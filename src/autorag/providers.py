"""LLM providers for hierarchical topic summarization.

Each provider consumes a flat list of `WordSpan` dicts (the Whisper transcript
flattened to absolute wall-clock and segment-relative time) and returns a
`TopicTree` dict of the shape:

    {
      "topics": [
        {
          "title": str,
          "start_s": float,
          "children": [
            {"title": str, "start_s": float, "children": [
              {"title": str, "start_s": float, "children": []}
            ]}
          ]
        },
        ...
      ]
    }

Max depth is 3 (root L1 topics, L2 children, L3 grandchildren).

Every implementation asks for structured JSON output via the provider's native
mechanism. Secrets/base URLs are read from the environment at request time.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypedDict

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- #
# Types                                                                          #
# ----------------------------------------------------------------------------- #


class WordSpan(TypedDict, total=False):
    w: str  # the token (including any leading space)
    s: float  # segment-relative start, seconds
    e: float  # segment-relative end, seconds
    abs_s: float  # absolute wall-clock offset from audio_start, seconds
    segment_id: str


class Topic(TypedDict, total=False):
    title: str
    summary: str
    start_s: float
    end_s: float
    children: list[Topic]


class TopicTree(TypedDict):
    topics: list[Topic]


ProviderName = Literal["ollama"]


PROVIDER_DEFAULT_MODELS: dict[str, str] = {
    "ollama": "llama3.1:8b",
}


def OLLAMA_BASE_URL() -> str:  # noqa: N802
    """Resolve the Ollama base URL from env, falling back to localhost."""
    raw = os.environ.get("AUTOLOGGER_OLLAMA_BASE_URL", "").strip()
    return raw or "http://localhost:11434"


# ----------------------------------------------------------------------------- #
# Prompt construction                                                            #
# ----------------------------------------------------------------------------- #


SYSTEM_PROMPT = (
    "You are a production-logging assistant. You receive a transcript of a "
    "recording with word-level timestamps and must produce a hierarchical "
    "3-level topic outline summarizing what was discussed.\n\n"
    "Rules:\n"
    "1. Produce at most 3 levels: top-level topics (L1), subtopics (L2), and "
    "sub-subtopics / beats (L3).\n"
    "2. Only nest a subtopic level if you have at least 2 siblings at that "
    "level. If a topic would have only one subtopic, fold it into the "
    "parent instead of creating a lone child.\n"
    "3. Each topic has a short sentence `title` (<=120 chars) describing "
    "what it covers, a `summary` (2-4 sentences describing in detail what "
    "was discussed in this section), and a `start_s` number: the earliest "
    "word start time (in seconds, relative to audio start) that the topic "
    "covers.\n"
    "4. `start_s` values must be monotonically non-decreasing within a "
    "sibling list.\n"
    "5. Output ONLY JSON matching this schema:\n"
    '{"topics":[{"title":str,"summary":str,"start_s":float,'
    '"children":[{"title":str,"summary":str,"start_s":float,'
    '"children":[{"title":str,"summary":str,"start_s":float,"children":[]}]}]}]}\n'
    "6. Do not invent topics that are not present in the transcript. Do not "
    "include timestamps outside the transcript's range.\n"
)


# Grouping words into ~10s chunks keeps prompts short and — more importantly —
# keeps the timestamp anchors spread across the whole audio so the LLM doesn't
# cluster every topic near the start of what it can attend to.
_PROMPT_CHUNK_SECONDS = 1.0


def _chunk_transcript(
    transcript: list[WordSpan], chunk_seconds: float = _PROMPT_CHUNK_SECONDS
) -> list[tuple[float, float, str]]:
    """Group consecutive words into (start_s, end_s, text) chunks.

    A new chunk starts whenever the current chunk's span has reached
    ``chunk_seconds``. Empty tokens are skipped.
    """
    chunks: list[tuple[float, float, str]] = []
    cur_start: float | None = None
    cur_end: float = 0.0
    cur_words: list[str] = []
    for ws in transcript:
        token = str(ws.get("w", "")).strip()
        if not token:
            continue
        s = float(ws.get("abs_s", ws.get("s", 0.0)) or 0.0)
        e = float(ws.get("e", s) or s)
        if cur_start is None:
            cur_start = s
        if s - cur_start >= chunk_seconds and cur_words:
            chunks.append((cur_start, cur_end, " ".join(cur_words)))
            cur_words = []
            cur_start = s
        cur_end = max(cur_end, e, s)
        cur_words.append(token)
    if cur_words and cur_start is not None:
        chunks.append((cur_start, cur_end, " ".join(cur_words)))
    return chunks


def _build_user_prompt(transcript: list[WordSpan], levels: int, prompt_extras: str) -> str:
    levels = max(1, min(3, int(levels or 3)))
    lines = [
        f"Produce up to {levels} levels of topic nesting.",
    ]
    if prompt_extras:
        lines.append(f"Additional instructions: {prompt_extras.strip()}")
    lines.append("")
    chunks = _chunk_transcript(transcript)
    audio_end_s = chunks[-1][1] if chunks else 0.0
    lines.append(
        f"Audio spans 0 to {audio_end_s:.1f} seconds. Topic `start_s` values "
        "MUST be drawn from across this full range — do not cluster all "
        "topics near the beginning."
    )
    lines.append("")
    lines.append(
        "Transcript (one JSON object per line: {s: seconds at chunk start, w: chunk text}):"
    )
    for s, _e, text in chunks:
        lines.append(json.dumps({"s": round(s, 2), "w": text}, ensure_ascii=False))
    lines.append("")
    lines.append(
        "Return ONLY the JSON object described in the system prompt. "
        "No preamble, no markdown, no code fences."
    )
    return "\n".join(lines)


# A JSON Schema for providers that support structured output.
TOPIC_LEAF_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "start_s": {"type": "number"},
        "children": {
            "type": "array",
            "items": {"type": "object"},
            "maxItems": 0,
        },
    },
    "required": ["title", "summary", "start_s", "children"],
    "additionalProperties": False,
}

TOPIC_L2_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "start_s": {"type": "number"},
        "children": {"type": "array", "items": TOPIC_LEAF_SCHEMA},
    },
    "required": ["title", "summary", "start_s", "children"],
    "additionalProperties": False,
}

TOPIC_L1_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "start_s": {"type": "number"},
        "children": {"type": "array", "items": TOPIC_L2_SCHEMA},
    },
    "required": ["title", "summary", "start_s", "children"],
    "additionalProperties": False,
}

TOPIC_TREE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "topics": {"type": "array", "items": TOPIC_L1_SCHEMA},
    },
    "required": ["topics"],
    "additionalProperties": False,
}


# ----------------------------------------------------------------------------- #
# Validation                                                                     #
# ----------------------------------------------------------------------------- #


def _coerce_tree(obj: Any, max_depth: int = 3) -> TopicTree:
    """Best-effort validation & normalization of a provider response.

    Raises `ValueError` if the shape is unusable.
    """
    if not isinstance(obj, dict) or "topics" not in obj:
        raise ValueError(f"response missing top-level `topics` array; got: {str(obj)[:200]!r}")
    raw_topics = obj.get("topics")
    if not isinstance(raw_topics, list):
        raise ValueError(f"`topics` must be an array; got: {str(raw_topics)[:200]!r}")

    def _walk(nodes: Any, depth: int) -> list[Topic]:
        out: list[Topic] = []
        if not isinstance(nodes, list):
            return out
        for n in nodes:
            if not isinstance(n, dict):
                continue
            title = str(n.get("title", "") or "").strip()
            try:
                start_s = float(n.get("start_s", 0.0) or 0.0)
            except (TypeError, ValueError):
                start_s = 0.0
            if not title:
                continue
            summary = str(n.get("summary", "") or "").strip()
            children_raw = n.get("children") or []
            children = _walk(children_raw, depth + 1) if depth < max_depth else []
            out.append(
                {"title": title, "summary": summary, "start_s": start_s, "children": children}
            )
        return out

    topics = _walk(raw_topics, 1)
    return {"topics": topics}


def _extract_json_text(text: str) -> str:
    """Strip common wrappers (markdown code fences, stray prose) from LLM output."""
    s = (text or "").strip()
    if not s:
        return s
    if s.startswith("```"):
        # ```json\n{...}\n``` or ```\n{...}\n```
        s = s.lstrip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    # Find outermost braces if there's surrounding prose
    if s and not s.lstrip().startswith("{"):
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            s = s[start : end + 1]
    return s


# ----------------------------------------------------------------------------- #
# Provider Protocol + Implementations                                            #
# ----------------------------------------------------------------------------- #


class LLMProvider(Protocol):
    def summarize(
        self,
        transcript: list[WordSpan],
        levels: int,
        prompt_extras: str,
    ) -> TopicTree: ...


@dataclass
class OllamaProvider:
    model: str = PROVIDER_DEFAULT_MODELS["ollama"]

    def summarize(self, transcript: list[WordSpan], levels: int, prompt_extras: str) -> TopicTree:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("The `httpx` package is not installed.") from exc

        base = OLLAMA_BASE_URL().rstrip("/")
        user_prompt = _build_user_prompt(transcript, levels, prompt_extras)
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"num_ctx": 64000, "num_predict": -1, "temperature": 0.0},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        try:
            with httpx.Client(timeout=600.0) as client:
                resp = client.post(f"{base}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            raise RuntimeError(f"Ollama request to {base} failed: {exc}") from exc

        # /api/chat returns {"message": {"role":"assistant","content":"..."}, ...}
        text = ""
        if isinstance(data, dict):
            msg = data.get("message") or {}
            if isinstance(msg, dict):
                text = str(msg.get("content") or "")
            if not text:
                text = str(data.get("response") or "")
        if not text:
            raise RuntimeError("Ollama returned empty content.")
        try:
            parsed = json.loads(_extract_json_text(text))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Ollama returned non-JSON text: {exc}") from exc
        return _coerce_tree(parsed)


# ----------------------------------------------------------------------------- #
# Factory                                                                        #
# ----------------------------------------------------------------------------- #


def get_provider(name: str, *, model: str) -> LLMProvider:
    key = (name or "").strip().lower()
    chosen_model = (model or "").strip() or PROVIDER_DEFAULT_MODELS.get(key, "")
    if key == "ollama":
        return OllamaProvider(model=chosen_model)
    raise ValueError(f"Unknown provider: {name!r}")


__all__ = [
    "OLLAMA_BASE_URL",
    "PROVIDER_DEFAULT_MODELS",
    "TOPIC_TREE_SCHEMA",
    "LLMProvider",
    "OllamaProvider",
    "ProviderName",
    "Topic",
    "TopicTree",
    "WordSpan",
    "get_provider",
]
