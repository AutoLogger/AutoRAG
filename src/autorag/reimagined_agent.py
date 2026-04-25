"""Clean-room LangChain pipeline that mirrors `agent.py`'s I/O contract.

Same input (`Path | str` to an audio file) and output shape
(`TranscriptionResult` with `transcription` + `topics`) as `agent.py`, but
shares no project code: Whisper is invoked directly, the LLM step uses a
`ChatPromptTemplate` with Pydantic-validated structured output, and types are
redefined locally so outputs remain interchangeable at runtime.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any, TypedDict, cast

import whisper
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


class WordSpan(TypedDict, total=False):
    w: str
    s: float
    e: float
    abs_s: float
    segment_id: str


class TopicDict(TypedDict, total=False):
    title: str
    summary: str
    s: float
    e: float
    children: list[TopicDict]


class TopicTree(TypedDict):
    topics: list[TopicDict]


class TranscriptionResult(TypedDict):
    transcription: list[WordSpan]
    topics: TopicTree


class TopicLeaf(BaseModel):
    """L3 (deepest) topic — `children` must always be an empty list."""

    title: str
    summary: str
    s: float
    e: float
    children: list[Any] = Field(max_length=0)


class TopicL2(BaseModel):
    title: str
    summary: str
    s: float
    e: float
    children: list[TopicLeaf]


class TopicL1(BaseModel):
    title: str
    summary: str
    s: float
    e: float
    children: list[TopicL2]


class TopicTreeModel(BaseModel):
    topics: list[TopicL1]


_SYSTEM = (
    "You are a transcript outliner. You receive a recording's word-level "
    "transcript and must produce a hierarchical 3-level topic outline "
    "summarizing what was discussed.\n\n"
    "Rules:\n"
    "1. Up to 3 levels: top-level topics (L1), subtopics (L2), sub-subtopics "
    "(L3). Only nest a level if at least 2 siblings would exist there; "
    "fold a lone child into its parent instead.\n"
    "2. Each topic has a `title` (<=120 chars), a `summary` (2-4 sentences), "
    "and an interval [s, e] in seconds. `s` is the start of the FIRST word "
    "the topic covers; `e` is the end of the LAST word it covers. Topics "
    "typically span tens to hundreds of seconds, not single words.\n"
    "3. Sibling topics TILE their parent's time range with no gaps and no "
    "overlap. Order siblings by time. For adjacent siblings A then B, set "
    "B.s = A.e. The first sibling's s equals the parent's s; the last "
    "sibling's e equals the parent's e.\n"
    "4. A parent's interval is the union of its children's intervals: "
    "parent.s = first_child.s, parent.e = last_child.e. Every child's "
    "interval is contained inside its parent's.\n"
    "5. The L1 siblings together tile the entire audio: the first L1 starts "
    "at 0.0 and the last L1 ends at the audio end.\n"
    "6. Each piece of content appears at exactly ONE position in the tree. "
    "Do NOT repeat the same topic both as an L1 and inside another L1's "
    "children. Do NOT duplicate a subtopic at multiple levels.\n"
    "7. Use only timestamps that come from the transcript. Do not invent "
    "topics that are not present in the transcript.\n"
)
_FEWSHOT = (
    "Example shape ONLY (illustrative — DO NOT copy these timestamps; the "
    "example uses a fictitious 29-minute (1740s) cooking-show audio whose "
    "duration will be very different from yours). Properties to imitate: "
    "4 distinct L1 topics that TILE the full audio end-to-end; every L2 "
    "tiles its L1; every L3 tiles its L2; no zero-duration intervals; no "
    "topic appears at more than one position.\n"
    "{{\n"
    '  "topics": [\n'
    "    {{\n"
    '      "title": "Mise en place and equipment",\n'
    '      "summary": "Host walks through the prep workflow and the '
    'knives, pans, and timer setup needed before cooking begins.",\n'
    '      "s": 0.0, "e": 423.0,\n'
    '      "children": [\n'
    "        {{\n"
    '          "title": "Prep workflow",\n'
    '          "summary": "Order of operations from washing to chopping.",\n'
    '          "s": 0.0, "e": 217.0, "children": []\n'
    "        }},\n"
    "        {{\n"
    '          "title": "Knives and pans",\n'
    '          "summary": "Which knife and pan for which task.",\n'
    '          "s": 217.0, "e": 423.0, "children": []\n'
    "        }}\n"
    "      ]\n"
    "    }},\n"
    "    {{\n"
    '      "title": "Building the base sauce",\n'
    '      "summary": "Sweats aromatics, deglazes the pan, and reduces '
    'the stock to a base.",\n'
    '      "s": 423.0, "e": 911.0,\n'
    '      "children": []\n'
    "    }},\n"
    "    {{\n"
    '      "title": "Searing and resting the protein",\n'
    '      "summary": "Pat-dry, high-heat sear, transfer to oven, then '
    'rest before slicing.",\n'
    '      "s": 911.0, "e": 1402.0,\n'
    '      "children": [\n'
    "        {{\n"
    '          "title": "Sear technique",\n'
    '          "summary": "Why a dry surface matters and how to avoid '
    'overcrowding the pan.",\n'
    '          "s": 911.0, "e": 1184.0, "children": []\n'
    "        }},\n"
    "        {{\n"
    '          "title": "Resting and slicing",\n'
    '          "summary": "Rest time relative to thickness; slicing '
    'against the grain.",\n'
    '          "s": 1184.0, "e": 1402.0, "children": []\n'
    "        }}\n"
    "      ]\n"
    "    }},\n"
    "    {{\n"
    '      "title": "Plating and tasting notes",\n'
    '      "summary": "Final assembly, seasoning adjustments, and the '
    "host's tasting commentary.\",\n"
    '      "s": 1402.0, "e": 1740.0,\n'
    '      "children": []\n'
    "    }}\n"
    "  ]\n"
    "}}\n"
)
_HUMAN = (
    "Audio runs from 0.00 to {audio_e} seconds (~{duration_min:.1f} min). "
    "Suggested top-level topic count: ~{target_count} (calibrated to "
    "duration; do not over-split into 15+ tiny topics, do not collapse "
    "into a single topic). Produce up to {levels} levels of nesting.\n\n"
    "Follow this example shape exactly (your titles and summaries must "
    "come from the actual transcript below, not from this example):\n"
    "{fewshot}\n"
    "Time anchors (evenly-sampled words across the audio - use these to "
    "see the full duration and spread topics across it; do NOT cluster "
    "topics near the start):\n"
    "{anchors}\n\n"
    "Full transcript (one word per line as 's=12.34 word'):\n{transcript}"
)


def _run_whisper(
    file: Path,
    *,
    model_size: str,
    language: str | None,
) -> list[WordSpan]:
    if not file.exists():
        raise FileNotFoundError(f"audio file not found: {file}")
    model = whisper.load_model(model_size)
    kwargs: dict[str, Any] = {"word_timestamps": True}
    if language:
        kwargs["language"] = language
    result: Any = model.transcribe(str(file), **kwargs)

    spans: list[WordSpan] = []
    for seg in result.get("segments") or []:
        for w in seg.get("words") or []:
            token = str(w.get("word", "") or "")
            if not token.strip():
                continue
            s = float(w.get("start", 0.0) or 0.0)
            e = float(w.get("end", s) or s)
            spans.append({"w": token, "s": s, "e": e, "abs_s": s, "segment_id": "single"})
    return spans


def _format_transcript(spans: list[WordSpan]) -> str:
    lines: list[str] = []
    for ws in spans:
        token = str(ws.get("w", "")).strip()
        if not token:
            continue
        lines.append(f"s={float(ws.get('s', 0.0)):.2f} {token}")
    return "\n".join(lines)


def _time_anchors(spans: list[WordSpan], n: int = 10) -> str:
    """Pick ~n evenly-spaced word lines and format them as anchor lines.

    Single-shot calls on long transcripts cluster topics in the first ~10s
    of audio because the LLM anchors on the first timestamps it sees. A
    short anchor block at the top of the prompt — `t=12.34s  word` lines
    spread evenly across the audio — gives the model explicit time
    references before it dives into the dense transcript and reduces the
    early-clustering failure mode.
    """
    real = [w for w in spans if str(w.get("w", "")).strip()]
    if not real:
        return "(empty transcript)"
    if len(real) <= n:
        picked = real
    else:
        step = len(real) / n
        picked = [real[min(len(real) - 1, int(i * step))] for i in range(n)]
        if picked[-1] is not real[-1]:
            picked.append(real[-1])
    return "\n".join(
        f"  t={float(w.get('s', 0.0)):.2f}s  {str(w.get('w', '')).strip()}" for w in picked
    )


def _target_count(audio_e: float) -> int:
    """Pick a top-level target topic count based on audio duration.

    Roughly one L1 per minute of speech, clamped to [2, 7]. Embedded as a
    soft suggestion in the prompt — not a schema constraint — to keep the
    LLM from over-splitting (15+ tiny topics) or under-splitting (one
    topic covering the whole audio).
    """
    target = round(max(0.0, audio_e) / 60.0)
    return max(2, min(7, target))


def _normalize_intervals(topics: list[TopicDict], audio_e: float) -> None:
    """Defensively clamp, sort, and tile every sibling list in the tree.

    Walks top-down. At each level: element-wise clamp every child interval
    into its parent's `[s, e]` (guards against hallucinated timestamps like
    `42526` for a 457s audio that would otherwise survive a sort-then-tile
    pass), sort by `s`, anchor `siblings[0].s = parent.s` and
    `siblings[-1].e = parent.e`, then force `cur.s = prev.e` for every
    adjacent pair so any remaining gaps OR overlaps collapse in one pass.
    Recurse with the now-fixed parent interval. The L1 list is tiled
    against `[0.0, audio_e]`.
    """

    def snap(siblings: list[TopicDict], slice_s: float, slice_e: float) -> None:
        if not siblings:
            return
        for c in siblings:
            cs = max(slice_s, min(slice_e, float(c.get("s", slice_s))))
            ce = max(slice_s, min(slice_e, float(c.get("e", slice_s))))
            if ce < cs:
                ce = cs
            c["s"] = cs
            c["e"] = ce
        siblings.sort(key=lambda c: float(c.get("s", 0.0)))
        siblings[0]["s"] = slice_s
        siblings[-1]["e"] = slice_e
        for prev, cur in pairwise(siblings):
            cur["s"] = float(prev.get("e", 0.0))
            if float(cur.get("e", 0.0)) < float(cur["s"]):
                cur["e"] = float(cur["s"])
        for c in siblings:
            snap(c.get("children") or [], float(c["s"]), float(c["e"]))

    def drop_zero(siblings: list[TopicDict]) -> list[TopicDict]:
        # Remove zero-duration siblings (LLM emitted boundaries past the
        # parent's end; clamp+tile collapsed them to point intervals). Keeping
        # them produces titled but timeless leaves that confuse downstream
        # consumers. Recurse first so a leaf-pruned parent that itself becomes
        # a no-information node still keeps its (non-zero) interval — only the
        # interval check decides retention.
        kept: list[TopicDict] = []
        for c in siblings:
            c["children"] = drop_zero(c.get("children") or [])
            if float(c.get("e", 0.0)) - float(c.get("s", 0.0)) > 1e-6:
                kept.append(c)
        return kept

    snap(topics, 0.0, audio_e)
    pruned = drop_zero(topics)
    topics[:] = pruned


def _extract_topics(
    spans: list[WordSpan],
    *,
    llm_model: str,
    base_url: str | None,
    levels: int,
    num_ctx: int,
) -> TopicTree:
    audio_e = max((float(ws.get("e", 0.0)) for ws in spans), default=0.0)
    # `num_ctx` matters: Ollama defaults to 2048, which silently truncates a
    # 7-minute transcript (~9-10K tokens) and starves the LLM of the audio's
    # later half. 16384 fits the 7-min reference clip with headroom. With a
    # 14B-q8 model (~15 GB weights) plus 16K f16 KV (~3 GB) plus runtime
    # overhead, full GPU offload requires ~20-22 GB; 32K KV pushed past the
    # 24 GB budget on this server (partial CPU spill). For longer audios on
    # a bigger card, raise this — and verify with `ollama ps` that all layers
    # stayed on the GPU.
    llm_kwargs: dict[str, Any] = {
        "model": llm_model,
        "temperature": 0.0,
        "num_ctx": num_ctx,
    }
    if base_url:
        llm_kwargs["base_url"] = base_url
    llm = ChatOllama(**llm_kwargs)
    prompt = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _HUMAN)])
    chain = prompt | llm.with_structured_output(TopicTreeModel, method="json_schema")
    tree = chain.invoke(
        {
            "transcript": _format_transcript(spans),
            "anchors": _time_anchors(spans),
            "fewshot": _FEWSHOT,
            "audio_e": f"{audio_e:.2f}",
            "duration_min": audio_e / 60.0,
            "target_count": _target_count(audio_e),
            "levels": max(1, min(3, int(levels or 3))),
        }
    )
    if not isinstance(tree, TopicTreeModel):
        raise RuntimeError(
            f"expected TopicTreeModel from structured output; got {type(tree).__name__}"
        )
    out = cast("TopicTree", tree.model_dump())
    _normalize_intervals(out["topics"], audio_e)
    return out


def build_reimagined_agent(
    *,
    whisper_model: str = "base",
    language: str | None = None,
    llm_model: str = "qwen2.5:14b-instruct-q8_0",
    ollama_base_url: str | None = None,
    levels: int = 3,
    num_ctx: int = 16384,
) -> Runnable[Path | str, TranscriptionResult]:
    """Build a `Runnable` that maps an audio file path to transcription+topics."""

    def _transcribe_step(file: Path | str) -> list[WordSpan]:
        return _run_whisper(Path(file), model_size=whisper_model, language=language)

    def _project(spans: list[WordSpan]) -> TranscriptionResult:
        return {
            "transcription": spans,
            "topics": _extract_topics(
                spans,
                llm_model=llm_model,
                base_url=ollama_base_url,
                levels=levels,
                num_ctx=num_ctx,
            ),
        }

    return RunnableLambda(_transcribe_step) | RunnableLambda(_project)


def transcribe(file: Path | str, **kwargs: Any) -> TranscriptionResult:
    """Convenience wrapper: build the agent and invoke it once."""
    return build_reimagined_agent(**kwargs).invoke(file)


__all__ = [
    "TopicDict",
    "TopicTree",
    "TranscriptionResult",
    "WordSpan",
    "build_reimagined_agent",
    "transcribe",
]
