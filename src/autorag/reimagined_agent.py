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
    "You are a production-logging assistant. You receive a transcript of a "
    "recording with word-level timestamps and must produce a hierarchical "
    "3-level topic outline summarizing what was discussed.\n\n"
    "Rules:\n"
    "1. Produce at most 3 levels: top-level topics (L1), subtopics (L2), and "
    "sub-subtopics / beats (L3).\n"
    "2. Only nest a subtopic level if you have at least 2 siblings at that "
    "level. If a topic would have only one subtopic, fold it into the "
    "parent instead of creating a lone child.\n"
    "3. Each topic has a `title` (<=120 chars), a `summary` (2-4 sentences), "
    "and an interval `[s, e]` in seconds (relative to audio start) "
    "marking when the topic BEGINS and ENDS in the recording. `s` is "
    "the start of the FIRST word the topic covers; `e` is the end of the "
    "LAST word the topic covers. Topics typically span tens to hundreds of "
    "seconds — not single words.\n"
    "4. Sibling topics TILE the parent's time range without gaps or overlap. "
    "Order siblings by time. For adjacent siblings A then B, set "
    "B.s = A.e. The first sibling's s equals the parent's "
    "s; the last sibling's e equals the parent's e.\n"
    "5. A parent's interval is the UNION of its children's intervals: "
    "parent.s = first_child.s, parent.e = last_child.e. "
    "Equivalently: every child interval is contained in its parent's.\n"
    "6. The L1 siblings together tile the entire audio: the first L1 starts "
    "at ~0.0 and the last L1 ends at ~the audio end.\n"
    "7. Do not invent topics that are not present in the transcript. Do not "
    "include timestamps outside the transcript's range.\n"
)
_HUMAN = (
    "Audio runs from 0.00 to {audio_e} seconds. Topic `s` values "
    "MUST be drawn from across this full range — do not cluster all "
    "topics near the beginning. Produce up to {levels} "
    "levels of nesting. Each line in the following trascript starts with a "
    "'s' followed by the word at that timestamp \n\n"
    "{transcript}"
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


def _normalize_intervals(topics: list[TopicDict], audio_e: float) -> None:
    """Derive parent intervals from children and tile siblings to remove overlap.

    Walks the tree bottom-up: each leaf keeps the LLM's `[s, e]`,
    each non-leaf is rewritten so `s = first_child.s` and
    `e = last_child.e` (the union-of-children rule). Then within each
    sibling list (top-down), adjacent topics are forced to tile: a sibling's
    `s` is clamped to the previous sibling's `e` if it would
    otherwise overlap. The final L1 has its `e` extended to `audio_e`
    if the LLM left a gap.
    """

    def fix_node(node: TopicDict) -> None:
        children = list(node.get("children") or [])
        for c in children:
            fix_node(c)
        if children:
            node["s"] = float(children[0].get("s", 0.0))
            node["e"] = float(children[-1].get("e", 0.0))

    for t in topics:
        fix_node(t)

    def tile_siblings(siblings: list[TopicDict]) -> None:
        for prev, cur in pairwise(siblings):
            prev_end = float(prev.get("e", 0.0))
            cur_start = float(cur.get("s", 0.0))
            cur_end = float(cur.get("e", 0.0))
            if cur_start < prev_end:
                cur["s"] = prev_end
                if cur_end < prev_end:
                    cur["e"] = prev_end
        for s in siblings:
            children = s.get("children") or []
            if children:
                tile_siblings(children)

    tile_siblings(topics)

    if topics:
        last = topics[-1]
        if float(last.get("e", 0.0)) < audio_e:
            last["e"] = float(audio_e)


def _extract_topics(
    spans: list[WordSpan],
    *,
    llm_model: str,
    base_url: str | None,
    levels: int,
) -> TopicTree:
    audio_e = max((float(ws.get("e", 0.0)) for ws in spans), default=0.0)
    llm_kwargs: dict[str, Any] = {"model": llm_model, "temperature": 0.0}
    if base_url:
        llm_kwargs["base_url"] = base_url
    llm = ChatOllama(**llm_kwargs)
    prompt = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _HUMAN)])
    chain = prompt | llm.with_structured_output(TopicTreeModel, method="json_schema")
    tree = chain.invoke(
        {
            "transcript": _format_transcript(spans),
            "audio_e": f"{audio_e:.2f}",
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
    llm_model: str = "llama3.1:8b",
    ollama_base_url: str | None = None,
    levels: int = 3,
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
