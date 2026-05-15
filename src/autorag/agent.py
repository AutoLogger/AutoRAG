"""Audio → hierarchical topic tree. The single agent for AutoRAG.

Multi-pass L0 / L1 / L2 extractor — each LLM stage has one focused job::

    1. Whisper                              -> list[WordSpan]               1 call
    2. L1 boundaries  (single LLM call)     -> list[{s,e}]                  1 LLM
    3a Decide subdivide  (per long L1)      -> list[bool]                   N LLM
    3b L2 boundaries  (per yes-L1, batched) -> list[list[{s,e}]]            M LLM (M<=N)
    4. Summarize nodes  (per L1+L2, batched)-> {title,summary} per node     K LLM
    5. L0 aggregate                         -> {title, summary}             1 LLM

Final shape: ``{"topics": [L0]}`` with ``L0.children = [L1...]``, each
``L1.children = [L2...]`` or ``[]``. The L0 root is the explicit "what is
this audio about" node.

Boundary calls receive a time-bucketed (``format_blocks``, 30s) transcript and
emit ``{s, e}`` as ``MM:SS`` strings, which we parse back to float seconds here
(never the LLM — no model-side arithmetic). Per-node summary calls operate on
the slice's plain text (no timestamps) and emit ``{title, summary}``. The
K=N1+N2 summary calls share an identical prompt prefix for cache reuse.
"""

from __future__ import annotations

import logging
import os
import time
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from autorag import diarize, whisper_runner
from autorag.blocks import format_blocks, mmss
from autorag.blocks import group_by_speaker as _group_by_speaker
from autorag.types import TopicDict, TopicTree, TranscriptionResult, WordSpan

logger = logging.getLogger(__name__)

# Window size for the block-formatted transcript fed to the L1/L2 boundary
# prompts. Smaller = more frequent MM:SS anchors (finer possible boundaries) at
# the cost of more lines; 30s gives a fresh anchor at least twice a minute even
# inside a long single-speaker monologue.
_BOUNDARY_BLOCK_SECONDS = 30


class _Boundary(BaseModel):
    # `str`, not `float`, on purpose: the structured-output JSON schema is
    # derived from these annotations, so a `float` here would instruct the
    # model to emit numbers. We want it to copy the transcript's `MM:SS`
    # markers verbatim; `_parse_ts` converts them to seconds before tiling.
    s: str
    e: str


class _BoundaryList(BaseModel):
    topics: list[_Boundary]


class _SubdivideDecision(BaseModel):
    # `reason` is placed BEFORE `subdivide` on purpose: small structured-output
    # models produce more accurate booleans when they emit a short rationale
    # first. The reason is parsed but discarded by the orchestrator.
    reason: str
    subdivide: bool


class _NodeSummary(BaseModel):
    title: str
    summary: str


class _L0Summary(BaseModel):
    title: str
    summary: str


_L1_SYS = (
    "You are a topic boundary detector. You receive a recording's "
    "transcript as time-bucketed blocks: blocks are separated by blank "
    "lines and every line is `MM:SS-MM:SS Speaker K: <words>` (the two "
    "MM:SS values are that turn's start and end). You must split the "
    "recording into ordered, non-overlapping top-level (L1) topics that "
    "TILE the audio from start to end.\n\n"
    "Speaker changes are useful evidence for topic boundaries, but a "
    "single topic may span multiple speakers.\n\n"
    "Rules:\n"
    "1. Return ONLY intervals -- no titles, no summaries. Each item is "
    '{{"s": "MM:SS", "e": "MM:SS"}}.\n'
    "2. The first topic's `s` equals the first MM:SS in the transcript; "
    "the last topic's `e` equals the last MM:SS in the transcript.\n"
    "3. Adjacent topics tile end-to-start: for siblings A then B, set "
    "B.s = A.e (no gaps, no overlap). Order siblings by time.\n"
    "4. Aim for roughly the suggested topic count -- it is calibrated to "
    "duration. Do NOT over-split into 15+ tiny topics; do NOT collapse "
    "into a single topic unless the audio is very short.\n"
    "5. Topics typically span tens to hundreds of seconds, not single "
    "lines.\n"
    "6. Copy MM:SS values directly from the transcript's range markers. "
    "Do not invent or reformat timestamps."
)
_L1_HUMAN = (
    "Audio runs from 00:00 to {audio_e} (~{duration_min:.1f} min). "
    "Suggested topic count: ~{target_count}. Spread topics across the "
    "FULL duration; do NOT cluster them near the start.\n\n"
    "Time-bucketed transcript (blocks separated by blank lines; each "
    "line is `MM:SS-MM:SS Speaker K: <words>`):\n{transcript}"
)

_DECIDE_SYS = (
    "You decide whether a passage of speech is substantial enough to be "
    "broken into 2 or more distinct subtopics, or whether it covers a "
    "single coherent point that should NOT be subdivided.\n\n"
    "Text may include `Speaker N:` prefixes when multiple speakers are "
    "present. Consider all speakers together when deciding; speaker "
    "turns alone are not subtopics.\n\n"
    "Rules:\n"
    "1. Set subdivide=true ONLY if you can identify at least 2 distinct, "
    "well-bounded subtopics inside the passage. Each subtopic must cover "
    "a meaningful span of speech (tens of seconds, not a few words).\n"
    "2. Set subdivide=false when the passage is on a single subject, when "
    "it is short, or when any split would be artificial.\n"
    "3. The `reason` field should be one short sentence describing why."
)
_DECIDE_HUMAN = (
    "Passage runs ~{duration_min:.1f} minutes.\n\nTranscript (plain text):\n{transcript}"
)

_L2_SYS = (
    "You are a topic boundary detector. You receive a SLICE of a longer "
    "recording's transcript as time-bucketed blocks: blocks are "
    "separated by blank lines and every line is "
    "`MM:SS-MM:SS Speaker K: <words>` (the two MM:SS values are that "
    "turn's start and end). You must split the slice into ordered, "
    "non-overlapping subtopics that TILE the slice from start to end.\n\n"
    "Speaker changes are useful evidence for subtopic boundaries, but a "
    "single subtopic may span multiple speakers.\n\n"
    "Rules:\n"
    "1. Return ONLY intervals -- no titles, no summaries. Each item is "
    '{{"s": "MM:SS", "e": "MM:SS"}}.\n'
    "2. The first subtopic's `s` equals the slice start; the last "
    "subtopic's `e` equals the slice end.\n"
    "3. Adjacent subtopics tile end-to-start: for siblings A then B, set "
    "B.s = A.e (no gaps, no overlap). Order by time.\n"
    "4. Copy MM:SS values directly from the transcript's range markers. "
    "Do not invent or reformat timestamps."
)
_L2_HUMAN = (
    "Slice spans [{slice_s} to {slice_e}] (~{duration_min:.1f} min). "
    "Suggested subtopic count: ~{target_count}. Produce at least 2 "
    "subtopics that together tile the slice; if you genuinely cannot "
    "find 2 distinct subjects, return exactly 2 anyway by splitting on "
    "the clearest natural break.\n\n"
    "Time-bucketed slice transcript (blocks separated by blank lines; "
    "each line is `MM:SS-MM:SS Speaker K: <words>`):\n{transcript}"
)

_NODE_SUM_SYS = (
    "You summarize a passage of transcribed speech. Given the passage "
    "text, return a short title and a 1-2 sentence summary describing "
    "what was said.\n\n"
    "Text may include `Speaker N:` prefixes when multiple speakers are "
    "present. Consider all speakers together; the summary should "
    "describe the passage's content, mentioning who said what only "
    "when it materially aids understanding.\n\n"
    "Rules:\n"
    "1. `title` is a noun phrase, at most 120 characters. No trailing "
    "punctuation. Not a full sentence.\n"
    "2. `summary` is 1-2 sentences describing the passage's content.\n"
    "3. Do not invent content beyond what the passage says. Do not "
    "speculate about surrounding context."
)
_NODE_SUM_HUMAN = "Passage:\n{text}"

_AGG_SYS = (
    "You are summarizing a whole audio from its top-level topics. Given "
    "the topics' titles and summaries, produce a single overall title "
    "(<=120 chars) and a 2-4 sentence summary capturing the unifying "
    "theme. Do not invent content beyond what the topics describe."
)
_AGG_HUMAN = "Top-level topics:\n{children}"


def _ollama_base_url() -> str:
    """Resolve the Ollama base URL from env, falling back to localhost."""
    raw = os.environ.get("AUTORAG_OLLAMA_BASE_URL", "").strip()
    return raw or "http://localhost:11434"


def _run_whisper(file: Path, *, model_size: str, language: str | None) -> list[WordSpan]:
    if not file.exists():
        raise FileNotFoundError(f"audio file not found: {file}")
    model = whisper_runner.get_model(model_size, device_hint="cuda")
    raw_words = whisper_runner.transcribe_segment(model, str(file), language)
    turns = diarize.diarize_file(str(file))
    labels = diarize.assign_speakers(raw_words, turns)

    spans: list[WordSpan] = []
    for w, label in zip(raw_words, labels, strict=True):
        s = float(w["s"])
        spans.append(
            {
                "w": str(w["w"]),
                "s": s,
                "e": float(w["e"]),
                "segment_id": "single",
                "speaker": label,
            }
        )
    return spans


def _parse_ts(value: str) -> float:
    """Parse an ``MM:SS`` / ``H:MM:SS`` (or bare-number) timestamp to seconds.

    The boundary LLM copies ``MM:SS`` markers straight from the block-formatted
    transcript; we do the arithmetic here rather than trusting the model to.
    Each ``:``-separated field is a base-60 digit, so minutes may exceed 59 for
    long audio (``"120:00"`` -> 7200.0). A bare number passes through. Anything
    unparseable returns ``0.0`` — ``_snap_tile`` / ``_drop_zero`` then repair
    the degenerate node.
    """
    raw = str(value).strip()
    if not raw:
        return 0.0
    try:
        if ":" in raw:
            total = 0.0
            for part in raw.split(":"):
                total = total * 60.0 + float(part)
            return total
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _format_words_only(spans: list[WordSpan]) -> str:
    lines: list[str] = []
    for speaker, group in _group_by_speaker(spans):
        tokens = [t for ws in group if (t := str(ws.get("w", "")).strip())]
        if tokens:
            lines.append(f"Speaker {speaker}: {' '.join(tokens)}")
    return "\n".join(lines)


def _format_children(children: list[TopicDict]) -> str:
    lines: list[str] = []
    for c in children:
        lines.append(f"- title: {c.get('title', '') or ''}")
        lines.append(f"  summary: {c.get('summary', '') or ''}")
    return "\n".join(lines)


def _slice_spans(spans: list[WordSpan], s: float, e: float) -> list[WordSpan]:
    return [w for w in spans if s <= float(w.get("s", 0.0)) <= e]


def _audio_end(spans: list[WordSpan]) -> float:
    return max((float(w.get("e", 0.0)) for w in spans), default=0.0)


def _target_count(slice_s: float, slice_e: float) -> int:
    duration_s = max(0.0, slice_e - slice_s)
    target = round(duration_s / 60.0)
    return max(2, min(7, target))


def _snap_tile(siblings: list[TopicDict], slice_s: float, slice_e: float) -> None:
    """Element-wise clamp into [slice_s, slice_e], sort by `s`, anchor first/last,
    then force `cur.s = prev.e` for every adjacent pair so any remaining gaps
    OR overlaps collapse in one pass. Mutates `siblings` in place.
    """
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


def _drop_zero(siblings: list[TopicDict]) -> list[TopicDict]:
    kept: list[TopicDict] = []
    for c in siblings:
        c["children"] = _drop_zero(c.get("children") or [])
        if float(c.get("e", 0.0)) - float(c.get("s", 0.0)) > 1e-6:
            kept.append(c)
    return kept


def _new_node(s: float, e: float, *, title: str = "", summary: str = "") -> TopicDict:
    return {"title": title, "summary": summary, "s": s, "e": e, "children": []}


def build_topic_runnable(
    *,
    llm_model: str = "qwen2.5:14b-instruct-q8_0",
    ollama_base_url: str | None = None,
    num_ctx_l1: int = 16384,
    num_ctx_fanout: int = 8192,
    max_concurrency: int = 4,
    min_subdivide_duration_s: float = 120.0,
) -> Runnable[list[WordSpan], TopicTree]:
    """Build a Runnable mapping list[WordSpan] -> TopicTree (L0/L1/L2 hierarchy).

    Notes on Ollama settings (server-side, controlled outside this module):

    - `temperature=0.0` plus identical system prompts per chain plus
      `OLLAMA_MULTIUSER_CACHE=true` give prefix-cache hits across all calls
      inside a single chain. We set `keep_alive=0` so Ollama unloads the
      model the moment the last in-flight request finishes, but during a
      batch the model stays loaded (Ollama ref-counts in-flight requests),
      so this only kicks in at stage boundaries.
    - With `OLLAMA_NUM_PARALLEL=1` (current server config for the 14B
      model) the server serializes batched requests, so Stage 3a/3b
      wall-clock is `N x per-call`, not `N/4 x per-call`. Raising
      `NUM_PARALLEL` requires more VRAM (the server reserves all slots'
      KV-cache up front at the request's `num_ctx`). See `CLAUDE.md`
      "Ollama tuning notes".
    """

    base_kwargs: dict[str, Any] = {
        "model": llm_model,
        "temperature": 0.0,
        "keep_alive": 0,
        "base_url": ollama_base_url or _ollama_base_url(),
    }

    boundary_llm_l1 = ChatOllama(num_ctx=num_ctx_l1, **base_kwargs).with_structured_output(
        _BoundaryList, method="json_schema"
    )
    boundary_llm_fanout = ChatOllama(num_ctx=num_ctx_fanout, **base_kwargs).with_structured_output(
        _BoundaryList, method="json_schema"
    )
    decide_llm = ChatOllama(num_ctx=num_ctx_fanout, **base_kwargs).with_structured_output(
        _SubdivideDecision, method="json_schema"
    )
    node_sum_llm = ChatOllama(num_ctx=num_ctx_fanout, **base_kwargs).with_structured_output(
        _NodeSummary, method="json_schema"
    )
    agg_llm = ChatOllama(num_ctx=num_ctx_fanout, **base_kwargs).with_structured_output(
        _L0Summary, method="json_schema"
    )

    l1_chain = (
        ChatPromptTemplate.from_messages([("system", _L1_SYS), ("human", _L1_HUMAN)])
        | boundary_llm_l1
    )
    decide_chain = (
        ChatPromptTemplate.from_messages([("system", _DECIDE_SYS), ("human", _DECIDE_HUMAN)])
        | decide_llm
    )
    l2_chain = (
        ChatPromptTemplate.from_messages([("system", _L2_SYS), ("human", _L2_HUMAN)])
        | boundary_llm_fanout
    )
    node_sum_chain = (
        ChatPromptTemplate.from_messages([("system", _NODE_SUM_SYS), ("human", _NODE_SUM_HUMAN)])
        | node_sum_llm
    )
    agg_chain = (
        ChatPromptTemplate.from_messages([("system", _AGG_SYS), ("human", _AGG_HUMAN)]) | agg_llm
    )

    batch_cfg: RunnableConfig = {"max_concurrency": max_concurrency}

    def _extract_l1_boundaries(spans: list[WordSpan], audio_e: float) -> list[TopicDict]:
        result = cast(
            "_BoundaryList",
            l1_chain.invoke(
                {
                    "transcript": format_blocks(spans, _BOUNDARY_BLOCK_SECONDS),
                    "audio_e": mmss(audio_e),
                    "duration_min": audio_e / 60.0,
                    "target_count": _target_count(0.0, audio_e),
                }
            ),
        )
        nodes: list[TopicDict] = [_new_node(_parse_ts(b.s), _parse_ts(b.e)) for b in result.topics]
        _snap_tile(nodes, 0.0, audio_e)
        nodes = _drop_zero(nodes)
        if not nodes:
            return [_new_node(0.0, audio_e)]
        return nodes

    def _decide_subdivide_batch(l1_nodes: list[TopicDict], spans: list[WordSpan]) -> list[bool]:
        # Build inputs only for L1s long enough to plausibly subdivide;
        # short ones force False without an LLM call.
        decisions: list[bool] = [False] * len(l1_nodes)
        long_indices: list[int] = []
        long_inputs: list[dict[str, Any]] = []
        for i, l1 in enumerate(l1_nodes):
            ls = float(l1.get("s", 0.0))
            le = float(l1.get("e", 0.0))
            if (le - ls) < min_subdivide_duration_s:
                continue
            sliced = _slice_spans(spans, ls, le)
            long_indices.append(i)
            long_inputs.append(
                {
                    "transcript": _format_words_only(sliced),
                    "duration_min": (le - ls) / 60.0,
                }
            )
        if not long_inputs:
            return decisions
        results = cast(
            "list[_SubdivideDecision]",
            decide_chain.batch(long_inputs, config=batch_cfg),
        )
        for idx, dec in zip(long_indices, results, strict=True):
            decisions[idx] = bool(dec.subdivide)
        return decisions

    def _extract_l2_boundaries_batch(
        yes_l1_nodes: list[TopicDict], spans: list[WordSpan]
    ) -> list[list[TopicDict]]:
        if not yes_l1_nodes:
            return []
        inputs: list[dict[str, Any]] = []
        for l1 in yes_l1_nodes:
            ls = float(l1.get("s", 0.0))
            le = float(l1.get("e", 0.0))
            sliced = _slice_spans(spans, ls, le)
            inputs.append(
                {
                    "transcript": format_blocks(sliced, _BOUNDARY_BLOCK_SECONDS),
                    "slice_s": mmss(ls),
                    "slice_e": mmss(le),
                    "duration_min": (le - ls) / 60.0,
                    "target_count": _target_count(ls, le),
                }
            )
        results = cast("list[_BoundaryList]", l2_chain.batch(inputs, config=batch_cfg))
        out: list[list[TopicDict]] = []
        for r, l1 in zip(results, yes_l1_nodes, strict=True):
            ls = float(l1.get("s", 0.0))
            le = float(l1.get("e", 0.0))
            kids: list[TopicDict] = [_new_node(_parse_ts(b.s), _parse_ts(b.e)) for b in r.topics]
            _snap_tile(kids, ls, le)
            kids = _drop_zero(kids)
            out.append(kids)
        return out

    def _summarize_nodes_batch(nodes: list[TopicDict], spans: list[WordSpan]) -> None:
        if not nodes:
            return
        inputs: list[dict[str, Any]] = []
        keep_idx: list[int] = []
        for i, n in enumerate(nodes):
            ns = float(n.get("s", 0.0))
            ne = float(n.get("e", 0.0))
            text = _format_words_only(_slice_spans(spans, ns, ne))
            if not text.strip():
                continue
            keep_idx.append(i)
            inputs.append({"text": text})
        if not inputs:
            return
        results = cast(
            "list[_NodeSummary]",
            node_sum_chain.batch(inputs, config=batch_cfg),
        )
        for idx, summ in zip(keep_idx, results, strict=True):
            nodes[idx]["title"] = summ.title
            nodes[idx]["summary"] = summ.summary

    def _aggregate_l0(l1_nodes: list[TopicDict], audio_e: float) -> TopicDict:
        if not l1_nodes:
            return _new_node(0.0, audio_e, title="(empty)", summary="")
        result = cast(
            "_L0Summary",
            agg_chain.invoke({"children": _format_children(l1_nodes)}),
        )
        return _new_node(0.0, audio_e, title=result.title, summary=result.summary)

    def _build_tree(spans: list[WordSpan]) -> TopicTree:
        if not spans:
            return {"topics": []}

        audio_e = _audio_end(spans)

        logger.info("Stage 2 (L1 boundaries): 1 call, num_ctx=%d", num_ctx_l1)
        t0 = time.time()
        l1_nodes = _extract_l1_boundaries(spans, audio_e)
        logger.info("Stage 2 done in %.1fs (%d L1 topics)", time.time() - t0, len(l1_nodes))

        logger.info(
            "Stage 3a (decide subdivide): up to %d batched calls (min slice=%.0fs)",
            len(l1_nodes),
            min_subdivide_duration_s,
        )
        t0 = time.time()
        decisions = _decide_subdivide_batch(l1_nodes, spans)
        yes_count = sum(1 for d in decisions if d)
        logger.info(
            "Stage 3a done in %.1fs (%d/%d subdivide=true)",
            time.time() - t0,
            yes_count,
            len(l1_nodes),
        )

        yes_l1_nodes = [l1 for l1, d in zip(l1_nodes, decisions, strict=True) if d]
        logger.info("Stage 3b (L2 boundaries): %d batched calls", len(yes_l1_nodes))
        t0 = time.time()
        l2_lists = _extract_l2_boundaries_batch(yes_l1_nodes, spans)
        for l1, kids in zip(yes_l1_nodes, l2_lists, strict=True):
            l1["children"] = kids
        l2_total = sum(len(k) for k in l2_lists)
        logger.info("Stage 3b done in %.1fs (%d L2 topics)", time.time() - t0, l2_total)

        nodes_to_summarize: list[TopicDict] = []
        for l1 in l1_nodes:
            nodes_to_summarize.append(l1)
            for l2 in l1.get("children") or []:
                nodes_to_summarize.append(l2)
        logger.info("Stage 4 (summarize nodes): %d batched calls", len(nodes_to_summarize))
        t0 = time.time()
        _summarize_nodes_batch(nodes_to_summarize, spans)
        logger.info("Stage 4 done in %.1fs", time.time() - t0)

        logger.info("Stage 5 (L0 aggregate): 1 call")
        t0 = time.time()
        l0 = _aggregate_l0(l1_nodes, audio_e)
        l0["children"] = l1_nodes
        logger.info("Stage 5 done in %.1fs", time.time() - t0)

        return {"topics": [l0]}

    return RunnableLambda(_build_tree)


def build_agent(
    *,
    whisper_model: str = "base",
    language: str | None = None,
    llm_model: str = "qwen2.5:14b-instruct-q8_0",
    ollama_base_url: str | None = None,
    num_ctx_l1: int = 16384,
    num_ctx_fanout: int = 8192,
    max_concurrency: int = 4,
    min_subdivide_duration_s: float = 120.0,
) -> Runnable[Path | str, TranscriptionResult]:
    """Build a Runnable mapping audio file -> {transcription, topics:{topics:[L0]}}."""
    topic_runnable = build_topic_runnable(
        llm_model=llm_model,
        ollama_base_url=ollama_base_url,
        num_ctx_l1=num_ctx_l1,
        num_ctx_fanout=num_ctx_fanout,
        max_concurrency=max_concurrency,
        min_subdivide_duration_s=min_subdivide_duration_s,
    )

    def _whisper_step(file: Path | str) -> list[WordSpan]:
        f = Path(file)
        logger.info("Stage 1 (Whisper): transcribing %s", f.name)
        t0 = time.time()
        spans = _run_whisper(f, model_size=whisper_model, language=language)
        logger.info("Stage 1 done in %.1fs (%d words)", time.time() - t0, len(spans))
        return spans

    def _assemble(spans: list[WordSpan]) -> TranscriptionResult:
        topics: TopicTree = topic_runnable.invoke(spans)
        return {"transcription": spans, "topics": topics}

    return RunnableLambda(_whisper_step) | RunnableLambda(_assemble)


def transcribe_audio(
    file: Path | str,
    *,
    whisper_model: str = "base",
    language: str | None = None,
) -> list[WordSpan]:
    """Run Whisper + diarization on a local audio file, returning word spans."""
    f = Path(file)
    logger.info("Stage 1 (Whisper): transcribing %s", f.name)
    t0 = time.time()
    spans = _run_whisper(f, model_size=whisper_model, language=language)
    logger.info("Stage 1 done in %.1fs (%d words)", time.time() - t0, len(spans))
    return spans


def generate_topics(words: list[WordSpan], **kwargs: Any) -> TopicTree:
    """Build the topic runnable and invoke it once."""
    return build_topic_runnable(**kwargs).invoke(words)


def transcribe(file: Path | str, **kwargs: Any) -> TranscriptionResult:
    """Build the agent and invoke it once."""
    return build_agent(**kwargs).invoke(file)


__all__ = [
    "TopicDict",
    "TopicTree",
    "TranscriptionResult",
    "WordSpan",
    "build_agent",
    "build_topic_runnable",
    "generate_topics",
    "transcribe",
    "transcribe_audio",
]
