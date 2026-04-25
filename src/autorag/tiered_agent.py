"""Tiered L0 / L1 / L2 multi-pass topic extractor.

Sibling to `agent.py`, `reimagined_agent.py`, and `hierarchical_agent.py`.
Same `transcribe(file)` / `build_tiered_agent()` surface and same
`TranscriptionResult` output keys (`s`/`e`/`title`/`summary`/`children`).

Pipeline:

    1. Whisper                             -> list[WordSpan]               1 call
    2. L1 extract  (single LLM call)       -> list[L1 topic]               1 LLM
    3a Decide subdivide  (per L1, batched) -> list[bool]                   N LLM
    3b L2 extract  (per yes-L1, batched)   -> list[list[L2 topic]]         M LLM (M<=N)
    4. L0 aggregate                        -> {title, summary}             1 LLM

Final shape: `{"topics": [L0]}` with `L0.children = [L1...]`, each
`L1.children = [L2...]` or `[]`. The L0 root is the explicit "what is this
audio about" node — replaces the L3 layer that reimagined / hierarchical
produce.

Why multi-pass: one-shot agents on small models routinely under-split or
over-split. Splitting the work into L1 -> decide -> L2 -> L0 lets each LLM
call have one focused job, and the explicit "should I subdivide?" gate
stops the over-eager nesting that produces zero-duration ghost L3s.
"""

from __future__ import annotations

import logging
import time
from itertools import pairwise
from pathlib import Path
from typing import Any, TypedDict, cast

import whisper
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


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


class _L1Topic(BaseModel):
    title: str
    summary: str
    s: float
    e: float
    children: list[Any] = Field(max_length=0)


class _L1List(BaseModel):
    topics: list[_L1Topic]


class _SubdivideDecision(BaseModel):
    # `reason` is placed BEFORE `subdivide` on purpose: small structured-output
    # models produce more accurate booleans when they emit a short rationale
    # first. The reason is parsed but discarded by the orchestrator.
    reason: str
    subdivide: bool


class _L2Topic(BaseModel):
    title: str
    summary: str
    s: float
    e: float
    children: list[Any] = Field(max_length=0)


class _L2List(BaseModel):
    topics: list[_L2Topic]


class _L0Summary(BaseModel):
    title: str
    summary: str


_L1_SYS = (
    "You are a transcript outliner. You receive a recording's word-level "
    "transcript and must produce a flat list of top-level (L1) topics that "
    "summarize what was discussed.\n\n"
    "Rules:\n"
    "1. Produce a flat list of L1 topics. Do NOT nest subtopics here -- "
    "subtopics will be requested in a later pass. Every topic's children "
    "list must be empty.\n"
    "2. Each topic has a `title` (<=120 chars), a `summary` (2-4 sentences), "
    "and an interval [s, e] in seconds. `s` is the start of the FIRST word "
    "the topic covers; `e` is the end of the LAST word it covers. Topics "
    "typically span tens to hundreds of seconds, not single words.\n"
    "3. Sibling topics TILE the audio with no gaps and no overlap. Order "
    "siblings by time. For adjacent siblings A then B, set B.s = A.e. The "
    "first topic's s equals 0.0 (or the audio start); the last topic's e "
    "equals the audio end.\n"
    "4. Each piece of content appears at exactly ONE position. Do NOT "
    "duplicate the same topic across siblings.\n"
    "5. Use only timestamps that come from the transcript. Do not invent "
    "topics that are not present in the transcript.\n"
)
_L1_FEWSHOT = (
    "Example shape ONLY (illustrative -- DO NOT copy these timestamps; the "
    "example uses a fictitious 29-minute (1740s) cooking-show audio whose "
    "duration will be very different from yours). Properties to imitate: a "
    "flat list of L1 topics that TILE the full audio end-to-end; every "
    "children array is exactly []; no zero-duration intervals.\n"
    "{{\n"
    '  "topics": [\n'
    "    {{\n"
    '      "title": "Mise en place and equipment",\n'
    '      "summary": "Host walks through the prep workflow and the '
    'knives, pans, and timer setup needed before cooking begins.",\n'
    '      "s": 0.0, "e": 423.0, "children": []\n'
    "    }},\n"
    "    {{\n"
    '      "title": "Building the base sauce",\n'
    '      "summary": "Sweats aromatics, deglazes the pan, and reduces '
    'the stock to a base.",\n'
    '      "s": 423.0, "e": 911.0, "children": []\n'
    "    }},\n"
    "    {{\n"
    '      "title": "Searing and resting the protein",\n'
    '      "summary": "Pat-dry, high-heat sear, transfer to oven, then '
    'rest before slicing.",\n'
    '      "s": 911.0, "e": 1402.0, "children": []\n'
    "    }},\n"
    "    {{\n"
    '      "title": "Plating and tasting notes",\n'
    '      "summary": "Final assembly, seasoning adjustments, and the '
    "host's tasting commentary.\",\n"
    '      "s": 1402.0, "e": 1740.0, "children": []\n'
    "    }}\n"
    "  ]\n"
    "}}\n"
)
_L1_HUMAN = (
    "Audio runs from 0.00 to {audio_e} seconds (~{duration_min:.1f} min). "
    "Suggested topic count: ~{target_count} (calibrated to duration; do "
    "not over-split into 15+ tiny topics, do not collapse into a single "
    "topic).\n\n"
    "Follow this example shape exactly (your titles and summaries must "
    "come from the actual transcript below, not from this example):\n"
    "{fewshot}\n"
    "Time anchors (evenly-sampled words across the audio - use these to "
    "see the full duration and spread topics across it; do NOT cluster "
    "topics near the start):\n"
    "{anchors}\n\n"
    "Full transcript (one word per line as 's=12.34 word'):\n{transcript}"
)

_DECIDE_SYS = (
    "You decide whether a passage of speech is substantial enough to be "
    "broken into 2 or more distinct subtopics, or whether it covers a "
    "single coherent point that should NOT be subdivided.\n\n"
    "Rules:\n"
    "1. Set subdivide=true ONLY if you can identify at least 2 distinct, "
    "well-bounded subtopics inside the passage. Each subtopic must cover "
    "a meaningful span of speech (tens of seconds, not a few words).\n"
    "2. Set subdivide=false when the passage is on a single subject, when "
    "it is short, or when any split would be artificial.\n"
    "3. The `reason` field should be one short sentence describing why."
)
_DECIDE_HUMAN = (
    "Passage spans [{slice_s} -> {slice_e}] seconds (~{duration_min:.1f} min).\n\n"
    "Transcript (one word per line as 's=12.34 word'):\n{transcript}"
)

_L2_SYS = (
    "You are a transcript outliner. You receive a SLICE of a longer "
    "recording's word-level transcript and must produce the subtopics "
    "that subdivide this slice.\n\n"
    "Rules:\n"
    "1. Produce a flat list of subtopics. Every topic's children list must "
    "be empty.\n"
    "2. Each topic has a `title` (<=120 chars), a `summary` (2-4 sentences), "
    "and an interval [s, e] in seconds. `s` is the start of the FIRST word "
    "the topic covers; `e` is the end of the LAST word it covers.\n"
    "3. Sibling topics TILE the slice with no gaps and no overlap. Order "
    "siblings by time. For adjacent siblings A then B, set B.s = A.e. The "
    "first topic's s equals the slice start; the last topic's e equals "
    "the slice end.\n"
    "4. Use only timestamps that come from the transcript slice. Do not "
    "invent subtopics that are not present.\n"
)
_L2_FEWSHOT = (
    "Example shape ONLY (illustrative -- DO NOT copy these timestamps; the "
    "example uses a fictitious 6-minute slice from 1200.0 to 1560.0 "
    "seconds). Properties to imitate: a flat list of subtopics that tile "
    "the slice end-to-end; every children array is exactly [].\n"
    "{{\n"
    '  "topics": [\n'
    "    {{\n"
    '      "title": "Sear technique",\n'
    '      "summary": "Why a dry surface matters and how to avoid '
    'overcrowding the pan.",\n'
    '      "s": 1200.0, "e": 1340.0, "children": []\n'
    "    }},\n"
    "    {{\n"
    '      "title": "Resting and slicing",\n'
    '      "summary": "Rest time relative to thickness; slicing against '
    'the grain.",\n'
    '      "s": 1340.0, "e": 1480.0, "children": []\n'
    "    }},\n"
    "    {{\n"
    '      "title": "Plating",\n'
    '      "summary": "Final assembly and seasoning adjustments.",\n'
    '      "s": 1480.0, "e": 1560.0, "children": []\n'
    "    }}\n"
    "  ]\n"
    "}}\n"
)
_L2_HUMAN = (
    "Slice spans [{slice_s} -> {slice_e}] seconds (~{duration_min:.1f} min). "
    "Suggested subtopic count: ~{target_count}. Produce at least 2 "
    "subtopics that together tile the slice; if you genuinely cannot find "
    "2 distinct subjects, return exactly 2 anyway by splitting on the "
    "clearest natural break.\n\n"
    "Follow this example shape exactly (your titles and summaries must "
    "come from the slice's transcript below, not from this example):\n"
    "{fewshot}\n"
    "Time anchors (evenly-sampled words across the slice):\n"
    "{anchors}\n\n"
    "Slice transcript (one word per line as 's=12.34 word'):\n{transcript}"
)

_AGG_SYS = (
    "You are summarizing a whole audio from its top-level topics. Given "
    "the topics' titles and summaries, produce a single overall title "
    "(<=120 chars) and a 2-4 sentence summary capturing the unifying "
    "theme. Do not invent content beyond what the topics describe."
)
_AGG_HUMAN = "Top-level topics:\n{children}"


def _run_whisper(file: Path, *, model_size: str, language: str | None) -> list[WordSpan]:
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


def _format_children(children: list[TopicDict]) -> str:
    lines: list[str] = []
    for c in children:
        lines.append(f"- title: {c.get('title', '') or ''}")
        lines.append(f"  summary: {c.get('summary', '') or ''}")
    return "\n".join(lines)


def _time_anchors(spans: list[WordSpan], n: int = 10) -> str:
    real = [w for w in spans if str(w.get("w", "")).strip()]
    if not real:
        return "(empty slice)"
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


def build_tiered_agent(
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
    """Build a Runnable mapping audio file -> {transcription, topics:{topics:[L0]}}.

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
    }
    if ollama_base_url:
        base_kwargs["base_url"] = ollama_base_url

    l1_llm = ChatOllama(num_ctx=num_ctx_l1, **base_kwargs).with_structured_output(
        _L1List, method="json_schema"
    )
    decide_llm = ChatOllama(num_ctx=num_ctx_fanout, **base_kwargs).with_structured_output(
        _SubdivideDecision, method="json_schema"
    )
    l2_llm = ChatOllama(num_ctx=num_ctx_fanout, **base_kwargs).with_structured_output(
        _L2List, method="json_schema"
    )
    agg_llm = ChatOllama(num_ctx=num_ctx_fanout, **base_kwargs).with_structured_output(
        _L0Summary, method="json_schema"
    )

    l1_chain = (
        ChatPromptTemplate.from_messages([("system", _L1_SYS), ("human", _L1_HUMAN)]) | l1_llm
    )
    decide_chain = (
        ChatPromptTemplate.from_messages([("system", _DECIDE_SYS), ("human", _DECIDE_HUMAN)])
        | decide_llm
    )
    l2_chain = (
        ChatPromptTemplate.from_messages([("system", _L2_SYS), ("human", _L2_HUMAN)]) | l2_llm
    )
    agg_chain = (
        ChatPromptTemplate.from_messages([("system", _AGG_SYS), ("human", _AGG_HUMAN)]) | agg_llm
    )

    batch_cfg: RunnableConfig = {"max_concurrency": max_concurrency}

    def _extract_l1(spans: list[WordSpan], audio_e: float) -> list[TopicDict]:
        result = cast(
            "_L1List",
            l1_chain.invoke(
                {
                    "transcript": _format_transcript(spans),
                    "anchors": _time_anchors(spans),
                    "fewshot": _L1_FEWSHOT,
                    "audio_e": f"{audio_e:.2f}",
                    "duration_min": audio_e / 60.0,
                    "target_count": _target_count(0.0, audio_e),
                }
            ),
        )
        nodes: list[TopicDict] = [
            _new_node(t.s, t.e, title=t.title, summary=t.summary) for t in result.topics
        ]
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
                    "transcript": _format_transcript(sliced),
                    "slice_s": f"{ls:.2f}",
                    "slice_e": f"{le:.2f}",
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

    def _extract_l2_batch(
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
                    "transcript": _format_transcript(sliced),
                    "anchors": _time_anchors(sliced),
                    "fewshot": _L2_FEWSHOT,
                    "slice_s": f"{ls:.2f}",
                    "slice_e": f"{le:.2f}",
                    "duration_min": (le - ls) / 60.0,
                    "target_count": _target_count(ls, le),
                }
            )
        results = cast("list[_L2List]", l2_chain.batch(inputs, config=batch_cfg))
        out: list[list[TopicDict]] = []
        for r, l1 in zip(results, yes_l1_nodes, strict=True):
            ls = float(l1.get("s", 0.0))
            le = float(l1.get("e", 0.0))
            kids: list[TopicDict] = [
                _new_node(t.s, t.e, title=t.title, summary=t.summary) for t in r.topics
            ]
            _snap_tile(kids, ls, le)
            kids = _drop_zero(kids)
            out.append(kids)
        return out

    def _aggregate_l0(l1_nodes: list[TopicDict], audio_e: float) -> TopicDict:
        if not l1_nodes:
            return _new_node(0.0, audio_e, title="(empty)", summary="")
        result = cast(
            "_L0Summary",
            agg_chain.invoke({"children": _format_children(l1_nodes)}),
        )
        return _new_node(0.0, audio_e, title=result.title, summary=result.summary)

    def _build_tree(spans: list[WordSpan]) -> TranscriptionResult:
        if not spans:
            return {"transcription": spans, "topics": {"topics": []}}

        audio_e = _audio_end(spans)

        logger.info("Stage 2 (L1 extract): 1 call, num_ctx=%d", num_ctx_l1)
        t0 = time.time()
        l1_nodes = _extract_l1(spans, audio_e)
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
        logger.info("Stage 3b (L2 extract): %d batched calls", len(yes_l1_nodes))
        t0 = time.time()
        l2_lists = _extract_l2_batch(yes_l1_nodes, spans)
        for l1, kids in zip(yes_l1_nodes, l2_lists, strict=True):
            l1["children"] = kids
        l2_total = sum(len(k) for k in l2_lists)
        logger.info("Stage 3b done in %.1fs (%d L2 topics)", time.time() - t0, l2_total)

        logger.info("Stage 4 (L0 aggregate): 1 call")
        t0 = time.time()
        l0 = _aggregate_l0(l1_nodes, audio_e)
        l0["children"] = l1_nodes
        logger.info("Stage 4 done in %.1fs", time.time() - t0)

        return {"transcription": spans, "topics": {"topics": [l0]}}

    def _whisper_step(file: Path | str) -> list[WordSpan]:
        f = Path(file)
        logger.info("Stage 1 (Whisper): transcribing %s", f.name)
        t0 = time.time()
        spans = _run_whisper(f, model_size=whisper_model, language=language)
        logger.info("Stage 1 done in %.1fs (%d words)", time.time() - t0, len(spans))
        return spans

    return RunnableLambda(_whisper_step) | RunnableLambda(_build_tree)


def transcribe(file: Path | str, **kwargs: Any) -> TranscriptionResult:
    """Build the agent and invoke it once."""
    return build_tiered_agent(**kwargs).invoke(file)


__all__ = [
    "TopicDict",
    "TopicTree",
    "TranscriptionResult",
    "WordSpan",
    "build_tiered_agent",
    "transcribe",
]
