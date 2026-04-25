"""Hierarchical multi-pass topic extractor.

Where `reimagined_agent.py` asks the LLM to produce the entire 3-level topic
tree (boundaries + titles + summaries) in one structured-output call, this
module fans the work out into per-level / per-parent calls:

    ① Whisper                — once
    ② L1 boundary detection  — one LLM call (whole transcript → list of [s, e])
    ③ L2 / L3 boundary detection — one LLM call per parent (batched)
    ④ Leaf title + summary   — one LLM call per leaf, given that leaf's word slice
    ⑤ Aggregate-up titles    — one LLM call per non-leaf, given its children's
                                titles + summaries (no transcript)

Containment is structural: an L2 boundary call only sees the words inside
its L1's `[s, e]` interval, so its outputs are by construction inside the
parent. Sibling tiling is enforced by a small deterministic snap pass.
"""

from __future__ import annotations

import logging
import time
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from autorag.whisper_runner import get_model, transcribe_segment

if TYPE_CHECKING:
    from autorag.reimagined_agent import TopicDict, TranscriptionResult, WordSpan

logger = logging.getLogger(__name__)


class _Boundary(BaseModel):
    s: float
    e: float


class _BoundaryList(BaseModel):
    topics: list[_Boundary]


class _Summary(BaseModel):
    title: str
    summary: str


_BOUNDARY_SYS = (
    "You are a topic boundary detector. Given a transcript slice with "
    "word-level timestamps (one word per line in the form 's=12.34 word'), "
    "split it into ordered, non-overlapping topics that TILE the slice "
    "from start to end.\n\n"
    "Rules:\n"
    "1. Return ONLY intervals — no titles, no summaries.\n"
    "2. The first topic's `s` equals the slice start; the last topic's "
    "`e` equals the slice end.\n"
    "3. Adjacent topics tile end-to-start: for siblings A then B, B.s = A.e "
    "(no gaps, no overlap).\n"
    "4. Aim for roughly the suggested topic count — that count is calibrated "
    "to the slice duration. Do NOT over-split (returning 15+ topics for a "
    "few-minute slice is wrong); do NOT under-split (returning a single "
    "topic that covers the whole slice is also wrong unless the slice is "
    "very short or genuinely on one subject).\n"
    "5. Topics typically span tens to hundreds of seconds of speech.\n"
    "6. Use timestamp values that come directly from the transcript lines."
)
_BOUNDARY_HUMAN = (
    "Slice spans [{slice_s} -> {slice_e}] seconds "
    "(~{duration_min:.1f} min). Suggested topic count: ~{target_count}.\n\n"
    "Time anchors (evenly-sampled words across the slice — use these to see "
    "the full duration and spread topics across it; do NOT cluster all "
    "topics near the start):\n"
    "{anchors}\n\n"
    "Full transcript:\n{transcript}"
)

_LEAF_SYS = (
    "You are a summarizer of transcribed speech. Given a passage (one word "
    "per line as 's=… word'), produce a short title (≤120 chars) and a "
    "2-4 sentence summary describing what was said in the passage."
)
_LEAF_HUMAN = "Passage spans [{slice_s} → {slice_e}] seconds.\n\nTranscript:\n{transcript}"

_AGG_SYS = (
    "You are summarizing a parent topic from its subtopics. Given the "
    "subtopics' titles and summaries, produce the parent's title (≤120 "
    "chars) and a 2-4 sentence summary capturing the unifying theme. Do "
    "not invent content beyond what the subtopics describe."
)
_AGG_HUMAN = "Subtopics:\n{children}"


def _to_word_spans(raw: list[dict[str, Any]]) -> list[WordSpan]:
    spans: list[WordSpan] = []
    for w in raw:
        s = float(w["s"])
        spans.append(
            {
                "w": str(w["w"]),
                "s": s,
                "e": float(w["e"]),
                "abs_s": s,
                "segment_id": "single",
            }
        )
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


def _slice_spans(spans: list[WordSpan], s: float, e: float) -> list[WordSpan]:
    return [w for w in spans if s <= float(w.get("s", 0.0)) <= e]


def _time_anchors(spans: list[WordSpan], n: int = 10) -> str:
    """Pick ~n evenly-spaced word lines from the slice, formatted as anchors.

    Boundary calls on long slices anchor the LLM on the early timestamps it
    sees and stop reaching into the rest of the prompt. A short anchor block
    at the top of the prompt — `t=12.34s  word` lines spread evenly across
    the slice — gives the model explicit time references before it dives
    into the dense transcript, and noticeably reduces clustering of returned
    boundaries near the start.
    """
    if not spans:
        return "(empty slice)"
    # Filter out empty tokens up front so the indices we sample are real words.
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


def _audio_end(spans: list[WordSpan]) -> float:
    return max((float(w.get("e", 0.0)) for w in spans), default=0.0)


def _target_count(slice_s: float, slice_e: float) -> int:
    """Pick a target topic count based on slice duration.

    Roughly one topic per minute of speech, clamped to [2, 7]. Keeps the LLM
    from producing a 1-topic-per-15-seconds explosion or a 1-topic-covers-it-
    all collapse. The number is a suggestion in the prompt, not a constraint
    in the schema.
    """
    duration_s = max(0.0, slice_e - slice_s)
    target = round(duration_s / 60.0)
    return max(2, min(7, target))


def _snap_tile(boundaries: list[_Boundary], slice_s: float, slice_e: float) -> list[_Boundary]:
    """Clamp every value into the slice, sort by `s`, and tile siblings.

    Returns `[]` if the LLM gave no boundaries — the caller decides whether
    that means "treat parent as leaf" or "fall back to single span".

    The element-wise clamp guards against hallucinated timestamps (e.g. the
    LLM emitting 42526 for a 457s slice). Without this, an out-of-range mid-
    list value would slip past the first/last anchoring step below and end up
    in the final tree.
    """
    if not boundaries:
        return []
    for b in boundaries:
        b.s = max(slice_s, min(slice_e, b.s))
        b.e = max(slice_s, min(slice_e, b.e))
        if b.e < b.s:
            b.e = b.s
    sorted_b = sorted(boundaries, key=lambda b: b.s)
    sorted_b[0].s = slice_s
    sorted_b[-1].e = slice_e
    # Tile end-to-start: every cur.s is forced to equal prev.e — closing
    # both gaps (cur.s > prev.e) and overlaps (cur.s < prev.e). Then push
    # cur.e forward if it now sits behind cur.s.
    for prev, cur in pairwise(sorted_b):
        cur.s = prev.e
        if cur.e < cur.s:
            cur.e = cur.s
    return sorted_b


def _new_node(s: float, e: float) -> TopicDict:
    return {"title": "", "summary": "", "s": s, "e": e, "children": []}


def build_hierarchical_agent(
    *,
    whisper_model: str = "base",
    language: str | None = None,
    llm_model: str = "llama3.1:8b",
    ollama_base_url: str | None = None,
    max_concurrency: int = 4,
) -> Runnable[Path | str, TranscriptionResult]:
    """Build a `Runnable` that runs the multi-pass hierarchical pipeline."""

    # `keep_alive=0` tells Ollama to unload the model the moment the last
    # in-flight request for it finishes. During a batch the model stays
    # loaded (Ollama ref-counts in-flight requests), so this only kicks in
    # at stage boundaries — exactly when we no longer need that config.
    # `temperature=0.0` plus identical system prompts per chain plus
    # OLLAMA_MULTIUSER_CACHE=true on the server gives us prefix-cache hits
    # across all calls inside a single chain.
    base_kwargs: dict[str, Any] = {
        "model": llm_model,
        "temperature": 0.0,
        "keep_alive": 0,
    }
    if ollama_base_url:
        base_kwargs["base_url"] = ollama_base_url

    # L1 boundary detection runs once on the full transcript. The 7-min 3b1b
    # clip yields ~7K tokens of prompt; 16K gives comfortable headroom.
    # NOTE: Ollama with NUM_PARALLEL=4 reserves all 4 slots' KV-cache up front
    # at the configured num_ctx, so this also caps that pre-reservation:
    # 4 x 16K x f16 ≈ 10 GB KV + 9 GB model ≈ 19 GB total VRAM. Bumping to
    # 32K would push past 30 GB and OOM most consumer GPUs. For audio longer
    # than ~15 min the L1 prompt may exceed 16K — chunk first or raise here
    # only if you have the VRAM headroom.
    boundary_llm_l1 = ChatOllama(num_ctx=16384, **base_kwargs).with_structured_output(
        _BoundaryList, method="json_schema"
    )
    boundary_llm_fanout = ChatOllama(num_ctx=8192, **base_kwargs).with_structured_output(
        _BoundaryList, method="json_schema"
    )
    summary_llm = ChatOllama(num_ctx=8192, **base_kwargs).with_structured_output(
        _Summary, method="json_schema"
    )

    boundary_prompt = ChatPromptTemplate.from_messages(
        [("system", _BOUNDARY_SYS), ("human", _BOUNDARY_HUMAN)]
    )
    boundary_chain_l1 = boundary_prompt | boundary_llm_l1
    boundary_chain_fanout = boundary_prompt | boundary_llm_fanout
    leaf_chain = (
        ChatPromptTemplate.from_messages([("system", _LEAF_SYS), ("human", _LEAF_HUMAN)])
        | summary_llm
    )
    agg_chain = (
        ChatPromptTemplate.from_messages([("system", _AGG_SYS), ("human", _AGG_HUMAN)])
        | summary_llm
    )

    batch_cfg: RunnableConfig = {"max_concurrency": max_concurrency}

    def _detect_boundaries(
        spans: list[WordSpan], slice_s: float, slice_e: float
    ) -> list[_Boundary]:
        result = cast(
            "_BoundaryList",
            boundary_chain_l1.invoke(
                {
                    "transcript": _format_transcript(spans),
                    "anchors": _time_anchors(spans),
                    "slice_s": f"{slice_s:.2f}",
                    "slice_e": f"{slice_e:.2f}",
                    "duration_min": (slice_e - slice_s) / 60.0,
                    "target_count": _target_count(slice_s, slice_e),
                }
            ),
        )
        return _snap_tile(result.topics, slice_s, slice_e)

    def _detect_boundaries_batch(
        parents: list[TopicDict], spans: list[WordSpan]
    ) -> list[list[_Boundary]]:
        if not parents:
            return []
        inputs: list[dict[str, Any]] = []
        for p in parents:
            ps = float(p.get("s", 0.0))
            pe = float(p.get("e", 0.0))
            sliced = _slice_spans(spans, ps, pe)
            inputs.append(
                {
                    "transcript": _format_transcript(sliced),
                    "anchors": _time_anchors(sliced),
                    "slice_s": f"{ps:.2f}",
                    "slice_e": f"{pe:.2f}",
                    "duration_min": (pe - ps) / 60.0,
                    "target_count": _target_count(ps, pe),
                }
            )
        results = cast("list[_BoundaryList]", boundary_chain_fanout.batch(inputs, config=batch_cfg))
        out: list[list[_Boundary]] = []
        for r, p in zip(results, parents, strict=True):
            ps = float(p.get("s", 0.0))
            pe = float(p.get("e", 0.0))
            out.append(_snap_tile(r.topics, ps, pe))
        return out

    def _summarize_leaves(leaves: list[TopicDict], spans: list[WordSpan]) -> list[_Summary]:
        if not leaves:
            return []
        inputs: list[dict[str, Any]] = []
        for leaf in leaves:
            ls = float(leaf.get("s", 0.0))
            le = float(leaf.get("e", 0.0))
            inputs.append(
                {
                    "transcript": _format_transcript(_slice_spans(spans, ls, le)),
                    "slice_s": f"{ls:.2f}",
                    "slice_e": f"{le:.2f}",
                }
            )
        return cast("list[_Summary]", leaf_chain.batch(inputs, config=batch_cfg))

    def _aggregate_internals(internals: list[TopicDict]) -> list[_Summary]:
        if not internals:
            return []
        inputs = [{"children": _format_children(node.get("children") or [])} for node in internals]
        return cast("list[_Summary]", agg_chain.batch(inputs, config=batch_cfg))

    def _build_full_tree(spans: list[WordSpan]) -> TranscriptionResult:
        if not spans:
            return {"transcription": spans, "topics": {"topics": []}}

        audio_e = _audio_end(spans)

        # ② L1 boundaries
        logger.info("Stage 2 (L1 boundaries): 1 call, num_ctx=16K")
        t0 = time.time()
        l1_bounds = _detect_boundaries(spans, 0.0, audio_e)
        if not l1_bounds:
            l1_bounds = [_Boundary(s=0.0, e=audio_e)]
        l1_nodes: list[TopicDict] = [_new_node(b.s, b.e) for b in l1_bounds]
        logger.info("Stage 2 done in %.1fs (%d L1 topics)", time.time() - t0, len(l1_nodes))

        # ③ pass 1 — L2 boundaries (one call per L1)
        logger.info("Stage 3a (L2 boundaries): %d parallel calls", len(l1_nodes))
        t0 = time.time()
        l2_lists = _detect_boundaries_batch(l1_nodes, spans)
        for l1, l2_bounds in zip(l1_nodes, l2_lists, strict=True):
            l1["children"] = [_new_node(b.s, b.e) for b in l2_bounds]
        l2_total = sum(len(x) for x in l2_lists)
        logger.info("Stage 3a done in %.1fs (%d L2 topics)", time.time() - t0, l2_total)

        # ③ pass 2 — L3 boundaries (one call per L2)
        l2_flat = [l2 for l1 in l1_nodes for l2 in (l1.get("children") or [])]
        logger.info("Stage 3b (L3 boundaries): %d parallel calls", len(l2_flat))
        t0 = time.time()
        l3_lists = _detect_boundaries_batch(l2_flat, spans)
        for l2, l3_bounds in zip(l2_flat, l3_lists, strict=True):
            l2["children"] = [_new_node(b.s, b.e) for b in l3_bounds]
        l3_total = sum(len(x) for x in l3_lists)
        logger.info("Stage 3b done in %.1fs (%d L3 topics)", time.time() - t0, l3_total)

        # ④ Collect leaves (no-children at any depth) and summarize them
        leaves: list[TopicDict] = []
        for l1 in l1_nodes:
            l1_kids = l1.get("children") or []
            if not l1_kids:
                leaves.append(l1)
                continue
            for l2 in l1_kids:
                l2_kids = l2.get("children") or []
                if not l2_kids:
                    leaves.append(l2)
                    continue
                leaves.extend(l2_kids)
        logger.info("Stage 4 (leaf summaries): %d parallel calls", len(leaves))
        t0 = time.time()
        leaf_summaries = _summarize_leaves(leaves, spans)
        for leaf, summ in zip(leaves, leaf_summaries, strict=True):
            leaf["title"] = summ.title
            leaf["summary"] = summ.summary
        logger.info("Stage 4 done in %.1fs", time.time() - t0)

        # ⑤ Aggregate up — deepest internals first (L2s with L3 kids, then L1s)
        internal_l2s = [
            l2 for l1 in l1_nodes for l2 in (l1.get("children") or []) if l2.get("children")
        ]
        logger.info("Stage 5a (L2 aggregation): %d parallel calls", len(internal_l2s))
        t0 = time.time()
        l2_summaries = _aggregate_internals(internal_l2s)
        for l2, summ in zip(internal_l2s, l2_summaries, strict=True):
            l2["title"] = summ.title
            l2["summary"] = summ.summary
        logger.info("Stage 5a done in %.1fs", time.time() - t0)

        internal_l1s = [l1 for l1 in l1_nodes if l1.get("children")]
        logger.info("Stage 5b (L1 aggregation): %d parallel calls", len(internal_l1s))
        t0 = time.time()
        l1_summaries = _aggregate_internals(internal_l1s)
        for l1, summ in zip(internal_l1s, l1_summaries, strict=True):
            l1["title"] = summ.title
            l1["summary"] = summ.summary
        logger.info("Stage 5b done in %.1fs", time.time() - t0)

        return {"transcription": spans, "topics": {"topics": l1_nodes}}

    def _whisper_step(file: Path | str) -> list[WordSpan]:
        f = Path(file)
        if not f.exists():
            raise FileNotFoundError(f"audio file not found: {f}")
        logger.info("Stage 1 (Whisper): transcribing %s", f.name)
        t0 = time.time()
        model = get_model(whisper_model)
        raw = transcribe_segment(model, str(f), language or None)
        spans = _to_word_spans(raw)
        logger.info("Stage 1 done in %.1fs (%d words)", time.time() - t0, len(spans))
        return spans

    return RunnableLambda(_whisper_step) | RunnableLambda(_build_full_tree)


def transcribe(file: Path | str, **kwargs: Any) -> TranscriptionResult:
    """Build the agent and invoke it once."""
    return build_hierarchical_agent(**kwargs).invoke(file)


__all__ = [
    "build_hierarchical_agent",
    "transcribe",
]
