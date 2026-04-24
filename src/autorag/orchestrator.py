"""End-to-end Whisper -> LLM topic summarization orchestration.

Exposes `run_session_transcription(db, session_id, ...)`. The caller is
responsible for ensuring the three "Topic L1/L2/L3" categories exist and
passing their ids via `topic_category_ids`. The orchestrator never creates
categories itself.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from autorag import whisper_runner
from autorag.providers import (
    PROVIDER_DEFAULT_MODELS,
    Topic,
    TopicTree,
    WordSpan,
    get_provider,
)
from autorag.signatures import compute_audio_signature

if TYPE_CHECKING:
    from collections.abc import Generator

    from autorag.db import Database

logger = logging.getLogger(__name__)

# v1 safety cap: drop words beyond ~90 minutes into the session.
MAX_TRANSCRIPT_SECONDS = 90 * 60


# --------------------------------------------------------------------------- #
# Result types                                                                 #
# --------------------------------------------------------------------------- #


class TranscriptSegment(TypedDict):
    id: str
    started_at_utc: str | None
    words: list[dict[str, Any]]


class TranscriptPayload(TypedDict):
    segments: list[TranscriptSegment]


class SessionTranscriptionResult(TypedDict):
    inserted: int
    levels: list[int]
    transcript_cached: bool
    provider: Literal["anthropic", "openai", "gemini", "ollama"]
    llm_model: str
    device_used: str
    duration_secs: float
    timings: dict[str, float]
    word_spans: list[WordSpan]
    pending_events: list[dict[str, Any]]


# --------------------------------------------------------------------------- #
# Time helpers                                                                 #
# --------------------------------------------------------------------------- #


def _parse_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        s = str(value).strip()
        if not s:
            return None
        # Accept trailing Z suffix
        s_norm = s.replace("Z", "+00:00") if s.endswith("Z") else s
        try:
            dt = datetime.fromisoformat(s_norm)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _seconds_to_timedelta(seconds: float) -> timedelta:
    try:
        return timedelta(seconds=float(seconds))
    except (TypeError, ValueError, OverflowError):
        return timedelta(seconds=0)


# --------------------------------------------------------------------------- #
# Segment enrichment                                                           #
# --------------------------------------------------------------------------- #


def _resolve_segment_file(db: Any, session_id: str, segment_id: str) -> tuple[Path, str] | None:
    resolver = getattr(db, "get_audio_segment_file", None)
    if resolver is None:
        return None
    try:
        return resolver(session_id, segment_id)  # type: ignore[no-any-return]
    except Exception:  # pragma: no cover - defensive
        return None


def _enrich_segments(
    db: Any, session_id: str, segments: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Attach `file_path` and `file_size` to each segment (best effort)."""
    enriched: list[dict[str, Any]] = []
    for seg in segments:
        out = dict(seg)
        seg_id = str(seg.get("id") or "")
        resolved = _resolve_segment_file(db, session_id, seg_id)
        if resolved is not None:
            path, _mime = resolved
            out["file_path"] = str(path)
            try:
                out["file_size"] = int(Path(path).stat().st_size)
            except OSError:
                out["file_size"] = None
        else:
            out.setdefault("file_path", None)
            out.setdefault("file_size", None)
        enriched.append(out)
    return enriched


# --------------------------------------------------------------------------- #
# Transcript build                                                             #
# --------------------------------------------------------------------------- #


def _run_whisper_on_segments(
    segments: list[dict[str, Any]], whisper_model: str, language: str | None
) -> tuple[TranscriptPayload, float, float]:
    """Run Whisper over every segment and return (cache payload, model_load_s, transcription_s)."""
    t0 = time.perf_counter()
    model = whisper_runner.get_model(whisper_model)
    model_load_s = time.perf_counter() - t0

    transcription_s = 0.0
    out_segments: list[TranscriptSegment] = []
    for seg in segments:
        file_path = seg.get("file_path")
        if not file_path or not Path(file_path).is_file():
            logger.warning(
                "Skipping segment %s: file_path missing or not a file (%r).",
                seg.get("id"),
                file_path,
            )
            continue
        t1 = time.perf_counter()
        words = whisper_runner.transcribe_segment(model, file_path, language)
        transcription_s += time.perf_counter() - t1
        out_segments.append(
            {
                "id": str(seg.get("id") or ""),
                "started_at_utc": seg.get("started_at_utc"),
                "words": words,
            }
        )
    return {"segments": out_segments}, model_load_s, transcription_s


def _flatten_words(transcript: TranscriptPayload, audio_start_wall: datetime) -> list[WordSpan]:
    spans: list[WordSpan] = []
    for seg in transcript.get("segments", []) or []:
        seg_started = _parse_utc(seg.get("started_at_utc"))
        if seg_started is None:
            continue
        seg_offset = (seg_started - audio_start_wall).total_seconds()
        seg_id = str(seg.get("id") or "")
        for w in seg.get("words", []) or []:
            token = str(w.get("w", "") or "")
            if not token.strip():
                continue
            try:
                s = float(w.get("s", 0.0) or 0.0)
                e = float(w.get("e", s) or s)
            except (TypeError, ValueError):
                continue
            abs_s = seg_offset + s
            if abs_s > MAX_TRANSCRIPT_SECONDS:
                logger.warning(
                    "Transcript exceeds %ds cap; trimming remaining words.",
                    MAX_TRANSCRIPT_SECONDS,
                )
                return spans
            spans.append(
                {
                    "w": token,
                    "s": s,
                    "e": e,
                    "abs_s": abs_s,
                    "segment_id": seg_id,
                }
            )
    return spans


# --------------------------------------------------------------------------- #
# Topic fanout                                                                 #
# --------------------------------------------------------------------------- #


def _collapse_lone_children(tree: TopicTree) -> TopicTree:
    """Enforce that a subtopic level only exists if it has >=2 siblings.

    If a node has exactly one child, drop the lone child and promote its
    grandchildren to become direct children of the node. Applied recursively
    so cascading single-child chains collapse in one pass.
    """

    def walk(nodes: list[Topic]) -> list[Topic]:
        out: list[Topic] = []
        for node in nodes:
            children = list(node.get("children") or [])
            while len(children) == 1:
                lone = children[0]
                children = list(lone.get("children") or [])
            node["children"] = walk(children)
            out.append(node)
        return out

    return {"topics": walk(tree.get("topics") or [])}


def _iter_topics_flat(
    tree: TopicTree,
) -> Generator[tuple[int, Topic, str], None, None]:
    """Yield (level, topic_dict, number_label).

    `number_label` is a hierarchical sibling index like "1", "1.2", "1.2.3"
    that skips empty-title nodes so the numbering stays gap-free.
    """

    def walk(
        nodes: list[Topic], level: int, parent_number: str
    ) -> Generator[tuple[int, Topic, str], None, None]:
        sibling_count = 0
        for node in nodes:
            title = str(node.get("title", "") or "").strip()
            if not title:
                continue
            sibling_count += 1
            number_label = (
                str(sibling_count) if not parent_number else f"{parent_number}.{sibling_count}"
            )
            yield level, node, number_label
            children = node.get("children") or []
            if level < 3 and children:
                yield from walk(children, level + 1, number_label)

    yield from walk(tree.get("topics") or [], 1, "")


# --------------------------------------------------------------------------- #
# Main orchestrator                                                            #
# --------------------------------------------------------------------------- #


def run_session_transcription(
    db: Database,
    session_id: str,
    *,
    whisper_model: str,
    language: str | None,
    provider_name: Literal["anthropic", "openai", "gemini", "ollama"],
    llm_model: str,
    replace_existing: bool,
    force_retranscribe: bool,
    topic_category_ids: tuple[str, str, str],
) -> SessionTranscriptionResult:
    """Run the full transcribe -> summarize -> fanout pipeline."""
    start_wall_time = time.monotonic()
    timings: dict[str, float] = {}

    # 1. Enumerate segments.
    _t = time.perf_counter()
    segments = db.list_audio_segments(session_id)
    if not segments:
        raise ValueError("No audio segments for this session.")
    enriched = _enrich_segments(db, session_id, segments)
    timings["db_enumerate"] = time.perf_counter() - _t

    # 2. Audio signature for cache invalidation.
    _t = time.perf_counter()
    signature = compute_audio_signature(enriched)
    timings["audio_signature"] = time.perf_counter() - _t

    # 3. Transcript: cache or fresh?
    _t = time.perf_counter()
    cached = None
    try:
        cached = db.get_transcript(session_id)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("get_transcript(%s) failed: %s", session_id, exc)
        cached = None
    timings["cache_lookup"] = time.perf_counter() - _t

    transcript_cached = False
    transcript_payload: TranscriptPayload
    if (
        cached is not None
        and not force_retranscribe
        and str(cached.get("audio_signature")) == signature
        and isinstance(cached.get("transcript_json"), dict)
    ):
        transcript_payload = cached["transcript_json"]
        transcript_cached = True
        timings["whisper_model_load"] = 0.0
        timings["whisper_transcription"] = 0.0
        timings["db_upsert_transcript"] = 0.0
    else:
        transcript_payload, model_load_s, transcription_s = _run_whisper_on_segments(
            enriched, whisper_model, language
        )
        timings["whisper_model_load"] = model_load_s
        timings["whisper_transcription"] = transcription_s
        now_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        _t = time.perf_counter()
        try:
            db.upsert_transcript(
                session_id,
                whisper_model=whisper_model,
                language=language,
                audio_signature=signature,
                transcript_json=transcript_payload,  # type: ignore[arg-type]
                generated_at_utc=now_iso,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("upsert_transcript(%s) failed: %s", session_id, exc)
        timings["db_upsert_transcript"] = time.perf_counter() - _t

    # 4. Compute audio_start_wall_time as the earliest segment start.
    earliest: datetime | None = None
    for seg in enriched:
        dt = _parse_utc(seg.get("started_at_utc"))
        if dt is None:
            continue
        if earliest is None or dt < earliest:
            earliest = dt
    if earliest is None:
        # Fall back to "now" so at least the events get created coherently.
        earliest = datetime.now(UTC)

    # 5. Flatten and call the LLM provider.
    _t = time.perf_counter()
    word_spans = _flatten_words(transcript_payload, earliest)
    timings["word_flatten"] = time.perf_counter() - _t

    provider = get_provider(
        provider_name,
        model=llm_model or PROVIDER_DEFAULT_MODELS.get(provider_name, ""),
    )
    _t = time.perf_counter()
    try:
        tree = provider.summarize(word_spans, levels=3, prompt_extras="")
    except Exception as exc:
        raise RuntimeError(f"Provider {provider_name} call failed: {exc}") from exc
    timings["llm_summarize"] = time.perf_counter() - _t

    if not isinstance(tree, dict) or "topics" not in tree:
        raise RuntimeError(
            f"Provider {provider_name} returned malformed summary: missing top-level 'topics' field"
        )

    _t = time.perf_counter()
    tree = _collapse_lone_children(tree)
    timings["topic_collapse"] = time.perf_counter() - _t

    # 6. Fanout topics -> analytics events.
    _t = time.perf_counter()
    cat_by_level: dict[int, str] = {
        1: topic_category_ids[0],
        2: topic_category_ids[1],
        3: topic_category_ids[2],
    }
    inserted = 0
    level_counts = [0, 0, 0]
    pending_events: list[dict[str, Any]] = []

    for level, node, number_label in _iter_topics_flat(tree):
        title = str(node.get("title", "") or "").strip()
        if not title:
            continue
        try:
            start_s = float(node.get("start_s", 0.0) or 0.0)
        except (TypeError, ValueError):
            start_s = 0.0
        # Optional end_s if the provider supplied one
        word_start_s = start_s
        word_end_s: float | None = None
        raw_end = node.get("end_s")
        try:
            if raw_end is not None:
                word_end_s = float(raw_end)
        except (TypeError, ValueError):
            word_end_s = None

        summary = str(node.get("summary", "") or "").strip()
        metadata = {
            "transcription": {
                "level": level,
                "provider": provider_name,
                "model": llm_model,
                "number_label": number_label,
                "word_start_s": word_start_s,
                "word_end_s": word_end_s,
                "summary": summary,
            }
        }

        # Compute wall-clock timestamp for this topic.
        topic_offset = max(0.0, start_s)
        marked_at = earliest + _seconds_to_timedelta(topic_offset)

        category_id = cat_by_level.get(level)
        if not category_id:
            continue

        try:
            event = db.add_analytics_event(
                session_id,
                category=category_id,
                message=title,
                metadata=metadata,
                marked_at_utc=marked_at,
            )
            pending_events.append(event)
            inserted += 1
            if 1 <= level <= 3:
                level_counts[level - 1] += 1
        except Exception as exc:
            logger.warning(
                "add_analytics_event failed for topic %r (level=%d): %s",
                title,
                level,
                exc,
            )

    timings["db_fanout"] = time.perf_counter() - _t
    duration_secs = round(time.monotonic() - start_wall_time, 3)

    return {
        "inserted": inserted,
        "levels": level_counts,
        "transcript_cached": transcript_cached,
        "provider": provider_name,
        "llm_model": llm_model,
        "device_used": whisper_runner.resolved_device(),
        "duration_secs": duration_secs,
        "timings": timings,
        "word_spans": word_spans,
        "pending_events": pending_events,
    }


__all__ = ["run_session_transcription"]
