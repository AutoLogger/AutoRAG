"""Topic-tree → SQLite/Chroma persistence helpers.

Pure functions extracted from the CLI so the SDK's
:meth:`autorag.core.AutoRAG.persist_transcription` can reuse them.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator
    from datetime import datetime

    from autorag.db import Database
    from autorag.types import TopicDict, TopicTree, WordSpan

logger = logging.getLogger(__name__)


def derive_session_id(file_or_url: str | Path) -> str:
    """Compute the same ``session_id`` :meth:`AutoRAG.persist_transcription`
    would write.

    Mirrors the inline logic in :meth:`AutoRAG.persist_transcription`:
      - YouTube URL → ``uuid5(NAMESPACE_URL, _canonical_youtube_url(url))``
      - Local Path → ``uuid5(NAMESPACE_URL, str(path.resolve()))``

    Only ``autorag.audio_source`` is imported (base-safe; ``yt_dlp`` stays
    behind its own lazy import). Safe to call without ``[audio]``/``[rag]``.
    """
    from autorag.audio_source import _canonical_youtube_url, is_youtube_url

    if isinstance(file_or_url, str) and is_youtube_url(file_or_url):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, _canonical_youtube_url(file_or_url)))
    path = Path(file_or_url)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(path.resolve())))


def load_transcription(db: Database, session_id: str) -> list[WordSpan] | None:
    """Return the stored word list for ``session_id``, or ``None`` if the row
    is missing or has no transcription.

    Parses the JSON string written by :meth:`Database.store_transcription`.
    Reads via raw ``sqlite_utils`` (matching :meth:`Database.list_clips`)
    so a freshly-opened :class:`Database` instance can read rows it didn't
    write — ``pydantic_sqlite``'s model registry is in-memory only.
    """
    inner = db.db._db  # pyright: ignore[reportPrivateUsage]
    try:
        if "audio_clips" not in inner.table_names():
            return None
        rows = list(inner["audio_clips"].rows_where("id = ?", [session_id]))
    except Exception:
        return None
    if not rows:
        return None
    raw = rows[0].get("transcription")
    if raw is None:
        return None
    return list(json.loads(raw))


def collapse_lone_children(tree: TopicTree) -> TopicTree:
    """Drop single-child chains so a subtopic level only exists with >=2 siblings."""

    def walk(nodes: list[TopicDict]) -> list[TopicDict]:
        out: list[TopicDict] = []
        for node in nodes:
            children = list(node.get("children") or [])
            while len(children) == 1:
                lone = children[0]
                children = list(lone.get("children") or [])
            node["children"] = walk(children)
            out.append(node)
        return out

    return {"topics": walk(tree.get("topics") or [])}


def iter_topics_flat(tree: TopicTree) -> Generator[tuple[int, TopicDict, str], None, None]:
    """Yield (level, node, number_label) like '1', '1.2', '1.2.3'."""

    def walk(
        nodes: list[TopicDict], level: int, parent_number: str
    ) -> Generator[tuple[int, TopicDict, str], None, None]:
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


def topics_to_events(
    db: Database,
    session_id: str,
    tree: TopicTree,
    *,
    audio_start: datetime,
    provider: str,
    llm_model: str,
    topic_category_ids: tuple[str, str, str],
) -> list[dict[str, Any]]:
    """Walk the topic tree and produce analytics events for each titled node.

    Reads the hierarchical-agent's ``s`` / ``e`` keys (not ``start_s`` / ``end_s``).
    """
    cat_by_level = {1: topic_category_ids[0], 2: topic_category_ids[1], 3: topic_category_ids[2]}
    events: list[dict[str, Any]] = []

    for level, node, number_label in iter_topics_flat(tree):
        title = str(node.get("title", "") or "").strip()
        if not title:
            continue
        try:
            start_s = float(node.get("s", 0.0) or 0.0)
        except (TypeError, ValueError):
            start_s = 0.0
        word_end_s: float | None = None
        raw_end = node.get("e")
        try:
            if raw_end is not None:
                word_end_s = float(raw_end)
        except (TypeError, ValueError):
            word_end_s = None

        summary = str(node.get("summary", "") or "").strip()
        metadata: dict[str, Any] = {
            "transcription": {
                "level": level,
                "provider": provider,
                "model": llm_model,
                "number_label": number_label,
                "word_start_s": start_s,
                "word_end_s": word_end_s,
                "summary": summary,
            }
        }

        marked_at = audio_start + timedelta(seconds=max(0.0, start_s))
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
            events.append(event)
        except Exception as exc:
            logger.warning(
                "add_analytics_event failed for topic %r (level=%d): %s", title, level, exc
            )

    return events
