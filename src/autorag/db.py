"""SQLite-backed database for audio clip transcription and topic storage."""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel
from pydantic_sqlite import DataBase

if TYPE_CHECKING:
    from pathlib import Path

_TABLE = "audio_clips"


class AudioClip(BaseModel):
    """One row of the ``audio_clips`` SQLite table.

    Fields ``transcription`` and ``topics`` are JSON-encoded strings;
    use :meth:`Database.get_clip` to fetch and decode them. The
    ``whisper_model`` / ``provider`` / ``llm_model`` columns are
    populated by :meth:`Database.finalize_topics` to record which
    backends produced the stored data.
    """

    id: str
    title: str
    file_path: str
    created_at: str
    transcription: str | None = None
    topics: str | None = None
    whisper_model: str | None = None
    provider: str | None = None
    llm_model: str | None = None


class Database:
    """Thin façade over :class:`pydantic_sqlite.DataBase` for AudioRAG state.

    Creates the SQLite file (and any missing parent directories) on
    construction. All methods are session-id keyed: callers derive a
    stable id (see :func:`autorag.persistence.derive_session_id`) and
    AutoRAG uses it as the primary key for upsert semantics — re-running
    a transcription overwrites the existing row instead of creating a
    duplicate.
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = DataBase(db_path)

    def _row(self, session_id: str) -> AudioClip | None:
        try:
            return cast("AudioClip", self.db.model_from_table(_TABLE, session_id))
        except KeyError:
            return None

    def add_analytics_event(
        self,
        session_id: str,
        *,
        category: str,
        message: str,
        metadata: dict[str, Any],
        marked_at_utc: Any,
    ) -> dict[str, Any]:
        """Build the analytics-event dict written into a clip's ``topics`` JSON.

        Does not touch the database itself — callers accumulate the
        returned dicts and pass them to :meth:`finalize_topics`.
        """
        event_id = str(uuid.uuid4())
        tx = metadata.get("transcription", {})
        return {
            "event_id": event_id,
            "category": category,
            "message": message,
            "level": int(tx.get("level") or 1),
            "start_s": float(tx.get("word_start_s") or 0.0),
            "number_label": str(tx.get("number_label") or ""),
            "summary": str(tx.get("summary") or ""),
            "marked_at_utc": marked_at_utc,
        }

    # --- CLI helpers ---

    def create_clip(
        self,
        session_id: str,
        *,
        title: str,
        file_path: str,
        created_at: str,
    ) -> None:
        """Create an :class:`AudioClip` row if one doesn't already exist.

        No-op when a row with this ``session_id`` is already present.
        Subsequent calls to :meth:`store_transcription` or
        :meth:`finalize_topics` overwrite the row in place.
        """
        if self._row(session_id) is not None:
            return
        clip = AudioClip(
            id=session_id,
            title=title,
            file_path=file_path,
            created_at=created_at,
        )
        self.db.add(_TABLE, clip, pk="id")

    def store_transcription(self, session_id: str, words: list[dict[str, Any]]) -> None:
        """Persist a JSON-encoded :class:`~autorag.types.WordSpan` list on the clip."""
        clip = self._row(session_id)
        if clip is None:
            return
        clip.transcription = json.dumps(words)
        self.db.add(_TABLE, clip, pk="id")

    def finalize_topics(
        self,
        session_id: str,
        transcript_end_s: float,
        *,
        events: list[dict[str, Any]],
        provider: str,
        llm_model: str,
        whisper_model: str,
    ) -> None:
        """Flatten topic events, compute durations, and write them to the clip.

        Within each L1/L2 level, ``duration_s`` is derived from the gap
        to the next sibling (or to ``transcript_end_s`` for the last
        node). The ``provider`` / ``llm_model`` / ``whisper_model``
        columns record which backends produced the data.
        """
        if not events:
            return

        by_level: dict[int, list[dict[str, Any]]] = {}
        for ev in events:
            by_level.setdefault(ev["level"], []).append(ev)

        for level_evs in by_level.values():
            level_evs.sort(key=lambda e: e["start_s"])
            for i, ev in enumerate(level_evs):
                if i + 1 < len(level_evs):
                    ev["duration_s"] = round(level_evs[i + 1]["start_s"] - ev["start_s"], 3)
                else:
                    ev["duration_s"] = round(max(0.0, transcript_end_s - ev["start_s"]), 3)

        topics = [
            {
                "title": ev["message"],
                "level": ev["level"],
                "start_s": ev["start_s"],
                "duration_s": ev.get("duration_s", 0.0),
                "number": ev["number_label"],
                "summary": ev.get("summary", ""),
            }
            for ev in events
        ]
        topics.sort(key=lambda t: (t["start_s"], t["level"]))

        clip = self._row(session_id)
        if clip is None:
            return
        clip.topics = json.dumps(topics)
        clip.provider = provider
        clip.llm_model = llm_model
        clip.whisper_model = whisper_model
        self.db.add(_TABLE, clip, pk="id")

    def get_clip(self, session_id: str) -> dict[str, Any] | None:
        """Return the clip as a plain dict, or ``None`` if missing."""
        clip = self._row(session_id)
        if not clip:
            return None
        return clip.model_dump()

    def list_clips(self) -> list[dict[str, Any]]:
        """Return every clip row as a plain dict (empty list if none / on error)."""
        try:
            inner = self.db._db  # pyright: ignore[reportPrivateUsage]
            if "audio_clips" not in inner.table_names():
                return []
            return [dict(row) for row in inner["audio_clips"].rows]
        except Exception:
            return []
