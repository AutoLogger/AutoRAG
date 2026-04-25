"""SQLite-backed database for audio clip transcription and topic storage."""

from __future__ import annotations

import json
import mimetypes
import uuid
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel
from pydantic_sqlite import DataBase

_TABLE = "audio_clips"


class AudioClip(BaseModel):
    id: str
    title: str
    file_path: str
    created_at: str
    audio_signature: str | None = None
    transcription: str | None = None
    whisper_cache: str | None = None
    topics: str | None = None
    whisper_model: str | None = None
    provider: str | None = None
    llm_model: str | None = None
    embeddings: str | None = None


class Database:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = DataBase(db_path)

    def _row(self, session_id: str) -> AudioClip | None:
        try:
            return cast("AudioClip", self.db.model_from_table(_TABLE, session_id))
        except KeyError:
            return None

    # --- orchestrator duck-typed interface ---

    def list_audio_segments(self, session_id: str) -> list[dict[str, Any]]:
        row = self._row(session_id)
        if not row:
            return []
        return [{"id": "0", "started_at_utc": row.created_at}]

    def get_audio_segment_file(self, session_id: str, segment_id: str) -> tuple[Path, str] | None:
        row = self._row(session_id)
        if not row:
            return None
        path = Path(row.file_path)
        mime = mimetypes.guess_type(str(path))[0] or "audio/webm"
        return (path, mime)

    def get_transcript(self, session_id: str) -> dict[str, Any] | None:
        row = self._row(session_id)
        if not row or not row.whisper_cache or not row.audio_signature:
            return None
        return {
            "audio_signature": row.audio_signature,
            "transcript_json": json.loads(row.whisper_cache),
        }

    def upsert_transcript(
        self,
        session_id: str,
        *,
        whisper_model: str,
        language: str | None,
        audio_signature: str,
        transcript_json: dict[str, Any],
        generated_at_utc: str,
    ) -> None:
        clip = self._row(session_id)
        if clip is None:
            return
        clip.audio_signature = audio_signature
        clip.whisper_cache = json.dumps(transcript_json)
        clip.whisper_model = whisper_model
        self.db.add(_TABLE, clip, pk="id")

    def add_analytics_event(
        self,
        session_id: str,
        *,
        category: str,
        message: str,
        metadata: dict[str, Any],
        marked_at_utc: Any,
    ) -> dict[str, Any]:
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

    def store_embeddings(self, session_id: str, embeddings: list[list[float]]) -> None:
        clip = self._row(session_id)
        if clip is None:
            return
        clip.embeddings = json.dumps(embeddings)
        self.db.add(_TABLE, clip, pk="id")

    def get_clip(self, session_id: str) -> dict[str, Any] | None:
        clip = self._row(session_id)
        if not clip:
            return None
        return clip.model_dump()

    def list_clips(self) -> list[dict[str, Any]]:
        try:
            inner = self.db._db  # pyright: ignore[reportPrivateUsage]
            if "audio_clips" not in inner.table_names():
                return []
            return [dict(row) for row in inner["audio_clips"].rows]
        except Exception:
            return []
