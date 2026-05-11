from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from autorag.blocks import format_blocks
from autorag.config import Settings, get_settings
from autorag.embed import Embedder
from autorag.errors import MissingExtraError, _missing_extra
from autorag.generate import Generator
from autorag.ingest import chunk_document, load_documents
from autorag.retrieve import Retriever
from autorag.schemas import IngestResponse, QueryResponse
from autorag.store import InMemoryStore, VectorStore

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable

    from autorag.schemas import Chunk
    from autorag.types import TranscriptionResult, WordSpan

logger = logging.getLogger(__name__)

__all__ = ["AutoRAG", "MissingExtraError"]


class AutoRAG:
    """Unified facade for the audio→topics agent and the document-RAG pipeline.

    Heavy dependencies (whisper, torch, pyannote, chromadb, ...) are loaded
    lazily on first use, so a base install can import :class:`AutoRAG` without
    pulling them. Methods raise :class:`MissingExtraError` with the specific
    extras hint when an extra is missing.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        store: VectorStore | None = None,
        embedder: Embedder | None = None,
        generator: Generator | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or InMemoryStore()
        self.embedder = embedder or Embedder()
        self.generator = generator or Generator(model=self.settings.model)
        self.retriever = Retriever(self.store, self.embedder)

    # ── Document RAG ──────────────────────────────────────────────────────

    def ingest(self, paths: list[str | Path]) -> IngestResponse:
        docs = load_documents(paths)
        all_chunks: list[Chunk] = []
        for doc in docs:
            all_chunks.extend(
                chunk_document(
                    doc,
                    chunk_size=self.settings.chunk_size,
                    chunk_overlap=self.settings.chunk_overlap,
                )
            )
        self.embedder.embed_chunks(all_chunks)
        self.store.add(all_chunks)
        return IngestResponse(ingested=len(docs), chunks=len(all_chunks))

    def query(self, question: str, top_k: int | None = None) -> QueryResponse:
        k = top_k or self.settings.top_k
        retrieved = self.retriever.retrieve(question, top_k=k)
        answer = self.generator.generate(question, retrieved)
        return QueryResponse(answer=answer, sources=retrieved)

    # ── Audio → topics ────────────────────────────────────────────────────

    def transcribe(
        self,
        file: Path | str,
        *,
        whisper_model: str = "base",
        llm_model: str = "qwen2.5:14b-instruct-q8_0",
        language: str | None = None,
    ) -> TranscriptionResult:
        """Run Whisper + LLM topic extraction on an audio file or YouTube URL.

        ``file`` is either a local audio file path or a YouTube URL
        (``youtube.com``, ``youtu.be``, ``m.youtube.com``,
        ``music.youtube.com``). YouTube URLs are downloaded to a temporary
        ``.webm`` for the duration of the call.

        Requires ``pip install 'autorag[audio,diarize]'`` for transcription,
        plus ``[youtube]`` when passing a URL. Returns the raw
        ``{transcription, topics}`` dict. Use :meth:`persist_transcription`
        to write to SQLite + Chroma (separate ``[rag]`` extra).
        """
        try:
            from autorag.agent import transcribe as _agent_transcribe
            from autorag.audio_source import resolve_audio_input
        except ModuleNotFoundError as exc:
            raise _missing_extra("audio,diarize", exc) from exc

        with resolve_audio_input(file) as src:
            return _agent_transcribe(
                src.path,
                whisper_model=whisper_model,
                llm_model=llm_model,
                language=language,
            )

    def build_agent(self, **kwargs: Any) -> Runnable[Path | str, TranscriptionResult]:
        """Return the LangChain :class:`Runnable` for batched / streaming use.

        Same extras as :meth:`transcribe`. Forwards ``**kwargs`` to
        :func:`autorag.agent.build_agent`.
        """
        try:
            from autorag.agent import build_agent as _agent_build
        except ModuleNotFoundError as exc:
            raise _missing_extra("audio,diarize", exc) from exc
        return _agent_build(**kwargs)

    def transcribe_blocks(
        self,
        file: Path | str,
        seconds: int = 10,
        *,
        force_retranscribe: bool = False,
        db_path: Path | None = None,
        whisper_model: str = "base",
        llm_model: str = "qwen2.5:14b-instruct-q8_0",
        language: str | None = None,
        title: str | None = None,
        provider: str = "ollama",
    ) -> str:
        """Return the transcription formatted as N-second time blocks.

        Resolution order:
          1. ``session_id = derive_session_id(file)``.
          2. If SQLite has a row for ``session_id`` with a non-null
             ``transcription`` and ``force_retranscribe`` is False, decode
             it and format — returns immediately (no ``[audio]`` needed).
          3. Else run :meth:`transcribe` then :meth:`persist_transcription`
             (matching the CLI ``transcribe`` command), then format the
             freshly produced words.

        Each non-empty bucket emits one line per speaker turn,
        ``MM:SS-MM:SS Speaker K: <words>``. See
        :func:`autorag.blocks.format_blocks` for the full algorithm.

        Requires ``pip install 'autorag[rag]'`` for the cached path;
        ``[audio,diarize]`` (+ ``[youtube]`` for URLs) on cache miss.
        """
        if seconds <= 0:
            raise ValueError("seconds must be a positive integer")

        try:
            from autorag.audio_source import default_title_from
            from autorag.db import Database
            from autorag.persistence import derive_session_id, load_transcription
        except ModuleNotFoundError as exc:
            raise _missing_extra("rag", exc) from exc

        session_id = derive_session_id(file)
        resolved_db = (db_path or self.settings.db_path).expanduser()
        db = Database(resolved_db)

        if not force_retranscribe:
            cached = load_transcription(db, session_id)
            if cached is not None:
                return format_blocks(cached, seconds)

        try:
            from autorag.audio_source import resolve_audio_input
        except ModuleNotFoundError as exc:
            raise _missing_extra("audio,diarize", exc) from exc

        source_str = file if isinstance(file, str) else str(file)
        with resolve_audio_input(file) as src:
            result = self.transcribe(
                src.path,
                whisper_model=whisper_model,
                llm_model=llm_model,
                language=language,
            )
            resolved_title = title or src.title or default_title_from(source_str)
            self.persist_transcription(
                src.path,
                result,
                title=resolved_title,
                provider=provider,
                llm_model=llm_model,
                whisper_model=whisper_model,
                db_path=db_path,
                source_url=src.source_url,
                upload_date=src.upload_date,
                duration_s=src.duration_s,
            )
        return format_blocks(result["transcription"], seconds)

    def persist_transcription(
        self,
        file: Path | str,
        result: TranscriptionResult,
        *,
        title: str | None = None,
        provider: str = "ollama",
        llm_model: str = "qwen2.5:14b-instruct-q8_0",
        whisper_model: str = "base",
        db_path: Path | None = None,
        source_url: str | None = None,
        upload_date: str | None = None,
        duration_s: float | None = None,
    ) -> dict[str, Any]:
        """Write a transcription + topic tree to SQLite (clip + words + events) and
        index topic-title embeddings into Chroma. Returns a dict with the stored
        clip row plus a ``timings`` breakdown.

        Requires ``pip install 'autorag[rag]'`` (chromadb + pydantic_sqlite).
        ``whisper_model`` is recorded as metadata only.

        ``source_url`` (optional) is the original input URL when ``file`` is a
        local copy of remote content (e.g. a yt-dlp download). When supplied,
        the clip's ``session_id`` is seeded from the canonical URL instead of
        the local path, so re-fetching the same URL replaces the existing
        clip rather than creating a duplicate. ``file_path`` in the stored row
        is also set to the canonical URL so it remains valid after the temp
        download is gone.

        ``upload_date`` (optional, ``"YYYYMMDD"`` from yt-dlp) anchors the
        clip's ``created_at`` (and absolute event timestamps) to when the
        video was published, instead of when the temp file was written.
        ``duration_s`` is currently informational and not persisted.
        """
        del duration_s  # informational; no schema column for it yet
        try:
            from autorag.audio_source import _canonical_youtube_url, is_youtube_url
            from autorag.chroma_store import ChromaStore, default_chroma_dir
            from autorag.db import Database
            from autorag.persistence import (
                collapse_lone_children,
                topics_to_events,
            )
        except ModuleNotFoundError as exc:
            raise _missing_extra("rag", exc) from exc

        path = Path(file)
        if not path.is_file():
            raise FileNotFoundError(f"{path} is not a file.")

        resolved_db = (db_path or self.settings.db_path).expanduser()
        db = Database(resolved_db)

        canonical_source_url: str | None = None
        if source_url is not None and is_youtube_url(source_url):
            canonical_source_url = _canonical_youtube_url(source_url)
            session_id = str(uuid.uuid5(uuid.NAMESPACE_URL, canonical_source_url))
        elif source_url is not None:
            canonical_source_url = source_url
            session_id = str(uuid.uuid5(uuid.NAMESPACE_URL, source_url))
        else:
            session_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(path.resolve())))

        clip_title = title or path.stem

        if upload_date:
            uploaded_at = datetime.strptime(upload_date, "%Y%m%d").replace(tzinfo=UTC)
            created_at = uploaded_at.isoformat().replace("+00:00", "Z")
            audio_start = uploaded_at
        else:
            mtime = path.stat().st_mtime
            created_at = datetime.fromtimestamp(mtime, tz=UTC).isoformat().replace("+00:00", "Z")
            audio_start = datetime.fromtimestamp(mtime, tz=UTC)

        stored_file_path = canonical_source_url or str(path.resolve())

        db.create_clip(
            session_id,
            title=clip_title,
            file_path=stored_file_path,
            created_at=created_at,
        )

        words: list[WordSpan] = result["transcription"]
        topic_tree = collapse_lone_children(result["topics"])

        t = time.perf_counter()
        db.store_transcription(session_id, words)  # type: ignore[arg-type]
        store_words_s = time.perf_counter() - t

        transcript_end_s = 0.0
        if words:
            last = words[-1]
            transcript_end_s = last.get("abs_s", 0.0) + (last.get("e", 0.0) - last.get("s", 0.0))

        t = time.perf_counter()
        pending_events = topics_to_events(
            db,
            session_id,
            topic_tree,
            audio_start=audio_start,
            provider=provider,
            llm_model=llm_model,
            topic_category_ids=("l1", "l2", "l3"),
        )
        db.finalize_topics(
            session_id,
            transcript_end_s,
            events=pending_events,
            provider=provider,
            llm_model=llm_model,
            whisper_model=whisper_model,
        )
        finalize_s = time.perf_counter() - t

        t = time.perf_counter()
        clip_data = db.get_clip(session_id)
        if clip_data and clip_data.get("topics"):
            topics = [t for t in json.loads(clip_data["topics"]) if t.get("title")]
            texts = [
                f"{t['title']}. {t['summary']}" if t.get("summary") else t["title"] for t in topics
            ]
            if texts:
                try:
                    embeddings = self.embedder.embed_texts(texts)
                    chroma = ChromaStore(default_chroma_dir(resolved_db))
                    chroma.delete_clip(session_id)
                    chroma.add_topic_embeddings(
                        session_id,
                        str(clip_data.get("title", "")),
                        topics,
                        embeddings,
                    )
                except Exception as exc:
                    logger.warning("embedding/index failed: %s", exc)
        embed_s = time.perf_counter() - t

        clip = db.get_clip(session_id)
        return {
            "clip": clip,
            "session_id": session_id,
            "timings": {
                "store_words": store_words_s,
                "finalize": finalize_s,
                "embed": embed_s,
            },
        }
