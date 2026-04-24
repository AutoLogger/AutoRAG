from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from autorag.schemas import Chunk, Document


def load_documents(paths: list[str | Path]) -> list[Document]:
    raise NotImplementedError


def load_audio_clips(paths: list[str | Path]) -> list[dict]:
    raise NotImplementedError


def chunk_document(doc: Document, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    raise NotImplementedError


def _new_id() -> str:
    return uuid4().hex
