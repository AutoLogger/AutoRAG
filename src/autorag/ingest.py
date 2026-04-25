from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from autorag.schemas import Chunk, Document


def load_documents(paths: list[str | Path]) -> list[Document]:
    raise NotImplementedError


def load_audio_clips(paths: list[str | Path]) -> list[dict[str, Any]]:
    raise NotImplementedError


def chunk_document(doc: Document, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    raise NotImplementedError
