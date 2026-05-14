"""Document loading and chunking primitives for the RAG pipeline.

These functions form the boundary between filesystem inputs (text
files, PDFs, audio clips) and the structured
:class:`~autorag.schemas.Document` / :class:`~autorag.schemas.Chunk`
shapes consumed by the embedder and vector store.

The current implementations are stubs that raise
:class:`NotImplementedError`; concrete loaders are wired up via
:class:`autorag.core.AutoRAG`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from autorag.schemas import Chunk, Document


def load_documents(paths: list[str | Path]) -> list[Document]:
    """Load text documents from disk into :class:`~autorag.schemas.Document` records."""
    raise NotImplementedError


def load_audio_clips(paths: list[str | Path]) -> list[dict[str, Any]]:
    """Load audio clip metadata for transcript-based ingestion."""
    raise NotImplementedError


def chunk_document(doc: Document, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Split a document into overlapping :class:`~autorag.schemas.Chunk` records."""
    raise NotImplementedError
