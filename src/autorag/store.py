"""Vector-store façade for the RAG pipeline.

Defines the :class:`VectorStore` interface that
:class:`~autorag.retrieve.Retriever` calls into. Concrete backends
(in-memory, Chroma, etc.) implement the four primitives; the topic-side
Chroma collection used by the ``/viz`` page lives separately in
:mod:`autorag.chroma_store`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autorag.schemas import Chunk, Retrieved


class VectorStore:
    """Abstract embedding-vector store.

    Subclasses provide a backend-specific implementation of each method.
    The interface is intentionally thin: the orchestration layer
    (:class:`~autorag.core.AutoRAG`) handles batching, persistence
    cadence, and tenant separation.
    """

    def add(self, chunks: list[Chunk]) -> None:
        """Insert chunks (with populated ``embedding`` fields) into the store."""
        raise NotImplementedError

    def search(self, query_embedding: list[float], top_k: int) -> list[Retrieved]:
        """Return the ``top_k`` nearest chunks to ``query_embedding``."""
        raise NotImplementedError

    def persist(self) -> None:
        """Flush in-memory state to durable storage."""
        raise NotImplementedError

    def load(self) -> None:
        """Restore previously persisted state."""
        raise NotImplementedError


class InMemoryStore(VectorStore):
    """Simple, non-persistent reference implementation.

    Stores chunks in a Python list. Useful for tests and small demos;
    not suitable for production retrieval workloads.
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk]) -> None:
        """Append chunks; no embedding-index structure is built."""
        self._chunks.extend(chunks)
