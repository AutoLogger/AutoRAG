"""Pydantic request/response and entity models for the RAG pipeline.

These models double as the on-the-wire schema for the HTTP API
(:mod:`autorag.api`) and as the in-process value types passed between
the embedder, store, retriever, and generator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from pydantic import BaseModel, Field


class Document(BaseModel):
    """One ingested source document, before chunking."""

    id: str
    source: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A retrieval-sized piece of a :class:`Document`.

    ``embedding`` is filled in by :class:`~autorag.embed.Embedder` and
    remains ``None`` until the chunk has been embedded.
    """

    id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None


class Retrieved(BaseModel):
    """A chunk plus its similarity score from a vector-store search."""

    chunk: Chunk
    score: float


class QueryRequest(BaseModel):
    """Request body for ``POST /query``."""

    question: str
    top_k: int | None = None


class QueryResponse(BaseModel):
    """Response body for ``POST /query``: generated answer plus its sources."""

    answer: str
    sources: list[Retrieved]


class IngestRequest(BaseModel):
    """Request body for ``POST /ingest``: filesystem paths to ingest."""

    paths: list[str | Path]


class IngestResponse(BaseModel):
    """Response body for ``POST /ingest``: counts of documents and chunks."""

    ingested: int
    chunks: int
