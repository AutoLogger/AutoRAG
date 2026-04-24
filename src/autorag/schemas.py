from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pathlib import Path


class Document(BaseModel):
    id: str
    source: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None


class Retrieved(BaseModel):
    chunk: Chunk
    score: float


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[Retrieved]


class IngestRequest(BaseModel):
    paths: list[str | Path]


class IngestResponse(BaseModel):
    ingested: int
    chunks: int
