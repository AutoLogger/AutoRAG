from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autorag.schemas import Chunk, Retrieved


class VectorStore:
    def add(self, chunks: list[Chunk]) -> None:
        raise NotImplementedError

    def search(self, query_embedding: list[float], top_k: int) -> list[Retrieved]:
        raise NotImplementedError

    def persist(self) -> None:
        raise NotImplementedError

    def load(self) -> None:
        raise NotImplementedError


class InMemoryStore(VectorStore):
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk]) -> None:
        self._chunks.extend(chunks)
