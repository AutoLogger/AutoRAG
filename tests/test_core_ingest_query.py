"""Wiring-contract tests for :meth:`AutoRAG.ingest` and :meth:`AutoRAG.query`.

The underlying primitives (``load_documents``, ``chunk_document``,
``Generator.generate``, ``VectorStore.{add,search}``) are
``NotImplementedError`` stubs today; these tests don't assert end-to-end
RAG behaviour, they pin the orchestration contract that every concrete
implementation must honour:

  ``ingest``: load → chunk(size, overlap from settings) → embed → store
  ``query``:  retrieve(question, top_k) → generate(question, sources)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from autorag.config import Settings
from autorag.core import AutoRAG
from autorag.embed import Embedder
from autorag.generate import Generator
from autorag.retrieve import Retriever
from autorag.schemas import Chunk, Document, IngestResponse, QueryResponse, Retrieved
from autorag.store import VectorStore

if TYPE_CHECKING:
    from pathlib import Path


class _FakeStore(VectorStore):
    def __init__(self, search_result: list[Retrieved] | None = None) -> None:
        self.added: list[Chunk] = []
        self.search_calls: list[tuple[list[float], int]] = []
        self._search_result = search_result or []

    def add(self, chunks: list[Chunk]) -> None:
        self.added.extend(chunks)

    def search(self, query_embedding: list[float], top_k: int) -> list[Retrieved]:
        self.search_calls.append((query_embedding, top_k))
        return list(self._search_result)


class _FakeEmbedder(Embedder):
    def __init__(self) -> None:
        # Skip the real OllamaEmbeddings construction; we don't touch the network.
        self.base_url = "http://fake"
        self.model = "fake-embed"
        self.embed_texts_calls: list[list[str]] = []
        self.embed_chunks_calls: list[list[Chunk]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.embed_texts_calls.append(list(texts))
        return [[0.1, 0.2] for _ in texts]

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        self.embed_chunks_calls.append(list(chunks))
        for c in chunks:
            c.embedding = [0.1, 0.2]
        return chunks


class _FakeGenerator(Generator):
    def __init__(self) -> None:
        super().__init__(model="fake-model")
        self.calls: list[tuple[str, list[Retrieved]]] = []

    def generate(self, question: str, context: list[Retrieved]) -> str:
        self.calls.append((question, list(context)))
        return "fake-answer"


def _doc(doc_id: str) -> Document:
    return Document(id=doc_id, source=f"{doc_id}.txt", text=f"text of {doc_id}")


def _chunks_for(doc: Document, n: int) -> list[Chunk]:
    return [Chunk(id=f"{doc.id}-c{i}", doc_id=doc.id, text=f"{doc.text} #{i}") for i in range(n)]


def test_ingest_orchestrates_load_chunk_embed_store(monkeypatch: pytest.MonkeyPatch) -> None:
    docs = [_doc("a"), _doc("b")]

    def _load(_paths: list[str | Path]) -> list[Document]:
        return list(docs)

    monkeypatch.setattr("autorag.core.load_documents", _load)

    chunk_calls: list[dict[str, Any]] = []

    def _chunk(doc: Document, *, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
        chunk_calls.append(
            {"doc_id": doc.id, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
        )
        return _chunks_for(doc, 3)

    monkeypatch.setattr("autorag.core.chunk_document", _chunk)

    store = _FakeStore()
    embedder = _FakeEmbedder()
    settings = Settings(chunk_size=777, chunk_overlap=111)
    autorag = AutoRAG(settings=settings, store=store, embedder=embedder)

    result = autorag.ingest(["a.txt", "b.txt"])

    assert result == IngestResponse(ingested=2, chunks=6)
    assert [c["doc_id"] for c in chunk_calls] == ["a", "b"]
    assert all(c["chunk_size"] == 777 and c["chunk_overlap"] == 111 for c in chunk_calls)
    assert len(embedder.embed_chunks_calls) == 1
    assert len(embedder.embed_chunks_calls[0]) == 6
    assert len(store.added) == 6
    assert all(c.embedding == [0.1, 0.2] for c in store.added)


def test_ingest_with_zero_docs(monkeypatch: pytest.MonkeyPatch) -> None:
    def _load_empty(_paths: list[str | Path]) -> list[Document]:
        return []

    def _chunk_should_not_run(
        _doc: Document, *, chunk_size: int, chunk_overlap: int
    ) -> list[Chunk]:
        pytest.fail("chunk_document must not run when there are no documents")

    monkeypatch.setattr("autorag.core.load_documents", _load_empty)
    monkeypatch.setattr("autorag.core.chunk_document", _chunk_should_not_run)

    store = _FakeStore()
    embedder = _FakeEmbedder()
    autorag = AutoRAG(store=store, embedder=embedder)
    result = autorag.ingest([])

    assert result == IngestResponse(ingested=0, chunks=0)
    # embed_chunks is called with [] so the contract is preserved, but nothing lands in the store.
    assert store.added == []


def test_query_calls_retriever_then_generator() -> None:
    chunk = Chunk(id="c1", doc_id="d1", text="grass is green")
    sources = [Retrieved(chunk=chunk, score=0.99)]
    store = _FakeStore(search_result=sources)
    embedder = _FakeEmbedder()
    generator = _FakeGenerator()
    autorag = AutoRAG(store=store, embedder=embedder, generator=generator)

    response = autorag.query("what colour is grass?")

    assert response == QueryResponse(answer="fake-answer", sources=sources)
    assert embedder.embed_texts_calls == [["what colour is grass?"]]
    assert len(store.search_calls) == 1
    assert generator.calls == [("what colour is grass?", sources)]


def test_query_top_k_override_beats_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _FakeStore(search_result=[])
    embedder = _FakeEmbedder()
    generator = _FakeGenerator()
    settings = Settings(top_k=5)
    autorag = AutoRAG(settings=settings, store=store, embedder=embedder, generator=generator)

    seen_top_k: list[int] = []
    original_retrieve = Retriever.retrieve

    def _spy(self: Retriever, question: str, top_k: int) -> list[Retrieved]:
        seen_top_k.append(top_k)
        return original_retrieve(self, question, top_k=top_k)

    monkeypatch.setattr(Retriever, "retrieve", _spy)

    autorag.query("q1", top_k=7)
    autorag.query("q2")

    assert seen_top_k == [7, 5]
