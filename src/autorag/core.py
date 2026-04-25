from __future__ import annotations

from typing import TYPE_CHECKING

from autorag.config import Settings, get_settings

if TYPE_CHECKING:
    from pathlib import Path

    from autorag.schemas import Chunk

from autorag.embed import Embedder
from autorag.generate import Generator
from autorag.ingest import chunk_document, load_documents
from autorag.retrieve import Retriever
from autorag.schemas import IngestResponse, QueryResponse
from autorag.store import InMemoryStore, VectorStore


class AutoRAG:
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
