from __future__ import annotations

from autorag.embed import Embedder
from autorag.schemas import Retrieved
from autorag.store import VectorStore


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    def retrieve(self, question: str, top_k: int) -> list[Retrieved]:
        [query_vec] = self.embedder.embed_texts([question])
        return self.store.search(query_vec, top_k=top_k)
