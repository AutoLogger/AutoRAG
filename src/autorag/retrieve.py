"""Vector-similarity retrieval over an ingested corpus.

Composes an :class:`~autorag.embed.Embedder` (to encode the query) with
a :class:`~autorag.store.VectorStore` (to do the nearest-neighbour
lookup) so that the choice of embedding model and backing store can
vary independently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autorag.embed import Embedder
    from autorag.schemas import Retrieved
    from autorag.store import VectorStore


class Retriever:
    """Embed a question and pull the ``top_k`` most similar chunks."""

    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    def retrieve(self, question: str, top_k: int) -> list[Retrieved]:
        """Return the ``top_k`` chunks most similar to ``question``."""
        [query_vec] = self.embedder.embed_texts([question])
        return self.store.search(query_vec, top_k=top_k)
