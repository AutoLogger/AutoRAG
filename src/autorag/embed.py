from __future__ import annotations

from autorag.schemas import Chunk


class Embedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        vectors = self.embed_texts([c.text for c in chunks])
        for c, v in zip(chunks, vectors, strict=True):
            c.embedding = v
        return chunks
