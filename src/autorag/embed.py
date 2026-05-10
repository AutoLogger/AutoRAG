from __future__ import annotations

import os
from typing import TYPE_CHECKING

from langchain_ollama import OllamaEmbeddings

if TYPE_CHECKING:
    from autorag.schemas import Chunk


class Embedder:
    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        resolved_base = (
            base_url or os.environ.get("AUTORAG_OLLAMA_BASE_URL", "http://localhost:11434")
        ).rstrip("/")
        resolved_model = model or os.environ.get("AUTOLOGGER_EMBED_MODEL", "nomic-embed-text")
        self.base_url = resolved_base
        self.model = resolved_model
        self._embeddings = OllamaEmbeddings(base_url=resolved_base, model=resolved_model)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            return self._embeddings.embed_documents(texts)
        except Exception as exc:
            raise RuntimeError(f"Ollama embedding request failed ({self.base_url}): {exc}") from exc

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        vectors = self.embed_texts([c.text for c in chunks])
        for c, v in zip(chunks, vectors, strict=True):
            c.embedding = v
        return chunks
