from __future__ import annotations

import json
import os
import urllib.request
from typing import TYPE_CHECKING, cast

import numpy as np
from chromadb import Documents, EmbeddingFunction, Embeddings

if TYPE_CHECKING:
    from autorag.schemas import Chunk


class Embedder:
    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        resolved_base = base_url or os.environ.get(
            "AUTOLOGGER_OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self.base_url = resolved_base.rstrip("/")
        self.model = model or os.environ.get("AUTOLOGGER_EMBED_MODEL", "nomic-embed-text")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = json.dumps({"model": self.model, "input": texts, "temperature": 0.0}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read())
        except Exception as exc:
            raise RuntimeError(f"Ollama embedding request failed ({self.base_url}): {exc}") from exc
        return cast("list[list[float]]", body["embeddings"])

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        vectors = self.embed_texts([c.text for c in chunks])
        for c, v in zip(chunks, vectors, strict=True):
            c.embedding = v
        return chunks


class EmbedderEmbeddingFunction(EmbeddingFunction[Documents]):
    """Adapt :class:`Embedder` to Chroma's ``EmbeddingFunction`` protocol."""

    def __init__(self, embedder: Embedder | None = None) -> None:
        self._embedder = embedder or Embedder()

    def __call__(self, input: Documents) -> Embeddings:
        vectors = self._embedder.embed_texts(list(input))
        return cast("Embeddings", [np.asarray(v, dtype=np.float32) for v in vectors])

    @staticmethod
    def name() -> str:
        return "autorag-ollama-embedder"
