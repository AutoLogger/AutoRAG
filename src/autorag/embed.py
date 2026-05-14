"""Ollama-backed embedder for document and chunk text.

Wraps :class:`langchain_ollama.OllamaEmbeddings` and resolves its base
URL and model from the environment so the same defaults apply across
the CLI, SDK, and server:

* ``AUTORAG_OLLAMA_BASE_URL`` — defaults to ``http://localhost:11434``.
* ``AUTORAG_EMBED_MODEL``     — defaults to ``nomic-embed-text``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from langchain_ollama import OllamaEmbeddings

if TYPE_CHECKING:
    from autorag.schemas import Chunk


class Embedder:
    """Thin wrapper around :class:`OllamaEmbeddings`.

    Exposes :meth:`embed_texts` for raw strings and :meth:`embed_chunks`
    for in-place enrichment of :class:`~autorag.schemas.Chunk` objects.

    Args:
        base_url: Override the Ollama server URL. Falls back to the
            ``AUTORAG_OLLAMA_BASE_URL`` env var, then
            ``http://localhost:11434``.
        model:    Override the embedding model name. Falls back to
            ``AUTORAG_EMBED_MODEL``, then ``nomic-embed-text``.
    """

    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        resolved_base = (
            base_url or os.environ.get("AUTORAG_OLLAMA_BASE_URL", "http://localhost:11434")
        ).rstrip("/")
        resolved_model = model or os.environ.get("AUTORAG_EMBED_MODEL", "nomic-embed-text")
        self.base_url = resolved_base
        self.model = resolved_model
        self._embeddings = OllamaEmbeddings(base_url=resolved_base, model=resolved_model)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings, returning one vector per input.

        Raises :class:`RuntimeError` with the configured Ollama URL on
        connection or model errors. Empty input returns ``[]`` without
        touching the network.
        """
        if not texts:
            return []
        try:
            return self._embeddings.embed_documents(texts)
        except Exception as exc:
            raise RuntimeError(f"Ollama embedding request failed ({self.base_url}): {exc}") from exc

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Set ``chunk.embedding`` on every chunk in place; return the same list."""
        vectors = self.embed_texts([c.text for c in chunks])
        for c, v in zip(chunks, vectors, strict=True):
            c.embedding = v
        return chunks
