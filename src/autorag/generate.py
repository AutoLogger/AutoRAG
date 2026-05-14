"""LLM response generation over retrieved context.

Defines the :class:`Generator` interface. Concrete backends (Ollama,
hosted APIs) override :meth:`Generator.generate` to call the model of
their choice with the prompt assembled by :meth:`Generator._build_prompt`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autorag.schemas import Retrieved


class Generator:
    """Base interface for RAG response generation.

    Subclasses must implement :meth:`generate`. The base class provides
    a default ``[idx] chunk`` prompt assembly via
    :meth:`_build_prompt` that subclasses may reuse.
    """

    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key

    def generate(self, question: str, context: list[Retrieved]) -> str:
        """Return a natural-language answer to ``question`` grounded in ``context``."""
        raise NotImplementedError

    def _build_prompt(self, question: str, context: list[Retrieved]) -> str:
        blocks = "\n\n".join(f"[{i}] {r.chunk.text}" for i, r in enumerate(context))
        return f"Context:\n{blocks}\n\nQuestion: {question}"
