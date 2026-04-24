from __future__ import annotations

from autorag.schemas import Retrieved


class Generator:
    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key

    def generate(self, question: str, context: list[Retrieved]) -> str:
        raise NotImplementedError

    def _build_prompt(self, question: str, context: list[Retrieved]) -> str:
        blocks = "\n\n".join(f"[{i}] {r.chunk.text}" for i, r in enumerate(context))
        return f"Context:\n{blocks}\n\nQuestion: {question}"
