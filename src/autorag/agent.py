"""LangChain agent that mimics the I/O contract of `cli._transcribe`.

Audio file in → transcription + topics out, as a deterministic LCEL chain
(`Runnable`). No DB, Chroma, or session side effects — this is a pure pipeline
intended for composition into other LangChain workflows.

The chain reuses the project's existing Whisper helpers and `OllamaProvider`,
so the prompt, structured-output schema, and validation match what the CLI
runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from autorag.providers import TopicTree, WordSpan, get_provider
from autorag.whisper_runner import get_model, transcribe_segment


class TranscriptionResult(TypedDict):
    transcription: list[WordSpan]
    topics: TopicTree


def _to_word_spans(raw: list[dict[str, Any]]) -> list[WordSpan]:
    spans: list[WordSpan] = []
    for w in raw:
        s = float(w["s"])
        span: WordSpan = {
            "w": str(w["w"]),
            "s": s,
            "e": float(w["e"]),
            "abs_s": s,
            "segment_id": "single",
        }
        spans.append(span)
    return spans


def build_transcription_agent(
    *,
    whisper_model: str = "base",
    language: str | None = None,
    llm_model: str = "llama3.1:8b",
    levels: int = 3,
) -> Runnable[Path | str, TranscriptionResult]:
    """Build a `Runnable` that maps an audio file path to transcription+topics."""

    def _transcribe_audio(state: dict[str, Any]) -> list[WordSpan]:
        file: Path = state["file"]
        if not file.exists():
            raise FileNotFoundError(f"audio file not found: {file}")
        model = get_model(whisper_model)
        raw = transcribe_segment(model, str(file), language or None)
        return _to_word_spans(raw)

    def _extract_topics(state: dict[str, Any]) -> TopicTree:
        provider = get_provider("ollama", model=llm_model)
        return provider.summarize(
            transcript=state["transcription"],
            levels=levels,
            prompt_extras="",
        )

    def _project(state: dict[str, Any]) -> TranscriptionResult:
        return {
            "transcription": state["transcription"],
            "topics": state["topics"],
        }

    pipeline: Runnable[dict[str, Any], TranscriptionResult] = (
        RunnablePassthrough.assign(transcription=RunnableLambda(_transcribe_audio))
        | RunnablePassthrough.assign(topics=RunnableLambda(_extract_topics))
        | RunnableLambda(_project)
    )

    def _wrap(file: Path | str) -> dict[str, Any]:
        return {"file": Path(file)}

    return RunnableLambda(_wrap) | pipeline


def transcribe(file: Path | str, **kwargs: Any) -> TranscriptionResult:
    """Convenience wrapper: build the agent and invoke it once."""
    return build_transcription_agent(**kwargs).invoke(file)


__all__ = [
    "TranscriptionResult",
    "build_transcription_agent",
    "transcribe",
]
