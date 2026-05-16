"""The topic agent keeps the Ollama model warm across stages and evicts
it once the run is done.

These tests fake ``autorag.agent.ChatOllama`` with a recorder so they can
assert *how* the chat clients are constructed and that the post-run
eviction call fires — including on the error path — without a live Ollama.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.runnables import RunnableLambda

from autorag import agent
from autorag.agent import (
    _Boundary,
    _BoundaryList,
    _L0Summary,
    _NodeSummary,
    _SubdivideDecision,
)

if TYPE_CHECKING:
    from autorag.types import WordSpan


def _spans() -> list[WordSpan]:
    # ~60 s, single speaker — short enough that Stage 3a/3b are skipped
    # (< the default min_subdivide_duration_s=120), minimising LLM calls.
    return [
        {"w": "hello", "s": 0.0, "e": 1.0, "segment_id": "0", "speaker": "0"},
        {"w": "world", "s": 1.0, "e": 2.0, "segment_id": "0", "speaker": "0"},
        {"w": "again", "s": 30.0, "e": 31.0, "segment_id": "1", "speaker": "0"},
        {"w": "bye", "s": 59.0, "e": 60.0, "segment_id": "1", "speaker": "0"},
    ]


_CANNED: dict[type, Any] = {
    _BoundaryList: _BoundaryList(topics=[_Boundary(s="00:00", e="01:00")]),
    _SubdivideDecision: _SubdivideDecision(reason="n/a", subdivide=False),
    _NodeSummary: _NodeSummary(title="T", summary="S"),
    _L0Summary: _L0Summary(title="Root", summary="About X"),
}


class _RecLLM:
    """Stand-in for ``ChatOllama``: records constructor kwargs and the
    bare-instance ``.invoke`` calls (the post-run eviction)."""

    def __init__(self, registry: list[_RecLLM], raise_for: type | None, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.structured_schema: type | None = None
        self.invocations = 0
        self._raise_for = raise_for
        registry.append(self)

    def with_structured_output(
        self, schema: type, *, method: str | None = None
    ) -> RunnableLambda[Any, Any]:
        self.structured_schema = schema

        def _run(_inp: Any) -> Any:
            if self._raise_for is not None and schema is self._raise_for:
                raise RuntimeError("boom in stage")
            return _CANNED[schema]

        return RunnableLambda(_run)

    def invoke(self, _inp: Any, *args: Any, **kwargs: Any) -> str:
        self.invocations += 1
        return "ok"


def _patch(monkeypatch: pytest.MonkeyPatch, raise_for: type | None = None) -> list[_RecLLM]:
    instances: list[_RecLLM] = []

    def _factory(**kwargs: Any) -> _RecLLM:
        return _RecLLM(instances, raise_for, **kwargs)

    monkeypatch.setattr(agent, "ChatOllama", _factory)
    return instances


def test_keeps_model_warm_and_evicts_after_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances = _patch(monkeypatch)

    tree = agent.generate_topics(_spans())

    # The pipeline still produces a well-formed tree.
    assert tree["topics"][0]["title"] == "Root"

    structured = [i for i in instances if i.structured_schema is not None]
    bare = [i for i in instances if i.structured_schema is None]

    # Every per-stage chat client is built warm at the unified context size.
    assert len(structured) == 5
    for i in structured:
        assert i.kwargs["keep_alive"] == "5m"
        assert i.kwargs["num_ctx"] == 8192

    # Exactly one bare client — the eviction call — built keep_alive=0,
    # num_predict=1, matching num_ctx, and invoked exactly once.
    assert len(bare) == 1
    (unload,) = bare
    assert unload.kwargs["keep_alive"] == 0
    assert unload.kwargs["num_predict"] == 1
    assert unload.kwargs["num_ctx"] == 8192
    assert unload.invocations == 1


def test_evicts_model_even_when_a_stage_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances = _patch(monkeypatch, raise_for=_BoundaryList)

    with pytest.raises(RuntimeError, match="boom in stage"):
        agent.generate_topics(_spans())

    # The finally-path eviction still fires and the original error propagates.
    (unload,) = [i for i in instances if i.structured_schema is None]
    assert unload.kwargs["keep_alive"] == 0
    assert unload.invocations == 1
