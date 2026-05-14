"""Miscellaneous edge-case tests for ``AutoRAG``.

Covers three small corners that don't warrant their own files:

* :meth:`AutoRAG.persist_transcription` raises ``FileNotFoundError``
  when handed a path that isn't a file.
* :meth:`AutoRAG.build_agent` forwards kwargs to
  :func:`autorag.agent.build_agent` and returns its result.
* :meth:`AutoRAG._resolve_clip_identity` falls back to ``datetime.now``
  when ``Path.stat`` fails (e.g. the temp YouTube download has already
  been cleaned up).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from autorag.core import AutoRAG
from autorag.embed import Embedder


@pytest.fixture
def autorag_no_embed(monkeypatch: pytest.MonkeyPatch) -> AutoRAG:
    def _fake(self: Embedder, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setattr(Embedder, "embed_texts", _fake)
    return AutoRAG()


def test_persist_transcription_missing_file_raises(
    autorag_no_embed: AutoRAG, tmp_path: Path
) -> None:
    db_path = tmp_path / "autorag.sqlite"
    missing = tmp_path / "ghost.webm"
    with pytest.raises(FileNotFoundError):
        autorag_no_embed.persist_transcription(missing, [], db_path=db_path)


def test_build_agent_forwards_kwargs_and_returns_runnable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    seen: dict[str, Any] = {}

    def _fake_build(**kwargs: Any) -> object:
        seen.update(kwargs)
        return sentinel

    monkeypatch.setattr("autorag.agent.build_agent", _fake_build)

    result = AutoRAG().build_agent(whisper_model="tiny", llm_model="qwen2.5:0.5b")

    assert result is sentinel
    assert seen == {"whisper_model": "tiny", "llm_model": "qwen2.5:0.5b"}


def test_resolve_clip_identity_falls_back_to_now_on_oserror(
    autorag_no_embed: AutoRAG,
) -> None:
    """``_resolve_clip_identity`` is private; we call it directly because the
    OSError branch is only reachable through callers like ``persist_topics``
    that take a YouTube URL string in the ``file`` slot. Asserting through a
    public caller would mix this branch with chroma/SQLite side-effects.
    """
    before = datetime.now(tz=UTC)
    _session, audio_start, _stored = autorag_no_embed._resolve_clip_identity(
        Path("/definitely/does/not/exist.webm"),
        source_url=None,
        upload_date=None,
    )
    after = datetime.now(tz=UTC)

    assert before <= audio_start <= after
