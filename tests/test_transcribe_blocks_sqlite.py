"""Integration tests for :meth:`AutoRAG.transcribe_blocks`.

Exercise the cache-first lookup, YouTube canonicalization, and force
re-transcribe path without invoking Whisper / Ollama. Embeddings are
monkeypatched the same way as ``tests/test_persistence_session_id.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from autorag.core import AutoRAG
from autorag.embed import Embedder

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def autorag_no_embed(monkeypatch: pytest.MonkeyPatch) -> AutoRAG:
    def _fake(self: Embedder, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setattr(Embedder, "embed_texts", _fake)
    return AutoRAG()


def _make_audio_file(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    audio = parent / "clip.webm"
    audio.write_bytes(b"fake-audio-bytes")
    return audio


def _result() -> dict[str, Any]:
    return {
        "transcription": [
            {"w": "hello", "s": 1.0, "e": 1.5, "abs_s": 1.0, "speaker": "0"},
            {"w": "there", "s": 1.5, "e": 2.0, "abs_s": 1.5, "speaker": "0"},
            {"w": "hi", "s": 12.0, "e": 12.5, "abs_s": 12.0, "speaker": "1"},
        ],
        "topics": {
            "topics": [
                {
                    "title": "root",
                    "summary": "root summary",
                    "s": 0.0,
                    "e": 13.0,
                    "children": [{"title": "intro", "summary": "opening", "s": 0.0, "e": 13.0}],
                }
            ]
        },
    }


def test_reads_from_sqlite_when_present(autorag_no_embed: AutoRAG, tmp_path: Path) -> None:
    audio = _make_audio_file(tmp_path)
    db_path = tmp_path / "autorag.sqlite"

    autorag_no_embed.persist_transcription(
        audio,
        _result(),  # type: ignore[arg-type]
        db_path=db_path,
    )

    out = autorag_no_embed.transcribe_blocks(audio, seconds=10, db_path=db_path)
    expected = "00:01-00:02 Speaker 1: hello there\n\n00:12-00:12 Speaker 2: hi"
    assert out == expected


def test_url_uses_canonical_session_id(
    autorag_no_embed: AutoRAG, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = _make_audio_file(tmp_path)
    db_path = tmp_path / "autorag.sqlite"

    autorag_no_embed.persist_transcription(
        audio,
        _result(),  # type: ignore[arg-type]
        db_path=db_path,
        source_url="https://youtu.be/dQw4w9WgXcQ",
    )

    def _fail(*_args: object, **_kwargs: object) -> dict[str, Any]:
        msg = "transcribe must not run when the cache is warm"
        raise AssertionError(msg)

    monkeypatch.setattr(AutoRAG, "transcribe", _fail)

    out = autorag_no_embed.transcribe_blocks(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        seconds=10,
        db_path=db_path,
    )
    expected = "00:01-00:02 Speaker 1: hello there\n\n00:12-00:12 Speaker 2: hi"
    assert out == expected


def test_force_retranscribe_bypasses_cache(
    autorag_no_embed: AutoRAG, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = _make_audio_file(tmp_path)
    db_path = tmp_path / "autorag.sqlite"

    autorag_no_embed.persist_transcription(
        audio,
        _result(),  # type: ignore[arg-type]
        db_path=db_path,
    )

    called = {"n": 0}

    def _sentinel(self: AutoRAG, file: object, **_kwargs: object) -> dict[str, Any]:
        called["n"] += 1
        return _result()

    monkeypatch.setattr(AutoRAG, "transcribe", _sentinel)

    autorag_no_embed.transcribe_blocks(audio, seconds=10, db_path=db_path, force_retranscribe=True)
    assert called["n"] == 1


def test_zero_seconds_raises(autorag_no_embed: AutoRAG, tmp_path: Path) -> None:
    audio = _make_audio_file(tmp_path)
    db_path = tmp_path / "autorag.sqlite"
    with pytest.raises(ValueError):
        autorag_no_embed.transcribe_blocks(audio, seconds=0, db_path=db_path)
