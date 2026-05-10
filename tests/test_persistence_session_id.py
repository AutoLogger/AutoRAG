"""Verify that ``AutoRAG.persist_transcription`` derives a deterministic
``session_id`` for YouTube-sourced clips, so re-fetching the same URL
replaces the existing clip rather than creating a duplicate.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import pytest

from autorag.audio_source import _canonical_youtube_url
from autorag.core import AutoRAG
from autorag.embed import Embedder

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def autorag_no_embed(monkeypatch: pytest.MonkeyPatch) -> AutoRAG:
    """`AutoRAG` whose embedder returns canned vectors (skips Ollama)."""

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
            {"w": "hello", "s": 0.0, "e": 0.5, "abs_s": 0.0, "speaker": "0"},
            {"w": "world", "s": 0.5, "e": 1.0, "abs_s": 0.5, "speaker": "0"},
        ],
        "topics": {
            "topics": [
                {
                    "title": "root",
                    "summary": "root summary",
                    "s": 0.0,
                    "e": 1.0,
                    "children": [
                        {
                            "title": "intro",
                            "summary": "opening",
                            "s": 0.0,
                            "e": 1.0,
                        }
                    ],
                }
            ]
        },
    }


def test_session_id_stable_across_calls_for_same_youtube_url(
    autorag_no_embed: AutoRAG, tmp_path: Path
) -> None:
    audio_a = _make_audio_file(tmp_path / "first")
    audio_b = _make_audio_file(tmp_path / "second")
    db_path = tmp_path / "autorag.sqlite"

    url_a = "https://youtu.be/dQw4w9WgXcQ"
    url_b = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    out_a = autorag_no_embed.persist_transcription(
        audio_a,
        _result(),  # type: ignore[arg-type]
        db_path=db_path,
        source_url=url_a,
    )
    out_b = autorag_no_embed.persist_transcription(
        audio_b,
        _result(),  # type: ignore[arg-type]
        db_path=db_path,
        source_url=url_b,
    )

    assert out_a["session_id"] == out_b["session_id"]
    expected = str(uuid.uuid5(uuid.NAMESPACE_URL, "https://www.youtube.com/watch?v=dQw4w9WgXcQ"))
    assert out_a["session_id"] == expected
    assert out_a["session_id"] == str(uuid.uuid5(uuid.NAMESPACE_URL, _canonical_youtube_url(url_a)))


def test_session_id_falls_back_to_path_when_no_source_url(
    autorag_no_embed: AutoRAG, tmp_path: Path
) -> None:
    audio = _make_audio_file(tmp_path)
    db_path = tmp_path / "autorag.sqlite"

    out = autorag_no_embed.persist_transcription(
        audio,
        _result(),  # type: ignore[arg-type]
        db_path=db_path,
    )

    expected = str(uuid.uuid5(uuid.NAMESPACE_URL, str(audio.resolve())))
    assert out["session_id"] == expected
