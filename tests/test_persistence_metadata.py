"""Verify that ``AutoRAG.persist_transcription`` records the YouTube info-dict
metadata (upload date, canonical URL) on the stored clip row instead of
falling back to the temp-file mtime / path.
"""

from __future__ import annotations

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


def test_youtube_metadata_anchors_created_at_and_file_path(
    autorag_no_embed: AutoRAG, tmp_path: Path
) -> None:
    audio = _make_audio_file(tmp_path)
    db_path = tmp_path / "autorag.sqlite"
    url = "https://youtu.be/dQw4w9WgXcQ"

    out = autorag_no_embed.persist_transcription(
        audio,
        _result(),  # type: ignore[arg-type]
        db_path=db_path,
        source_url=url,
        upload_date="20091025",
        duration_s=213.0,
    )

    clip = out["clip"]
    assert clip is not None
    assert clip["created_at"] == "2009-10-25T00:00:00Z"
    assert clip["file_path"] == _canonical_youtube_url(url)


def test_persist_without_metadata_keeps_legacy_behavior(
    autorag_no_embed: AutoRAG, tmp_path: Path
) -> None:
    audio = _make_audio_file(tmp_path)
    db_path = tmp_path / "autorag.sqlite"

    out = autorag_no_embed.persist_transcription(
        audio,
        _result(),  # type: ignore[arg-type]
        db_path=db_path,
    )

    clip = out["clip"]
    assert clip is not None
    assert clip["file_path"] == str(audio.resolve())
    assert clip["created_at"].endswith("Z")
