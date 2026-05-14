"""Tests for the cache-miss branch of :meth:`AutoRAG.transcribe_blocks`.

The cache-hit / canonical-URL / force-retranscribe paths are covered in
``tests/test_transcribe_blocks_sqlite.py``. This module fills the
remaining hole — the first-call path that runs ``transcribe``,
persists, and returns formatted blocks — without invoking Whisper or
yt-dlp. ``resolve_audio_input`` is replaced with a fake context manager
that yields a local-file ``AudioSource``; ``AutoRAG.transcribe`` is
replaced with a recording stub.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import pytest

from autorag import format_blocks
from autorag.audio_source import AudioSource
from autorag.core import AutoRAG
from autorag.db import Database
from autorag.embed import Embedder

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from autorag.types import WordSpan


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


def _canned_words() -> list[WordSpan]:
    return [
        {"w": "hello", "s": 1.0, "e": 1.5, "speaker": "0"},
        {"w": "there", "s": 1.5, "e": 2.0, "speaker": "0"},
        {"w": "hi", "s": 12.0, "e": 12.5, "speaker": "1"},
    ]


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    audio: Path,
    *,
    src_title: str | None = None,
) -> dict[str, int]:
    """Patch ``resolve_audio_input`` + ``AutoRAG.transcribe`` for the cache-miss path.

    Returns a counter dict so callers can assert how many times each fake
    was invoked.
    """
    counts = {"transcribe": 0, "resolve": 0}

    @contextlib.contextmanager
    def _fake_resolve(source: Path | str) -> Iterator[AudioSource]:
        counts["resolve"] += 1
        yield AudioSource(
            path=audio,
            source_url=None,
            video_id=None,
            title=src_title,
            upload_date=None,
            duration_s=None,
            uploader=None,
        )

    monkeypatch.setattr("autorag.audio_source.resolve_audio_input", _fake_resolve)

    def _fake_transcribe(self: AutoRAG, file: object, **_kwargs: Any) -> list[WordSpan]:
        counts["transcribe"] += 1
        return _canned_words()

    monkeypatch.setattr(AutoRAG, "transcribe", _fake_transcribe)
    return counts


def test_cache_miss_invokes_transcribe_and_persists(
    autorag_no_embed: AutoRAG, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = _make_audio_file(tmp_path)
    db_path = tmp_path / "autorag.sqlite"
    counts = _install_fakes(monkeypatch, audio)

    out = autorag_no_embed.transcribe_blocks(audio, seconds=10, db_path=db_path)

    expected = format_blocks(_canned_words(), 10)
    assert out == expected
    assert counts["transcribe"] == 1
    assert counts["resolve"] == 1

    # Second call must hit the cache and skip transcribe entirely.
    out2 = autorag_no_embed.transcribe_blocks(audio, seconds=10, db_path=db_path)
    assert out2 == expected
    assert counts["transcribe"] == 1


def test_cache_miss_title_resolution_caller_wins(
    autorag_no_embed: AutoRAG, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = _make_audio_file(tmp_path)
    db_path = tmp_path / "autorag.sqlite"
    _install_fakes(monkeypatch, audio, src_title="from-yt-dlp")

    autorag_no_embed.transcribe_blocks(audio, seconds=10, db_path=db_path, title="caller-title")

    db = Database(db_path)
    clips = db.list_clips()
    assert len(clips) == 1
    assert clips[0]["title"] == "caller-title"


def test_cache_miss_title_resolution_falls_back_to_source(
    autorag_no_embed: AutoRAG, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = _make_audio_file(tmp_path)
    db_path = tmp_path / "autorag.sqlite"
    _install_fakes(monkeypatch, audio, src_title="from-yt-dlp")

    autorag_no_embed.transcribe_blocks(audio, seconds=10, db_path=db_path)

    db = Database(db_path)
    clips = db.list_clips()
    assert len(clips) == 1
    assert clips[0]["title"] == "from-yt-dlp"


def test_cache_miss_title_resolution_defaults_to_stem(
    autorag_no_embed: AutoRAG, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = _make_audio_file(tmp_path)
    db_path = tmp_path / "autorag.sqlite"
    _install_fakes(monkeypatch, audio, src_title=None)

    autorag_no_embed.transcribe_blocks(audio, seconds=10, db_path=db_path)

    db = Database(db_path)
    clips = db.list_clips()
    assert len(clips) == 1
    # default_title_from(str(audio)) == audio.stem == "clip"
    assert clips[0]["title"] == "clip"
