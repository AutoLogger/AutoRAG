"""Tests for :meth:`AutoRAG.persist_topics`.

Covers the happy path, the three ``transcript_end_s`` branches
(explicit / derived-from-words / zero fallback), the YouTube-URL guard
that skips the file-existence check, the ``FileNotFoundError`` raised
for non-existent local files, and the embedding-failure swallow at
:mod:`autorag.core` lines 418-419.

The embedder is monkeypatched per the project's
``autorag_no_embed`` pattern; the SQLite + Chroma backends run for real
in ``tmp_path``.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import pytest

from autorag.chroma_store import ChromaStore, default_chroma_dir
from autorag.core import AutoRAG
from autorag.embed import Embedder

if TYPE_CHECKING:
    from pathlib import Path

    from autorag.types import TopicTree, WordSpan


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


def _topic_tree() -> TopicTree:
    return {
        "topics": [
            {
                "title": "root",
                "summary": "root summary",
                "s": 0.0,
                "e": 10.0,
                "children": [
                    {"title": "intro", "summary": "opening", "s": 0.0, "e": 5.0},
                    {"title": "outro", "summary": "closing", "s": 5.0, "e": 10.0},
                ],
            }
        ]
    }


def _words(end_s: float) -> list[WordSpan]:
    return [
        {"w": "hello", "s": 0.0, "e": 0.5, "speaker": "0"},
        {"w": "world", "s": end_s - 0.5, "e": end_s, "speaker": "0"},
    ]


def _root_duration(clip_topics_json: str | None) -> float:
    """Return ``duration_s`` of the level-1 ``"root"`` topic in the stored JSON.

    With a single L1 node anchored at ``start_s=0.0``, its ``duration_s``
    equals ``transcript_end_s`` outright, so this is the cleanest knob
    to assert which branch of ``persist_topics`` set the end time.
    """
    assert clip_topics_json is not None
    topics = json.loads(clip_topics_json)
    for t in topics:
        if t["level"] == 1 and t["title"] == "root":
            return float(t["duration_s"])
    raise AssertionError(f"no L1 root in {topics!r}")


def test_persist_topics_happy_path(autorag_no_embed: AutoRAG, tmp_path: Path) -> None:
    audio = _make_audio_file(tmp_path)
    db_path = tmp_path / "autorag.sqlite"
    autorag_no_embed.persist_transcription(audio, _words(10.0), db_path=db_path)

    out = autorag_no_embed.persist_topics(
        audio, _topic_tree(), words=_words(10.0), db_path=db_path, title="My Clip"
    )

    clip = out["clip"]
    assert clip is not None
    assert clip["topics"]
    titles = [t["title"] for t in json.loads(clip["topics"])]
    assert set(titles) == {"root", "intro", "outro"}
    assert clip["provider"] == "ollama"
    assert "finalize" in out["timings"] and "embed" in out["timings"]

    chroma = ChromaStore(default_chroma_dir(db_path))
    assert chroma.count() >= 3


def test_persist_topics_end_s_explicit_overrides_words(
    autorag_no_embed: AutoRAG, tmp_path: Path
) -> None:
    audio = _make_audio_file(tmp_path)
    db_path = tmp_path / "autorag.sqlite"
    autorag_no_embed.persist_transcription(audio, _words(10.0), db_path=db_path)

    out = autorag_no_embed.persist_topics(
        audio,
        _topic_tree(),
        words=_words(10.0),
        transcript_end_s=42.0,
        db_path=db_path,
    )

    # Explicit transcript_end_s wins over the words[-1] hint (which would give 10.0).
    # Root (single L1, s=0.0) → duration = end_s - 0.0 = 42.0;
    # last L2 sibling ("outro", s=5.0) → duration = end_s - 5.0 = 37.0.
    assert _root_duration(out["clip"]["topics"]) == 42.0
    topics = json.loads(out["clip"]["topics"])
    outro = next(t for t in topics if t["title"] == "outro")
    assert outro["duration_s"] == 37.0


def test_persist_topics_end_s_derived_from_words(autorag_no_embed: AutoRAG, tmp_path: Path) -> None:
    audio = _make_audio_file(tmp_path)
    db_path = tmp_path / "autorag.sqlite"
    autorag_no_embed.persist_transcription(audio, _words(13.5), db_path=db_path)

    out = autorag_no_embed.persist_topics(
        audio,
        _topic_tree(),
        words=_words(13.5),
        db_path=db_path,
    )
    assert _root_duration(out["clip"]["topics"]) == 13.5


def test_persist_topics_end_s_zero_when_no_words_no_end_s(
    autorag_no_embed: AutoRAG, tmp_path: Path
) -> None:
    audio = _make_audio_file(tmp_path)
    db_path = tmp_path / "autorag.sqlite"
    autorag_no_embed.persist_transcription(audio, _words(10.0), db_path=db_path)

    out = autorag_no_embed.persist_topics(audio, _topic_tree(), db_path=db_path)
    # No transcript_end_s, no words → end_s = 0.0 → root's duration clamps to 0.0.
    assert _root_duration(out["clip"]["topics"]) == 0.0


def test_persist_topics_youtube_url_skips_file_check(
    autorag_no_embed: AutoRAG, tmp_path: Path
) -> None:
    """When ``file`` itself is a YouTube URL, the local-file check is bypassed.

    The canonical/upserts-overwrites behaviour for URLs is exercised in
    ``tests/test_persistence_session_id.py``; here we just verify the
    method runs without raising and writes a clip row.
    """
    db_path = tmp_path / "autorag.sqlite"
    out = autorag_no_embed.persist_topics(
        "https://youtu.be/dQw4w9WgXcQ",
        _topic_tree(),
        db_path=db_path,
        title="youtube-clip",
    )
    assert out["clip"] is not None
    assert out["clip"]["title"] == "youtube-clip"


def test_persist_topics_missing_file_raises(autorag_no_embed: AutoRAG, tmp_path: Path) -> None:
    db_path = tmp_path / "autorag.sqlite"
    missing = tmp_path / "does-not-exist.webm"
    with pytest.raises(FileNotFoundError):
        autorag_no_embed.persist_topics(missing, _topic_tree(), db_path=db_path)


def test_persist_topics_embedding_failure_is_swallowed(
    autorag_no_embed: AutoRAG,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    audio = _make_audio_file(tmp_path)
    db_path = tmp_path / "autorag.sqlite"
    autorag_no_embed.persist_transcription(audio, _words(10.0), db_path=db_path)

    def _boom(self: Embedder, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("ollama down")

    monkeypatch.setattr(Embedder, "embed_texts", _boom)

    with caplog.at_level(logging.WARNING, logger="autorag.core"):
        out = autorag_no_embed.persist_topics(
            audio, _topic_tree(), words=_words(10.0), db_path=db_path
        )

    # SQLite write must have succeeded even though embeddings failed.
    assert out["clip"] is not None
    assert out["clip"]["topics"]
    assert "embed" in out["timings"]
    assert any("embedding/index failed" in r.getMessage() for r in caplog.records)
