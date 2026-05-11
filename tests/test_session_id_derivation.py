"""Unit tests for :func:`autorag.persistence.derive_session_id`.

Mirror the persistence-layer logic in :meth:`AutoRAG.persist_transcription`
so re-querying the same source resolves to the same SQLite row.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from autorag.audio_source import _canonical_youtube_url
from autorag.persistence import derive_session_id

if TYPE_CHECKING:
    from pathlib import Path


def test_youtube_url_matches_persist_logic() -> None:
    url = "https://youtu.be/dQw4w9WgXcQ"
    expected = str(uuid.uuid5(uuid.NAMESPACE_URL, _canonical_youtube_url(url)))
    assert derive_session_id(url) == expected


def test_local_path_matches_persist_logic(tmp_path: Path) -> None:
    audio = tmp_path / "clip.webm"
    audio.write_bytes(b"fake")
    expected = str(uuid.uuid5(uuid.NAMESPACE_URL, str(audio.resolve())))
    assert derive_session_id(audio) == expected
    # Same for str input.
    assert derive_session_id(str(audio)) == expected


def test_url_variants_collapse() -> None:
    a = derive_session_id("https://youtu.be/dQw4w9WgXcQ")
    b = derive_session_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    c = derive_session_id("https://m.youtube.com/watch?v=dQw4w9WgXcQ")
    assert a == b == c
