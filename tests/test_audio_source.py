from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from autorag.audio_source import is_youtube_url, resolve_audio_input

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com/watch?v=dQw4w9WgXcQ",
        "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://music.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "http://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://WWW.YOUTUBE.COM/watch?v=dQw4w9WgXcQ",
    ],
)
def test_is_youtube_url_accepts(url: str) -> None:
    assert is_youtube_url(url)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "youtube.com/watch?v=dQw4w9WgXcQ",
        "ftp://youtube.com/watch?v=dQw4w9WgXcQ",
        "http://evil.com/youtube.com/watch?v=dQw4w9WgXcQ",
        "https://vimeo.com/123",
        "/local/path/clip.webm",
        "./relative/path.webm",
        "dQw4w9WgXcQ",
    ],
)
def test_is_youtube_url_rejects(value: str) -> None:
    assert not is_youtube_url(value)


def test_resolve_audio_input_local_file_passthrough(tmp_path: Path) -> None:
    audio = tmp_path / "clip.webm"
    audio.write_bytes(b"fake-audio")

    with resolve_audio_input(audio) as resolved:
        assert resolved == audio
        assert resolved.is_file()


def test_resolve_audio_input_local_string_passthrough(tmp_path: Path) -> None:
    audio = tmp_path / "clip.webm"
    audio.write_bytes(b"fake-audio")

    with resolve_audio_input(str(audio)) as resolved:
        assert resolved == audio


def test_resolve_audio_input_missing_local_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.webm"
    with pytest.raises(FileNotFoundError):  # noqa: SIM117
        with resolve_audio_input(missing):
            pass


def test_resolve_audio_input_non_youtube_url_treated_as_path() -> None:
    """Strict allowlist: a non-YouTube URL is treated as a local path and fails the exists check."""
    with pytest.raises(FileNotFoundError):  # noqa: SIM117
        with resolve_audio_input("https://vimeo.com/12345"):
            pass
