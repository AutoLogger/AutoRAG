from __future__ import annotations

from pathlib import Path

import pytest

from autorag.audio_source import (
    AudioSource,
    _canonical_youtube_url,
    is_youtube_url,
    resolve_audio_input,
)


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


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com/watch?v=dQw4w9WgXcQ",
        "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://music.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "http://www.youtube.com/watch?v=dQw4w9WgXcQ&feature=share",
    ],
)
def test_canonical_youtube_url_collapses_host_variants(url: str) -> None:
    assert _canonical_youtube_url(url) == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def test_canonical_youtube_url_raises_when_id_missing() -> None:
    with pytest.raises(ValueError):
        _canonical_youtube_url("https://www.youtube.com/feed/subscriptions")


def test_resolve_audio_input_local_file_passthrough(tmp_path: Path) -> None:
    audio = tmp_path / "clip.webm"
    audio.write_bytes(b"fake-audio")

    with resolve_audio_input(audio) as resolved:
        assert isinstance(resolved, AudioSource)
        assert resolved.path == audio
        assert resolved.path.is_file()
        assert resolved.source_url is None
        assert resolved.video_id is None
        assert resolved.title is None
        assert resolved.upload_date is None
        assert resolved.duration_s is None
        assert resolved.uploader is None


def test_resolve_audio_input_local_string_passthrough(tmp_path: Path) -> None:
    audio = tmp_path / "clip.webm"
    audio.write_bytes(b"fake-audio")

    with resolve_audio_input(str(audio)) as resolved:
        assert resolved.path == audio
        assert resolved.source_url is None
        assert resolved.video_id is None
        assert resolved.title is None
        assert resolved.upload_date is None
        assert resolved.duration_s is None
        assert resolved.uploader is None


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


def test_resolve_audio_input_youtube_populates_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """yt-dlp's info dict is mapped onto AudioSource fields end-to-end."""
    import sys
    from typing import Any

    fake_info: dict[str, Any] = {
        "id": "dQw4w9WgXcQ",
        "ext": "webm",
        "title": "Never Gonna Give You Up",
        "upload_date": "20091025",
        "duration": 213,
        "uploader": "Rick Astley",
    }

    class _FakeYDL:
        def __init__(self, opts: dict[str, Any]) -> None:
            self._opts = opts

        def __enter__(self) -> _FakeYDL:
            return self

        def __exit__(self, *a: object) -> None:
            return None

        def extract_info(self, url: str, download: bool = True) -> dict[str, Any]:
            assert url == "https://youtu.be/dQw4w9WgXcQ"
            assert download is True
            outtmpl = str(self._opts["outtmpl"])
            dest_dir = Path(outtmpl).parent
            (dest_dir / f"{fake_info['id']}.{fake_info['ext']}").write_bytes(b"fake-audio")
            return fake_info

        def prepare_filename(self, info: dict[str, Any]) -> str:
            outtmpl = str(self._opts["outtmpl"])
            dest_dir = Path(outtmpl).parent
            return str(dest_dir / f"{info['id']}.{info['ext']}")

    fake_module = type("_M", (), {"YoutubeDL": _FakeYDL})()
    monkeypatch.setitem(sys.modules, "yt_dlp", fake_module)

    with resolve_audio_input("https://youtu.be/dQw4w9WgXcQ") as resolved:
        assert resolved.path.is_file()
        assert resolved.path.name == "dQw4w9WgXcQ.webm"
        assert resolved.source_url == "https://youtu.be/dQw4w9WgXcQ"
        assert resolved.video_id == "dQw4w9WgXcQ"
        assert resolved.title == "Never Gonna Give You Up"
        assert resolved.upload_date == "20091025"
        assert resolved.duration_s == 213.0
        assert resolved.uploader == "Rick Astley"
