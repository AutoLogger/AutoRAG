from __future__ import annotations

import logging
import tempfile
import urllib.parse
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from autorag.errors import _missing_extra

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

_YOUTUBE_HOSTS: frozenset[str] = frozenset(
    {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtu.be",
    }
)


def is_youtube_url(value: str) -> bool:
    """Return True iff ``value`` parses as an http(s) URL on a YouTube host."""
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").lower()
    return host in _YOUTUBE_HOSTS


@contextmanager
def resolve_audio_input(source: Path | str) -> Iterator[Path]:
    """Yield a local ``Path`` for ``source``.

    If ``source`` is a YouTube URL, download the best audio stream into a
    temporary directory and yield that file's path. The tempdir is removed
    on exit. Otherwise treat ``source`` as a local path and yield it after
    verifying it exists.
    """
    if isinstance(source, str) and is_youtube_url(source):
        with tempfile.TemporaryDirectory(prefix="autorag-yt-") as tmp:
            yield _download_youtube_audio(source, Path(tmp))
        return

    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"audio source not found: {path}")
    yield path


def _download_youtube_audio(url: str, dest_dir: Path) -> Path:
    try:
        import yt_dlp
    except ModuleNotFoundError as exc:
        raise _missing_extra("youtube", exc) from exc

    opts: dict[str, object] = {
        "format": "bestaudio[ext=webm]/bestaudio",
        "outtmpl": str(dest_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    logger.info("Downloading YouTube audio: %s", url)
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        out = Path(ydl.prepare_filename(info))
    if not out.is_file():
        raise RuntimeError(f"yt-dlp did not produce expected file: {out}")
    return out
