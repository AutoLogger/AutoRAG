from __future__ import annotations

import logging
import tempfile
import urllib.parse
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

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


@dataclass(frozen=True)
class AudioSource:
    """Resolved audio input plus its original-source identity and metadata.

    ``path`` is the local file the rest of the pipeline reads. ``source_url``
    and ``video_id`` are populated only when the input was a YouTube URL.
    The remaining fields surface yt-dlp's info dict (title, upload date,
    duration, uploader) so downstream persistence can record human-readable
    metadata instead of falling back to the temp filename / mtime.
    """

    path: Path
    source_url: str | None
    video_id: str | None
    title: str | None = None
    upload_date: str | None = None
    duration_s: float | None = None
    uploader: str | None = None


def is_youtube_url(value: str) -> bool:
    """Return True iff ``value`` parses as an http(s) URL on a YouTube host."""
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").lower()
    return host in _YOUTUBE_HOSTS


def default_title_from(source: str) -> str:
    """Derive a clip title from a local path or YouTube URL.

    YouTube URLs resolve to the video id; local paths resolve to the file
    stem. Used as a fallback when neither a caller-supplied title nor a
    yt-dlp-provided title is available.
    """
    if is_youtube_url(source):
        parsed = urllib.parse.urlparse(source)
        qs = urllib.parse.parse_qs(parsed.query)
        video_id = (qs.get("v", [""])[0] or parsed.path.lstrip("/")).strip("/")
        return video_id or "youtube-clip"
    return Path(source).stem


def _canonical_youtube_url(url: str) -> str:
    """Return a normalized ``https://www.youtube.com/watch?v=<id>`` URL.

    Collapses ``youtu.be/<id>``, ``m.youtube.com/watch?v=<id>``, etc. to a
    single canonical string so the same video hashes to the same id under
    :func:`uuid.uuid5`. Raises :class:`ValueError` if no video id can be
    extracted.
    """
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    video_id: str | None = None
    if host == "youtu.be":
        video_id = parsed.path.lstrip("/").split("/", 1)[0] or None
    elif host in _YOUTUBE_HOSTS:
        qs = urllib.parse.parse_qs(parsed.query)
        v = qs.get("v", [""])[0].strip()
        if v:
            video_id = v
    if not video_id:
        raise ValueError(f"could not extract YouTube video id from URL: {url}")
    return f"https://www.youtube.com/watch?v={video_id}"


@contextmanager
def resolve_audio_input(source: Path | str) -> Iterator[AudioSource]:
    """Yield an :class:`AudioSource` for ``source``.

    If ``source`` is a YouTube URL, download the best audio stream into a
    temporary directory and yield a populated ``AudioSource`` whose ``path``
    points into that tempdir. The tempdir is removed on exit. Otherwise treat
    ``source`` as a local path and yield a path-only ``AudioSource`` after
    verifying the file exists.
    """
    if isinstance(source, str) and is_youtube_url(source):
        with tempfile.TemporaryDirectory(prefix="autorag-yt-") as tmp:
            path, info = _download_youtube_audio(source, Path(tmp))
            duration = info.get("duration")
            yield AudioSource(
                path=path,
                source_url=source,
                video_id=str(info["id"]),
                title=(info.get("title") or None),
                upload_date=(info.get("upload_date") or None),
                duration_s=float(duration) if duration is not None else None,
                uploader=(info.get("uploader") or info.get("channel") or None),
            )
        return

    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"audio source not found: {path}")
    yield AudioSource(path=path, source_url=None, video_id=None)


def _download_youtube_audio(url: str, dest_dir: Path) -> tuple[Path, dict[str, Any]]:
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
    if not str(info.get("id") or "").strip():
        raise RuntimeError(f"yt-dlp did not return a video id for: {url}")
    return out, info
