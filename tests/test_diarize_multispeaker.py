"""End-to-end diarization test on the multi-speaker fox-new.webm sample.

Skipped when HF_TOKEN is not available (e.g., CI). Runs pyannote only — no
Whisper transcription, no Ollama LLM calls.

pyannote's audio loader can't decode .webm directly (it raises a torchaudio
internal error); the test transcodes to a 16 kHz mono wav with ffmpeg first.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIO = REPO_ROOT / "tests" / "fox-new.webm"
ENV_FILE = REPO_ROOT / ".env"


def _ffmpeg_exe() -> str:
    import shutil

    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg

    return str(imageio_ffmpeg.get_ffmpeg_exe())


def _transcode_to_wav(src: Path, dst: Path) -> None:
    """Decode webm/whatever → 16 kHz mono PCM wav so pyannote/torchaudio can read it."""
    subprocess.run(
        [
            _ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-vn",
            str(dst),
        ],
        check=True,
    )


def _resolve_hf_token() -> str | None:
    """Return HF_TOKEN from env; fall back to project .env. Exports to os.environ."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    if not ENV_FILE.exists():
        return None
    for raw_line in ENV_FILE.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        if key.strip() != "HF_TOKEN":
            continue
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ["HF_TOKEN"] = value
            return value
    return None


_HF_TOKEN = _resolve_hf_token()


@pytest.mark.skipif(
    _HF_TOKEN is None,
    reason="HF_TOKEN unavailable (env or .env); pyannote diarization disabled",
)
@pytest.mark.skipif(
    not AUDIO.exists(),
    reason=f"sample audio missing at {AUDIO}",
)
def test_fox_new_has_multiple_speakers(tmp_path: Path) -> None:
    from autorag import diarize

    wav = tmp_path / "fox-new.wav"
    _transcode_to_wav(AUDIO, wav)

    turns = diarize.diarize_file(str(wav))

    assert turns, "diarize_file returned no turns; pipeline likely failed to load"

    speakers = {label for _, _, label in turns}
    assert len(speakers) >= 2, (
        f"expected >= 2 distinct speakers in fox-new.webm, got {len(speakers)}: {speakers}"
    )

    for start, end, _ in turns:
        assert start < end, f"turn has non-positive duration: ({start}, {end})"

    starts = [t[0] for t in turns]
    assert starts == sorted(starts), "turns should be sorted by start time"
