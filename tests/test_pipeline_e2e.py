"""End-to-end test for autorag.agent.transcribe() against fox-new.webm.

Exercises the full pipeline: Whisper transcription + pyannote diarization +
Ollama topic extraction. Auto-skips when any of the three external dependencies
(HF_TOKEN, Ollama with the chosen model, ffmpeg) is missing — so CI passes
cleanly while local dev still gets the integration coverage.

Runtime when not skipped: ~1-3 min depending on hardware.
"""

from __future__ import annotations

import json
import os
import shutil
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIO = REPO_ROOT / "tests" / "fox-new.webm"
ENV_FILE = REPO_ROOT / ".env"
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_URL = os.environ.get("AUTORAG_OLLAMA_BASE_URL", "").strip() or "http://localhost:11434"


# TODO: extract _resolve_hf_token() to tests/conftest.py if a third caller appears.
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


def _ollama_reachable(base_url: str, model: str) -> bool:
    """True iff Ollama responds at /api/tags AND advertises `model`."""
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=2.0) as resp:
            data = json.loads(resp.read())
    except Exception:
        return False
    names = {m.get("name", "") for m in data.get("models", []) if isinstance(m, dict)}
    return model in names


def _ffmpeg_present() -> bool:
    if shutil.which("ffmpeg"):
        return True
    try:
        import imageio_ffmpeg

        imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return False
    return True


def _yt_dlp_available() -> bool:
    try:
        import yt_dlp  # type: ignore[import-untyped]  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


_HF_TOKEN = _resolve_hf_token()
_OLLAMA_OK = _ollama_reachable(OLLAMA_URL, OLLAMA_MODEL)
_FFMPEG_OK = _ffmpeg_present()
_YT_URL = os.environ.get("AUTORAG_TEST_YT_URL", "").strip()
_YT_DLP_OK = _yt_dlp_available()


@pytest.mark.skipif(not AUDIO.exists(), reason=f"sample audio missing at {AUDIO}")
@pytest.mark.skipif(
    _HF_TOKEN is None,
    reason="HF_TOKEN unavailable; pyannote diarization disabled",
)
@pytest.mark.skipif(
    not _OLLAMA_OK,
    reason=f"Ollama at {OLLAMA_URL} missing model {OLLAMA_MODEL}",
)
@pytest.mark.skipif(
    not _FFMPEG_OK,
    reason="ffmpeg not available for diarization transcode",
)
def test_fox_new_full_pipeline() -> None:
    from autorag.agent import transcribe

    result = transcribe(AUDIO, llm_model=OLLAMA_MODEL)

    spans = result["transcription"]
    assert spans, "transcription is empty"

    topics_root = result["topics"]["topics"]
    assert len(topics_root) == 1, f"expected 1 L0 root, got {len(topics_root)}"

    for w in spans:
        assert {"w", "s", "e", "speaker"}.issubset(w.keys()), f"span missing keys: {w}"
        assert float(w["s"]) <= float(w["e"]), f"span has s>e: {w}"

    starts = [float(w["s"]) for w in spans]
    assert starts == sorted(starts), "spans should be non-decreasing by start time"

    speakers = {w["speaker"] for w in spans}
    assert len(speakers) >= 2, (
        f"expected >= 2 distinct speakers in fox-new.webm, got {len(speakers)}: {speakers}"
        " — Part A transcode fix may not be wired into the agent path"
    )

    l0 = topics_root[0]
    l1_children = l0.get("children", [])
    # Note: L2 presence is intentionally not asserted — with the default
    # min_subdivide_duration_s=120.0 the subdivide branch may or may not
    # fire depending on the clip's L1 boundary lengths.
    assert l1_children, "L0 root has no L1 children"

    for l1 in l1_children:
        assert l1.get("title", ""), f"L1 missing title: {l1}"
        assert l1.get("summary", ""), f"L1 missing summary: {l1}"
        for l2 in l1.get("children", []):
            assert l2.get("title", ""), f"L2 missing title: {l2}"
            assert l2.get("summary", ""), f"L2 missing summary: {l2}"


@pytest.mark.skipif(not _YT_URL, reason="AUTORAG_TEST_YT_URL unset; URL e2e test opt-in")
@pytest.mark.skipif(not _YT_DLP_OK, reason="yt-dlp not installed (autorag[youtube] extra)")
@pytest.mark.skipif(
    _HF_TOKEN is None,
    reason="HF_TOKEN unavailable; pyannote diarization disabled",
)
@pytest.mark.skipif(
    not _OLLAMA_OK,
    reason=f"Ollama at {OLLAMA_URL} missing model {OLLAMA_MODEL}",
)
@pytest.mark.skipif(
    not _FFMPEG_OK,
    reason="ffmpeg not available for diarization transcode",
)
def test_youtube_url_full_pipeline() -> None:
    from autorag import AutoRAG

    rag = AutoRAG()
    spans = rag.transcribe(_YT_URL)
    assert spans, "transcription is empty"

    topics_result = rag.generate_topics(spans, llm_model=OLLAMA_MODEL)
    topics_root = topics_result["topics"]
    assert len(topics_root) == 1, f"expected 1 L0 root, got {len(topics_root)}"

    for w in spans:
        assert {"w", "s", "e", "speaker"}.issubset(w.keys()), f"span missing keys: {w}"
