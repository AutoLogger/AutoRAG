"""Lazy-loaded pyannote speaker-diarization pipeline with CUDA->CPU fallback.

Mirrors `whisper_runner.py` in structure: a module-level cache, a `threading.Lock`,
and a single CUDA failure flips the process to CPU for the rest of its life.

Public surface:

- `get_pipeline()` -> Pipeline | None  (None means "no token / load failed";
  callers should fall back to single-speaker behavior)
- `diarize_file(path)` -> list[(start_s, end_s, speaker_label)]
- `assign_speakers(words, turns)` -> list[str]  (parallel labels, '0' fallback)
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import threading
import warnings
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Module-level state -----------------------------------------------------------

_PIPELINE_LOCK = threading.Lock()
_PIPELINE: Any | None = None
_PIPELINE_LOAD_ATTEMPTED = False
_cpu_pinned = False  # set True after any CUDA failure for the process lifetime
_pipeline_device: str = "cpu"  # tracks current device of _PIPELINE after load

_MODEL_NAME = "pyannote/speaker-diarization-3.1"

# pyannote/torchaudio decodes these reliably; everything else gets transcoded.
_NATIVE_AUDIO_EXTS = frozenset({".wav", ".flac"})


def _torch_cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _device_preference() -> str:
    if _cpu_pinned:
        return "cpu"
    raw = os.environ.get("AUTORAG_WHISPER_DEVICE", "auto").strip().lower()
    if raw == "cpu":
        return "cpu"
    return "cuda" if _torch_cuda_available() else "cpu"


def _ffmpeg_exe() -> str | None:
    """Locate ffmpeg, preferring system PATH then imageio_ffmpeg's bundled binary."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return None


def _ensure_ffmpeg_on_path() -> None:
    """Same trick as whisper_runner: pyannote loads audio via torchaudio,
    which shells out to ffmpeg on some backends."""
    ff = _ffmpeg_exe()
    if not ff:
        return
    ff_dir = str(Path(ff).parent)
    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep) if current else []
    if ff_dir and ff_dir not in parts:
        os.environ["PATH"] = (ff_dir + os.pathsep + current) if current else ff_dir


def _transcode_to_wav(src: str, dst: str) -> bool:
    """Decode `src` → 16 kHz mono PCM wav at `dst`. Returns True on success."""
    ff = _ffmpeg_exe()
    if not ff:
        logger.warning("ffmpeg unavailable; cannot transcode %s for diarization.", src)
        return False
    try:
        subprocess.run(
            [ff, "-y", "-loglevel", "error", "-i", src, "-ac", "1", "-ar", "16000", "-vn", dst],
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning("ffmpeg transcode failed for %s (%s); diarization skipped.", src, exc)
        return False
    return True


def _hf_token() -> str | None:
    raw = os.environ.get("HF_TOKEN", "").strip()
    return raw or None


def _is_cuda_error(exc: BaseException) -> bool:
    cls = type(exc).__name__.lower()
    msg = (str(exc) or "").lower()
    if "cuda" in cls or "cuda" in msg:
        return True
    if "out of memory" in msg:
        return True
    return "nvml" in msg or ("driver" in msg and "cuda" in msg)


def _pin_cpu(reason: str) -> None:
    global _cpu_pinned
    if not _cpu_pinned:
        logger.warning("Pinning pyannote to CPU for remainder of process: %s", reason)
    _cpu_pinned = True


def _ensure_pipeline_on_cuda(pipeline: Any) -> None:
    """Re-move an offloaded pipeline back to CUDA before inference."""
    global _pipeline_device
    if _pipeline_device == "cuda" or _cpu_pinned or _device_preference() != "cuda":
        return
    try:
        import torch

        pipeline.to(torch.device("cuda"))
        _pipeline_device = "cuda"
        logger.debug("pyannote pipeline moved back to CUDA for inference.")
    except Exception as exc:
        _pin_cpu(str(exc))


def _offload_pipeline(pipeline: Any) -> None:
    """Move pipeline to CPU and free VRAM after inference completes."""
    global _pipeline_device
    if _pipeline_device != "cuda":
        return
    try:
        import torch

        pipeline.to(torch.device("cpu"))
        torch.cuda.empty_cache()
        _pipeline_device = "cpu"
        logger.debug("pyannote pipeline offloaded to CPU; VRAM freed.")
    except Exception as exc:
        logger.debug("pyannote VRAM offload failed (%s); continuing.", exc)


def _load_pipeline_on(device: str) -> Any | None:
    token = _hf_token()
    if not token:
        logger.warning(
            "HF_TOKEN not set; speaker diarization disabled. "
            "All words will be labeled as a single speaker."
        )
        return None

    try:
        from pyannote.audio import Pipeline
    except Exception as exc:
        logger.warning("pyannote.audio import failed (%s); diarization disabled.", exc)
        return None

    try:
        pipeline = Pipeline.from_pretrained(_MODEL_NAME, token=token)
    except Exception as exc:
        logger.warning("Failed to load %s (%s); diarization disabled.", _MODEL_NAME, exc)
        return None
    if pipeline is None:
        # pyannote returns None when the model cannot be loaded (e.g. accepted
        # license missing for the gated repo).
        logger.warning(
            "%s could not be loaded (token may lack access to the gated model); "
            "diarization disabled.",
            _MODEL_NAME,
        )
        return None

    if device == "cuda":
        try:
            import torch

            pipeline.to(torch.device("cuda"))
            global _pipeline_device
            _pipeline_device = "cuda"
        except Exception as exc:
            if _is_cuda_error(exc):
                _pin_cpu(str(exc))
                # Pipeline already loaded on CPU by from_pretrained default;
                # no reload needed.
                logger.warning("pyannote CUDA move failed (%s); using CPU.", exc)
            else:
                raise
    logger.info("Loaded pyannote pipeline %s on device=%s", _MODEL_NAME, device)
    return pipeline


def get_pipeline() -> Any | None:
    """Return the cached pyannote pipeline, loading on first call.

    Returns None if HF_TOKEN is missing or load failed; callers MUST handle
    None by skipping diarization.
    """
    global _PIPELINE, _PIPELINE_LOAD_ATTEMPTED
    with _PIPELINE_LOCK:
        if _PIPELINE_LOAD_ATTEMPTED:
            return _PIPELINE
        _PIPELINE_LOAD_ATTEMPTED = True
        _ensure_ffmpeg_on_path()
        _PIPELINE = _load_pipeline_on(_device_preference())
        return _PIPELINE


def diarize_file(file_path: str) -> list[tuple[float, float, str]]:
    """Run diarization. Returns sorted [(start, end, label), ...] or [] on failure.

    pyannote/torchaudio only decodes a small set of containers reliably (wav, flac);
    everything else (webm, mp3, m4a, ogg, ...) is transcoded to a temporary 16 kHz
    mono wav with ffmpeg first.
    """
    pipeline = get_pipeline()
    if pipeline is None:
        return []

    ext = Path(file_path).suffix.lower()
    if ext in _NATIVE_AUDIO_EXTS:
        return _run_diarization(pipeline, file_path)

    with tempfile.TemporaryDirectory(prefix="autorag-diarize-") as tmpdir:
        wav_path = str(Path(tmpdir) / "diarize_input.wav")
        if not _transcode_to_wav(file_path, wav_path):
            return []
        return _run_diarization(pipeline, wav_path)


def _run_diarization(pipeline: Any, audio_path: str) -> list[tuple[float, float, str]]:
    global _pipeline_device
    _ensure_pipeline_on_cuda(pipeline)
    try:
        with warnings.catch_warnings():
            # pyannote's StatsPool calls std(correction=1) on single-frame segments
            # (dof=0 → NaN), which it handles internally. Suppress the noise.
            warnings.filterwarnings(
                "ignore",
                message=r"std\(\).*degrees of freedom",
                category=UserWarning,
            )
            diarization = pipeline(audio_path)
    except Exception as exc:  # pragma: no cover - hardware-dependent
        if _is_cuda_error(exc) and not _cpu_pinned:
            logger.warning("pyannote CUDA failure on %s (%s); retrying on CPU.", audio_path, exc)
            _pin_cpu(str(exc))
            try:
                import torch

                pipeline.to(torch.device("cpu"))
                _pipeline_device = "cpu"
                diarization = pipeline(audio_path)
            except Exception as exc2:
                logger.warning("pyannote CPU retry failed (%s); skipping diarization.", exc2)
                return []
        else:
            logger.warning("pyannote diarization failed on %s (%s); skipping.", audio_path, exc)
            return []

    # pyannote 4.x wraps the Annotation in a DiarizeOutput; older versions
    # return the Annotation directly. Unwrap if needed.
    annotation = getattr(diarization, "speaker_diarization", diarization)

    turns: list[tuple[float, float, str]] = []
    try:
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            turns.append((float(turn.start), float(turn.end), str(speaker)))
    except Exception as exc:
        logger.warning("Failed to extract pyannote tracks (%s); skipping.", exc)
        return []

    turns.sort(key=lambda t: t[0])
    result = _normalize_labels(turns)
    _offload_pipeline(pipeline)
    return result


def _normalize_labels(
    turns: list[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    """Map pyannote labels (SPEAKER_00, SPEAKER_01, ...) to '0', '1', ... in
    first-appearance order. Keeps prompt output clean and stable."""
    mapping: dict[str, str] = {}
    out: list[tuple[float, float, str]] = []
    for s, e, raw in turns:
        if raw not in mapping:
            mapping[raw] = str(len(mapping))
        out.append((s, e, mapping[raw]))
    return out


def assign_speakers(
    words: list[dict[str, Any]],
    turns: list[tuple[float, float, str]],
) -> list[str]:
    """Assign a speaker label to each word.

    Strategy: pick the turn with maximum temporal overlap with the word's
    [s, e] interval. If no turn overlaps, fall back to the nearest turn
    (by midpoint distance). If `turns` is empty, every word becomes "0".
    """
    if not turns:
        return ["0"] * len(words)

    labels: list[str] = []
    for w in words:
        try:
            ws = float(w.get("s", 0.0))
            we = float(w.get("e", ws))
        except (TypeError, ValueError):
            ws = we = 0.0
        if we < ws:
            we = ws

        best_overlap = 0.0
        best_label: str | None = None
        for ts, te, label in turns:
            overlap = max(0.0, min(we, te) - max(ws, ts))
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = label

        if best_label is None:
            wmid = (ws + we) / 2.0
            best_dist = float("inf")
            for ts, te, label in turns:
                tmid = (ts + te) / 2.0
                dist = abs(wmid - tmid)
                if dist < best_dist:
                    best_dist = dist
                    best_label = label

        labels.append(best_label or "0")
    return labels


__all__ = [
    "assign_speakers",
    "diarize_file",
    "get_pipeline",
]
