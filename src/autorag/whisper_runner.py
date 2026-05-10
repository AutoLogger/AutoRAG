"""Lazy-loaded Whisper model cache with CUDA-preferred / CPU-fallback device selection.

The model cache is keyed by `(size, device)` and guarded by a module-level
`threading.Lock`. We choose `cuda` when available (and not explicitly forced
off by `AUTORAG_WHISPER_DEVICE=cpu`); on the first CUDA failure (OOM,
driver mismatch, etc.) we catch it once, log a warning, reload on CPU, and
pin the whole process to CPU for the rest of its lifetime.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Any

import whisper

logger = logging.getLogger(__name__)

# Module-level state -----------------------------------------------------------

_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_cpu_pinned = False  # set True after any CUDA failure for the process lifetime
_device_log_emitted = False
_resolved_device: str | None = None


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
    """Resolve the preferred device honoring `AUTORAG_WHISPER_DEVICE`."""
    if _cpu_pinned:
        return "cpu"
    raw = os.environ.get("AUTORAG_WHISPER_DEVICE", "auto").strip().lower()
    if raw == "cpu":
        return "cpu"
    if raw == "cuda":
        return "cuda" if _torch_cuda_available() else "cpu"
    # "auto" or anything else
    return "cuda" if _torch_cuda_available() else "cpu"


def _ensure_ffmpeg_on_path() -> None:
    """Reuse audio_chunk_lab's ffmpeg resolution so Whisper's subprocess finds it."""
    ff = shutil.which("ffmpeg")
    if not ff:
        try:
            import imageio_ffmpeg

            exe = imageio_ffmpeg.get_ffmpeg_exe()
            ff = str(exe)
        except Exception:
            raise RuntimeError("missing ffmpeg") from None

    ff_path = Path(ff)
    ff_dir = str(ff_path.parent)
    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep) if current else []
    if ff_dir and ff_dir not in parts:
        os.environ["PATH"] = (ff_dir + os.pathsep + current) if current else ff_dir


def resolved_device() -> str:
    """Return the device most recently used (or preferred if nothing loaded yet)."""
    if _resolved_device is not None:
        return _resolved_device
    return _device_preference()


def _load_model_on(size: str, device: str) -> Any:

    logger.info("Loading Whisper model size=%s device=%s", size, device)
    return whisper.load_model(size, device=device)


def get_model(size: str, device_hint: str | None = None) -> Any:
    """Return a cached Whisper model for the given size.

    `device_hint` is advisory: if the process is already CPU-pinned because of
    an earlier GPU failure, the hint is ignored.
    """
    global _device_log_emitted, _resolved_device

    _ensure_ffmpeg_on_path()

    device = (device_hint or "").strip().lower() or _device_preference()
    if _cpu_pinned:
        device = "cpu"
    if device not in ("cuda", "cpu"):
        device = "cpu"

    key = (size, device)
    with _MODEL_LOCK:
        if not _device_log_emitted:
            logger.info("Whisper device preference resolved to: %s", device)
            _device_log_emitted = True
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            _resolved_device = device
            return cached
        model = _load_model_on(size, device)
        _MODEL_CACHE[key] = model
        _resolved_device = device
        return model


def _pin_cpu(reason: str) -> None:
    global _cpu_pinned
    if not _cpu_pinned:
        logger.warning("Pinning Whisper to CPU for remainder of process: %s", reason)
    _cpu_pinned = True


def _is_cuda_error(exc: BaseException) -> bool:
    cls = type(exc).__name__.lower()
    msg = (str(exc) or "").lower()
    if "cuda" in cls or "cuda" in msg:
        return True
    if "out of memory" in msg:
        return True
    return "nvml" in msg or ("driver" in msg and "cuda" in msg)


def transcribe_segment(
    model: Any,
    file_path: str,
    language: str | None,
) -> list[dict[str, Any]]:
    """Transcribe a single audio file and return a flat list of word dicts.

    Each word dict is `{"w": str, "s": float, "e": float, "p": float}`.
    `language=None` means auto-detect.
    """
    kwargs: dict[str, Any] = {"word_timestamps": True}
    if language:
        kwargs["language"] = language

    try:
        result = model.transcribe(str(file_path), **kwargs)
    except Exception as exc:  # pragma: no cover - hardware-dependent
        if _is_cuda_error(exc) and not _cpu_pinned:
            logger.warning(
                "Whisper CUDA failure on %s (%s); reloading on CPU and retrying once.",
                file_path,
                exc,
            )
            _pin_cpu(str(exc))
            size_guess = _current_model_size(model) or "base"
            cpu_model = get_model(size_guess, device_hint="cpu")
            result = cpu_model.transcribe(str(file_path), **kwargs)
        else:
            raise

    words_out: list[dict[str, Any]] = []
    segments = result.get("segments") if isinstance(result, dict) else None
    if not segments:
        return words_out
    for seg in segments:
        raw_words = seg.get("words") if isinstance(seg, dict) else None
        if not raw_words:
            continue
        for w in raw_words:
            try:
                token = str(w.get("word", "") or "")
                start = float(w.get("start", 0.0) or 0.0)
                end = float(w.get("end", start) or start)
                prob = float(w.get("probability", w.get("prob", 0.0)) or 0.0)
            except (TypeError, ValueError):
                continue
            if not token.strip():
                continue
            words_out.append({"w": token, "s": start, "e": end, "p": prob})
    return words_out


def _current_model_size(model: Any) -> str | None:
    """Best-effort reverse lookup of a model's cache key for size."""
    with _MODEL_LOCK:
        for (size, _device), m in _MODEL_CACHE.items():
            if m is model:
                return size
    return None


__all__ = [
    "get_model",
    "resolved_device",
    "transcribe_segment",
]
