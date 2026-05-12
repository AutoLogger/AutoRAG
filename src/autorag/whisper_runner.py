"""Lazy-loaded whisperX model cache with CUDA-preferred / CPU-fallback device selection.

The main transcription model (CTranslate2 / faster-whisper backend) is removed
from the module cache after each run so Python GC can free VRAM; the smaller
wav2vec2 alignment model is offloaded to CPU after aligning and restored on the
next call (PyTorch .to() round-trip).  Both are re-created from local HF cache
on the next pipeline run, which is fast (<1 s for models already downloaded).
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Module-level state -----------------------------------------------------------

_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: dict[tuple[str, str], Any] = {}  # (size, device) → whisperx model
_ALIGN_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}  # (lang, device) → (align_model, meta)
_cpu_pinned = False
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
    """Resolve the preferred device honoring ``AUTORAG_WHISPER_DEVICE``."""
    if _cpu_pinned:
        return "cpu"
    raw = os.environ.get("AUTORAG_WHISPER_DEVICE", "auto").strip().lower()
    if raw == "cpu":
        return "cpu"
    if raw == "cuda":
        return "cuda" if _torch_cuda_available() else "cpu"
    return "cuda" if _torch_cuda_available() else "cpu"


def _ensure_ffmpeg_on_path() -> None:
    """Reuse imageio_ffmpeg's bundled binary so whisperX's subprocess finds it."""
    ff = shutil.which("ffmpeg")
    if not ff:
        try:
            import imageio_ffmpeg

            ff = str(imageio_ffmpeg.get_ffmpeg_exe())
        except Exception:
            raise RuntimeError("missing ffmpeg") from None
    ff_dir = str(Path(ff).parent)
    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep) if current else []
    if ff_dir and ff_dir not in parts:
        os.environ["PATH"] = (ff_dir + os.pathsep + current) if current else ff_dir


def resolved_device() -> str:
    """Return the device most recently used (or the preference if nothing loaded yet)."""
    if _resolved_device is not None:
        return _resolved_device
    return _device_preference()


def _compute_type(device: str) -> str:
    return "float16" if device == "cuda" else "int8"


def _load_model_on(size: str, device: str) -> Any:
    import whisperx

    compute = _compute_type(device)
    logger.info("Loading whisperX model size=%s device=%s compute_type=%s", size, device, compute)
    return whisperx.load_model(size, device, compute_type=compute)


def get_model(size: str, device_hint: str | None = None) -> Any:
    """Return a cached whisperX model for *size*.

    ``device_hint`` is advisory: ignored when the process is already CPU-pinned.
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
            logger.info("whisperX device preference resolved to: %s", device)
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
        logger.warning("Pinning whisperX to CPU for remainder of process: %s", reason)
    _cpu_pinned = True


def _is_cuda_error(exc: BaseException) -> bool:
    cls = type(exc).__name__.lower()
    msg = (str(exc) or "").lower()
    if "cuda" in cls or "cuda" in msg:
        return True
    if "out of memory" in msg:
        return True
    return "nvml" in msg or ("driver" in msg and "cuda" in msg)


def _get_align_model(language: str, device: str) -> tuple[Any, Any]:
    """Return (align_model, metadata) for *language*, restoring from CPU cache when possible."""
    key = (language, device)
    with _MODEL_LOCK:
        cached = _ALIGN_CACHE.get(key)
        if cached is not None:
            return cached
        cpu_cached = _ALIGN_CACHE.get((language, "cpu"))

    if device == "cuda" and cpu_cached is not None:
        model_a, metadata = cpu_cached
        try:
            import torch

            model_a.to(torch.device("cuda"))
            with _MODEL_LOCK:
                _ALIGN_CACHE[(language, "cuda")] = (model_a, metadata)
                _ALIGN_CACHE.pop((language, "cpu"), None)
            logger.debug("whisperX align model restored to CUDA.")
            return model_a, metadata
        except Exception as exc:
            logger.warning("whisperX align model CUDA restore failed (%s); reloading.", exc)

    import whisperx

    logger.info("Loading whisperX align model language=%s device=%s", language, device)
    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
    with _MODEL_LOCK:
        _ALIGN_CACHE[(language, device)] = (model_a, metadata)
    return model_a, metadata


def _offload_align_model(language: str) -> None:
    """Move the wav2vec2 alignment model to CPU and free VRAM."""
    try:
        import torch

        with _MODEL_LOCK:
            cuda_key = (language, "cuda")
            cached = _ALIGN_CACHE.get(cuda_key)
            if cached is None:
                return
            model_a, metadata = cached
            model_a.to(torch.device("cpu"))
            _ALIGN_CACHE[(language, "cpu")] = (model_a, metadata)
            del _ALIGN_CACHE[cuda_key]
        torch.cuda.empty_cache()
        logger.debug("whisperX align model offloaded to CPU; VRAM freed.")
    except Exception as exc:
        logger.debug("whisperX align model offload failed (%s); continuing.", exc)


def transcribe_segment(
    model: Any,
    file_path: str,
    language: str | None,
) -> list[dict[str, Any]]:
    """Transcribe *file_path* and return frame-aligned word dicts.

    Each dict: ``{"w": str, "s": float, "e": float, "p": float}``.
    The alignment pass uses wav2vec2 for frame-accurate word timestamps; if it
    fails the unaligned faster-whisper timestamps are used as a fallback.
    """
    import whisperx

    _ensure_ffmpeg_on_path()
    audio = whisperx.load_audio(file_path)
    device = resolved_device() or "cpu"
    batch_size = 16 if device == "cuda" else 4

    transcribe_kwargs: dict[str, Any] = {"batch_size": batch_size}
    if language:
        transcribe_kwargs["language"] = language

    try:
        result: dict[str, Any] = model.transcribe(audio, **transcribe_kwargs)
    except Exception as exc:  # pragma: no cover - hardware-dependent
        if _is_cuda_error(exc) and not _cpu_pinned:
            logger.warning(
                "whisperX CUDA failure on %s (%s); reloading on CPU and retrying once.",
                file_path,
                exc,
            )
            _pin_cpu(str(exc))
            size_guess = _current_model_size(model) or "base"
            cpu_model = get_model(size_guess, device_hint="cpu")
            cpu_kwargs: dict[str, Any] = {"batch_size": 4}
            if language:
                cpu_kwargs["language"] = language
            result = cpu_model.transcribe(audio, **cpu_kwargs)
            model = cpu_model
            device = "cpu"
        else:
            raise

    detected_language: str = result.get("language") or language or "en"

    try:
        model_a, metadata = _get_align_model(detected_language, device)
        aligned: dict[str, Any] = whisperx.align(
            result["segments"], model_a, metadata, audio, device, return_char_alignments=False
        )
        segments: list[Any] = aligned.get("segments", result.get("segments", []))
    except Exception as exc:
        logger.warning("whisperX alignment failed (%s); using unaligned timestamps.", exc)
        segments = result.get("segments", [])

    words_out: list[dict[str, Any]] = []
    for seg in segments:
        raw_words = seg.get("words") if isinstance(seg, dict) else None
        if not raw_words:
            continue
        for w in raw_words:
            try:
                token = str(w.get("word", "") or "")
                start_raw = w.get("start")
                end_raw = w.get("end")
                # whisperX omits start/end for words it could not align — skip them.
                if start_raw is None and end_raw is None:
                    continue
                start = float(start_raw or 0.0)
                end = float(end_raw or start)
                prob = float(w.get("score", w.get("probability", 0.0)) or 0.0)
            except (TypeError, ValueError):
                continue
            if not token.strip():
                continue
            words_out.append({"w": token, "s": start, "e": end, "p": prob})

    _offload_main_model(model)
    _offload_align_model(detected_language)
    return words_out


def _offload_main_model(model: Any) -> None:
    """Remove the whisperX model from cache; CTranslate2 frees VRAM when the object is GC'd."""
    with _MODEL_LOCK:
        keys_to_del = [k for k, v in _MODEL_CACHE.items() if v is model]
        for k in keys_to_del:
            del _MODEL_CACHE[k]
    if keys_to_del:
        logger.debug("whisperX model removed from cache; VRAM freed on GC.")


def _current_model_size(model: Any) -> str | None:
    """Best-effort reverse lookup of a model's cache key for its size."""
    with _MODEL_LOCK:
        for (size, _), m in _MODEL_CACHE.items():
            if m is model:
                return size
    return None


__all__ = [
    "get_model",
    "resolved_device",
    "transcribe_segment",
]
