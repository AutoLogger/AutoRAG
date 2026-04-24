"""Audio signature hashing for transcript cache invalidation.

The signature is a deterministic hash over (segment_id, file_size, ended_at_utc)
for every audio segment in a session, sorted by segment id. It changes whenever
a segment is added, replaced, or re-encoded — which is exactly when the cached
Whisper transcript should be considered stale.
"""

from __future__ import annotations

import hashlib
from typing import Any


def compute_audio_signature(segments: list[dict[str, Any]]) -> str:
    """SHA-256 over sorted (id, file_size, ended_at_utc) tuples.

    Returns a hex digest string. Callers enrich segment dicts with a
    `file_size` field prior to calling this helper (since
    `AutoLoggerDB.list_audio_segments` does not expose file size or
    file_path directly).
    """
    normalized: list[tuple[str, str, str]] = []
    for seg in segments:
        seg_id = str(seg.get("id") or "")
        size_val = seg.get("file_size")
        try:
            size_str = str(int(size_val)) if size_val is not None else ""
        except (TypeError, ValueError):
            size_str = str(size_val) if size_val is not None else ""
        ended = seg.get("ended_at_utc")
        ended_str = "" if ended is None else str(ended)
        normalized.append((seg_id, size_str, ended_str))
    normalized.sort(key=lambda t: t[0])

    hasher = hashlib.sha256()
    for seg_id, size_str, ended_str in normalized:
        hasher.update(seg_id.encode("utf-8"))
        hasher.update(b"\x1f")
        hasher.update(size_str.encode("utf-8"))
        hasher.update(b"\x1f")
        hasher.update(ended_str.encode("utf-8"))
        hasher.update(b"\x1e")
    return hasher.hexdigest()
