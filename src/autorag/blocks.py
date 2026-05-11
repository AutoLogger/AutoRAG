"""Pure-stdlib transcript-formatting helpers.

Kept dependency-free so a base install (no ``[audio]`` / ``[rag]``) can call
:func:`format_blocks` on any :class:`autorag.types.WordSpan` list it already
has — e.g. one loaded straight from the SQLite cache or built externally.
"""

from __future__ import annotations

from math import floor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autorag.types import WordSpan

__all__ = ["format_blocks", "group_by_speaker"]


def group_by_speaker(spans: list[WordSpan]) -> list[tuple[str, list[WordSpan]]]:
    """Walk spans in order; coalesce consecutive same-speaker runs.

    Words missing a `speaker` key are treated as speaker "0", which keeps
    single-speaker behavior identical to pre-diarization output.
    """
    groups: list[tuple[str, list[WordSpan]]] = []
    for ws in spans:
        speaker = str(ws.get("speaker", "0") or "0")
        if groups and groups[-1][0] == speaker:
            groups[-1][1].append(ws)
        else:
            groups.append((speaker, [ws]))
    return groups


def _mmss(t: float) -> str:
    total = max(0, floor(t))
    return f"{total // 60:02d}:{total % 60:02d}"


def _speaker_label(raw: str) -> str:
    try:
        return f"Speaker {int(raw) + 1}"
    except (TypeError, ValueError):
        return f"Speaker {raw}"


def format_blocks(transcription: list[WordSpan], seconds: int) -> str:
    """Render `transcription` as N-second time blocks with per-turn speaker lines.

    Buckets each :class:`WordSpan` into ``[floor(s/N)*N, floor(s/N)*N + N)``.
    Within each non-empty bucket, groups consecutive same-speaker spans into
    turns via :func:`group_by_speaker` and emits one line per turn:
    ``MM:SS-MM:SS Speaker K: <words>`` where K is ``int(speaker) + 1``
    (1-indexed display; non-numeric labels pass through verbatim).
    Skips empty buckets; separates non-empty buckets by one blank line.
    No trailing newline.

    A turn whose words span multiple buckets produces one line per bucket —
    each line covers only that bucket's portion of the turn.

    Raises:
        ValueError: if ``seconds <= 0``.
    """
    if seconds <= 0:
        raise ValueError("seconds must be a positive integer")

    buckets: dict[int, list[WordSpan]] = {}
    for ws in transcription:
        token = str(ws.get("w", "") or "").strip()
        if not token:
            continue
        start = float(ws.get("s", 0.0) or 0.0)
        key = floor(start / seconds) * seconds
        buckets.setdefault(key, []).append(ws)

    if not buckets:
        return ""

    block_texts: list[str] = []
    for key in sorted(buckets):
        lines: list[str] = []
        for speaker, group in group_by_speaker(buckets[key]):
            tokens: list[str] = []
            first_s: float | None = None
            last_e: float | None = None
            for ws in group:
                token = str(ws.get("w", "") or "").strip()
                if not token:
                    continue
                s = float(ws.get("s", 0.0) or 0.0)
                e_raw = ws.get("e")
                e = float(e_raw) if e_raw is not None else s
                if first_s is None:
                    first_s = s
                last_e = e
                tokens.append(token)
            if not tokens or first_s is None or last_e is None:
                continue
            lines.append(
                f"{_mmss(first_s)}-{_mmss(last_e)} {_speaker_label(speaker)}: {' '.join(tokens)}"
            )
        if lines:
            block_texts.append("\n".join(lines))

    return "\n\n".join(block_texts)
