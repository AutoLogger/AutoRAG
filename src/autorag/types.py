"""Public typed-dict shapes for the audio→topics pipeline.

Kept dependency-free so SDK consumers can reference these types without
forcing the optional `[audio]` / `[diarize]` extras (langchain, whisper,
pyannote) to be importable.
"""

from __future__ import annotations

from typing import TypedDict


class WordSpan(TypedDict, total=False):
    """One word emitted by the transcription pipeline.

    Keys: ``w`` (word), ``s``/``e`` (start/end seconds), ``segment_id``
    (Whisper segment id), and ``speaker`` (string id assigned by
    diarization; ``"0"`` when diarization is disabled).
    """

    w: str
    s: float
    e: float
    segment_id: str
    speaker: str


class TopicDict(TypedDict, total=False):
    """One node in the L0/L1/L2 topic tree."""

    title: str
    summary: str
    s: float
    e: float
    children: list[TopicDict]


class TopicTree(TypedDict):
    """Container returned by :meth:`autorag.core.AutoRAG.generate_topics`."""

    topics: list[TopicDict]


class TranscriptionResult(TypedDict):
    """Combined transcript + topics, the output of ``build_agent``."""

    transcription: list[WordSpan]
    topics: TopicTree
