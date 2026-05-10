"""Public typed-dict shapes for the audio→topics pipeline.

Kept dependency-free so SDK consumers can reference these types without
forcing the optional `[audio]` / `[diarize]` extras (langchain, whisper,
pyannote) to be importable.
"""

from __future__ import annotations

from typing import TypedDict


class WordSpan(TypedDict, total=False):
    w: str
    s: float
    e: float
    abs_s: float
    segment_id: str
    speaker: str


class TopicDict(TypedDict, total=False):
    title: str
    summary: str
    s: float
    e: float
    children: list[TopicDict]


class TopicTree(TypedDict):
    topics: list[TopicDict]


class TranscriptionResult(TypedDict):
    transcription: list[WordSpan]
    topics: TopicTree
