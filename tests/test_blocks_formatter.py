"""Unit tests for :func:`autorag.blocks.format_blocks` (pure stdlib)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from autorag.blocks import format_blocks

if TYPE_CHECKING:
    from autorag.types import WordSpan


def test_basic_two_buckets_two_speakers() -> None:
    spans: list[WordSpan] = [
        {"w": "hi", "s": 1.20, "e": 1.50, "speaker": "0"},
        {"w": "there", "s": 1.50, "e": 1.90, "speaker": "0"},
        {"w": "welcome", "s": 4.00, "e": 4.80, "speaker": "1"},
        {"w": "thanks", "s": 12.30, "e": 12.90, "speaker": "0"},
    ]
    expected = (
        "00:01-00:01 Speaker 1: hi there\n"
        "00:04-00:04 Speaker 2: welcome\n"
        "\n"
        "00:12-00:12 Speaker 1: thanks"
    )
    assert format_blocks(spans, 10) == expected


def test_empty_transcription_returns_empty_string() -> None:
    assert format_blocks([], 10) == ""


def test_skips_empty_windows() -> None:
    spans: list[WordSpan] = [
        {"w": "alpha", "s": 0.5, "e": 0.8, "speaker": "0"},
        {"w": "omega", "s": 20.5, "e": 21.0, "speaker": "0"},
    ]
    expected = "00:00-00:00 Speaker 1: alpha\n\n00:20-00:21 Speaker 1: omega"
    out = format_blocks(spans, 10)
    assert out == expected
    # No block emitted for the empty [10, 20) bucket: exactly one blank line.
    assert out.count("\n\n") == 1


def test_groups_consecutive_same_speaker_within_bucket() -> None:
    spans: list[WordSpan] = [
        {"w": "one", "s": 0.0, "e": 0.5, "speaker": "0"},
        {"w": "two", "s": 0.5, "e": 1.0, "speaker": "0"},
        {"w": "three", "s": 1.0, "e": 1.5, "speaker": "0"},
    ]
    assert format_blocks(spans, 10) == "00:00-00:01 Speaker 1: one two three"


def test_long_turn_split_across_buckets() -> None:
    spans: list[WordSpan] = [
        {"w": "before", "s": 5.0, "e": 5.5, "speaker": "0"},
        {"w": "after", "s": 12.0, "e": 12.5, "speaker": "0"},
    ]
    expected = "00:05-00:05 Speaker 1: before\n\n00:12-00:12 Speaker 1: after"
    assert format_blocks(spans, 10) == expected


def test_default_speaker_when_missing_field() -> None:
    spans: list[WordSpan] = [{"w": "hello", "s": 0.0, "e": 0.5}]
    assert format_blocks(spans, 10) == "00:00-00:00 Speaker 1: hello"


def test_skips_empty_token_words() -> None:
    spans: list[WordSpan] = [
        {"w": "  ", "s": 0.5, "e": 0.6, "speaker": "0"},
        {"w": "", "s": 1.0, "e": 1.1, "speaker": "0"},
        {"w": "real", "s": 20.0, "e": 20.3, "speaker": "0"},
    ]
    # Bucket [0, 10) has only empty-token words → skipped entirely.
    assert format_blocks(spans, 10) == "00:20-00:20 Speaker 1: real"


def test_raises_on_nonpositive_seconds() -> None:
    spans: list[WordSpan] = [{"w": "x", "s": 0.0, "e": 0.5}]
    with pytest.raises(ValueError):
        format_blocks(spans, 0)
    with pytest.raises(ValueError):
        format_blocks(spans, -5)


def test_minutes_over_99() -> None:
    spans: list[WordSpan] = [{"w": "late", "s": 6000.0, "e": 6000.5, "speaker": "0"}]
    assert format_blocks(spans, 10) == "100:00-100:00 Speaker 1: late"


def test_range_end_uses_word_e() -> None:
    # Last word ends at 7.9 — range-end MM:SS should reflect e (07), not s (06)
    # and not the bucket boundary (10).
    spans: list[WordSpan] = [
        {"w": "first", "s": 1.0, "e": 1.4, "speaker": "0"},
        {"w": "last", "s": 6.0, "e": 7.9, "speaker": "0"},
    ]
    assert format_blocks(spans, 10) == "00:01-00:07 Speaker 1: first last"
