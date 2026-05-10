from __future__ import annotations

from autorag.agent import (
    TopicDict,
    WordSpan,
    _drop_zero,
    _format_transcript,
    _format_words_only,
    _group_by_speaker,
    _slice_spans,
    _snap_tile,
    _target_count,
)


def test_target_count_short_audio_floors_to_two() -> None:
    assert _target_count(0.0, 0.0) == 2
    assert _target_count(0.0, 60.0) == 2


def test_target_count_scales_with_minutes() -> None:
    assert _target_count(0.0, 180.0) == 3
    assert _target_count(0.0, 360.0) == 6


def test_target_count_clamps_at_seven() -> None:
    assert _target_count(0.0, 1200.0) == 7
    assert _target_count(0.0, 6000.0) == 7


def test_target_count_uses_relative_slice() -> None:
    # 10 minutes inside a slice that starts late.
    assert _target_count(300.0, 900.0) == 7


def test_slice_spans_inclusive_bounds() -> None:
    spans: list[WordSpan] = [
        {"w": "a", "s": 0.0, "e": 0.5},
        {"w": "b", "s": 1.0, "e": 1.5},
        {"w": "c", "s": 2.0, "e": 2.5},
        {"w": "d", "s": 3.0, "e": 3.5},
    ]
    out = _slice_spans(spans, 1.0, 2.0)
    assert [w["w"] for w in out] == ["b", "c"]


def test_slice_spans_empty_when_out_of_range() -> None:
    spans: list[WordSpan] = [{"w": "a", "s": 5.0, "e": 5.5}]
    assert _slice_spans(spans, 0.0, 1.0) == []


def test_slice_spans_handles_missing_s_as_zero() -> None:
    spans: list[WordSpan] = [{"w": "a"}, {"w": "b", "s": 1.0, "e": 1.5}]
    out = _slice_spans(spans, 0.0, 0.5)
    assert [w["w"] for w in out] == ["a"]


def _node(s: float, e: float, *, children: list[TopicDict] | None = None) -> TopicDict:
    return {"title": "", "summary": "", "s": s, "e": e, "children": children or []}


def test_snap_tile_anchors_endpoints_and_closes_gaps() -> None:
    siblings: list[TopicDict] = [_node(5.0, 20.0), _node(25.0, 40.0)]
    _snap_tile(siblings, 0.0, 50.0)
    # First anchored to slice_s, last anchored to slice_e, no gap between them.
    assert siblings[0]["s"] == 0.0
    assert siblings[-1]["e"] == 50.0
    assert siblings[1]["s"] == siblings[0]["e"]


def test_snap_tile_sorts_and_collapses_overlaps() -> None:
    siblings: list[TopicDict] = [_node(20.0, 30.0), _node(0.0, 25.0)]
    _snap_tile(siblings, 0.0, 30.0)
    # After sorting by s: [0..25], [20..30] — overlap collapsed via prev.e -> cur.s.
    assert siblings[0]["s"] == 0.0
    assert siblings[1]["s"] == siblings[0]["e"]
    assert siblings[1]["e"] == 30.0


def test_snap_tile_clamps_out_of_range_values() -> None:
    siblings: list[TopicDict] = [_node(-5.0, 100.0)]
    _snap_tile(siblings, 0.0, 10.0)
    assert siblings[0]["s"] == 0.0
    assert siblings[0]["e"] == 10.0


def test_snap_tile_noop_on_empty() -> None:
    siblings: list[TopicDict] = []
    _snap_tile(siblings, 0.0, 10.0)
    assert siblings == []


def test_drop_zero_removes_zero_duration_leaves() -> None:
    siblings: list[TopicDict] = [
        _node(0.0, 5.0),
        _node(5.0, 5.0),  # zero duration, dropped
        _node(5.0, 10.0),
    ]
    out = _drop_zero(siblings)
    assert [(n["s"], n["e"]) for n in out] == [(0.0, 5.0), (5.0, 10.0)]


def test_drop_zero_recurses_into_children() -> None:
    siblings: list[TopicDict] = [
        _node(
            0.0,
            10.0,
            children=[_node(0.0, 5.0), _node(5.0, 5.0), _node(5.0, 10.0)],
        ),
    ]
    out = _drop_zero(siblings)
    assert len(out) == 1
    assert [(c["s"], c["e"]) for c in out[0]["children"]] == [(0.0, 5.0), (5.0, 10.0)]


def test_group_by_speaker_coalesces_runs() -> None:
    spans: list[WordSpan] = [
        {"w": "a", "s": 0.0, "e": 0.5, "speaker": "0"},
        {"w": "b", "s": 0.5, "e": 1.0, "speaker": "0"},
        {"w": "c", "s": 1.0, "e": 1.5, "speaker": "1"},
        {"w": "d", "s": 1.5, "e": 2.0, "speaker": "0"},
    ]
    groups = _group_by_speaker(spans)
    assert [(s, [w["w"] for w in g]) for s, g in groups] == [
        ("0", ["a", "b"]),
        ("1", ["c"]),
        ("0", ["d"]),
    ]


def test_group_by_speaker_single_speaker_one_group() -> None:
    spans: list[WordSpan] = [
        {"w": "a", "speaker": "0"},
        {"w": "b", "speaker": "0"},
        {"w": "c", "speaker": "0"},
    ]
    groups = _group_by_speaker(spans)
    assert len(groups) == 1
    assert groups[0][0] == "0"
    assert [w["w"] for w in groups[0][1]] == ["a", "b", "c"]


def test_group_by_speaker_empty() -> None:
    assert _group_by_speaker([]) == []


def test_group_by_speaker_missing_key_defaults_to_zero() -> None:
    # Spans without a `speaker` field (legacy fixtures, fallback path) coalesce
    # into a single "0" group, matching pre-diarization behavior.
    spans: list[WordSpan] = [{"w": "a"}, {"w": "b"}]
    groups = _group_by_speaker(spans)
    assert len(groups) == 1
    assert groups[0][0] == "0"


def test_format_transcript_multi_speaker_emits_speaker_headers() -> None:
    spans: list[WordSpan] = [
        {"w": "hello", "s": 0.00, "e": 0.40, "speaker": "0"},
        {"w": "world", "s": 0.40, "e": 0.80, "speaker": "0"},
        {"w": "hi", "s": 1.20, "e": 1.40, "speaker": "1"},
    ]
    out = _format_transcript(spans)
    assert out == ("[Speaker 0]\ns=0.00 hello\ns=0.40 world\n[Speaker 1]\ns=1.20 hi")


def test_format_transcript_single_speaker_one_header() -> None:
    spans: list[WordSpan] = [
        {"w": "a", "s": 0.0, "e": 0.5, "speaker": "0"},
        {"w": "b", "s": 0.5, "e": 1.0, "speaker": "0"},
    ]
    out = _format_transcript(spans)
    assert out == "[Speaker 0]\ns=0.00 a\ns=0.50 b"


def test_format_words_only_multi_speaker_per_line() -> None:
    spans: list[WordSpan] = [
        {"w": "hello", "speaker": "0"},
        {"w": "and", "speaker": "0"},
        {"w": "welcome", "speaker": "0"},
        {"w": "thanks", "speaker": "1"},
        {"w": "for", "speaker": "1"},
        {"w": "having", "speaker": "1"},
        {"w": "me", "speaker": "1"},
        {"w": "lets", "speaker": "0"},
        {"w": "go", "speaker": "0"},
    ]
    out = _format_words_only(spans)
    expected = "Speaker 0: hello and welcome\nSpeaker 1: thanks for having me\nSpeaker 0: lets go"
    assert out == expected


def test_format_words_only_falls_back_when_no_speaker_key() -> None:
    spans: list[WordSpan] = [{"w": "alpha"}, {"w": "beta"}]
    out = _format_words_only(spans)
    assert out == "Speaker 0: alpha beta"
