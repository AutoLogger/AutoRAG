"""CLI option coverage for the language default and LLM tuning knobs.

These tests drive the Typer app with a mocked :class:`AutoRAG` so no
audio / Whisper / Ollama runs. They pin the two user-facing contracts
added alongside the LLM-knob exposure:

* ``--language`` defaults to English (``en``); ``--language ""`` is the
  auto-detect escape hatch (forwarded to the SDK as ``language=None``).
* ``generate-topics`` forwards the full facade tuning set
  (``num_ctx_l1`` / ``num_ctx_fanout`` / ``max_concurrency`` /
  ``min_subdivide_duration_s`` / ``reasoning``) with the facade defaults,
  and respects overrides.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

import autorag.audio_source as audio_source
import autorag.cli as cli
import autorag.whisper_runner as whisper_runner

if TYPE_CHECKING:
    from collections.abc import Iterator

runner = CliRunner()


@pytest.fixture
def mock_rag(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch the ``AutoRAG`` the CLI instantiates; return the instance mock."""
    instance = MagicMock()
    instance.transcribe.return_value = []
    instance.generate_topics.return_value = {"topics": []}
    monkeypatch.setattr(cli, "AutoRAG", MagicMock(return_value=instance))
    return instance


@pytest.fixture
def stub_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the audio-source resolver + device probe used by ``transcribe``."""

    @contextlib.contextmanager
    def _resolve(source: str) -> Iterator[SimpleNamespace]:
        yield SimpleNamespace(
            path=Path("clip.wav"),
            title=None,
            source_url=None,
            upload_date=None,
            duration_s=None,
        )

    monkeypatch.setattr(audio_source, "resolve_audio_input", _resolve)
    monkeypatch.setattr(whisper_runner, "resolved_device", lambda: "cpu")


def test_transcribe_defaults_to_english(mock_rag: MagicMock, stub_audio: None) -> None:
    result = runner.invoke(cli.app, ["transcribe", "clip.wav", "--no-persist"])

    assert result.exit_code == 0, result.output
    assert mock_rag.transcribe.call_args.kwargs["language"] == "en"


def test_transcribe_empty_language_auto_detects(mock_rag: MagicMock, stub_audio: None) -> None:
    result = runner.invoke(cli.app, ["transcribe", "clip.wav", "--no-persist", "--language", ""])

    assert result.exit_code == 0, result.output
    assert mock_rag.transcribe.call_args.kwargs["language"] is None


def test_generate_topics_forwards_tuning_defaults(mock_rag: MagicMock) -> None:
    result = runner.invoke(
        cli.app,
        ["generate-topics", "clip.wav", "--transcription", "[]", "--no-persist"],
    )

    assert result.exit_code == 0, result.output
    kwargs = mock_rag.generate_topics.call_args.kwargs
    assert kwargs == {
        "llm_model": "gemma4:latest",
        "num_ctx_l1": 8192,
        "num_ctx_fanout": 8192,
        "max_concurrency": 4,
        "min_subdivide_duration_s": 120.0,
        "reasoning": False,
        "boundary_block_seconds": 30,
    }


def test_generate_topics_forwards_tuning_overrides(mock_rag: MagicMock) -> None:
    result = runner.invoke(
        cli.app,
        [
            "generate-topics",
            "clip.wav",
            "--transcription",
            "[]",
            "--no-persist",
            "--num-ctx-l1",
            "16384",
            "--num-ctx-fanout",
            "4096",
            "--max-concurrency",
            "1",
            "--min-subdivide-duration-s",
            "60",
            "--reasoning",
            "--boundary-block-seconds",
            "15",
        ],
    )

    assert result.exit_code == 0, result.output
    kwargs = mock_rag.generate_topics.call_args.kwargs
    assert kwargs["num_ctx_l1"] == 16384
    assert kwargs["num_ctx_fanout"] == 4096
    assert kwargs["max_concurrency"] == 1
    assert kwargs["min_subdivide_duration_s"] == 60.0
    assert kwargs["reasoning"] is True
    assert kwargs["boundary_block_seconds"] == 15
