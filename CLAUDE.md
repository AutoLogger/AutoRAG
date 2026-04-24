# AutoRAG Project

Python 3.12 project managed with `uv`.

## Package Manager

Always use `uv`, never `pip` directly:
- `uv add <package>` / `uv add --dev <package>`
- `uv run <command>` to execute in the project venv
- `uv sync` to synchronize the environment

## Project Layout

- `src/autorag/` — main package (src layout)
- `tests/` — pytest tests
- Entry point: `autorag` CLI (`src/autorag/cli.py`)
- API server: `src/autorag/api.py`

## Existing Conventions (preserve these)

- Every module begins with `from __future__ import annotations`.
- Pydantic v2 `BaseModel` for API schemas; `SettingsConfigDict` for config.
- `TypedDict` in `providers.py` for `WordSpan`, `Topic`, `TopicTree` — extend this pattern for new typed dicts.
- `TypedDict` in `orchestrator.py` for `TranscriptSegment`, `TranscriptPayload`, `SessionTranscriptionResult`.
- `Protocol` used for `LLMProvider` — use Protocol for new abstract interfaces, not ABC.
- `numpy.typing.NDArray[np.float64]` for numpy array return types (see `viz.umap_3d`).

## Third-Party Stubs

These packages have no stubs — covered by mypy `ignore_missing_imports` overrides:
- `whisper`, `umap`, `pydantic_sqlite`, `imageio_ffmpeg`

These packages have no stubs — suppress with `# type: ignore[import-untyped]` at the import site:
- `sklearn` (used in `viz.py` and `topic_cluster.py`)

## Static Analysis Commands

```bash
uv run mypy src/autorag/
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run pytest
```
