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

### LangChain agents (audio → transcript + topics)

Three modules expose the same `transcribe(file)` / `build_*_agent()` surface,
returning `{transcription, topics}`. Pick by trade-off:

- `agent.py` — thin LCEL chain that reuses project helpers
  (`whisper_runner`, `OllamaProvider`). Single LLM call. Use for
  parity with the existing CLI pipeline.
- `reimagined_agent.py` — clean-room single-shot pipeline; defines its own
  Pydantic schema (3-level recursive type) and renamed dict keys (`s`/`e`
  instead of `start_s`/`end_s`). Single LLM call. Output dict shape is the
  reference contract for the other agents.
- `hierarchical_agent.py` — multi-pass divide-and-conquer pipeline (5
  stages, ~50–80 LLM calls, parallel-batched). Each call sees only its
  parent's transcript slice, so containment is structural. Better topic
  spans on weaker models (e.g. `llama3.1:8b`) at the cost of latency.

### Ollama tuning notes (server-side)

Required for the hierarchical agent to work without crashes:

- `OLLAMA_NUM_PARALLEL` ≥ 4 — gives `Runnable.batch` real concurrency.
- **Do NOT** combine `OLLAMA_FLASH_ATTENTION=1` with
  `OLLAMA_MULTIUSER_CACHE=true` and concurrent slots — triggers
  `GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers")`.
  Drop `MULTIUSER_CACHE` (per-slot prefix cache still works).
- Per-slot KV-cache sizing: at f16, 4 slots × 32K context exceeds 24 GB
  VRAM. The hierarchical agent caps `num_ctx` at 16K for the L1 call and
  8K for fan-out / summary calls to fit a 24 GB budget.

## Existing Conventions (preserve these)

- Every module begins with `from __future__ import annotations`.
- Pydantic v2 `BaseModel` for API schemas; `SettingsConfigDict` for config.
- `TypedDict` in `providers.py` for `WordSpan`, `Topic`, `TopicTree` — extend this pattern for new typed dicts.
- `TypedDict` in `orchestrator.py` for `TranscriptSegment`, `TranscriptPayload`, `SessionTranscriptionResult`.
- `Protocol` used for `LLMProvider` — use Protocol for new abstract interfaces, not ABC.
- `numpy.typing.NDArray[np.float64]` for numpy array return types (see `viz.umap_3d`).

## Third-Party Stubs

These packages have no stubs — covered by mypy `ignore_missing_imports` overrides:
- `whisper`, `umap`, `pydantic_sqlite`, `imageio_ffmpeg`, `chromadb`

`langchain-ollama` and `langchain-core` ship inline types and need no mypy overrides.

These packages have no stubs — suppress with `# type: ignore[import-untyped]` at the import site:
- `sklearn` (used in `viz.py` and `topic_cluster.py`)

## Pylance / Pyright

`.vscode/settings.json` enables Pylance with `typeCheckingMode: "strict"`. Because Pylance does not read `[tool.mypy]` overrides, the `[tool.pyright]` block in `pyproject.toml` mirrors them: `reportMissingTypeStubs = "none"` (matches the mypy `ignore_missing_imports` set above) and `reportPrivateUsage = "none"` (for accessing `pydantic_sqlite.DataBase._db` directly, which has no public reader). It also disables `reportUnknownArgumentType`/`VariableType`/`MemberType`, since mypy strict already catches the cases we care about and Pylance's strict mode flags `Any` propagation more aggressively than the codebase wants.

If a new untyped third-party dep is added: add it to BOTH the mypy overrides and (implicitly) to the pyright config — `reportMissingTypeStubs = "none"` covers all unstubbed deps in one shot.

## Static Analysis Commands

```bash
uv run mypy src/autorag/
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run pytest
```

## CI Pipeline

`.github/workflows/ci.yml` runs on every push and PR to `main`. Two parallel jobs:

- **Lint & Type Check** — `ruff check`, `ruff format --check`, `mypy`
- **Tests** — `pytest -v`

The workflow uses `uv sync --frozen` (fails if `uv.lock` is out of sync with `pyproject.toml`). If you add or change dependencies, run `uv lock` locally before pushing to keep the lock file current.
