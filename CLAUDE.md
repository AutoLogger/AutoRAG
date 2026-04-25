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

Four modules expose the same `transcribe(file)` / `build_*_agent()` surface,
returning `{transcription, topics}`. Pick by trade-off:

- `agent.py` — thin LCEL chain that reuses project helpers
  (`whisper_runner`, `OllamaProvider`). Single LLM call. Use for
  parity with the existing CLI pipeline.
- `reimagined_agent.py` — clean-room single-shot pipeline; defines its own
  Pydantic schema (3-level recursive type) and renamed dict keys (`s`/`e`
  instead of `start_s`/`end_s`). Single LLM call. Output dict shape is the
  reference contract for the other agents. Default model
  `qwen2.5:14b-instruct-q8_0`, `num_ctx=16384`.
- `tiered_agent.py` — multi-pass L0/L1/L2 with an explicit "decide
  subdivide" gate per L1 and an aggregate L0 root. ~N+M+3 LLM calls
  (~10 for a 7-min clip). Output is `{"topics": [L0]}` — a single root
  with `L0.children = [L1...]`. Best balance of quality and cost; the
  decide-gate prevents the over-eager nesting that produces zero-duration
  ghost L3s. Recommended starting point for new work. Default model
  `qwen2.5:14b-instruct-q8_0`.
- `hierarchical_agent.py` — multi-pass divide-and-conquer pipeline (5
  stages, ~50–80 LLM calls, parallel-batched). Each call sees only its
  parent's transcript slice, so containment is structural. Use only when
  you need the full L3 nesting and have the parallelism budget.

The CLI (`cli.py`) currently invokes `reimagined_agent`. Switching it to
`tiered_agent` requires no code changes other than the import + the call
site, because `cli.py`'s 3-level traversal maps L0 → category `l1`,
L1 → `l2`, L2 → `l3` (the L0 root replaces what reimagined called L1).

### Ollama tuning notes (server-side)

`OLLAMA_NUM_PARALLEL` is the per-agent split:

- **`>= 4`** for multi-pass agents that batch (`hierarchical_agent`,
  `tiered_agent` Stage 3a/3b). Required for `Runnable.batch` to actually
  parallelize.
- **`= 1`** for one-shot on a *bigger* model. Ollama pre-reserves all
  `NUM_PARALLEL` slots' KV cache at the configured `num_ctx`, so 4 idle
  slots steal VRAM that the bigger model needs. With `NUM_PARALLEL=1` on
  a 24 GB GPU you can run `qwen2.5:14b-q8_0` (~15 GB) + `num_ctx=16384`
  (~3 GB KV) with full GPU offload; 32K KV pushes some layers onto CPU.
  Verify with `ollama ps` after a load.

Other settings:

- **Do NOT** combine `OLLAMA_FLASH_ATTENTION=1` with
  `OLLAMA_MULTIUSER_CACHE=true` and concurrent slots — triggers
  `GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers")`.
  Drop `MULTIUSER_CACHE` (per-slot prefix cache still works).
- Per-slot KV-cache sizing (f16): the hierarchical agent caps `num_ctx`
  at 16K for the L1 call and 8K for fan-out / summary calls to fit
  4 slots × KV + ~9 GB model in a 24 GB budget. The tiered agent uses
  the same defaults.

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
