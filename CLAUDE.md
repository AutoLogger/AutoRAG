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

### Audio → transcript + topics agent

`src/autorag/agent.py` is the single audio→topics pipeline. Public surface:
`transcribe(file, **kwargs) -> TranscriptionResult` and `build_agent(**kwargs)`,
returning `{transcription, topics}` where `topics = {"topics": [L0]}` and `L0`
is a root node whose `children` are the L1 topics (each with optional `L2`
`children`).

Multi-pass L0/L1/L2 with boundary detection separated from summarization.
Stages: L1 boundaries (1 call) → decide subdivide on plain text (per long L1) →
L2 boundaries (per yes-L1) → per-node summarize for every L1 + L2 leaf
(batched, plain text in / `{title, summary}` out) → L0 aggregate. Total
~`2 + N1_long + N1_yes + N1 + N2_total` LLM calls (~20 for a 7-min clip).
The boundaries-vs-summaries split lets each call have one focused job and
gives the K summary calls an identical prompt prefix for cache reuse.
Default model `qwen2.5:14b-instruct-q8_0`.

Whisper is invoked through `autorag.whisper_runner` (cached per (size, device),
CUDA→CPU fallback on first failure). The Ollama base URL is resolved via
`AUTORAG_OLLAMA_BASE_URL` (falls back to `http://localhost:11434`).

Speaker diarization runs via `autorag.diarize` using
`pyannote/speaker-diarization-3.1`, which is HuggingFace-gated and requires
`HF_TOKEN`. Each `WordSpan` carries a `speaker` field (`"0"`, `"1"`, …
normalized in first-appearance order). Without a token (or on pyannote
load/runtime failure) the agent logs a warning and labels every word
`"0"` — output then matches pre-diarization behavior. `_format_transcript`
emits `[Speaker N]` headers above per-word timestamp lines; the per-node
summary input emits `Speaker N: <words>` per turn so the LLM sees explicit
turn-taking.

The CLI (`cli.py`) calls `agent.transcribe()`. Its 3-level traversal maps
the agent's L0 children → category `l1`, L1 children → `l2`, L2 children → `l3`.

### Ollama tuning notes (server-side)

`OLLAMA_NUM_PARALLEL` is the per-agent split:

- **`>= 4`** for the agent's batched stages (Stage 3a decide, Stage 3b L2
  boundaries, Stage 4 per-node summaries). Required for `Runnable.batch`
  to actually parallelize.
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
- Per-slot KV-cache sizing (f16): the agent caps `num_ctx` at 16K for the L1
  call and 8K for fan-out / summary calls to fit 4 slots × KV + ~9 GB model
  in a 24 GB budget.

## Existing Conventions (preserve these)

- Every module begins with `from __future__ import annotations`.
- Pydantic v2 `BaseModel` for API schemas; `SettingsConfigDict` for config.
- `TypedDict` in `agent.py` for `WordSpan`, `TopicDict`, `TopicTree`, `TranscriptionResult` — extend this pattern for new typed dicts.
- `numpy.typing.NDArray[np.float64]` for numpy array return types (see `viz.umap_3d`).

## Third-Party Stubs

These packages have no stubs — covered by mypy `ignore_missing_imports` overrides:
- `whisper`, `umap`, `pydantic_sqlite`, `imageio_ffmpeg`, `chromadb`, `pyannote`

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
