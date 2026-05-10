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
- **SDK entry point**: `from autorag import AutoRAG` (`src/autorag/core.py`)
- CLI: `autorag` (`src/autorag/cli.py`) — thin wrapper over `AutoRAG`
- API server: `src/autorag/api.py` — also wraps `AutoRAG`

### SDK facade (`core.py`)

`AutoRAG` is the single public class. Flat methods, lazy imports for heavy deps:

| Method                   | Extras needed     | Purpose                                       |
|--------------------------|-------------------|-----------------------------------------------|
| `transcribe(file, ...)`  | `[audio,diarize]` (+ `[youtube]` for URLs) | Whisper + LLM topic tree → `TranscriptionResult`. `file` is a local path **or** a YouTube URL (downloaded to a temp `.webm` via `autorag.audio_source.resolve_audio_input`) |
| `build_agent(**kwargs)`  | `[audio,diarize]` | Returns the LangChain `Runnable` directly     |
| `persist_transcription(file, result, ...)` | `[rag]` | Writes clip + words + events to SQLite, indexes topic embeddings in Chroma |
| `ingest(paths)`          | base              | Document RAG: load → chunk → embed → store    |
| `query(question, ...)`   | base              | Retrieve + generate over ingested corpus      |

`MissingExtraError` and `_missing_extra` live in `src/autorag/errors.py` (`core.py` re-exports `MissingExtraError` for backwards compat). Each audio/RAG method does `from autorag.X import ...` *inside the method body* and re-raises `ModuleNotFoundError` as `MissingExtraError` with a friendly extras hint. **Do not move these imports to module-top** — base install (`pip install autorag`) must boot without `chromadb`/`torch`/`whisper`/`pyannote`/`yt_dlp` installed. The CI `test-base` job enforces this.

### Audio → transcript + topics agent

`src/autorag/agent.py` is the single audio→topics pipeline. Public surface:
`transcribe(file, **kwargs) -> TranscriptionResult` and `build_agent(**kwargs)`,
returning `{transcription, topics}` where `topics = {"topics": [L0]}` and `L0`
is a root node whose `children` are the L1 topics (each with optional `L2`
`children`). Most callers should use `AutoRAG.transcribe()` instead of importing
this module directly — the facade handles the lazy-import / extras-error story.

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

The CLI (`cli.py`) calls `AutoRAG.transcribe()` then `AutoRAG.persist_transcription()`.
Persistence helpers (`collapse_lone_children`, `iter_topics_flat`, `topics_to_events`)
live in `src/autorag/persistence.py`. The 3-level traversal maps the agent's L0
children → category `l1`, L1 children → `l2`, L2 children → `l3`.

### YouTube URL input (`audio_source.py`)

`src/autorag/audio_source.py` provides URL detection (`is_youtube_url`,
host-allowlisted via `urllib.parse`) and a `resolve_audio_input(source)`
context manager that yields an `AudioSource` dataclass with `path`,
`source_url`, and `video_id`. For YouTube URLs it downloads the best
audio stream into a `tempfile.TemporaryDirectory(prefix="autorag-yt-")`
via `yt_dlp` (lazy-imported inside the helper, gated by the `[youtube]`
extra; raises `MissingExtraError("youtube", ...)` when missing) and
populates `source_url` / `video_id` from yt-dlp's info dict. For non-URL
inputs it verifies the path exists and yields a path-only `AudioSource`
(`source_url=None`, `video_id=None`).

Both `core.AutoRAG.transcribe()` and the CLI wrap their work in
`resolve_audio_input`. The CLI must own the temp lifetime itself because
it calls both `transcribe` and `persist_transcription` on the same path —
inner `core.transcribe`'s wrapper is a no-op pass-through for an
already-local Path, so the double-wrap is safe and idempotent. The CLI
also forwards `src.source_url` to `persist_transcription` so the clip's
`session_id` is seeded from the canonical YouTube URL
(`_canonical_youtube_url` collapses `youtu.be` / `m.youtube.com` /
`www.youtube.com` variants to one form), making re-runs overwrite the
same SQLite row instead of producing duplicates.

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
- `TypedDict` lives in `src/autorag/types.py` (dep-free) so SDK consumers can reference `WordSpan`, `TopicDict`, `TopicTree`, `TranscriptionResult` without importing langchain/whisper. New public typed-dicts go here, not in `agent.py`.
- `numpy.typing.NDArray[np.float64]` for numpy array return types (see `viz.umap_3d`).
- **Heavy deps stay lazy.** Base install (`pip install autorag`) only has typer + pydantic + langchain-{core,ollama}. Anything that imports `chromadb` / `torch` / `whisper` / `pyannote` / `umap` / `sklearn` / `pydantic_sqlite` belongs behind a method-body `import` in `core.py` (or the appropriate extras-gated module). When adding a new public method, decide which extra it needs and follow the existing `MissingExtraError` pattern.

### Packaging (`pyproject.toml`)

| Extra      | Modules that import it                                  | Adds                                  |
|------------|---------------------------------------------------------|---------------------------------------|
| `audio`    | `whisper_runner.py`, `agent.py` (whisper)               | openai-whisper, torch, imageio-ffmpeg |
| `diarize`  | `diarize.py`                                            | pyannote.audio, huggingface-hub       |
| `youtube`  | `audio_source.py` (lazy in `_download_youtube_audio`)   | yt-dlp                                |
| `rag`      | `chroma_store.py`, `db.py`, `viz.py`, `topic_cluster.py`| chromadb, umap-learn, scikit-learn, numpy, pydantic_sqlite |
| `server`   | `api.py` (FastAPI app)                                  | fastapi, uvicorn[standard]            |
| `all`      | —                                                       | union of the above                     |

Distribution is **GitHub-hosted, not PyPI**. Consumers install with
`pip install "autorag[...] @ git+https://github.com/AutoLogger/AutoRAG@v0.x.0"`.
Releases are made by bumping `__version__` in `src/autorag/__init__.py` and
`version` in `pyproject.toml`, running `uv lock`, committing, then
`git tag v0.x.0 && git push --tags`.

## Third-Party Stubs

These packages have no stubs — covered by mypy `ignore_missing_imports` overrides:
- `whisper`, `umap`, `pydantic_sqlite`, `imageio_ffmpeg`, `chromadb`, `pyannote`, `yt_dlp`

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

`.github/workflows/ci.yml` runs on every push and PR to `main`. Three parallel jobs:

- **Lint & Type Check** — `ruff check`, `ruff format --check`, `mypy` (installs `--all-extras` so mypy can see torch/chromadb/etc.)
- **Tests (all extras)** — `pytest -v` against the full dependency stack
- **SDK base install (no extras)** — `uv sync --frozen --no-dev` then asserts `from autorag import AutoRAG` boots and the SDK methods are callable. **This is the regression guard for the lazy-import contract** — if anyone re-introduces a `chromadb`/`torch`/`whisper`/`pyannote`/`yt_dlp` import at module top in `core.py` / `embed.py` / `__init__.py` / `store.py` / `audio_source.py`, this job fails.

The workflow uses `uv sync --frozen` (fails if `uv.lock` is out of sync with `pyproject.toml`). If you add or change dependencies, run `uv lock` locally before pushing to keep the lock file current.
