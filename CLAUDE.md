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
| `transcribe_blocks(file, seconds=10, ...)` | `[rag]` cache hit; `[audio,diarize]` (+ `[youtube]`) on miss | Returns the transcript as N-second time blocks (one `MM:SS-MM:SS Speaker K: ...` line per speaker turn). Reads from the SQLite cache when present (via `persistence.derive_session_id` + `load_transcription`), else runs `transcribe` + `persist_transcription` first |
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
CUDA→CPU fallback on first failure). After each `transcribe_segment` call the
model is offloaded to CPU and VRAM freed via `torch.cuda.empty_cache()`; the
next `get_model` call for the same size/device restores it to CUDA (cheap
memory copy, no disk reload). The Ollama base URL is resolved via
`AUTORAG_OLLAMA_BASE_URL` (falls back to `http://localhost:11434`).

Speaker diarization runs via `autorag.diarize` using
`pyannote/speaker-diarization-3.1`, which is HuggingFace-gated and requires
`HF_TOKEN`. Each `WordSpan` carries a `speaker` field (`"0"`, `"1"`, …
normalized in first-appearance order). Without a token (or on pyannote
load/runtime failure) the agent logs a warning and labels every word
`"0"` — output then matches pre-diarization behavior. After each
`_run_diarization` call the pipeline is offloaded to CPU and VRAM freed;
`_ensure_pipeline_on_cuda` restores it to CUDA on the next call. `_format_transcript`
emits `[Speaker N]` headers above per-word timestamp lines; the per-node
summary input emits `Speaker N: <words>` per turn so the LLM sees explicit
turn-taking.

The CLI (`cli.py`) calls `AutoRAG.transcribe()` then `AutoRAG.persist_transcription()`.
Persistence helpers (`collapse_lone_children`, `iter_topics_flat`, `topics_to_events`,
`derive_session_id`, `load_transcription`) live in `src/autorag/persistence.py`.
The first three are write-side; the latter two are base-safe readers used by
`AutoRAG.transcribe_blocks` to short-circuit on a cache hit. The 3-level
traversal maps the agent's L0 children → category `l1`, L1 children → `l2`,
L2 children → `l3`.

### Transcript block formatting (`blocks.py`)

`src/autorag/blocks.py` is dependency-free (stdlib only) and re-exported as
`from autorag import format_blocks`. `format_blocks(transcription, seconds)`
buckets `WordSpan`s by `floor(s/seconds) * seconds`, then within each
non-empty bucket coalesces consecutive same-speaker spans (via
`group_by_speaker`) into one `MM:SS-MM:SS Speaker K: <words>` line per turn
(K = `int(speaker) + 1`, 1-indexed display). Buckets are separated by a blank
line; empty buckets are skipped. A turn that crosses a bucket boundary
produces one line per bucket. `agent._format_transcript` re-uses
`group_by_speaker` from this module so there is no duplicate definition.

### `default_title_from` lives in `audio_source.py`

`autorag.audio_source.default_title_from(source)` resolves a YouTube URL to
its video id (or a local path to its file stem) and is used by both the CLI
`transcribe`/`blocks` commands and `AutoRAG.transcribe_blocks` as the last
fallback when neither a caller-supplied `--title` nor a yt-dlp-provided title
is available. Was `cli._default_title_from` in 0.2.0; promoted to a public
helper in 0.3.0 so the SDK doesn't have to import from `cli`.

### YouTube URL input (`audio_source.py`)

`src/autorag/audio_source.py` provides URL detection (`is_youtube_url`,
host-allowlisted via `urllib.parse`) and a `resolve_audio_input(source)`
context manager that yields an `AudioSource` dataclass with `path`,
`source_url`, `video_id`, plus four optional metadata fields surfaced
from yt-dlp's info dict: `title`, `upload_date` (`"YYYYMMDD"`),
`duration_s`, and `uploader` (falls back to `info["channel"]`). For
YouTube URLs it downloads the best audio stream into a
`tempfile.TemporaryDirectory(prefix="autorag-yt-")` via `yt_dlp`
(lazy-imported inside the helper, gated by the `[youtube]` extra;
raises `MissingExtraError("youtube", ...)` when missing).
`_download_youtube_audio` returns `(Path, dict[str, Any])` — the full
info dict — so `resolve_audio_input` can map fields onto `AudioSource`
without a parallel argument explosion. For non-URL inputs it verifies
the path exists and yields a path-only `AudioSource` (all optional
fields `None`).

Both `core.AutoRAG.transcribe()` and the CLI wrap their work in
`resolve_audio_input`. The CLI must own the temp lifetime itself because
it calls both `transcribe` and `persist_transcription` on the same path —
inner `core.transcribe`'s wrapper is a no-op pass-through for an
already-local Path, so the double-wrap is safe and idempotent. The CLI
forwards `src.source_url`, `src.upload_date`, `src.duration_s`, and
`src.title` to `persist_transcription`. `source_url` seeds the clip's
`session_id` from the canonical YouTube URL (`_canonical_youtube_url`
collapses `youtu.be` / `m.youtube.com` / `www.youtube.com` variants to
one form) so re-runs overwrite the same SQLite row, and is also stored
as the row's `file_path` so the entry remains valid after the temp
download is gone. `upload_date` (when present) anchors `created_at` and
the absolute event timestamps to midnight-UTC of the publish date
instead of the temp-file mtime. `duration_s` is currently informational
(no schema column). The CLI's `--title` still wins; `src.title` is the
fallback, and `_default_title_from(source)` is the last resort when
yt-dlp returned no title.

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

## Frontend (`/viz` page)

`/viz` is the project's only browser surface. As of v0.3.1 it is a
Vite + React 18 + TypeScript + `@react-three/fiber` app, replacing the previous
single-file vanilla-Three.js page.

### Layout

| Path                            | What                                                  |
|---------------------------------|-------------------------------------------------------|
| `frontend/`                     | Source — React/TS, not shipped to PyPI                |
| `frontend/index.html`           | Vite entry                                            |
| `frontend/vite.config.ts`       | `base: '/viz-assets/'` + `outDir: ../src/autorag/static/viz` |
| `frontend/src/main.tsx`         | `ReactDOM.createRoot`                                 |
| `frontend/src/App.tsx`          | Root component                                        |
| `frontend/src/styles.css`       | Global CSS (ported from the old `viz.html`)           |
| `frontend/src/api/`             | Hand-typed mirror of `src/autorag/viz.py` schemas + fetch wrappers |
| `frontend/src/state/`           | Zustand store for cross-component scene state         |
| `frontend/src/hooks/`           | `useVizData`, `useDebouncedSearch`                    |
| `frontend/src/three/`           | r3f components — `Scene`, `PointsLayer`, etc.         |
| `frontend/src/ui/`              | DOM components — `Rail`, `Legend`, `SearchBox`, `Tooltip`, etc. |
| `src/autorag/static/viz/`       | **Committed build output.** `index.html` + hashed `assets/*` |

`frontend/` lives outside `src/autorag/` so `uv` / `ruff` / `mypy` don't scan
TypeScript. The build output lives **inside** the Python package so wheel
packaging picks it up via the existing `static/` glob — no `MANIFEST.in`
changes.

### Build flow

```bash
cd frontend && npm install && npm run build
```

`tsc -b && vite build` runs the TypeScript project-references build for
typecheck, then emits `index.html` + hashed `assets/index-<hash>.{js,css}`
into `src/autorag/static/viz/` (Vite's `emptyOutDir: true` clears stale
hashes). Commit the rebuilt bundle alongside any `frontend/src/` changes
in the same commit so HTML, source, and assets never drift.

For interactive iteration:

```bash
cd frontend && npm run dev    # Vite on http://localhost:5173
```

The dev server proxies `/viz/data` and `/viz/search` to a separately running
`autorag serve` on port 8000 (see `server.proxy` in `vite.config.ts`).

### FastAPI wiring

- `src/autorag/viz.py` resolves `_VIZ_DIR = static/viz/`, serves
  `_VIZ_DIR / "index.html"` at `GET /viz`, and exports `viz_assets_dir` for
  the static mount.
- `src/autorag/api.py` mounts the assets dir at `/viz-assets` *inside* the
  existing `[rag]` `try/else`, so `[server]`-only installs (without `[rag]`)
  skip both the viz endpoints and the assets mount.
- `base: '/viz-assets/'` in `vite.config.ts` is load-bearing — it makes
  built asset URLs (`<script src="/viz-assets/assets/index-<hash>.js">`)
  match the mount.

### CI / build decision

**Built bundle is committed; CI does not run a node build.** Rationale:

1. Python-only CI keeps passing with zero new infra.
2. PyPI/git-installed wheels need the built assets anyway — they ship via the
   existing `static/` glob.
3. The viz changes infrequently relative to the Python backend.

If a CI build is wanted later: add one GH Actions job with `setup-node@v4`
running `npm ci && npm run build` in `frontend/`. Additive.

### Version pinning

Three.js and `@types/three` are pinned **exactly** (no `^`) — drei 9.x must
move in lockstep with any Three bump. Current: `three@0.165.0`.

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
