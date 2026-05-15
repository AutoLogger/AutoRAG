# AutoRAG Project

Python 3.12 project managed with `uv`. Distribution is **GitHub-hosted, not PyPI** — consumers pin to a git tag (`pip install "autorag[...] @ git+https://github.com/AutoLogger/AutoRAG@v0.6.0`). User-facing documentation is in `docs/` (Sphinx); keep this file in sync with it.

## Package Manager

Always use `uv`, never `pip` directly:

- `uv add <package>` / `uv add --dev <package>`
- `uv run <command>` to execute in the project venv
- `uv sync --all-extras` to install everything
- `uv sync --group docs` to add the docs build deps
- `uv sync --frozen --no-dev` to reproduce the CI base-install job

## Project Layout

- `src/autorag/` — main package (src layout)
- `src/autorag/static/viz/` — **committed** React/r3f build output (shipped in the wheel)
- `frontend/` — TypeScript source for the `/viz` app (not shipped to PyPI)
- `tests/` — pytest tests
- `docs/` — Sphinx documentation (rST + autodoc)
- **SDK entry point**: `from autorag import AutoRAG` (`src/autorag/core.py`)
- CLI: `autorag` (`src/autorag/cli.py`) — thin Typer wrapper over `AutoRAG`
- API server: `src/autorag/api.py` — FastAPI app, also wraps `AutoRAG`

### SDK facade (`core.py`)

`AutoRAG` is the single public class. Flat methods, lazy imports for heavy deps:

| Method                                       | Extras needed                                                    | Purpose                                                                                                |
| -------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `transcribe(file, ...)`                      | `[audio,diarize]` (+ `[youtube]` for URLs)                       | Whisper + diarization → `list[WordSpan]`. `file` is a local path **or** a YouTube URL.                  |
| `generate_topics(words, ...)`                | `[audio,diarize]`                                                | LLM topic extraction on a pre-computed `list[WordSpan]` → `TopicTree`.                                  |
| `build_agent(**kwargs)`                      | `[audio,diarize]`                                                | Returns the combined LangChain `Runnable[Path \| str, TranscriptionResult]` directly.                   |
| `transcribe_blocks(file, seconds=10, ...)`   | `[rag]` on cache hit; `[audio,diarize]` (+ `[youtube]`) on miss  | N-second time-bucketed transcript view. Reads SQLite cache when present, else transcribes + persists.   |
| `persist_transcription(file, words, ...)`    | `[rag]`                                                          | Writes clip row + word spans to SQLite.                                                                |
| `persist_topics(file, topics, ...)`          | `[rag]`                                                          | Writes topic tree to SQLite + indexes topic embeddings in Chroma.                                       |
| `ingest(paths)`                              | base                                                             | Document RAG: load → chunk → embed → store.                                                            |
| `query(question, ...)`                       | base                                                             | Retrieve + generate over ingested corpus.                                                              |

`MissingExtraError` and `_missing_extra` live in `src/autorag/errors.py` (`core.py` re-exports `MissingExtraError` for backwards compat). `MissingExtraError` is a subclass of `ImportError`, so callers that want a single `except` for "AutoRAG isn't fully installed" can catch `ImportError`. Each audio/RAG method does `from autorag.X import ...` *inside the method body* and re-raises `ModuleNotFoundError` as `MissingExtraError` with a friendly extras hint. **Do not move these imports to module-top** — base install (`pip install autorag`) must boot without `chromadb`/`torch`/`whisperx`/`pyannote`/`yt_dlp` installed. The CI `test-base` job enforces this.

### Audio → transcript + topics agent

`src/autorag/agent.py` is the single audio→topics pipeline. Public surface:

- `transcribe_audio(file, **kwargs) -> list[WordSpan]` — Whisper + diarize only.
- `generate_topics(words, **kwargs) -> TopicTree` — pure LLM, no audio.
- `build_topic_runnable(**kwargs) -> Runnable[list[WordSpan], TopicTree]` — LangChain runnable for the topic pass.
- `build_agent(**kwargs) -> Runnable[Path|str, TranscriptionResult]` — Whisper + topics combined.

Most callers should use the `AutoRAG` facade (`transcribe`, `generate_topics`) instead of importing this module directly — the facade handles the lazy-import / extras-error story.

Multi-pass L0/L1/L2 with boundary detection separated from summarization. Five stages:

1. **Whisper** → `list[WordSpan]` (1 call).
2. **L1 boundaries** — single LLM call → `list[{s, e}]`.
3. **3a — Decide subdivide** on plain text, per long L1 → `list[bool]`.
4. **3b — L2 boundaries** per yes-L1, batched → `list[list[{s, e}]]`.
5. **Per-node summarize** every L1 + L2 leaf, batched; plain text in / `{title, summary}` out.
6. **L0 aggregate** → `{title, summary}` for the explicit "what is this audio about" root.

Total LLM calls per clip ≈ `2 + N1_long + N1_yes + N1 + N2_total` (~20 for a 7-minute clip). Final shape: `{"topics": [L0]}` with `L0.children = [L1...]`, each `L1.children = [L2...]` or `[]`. Boundary calls receive a time-bucketed transcript view (`blocks.format_blocks`, fixed 30s windows — one `MM:SS-MM:SS Speaker K: <words>` line per turn instead of one timestamped line per word, keeping boundary prompts compact) and emit `{s, e}` as `MM:SS` strings copied from those range markers; `agent._parse_ts` parses them back to float seconds before tiling (the LLM never does arithmetic). The K summary calls share an identical prompt prefix so Ollama's per-slot prefix cache pays once. Independent retry: a bad boundary call can be replayed without redoing summaries. **Why split boundaries from summaries:** combined "find sections AND summarize" calls caused earlier LLM versions to bleed section content across boundaries or conflate distinct topics; two focused prompts are more reliable and each stage can be retried independently.

Default LLM model: `qwen2.5:14b-instruct-q8_0`. Override via `--llm-model` (CLI) or `llm_model=` (SDK).

Transcription is handled by `autorag.whisper_runner` via **whisperX** (faster-whisper / CTranslate2 backend + wav2vec2 forced-alignment pass for frame-accurate word timestamps). After each `transcribe_segment` call the main CTranslate2 model is removed from the module cache so Python GC can free VRAM; the smaller wav2vec2 alignment model is offloaded to CPU (PyTorch `.to("cpu")`) and restored on the next call. CUDA→CPU fallback on first CUDA error.

Speaker diarization runs via `autorag.diarize` using `pyannote/speaker-diarization-3.1`, which is HuggingFace-gated and requires `HF_TOKEN`. Each `WordSpan` carries a `speaker` field (`"0"`, `"1"`, … normalized in first-appearance order). Without a token (or on pyannote load/runtime failure) the agent logs a warning and labels every word `"0"` — output then matches pre-diarization behavior. After each `_run_diarization` call the pipeline is offloaded to CPU and VRAM freed; `_ensure_pipeline_on_cuda` restores it on the next call. Both transcript views the agent feeds the LLM build on `blocks.group_by_speaker` to coalesce consecutive same-speaker spans into turns: the boundary stages use `blocks.format_blocks` (`MM:SS-MM:SS Speaker K: <words>`, 30s windows) and the per-node summary input emits `Speaker N: <words>` per turn — the LLM always sees explicit turn-taking.

The CLI (`cli.py`) calls `AutoRAG.transcribe()` then `AutoRAG.persist_transcription()`. Persistence helpers (`collapse_lone_children`, `iter_topics_flat`, `topics_to_events`, `derive_session_id`, `load_transcription`) live in `src/autorag/persistence.py`. The first three are write-side; the latter two are base-safe readers used by `AutoRAG.transcribe_blocks` to short-circuit on a cache hit. The 3-level traversal maps the agent's L0 children → category `l1`, L1 children → `l2`, L2 children → `l3`.

### Transcript block formatting (`blocks.py`)

`src/autorag/blocks.py` is dependency-free (stdlib only) and re-exported as `from autorag import format_blocks`. `format_blocks(transcription, seconds)` buckets `WordSpan`s by `floor(s/seconds) * seconds`, then within each non-empty bucket coalesces consecutive same-speaker spans (via `group_by_speaker`) into one `MM:SS-MM:SS Speaker K: <words>` line per turn (K = `int(speaker) + 1`, 1-indexed display). Buckets are separated by a blank line; empty buckets are skipped. A turn that crosses a bucket boundary produces one line per bucket.

### YouTube URL input (`audio_source.py`)

`src/autorag/audio_source.py` provides URL detection (`is_youtube_url`, host-allowlisted via `urllib.parse`) and a `resolve_audio_input(source)` context manager that yields an `AudioSource` dataclass with `path`, `source_url`, `video_id`, plus four optional metadata fields surfaced from yt-dlp's info dict: `title`, `upload_date` (`"YYYYMMDD"`), `duration_s`, and `uploader` (falls back to `info["channel"]`).

Allowlisted hosts: `youtube.com`, `www.youtube.com`, `m.youtube.com`, `music.youtube.com`, `youtu.be`.

For YouTube URLs the helper downloads the best audio stream into a `tempfile.TemporaryDirectory(prefix="autorag-yt-")` via `yt_dlp` (lazy-imported, gated by the `[youtube]` extra; raises `MissingExtraError("youtube", ...)` when missing). `_download_youtube_audio` returns `(Path, dict[str, Any])` — the full info dict — so `resolve_audio_input` can map fields onto `AudioSource` without a parallel argument explosion. For non-URL inputs it verifies the path exists and yields a path-only `AudioSource` (all optional fields `None`).

Both `core.AutoRAG.transcribe()` and the CLI wrap their work in `resolve_audio_input`. The CLI must own the temp lifetime itself because it calls both `transcribe` and `persist_transcription` on the same path — the inner wrapper inside `core.transcribe` is a no-op pass-through for an already-local `Path`, so the double-wrap is safe.

The CLI forwards `src.source_url`, `src.upload_date`, `src.duration_s`, and `src.title` to `persist_transcription`:

- **`source_url`** seeds the clip's `session_id` from the canonical YouTube URL (`_canonical_youtube_url` collapses `youtu.be` / `m.youtube.com` / `www.youtube.com` variants to one form) and is stored as the row's `file_path` so the entry remains valid after the temp download is gone.
- **`upload_date`** (when present) anchors `created_at` and the absolute event timestamps to midnight-UTC of the publish date instead of the temp-file mtime.
- **`duration_s`** is currently informational (no schema column).
- **`title`** is the clip-title fallback when `--title` is absent.

For local paths the session id is the UUID-5 of the resolved path. `autorag.audio_source.default_title_from(source)` resolves a YouTube URL to its video id (or a local path to its file stem) and is the last fallback when neither `--title` nor a yt-dlp title is available.

### CLI surface (`cli.py`)

Every subcommand maps to one or two SDK methods:

- `autorag transcribe SOURCE [--title -t] [--whisper-model -w] [--language -l] [--persist/--no-persist] [--db]` — Whisper + diarization → `WordSpan` list. With `--persist` (default) writes to SQLite.
- `autorag generate-topics SOURCE [--provider -p] [--llm-model -m] [--transcription -T] [--persist/--no-persist]` — full audio→topics pipeline. Accepts a pre-computed `WordSpan` JSON via `-T` to skip Whisper. Persisted topic JSON goes to stdout; a timing breakdown (`whisper`, `agent`, `cli_store_words`, `cli_finalize`, `cli_embed`) goes to stderr.
- `autorag blocks SOURCE [--seconds -n] [--force-retranscribe]` — cached `MM:SS-MM:SS Speaker K: …` view. Reads straight from SQLite when cached (only `[rag]` needed); on miss imports `[audio]`/`[diarize]`/`[youtube]` lazily and runs the full pipeline first. Equivalent to `AutoRAG.transcribe_blocks`.
- `autorag ingest PATH [PATH ...]` — ingest files/dirs into the vector store.
- `autorag query QUESTION [--top-k]` — retrieve + generate over the ingested corpus.
- `autorag serve [--host] [--port] [--reload]` — run the FastAPI server (default `127.0.0.1:8000`).

### HTTP server (`api.py`)

FastAPI app behind `autorag serve`. Requires `[server]`; `/viz*` endpoints additionally need `[rag]`.

| Method | Path             | Description                                                                |
| ------ | ---------------- | -------------------------------------------------------------------------- |
| `GET`  | `/health`        | Liveness probe — always `{"status": "ok"}`.                                |
| `POST` | `/ingest`        | Document ingestion (`schemas.IngestRequest`).                              |
| `POST` | `/query`         | RAG query (`schemas.QueryRequest`).                                        |
| `GET`  | `/viz`           | React 3-D scatter HTML (needs `[rag]`).                                    |
| `GET`  | `/viz/data`      | UMAP coordinates + cluster labels + edges (`viz.VizData`).                 |
| `GET`  | `/viz/search`    | Semantic search over topic embeddings (`list[viz.SearchResult]`).          |
| `GET`  | `/viz-assets/*`  | Static file mount for the React bundle.                                    |

`get_rag()` (`api.py`) returns a process-wide `AutoRAG` singleton via `functools.lru_cache(maxsize=1)`, so reusing the app across requests doesn't re-instantiate the embedder or vector store. Consumers can mount the app inside a larger FastAPI app with `parent.mount("/autorag", autorag_app)`.

### Document RAG settings (`config.py`)

The document-RAG modules (`ingest`, `store`, `retrieve`, `generate`) currently ship as **stub interfaces** — the audio→topics pipeline is fully implemented, document-RAG is the natural plug-in point. Defaults on `autorag.config.Settings` (Pydantic, prefix `AUTORAG_`):

| Field           | Default                  | Env var               |
| --------------- | ------------------------ | --------------------- |
| `chunk_size`    | 1000                     | `AUTORAG_CHUNK_SIZE`  |
| `chunk_overlap` | 200                      | `AUTORAG_CHUNK_OVERLAP` |
| `top_k`         | 5                        | `AUTORAG_TOP_K`       |
| `db_path`       | `~/.autorag/autorag.db`  | `AUTORAG_DB_PATH`     |

### `/viz` data pipeline

The `/viz/data` endpoint runs four steps:

1. **Embed** — every persisted topic's `"<title>. <summary>"` is embedded with the Ollama embedding model. Cached vectors come from Chroma; misses are computed by `autorag.embed.Embedder` and written back.
2. **Project** — `autorag.viz.umap_3d` projects vectors to 3-D with `metric="cosine"`, `n_neighbors=15`.
3. **Cluster** — `autorag.topic_cluster.cluster_embeddings` runs `AgglomerativeClustering(linkage="average")` with cut controlled by `distance_threshold` (default `0.35`).
4. **Edges** — `autorag.topic_cluster.build_edges` wires each topic's top-5 cosine neighbours above `0.60` similarity as undirected edges.

### Ollama tuning (server-side)

Resolve the base URL via `AUTORAG_OLLAMA_BASE_URL` (default `http://localhost:11434`); the embedding model via `AUTORAG_EMBED_MODEL` (default `nomic-embed-text`).

`OLLAMA_NUM_PARALLEL` is the per-agent split:

- **`>= 4`** for the agent's batched stages (3a decide, 3b L2 boundaries, 4 per-node summaries). Required for `Runnable.batch` to actually parallelize.
- **`= 1`** for one-shot calls on a *bigger* model. Ollama pre-reserves all `NUM_PARALLEL` slots' KV cache at the configured `num_ctx`, so 4 idle slots steal VRAM that the bigger model needs. With `NUM_PARALLEL=1` on a 24 GB GPU you can run `qwen2.5:14b-q8_0` (~15 GB) + `num_ctx=16384` (~3 GB KV) with full GPU offload; 32K KV pushes some layers onto CPU. Verify with `ollama ps` after a load.

Other settings:

- **Do NOT** combine `OLLAMA_FLASH_ATTENTION=1` with `OLLAMA_MULTIUSER_CACHE=true` and concurrent slots — triggers `GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers")`. Drop `MULTIUSER_CACHE` (per-slot prefix cache still works, which is what the K identical summary prompts benefit from).
- Per-slot KV-cache sizing (f16): the agent caps `num_ctx` at 16K for the L1 call and 8K for fan-out / summary calls to fit 4 slots × KV + ~9 GB model in a 24 GB budget. These values are conservative enough that bumping the LLM to `qwen2.5:32b-q4_K_M` typically just needs `NUM_PARALLEL=1` and no other changes.

## Conventions

- Every module begins with `from __future__ import annotations`.
- Pydantic v2 `BaseModel` for API schemas; `SettingsConfigDict` for config.
- `TypedDict` lives in `src/autorag/types.py` (dep-free) so SDK consumers can reference `WordSpan`, `TopicDict`, `TopicTree`, `TranscriptionResult` without importing langchain/whisper. New public typed-dicts go here, not in `agent.py`.
- `numpy.typing.NDArray[np.float64]` for numpy array return types (see `viz.umap_3d`).
- **Heavy deps stay lazy.** Base install only has typer + pydantic + langchain-{core,ollama}. Anything that imports `chromadb` / `torch` / `whisperx` / `faster_whisper` / `pyannote` / `umap` / `sklearn` / `pydantic_sqlite` / `yt_dlp` belongs behind a method-body `import` in `core.py` (or the appropriate extras-gated module). When adding a new public method, decide which extra it needs and follow the existing `MissingExtraError` pattern.

### Packaging (`pyproject.toml`)

| Extra      | Modules that import it                                  | Adds                                                       |
| ---------- | ------------------------------------------------------- | ---------------------------------------------------------- |
| `audio`    | `whisper_runner.py`, `agent.py` (whisper)               | whisperx, torch, imageio-ffmpeg                            |
| `diarize`  | `diarize.py`                                            | pyannote.audio, huggingface-hub                            |
| `youtube`  | `audio_source.py` (lazy in `_download_youtube_audio`)   | yt-dlp                                                     |
| `rag`      | `chroma_store.py`, `db.py`, `viz.py`, `topic_cluster.py` | chromadb, umap-learn, scikit-learn, numpy, pydantic_sqlite |
| `server`   | `api.py`                                                | fastapi, uvicorn[standard]                                 |
| `all`      | —                                                       | union of the above                                         |

`[diarize]` rides on top of `[audio]` — pyannote needs the same torch + ffmpeg stack; install them together.

Build backend: `uv_build`. Releases: bump `__version__` in `src/autorag/__init__.py` and `version` in `pyproject.toml`, `uv lock`, commit, `git tag v0.x.0 && git push --tags`. Current version: `0.6.0`.

## Frontend (`/viz` page)

`/viz` is the project's only browser surface. It is a Vite + React 18 + TypeScript + `@react-three/fiber` app, served as a built static bundle by FastAPI.

### Layout

| Path                            | What                                                              |
| ------------------------------- | ----------------------------------------------------------------- |
| `frontend/`                     | Source — TypeScript, **not** shipped to PyPI                       |
| `frontend/index.html`           | Vite entry                                                        |
| `frontend/vite.config.ts`       | `base: '/viz-assets/'` + `outDir: ../src/autorag/static/viz`      |
| `frontend/src/main.tsx`         | `ReactDOM.createRoot`                                             |
| `frontend/src/App.tsx`          | Root component                                                    |
| `frontend/src/styles.css`       | Global CSS                                                        |
| `frontend/src/api/`             | Hand-typed mirror of `src/autorag/viz.py` schemas + fetch wrappers |
| `frontend/src/state/`           | Zustand store for cross-component scene state                      |
| `frontend/src/hooks/`           | `useVizData`, `useDebouncedSearch`                                |
| `frontend/src/three/`           | r3f components — `Scene`, `PointsLayer`, etc.                      |
| `frontend/src/ui/`              | DOM components — `Rail`, `Legend`, `SearchBox`, `Tooltip`, etc.    |
| `src/autorag/static/viz/`       | **Committed build output.** `index.html` + hashed `assets/*`       |

`frontend/` lives outside `src/autorag/` so `uv` / `ruff` / `mypy` don't scan TypeScript. The build output lives **inside** the Python package so wheel packaging picks it up via the existing `static/` glob — no `MANIFEST.in` changes.

`App.tsx` wraps `<Scene>` in `ui/SceneBoundary` (an error boundary): r3f's `<Canvas>` throws synchronously when a WebGL context can't be created (software/headless GL, blocklisted GPU), so the boundary keeps the rail + overlays alive with a "3D view unavailable" notice instead of unmounting the whole app to a blank page. The scene normalizes UMAP coords (bbox centroid → origin, longest axis → 7 world units in `three/layout.ts`) — raw `/viz/data` coordinates are not origin-centred, so without this the cloud renders off-camera.

### Build flow

```bash
cd frontend && npm install && npm run build
```

`tsc -b && vite build` runs the TypeScript project-references build for typecheck, then emits `index.html` + hashed `assets/index-<hash>.{js,css}` into `src/autorag/static/viz/` (Vite's `emptyOutDir: true` clears stale hashes). Commit the rebuilt bundle alongside any `frontend/src/` changes in the same commit so HTML, source, and assets never drift.

For interactive iteration:

```bash
cd frontend && npm run dev    # Vite on http://localhost:5173
```

The dev server proxies `/viz/data` and `/viz/search` to a separately running `autorag serve` on port 8000 (see `server.proxy` in `vite.config.ts`).

### FastAPI wiring

- `src/autorag/viz.py` resolves `_VIZ_DIR = static/viz/`, serves `_VIZ_DIR / "index.html"` at `GET /viz`, and exports `viz_assets_dir` for the static mount.
- `src/autorag/api.py` mounts the assets dir at `/viz-assets` *inside* the existing `[rag]` `try/else`, so `[server]`-only installs (without `[rag]`) silently skip both the viz endpoints and the assets mount.
- `base: '/viz-assets/'` in `vite.config.ts` is load-bearing — it makes built asset URLs (`<script src="/viz-assets/assets/index-<hash>.js">`) match the mount.

### CI / build decision

**Built bundle is committed; CI does not run a node build.** Rationale:

1. Python-only CI keeps passing with zero new infra.
2. PyPI/git-installed wheels need the built assets anyway — they ship via the existing `static/` glob.
3. The viz changes infrequently relative to the Python backend.

If a CI build is wanted later: add one GH Actions job with `setup-node@v4` running `npm ci && npm run build` in `frontend/`. Additive.

### Version pinning

Three.js and `@types/three` are pinned **exactly** (no `^`) — drei 9.x must move in lockstep with any Three bump. Current: `three@0.165.0`.

## Third-Party Stubs

These packages have no stubs — covered by mypy `ignore_missing_imports` overrides:

- `whisperx`, `faster_whisper`, `umap`, `pydantic_sqlite`, `imageio_ffmpeg`, `chromadb`, `pyannote`, `yt_dlp`.

`langchain-ollama` and `langchain-core` ship inline types and need no mypy overrides.

These packages have no stubs — suppress with `# type: ignore[import-untyped]` at the import site:

- `sklearn` (used in `viz.py` and `topic_cluster.py`).

## Pylance / Pyright

`.vscode/settings.json` enables Pylance with `typeCheckingMode: "strict"`. Because Pylance does not read `[tool.mypy]` overrides, the `[tool.pyright]` block in `pyproject.toml` mirrors them: `reportMissingTypeStubs = "none"` (matches the mypy `ignore_missing_imports` set above) and `reportPrivateUsage = "none"` (for accessing `pydantic_sqlite.DataBase._db` directly, which has no public reader). It also disables `reportUnknownArgumentType`/`VariableType`/`MemberType`, since mypy strict already catches the cases we care about and Pylance's strict mode flags `Any` propagation more aggressively than the codebase wants.

If a new untyped third-party dep is added: add it to BOTH the mypy overrides and the pyright config — `reportMissingTypeStubs = "none"` covers all unstubbed deps in one shot.

## Static Analysis Commands

```bash
uv run mypy src/autorag/
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run pytest
```

## Docs build

Local build:

```bash
uv sync --group docs
uv run make -C docs strict        # treats warnings as errors
```

CI also builds and publishes the docs to **GitHub Pages** via `.github/workflows/docs.yml` (see CI Pipeline below). The Pages source must be set to "GitHub Actions" in repo Settings → Pages (one-time, UI-only).

The `[docs]` extra (sphinx, furo, sphinx-autodoc-typehints, myst-parser) lives in `[dependency-groups]` rather than `[project.optional-dependencies]` so it doesn't appear in published wheels.

`docs/conf.py` mirrors the runtime extras in `autodoc_mock_imports` so the strict docs build works from a base+docs install too (no extras) — this is exactly what the Pages workflow installs. **When you add a new extras-gated import, add it to that list as well** — same rule as the mypy overrides / pyright config.

Pydantic API-schema models (`schemas.py`) resolve field annotations at runtime, so imports used only in their annotations (e.g. `pathlib.Path`) must stay as **runtime** imports, never moved into a `TYPE_CHECKING` block. `[tool.ruff.lint.flake8-type-checking] runtime-evaluated-base-classes = ["pydantic.BaseModel"]` stops TC003 from auto-suggesting that move; doing it anyway leaves the model "not fully defined" and breaks `model_rebuild()` / FastAPI schema gen / the autodoc build.

## CI Pipeline

`.github/workflows/ci.yml` runs on every push and PR to `main`. Three parallel jobs:

- **Lint & Type Check** — `ruff check`, `ruff format --check`, `mypy` (installs `--all-extras` so mypy can see torch/chromadb/etc.).
- **Tests (all extras)** — `pytest -v` against the full dependency stack.
- **SDK base install (no extras)** — `uv sync --frozen --no-dev` then asserts `from autorag import AutoRAG` boots and the SDK methods are callable. **This is the regression guard for the lazy-import contract** — if anyone re-introduces a `chromadb`/`torch`/`whisperx`/`pyannote`/`yt_dlp` import at module top in `core.py` / `embed.py` / `__init__.py` / `store.py` / `audio_source.py`, this job fails.

The workflow uses `uv sync --frozen` (fails if `uv.lock` is out of sync with `pyproject.toml`). If you add or change dependencies, run `uv lock` locally before pushing.

`.github/workflows/docs.yml` is a separate workflow: on push to `main` (path-filtered to `docs/**`, `src/autorag/**`, `pyproject.toml`, `uv.lock`, and the workflow itself) plus `workflow_dispatch`, it does `uv sync --frozen --group docs`, `make -C docs strict`, drops a `.nojekyll`, and publishes `docs/_build/html` to GitHub Pages with the official `upload-pages-artifact` / `deploy-pages` actions (build + deploy jobs, `concurrency: pages`, `cancel-in-progress: false`). It installs **base + docs only**, so it doubles as a second guard that the `autodoc_mock_imports` list stays complete. Requires Settings → Pages → Source = "GitHub Actions" (one-time, UI-only); served at `https://autologger.github.io/AutoRAG/`.
