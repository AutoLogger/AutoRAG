# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.0] - 2026-05-16

### Added
- `autorag generate-topics` now exposes the LLM tuning knobs that
  `AutoRAG.generate_topics` already accepted: `--num-ctx-l1`,
  `--num-ctx-fanout`, `--max-concurrency`, `--min-subdivide-duration-s`,
  and `--reasoning/--no-reasoning`. Forwarded 1:1 to the facade with the
  same defaults (`8192` / `8192` / `4` / `120.0` / `False`);
  `ollama_base_url` stays env-only via `AUTORAG_OLLAMA_BASE_URL`.
- New `boundary_block_seconds` tuning kwarg (default `30`) on
  `AutoRAG.generate_topics` / `agent.build_topic_runnable` /
  `agent.build_agent`, exposed as `--boundary-block-seconds` on `autorag
  generate-topics`. Sizes the time-bucketed transcript fed to the L1/L2
  boundary prompts (was the hardcoded private `_BOUNDARY_BLOCK_SECONDS`);
  smaller windows give finer `MM:SS` anchors at the cost of more
  boundary-prompt tokens.

### Changed
- **Default topic LLM is now `gemma4:latest`** (8B Q4_K_M, ~9.6 GB), replacing
  `qwen2.5:14b-instruct-q8_0`, across `AutoRAG.generate_topics` /
  `agent.build_topic_runnable` / `build_agent` and the `autorag
  generate-topics` CLI. `gemma4:latest` is a thinking-capable model; because
  all five stages do mechanical JSON extraction, the agent disables thinking
  by default. New overridable `reasoning: bool = False` kwarg on
  `build_topic_runnable` / `build_agent` / `AutoRAG.generate_topics` (sends
  `think: false` to Ollama on thinking models; harmless no-op otherwise) —
  pass `reasoning=True` to trade latency for chain-of-thought. The lighter
  default also frees VRAM: the 4 agent slots + model now sit at ~11 GB on a
  24 GB card (was ~15 GB+ for the qwen 14B).
- The topic agent now keeps the Ollama model resident in VRAM for the whole
  run instead of cold-reloading it (~15 GB) at every stage boundary. All five
  stages share one `num_ctx` and `keep_alive="5m"` (Ollama reloads on any
  `num_ctx` change, so a uniform size is what keeps it warm); `_build_tree`
  issues one throwaway `keep_alive=0` call after the run — or on a stage error
  — to evict the model so it doesn't squat VRAM during the downstream
  embed/viz step. Substantially cuts topic-generation wall-clock.
- `num_ctx_l1` now defaults to `8192` (was `16384`) in
  `AutoRAG.generate_topics` / `agent.build_topic_runnable` / `build_agent`, so
  the L1 call shares the fan-out context size. Trade-off: on very long audio
  (≈1 hr+) the L1 transcript can truncate at 8192 and degrade boundary
  quality — raise `num_ctx_l1` back to `16384` to restore fidelity, at the
  cost of one model reload at the Stage 2→3a boundary.
- **Transcription now defaults to English.** `--language` defaults to `en`
  on `autorag transcribe` / `generate-topics` / `blocks`, and the
  `language` parameter defaults to `"en"` on `AutoRAG.transcribe` /
  `AutoRAG.transcribe_blocks` / `agent.transcribe_audio` /
  `agent.build_agent` (was Whisper auto-detect). Behavior change for SDK
  consumers relying on auto-detect: pass `language=None` (SDK) or
  `--language ""` (CLI) to restore it.

## [0.7.0] - 2026-05-15

### Added
- `GET /viz` now renders the interactive 3-D topic constellation: per-level
  glowing points, clip/cluster coloring, additive knowledge-graph edges, a
  pointer tooltip, two-way rail↔scene hover sync, and debounced semantic
  search with click-to-focus. The React rewrite had previously shipped only
  the left rail, so the page showed no embeddings; the r3f scene
  (`frontend/src/three/`) is now implemented and the committed bundle
  rebuilt. UMAP coordinates are recentred/scaled in `three/layout.ts` (raw
  `/viz/data` coords are not origin-centred), and an error boundary keeps
  the rail usable if WebGL is unavailable.
- Hosted documentation at <https://autologger.github.io/AutoRAG/>, published
  to GitHub Pages on every push to `main` (`.github/workflows/docs.yml`).
- `autorag.blocks.mmss(t)` — public `MM:SS` second-formatter (promoted from
  the private `_mmss`), now exported in `autorag.blocks.__all__`.

### Changed
- The topic agent's L1/L2 boundary detection now feeds the LLM a 30-second
  time-bucketed transcript via `blocks.format_blocks` (one
  `MM:SS-MM:SS Speaker K: <words>` line per turn) instead of one timestamped
  line per word, and the boundary LLM emits `MM:SS` offsets that
  `agent._parse_ts` converts back to seconds in code. Cuts boundary-prompt
  size sharply; `AutoRAG.generate_topics` / `build_agent` signatures and the
  `Runnable[list[WordSpan], TopicTree]` contract are unchanged.

### Removed
- `src/autorag/static/viz.html` — the original vanilla Three.js `/viz`
  page. It was orphaned once `/viz` switched to the React bundle (`viz.py`
  serves `static/viz/index.html`, never this file) and had been shipping
  unused in the wheel via the `static/` glob.

### Fixed
- `IngestRequest` (`POST /ingest`) is no longer left "not fully defined":
  `pathlib.Path` is imported at runtime again so Pydantic can resolve the
  `paths` field. Restores `IngestRequest.model_rebuild()`, FastAPI OpenAPI
  schema generation, and the Sphinx autodoc build.
- The strict docs build no longer fails under `--all-extras`:
  `transformers` (pulled transitively by `langchain_core`, a base dep) is
  now mocked in `autodoc_mock_imports`, so base+docs and all-extras builds
  take the same path.

## [0.6.0] - 2026-05-12

### Changed
- Replaced `openai-whisper` with **whisperX** (faster-whisper / CTranslate2
  backend + wav2vec2 forced-alignment pass). Transcription is ~4× faster and
  word-level timestamps are frame-accurate rather than Whisper-estimated.
  The `[audio]` extra now pulls `whisperx` instead of `openai-whisper`; the
  public API (`AutoRAG.transcribe`, `WordSpan` shape) is unchanged.

## [0.5.0] - 2026-05-11

### Changed
- `AutoRAG.generate_topics()` now applies `collapse_lone_children` before
  returning, so callers always receive a normalized `TopicTree` regardless of
  whether `persist_topics` is called. `persist_topics` no longer collapses the
  tree itself.

### Fixed
- Suppress spurious pyannote `UserWarning` about `std()` degrees of freedom
  from `StatsPool` on single-frame diarization segments; the warning was
  harmless (pyannote handles the NaN internally) but polluted log output.

## [0.4.0] - 2026-05-11

### Added
- `AutoRAG.generate_topics(words, ...)` → `TopicTree`: pure LLM topic extraction
  on pre-computed `list[WordSpan]`, no audio involved.
- `AutoRAG.persist_topics(file, topics, ...)`: stores the topic tree to SQLite
  and embeds topic titles into Chroma. Call after `persist_transcription`.
- `build_topic_runnable()` in `agent.py` — LangChain
  `Runnable[list[WordSpan], TopicTree]` (Whisper-free; `build_agent` wraps it).
- `agent.transcribe_audio(file)` → `list[WordSpan]` and
  `agent.generate_topics(words)` → `TopicTree` as standalone module-level
  helpers (lower-level alternatives to the `AutoRAG` facade).
- `autorag generate-topics` CLI command: transcribes (or reads from cache),
  generates LLM topics, and persists transcription + topics + embeddings.

### Changed
- `AutoRAG.transcribe()` now returns `list[WordSpan]` instead of
  `TranscriptionResult`; call `generate_topics()` separately for the LLM topic
  tree.
- `AutoRAG.persist_transcription()` now stores word spans only; call
  `persist_topics()` to persist the topic tree and Chroma embeddings.
- `autorag transcribe` CLI now only transcribes and persists word spans (no LLM
  topic generation). Use `autorag generate-topics` for the full pipeline.

### Removed
- `abs_s` field removed from `WordSpan` dict construction in `agent.py`
  (was redundant with `s` and was never declared in the `WordSpan` TypedDict).

## [0.3.3] - 2026-05-11

### Fixed
- Whisper and pyannote pipeline VRAM is released immediately after inference:
  `transcribe_segment` and `_run_diarization` now move their models to CPU and
  call `torch.cuda.empty_cache()` so Ollama's LLM stages start with the GPU
  unencumbered. Both modules restore to CUDA automatically on the next call.

## [0.3.2] - 2026-05-10

### Changed
- `/viz` rail (header / stats / legend / size legend / controls / search /
  topic list) now renders from the React app, fed by a typed `useVizData()`
  hook hitting `/viz/data`. Color-mode and edges-visible state are held in a
  Zustand store (`frontend/src/state/vizStore.ts`) so the canvas (Phase C+)
  can read the same toggles. Phase B: DOM only — `<canvas>`, raycast,
  tooltip, and search wiring are still in the unmodified `viz.html` until
  later phases land them in `frontend/src/three/`.

## [0.3.1] - 2026-05-10

### Changed
- `/viz` is now served from a Vite-built React + TypeScript bundle under
  `src/autorag/static/viz/index.html`, mounted alongside a new `/viz-assets`
  static route. Source lives in the new top-level `frontend/` directory
  (outside `src/autorag/` so `uv`/`ruff`/`mypy` don't scan TypeScript).
  Phase A: scaffold + FastAPI wiring only — the existing Three.js scene is
  preserved in `viz.html` and will be ported to `react-three-fiber` in
  subsequent commits.

## [0.3.0] - 2026-05-10

### Added
- `transcribe` accepts YouTube URLs via the `[youtube]` extra; URL is downloaded
  to a temp `.webm` through `autorag.audio_source.resolve_audio_input` (lazy
  `yt_dlp` import).
- `AudioSource` carries `source_url`, `video_id`, `title`, `upload_date`,
  `duration_s`, and `uploader` lifted from yt-dlp's info dict. The CLI forwards
  these to `persist_transcription`.
- `autorag.blocks.format_blocks` (re-exported as `from autorag import
  format_blocks`) renders a `WordSpan` list as N-second time blocks with one
  `MM:SS-MM:SS Speaker K: ...` line per speaker turn. Pure stdlib — callable
  from a base install.
- `AutoRAG.transcribe_blocks(file, seconds=10, ...)` returns the same formatted
  output, reading from the SQLite cache when available and otherwise running
  the full transcribe + persist pipeline first. Requires `[rag]` for the cache
  path, `[audio,diarize]` (+ `[youtube]` for URLs) on cache miss.
- `autorag blocks SOURCE [-n SECONDS]` CLI command wrapping
  `transcribe_blocks`.
- `autorag.persistence.derive_session_id(file_or_url)` and
  `load_transcription(db, session_id)` expose the session-id derivation and
  the cached-transcription read path as base-safe public helpers.

### Changed
- `session_id` is derived deterministically from the canonical YouTube URL
  (`youtu.be` / `m.youtube.com` / `www.youtube.com` variants collapse to one
  form) so re-runs overwrite the same SQLite row.
- Renamed remaining `AUTOLOGGER_*` env vars to `AUTORAG_*`; devcontainer mount
  updated to match.
- Clip `created_at` and absolute event timestamps anchor to the YouTube
  `upload_date` (midnight UTC) when present, instead of the temp-file mtime.
- `default_title_from(source)` moved from `cli.py` (private
  `_default_title_from`) to `autorag.audio_source` as a public helper.
- `group_by_speaker` moved from `agent.py` to `autorag.blocks` and is now part
  of the public surface; `agent._format_transcript` re-imports it from there.

## [0.2.0] - 2026-05-10

### Added
- SDK facade `from autorag import AutoRAG` with flat methods (`transcribe`,
  `build_agent`, `persist_transcription`, `ingest`, `query`).
- Pip-installable from GitHub:
  `pip install "autorag[...] @ git+https://github.com/AutoLogger/AutoRAG@v0.2.0"`.
- Optional extras: `[audio]`, `[diarize]`, `[rag]`, `[server]`, `[all]`.
  `MissingExtraError` is raised with a friendly hint when an extra is missing.
- Speaker diarization via `pyannote/speaker-diarization-3.1` (gated by
  `[diarize]` + `HF_TOKEN`). Each `WordSpan` carries a `speaker` field.
- Unified multi-pass L0/L1/L2 topic agent in `src/autorag/agent.py`, with
  boundary detection separated from per-node summarization.
- GitHub Actions CI: lint/type-check, full-extras tests, and an SDK base-install
  regression guard for the lazy-import contract.

### Changed
- All LLM and embedding calls migrated to `langchain-ollama`.
- Topic embeddings moved from a SQLite column into a persistent Chroma store.
- Default topic model is `qwen2.5:14b-instruct-q8_0`.

### Removed
- Non-Ollama LLM providers.
- Unused `replace_existing` parameter from the transcription flow.

[Unreleased]: https://github.com/AutoLogger/AutoRAG/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/AutoLogger/AutoRAG/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/AutoLogger/AutoRAG/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/AutoLogger/AutoRAG/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/AutoLogger/AutoRAG/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/AutoLogger/AutoRAG/compare/v0.3.3...v0.4.0
[0.3.3]: https://github.com/AutoLogger/AutoRAG/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/AutoLogger/AutoRAG/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/AutoLogger/AutoRAG/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/AutoLogger/AutoRAG/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/AutoLogger/AutoRAG/releases/tag/v0.2.0
