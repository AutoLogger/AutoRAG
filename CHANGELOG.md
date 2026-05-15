# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Hosted documentation at <https://autologger.github.io/AutoRAG/>, published
  to GitHub Pages on every push to `main` (`.github/workflows/docs.yml`).

### Fixed
- `IngestRequest` (`POST /ingest`) is no longer left "not fully defined":
  `pathlib.Path` is imported at runtime again so Pydantic can resolve the
  `paths` field. Restores `IngestRequest.model_rebuild()`, FastAPI OpenAPI
  schema generation, and the Sphinx autodoc build.

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

[Unreleased]: https://github.com/AutoLogger/AutoRAG/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/AutoLogger/AutoRAG/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/AutoLogger/AutoRAG/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/AutoLogger/AutoRAG/compare/v0.3.3...v0.4.0
[0.3.3]: https://github.com/AutoLogger/AutoRAG/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/AutoLogger/AutoRAG/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/AutoLogger/AutoRAG/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/AutoLogger/AutoRAG/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/AutoLogger/AutoRAG/releases/tag/v0.2.0
