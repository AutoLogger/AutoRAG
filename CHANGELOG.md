# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/AutoLogger/AutoRAG/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/AutoLogger/AutoRAG/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/AutoLogger/AutoRAG/releases/tag/v0.2.0
