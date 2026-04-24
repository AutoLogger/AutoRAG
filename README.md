# AutoRAG

Transcribe audio files with Whisper, summarize into hierarchical topics with an LLM, and store everything in a local SQLite database. Also includes a RAG scaffold (ingest → embed → retrieve → generate) exposed via CLI and HTTP API.

## Quickstart

```bash
# Install (Whisper + Torch are core deps; add a cloud provider if needed)
uv pip install -e "."
uv pip install -e ".[anthropic]"   # optional: Anthropic cloud provider

# Transcribe a file using Ollama (default — no API key needed)
autorag transcribe session.webm

# Transcribe using Anthropic
export AUTOLOGGER_ANTHROPIC_API_KEY=sk-ant-...
autorag transcribe session.webm --provider anthropic
```

Output is a JSON list of topics printed to stdout. The database is written to `~/.autorag/autorag.db` by default.

## CLI

### `autorag transcribe`

```
autorag transcribe FILE [OPTIONS]

  --title            -t  TEXT   Clip title (defaults to filename stem)
  --whisper-model    -w  TEXT   Whisper model: tiny/base/small/medium/large  [default: base]
  --provider         -p  TEXT   LLM provider: anthropic, openai, gemini, ollama  [default: ollama]
  --llm-model        -m  TEXT   LLM model name (uses provider default if omitted)
  --language         -l  TEXT   Whisper language code (auto-detect if empty)
  --force-retranscribe    FLAG  Re-run Whisper even if cached
  --db                   PATH  Override database path
```

The same file always maps to the same session ID (UUID5 of the resolved path), so Whisper output is cached across runs. Re-running without `--force-retranscribe` hits the Whisper cache and only re-runs the LLM step.

Timing information is printed to stderr after each run.

### `autorag ingest`

```
autorag ingest PATH [PATH ...]

Ingest one or more files or directories into the in-memory vector store.
```

### `autorag query`

```
autorag query QUESTION [--top-k K]

Ask a question against the ingested corpus and print the generated answer.
```

### `autorag serve`

```
autorag serve [--host HOST] [--port PORT] [--reload]

Run the HTTP API server (default: http://127.0.0.1:8000).
```

## HTTP API

Start the server with `autorag serve`, then:

| Method | Path        | Description                                      |
|--------|-------------|--------------------------------------------------|
| GET    | `/health`   | Returns `{"status": "ok"}`                       |
| POST   | `/ingest`   | Ingest files — body: `{"paths": [...]}`          |
| POST   | `/query`    | Ask a question — body: `{"question": "...", "top_k": 5}` |
| GET    | `/viz`      | Interactive 3-D topic scatter (HTML)             |
| GET    | `/viz/data` | PCoA 3-D coordinates for all stored topics (JSON) |

## Providers

| Provider  | Env var                        | Default model        |
|-----------|--------------------------------|----------------------|
| ollama    | *(none — local)*               | granite3.3:8b        |
| anthropic | `AUTOLOGGER_ANTHROPIC_API_KEY` | claude-sonnet-4-6    |
| openai    | `AUTOLOGGER_OPENAI_API_KEY`    | gpt-4o-mini          |
| gemini    | `AUTOLOGGER_GEMINI_API_KEY`    | gemini-2.0-flash     |

## Environment variables

### Transcription (`AUTOLOGGER_` prefix)

| Variable                      | Default                     | Description                        |
|-------------------------------|-----------------------------|------------------------------------|
| `AUTOLOGGER_ANTHROPIC_API_KEY`| *(unset)*                   | API key for Anthropic provider     |
| `AUTOLOGGER_OPENAI_API_KEY`   | *(unset)*                   | API key for OpenAI provider        |
| `AUTOLOGGER_GEMINI_API_KEY`   | *(unset)*                   | API key for Gemini provider        |
| `AUTOLOGGER_OLLAMA_BASE_URL`  | `http://localhost:11434`    | Ollama server URL                  |
| `AUTOLOGGER_WHISPER_DEVICE`   | `auto`                      | `auto`, `cpu`, or `cuda`           |
| `AUTOLOGGER_EMBED_MODEL`      | `nomic-embed-text`          | Ollama model used for `/viz/data`  |

### RAG / general (`AUTORAG_` prefix)

| Variable                | Default                  | Description                          |
|-------------------------|--------------------------|--------------------------------------|
| `AUTORAG_DB_PATH`       | `~/.autorag/autorag.db`  | SQLite database path                 |
| `AUTORAG_MODEL`         | `claude-sonnet-4-6`      | LLM used by the RAG generator        |
| `AUTORAG_ANTHROPIC_API_KEY` | *(unset)*            | API key used by the RAG generator    |
| `AUTORAG_CHUNK_SIZE`    | `1000`                   | Characters per chunk when ingesting  |
| `AUTORAG_CHUNK_OVERLAP` | `200`                    | Overlap between consecutive chunks  |
| `AUTORAG_TOP_K`         | `5`                      | Default number of chunks to retrieve |

## Optional dependencies

```bash
uv pip install -e ".[anthropic]"         # Anthropic SDK
uv pip install -e ".[openai]"            # OpenAI SDK
uv pip install -e ".[gemini]"            # Google GenAI SDK
uv pip install -e ".[all]"              # All cloud providers
```

Whisper and PyTorch are **core** dependencies and are always installed.

## Database schema

Single SQLite database at `~/.autorag/autorag.db` (override with `AUTORAG_DB_PATH`).

```sql
CREATE TABLE audio_clips (
    id              TEXT PRIMARY KEY,   -- UUID5 (stable per resolved file path)
    title           TEXT NOT NULL,      -- user-supplied or filename stem
    file_path       TEXT NOT NULL,
    audio_signature TEXT,               -- SHA-256 of audio content; used for Whisper cache invalidation
    transcription   TEXT,               -- JSON: word-level transcript (see below)
    whisper_cache   TEXT,               -- raw Whisper output; internal cache only
    topics          TEXT,               -- JSON: topic list (see below)
    whisper_model   TEXT,               -- e.g. "base"
    provider        TEXT,               -- e.g. "anthropic"
    llm_model       TEXT,               -- e.g. "claude-sonnet-4-6"
    created_at      TEXT NOT NULL       -- ISO 8601 UTC (file mtime)
);
```

### `transcription` column

Word-level timestamps from Whisper, flattened to absolute offsets from audio start:

```json
[
  {"w": " Hello", "s": 0.0, "e": 0.4, "abs_s": 0.0},
  {"w": " world", "s": 0.4, "e": 0.8, "abs_s": 0.4}
]
```

| Field      | Description                                         |
|------------|-----------------------------------------------------|
| `w`        | Word token (may include leading space)              |
| `s`        | Segment-relative start time (seconds)               |
| `e`        | Segment-relative end time (seconds)                 |
| `abs_s`    | Absolute start offset from audio start (seconds)    |

### `topics` column

Hierarchical topics produced by the LLM, flattened to a list sorted by `start_s`:

```json
[
  {"title": "Introduction",  "level": 1, "start_s": 0.0,  "duration_s": 42.1, "number": "1"},
  {"title": "Setup",         "level": 2, "start_s": 5.2,  "duration_s": 15.0, "number": "1.1"},
  {"title": "Configuration", "level": 2, "start_s": 20.4, "duration_s": 21.7, "number": "1.2"}
]
```

| Field        | Description                                                            |
|--------------|------------------------------------------------------------------------|
| `title`      | LLM-generated topic summary (≤120 chars)                               |
| `level`      | Depth: 1 = top-level, 2 = subtopic, 3 = sub-subtopic                  |
| `start_s`    | Offset from audio start where this topic begins (seconds)             |
| `duration_s` | Duration of this topic; last sibling at each level extends to end     |
| `number`     | Hierarchical label, e.g. `"1.2.3"`                                     |

## Visualization

`GET /viz` serves an interactive 3-D scatter of all stored topics. Topic titles are embedded with Ollama (`nomic-embed-text` by default), projected to 3 dimensions via PCoA (classical MDS), and rendered in the browser. Set `AUTOLOGGER_EMBED_MODEL` to switch the embedding model.
