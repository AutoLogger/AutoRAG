# AutoRAG

[![CI](https://github.com/AutoLogger/AutoRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/AutoLogger/AutoRAG/actions/workflows/ci.yml)

Transcribe audio files with Whisper, summarize into a 3-level hierarchical topic outline with an LLM, and store everything in a local SQLite database. Includes a semantic visualization layer (UMAP 3-D scatter, agglomerative clustering, cosine-similarity search) and a RAG scaffold (ingest → embed → retrieve → generate) exposed via CLI and HTTP API.

## Quickstart

```bash
# Install full stack (audio + diarization + RAG + server + YouTube download)
uv sync --all-extras

# Transcribe a local audio file using Ollama (no API key needed)
autorag transcribe session.webm

# …or a YouTube URL — yt-dlp downloads the audio to a temp .webm first
autorag transcribe https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

Output is a JSON list of topics printed to stdout. Timing info goes to stderr. The database is written to `~/.autorag/autorag.db` by default.

## Install as a library

AutoRAG is also a pip-installable SDK. Install from a tagged release on GitHub:

```bash
# Audio → topics agent only (Whisper + diarization)
pip install "autorag[audio,diarize] @ git+https://github.com/AutoLogger/AutoRAG@v0.2.0"

# Add YouTube URL support (yt-dlp)
pip install "autorag[audio,diarize,youtube] @ git+https://github.com/AutoLogger/AutoRAG@v0.2.0"

# Full stack (also installs Chroma + UMAP + FastAPI)
pip install "autorag[all] @ git+https://github.com/AutoLogger/AutoRAG@v0.2.0"
```

```python
from autorag import AutoRAG

rag = AutoRAG()

# Local file
result = rag.transcribe("meeting.wav")

# Or a YouTube URL — downloaded to a temp .webm for the call's duration.
# Requires the [youtube] extra.
result = rag.transcribe("https://youtu.be/dQw4w9WgXcQ")

print(result["topics"])           # hierarchical topic tree
print(result["transcription"])    # word-level spans with speaker labels

# Optional: persist to SQLite + index topic embeddings (requires [rag] extra)
rag.persist_transcription("meeting.wav", result, title="Weekly sync")
```

### Extras

| Extra      | Adds                                         | Use when you want…                                  |
|------------|----------------------------------------------|------------------------------------------------------|
| `audio`    | openai-whisper, torch, imageio-ffmpeg        | …to call `rag.transcribe()` / `rag.build_agent()` |
| `diarize`  | pyannote.audio, huggingface-hub              | …speaker labels (combine with `audio`)               |
| `youtube`  | yt-dlp                                       | …to pass a YouTube URL to `rag.transcribe()` / `autorag transcribe` |
| `rag`      | chromadb, umap-learn, scikit-learn, pydantic_sqlite, numpy | …`rag.persist_transcription()`, viz, or document RAG |
| `server`   | fastapi, uvicorn[standard]                   | …`autorag serve` / the HTTP API                       |
| `all`      | everything above                             | …the full local-dev stack                             |

`[diarize]` is meant to ride on top of `[audio]` — pyannote needs the same torch + ffmpeg stack. Install both together: `pip install 'autorag[audio,diarize]'`.

### Releasing a new version

1. Bump `__version__` in `src/autorag/__init__.py` and `version` in `pyproject.toml`.
2. `uv lock` to refresh, commit.
3. `git tag v0.x.0 && git push --tags`.

Consumers then pin to the tag: `pip install "autorag[...] @ git+https://github.com/AutoLogger/AutoRAG@v0.x.0"`.

## CLI

### `autorag transcribe`

```
autorag transcribe SOURCE [OPTIONS]

  SOURCE                        Audio file path or YouTube URL
                                (youtube.com / youtu.be / m.youtube.com / music.youtube.com)
  --title            -t  TEXT   Clip title (defaults to YouTube video title for URLs, else filename stem / video id)
  --whisper-model    -w  TEXT   Whisper model: tiny/base/small/medium/large  [default: base]
  --provider         -p  TEXT   LLM provider (ollama)  [default: ollama]
  --llm-model        -m  TEXT   LLM model name  [default: qwen2.5:14b-instruct-q8_0]
  --language         -l  TEXT   Whisper language code (auto-detect if empty)
  --db                   PATH  Override database path
```

For local files, the same path always maps to the same session ID (UUID5 of its resolved path), so re-runs overwrite the same row. YouTube URLs are downloaded to a temp `.webm` (via yt-dlp, requires the `[youtube]` extra) and the session ID is seeded from the canonical `https://www.youtube.com/watch?v=<id>` URL — `youtu.be/X`, `m.youtube.com/watch?v=X`, and `www.youtube.com/watch?v=X` all collapse to the same row. The stored row's `title`, `created_at`, and `file_path` are populated from yt-dlp's info dict (video title, upload date as midnight UTC, canonical URL) instead of the now-deleted temp file. After topics are stored, topic-title embeddings are computed via Ollama and written to a persistent Chroma collection (alongside the SQLite db) for use by `/viz`.

Timing breakdown is printed to stderr after each run:

```
=== Transcription Timing Breakdown ===
  agent             21.842s
  cli_store_words    0.003s
  cli_finalize       0.005s
  cli_embed          0.231s
  ───────────────────────────
  TOTAL             22.081s
  device: cuda
```

The `agent` stage covers Whisper transcription plus all five LLM passes (L1 boundaries, subdivide decisions, L2 boundaries, per-node summarization, L0 aggregation).

### `autorag ingest`

```
autorag ingest PATH [PATH ...]

Ingest one or more files or directories into the vector store.
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

| Method | Path              | Description                                                           |
|--------|-------------------|-----------------------------------------------------------------------|
| GET    | `/health`         | Returns `{"status": "ok"}`                                            |
| POST   | `/ingest`         | Ingest files — body: `{"paths": [...]}`                               |
| POST   | `/query`          | Ask a question — body: `{"question": "...", "top_k": 5}`             |
| GET    | `/viz`            | Interactive 3-D topic scatter (HTML)                                  |
| GET    | `/viz/data`       | UMAP 3-D coordinates + cluster labels + similarity edges (JSON)       |
| GET    | `/viz/search`     | Semantic search over topics — params: `q=<query>`, `top_k=10` (JSON) |

## Provider

Ollama is the only supported provider. It runs locally — no API key required.

| Provider | Env var                      | Default model              | Notes         |
|----------|------------------------------|----------------------------|---------------|
| ollama   | *(none — local)*             | qwen2.5:14b-instruct-q8_0  | *(built-in)*  |

Ollama is invoked via [LangChain (`langchain-ollama`)](https://pypi.org/project/langchain-ollama/). The provider constructs messages with `SystemMessage`/`HumanMessage` and calls `ChatOllama.with_structured_output(schema, method="json_schema")` to enforce the topic-tree JSON schema. Embeddings are generated with `OllamaEmbeddings.embed_documents()`.

## Environment variables

| Variable                    | Default                  | Description                                                                                              |
|-----------------------------|--------------------------|----------------------------------------------------------------------------------------------------------|
| `AUTORAG_OLLAMA_BASE_URL`   | `http://localhost:11434` | Ollama server URL (used by both the agent and the embedder)                                              |
| `AUTORAG_DB_PATH`           | `~/.autorag/autorag.db`  | SQLite database path                                                                                     |
| `AUTORAG_CHUNK_SIZE`        | `1000`                   | Characters per chunk when ingesting                                                                      |
| `AUTORAG_CHUNK_OVERLAP`     | `200`                    | Overlap between consecutive chunks                                                                       |
| `AUTOLOGGER_WHISPER_DEVICE` | `auto`                   | `auto`, `cpu`, or `cuda` (Whisper + pyannote)                                                            |
| `AUTOLOGGER_EMBED_MODEL`    | `nomic-embed-text`       | Ollama model for topic title embeddings                                                                  |
| `HF_TOKEN`                  | *(unset)*                | HuggingFace token for `pyannote/speaker-diarization-3.1`. Without it, every word is labeled speaker `"0"`. |

Whisper and PyTorch ship with the `[audio]` extra; pyannote with `[diarize]`. See **Install as a library** for the extras matrix.

## Database schema

Single SQLite database at `~/.autorag/autorag.db` (override with `AUTORAG_DB_PATH`).

```sql
CREATE TABLE audio_clips (
    id              TEXT PRIMARY KEY,   -- UUID5 (stable per resolved file path)
    title           TEXT NOT NULL,      -- user-supplied or filename stem
    file_path       TEXT NOT NULL,
    created_at      TEXT NOT NULL,      -- ISO 8601 UTC (file mtime)
    transcription   TEXT,               -- JSON: word-level transcript (see below)
    topics          TEXT,               -- JSON: topic list (see below)
    whisper_model   TEXT,               -- e.g. "base"
    provider        TEXT,               -- e.g. "ollama"
    llm_model       TEXT                -- e.g. "qwen2.5:14b-instruct-q8_0"
);
```

Topic embeddings live alongside the SQLite db in a persistent Chroma collection (`<db_dir>/chroma/`, collection `audio_clip_topics`, cosine distance), keyed by `<clip_id>:<topic_index>`.

### `transcription` column

Word-level timestamps from Whisper, flattened to absolute offsets from audio start:

```json
[
  {"w": " Hello", "s": 0.0, "e": 0.4, "abs_s": 0.0, "speaker": "0"},
  {"w": " world", "s": 0.4, "e": 0.8, "abs_s": 0.4, "speaker": "1"}
]
```

| Field     | Description                                                                                            |
|-----------|--------------------------------------------------------------------------------------------------------|
| `w`       | Word token (may include leading space)                                                                 |
| `s`       | Segment-relative start time (seconds)                                                                  |
| `e`       | Segment-relative end time (seconds)                                                                    |
| `abs_s`   | Absolute start offset from audio start (seconds)                                                       |
| `speaker` | Speaker label `"0"`, `"1"`, … normalized in first-appearance order. `"0"` when diarization is disabled. |

### `topics` column

Hierarchical topics produced by the LLM, flattened to a list sorted by `start_s`:

```json
[
  {"title": "Introduction", "level": 1, "start_s": 0.0,  "duration_s": 42.1, "number": "1",   "summary": "Speaker introduces the session goals."},
  {"title": "Setup",        "level": 2, "start_s": 5.2,  "duration_s": 15.0, "number": "1.1", "summary": "Environment prerequisites are reviewed."},
  {"title": "Config",       "level": 2, "start_s": 20.4, "duration_s": 21.7, "number": "1.2", "summary": "Key config values and their effects."}
]
```

| Field        | Description                                                            |
|--------------|------------------------------------------------------------------------|
| `title`      | LLM-generated topic label (≤120 chars)                                 |
| `summary`    | 2–4 sentence description of what was discussed                         |
| `level`      | Depth: 1 = top-level, 2 = subtopic, 3 = sub-subtopic                  |
| `start_s`    | Offset from audio start where this topic begins (seconds)             |
| `duration_s` | Duration; the last sibling at each level extends to the transcript end |
| `number`     | Hierarchical label, e.g. `"1.2.3"`                                     |

### Topic embeddings (Chroma)

Each topic's `"<title>. <summary>"` (or just `title` when there is no summary) is embedded with the Ollama embedding model (default: `nomic-embed-text`) and upserted into the `audio_clip_topics` Chroma collection. Each record carries the embedding plus metadata (`clip_id`, `clip_title`, `topic_index`, `title`, `summary`, `level`, `start_s`, `duration_s`, `number`); ids are `<clip_id>:<topic_index>` and `topic_index` refers to the position within the clip's filtered (title-bearing) topic list. Used by `/viz/data` and `/viz/search`.

## Visualization

`GET /viz` serves an interactive 3-D scatter of all stored topics. The pipeline:

1. **Embed** — topic titles + summaries are embedded via Ollama. Stored embeddings are reused; missing ones are computed on demand.
2. **Project** — embeddings are projected to 3 dimensions via UMAP (`metric=cosine`, `n_neighbors=15`).
3. **Cluster** — topics are grouped with agglomerative clustering (`metric=cosine`, `linkage=average`, `distance_threshold=0.35`). Threshold is tunable via the `distance_threshold` query param (0.0–1.0).
4. **Edges** — for each topic the top-5 cosine-similar neighbours above 0.60 similarity are wired as undirected edges in the scatter.
5. **Render** — the browser renders the 3-D scatter with Three.js. Points are coloured by cluster; edges are drawn as lines. Hovering shows the topic title, clip, and summary.

### `/viz/data` response

```json
{
  "points": [
    {
      "topic_title": "Introduction",
      "clip_id": "...",
      "clip_title": "Session 1",
      "level": 1,
      "start_s": 0.0,
      "duration_s": 42.1,
      "number": "1",
      "summary": "...",
      "x": 0.12,
      "y": -0.34,
      "z": 0.09,
      "cluster_id": 2
    }
  ],
  "edges": [{"a": 0, "b": 4, "similarity": 0.82}],
  "clip_ids": ["..."],
  "clip_titles": {"...": "Session 1"},
  "total_topics": 47,
  "total_clips": 3,
  "total_clusters": 8
}
```

### `/viz/search` response

```
GET /viz/search?q=gradient+descent&top_k=5
```

```json
[
  {
    "point_index": 12,
    "topic_title": "Backpropagation deep-dive",
    "clip_title": "ML Lecture 3",
    "clip_id": "...",
    "similarity": 0.91,
    "summary": "..."
  }
]
```

## Architecture

```
autorag transcribe FILE
  │
  ├─ db.create_clip()              Register file in SQLite
  ├─ agent.transcribe()            Whisper + 5-stage LLM pipeline
  │    ├─ whisper_runner.get_model() → .transcribe_segment()
  │    ├─ Stage 2: L1 boundaries          (1 LLM call)
  │    ├─ Stage 3a: decide subdivide      (N LLM calls, batched)
  │    ├─ Stage 3b: L2 boundaries         (M LLM calls, batched)
  │    ├─ Stage 4: per-node summaries     (K LLM calls, batched)
  │    └─ Stage 5: L0 aggregate           (1 LLM call)
  ├─ _collapse_lone_children()     Drop single-child chains
  ├─ db.store_transcription()      Persist word spans
  ├─ _topics_to_events() → db.add_analytics_event() × N
  ├─ db.finalize_topics()          Compute durations, persist topics JSON
  └─ Embedder().embed_texts()      Ollama embed → ChromaStore.add_topic_embeddings()

autorag serve
  └─ FastAPI
       ├─ /ingest  POST → core.AutoRAG.ingest()
       ├─ /query   POST → core.AutoRAG.query()
       ├─ /viz          → viz.html (static)
       ├─ /viz/data GET → viz.viz_data()
       │    ├─ db.list_clips()
       │    ├─ ChromaStore.get_clip_embeddings()  (per clip)
       │    ├─ Embedder().embed_texts()           (missing vecs only)
       │    ├─ viz.umap_3d()
       │    ├─ topic_cluster.cluster_embeddings()
       │    └─ topic_cluster.build_edges()
       └─ /viz/search GET → viz.viz_search()
            ├─ Embedder().embed_texts([q])
            └─ ChromaStore.query(query_vec, top_k)
```
