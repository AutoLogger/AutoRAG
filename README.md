# AutoRAG

[![CI](https://github.com/AutoLogger/AutoRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/AutoLogger/AutoRAG/actions/workflows/ci.yml)

Transcribe audio files with Whisper, summarize into a 3-level hierarchical topic outline with an LLM, and store everything in a local SQLite database. Includes a semantic visualization layer (UMAP 3-D scatter, agglomerative clustering, cosine-similarity search) and a RAG scaffold (ingest → embed → retrieve → generate) exposed via CLI and HTTP API.

## Quickstart

```bash
# Install (Whisper + Torch are core deps)
uv sync                        # installs core deps from the lock file

# Transcribe using Ollama (no API key needed)
autorag transcribe session.webm
```

Output is a JSON list of topics printed to stdout. Timing info goes to stderr. The database is written to `~/.autorag/autorag.db` by default.

## CLI

### `autorag transcribe`

```
autorag transcribe FILE [OPTIONS]

  --title            -t  TEXT   Clip title (defaults to filename stem)
  --whisper-model    -w  TEXT   Whisper model: tiny/base/small/medium/large  [default: base]
  --provider         -p  TEXT   LLM provider (ollama)  [default: ollama]
  --llm-model        -m  TEXT   LLM model name (uses provider default if omitted)
  --language         -l  TEXT   Whisper language code (auto-detect if empty)
  --force-retranscribe    FLAG  Re-run Whisper even if cached
  --db                   PATH  Override database path
```

The same file always maps to the same session ID (UUID5 of its resolved path), so Whisper output is cached across runs. Re-running without `--force-retranscribe` skips Whisper and only re-runs the LLM topic extraction. After topics are stored, topic-title embeddings are computed via Ollama and written to a persistent Chroma collection (alongside the SQLite db) for use by `/viz`.

Timing breakdown is printed to stderr after each run:

```
=== Transcription Timing Breakdown ===
  db_enumerate           0.003s
  audio_signature        0.012s
  cache_lookup           0.001s
  whisper_model_load     1.843s
  whisper_transcription  8.201s
  db_upsert_transcript   0.004s
  word_flatten           0.002s
  llm_summarize         12.114s
  topic_collapse         0.000s
  db_fanout              0.005s
  cli_store_words        0.003s
  cli_finalize           0.004s
  cli_embed              0.231s
  ─────────────────────────────
  TOTAL                 22.423s
  device: cuda
```

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

| Provider | Env var                      | Default model | Notes         |
|----------|------------------------------|---------------|---------------|
| ollama   | *(none — local)*             | llama3.1:8b   | *(built-in)*  |

Ollama is invoked via [LangChain (`langchain-ollama`)](https://pypi.org/project/langchain-ollama/). The provider constructs messages with `SystemMessage`/`HumanMessage` and calls `ChatOllama.with_structured_output(schema, method="json_schema")` to enforce the topic-tree JSON schema. Embeddings are generated with `OllamaEmbeddings.embed_documents()`.

## Environment variables

### Transcription / providers (`AUTOLOGGER_` prefix)

| Variable                     | Default                  | Description                             |
|------------------------------|--------------------------|-----------------------------------------|
| `AUTOLOGGER_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL                       |
| `AUTOLOGGER_WHISPER_DEVICE`  | `auto`                   | `auto`, `cpu`, or `cuda`                |
| `AUTOLOGGER_EMBED_MODEL`     | `nomic-embed-text`       | Ollama model for topic title embeddings |

### RAG / general (`AUTORAG_` prefix)

| Variable            | Default                 | Description                          |
|---------------------|-------------------------|--------------------------------------|
| `AUTORAG_DB_PATH`   | `~/.autorag/autorag.db` | SQLite database path                 |
| `AUTORAG_MODEL`     | `claude-sonnet-4-6`     | LLM used by the RAG generator        |
| `AUTORAG_CHUNK_SIZE`    | `1000`              | Characters per chunk when ingesting  |
| `AUTORAG_CHUNK_OVERLAP` | `200`               | Overlap between consecutive chunks  |
| `AUTORAG_TOP_K`         | `5`                 | Default number of chunks to retrieve |

Whisper and PyTorch are **core** dependencies and are always installed.

## Database schema

Single SQLite database at `~/.autorag/autorag.db` (override with `AUTORAG_DB_PATH`).

```sql
CREATE TABLE audio_clips (
    id              TEXT PRIMARY KEY,   -- UUID5 (stable per resolved file path)
    title           TEXT NOT NULL,      -- user-supplied or filename stem
    file_path       TEXT NOT NULL,
    created_at      TEXT NOT NULL,      -- ISO 8601 UTC (file mtime)
    audio_signature TEXT,               -- SHA-256 of audio content; cache key for Whisper
    transcription   TEXT,               -- JSON: word-level transcript (see below)
    whisper_cache   TEXT,               -- raw Whisper output; internal use only
    topics          TEXT,               -- JSON: topic list (see below)
    whisper_model   TEXT,               -- e.g. "base"
    provider        TEXT,               -- e.g. "ollama"
    llm_model       TEXT                -- e.g. "claude-sonnet-4-6"
);
```

Topic embeddings live alongside the SQLite db in a persistent Chroma collection (`<db_dir>/chroma/`, collection `audio_clip_topics`, cosine distance), keyed by `<clip_id>:<topic_index>`.

### `transcription` column

Word-level timestamps from Whisper, flattened to absolute offsets from audio start:

```json
[
  {"w": " Hello", "s": 0.0, "e": 0.4, "abs_s": 0.0},
  {"w": " world", "s": 0.4, "e": 0.8, "abs_s": 0.4}
]
```

| Field   | Description                                         |
|---------|-----------------------------------------------------|
| `w`     | Word token (may include leading space)              |
| `s`     | Segment-relative start time (seconds)               |
| `e`     | Segment-relative end time (seconds)                 |
| `abs_s` | Absolute start offset from audio start (seconds)    |

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
  ├─ orchestrator.run_session_transcription()
  │    ├─ db.list_audio_segments() → db.get_audio_segment_file()
  │    ├─ signatures.compute_audio_signature()  SHA-256 cache key
  │    ├─ db.get_transcript()      Cache hit?
  │    │    no → whisper_runner.get_model() → .transcribe_segment()
  │    │         db.upsert_transcript()
  │    ├─ providers.get_provider().summarize()  LLM call → TopicTree
  │    ├─ _collapse_lone_children()             Enforce ≥2 siblings
  │    └─ db.add_analytics_event() × N          Fanout topics
  ├─ db.store_transcription()      Persist word spans
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
