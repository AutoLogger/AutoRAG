CLI reference
=============

``autorag`` is a thin Typer wrapper over
:class:`~autorag.core.AutoRAG`. Every subcommand maps to one (or two)
SDK methods.

``autorag transcribe``
----------------------

Whisper + diarization → ``WordSpan`` list. With ``--persist`` (the
default), the words are written to SQLite.

.. code-block:: text

    autorag transcribe SOURCE [OPTIONS]

    SOURCE                       Audio file path or YouTube URL.
    --title         -t  TEXT     Clip title (defaults to YouTube video
                                 title for URLs, else filename stem /
                                 video id).
    --whisper-model -w  TEXT     tiny / base / small / medium / large
                                 [default: base]
    --language      -l  TEXT     Whisper language code (auto-detect if
                                 empty).
    --persist/--no-persist       Write word spans to SQLite (default: true).
    --db                PATH     Override database path.

``autorag generate-topics``
---------------------------

Full audio→topics pipeline: transcribe (or read from cache, or accept
a pre-computed ``--transcription`` JSON), run the LLM topic
extraction, persist everything.

.. code-block:: text

    autorag generate-topics SOURCE [OPTIONS]

    --provider      -p  TEXT     LLM provider [default: ollama]
    --llm-model     -m  TEXT     LLM model [default: gemma4:latest]
    --transcription -T  TEXT     Pre-computed WordSpan JSON (skip Whisper)
    --persist/--no-persist       Write transcription + topics to
                                 SQLite/Chroma (default: true).

Outputs the persisted topic JSON to stdout; a timing breakdown
(whisper / agent / cli_store_words / cli_finalize / cli_embed) goes to
stderr.

``autorag blocks``
------------------

Cached, dependency-friendly view of a previously transcribed clip:
``MM:SS-MM:SS Speaker K: …`` lines bucketed into N-second blocks.

.. code-block:: text

    autorag blocks SOURCE [OPTIONS]

    --seconds       -n  INT      Block length [default: 10]
    --force-retranscribe         Re-run transcription even if cached.

Reads straight from SQLite when the clip is already there — only the
``[rag]`` extra is needed for the cache hit. On a miss the
``[audio]`` / ``[diarize]`` / ``[youtube]`` extras are imported lazily
to run the full pipeline first, then format. Equivalent SDK call:
:meth:`AutoRAG.transcribe_blocks
<autorag.core.AutoRAG.transcribe_blocks>`.

``autorag ingest``
------------------

.. code-block:: text

    autorag ingest PATH [PATH ...]

Ingest one or more files or directories into the vector store.

``autorag query``
-----------------

.. code-block:: text

    autorag query QUESTION [--top-k K]

Ask a question against the ingested corpus and print the generated
answer.

``autorag serve``
-----------------

.. code-block:: text

    autorag serve [--host HOST] [--port PORT] [--reload]

Run the HTTP API server (default ``http://127.0.0.1:8000``). See
:doc:`server`.
