Quickstart
==========

This page walks through the three most common ways to use AutoRAG.
Each example assumes :doc:`installation` is done and an Ollama server
is running on ``http://localhost:11434``.

From the CLI
------------

.. code-block:: bash

    # Whisper + diarization → word spans, persisted to SQLite
    autorag transcribe meeting.wav

    # Same, but also run the LLM topic pipeline and embed topics into Chroma
    autorag generate-topics meeting.wav

    # YouTube URLs work everywhere a path does (needs the [youtube] extra)
    autorag generate-topics https://youtu.be/dQw4w9WgXcQ

The CLI writes the topic JSON to stdout and a timing breakdown to
stderr. SQLite lives at ``~/.autorag/autorag.db`` by default; override
with ``--db`` or ``AUTORAG_DB_PATH``.

From the Python SDK
-------------------

.. code-block:: python

    from autorag import AutoRAG

    rag = AutoRAG()

    # 1. Transcribe (Whisper + diarization). Returns list[WordSpan].
    words = rag.transcribe("meeting.wav")

    # 2. LLM topic extraction over the pre-computed transcript.
    topics = rag.generate_topics(words)
    print(topics["topics"])

    # 3. Persist (requires the [rag] extra).
    rag.persist_transcription("meeting.wav", words, title="Weekly sync")
    rag.persist_topics("meeting.wav", topics, words=words, title="Weekly sync")

For dependency-free transcript formatting (e.g. on top of a cached
SQLite row) use :func:`autorag.format_blocks` — no extras required.

From the HTTP API
-----------------

.. code-block:: bash

    autorag serve --host 0.0.0.0 --port 8000 &

    curl -X POST http://localhost:8000/ingest \
         -H 'content-type: application/json' \
         -d '{"paths": ["./notes"]}'

    curl -X POST http://localhost:8000/query \
         -H 'content-type: application/json' \
         -d '{"question": "What did we decide about retries?", "top_k": 5}'

Open ``http://localhost:8000/viz`` in a browser to see the 3-D topic
scatter. See :doc:`user-guide/server` for the full endpoint reference.

Where to go next
----------------

* :doc:`user-guide/transcription` — audio → word spans → topics in detail.
* :doc:`user-guide/youtube` — passing YouTube URLs.
* :doc:`user-guide/document-rag` — ingest + query over text documents.
* :doc:`user-guide/visualization` — driving the ``/viz`` page.
* :doc:`internals/architecture` — the SDK facade and lazy-import contract.
