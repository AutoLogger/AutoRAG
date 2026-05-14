Extras model
============

AutoRAG's install extras gate which methods you can call. The base
install only carries ``typer + pydantic + langchain-{core,ollama}``.

Methods → extras
----------------

.. list-table::
   :header-rows: 1
   :widths: 35 25 40

   * - Method
     - Extras needed
     - Purpose
   * - :meth:`AutoRAG.transcribe
       <autorag.core.AutoRAG.transcribe>`
     - ``[audio,diarize]`` (+ ``[youtube]`` for URLs)
     - Whisper + diarization → :data:`~autorag.types.WordSpan` list.
       ``file=`` is a local path or a YouTube URL.
   * - :meth:`AutoRAG.generate_topics
       <autorag.core.AutoRAG.generate_topics>`
     - ``[audio,diarize]``
     - LLM topic extraction on a pre-computed transcript →
       :data:`~autorag.types.TopicTree`.
   * - :meth:`AutoRAG.build_agent
       <autorag.core.AutoRAG.build_agent>`
     - ``[audio,diarize]``
     - The combined Whisper + topics ``Runnable[Path | str,
       TranscriptionResult]``.
   * - :meth:`AutoRAG.transcribe_blocks
       <autorag.core.AutoRAG.transcribe_blocks>`
     - ``[rag]`` on cache hit; ``[audio,diarize]`` (+ ``[youtube]``)
       on miss
     - N-second time-bucketed transcript view. Cache reads need only
       ``[rag]``.
   * - :meth:`AutoRAG.persist_transcription
       <autorag.core.AutoRAG.persist_transcription>`
     - ``[rag]``
     - Write clip row + word spans to SQLite.
   * - :meth:`AutoRAG.persist_topics
       <autorag.core.AutoRAG.persist_topics>`
     - ``[rag]``
     - Persist topic tree + embed topic titles into Chroma.
   * - :meth:`AutoRAG.ingest
       <autorag.core.AutoRAG.ingest>`
     - base
     - Document RAG: load → chunk → embed → store.
   * - :meth:`AutoRAG.query
       <autorag.core.AutoRAG.query>`
     - base
     - Retrieve + generate over the ingested corpus.

Extras → modules
----------------

.. list-table::
   :header-rows: 1
   :widths: 15 55 30

   * - Extra
     - Modules that import it
     - Adds
   * - ``audio``
     - :mod:`autorag.whisper_runner`, :mod:`autorag.agent` (whisper)
     - whisperx, torch, imageio-ffmpeg
   * - ``diarize``
     - :mod:`autorag.diarize`
     - pyannote.audio, huggingface-hub
   * - ``youtube``
     - :mod:`autorag.audio_source` (lazy in
       ``_download_youtube_audio``)
     - yt-dlp
   * - ``rag``
     - :mod:`autorag.chroma_store`, :mod:`autorag.db`,
       :mod:`autorag.viz`, :mod:`autorag.topic_cluster`
     - chromadb, umap-learn, scikit-learn, numpy, pydantic_sqlite
   * - ``server``
     - :mod:`autorag.api`
     - fastapi, uvicorn[standard]
   * - ``all``
     - —
     - union of the above

The contract
------------

A new method on :class:`~autorag.core.AutoRAG` must decide up-front
which extra(s) it needs and follow the existing pattern: do the heavy
import inside the method body, catch :exc:`ModuleNotFoundError`, and
re-raise via :func:`autorag.errors._missing_extra` with the extras
string. Heavy imports at module top in any of ``core.py``,
``embed.py``, ``__init__.py``, ``store.py``, or ``audio_source.py``
will fail the CI ``test-base`` job.

:class:`~autorag.errors.MissingExtraError` is a subclass of
:exc:`ImportError`, so callers that want a single ``except`` for
"AutoRAG isn't fully installed" can catch ``ImportError``.
