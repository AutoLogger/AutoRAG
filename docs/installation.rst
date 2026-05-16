Installation
============

AutoRAG is distributed from GitHub, not PyPI. The base install carries
only ``typer``, ``pydantic``, ``langchain-core``, and
``langchain-ollama`` — anything heavier (Whisper, pyannote, Chroma,
UMAP, yt-dlp, FastAPI) is gated behind an install extra so the import
``from autorag import AutoRAG`` stays fast.

Choosing your extras
--------------------

Pick the smallest set that unlocks the methods you intend to call:

.. list-table::
   :header-rows: 1
   :widths: 15 50 35

   * - Extra
     - Adds
     - Use when you want…
   * - ``audio``
     - whisperx, torch, imageio-ffmpeg
     - …to call :meth:`AutoRAG.transcribe
       <autorag.core.AutoRAG.transcribe>` or
       :meth:`build_agent <autorag.core.AutoRAG.build_agent>`.
   * - ``diarize``
     - pyannote.audio, huggingface-hub
     - …speaker labels on every word (combine with ``audio``).
   * - ``youtube``
     - yt-dlp
     - …to pass a YouTube URL as ``file=``.
   * - ``rag``
     - chromadb, umap-learn, scikit-learn, pydantic_sqlite, numpy
     - …``persist_transcription``, ``persist_topics``, document
       ingest/query, or the ``/viz`` page.
   * - ``server``
     - fastapi, uvicorn[standard]
     - …``autorag serve`` or the HTTP API.
   * - ``all``
     - everything above
     - …the full local-dev stack.

``[diarize]`` rides on top of ``[audio]`` — pyannote needs the same
torch + ffmpeg stack. Install them together.

Installing from a tagged release
--------------------------------

.. code-block:: bash

    # Audio → topics agent only
    pip install "autorag[audio,diarize] @ git+https://github.com/AutoLogger/AutoRAG@v0.7.0"

    # Audio + YouTube URL support
    pip install "autorag[audio,diarize,youtube] @ git+https://github.com/AutoLogger/AutoRAG@v0.7.0"

    # Full stack (audio, diarize, rag, server, youtube)
    pip install "autorag[all] @ git+https://github.com/AutoLogger/AutoRAG@v0.7.0"

Calling a method whose extra is missing raises
:class:`~autorag.errors.MissingExtraError` with a hint naming the
``pip install`` command that fixes it.

Local development
-----------------

Inside a checkout of the repository, AutoRAG uses ``uv`` (not ``pip``):

.. code-block:: bash

    uv sync --all-extras       # install everything
    uv sync --group docs       # add the docs build deps
    uv run pytest              # run the test suite
    uv run autorag --help      # invoke the CLI

See :doc:`internals/packaging` for the release flow.

Required external services
--------------------------

AutoRAG calls Ollama for LLM chat and embeddings; you need a local (or
remote) Ollama running before invoking ``generate_topics``, ``query``,
or the ``/viz`` page. Diarization needs an HF token:

* **Ollama** — ``AUTORAG_OLLAMA_BASE_URL`` (default
  ``http://localhost:11434``) and ``AUTORAG_EMBED_MODEL`` (default
  ``nomic-embed-text``). LLM chat uses ``gemma4:latest`` by
  default (a thinking-capable model; the agent disables thinking by
  default — see the audio-pipeline-design internals page).
* **Hugging Face** — ``HF_TOKEN`` is required for the gated
  ``pyannote/speaker-diarization-3.1`` model. Without it, every word is
  labelled ``"0"``.
