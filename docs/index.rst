AutoRAG documentation
=====================

AutoRAG is an audio-to-topics agent and document retrieval toolkit. It
ships as:

* an importable Python SDK (``from autorag import AutoRAG``),
* a ``autorag`` command-line tool,
* and a FastAPI server (``autorag serve``) with an interactive ``/viz``
  WebGL page for inspecting topic embeddings.

It is built around a single :class:`autorag.core.AutoRAG` facade with
flat methods. Heavy dependencies (Whisper, pyannote, Chroma, UMAP,
yt-dlp) are gated behind install extras so the base import remains
lightweight.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User guide

   user-guide/transcription
   user-guide/youtube
   user-guide/document-rag
   user-guide/visualization
   user-guide/server
   user-guide/cli

.. toctree::
   :maxdepth: 2
   :caption: Internals

   internals/architecture
   internals/extras-model
   internals/audio-pipeline-design
   internals/ollama-tuning
   internals/frontend
   internals/packaging

.. toctree::
   :maxdepth: 2
   :caption: API reference

   reference/index

.. toctree::
   :maxdepth: 1
   :caption: Project

   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
