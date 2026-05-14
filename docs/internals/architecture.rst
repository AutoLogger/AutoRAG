Architecture
============

AutoRAG is organized around a single public class —
:class:`~autorag.core.AutoRAG` — with flat methods. The CLI
(``autorag``) and the HTTP server (``autorag.api``) are both thin
wrappers around that class.

::

    autorag.cli                 autorag.api (FastAPI)
         │                            │
         └──────────► AutoRAG ◄──────┘
                       │
       ┌───────────────┼─────────────────────┐
       │               │                     │
    audio          rag (docs)            persistence
    ────           ────────              ──────────
    whisper_runner ingest               db (sqlite)
    diarize        embed                persistence helpers
    agent          retrieve             chroma_store
    audio_source   generate
                   store

Every audio / RAG method on the facade follows the same shape:

1. Method body imports the heavy dependency
   (``import whisperx`` / ``import chromadb`` / ``import yt_dlp`` / …).
2. ``ModuleNotFoundError`` is caught and re-raised as
   :class:`~autorag.errors.MissingExtraError` via
   :func:`autorag.errors._missing_extra`, with a hint naming the
   extra to install.
3. The real work happens inside a sub-module that's free to import
   whatever it wants — only the facade has the lazy-import contract.

This is the **lazy-import contract**: ``from autorag import AutoRAG``
must boot from a base install with no torch / chromadb / whisper /
pyannote / yt-dlp installed. The CI ``test-base`` job enforces it by
running ``uv sync --frozen --no-dev`` and asserting the import +
method names exist.

Why this matters
----------------

* SDK consumers can ``pip install autorag`` to get the surface area
  visible (signatures, type hints, ``--help``) and pay for extras
  only when they call into them.
* The CI matrix can run ``mypy`` and the strict Sphinx build under a
  ``--all-extras`` environment without those deps leaking back into
  the published wheel's runtime requirements.
* The Sphinx build uses
  :data:`autodoc_mock_imports <docs.conf.autodoc_mock_imports>` (see
  ``docs/conf.py``) to mirror the extras list so the documentation
  builds from a base install too.

Code-level details, including which extra gates which method, are in
:doc:`extras-model`. The five-stage LLM pipeline behind
``generate_topics`` is documented in :doc:`audio-pipeline-design`.
