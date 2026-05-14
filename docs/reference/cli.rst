CLI (``autorag.cli``)
=====================

The ``autorag`` console script is a thin Typer wrapper over
:class:`~autorag.core.AutoRAG`. The exposed commands —
``ingest``, ``query``, ``serve``, ``transcribe``,
``transcribe-blocks`` — own temp-file lifetimes for YouTube URLs and
forward optional metadata (title, upload date, source URL) to
:meth:`AutoRAG.persist_transcription
<autorag.core.AutoRAG.persist_transcription>`.

.. automodule:: autorag.cli
   :members:
   :show-inheritance:
   :member-order: bysource
