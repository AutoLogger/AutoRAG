Persistence
===========

SQLite + Chroma persistence for transcripts, topic trees, and topic
embeddings:

* :mod:`autorag.db` — SQLite handle and ``AudioClip`` row schema via
  ``pydantic_sqlite``.
* :mod:`autorag.persistence` — topic-tree serialization, session-id
  derivation, base-safe transcript readers.
* :mod:`autorag.chroma_store` — persistent Chroma collection for
  topic embeddings, used by the ``/viz`` page.

.. toctree::
   :maxdepth: 1

   persistence/db
   persistence/persistence
   persistence/chroma_store
