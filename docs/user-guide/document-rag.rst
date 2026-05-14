Document RAG
============

In addition to the audio→topics pipeline, AutoRAG exposes a generic
document retrieval-augmented generation flow:

.. code-block:: python

    from autorag import AutoRAG

    rag = AutoRAG()
    rag.ingest(["./notes", "./design-docs"])
    answer = rag.query("What did we decide about retries?", top_k=5)
    print(answer)

The flow:

1. :func:`autorag.ingest.load_documents` reads each path into a
   :class:`~autorag.schemas.Document`.
2. :func:`autorag.ingest.chunk_document` splits each document into
   overlapping :class:`~autorag.schemas.Chunk` records sized by
   :attr:`Settings.chunk_size <autorag.config.Settings.chunk_size>` /
   :attr:`Settings.chunk_overlap <autorag.config.Settings.chunk_overlap>`.
3. :class:`autorag.embed.Embedder` calls Ollama
   (``nomic-embed-text`` by default) and writes the resulting vectors
   into each chunk.
4. A :class:`autorag.store.VectorStore` persists them.
5. At query time, :class:`autorag.retrieve.Retriever` embeds the
   question and pulls the ``top_k`` nearest chunks.
6. :class:`autorag.generate.Generator` assembles a context-grounded
   prompt and returns an answer string.

Configuration
-------------

The defaults live on :class:`autorag.config.Settings` and can be
overridden via environment variables (all prefixed with
``AUTORAG_``):

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Setting
     - Default
     - Env var
   * - ``chunk_size``
     - 1000
     - ``AUTORAG_CHUNK_SIZE``
   * - ``chunk_overlap``
     - 200
     - ``AUTORAG_CHUNK_OVERLAP``
   * - ``top_k``
     - 5
     - ``AUTORAG_TOP_K``
   * - ``db_path``
     - ``~/.autorag/autorag.db``
     - ``AUTORAG_DB_PATH``

Status
------

The RAG pipeline modules (``ingest``, ``store``, ``generate``,
``retrieve``) ship as stub interfaces. The audio→topics pipeline is
fully implemented; the document-RAG side is the natural place to plug
in your own loaders and vector store.

The CLI commands ``autorag ingest`` and ``autorag query`` are wired up
to the SDK so a backend swap surfaces through the CLI and HTTP API
without further changes.
