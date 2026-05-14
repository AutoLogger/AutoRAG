Running the HTTP server
=======================

The ``autorag serve`` command runs a FastAPI server with uvicorn.
Requires the ``[server]`` extra; the ``/viz`` endpoints additionally
need ``[rag]``.

.. code-block:: bash

    autorag serve --host 0.0.0.0 --port 8000

    # With auto-reload (development)
    autorag serve --reload

Endpoints
---------

.. list-table::
   :header-rows: 1
   :widths: 8 22 70

   * - Method
     - Path
     - Description
   * - ``GET``
     - ``/health``
     - Liveness probe — always ``{"status": "ok"}``.
   * - ``POST``
     - ``/ingest``
     - Document ingestion. Body:
       :class:`~autorag.schemas.IngestRequest`.
   * - ``POST``
     - ``/query``
     - RAG query. Body: :class:`~autorag.schemas.QueryRequest`.
   * - ``GET``
     - ``/viz``
     - React 3-D scatter (needs ``[rag]``).
   * - ``GET``
     - ``/viz/data``
     - UMAP coordinates + cluster labels + edges as JSON.
   * - ``GET``
     - ``/viz/search``
     - Semantic search over topic embeddings.
   * - ``GET``
     - ``/viz-assets/*``
     - Static file mount for the React bundle.

Calling the API
---------------

.. code-block:: bash

    curl http://localhost:8000/health
    # {"status":"ok"}

    curl -X POST http://localhost:8000/ingest \
         -H 'content-type: application/json' \
         -d '{"paths": ["./notes"]}'

    curl -X POST http://localhost:8000/query \
         -H 'content-type: application/json' \
         -d '{"question":"What did we decide about retries?","top_k":5}'

Mounting the app yourself
-------------------------

If you need to embed AutoRAG inside a larger FastAPI app, import
``autorag.api:app`` directly and re-mount it or merge its routers:

.. code-block:: python

    from fastapi import FastAPI
    from autorag.api import app as autorag_app

    parent = FastAPI()
    parent.mount("/autorag", autorag_app)

The :func:`~autorag.api.get_rag` helper returns a process-wide
``AutoRAG`` singleton via ``functools.lru_cache``, so reusing the app
across requests doesn't re-instantiate the embedder or vector store.
