Visualization (``/viz``)
========================

``GET /viz`` serves an interactive 3-D scatter plot of every topic in
the SQLite database. The page is a Vite-built React + r3f bundle (see
:doc:`../internals/frontend`); the data behind it comes from two
JSON endpoints.

Requirements: the ``[rag]`` extra (Chroma + UMAP + sklearn) and the
``[server]`` extra (FastAPI). Without ``[rag]``, the FastAPI app boots
fine but the ``/viz`` route and ``/viz-assets`` mount are silently
skipped.

Running it
----------

.. code-block:: bash

    autorag serve --host 127.0.0.1 --port 8000

Then open `http://127.0.0.1:8000/viz <http://127.0.0.1:8000/viz>`_.

The pipeline behind the page
----------------------------

1. **Embed** — every persisted topic's ``"<title>. <summary>"`` is
   embedded with the Ollama embedding model (default
   ``nomic-embed-text``). Cached vectors are read from Chroma; missing
   ones are computed on demand by :class:`autorag.embed.Embedder` and
   written back.
2. **Project** — :func:`autorag.viz.umap_3d` projects every vector to
   3 dimensions with ``metric="cosine"`` and ``n_neighbors=15``.
3. **Cluster** — :func:`autorag.topic_cluster.cluster_embeddings`
   groups topics via agglomerative clustering
   (``linkage="average"``). The cut is controlled by the
   ``distance_threshold`` query param (default 0.35).
4. **Edges** — :func:`autorag.topic_cluster.build_edges` wires each
   topic's top-5 cosine neighbours above 0.60 similarity as
   undirected edges in the scatter.

Endpoints
---------

* ``GET /viz`` — HTML for the React app.
* ``GET /viz/data?distance_threshold=0.35`` — UMAP coordinates,
  cluster labels, and edges as :class:`autorag.viz.VizData`.
* ``GET /viz/search?q=<query>&top_k=10`` — semantic search hits as a
  list of :class:`autorag.viz.SearchResult`.

Example ``/viz/search`` query:

.. code-block:: text

    GET /viz/search?q=gradient+descent&top_k=5

    [
      {
        "point_index": 12,
        "topic_title": "Backpropagation deep-dive",
        "clip_title": "ML Lecture 3",
        "clip_id": "...",
        "similarity": 0.91,
        "summary": "..."
      }
    ]
