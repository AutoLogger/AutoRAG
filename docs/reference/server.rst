HTTP server (``autorag.api``)
=============================

FastAPI app behind the ``autorag serve`` command. Exposes
``/health``, ``/ingest``, ``/query``, and — when the ``[rag]`` extra
is installed — the ``/viz`` page and ``/viz/data`` / ``/viz/search``
endpoints. The static viz bundle is mounted at ``/viz-assets`` and is
served from :data:`autorag.viz.viz_assets_dir`.

.. automodule:: autorag.api
   :members:
   :show-inheritance:
   :member-order: bysource

The Pydantic request/response models (``QueryRequest``,
``IngestRequest``, etc.) are documented in
:doc:`types-config`.
