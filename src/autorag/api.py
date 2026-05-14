"""FastAPI server for AutoRAG.

Wraps :class:`autorag.core.AutoRAG` behind HTTP endpoints. Mount this
``app`` with ``autorag serve`` or any ASGI runner.

Endpoints:

* ``GET /health`` — liveness probe.
* ``POST /ingest`` — document ingestion (see
  :class:`~autorag.schemas.IngestRequest`).
* ``POST /query`` — RAG query (see
  :class:`~autorag.schemas.QueryRequest`).

When the ``[rag]`` extra is installed, the ``/viz`` page, its JSON
endpoints (``/viz/data``, ``/viz/search``), and the React asset mount
at ``/viz-assets`` are added on top. A ``[server]``-only install
silently skips them.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import FastAPI

from autorag.core import AutoRAG
from autorag.schemas import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="AutoRAG", version="0.2.0")
"""The FastAPI application instance. Importable as ``autorag.api:app``."""

# Viz endpoints depend on the `[rag]` extra (umap, sklearn, chromadb).
# `[server]` users without `[rag]` get the API minus /viz.
try:
    from autorag.viz import router as viz_router
    from autorag.viz import viz_assets_dir
except ModuleNotFoundError as exc:
    logger.info("viz endpoints disabled (install autorag[rag] to enable): %s", exc)
else:
    from fastapi.staticfiles import StaticFiles

    app.include_router(viz_router)
    app.mount("/viz-assets", StaticFiles(directory=viz_assets_dir), name="viz-assets")


@lru_cache(maxsize=1)
def get_rag() -> AutoRAG:
    """Return a process-wide :class:`~autorag.core.AutoRAG` singleton.

    Cached so request handlers reuse the same vector store / embedder
    instead of re-instantiating per call.
    """
    return AutoRAG()


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe. Always returns ``{"status": "ok"}``."""
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    """Load, chunk, embed, and store the documents referenced by ``req.paths``."""
    return get_rag().ingest(req.paths)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Retrieve relevant chunks and generate an answer for ``req.question``."""
    return get_rag().query(req.question, top_k=req.top_k)
