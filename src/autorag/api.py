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
    return AutoRAG()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    return get_rag().ingest(req.paths)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    return get_rag().query(req.question, top_k=req.top_k)
