from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI

from autorag.core import AutoRAG
from autorag.schemas import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from autorag.viz import router as viz_router

app = FastAPI(title="AutoRAG", version="0.1.0")
app.include_router(viz_router)


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
