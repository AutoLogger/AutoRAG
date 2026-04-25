"""Topic embedding visualization: Ollama embeddings + UMAP + FastAPI endpoints."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np
import numpy.typing as npt
import umap
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine  # type: ignore[import-untyped]

from autorag.config import get_settings
from autorag.db import Database
from autorag.topic_cluster import build_edges, cluster_embeddings
from autorag.topic_embed import embed_topic_titles

router = APIRouter()

_HTML_PATH = pathlib.Path(__file__).parent / "static" / "viz.html"


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class TopicPoint(BaseModel):
    topic_title: str
    clip_id: str
    clip_title: str
    level: int
    start_s: float
    duration_s: float
    number: str
    summary: str = ""
    x: float
    y: float
    z: float
    cluster_id: int = 0


class Edge(BaseModel):
    a: int
    b: int
    similarity: float


class VizData(BaseModel):
    points: list[TopicPoint]
    clip_ids: list[str]
    clip_titles: dict[str, str]
    total_topics: int
    total_clips: int
    edges: list[Edge] = []
    total_clusters: int = 0


class SearchResult(BaseModel):
    point_index: int
    topic_title: str
    clip_title: str
    clip_id: str
    similarity: float
    summary: str = ""


# ---------------------------------------------------------------------------
# UMAP dimensionality reduction
# ---------------------------------------------------------------------------


def umap_3d(embeddings: list[list[float]]) -> npt.NDArray[np.float64]:
    emb = np.array(embeddings, dtype=np.float64)
    n = len(emb)
    if n == 1:
        return np.zeros((1, 3))
    n_components = min(3, n - 1)
    n_neighbors = min(15, n - 1)
    reducer = umap.UMAP(
        n_components=n_components,
        metric="cosine",
        n_neighbors=n_neighbors,
        random_state=42,
    )
    coords: npt.NDArray[np.float64] = np.asarray(reducer.fit_transform(emb), dtype=np.float64)
    if coords.shape[1] < 3:
        coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])))
    return coords  # (N, 3)


# ---------------------------------------------------------------------------
# Shared row/embedding collection
# ---------------------------------------------------------------------------


def _collect_rows_embeddings(
    clips: list[dict[str, Any]],
) -> tuple[list[tuple[str, str, dict[str, Any]]], list[list[float]]]:
    """Build (rows, embeddings) from clip records, filling missing vecs via Ollama."""
    rows: list[tuple[str, str, dict[str, Any]]] = []
    for clip in clips:
        raw = clip.get("topics")
        if not raw:
            continue
        try:
            topics = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        for t in topics:
            if t.get("title"):
                rows.append((clip["id"], clip["title"], t))

    if not rows:
        return [], []

    stored: dict[str, dict[str, list[float]]] = {}
    for clip in clips:
        raw_emb = clip.get("embeddings")
        raw_topics = clip.get("topics")
        if not raw_emb or not raw_topics:
            continue
        try:
            emb_list: list[list[float]] = json.loads(raw_emb)
            topic_list: list[dict[str, Any]] = json.loads(raw_topics)
            stored[clip["id"]] = {
                t["title"]: emb_list[i]
                for i, t in enumerate(topic_list)
                if t.get("title") and i < len(emb_list)
            }
        except (json.JSONDecodeError, TypeError, IndexError):
            pass

    embeddings: list[list[float] | None] = []
    missing_indices: list[int] = []

    for i, (clip_id, _clip_title, t) in enumerate(rows):
        vec = stored.get(clip_id, {}).get(t["title"])
        embeddings.append(vec)
        if vec is None:
            missing_indices.append(i)

    if missing_indices:
        missing_texts = [
            f"{rows[idx][2]['title']}. {rows[idx][2]['summary']}"
            if rows[idx][2].get("summary")
            else rows[idx][2]["title"]
            for idx in missing_indices
        ]
        computed = embed_topic_titles(missing_texts)
        for idx, vec in zip(missing_indices, computed, strict=False):
            embeddings[idx] = vec

    return rows, [e for e in embeddings if e is not None]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/viz", response_class=HTMLResponse, include_in_schema=False)
def viz_page() -> HTMLResponse:
    return HTMLResponse(_HTML_PATH.read_text(encoding="utf-8"))


@router.get("/viz/data", response_model=VizData)
def viz_data(
    distance_threshold: float = Query(default=0.35, ge=0.0, le=1.0),
) -> VizData:
    settings = get_settings()
    db = Database(settings.db_path.expanduser())
    clips = db.list_clips()

    try:
        rows, embeddings = _collect_rows_embeddings(clips)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not rows:
        return VizData(
            points=[],
            clip_ids=[],
            clip_titles={},
            total_topics=0,
            total_clips=len(clips),
        )

    coords = umap_3d(embeddings)

    emb_matrix = np.array(embeddings, dtype=np.float64)
    cluster_labels = cluster_embeddings(emb_matrix, distance_threshold=distance_threshold)
    raw_edges = build_edges(emb_matrix)

    seen: dict[str, str] = {}
    for clip_id, clip_title, _ in rows:
        seen.setdefault(clip_id, clip_title)
    clip_ids = list(seen.keys())

    points = [
        TopicPoint(
            topic_title=t["title"],
            clip_id=clip_id,
            clip_title=clip_title,
            level=int(t.get("level", 1)),
            start_s=float(t.get("start_s", 0.0)),
            duration_s=float(t.get("duration_s", 0.0)),
            number=str(t.get("number", "")),
            summary=str(t.get("summary", "")),
            x=float(coords[i, 0]),
            y=float(coords[i, 1]),
            z=float(coords[i, 2]),
            cluster_id=int(cluster_labels[i]) if i < len(cluster_labels) else 0,
        )
        for i, (clip_id, clip_title, t) in enumerate(rows)
    ]

    edges = [Edge(a=a, b=b, similarity=s) for a, b, s in raw_edges]
    total_clusters = int(cluster_labels.max()) + 1 if len(cluster_labels) > 0 else 0

    return VizData(
        points=points,
        clip_ids=clip_ids,
        clip_titles=seen,
        total_topics=len(points),
        total_clips=len(clips),
        edges=edges,
        total_clusters=total_clusters,
    )


@router.get("/viz/search", response_model=list[SearchResult])
def viz_search(
    q: str = Query(..., min_length=1),
    top_k: int = Query(default=10, ge=1, le=100),
) -> list[SearchResult]:
    q = q.strip()
    if not q:
        return []

    try:
        query_vec = embed_topic_titles([q])[0]
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    settings = get_settings()
    db = Database(settings.db_path.expanduser())
    clips = db.list_clips()

    try:
        rows, embeddings = _collect_rows_embeddings(clips)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not rows:
        return []

    query_arr = np.array(query_vec, dtype=np.float64).reshape(1, -1)
    emb_matrix = np.array(embeddings, dtype=np.float64)
    sims = sk_cosine(query_arr, emb_matrix)[0]

    top_indices = np.argsort(sims)[::-1][:top_k]
    return [
        SearchResult(
            point_index=int(idx),
            topic_title=rows[idx][2]["title"],
            clip_title=rows[idx][1],
            clip_id=rows[idx][0],
            similarity=float(sims[idx]),
            summary=str(rows[idx][2].get("summary", "")),
        )
        for idx in top_indices
    ]
