"""Topic embedding visualization: Chroma-backed embeddings + UMAP + FastAPI endpoints."""

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

from autorag.chroma_store import ChromaStore, default_chroma_dir
from autorag.config import get_settings
from autorag.db import Database
from autorag.embed import Embedder
from autorag.topic_cluster import build_edges, cluster_embeddings

router = APIRouter()

_VIZ_DIR = pathlib.Path(__file__).parent / "static" / "viz"
_HTML_PATH = _VIZ_DIR / "index.html"
viz_assets_dir = _VIZ_DIR


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


Row = tuple[str, str, dict[str, Any], int]
"""A single point in the viz: (clip_id, clip_title, topic, topic_index)."""


def _collect_rows_embeddings(
    clips: list[dict[str, Any]],
    chroma: ChromaStore,
) -> tuple[list[Row], list[list[float]]]:
    """Build (rows, embeddings) from clip records, filling missing vecs via Ollama.

    ``topic_index`` is the position of the topic within the clip's filtered
    (title-bearing) topic list — same convention used by ``cli._transcribe``
    when writing into Chroma.
    """
    rows: list[Row] = []
    for clip in clips:
        raw = clip.get("topics")
        if not raw:
            continue
        try:
            topics = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        topics = [t for t in topics if t.get("title")]
        for i, t in enumerate(topics):
            rows.append((clip["id"], clip["title"], t, i))

    if not rows:
        return [], []

    stored: dict[str, dict[int, list[float]]] = {}
    for clip in clips:
        try:
            stored[clip["id"]] = chroma.get_clip_embeddings(clip["id"])
        except Exception:
            stored[clip["id"]] = {}

    embeddings: list[list[float] | None] = []
    missing_per_clip: dict[str, list[tuple[int, dict[str, Any], int]]] = {}
    for row_idx, (clip_id, _clip_title, t, topic_index) in enumerate(rows):
        vec = stored.get(clip_id, {}).get(topic_index)
        embeddings.append(vec)
        if vec is None:
            missing_per_clip.setdefault(clip_id, []).append((row_idx, t, topic_index))

    if missing_per_clip:
        all_texts: list[str] = []
        flat: list[tuple[int, dict[str, Any], int]] = []
        for items in missing_per_clip.values():
            for row_idx, t, topic_index in items:
                text = f"{t['title']}. {t['summary']}" if t.get("summary") else t["title"]
                all_texts.append(text)
                flat.append((row_idx, t, topic_index))

        computed = Embedder().embed_texts(all_texts)
        for (row_idx, _t, _topic_index), vec in zip(flat, computed, strict=True):
            embeddings[row_idx] = vec

        clip_titles = {clip["id"]: clip.get("title", "") for clip in clips}
        for clip_id, items in missing_per_clip.items():
            current = dict(stored.get(clip_id, {}))
            for row_idx, _t, topic_index in items:
                vec = embeddings[row_idx]
                if vec is not None:
                    current[topic_index] = vec
            topic_lookup = {ti: t for cid, _, t, ti in rows if cid == clip_id}
            ordered_indices = [idx for idx in sorted(current.keys()) if idx in topic_lookup]
            try:
                chroma.delete_clip(clip_id)
                chroma.add_topic_embeddings(
                    clip_id,
                    str(clip_titles.get(clip_id, "")),
                    [topic_lookup[idx] for idx in ordered_indices],
                    [current[idx] for idx in ordered_indices],
                )
            except Exception:
                pass

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
    db_path = settings.db_path.expanduser()
    db = Database(db_path)
    clips = db.list_clips()
    chroma = ChromaStore(default_chroma_dir(db_path))

    try:
        rows, embeddings = _collect_rows_embeddings(clips, chroma)
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
    for clip_id, clip_title, _, _ in rows:
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
        for i, (clip_id, clip_title, t, _topic_index) in enumerate(rows)
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

    settings = get_settings()
    db_path = settings.db_path.expanduser()
    db = Database(db_path)
    clips = db.list_clips()
    chroma = ChromaStore(default_chroma_dir(db_path))

    try:
        query_vec = Embedder().embed_texts([q])[0]
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        rows, _embeddings = _collect_rows_embeddings(clips, chroma)
        results = chroma.query(query_vec, top_k=top_k)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not rows:
        return []

    row_lookup = {(r[0], r[3]): i for i, r in enumerate(rows)}
    out: list[SearchResult] = []
    for r in results:
        key = (r["clip_id"], r["topic_index"])
        if key not in row_lookup:
            continue
        out.append(
            SearchResult(
                point_index=row_lookup[key],
                topic_title=str(r["title"]),
                clip_title=str(r["clip_title"]),
                clip_id=str(r["clip_id"]),
                similarity=float(r["similarity"]),
                summary=str(r["summary"]),
            )
        )
    return out
