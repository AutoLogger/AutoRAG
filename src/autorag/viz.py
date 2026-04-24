"""Topic embedding visualization: Ollama embeddings + PCoA + FastAPI endpoints."""

from __future__ import annotations

import json
import os
import pathlib
import urllib.request
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from autorag.config import get_settings
from autorag.db import Database

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
    x: float
    y: float
    z: float


class VizData(BaseModel):
    points: list[TopicPoint]
    clip_ids: list[str]
    clip_titles: dict[str, str]
    total_topics: int
    total_clips: int


# ---------------------------------------------------------------------------
# Ollama embedding
# ---------------------------------------------------------------------------


def embed_topic_titles(titles: list[str]) -> list[list[float]]:
    base_url = os.environ.get(
        "AUTOLOGGER_OLLAMA_BASE_URL", "http://localhost:11434"
    ).rstrip("/")
    model = os.environ.get("AUTOLOGGER_EMBED_MODEL", "nomic-embed-text")
    payload = json.dumps({"model": model, "input": titles}).encode()
    req = urllib.request.Request(
        f"{base_url}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())
    except Exception as exc:
        raise RuntimeError(
            f"Ollama embedding request failed ({base_url}): {exc}"
        ) from exc
    return body["embeddings"]


# ---------------------------------------------------------------------------
# Cosine distance matrix
# ---------------------------------------------------------------------------


def cosine_distance_matrix(embeddings: list[list[float]]) -> np.ndarray:
    E = np.array(embeddings, dtype=np.float64)
    norms = np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-10)
    E_norm = E / norms
    cos_sim = np.clip(E_norm @ E_norm.T, -1.0, 1.0)
    return 1.0 - cos_sim  # (N, N) in [0, 2]


# ---------------------------------------------------------------------------
# PCoA (classical MDS via eigendecomposition)
# ---------------------------------------------------------------------------


def pcoa_3d(dist_matrix: np.ndarray) -> np.ndarray:
    N = dist_matrix.shape[0]
    if N == 1:
        return np.zeros((1, 3))
    D2 = dist_matrix**2
    H = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * (H @ D2 @ H)
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    k = min(3, N - 1)
    lam = np.maximum(eigenvalues[:k], 0.0)
    coords = eigenvectors[:, :k] * np.sqrt(lam)
    if coords.shape[1] < 3:
        coords = np.hstack([coords, np.zeros((N, 3 - coords.shape[1]))])
    return coords  # (N, 3)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/viz", response_class=HTMLResponse, include_in_schema=False)
def viz_page() -> Any:
    return HTMLResponse(_HTML_PATH.read_text(encoding="utf-8"))


@router.get("/viz/data", response_model=VizData)
def viz_data() -> VizData:
    settings = get_settings()
    db = Database(settings.db_path.expanduser())
    clips = db.list_clips()

    rows: list[tuple[str, str, dict]] = []
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
        return VizData(
            points=[],
            clip_ids=[],
            clip_titles={},
            total_topics=0,
            total_clips=len(clips),
        )

    titles = [r[2]["title"] for r in rows]
    try:
        embeddings = embed_topic_titles(titles)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    dist = cosine_distance_matrix(embeddings)
    coords = pcoa_3d(dist)

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
            x=float(coords[i, 0]),
            y=float(coords[i, 1]),
            z=float(coords[i, 2]),
        )
        for i, (clip_id, clip_title, t) in enumerate(rows)
    ]

    return VizData(
        points=points,
        clip_ids=clip_ids,
        clip_titles=seen,
        total_topics=len(points),
        total_clips=len(clips),
    )
