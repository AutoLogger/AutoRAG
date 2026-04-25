"""Semantic clustering and similarity-edge construction for topic embeddings."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering  # type: ignore[import-untyped]
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import-untyped]


def cluster_embeddings(
    embeddings: np.ndarray,
    distance_threshold: float = 0.35,
) -> np.ndarray:
    """Assign cluster labels to topic embeddings using agglomerative clustering.

    distance_threshold is cosine distance (0-2); 0.35 ~ similarity >= 0.65.
    Returns an int array of shape (N,) with labels 0..K-1.
    """
    n = len(embeddings)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1e-10, embeddings)
    clust = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    labels: np.ndarray = np.asarray(clust.fit_predict(safe), dtype=int)
    return labels


def build_edges(
    embeddings: np.ndarray,
    top_n: int = 5,
    min_similarity: float = 0.60,
) -> list[tuple[int, int, float]]:
    """Return undirected similarity edges between topics.

    For each topic, finds top_n most similar neighbours above min_similarity.
    Returns a deduplicated list of (idx_a, idx_b, similarity) with idx_a < idx_b.
    """
    if len(embeddings) < 2:
        return []
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1e-10, embeddings)
    sim = cosine_similarity(safe)
    np.fill_diagonal(sim, -1.0)
    seen: dict[tuple[int, int], float] = {}
    for i in range(len(sim)):
        candidates = np.argsort(sim[i])[::-1][:top_n]
        for j in candidates:
            s = float(sim[i, j])
            if s < min_similarity:
                break
            key = (min(i, int(j)), max(i, int(j)))
            if key not in seen:
                seen[key] = s
    return [(a, b, s) for (a, b), s in seen.items()]
