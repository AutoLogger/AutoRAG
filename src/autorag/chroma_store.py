from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import chromadb
import numpy as np
from chromadb import Documents, EmbeddingFunction, Embeddings

from autorag.embed import Embedder

if TYPE_CHECKING:
    from pathlib import Path


class EmbedderEmbeddingFunction(EmbeddingFunction[Documents]):
    """Adapt :class:`Embedder` to Chroma's ``EmbeddingFunction`` protocol."""

    def __init__(self, embedder: Embedder | None = None) -> None:
        self._embedder = embedder or Embedder()

    def __call__(self, input: Documents) -> Embeddings:
        vectors = self._embedder.embed_texts(list(input))
        return cast("Embeddings", [np.asarray(v, dtype=np.float32) for v in vectors])

    @staticmethod
    def name() -> str:
        return "autorag-ollama-embedder"


def default_chroma_dir(db_path: Path) -> Path:
    """Return the Chroma persistence directory derived from a SQLite db path."""
    return db_path.expanduser().parent / "chroma"


class ChromaStore:
    """Persistent Chroma collection of per-clip topic embeddings."""

    COLLECTION = "audio_clip_topics"

    def __init__(
        self,
        persist_dir: Path,
        embedding_function: EmbeddingFunction[Documents] | None = None,
    ) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._ef = embedding_function or EmbedderEmbeddingFunction()
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION,
            embedding_function=cast("Any", self._ef),
            metadata={"hnsw:space": "cosine"},
        )

    def add_topic_embeddings(
        self,
        clip_id: str,
        clip_title: str,
        topics: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        if not topics:
            return
        if len(topics) != len(embeddings):
            raise ValueError(
                f"topics ({len(topics)}) and embeddings ({len(embeddings)}) length mismatch"
            )
        ids = [f"{clip_id}:{i}" for i in range(len(topics))]
        documents = [
            f"{t['title']}. {t['summary']}" if t.get("summary") else str(t.get("title", ""))
            for t in topics
        ]
        metadatas: list[dict[str, str | int | float]] = [
            {
                "clip_id": clip_id,
                "clip_title": clip_title,
                "topic_index": i,
                "title": str(t.get("title", "")),
                "summary": str(t.get("summary", "")),
                "level": int(t.get("level", 1)),
                "start_s": float(t.get("start_s", 0.0)),
                "duration_s": float(t.get("duration_s", 0.0)),
                "number": str(t.get("number", "")),
            }
            for i, t in enumerate(topics)
        ]
        self._collection.upsert(
            ids=ids,
            embeddings=cast("Any", embeddings),
            documents=documents,
            metadatas=cast("Any", metadatas),
        )

    def get_clip_embeddings(self, clip_id: str) -> dict[int, list[float]]:
        result = self._collection.get(
            where={"clip_id": clip_id},
            include=["embeddings", "metadatas"],
        )
        embeddings = result.get("embeddings")
        metadatas = result.get("metadatas")
        if embeddings is None or metadatas is None:
            return {}
        out: dict[int, list[float]] = {}
        for emb, meta in zip(embeddings, metadatas, strict=True):
            idx = meta.get("topic_index")
            if isinstance(idx, int):
                out[idx] = [float(x) for x in emb]
        return out

    def query(self, query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        result = self._collection.query(
            query_embeddings=cast("Any", [query_embedding]),
            n_results=top_k,
            include=["metadatas", "distances", "documents"],
        )
        ids = (result.get("ids") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        out: list[dict[str, Any]] = []
        for _id, meta, dist in zip(ids, metadatas, distances, strict=True):
            topic_index = meta.get("topic_index", 0)
            out.append(
                {
                    "clip_id": str(meta.get("clip_id", "")),
                    "clip_title": str(meta.get("clip_title", "")),
                    "topic_index": int(topic_index) if isinstance(topic_index, int) else 0,
                    "title": str(meta.get("title", "")),
                    "summary": str(meta.get("summary", "")),
                    "similarity": 1.0 - float(dist),
                }
            )
        return out

    def delete_clip(self, clip_id: str) -> None:
        self._collection.delete(where={"clip_id": clip_id})

    def count(self) -> int:
        return int(self._collection.count())
