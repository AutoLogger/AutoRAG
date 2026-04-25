from __future__ import annotations

from typing import TYPE_CHECKING

from autorag.embed import Embedder, EmbedderEmbeddingFunction
from autorag.store import ChromaStore

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _topics() -> list[dict[str, object]]:
    return [
        {
            "title": "intro",
            "summary": "opening remarks",
            "level": 1,
            "start_s": 0.0,
            "duration_s": 12.0,
            "number": "1",
        },
        {
            "title": "details",
            "summary": "core content",
            "level": 1,
            "start_s": 12.0,
            "duration_s": 45.0,
            "number": "2",
        },
    ]


def test_chroma_store_roundtrip(tmp_path: Path) -> None:
    store = ChromaStore(tmp_path)
    vecs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    store.add_topic_embeddings("clip-1", "Clip One", _topics(), vecs)

    got = store.get_clip_embeddings("clip-1")
    assert set(got.keys()) == {0, 1}
    assert got[0] == [1.0, 0.0, 0.0]
    assert got[1] == [0.0, 1.0, 0.0]


def test_chroma_store_query_returns_nearest(tmp_path: Path) -> None:
    store = ChromaStore(tmp_path)
    topics = [
        *_topics(),
        {
            "title": "outro",
            "summary": "closing",
            "level": 1,
            "start_s": 60.0,
            "duration_s": 5.0,
            "number": "3",
        },
    ]
    vecs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    store.add_topic_embeddings("clip-q", "Clip Q", topics, vecs)

    results = store.query([1.0, 0.0, 0.0], top_k=3)
    assert len(results) == 3
    assert results[0]["topic_index"] == 0
    assert results[0]["clip_id"] == "clip-q"
    assert results[0]["title"] == "intro"
    assert results[0]["similarity"] > 0.99


def test_chroma_store_delete_clip(tmp_path: Path) -> None:
    store = ChromaStore(tmp_path)
    vecs = [[1.0, 0.0], [0.0, 1.0]]
    store.add_topic_embeddings("clip-d", "Clip D", _topics(), vecs)
    assert store.count() == 2

    store.delete_clip("clip-d")
    assert store.count() == 0
    assert store.get_clip_embeddings("clip-d") == {}


def test_embedder_embedding_function(monkeypatch: pytest.MonkeyPatch) -> None:
    canned = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    def fake_embed_texts(self: Embedder, texts: list[str]) -> list[list[float]]:
        assert texts == ["a", "b"]
        return canned

    monkeypatch.setattr(Embedder, "embed_texts", fake_embed_texts)
    ef = EmbedderEmbeddingFunction()
    out = ef(["a", "b"])
    assert len(out) == 2
    assert list(out[0]) == [0.1, 0.2, 0.3]
    assert list(out[1]) == [0.4, 0.5, 0.6]
    assert EmbedderEmbeddingFunction.name() == "autorag-ollama-embedder"
