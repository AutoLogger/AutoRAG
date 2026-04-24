"""Ollama embedding helper for topic titles."""

from __future__ import annotations

import json
import os
import urllib.request


def embed_topic_titles(titles: list[str]) -> list[list[float]]:
    base_url = os.environ.get("AUTOLOGGER_OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
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
        raise RuntimeError(f"Ollama embedding request failed ({base_url}): {exc}") from exc
    return body["embeddings"]
