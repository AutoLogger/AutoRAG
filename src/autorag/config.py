"""Process-wide settings loaded from environment variables and ``.env``.

Every field is prefixed with ``AUTORAG_`` in the environment — e.g.
``AUTORAG_TOP_K=8`` overrides :attr:`Settings.top_k`. Unrecognized
variables are ignored so callers can share an environment with other
tools.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Default knobs for the RAG pipeline."""

    model_config = SettingsConfigDict(env_prefix="AUTORAG_", env_file=".env", extra="ignore")

    chunk_size: int = 1000
    """Target character count for each chunk."""

    chunk_overlap: int = 200
    """Character overlap between adjacent chunks."""

    top_k: int = 5
    """Default number of chunks to retrieve per query."""

    model: str = "claude-sonnet-4-6"
    """Default LLM model name for generation."""

    db_path: Path = Path("~/.autorag/autorag.db")
    """Location of the SQLite clip database."""


def get_settings() -> Settings:
    """Build a :class:`Settings` instance from the current environment."""
    return Settings()
