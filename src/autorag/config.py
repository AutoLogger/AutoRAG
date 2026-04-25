from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AUTORAG_", env_file=".env", extra="ignore")

    data_dir: Path = Path("./data")
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    model: str = "claude-sonnet-4-6"
    db_path: Path = Path("~/.autorag/autorag.db")


def get_settings() -> Settings:
    return Settings()
