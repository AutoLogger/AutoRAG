from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

import typer

from autorag.config import get_settings
from autorag.core import AutoRAG

app = typer.Typer(help="AutoRAG — automated retrieval-augmented generation.")


@app.command()
def ingest(paths: list[Path] = typer.Argument(..., exists=True, readable=True)) -> None:
    """Ingest one or more files/directories into the vector store."""
    rag = AutoRAG()
    result = rag.ingest([str(p) for p in paths])
    typer.echo(f"Ingested {result.ingested} docs → {result.chunks} chunks.")


@app.command()
def query(
    question: str = typer.Argument(...),
    top_k: int | None = typer.Option(None, "--top-k", "-k"),
) -> None:
    """Ask a question against the ingested corpus."""
    rag = AutoRAG()
    result = rag.query(question, top_k=top_k)
    typer.echo(result.answer)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Run the HTTP API server."""
    import uvicorn

    uvicorn.run("autorag.api:app", host=host, port=port, reload=reload)


def _transcribe(
    file: Path,
    title: Optional[str] = None,
    whisper_model: str = "base",
    provider: Literal["anthropic", "openai", "gemini", "ollama"] = "ollama",
    llm_model: str = "granite3.3:8b",
    language: str = "",
    force_retranscribe: bool = False,
    db_override: Optional[Path] = None,
) -> tuple[list[str], dict[Any, Any], dict[Any, Any] | None, Any]:
    """Transcribe an audio file and output topics as a JSON list."""
    from autorag.db import Database
    from autorag.orchestrator import run_session_transcription

    if not file.is_file():
        typer.echo(f"Error: {file} is not a file.", err=True)
        raise typer.Exit(1)

    settings = get_settings()
    db_path = (db_override or settings.db_path).expanduser()
    db = Database(db_path)

    session_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(file.resolve())))
    clip_title = title or file.stem
    mtime = file.stat().st_mtime
    created_at = (
        datetime.fromtimestamp(mtime, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )

    db.create_clip(
        session_id,
        title=clip_title,
        file_path=str(file.resolve()),
        created_at=created_at,
    )

    import time as _time

    result = run_session_transcription(
        db,
        session_id,
        whisper_model=whisper_model,
        language=language or None,
        provider_name=provider,
        llm_model=llm_model,
        replace_existing=True,
        force_retranscribe=force_retranscribe,
        topic_category_ids=("l1", "l2", "l3"),
    )

    _t = _time.perf_counter()
    words: list[dict] = result["word_spans"]
    pending_events: list[dict] = result["pending_events"]
    db.store_transcription(session_id, words)
    cli_store_words_s = _time.perf_counter() - _t

    transcript_end_s = 0.0
    if words:
        last = words[-1]
        transcript_end_s = last.get("abs_s", 0.0) + (
            last.get("e", 0.0) - last.get("s", 0.0)
        )

    _t = _time.perf_counter()
    db.finalize_topics(
        session_id,
        transcript_end_s,
        events=pending_events,
        provider=provider,
        llm_model=llm_model,
        whisper_model=whisper_model,
    )
    cli_finalize_s = _time.perf_counter() - _t

    _t = _time.perf_counter()
    clip_data = db.get_clip(session_id)
    if clip_data and clip_data.get("topics"):
        _topics = json.loads(clip_data["topics"])
        from autorag.topic_embed import embed_topic_titles
        _texts = [
            f"{t['title']}. {t['summary']}" if t.get("summary") else t["title"]
            for t in _topics if t.get("title")
        ]
        if _texts:
            try:
                _embeddings = embed_topic_titles(_texts)
                db.store_embeddings(session_id, _embeddings)
            except Exception as _exc:
                typer.echo(f"Warning: embedding computation failed: {_exc}", err=True)
    cli_embed_s = _time.perf_counter() - _t

    timings = result.get("timings", {})
    timings["cli_store_words"] = cli_store_words_s
    timings["cli_finalize"] = cli_finalize_s
    timings["cli_embed"] = cli_embed_s

    stage_order = [
        "db_enumerate",
        "audio_signature",
        "cache_lookup",
        "whisper_model_load",
        "whisper_transcription",
        "db_upsert_transcript",
        "word_flatten",
        "llm_summarize",
        "topic_collapse",
        "db_fanout",
        "cli_store_words",
        "cli_finalize",
        "cli_embed",
    ]

    clip = db.get_clip(session_id)
    return stage_order, result, clip, timings


@app.command()
def transcribe(
    file: Path = typer.Argument(
        ..., help="Audio file to transcribe (.webm, .mp4, etc.)"
    ),
    title: Optional[str] = typer.Option(
        None, "--title", "-t", help="Clip title (defaults to filename stem)"
    ),
    whisper_model: str = typer.Option(
        "base",
        "--whisper-model",
        "-w",
        help="Whisper model size: tiny/base/small/medium/large",
    ),
    provider: str = typer.Option(
        "ollama",
        "--provider",
        "-p",
        help="LLM provider: anthropic, openai, gemini, ollama",
    ),
    llm_model: str = typer.Option(
        "granite3.3:8b",
        "--llm-model",
        "-m",
        help="LLM model name (uses provider default if empty)",
    ),
    language: str = typer.Option(
        "", "--language", "-l", help="Whisper language code (auto-detect if empty)"
    ),
    force_retranscribe: bool = typer.Option(
        False, "--force-retranscribe", help="Re-run Whisper even if cached"
    ),
    db_override: Optional[Path] = typer.Option(
        None, "--db", help="Override database path"
    ),
) -> None:
    """Transcribe an audio file and output topics as a JSON list."""
    stage_order, result, clip, timings = _transcribe(
        file,
        title,
        whisper_model,
        provider,
        llm_model,
        language,
        force_retranscribe,
        db_override,
    )
    cached_note = " (cached — skipped)" if result.get("transcript_cached") else ""
    typer.echo("", err=True)
    typer.echo("=== Transcription Timing Breakdown ===", err=True)
    max_label = max(len(s) for s in stage_order)
    for stage in stage_order:
        secs = timings.get(stage, 0.0)
        label = stage.ljust(max_label)
        note = (
            cached_note
            if stage in ("whisper_model_load", "whisper_transcription")
            else ""
        )
        typer.echo(f"  {label}  {secs:8.3f}s{note}", err=True)
    typer.echo(f"  {'─' * (max_label + 11)}", err=True)
    typer.echo(
        f"  {'TOTAL'.ljust(max_label)}  {result['duration_secs']:8.3f}s", err=True
    )
    typer.echo(f"  device: {result.get('device_used', 'unknown')}", err=True)
    typer.echo("", err=True)

    if clip and clip.get("topics"):
        typer.echo(clip["created_at"])
        typer.echo(json.dumps(json.loads(clip["topics"]), indent=2))
    else:
        typer.echo("[]")


if __name__ == "__main__":
    app()
