from __future__ import annotations

import json
import time
import urllib.parse
from pathlib import Path
from typing import Any

import typer

from autorag.audio_source import is_youtube_url
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


def _default_title_from(source: str) -> str:
    """Derive a clip title from a local path or YouTube URL."""
    if is_youtube_url(source):
        parsed = urllib.parse.urlparse(source)
        qs = urllib.parse.parse_qs(parsed.query)
        video_id = (qs.get("v", [""])[0] or parsed.path.lstrip("/")).strip("/")
        return video_id or "youtube-clip"
    return Path(source).stem


@app.command()
def transcribe(
    source: str = typer.Argument(
        ...,
        help="Audio file path or YouTube URL (youtube.com / youtu.be / ...).",
    ),
    title: str | None = typer.Option(
        None, "--title", "-t", help="Clip title (defaults to filename stem or video id)"
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
        help="LLM provider (ollama)",
    ),
    llm_model: str = typer.Option(
        "qwen2.5:14b-instruct-q8_0",
        "--llm-model",
        "-m",
        help="LLM model name (uses provider default if empty)",
    ),
    language: str = typer.Option(
        "", "--language", "-l", help="Whisper language code (auto-detect if empty)"
    ),
    db_override: Path | None = typer.Option(None, "--db", help="Override database path"),
) -> None:
    """Transcribe an audio file or YouTube URL and output topics as a JSON list."""
    from autorag.audio_source import resolve_audio_input

    rag = AutoRAG()

    with resolve_audio_input(source) as src:
        t0 = time.perf_counter()
        result = rag.transcribe(
            src.path,
            whisper_model=whisper_model,
            llm_model=llm_model,
            language=language or None,
        )
        agent_secs = time.perf_counter() - t0

        resolved_title = title or src.title or _default_title_from(source)

        persisted = rag.persist_transcription(
            src.path,
            result,
            title=resolved_title,
            provider=provider,
            llm_model=llm_model,
            whisper_model=whisper_model,
            db_path=db_override,
            source_url=src.source_url,
            upload_date=src.upload_date,
            duration_s=src.duration_s,
        )
    p_timings = persisted["timings"]
    timings: dict[str, float] = {
        "agent": agent_secs,
        "cli_store_words": float(p_timings["store_words"]),
        "cli_finalize": float(p_timings["finalize"]),
        "cli_embed": float(p_timings["embed"]),
    }
    stage_order = ["agent", "cli_store_words", "cli_finalize", "cli_embed"]

    from autorag import whisper_runner  # lazy: requires [audio] extra

    summary: dict[str, Any] = {
        "duration_secs": round(sum(timings.values()), 3),
        "device_used": whisper_runner.resolved_device(),
    }
    clip = persisted["clip"]

    typer.echo("", err=True)
    typer.echo("=== Transcription Timing Breakdown ===", err=True)
    max_label = max(len(s) for s in stage_order)
    for stage in stage_order:
        secs = timings.get(stage, 0.0)
        label = stage.ljust(max_label)
        typer.echo(f"  {label}  {secs:8.3f}s", err=True)
    typer.echo(f"  {'─' * (max_label + 11)}", err=True)
    typer.echo(f"  {'TOTAL'.ljust(max_label)}  {summary['duration_secs']:8.3f}s", err=True)
    typer.echo(f"  device: {summary.get('device_used', 'unknown')}", err=True)
    typer.echo("", err=True)

    if clip and clip.get("topics"):
        typer.echo(clip["created_at"])
        typer.echo(json.dumps(json.loads(clip["topics"]), indent=2))
    else:
        typer.echo("[]")


if __name__ == "__main__":
    app()
