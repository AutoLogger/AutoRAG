from __future__ import annotations

import json
import time
from pathlib import Path  # noqa: TC003
from typing import Any

import typer

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


@app.command()
def transcribe(
    file: Path = typer.Argument(..., help="Audio file to transcribe (.webm, .mp4, etc.)"),
    title: str | None = typer.Option(
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
    """Transcribe an audio file and output topics as a JSON list."""
    if not file.is_file():
        typer.echo(f"Error: {file} is not a file.", err=True)
        raise typer.Exit(1)

    rag = AutoRAG()

    t0 = time.perf_counter()
    result = rag.transcribe(
        file,
        whisper_model=whisper_model,
        llm_model=llm_model,
        language=language or None,
    )
    agent_secs = time.perf_counter() - t0

    persisted = rag.persist_transcription(
        file,
        result,
        title=title,
        provider=provider,
        llm_model=llm_model,
        whisper_model=whisper_model,
        db_path=db_override,
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
