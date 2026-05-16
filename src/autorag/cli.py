from __future__ import annotations

import json
import time
from pathlib import Path  # noqa: TC003 — typer needs the runtime type
from typing import TYPE_CHECKING, Any

import typer

from autorag.audio_source import default_title_from
from autorag.core import AutoRAG

if TYPE_CHECKING:
    from autorag.types import WordSpan

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
    language: str = typer.Option(
        "", "--language", "-l", help="Whisper language code (auto-detect if empty)"
    ),
    persist: bool = typer.Option(
        True, "--persist/--no-persist", help="Write word spans to SQLite (default: true)."
    ),
    db_override: Path | None = typer.Option(None, "--db", help="Override database path"),
) -> None:
    """Transcribe an audio file or YouTube URL and output word spans as JSON."""
    from autorag.audio_source import resolve_audio_input

    rag = AutoRAG()

    with resolve_audio_input(source) as src:
        t0 = time.perf_counter()
        words = rag.transcribe(
            src.path,
            whisper_model=whisper_model,
            language=language or None,
        )
        whisper_secs = time.perf_counter() - t0

        store_words_secs = 0.0
        if persist:
            resolved_title = title or src.title or default_title_from(source)
            persisted_words = rag.persist_transcription(
                src.path,
                words,
                title=resolved_title,
                db_path=db_override,
                source_url=src.source_url,
                upload_date=src.upload_date,
                duration_s=src.duration_s,
            )
            store_words_secs = float(persisted_words["timings"]["store_words"])

    timings: dict[str, float] = {
        "whisper": whisper_secs,
        "cli_store_words": store_words_secs,
    }
    stage_order = ["whisper", "cli_store_words"]

    from autorag import whisper_runner  # lazy: requires [audio] extra

    typer.echo("", err=True)
    typer.echo("=== Transcription Timing Breakdown ===", err=True)
    max_label = max(len(s) for s in stage_order)
    for stage in stage_order:
        secs = timings.get(stage, 0.0)
        label = stage.ljust(max_label)
        typer.echo(f"  {label}  {secs:8.3f}s", err=True)
    typer.echo(f"  {'─' * (max_label + 11)}", err=True)
    total = sum(timings.values())
    typer.echo(f"  {'TOTAL'.ljust(max_label)}  {total:8.3f}s", err=True)
    typer.echo(f"  device: {whisper_runner.resolved_device()}", err=True)
    typer.echo("", err=True)

    typer.echo(json.dumps(words))


@app.command()
def generate_topics(
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
        "gemma4:latest",
        "--llm-model",
        "-m",
        help="LLM model name (uses provider default if empty)",
    ),
    language: str = typer.Option(
        "", "--language", "-l", help="Whisper language code (auto-detect if empty)"
    ),
    transcription_json: str | None = typer.Option(
        None,
        "--transcription",
        "-T",
        help="Pre-computed word spans as a JSON string (skips audio transcription).",
    ),
    persist: bool = typer.Option(
        True,
        "--persist/--no-persist",
        help="Write transcription and topics to SQLite/Chroma (default: true).",
    ),
    db_override: Path | None = typer.Option(None, "--db", help="Override database path"),
) -> None:
    """Generate topics for an audio file or YouTube URL, transcribing first if not cached."""
    from autorag.audio_source import is_youtube_url, resolve_audio_input

    rag = AutoRAG()

    words: list[WordSpan] | None = None
    whisper_secs = 0.0
    store_words_secs = 0.0
    ran_whisper = False
    resolved_title: str | None = None
    upload_date_for_persist: str | None = None
    source_url_for_persist: str | None = None

    # Priority 1: caller-supplied transcription JSON
    if transcription_json:
        words = json.loads(transcription_json)
        resolved_title = title or default_title_from(source)
        source_url_for_persist = source if is_youtube_url(source) else None

    # Priority 2: SQLite cache (only when --persist so we have a DB to look up)
    if words is None and persist:
        from autorag.db import Database
        from autorag.persistence import derive_session_id, load_transcription

        session_id = derive_session_id(source)
        resolved_db = (db_override or rag.settings.db_path).expanduser()
        db = Database(resolved_db)
        cached = load_transcription(db, session_id)
        if cached is not None:
            words = cached
            clip: dict[str, Any] | None = db.get_clip(session_id)
            upload_date_for_persist = clip["created_at"][:10].replace("-", "") if clip else None
            source_url_for_persist = source if is_youtube_url(source) else None
            resolved_title = (
                title or (clip.get("title") if clip else None) or default_title_from(source)
            )

    # Priority 3: run Whisper
    if words is None:
        with resolve_audio_input(source) as src:
            t0 = time.perf_counter()
            words = rag.transcribe(
                src.path,
                whisper_model=whisper_model,
                language=language or None,
            )
            whisper_secs = time.perf_counter() - t0
            ran_whisper = True
            resolved_title = title or src.title or default_title_from(source)
            upload_date_for_persist = src.upload_date
            source_url_for_persist = src.source_url

            if persist:
                persisted_words = rag.persist_transcription(
                    src.path,
                    words,
                    title=resolved_title,
                    db_path=db_override,
                    source_url=src.source_url,
                    upload_date=src.upload_date,
                    duration_s=src.duration_s,
                )
                store_words_secs = float(persisted_words["timings"]["store_words"])

    # Generate topics
    t0 = time.perf_counter()
    topics = rag.generate_topics(words, llm_model=llm_model)
    agent_secs = time.perf_counter() - t0

    # Persist topics
    finalize_secs = 0.0
    embed_secs = 0.0
    persisted_clip: dict[str, Any] | None = None
    if persist:
        persisted_topics = rag.persist_topics(
            source,
            topics,
            words=words,
            title=resolved_title,
            provider=provider,
            llm_model=llm_model,
            whisper_model=whisper_model,
            db_path=db_override,
            source_url=source_url_for_persist,
            upload_date=upload_date_for_persist,
        )
        p_timings = persisted_topics["timings"]
        finalize_secs = float(p_timings["finalize"])
        embed_secs = float(p_timings["embed"])
        persisted_clip = persisted_topics["clip"]

    timings: dict[str, float] = {
        "whisper": whisper_secs,
        "agent": agent_secs,
        "cli_store_words": store_words_secs,
        "cli_finalize": finalize_secs,
        "cli_embed": embed_secs,
    }
    stage_order = ["whisper", "agent", "cli_store_words", "cli_finalize", "cli_embed"]

    typer.echo("", err=True)
    typer.echo("=== Topic Generation Timing Breakdown ===", err=True)
    max_label = max(len(s) for s in stage_order)
    for stage in stage_order:
        secs = timings.get(stage, 0.0)
        label = stage.ljust(max_label)
        typer.echo(f"  {label}  {secs:8.3f}s", err=True)
    typer.echo(f"  {'─' * (max_label + 11)}", err=True)
    total = sum(timings.values())
    typer.echo(f"  {'TOTAL'.ljust(max_label)}  {total:8.3f}s", err=True)
    if ran_whisper:
        from autorag import whisper_runner  # lazy: requires [audio] extra

        typer.echo(f"  device: {whisper_runner.resolved_device()}", err=True)
    typer.echo("", err=True)

    if persist and persisted_clip and persisted_clip.get("topics"):
        typer.echo(persisted_clip["created_at"])
        typer.echo(json.dumps(json.loads(persisted_clip["topics"]), indent=2))
    else:
        typer.echo(json.dumps(topics, indent=2))


@app.command()
def blocks(
    source: str = typer.Argument(
        ...,
        help="Audio file path or YouTube URL (youtube.com / youtu.be / ...).",
    ),
    seconds: int = typer.Option(
        10, "--seconds", "-n", min=1, help="Time-block window length in seconds."
    ),
    force_retranscribe: bool = typer.Option(
        False,
        "--force-retranscribe",
        help="Re-run transcription even if a cached copy exists.",
    ),
    title: str | None = typer.Option(
        None, "--title", "-t", help="Clip title (only used on cache miss)"
    ),
    db_override: Path | None = typer.Option(None, "--db", help="Override database path"),
    whisper_model: str = typer.Option("base", "--whisper-model", "-w"),
    language: str = typer.Option(
        "", "--language", "-l", help="Whisper language code (auto-detect if empty)"
    ),
) -> None:
    """Print the transcription as N-second time blocks, one line per speaker turn.

    Reads from the cached SQLite row when present; otherwise runs Whisper
    transcription and persists the words first. Topic generation is not
    performed here; use the ``transcribe`` command for that.
    """
    rag = AutoRAG()
    text = rag.transcribe_blocks(
        source,
        seconds=seconds,
        force_retranscribe=force_retranscribe,
        db_path=db_override,
        whisper_model=whisper_model,
        language=language or None,
        title=title,
    )
    typer.echo(text)


if __name__ == "__main__":
    app()
