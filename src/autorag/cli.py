from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Any, Literal

import typer

from autorag.config import get_settings
from autorag.core import AutoRAG
from autorag.embed import Embedder

if TYPE_CHECKING:
    from collections.abc import Generator

    from autorag.agent import TopicDict, TopicTree, WordSpan
    from autorag.db import Database

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


def _collapse_lone_children(tree: TopicTree) -> TopicTree:
    """Drop single-child chains so a subtopic level only exists with ≥2 siblings."""

    def walk(nodes: list[TopicDict]) -> list[TopicDict]:
        out: list[TopicDict] = []
        for node in nodes:
            children = list(node.get("children") or [])
            while len(children) == 1:
                lone = children[0]
                children = list(lone.get("children") or [])
            node["children"] = walk(children)
            out.append(node)
        return out

    return {"topics": walk(tree.get("topics") or [])}


def _iter_topics_flat(tree: TopicTree) -> Generator[tuple[int, TopicDict, str], None, None]:
    """Yield (level, node, number_label) like '1', '1.2', '1.2.3'."""

    def walk(
        nodes: list[TopicDict], level: int, parent_number: str
    ) -> Generator[tuple[int, TopicDict, str], None, None]:
        sibling_count = 0
        for node in nodes:
            title = str(node.get("title", "") or "").strip()
            if not title:
                continue
            sibling_count += 1
            number_label = (
                str(sibling_count) if not parent_number else f"{parent_number}.{sibling_count}"
            )
            yield level, node, number_label
            children = node.get("children") or []
            if level < 3 and children:
                yield from walk(children, level + 1, number_label)

    yield from walk(tree.get("topics") or [], 1, "")


def _topics_to_events(
    db: Database,
    session_id: str,
    tree: TopicTree,
    *,
    audio_start: datetime,
    provider: str,
    llm_model: str,
    topic_category_ids: tuple[str, str, str],
) -> list[dict[str, Any]]:
    """Walk the topic tree and produce analytics events for each titled node.

    Reads the hierarchical-agent's `s`/`e` keys (not `start_s`/`end_s`).
    """
    cat_by_level = {1: topic_category_ids[0], 2: topic_category_ids[1], 3: topic_category_ids[2]}
    events: list[dict[str, Any]] = []

    for level, node, number_label in _iter_topics_flat(tree):
        title = str(node.get("title", "") or "").strip()
        if not title:
            continue
        try:
            start_s = float(node.get("s", 0.0) or 0.0)
        except (TypeError, ValueError):
            start_s = 0.0
        word_end_s: float | None = None
        raw_end = node.get("e")
        try:
            if raw_end is not None:
                word_end_s = float(raw_end)
        except (TypeError, ValueError):
            word_end_s = None

        summary = str(node.get("summary", "") or "").strip()
        metadata: dict[str, Any] = {
            "transcription": {
                "level": level,
                "provider": provider,
                "model": llm_model,
                "number_label": number_label,
                "word_start_s": start_s,
                "word_end_s": word_end_s,
                "summary": summary,
            }
        }

        marked_at = audio_start + timedelta(seconds=max(0.0, start_s))
        category_id = cat_by_level.get(level)
        if not category_id:
            continue

        try:
            event = db.add_analytics_event(
                session_id,
                category=category_id,
                message=title,
                metadata=metadata,
                marked_at_utc=marked_at,
            )
            events.append(event)
        except Exception as exc:
            typer.echo(
                f"Warning: add_analytics_event failed for topic {title!r} (level={level}): {exc}",
                err=True,
            )

    return events


def _transcribe(
    file: Path,
    title: str | None = None,
    whisper_model: str = "base",
    provider: Literal["ollama"] = "ollama",
    llm_model: str = "qwen2.5:14b-instruct-q8_0",
    language: str = "",
    db_override: Path | None = None,
) -> tuple[list[str], dict[str, Any], dict[Any, Any] | None, dict[str, float]]:
    """Transcribe an audio file and output topics as a JSON list."""
    from autorag import whisper_runner
    from autorag.agent import transcribe as run_agent
    from autorag.db import Database

    if not file.is_file():
        typer.echo(f"Error: {file} is not a file.", err=True)
        raise typer.Exit(1)

    settings = get_settings()
    db_path = (db_override or settings.db_path).expanduser()
    db = Database(db_path)

    session_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(file.resolve())))
    clip_title = title or file.stem
    mtime = file.stat().st_mtime
    created_at = datetime.fromtimestamp(mtime, tz=UTC).isoformat().replace("+00:00", "Z")

    db.create_clip(
        session_id,
        title=clip_title,
        file_path=str(file.resolve()),
        created_at=created_at,
    )

    import time as _time

    t_agent = _time.perf_counter()
    agent_out = run_agent(
        file,
        whisper_model=whisper_model,
        language=language or None,
        llm_model=llm_model,
    )
    agent_secs = _time.perf_counter() - t_agent

    words: list[WordSpan] = agent_out["transcription"]
    topic_tree: TopicTree = _collapse_lone_children(agent_out["topics"])

    _t = _time.perf_counter()
    db.store_transcription(session_id, words)  # type: ignore[arg-type]
    cli_store_words_s = _time.perf_counter() - _t

    transcript_end_s = 0.0
    if words:
        last = words[-1]
        transcript_end_s = last.get("abs_s", 0.0) + (last.get("e", 0.0) - last.get("s", 0.0))

    audio_start = datetime.fromtimestamp(mtime, tz=UTC)

    _t = _time.perf_counter()
    pending_events = _topics_to_events(
        db,
        session_id,
        topic_tree,
        audio_start=audio_start,
        provider=provider,
        llm_model=llm_model,
        topic_category_ids=("l1", "l2", "l3"),
    )
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
        from autorag.store import ChromaStore, default_chroma_dir

        _topics = [t for t in json.loads(clip_data["topics"]) if t.get("title")]
        _texts = [
            f"{t['title']}. {t['summary']}" if t.get("summary") else t["title"] for t in _topics
        ]
        if _texts:
            try:
                _embeddings = Embedder().embed_texts(_texts)
                _chroma = ChromaStore(default_chroma_dir(db_path))
                _chroma.delete_clip(session_id)
                _chroma.add_topic_embeddings(
                    session_id,
                    str(clip_data.get("title", "")),
                    _topics,
                    _embeddings,
                )
            except Exception as _exc:
                typer.echo(f"Warning: embedding/index failed: {_exc}", err=True)
    cli_embed_s = _time.perf_counter() - _t

    timings: dict[str, float] = {
        "agent": agent_secs,
        "cli_store_words": cli_store_words_s,
        "cli_finalize": cli_finalize_s,
        "cli_embed": cli_embed_s,
    }
    stage_order = ["agent", "cli_store_words", "cli_finalize", "cli_embed"]

    result: dict[str, Any] = {
        "duration_secs": round(sum(timings.values()), 3),
        "device_used": whisper_runner.resolved_device(),
    }

    clip = db.get_clip(session_id)
    return stage_order, result, clip, timings


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
    stage_order, result, clip, timings = _transcribe(
        file,
        title,
        whisper_model,
        provider,  # type: ignore[arg-type]
        llm_model,
        language,
        db_override,
    )
    typer.echo("", err=True)
    typer.echo("=== Transcription Timing Breakdown ===", err=True)
    max_label = max(len(s) for s in stage_order)
    for stage in stage_order:
        secs = timings.get(stage, 0.0)
        label = stage.ljust(max_label)
        typer.echo(f"  {label}  {secs:8.3f}s", err=True)
    typer.echo(f"  {'─' * (max_label + 11)}", err=True)
    typer.echo(f"  {'TOTAL'.ljust(max_label)}  {result['duration_secs']:8.3f}s", err=True)
    typer.echo(f"  device: {result.get('device_used', 'unknown')}", err=True)
    typer.echo("", err=True)

    if clip and clip.get("topics"):
        typer.echo(clip["created_at"])
        typer.echo(json.dumps(json.loads(clip["topics"]), indent=2))
    else:
        typer.echo("[]")


if __name__ == "__main__":
    app()
