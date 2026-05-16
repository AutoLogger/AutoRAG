"""Benchmark / compare AutoRAG topic-agent designs and append to the ledger.

This is the bundled runner for the ``autorag-agent-lab`` skill. It runs the
real topic agent (``autorag.agent.build_topic_runnable``, stages 2-5 — Whisper
is skipped by reusing cached word spans), captures mechanical metrics + an
LLM-judge quality score, and appends a reproducible entry to ``LEDGER.md``.

It deliberately does **not** modify ``src/autorag/agent.py``: prompt variants
are applied by monkey-patching the agent's module-level prompt constants in
this process *before* ``build_topic_runnable`` reads them.

Run it from the repo root with the project venv::

    uv run python .claude/skills/autorag-agent-lab/bench.py --prepare-fixtures
    uv run python .claude/skills/autorag-agent-lab/bench.py --design baseline --fixtures fox-new

See ``SKILL.md`` for the full workflow.
"""

# This script lives outside the `autorag` package and imports it as an
# installed dependency. The package ships no `py.typed` marker (it is
# type-checked in place via `mypy src/autorag`), so from here every
# `import autorag…` is "untyped". Disable that one error code file-wide
# rather than scattering brittle per-line ignores (mypy only emits the
# error on the first import of each module, which makes line-level
# `type: ignore`s trip `warn_unused_ignores`).
# mypy: disable-error-code="import-untyped"

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import statistics
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import LLMResult

SKILL_DIR = Path(__file__).resolve().parent
REPO_ROOT = SKILL_DIR.parents[2]
FIXTURES_DIR = SKILL_DIR / "fixtures"
RUNS_DIR = SKILL_DIR / "runs"
PROMPTS_DIR = SKILL_DIR / "prompts"
LEDGER_PATH = SKILL_DIR / "LEDGER.md"
DESIGNS_PATH = SKILL_DIR / "designs.json"

# Source audio for each fixture stem. These ship in the repo's tests/ dir.
FIXTURE_SOURCES: dict[str, str] = {
    "fox-new": "tests/fox-new.webm",
    "3b1b-llm": "tests/3b1b-llm.webm",
    "3b1b-llm2": "tests/3b1b-llm2.webm",
    "quin-rs-tut": "tests/quin-rs-tut.webm",
}

# Knobs accepted by autorag.agent.build_topic_runnable. A design's "knobs"
# object may set any subset; omitted keys fall back to the agent's defaults.
ALLOWED_KNOBS = {
    "llm_model",
    "ollama_base_url",
    "num_ctx_l1",
    "num_ctx_fanout",
    "max_concurrency",
    "min_subdivide_duration_s",
}

# Prompt constants on autorag.agent a prompt-override file may redefine.
AGENT_PROMPT_NAMES = {
    "_L1_SYS",
    "_L1_HUMAN",
    "_DECIDE_SYS",
    "_DECIDE_HUMAN",
    "_L2_SYS",
    "_L2_HUMAN",
    "_NODE_SUM_SYS",
    "_NODE_SUM_HUMAN",
    "_AGG_SYS",
    "_AGG_HUMAN",
    "_BOUNDARY_BLOCK_SECONDS",
}

# agent.py logs "Stage <N> done in <secs>s ..." per stage; map the stage tag
# to a stable metric key.
_STAGE_RE = re.compile(r"Stage (\S+) done in ([\d.]+)s")
_STAGE_KEYS = {"2": "l1", "3a": "decide", "3b": "l2", "4": "summarize", "5": "l0"}

log = logging.getLogger("agent-lab")


# --------------------------------------------------------------------------- #
# Data shapes
# --------------------------------------------------------------------------- #
@dataclass
class Design:
    name: str
    description: str
    knobs: dict[str, Any]
    prompt_override: str | None


class JudgeRubric(BaseModel):
    """Structured output for the LLM judge. Each dimension is scored 1-5."""

    boundary_coherence: int = Field(ge=1, le=5)
    coverage_completeness: int = Field(ge=1, le=5)
    summary_faithfulness: int = Field(ge=1, le=5)
    hierarchy_appropriateness: int = Field(ge=1, le=5)
    overall: int = Field(ge=1, le=5)
    rationale: str


@dataclass
class RunMetrics:
    fixture: str
    total_s: float
    stage_s: dict[str, float]
    llm_calls_pipeline: int
    llm_calls_total: int
    input_tokens: int
    output_tokens: int
    vram_delta_mb: float | None
    vram_peak_mb: float | None
    ollama_size_mb: float | None
    n_l1: int
    n_l2: int
    depth: int
    judge: JudgeRubric | None = None
    tree: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Small subprocess / env helpers
# --------------------------------------------------------------------------- #
def _run(cmd: list[str], timeout: float = 15.0) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def _git_sha() -> str:
    return _run(["git", "rev-parse", "--short", "HEAD"]) or "unknown"


def _ollama_version() -> str:
    return _run(["ollama", "--version"]) or "unknown"


def _gpu_name() -> str:
    out = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    return out.splitlines()[0].strip() if out else "none"


def _ollama_base_url() -> str:
    return os.environ.get("AUTORAG_OLLAMA_BASE_URL", "http://localhost:11434")


# --------------------------------------------------------------------------- #
# Designs registry
# --------------------------------------------------------------------------- #
def load_designs() -> dict[str, Design]:
    raw = cast("dict[str, Any]", json.loads(DESIGNS_PATH.read_text()))
    designs: dict[str, Design] = {}
    for name, spec in raw.items():
        knobs = cast("dict[str, Any]", spec.get("knobs", {}))
        bad = set(knobs) - ALLOWED_KNOBS
        if bad:
            raise SystemExit(
                f"design {name!r}: unknown knob(s) {sorted(bad)}; allowed: {sorted(ALLOWED_KNOBS)}"
            )
        designs[name] = Design(
            name=name,
            description=str(spec.get("description", "")),
            knobs=knobs,
            prompt_override=spec.get("prompt_override"),
        )
    return designs


def resolve_design(args: argparse.Namespace) -> Design:
    designs = load_designs()
    if args.design not in designs:
        raise SystemExit(f"unknown design {args.design!r}; known: {sorted(designs)}")
    d = designs[args.design]
    # Inline CLI overrides win over the registry entry.
    overrides: dict[str, Any] = {}
    if args.llm_model is not None:
        overrides["llm_model"] = args.llm_model
    if args.num_ctx_l1 is not None:
        overrides["num_ctx_l1"] = args.num_ctx_l1
    if args.num_ctx_fanout is not None:
        overrides["num_ctx_fanout"] = args.num_ctx_fanout
    if args.max_concurrency is not None:
        overrides["max_concurrency"] = args.max_concurrency
    if args.min_subdivide_s is not None:
        overrides["min_subdivide_duration_s"] = args.min_subdivide_s
    knobs = {**d.knobs, **overrides}
    prompt_override = args.prompt_override or d.prompt_override
    return Design(d.name, d.description, knobs, prompt_override)


# --------------------------------------------------------------------------- #
# Prompt-override monkey-patch
# --------------------------------------------------------------------------- #
def apply_prompt_override(rel_path: str) -> dict[str, Any]:
    """Import the override file and setattr matching constants on the agent.

    Returns the original values so the caller can restore them.
    """
    import importlib.util

    import autorag.agent as agent

    path = (SKILL_DIR / rel_path).resolve()
    if not path.is_file():
        raise SystemExit(f"prompt override not found: {path}")
    spec = importlib.util.spec_from_file_location("agent_lab_prompt_override", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import prompt override: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    originals: dict[str, Any] = {}
    applied: list[str] = []
    for name in dir(mod):
        if name not in AGENT_PROMPT_NAMES:
            continue
        if not hasattr(agent, name):
            log.warning("override defines %s but agent has no such constant", name)
            continue
        originals[name] = getattr(agent, name)
        setattr(agent, name, getattr(mod, name))
        applied.append(name)
    if not applied:
        log.warning("prompt override %s changed nothing", rel_path)
    else:
        log.info("prompt override %s applied: %s", rel_path, ", ".join(applied))
    return originals


def restore_prompts(originals: dict[str, Any]) -> None:
    import autorag.agent as agent

    for name, value in originals.items():
        setattr(agent, name, value)


# --------------------------------------------------------------------------- #
# Metric capture: LLM calls + tokens (LangChain callback)
# --------------------------------------------------------------------------- #
class MetricsCallback(BaseCallbackHandler):
    """Counts LLM calls and sums token usage across the whole runnable.

    Callbacks passed via ``config`` to ``RunnableLambda(_build_tree).invoke``
    propagate (via langchain's child-config contextvar) to every nested
    chain call inside ``_build_tree`` — including the one ``keep_alive=0``
    eviction call the agent fires in its ``finally``. We track that separately
    so the leaderboard's pipeline call count matches CLAUDE.md's cost formula.
    """

    def __init__(self) -> None:
        super().__init__()
        self._starts: set[uuid.UUID] = set()
        self.input_tokens = 0
        self.output_tokens = 0

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._starts.add(run_id)

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._starts.add(run_id)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: uuid.UUID,
        parent_run_id: uuid.UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        for gens in response.generations:
            for gen in gens:
                msg = getattr(gen, "message", None)
                if msg is None:
                    continue
                um = getattr(msg, "usage_metadata", None)
                if isinstance(um, dict):
                    self.input_tokens += int(um.get("input_tokens", 0) or 0)
                    self.output_tokens += int(um.get("output_tokens", 0) or 0)
                    continue
                rm = getattr(msg, "response_metadata", {}) or {}
                self.input_tokens += int(rm.get("prompt_eval_count", 0) or 0)
                self.output_tokens += int(rm.get("eval_count", 0) or 0)

    @property
    def calls_total(self) -> int:
        return len(self._starts)


class StageTimingHandler(logging.Handler):
    """Scrapes ``Stage <N> done in <s>s`` lines off the autorag.agent logger."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.stage_s: dict[str, float] = {}

    def emit(self, record: logging.LogRecord) -> None:
        m = _STAGE_RE.search(record.getMessage())
        if m:
            key = _STAGE_KEYS.get(m.group(1), m.group(1))
            self.stage_s[key] = float(m.group(2))


# --------------------------------------------------------------------------- #
# Metric capture: VRAM sampler
# --------------------------------------------------------------------------- #
class VRAMSampler(threading.Thread):
    """Polls nvidia-smi + ``ollama ps`` on a background thread for peak VRAM."""

    _SIZE_RE = re.compile(r"([\d.]+)\s*(GB|MB|GiB|MiB)", re.IGNORECASE)

    def __init__(self, interval: float = 0.4) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        # NB: not `_stop` — that name shadows threading.Thread._stop and
        # breaks Thread.join().
        self._stop_evt = threading.Event()
        self.baseline_mb: float | None = None
        self.peak_mb: float | None = None
        self.ollama_size_mb: float | None = None

    def _sample_gpu(self) -> float | None:
        out = _run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=5.0,
        )
        if not out:
            return None
        vals = [float(x) for x in out.splitlines() if x.strip().isdigit()]
        return max(vals) if vals else None

    def _sample_ollama(self) -> float | None:
        out = _run(["ollama", "ps"], timeout=5.0)
        if not out:
            return None
        best: float | None = None
        for line in out.splitlines()[1:]:
            m = self._SIZE_RE.search(line)
            if not m:
                continue
            val = float(m.group(1))
            unit = m.group(2).lower()
            mb = val * 1024 if unit in ("gb", "gib") else val
            best = mb if best is None else max(best, mb)
        return best

    def run(self) -> None:
        first = self._sample_gpu()
        self.baseline_mb = first
        self.peak_mb = first
        while not self._stop_evt.is_set():
            g = self._sample_gpu()
            if g is not None:
                self.peak_mb = g if self.peak_mb is None else max(self.peak_mb, g)
            o = self._sample_ollama()
            if o is not None:
                self.ollama_size_mb = (
                    o if self.ollama_size_mb is None else max(self.ollama_size_mb, o)
                )
            self._stop_evt.wait(self.interval)

    def stop(self) -> tuple[float | None, float | None, float | None]:
        self._stop_evt.set()
        self.join(timeout=5.0)
        delta = (
            None
            if self.peak_mb is None or self.baseline_mb is None
            else round(self.peak_mb - self.baseline_mb, 1)
        )
        return delta, self.peak_mb, self.ollama_size_mb


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def prepare_fixtures(stems: list[str]) -> None:
    from autorag import AutoRAG

    FIXTURES_DIR.mkdir(exist_ok=True)
    rag = AutoRAG()
    for stem in stems:
        dst = FIXTURES_DIR / f"{stem}.words.json"
        if dst.exists():
            log.info("fixture %s already prepared (%s)", stem, dst.name)
            continue
        src = REPO_ROOT / FIXTURE_SOURCES[stem]
        if not src.exists():
            raise SystemExit(f"fixture source missing: {src}")
        log.info("transcribing %s (one-time) ...", src.name)
        t0 = time.time()
        spans = rag.transcribe(src)
        dst.write_text(json.dumps(spans))
        log.info(
            "fixture %s: %d spans in %.0fs -> %s",
            stem,
            len(spans),
            time.time() - t0,
            dst.name,
        )


def load_spans(stem: str) -> list[dict[str, Any]]:
    p = FIXTURES_DIR / f"{stem}.words.json"
    if not p.exists():
        raise SystemExit(
            f"fixture {stem!r} not prepared. Run: bench.py --prepare-fixtures --fixtures {stem}"
        )
    return cast("list[dict[str, Any]]", json.loads(p.read_text()))


# --------------------------------------------------------------------------- #
# Tree introspection + judge
# --------------------------------------------------------------------------- #
def _tree_stats(tree: dict[str, Any]) -> tuple[int, int, int]:
    topics = tree.get("topics", [])
    if not topics:
        return 0, 0, 0
    l0 = topics[0]
    l1 = l0.get("children", []) or []
    l2 = sum(len(n.get("children", []) or []) for n in l1)
    depth = 1 + (1 if l1 else 0) + (1 if l2 else 0)
    return len(l1), l2, depth


def _compact_tree(tree: dict[str, Any]) -> Any:
    def node(n: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {
            "title": n.get("title", ""),
            "summary": n.get("summary", ""),
            "s": round(float(n.get("s", 0.0)), 1),
            "e": round(float(n.get("e", 0.0)), 1),
        }
        kids = n.get("children") or []
        if kids:
            out["children"] = [node(k) for k in kids]
        return out

    return [node(t) for t in tree.get("topics", [])]


def judge_tree(
    spans: list[dict[str, Any]],
    tree: dict[str, Any],
    *,
    model: str,
    num_ctx: int,
    char_budget: int,
) -> JudgeRubric:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_ollama import ChatOllama

    from autorag.blocks import format_blocks

    transcript = format_blocks(cast("Any", spans), 30)
    truncated = len(transcript) > char_budget
    if truncated:
        transcript = transcript[:char_budget] + "\n…[transcript truncated]"

    sys_prompt = (
        "You are a strict evaluator of an automatic topic-segmentation system. "
        "You are given a recording's transcript (time-bucketed blocks, "
        "`MM:SS-MM:SS Speaker K: words`) and the topic tree the system produced "
        "(L0 root → L1 topics → optional L2 subtopics, each with title, "
        "summary, and s/e seconds). Score 1-5 (5 = excellent) on:\n"
        "- boundary_coherence: do segment boundaries fall at real topic shifts?\n"
        "- coverage_completeness: do segments tile the whole recording with no "
        "large gaps/overlaps?\n"
        "- summary_faithfulness: are titles/summaries grounded in that span "
        "with no hallucination?\n"
        "- hierarchy_appropriateness: is the L1/L2 nesting warranted (not "
        "over/under-split)?\n"
        "- overall: holistic quality.\n"
        "rationale: one or two sentences citing the most decisive evidence. "
        "Be calibrated: reserve 5 for genuinely excellent output."
    )
    human = (
        "TRANSCRIPT:\n{transcript}\n\n"
        "PRODUCED TOPIC TREE (JSON):\n{tree}\n\n"
        "Return the structured scores."
    )
    chat = ChatOllama(
        model=model,
        temperature=0.0,
        keep_alive=0,
        num_ctx=num_ctx,
        base_url=_ollama_base_url(),
    ).with_structured_output(JudgeRubric, method="json_schema")
    prompt = ChatPromptTemplate.from_messages([("system", sys_prompt), ("human", human)])
    result = (prompt | chat).invoke(
        {"transcript": transcript, "tree": json.dumps(_compact_tree(tree))}
    )
    return cast("JudgeRubric", result)


# --------------------------------------------------------------------------- #
# One run
# --------------------------------------------------------------------------- #
def run_once(
    design: Design,
    stem: str,
    spans: list[dict[str, Any]],
    *,
    judge: bool,
    judge_model: str,
    judge_num_ctx: int,
    judge_char_budget: int,
) -> RunMetrics:
    import autorag.agent as agent

    agent_logger = logging.getLogger("autorag.agent")
    stage_handler = StageTimingHandler()
    prev_level = agent_logger.level
    agent_logger.setLevel(logging.INFO)
    agent_logger.addHandler(stage_handler)

    cb = MetricsCallback()
    sampler = VRAMSampler()
    sampler.start()
    time.sleep(0.5)  # let the sampler grab a pre-invoke baseline

    try:
        runnable = agent.build_topic_runnable(**design.knobs)
        t0 = time.time()
        tree = cast(
            "dict[str, Any]",
            runnable.invoke(cast("Any", spans), config=cast("Any", {"callbacks": [cb]})),
        )
        total_s = time.time() - t0
    finally:
        agent_logger.removeHandler(stage_handler)
        agent_logger.setLevel(prev_level)
        vram_delta, vram_peak, ollama_size = sampler.stop()

    n_l1, n_l2, depth = _tree_stats(tree)
    rubric: JudgeRubric | None = None
    if judge:
        log.info("judging %s with %s ...", stem, judge_model)
        rubric = judge_tree(
            spans,
            tree,
            model=judge_model,
            num_ctx=judge_num_ctx,
            char_budget=judge_char_budget,
        )

    return RunMetrics(
        fixture=stem,
        total_s=round(total_s, 1),
        stage_s=stage_handler.stage_s,
        # The agent always fires exactly one keep_alive=0 eviction call in its
        # finally; everything else is pipeline work.
        llm_calls_pipeline=max(cb.calls_total - 1, 0),
        llm_calls_total=cb.calls_total,
        input_tokens=cb.input_tokens,
        output_tokens=cb.output_tokens,
        vram_delta_mb=vram_delta,
        vram_peak_mb=vram_peak,
        ollama_size_mb=ollama_size,
        n_l1=n_l1,
        n_l2=n_l2,
        depth=depth,
        judge=rubric,
        tree=tree,
    )


# --------------------------------------------------------------------------- #
# Aggregation across --repeat / fixtures
# --------------------------------------------------------------------------- #
def _mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    m = statistics.fmean(values)
    sd = statistics.pstdev(values) if len(values) > 1 else 0.0
    return round(m, 1), round(sd, 1)


def judge_avg(r: JudgeRubric | None) -> float | None:
    if r is None:
        return None
    return round(
        statistics.fmean(
            [
                r.boundary_coherence,
                r.coverage_completeness,
                r.summary_faithfulness,
                r.hierarchy_appropriateness,
            ]
        ),
        2,
    )


# --------------------------------------------------------------------------- #
# Ledger writing
# --------------------------------------------------------------------------- #
def _designs_table() -> str:
    designs = load_designs()
    rows = [
        "| Design | Description | Knobs | Prompt override |",
        "| --- | --- | --- | --- |",
    ]
    for d in designs.values():
        knobs = ", ".join(f"{k}={v}" for k, v in d.knobs.items()) or "_(defaults)_"
        rows.append(f"| `{d.name}` | {d.description} | {knobs} | {d.prompt_override or '—'} |")
    return "\n".join(rows)


def _edit_ledger(leaderboard_row: str, run_entry: str) -> None:
    text = LEDGER_PATH.read_text()
    designs_block = f"<!-- DESIGNS:START -->\n{_designs_table()}\n<!-- DESIGNS:END -->"
    text = re.sub(
        r"<!-- DESIGNS:START -->.*?<!-- DESIGNS:END -->",
        lambda _: designs_block,
        text,
        flags=re.DOTALL,
    )
    text = text.replace(
        "<!-- LEADERBOARD:ROWS -->",
        f"{leaderboard_row}\n<!-- LEADERBOARD:ROWS -->",
    )
    text = text.replace(
        "<!-- RUNS:END -->",
        f"{run_entry}\n<!-- RUNS:END -->",
    )
    LEDGER_PATH.write_text(text)


def find_baseline(name: str) -> dict[str, Any] | None:
    if not RUNS_DIR.exists():
        return None
    cands = sorted(RUNS_DIR.glob("*.json"))
    for p in reversed(cands):
        rec = cast("dict[str, Any]", json.loads(p.read_text()))
        if rec.get("design") == name:
            return rec
    return None


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare-fixtures", action="store_true")
    p.add_argument("--design", default="baseline")
    p.add_argument(
        "--fixtures",
        default="",
        help="comma-separated fixture stems (default: all prepared, or all "
        "known with --prepare-fixtures)",
    )
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--judge-model", default="qwen2.5:32b-instruct-q4_K_M")
    p.add_argument("--judge-num-ctx", type=int, default=16384)
    p.add_argument("--judge-char-budget", type=int, default=48000)
    p.add_argument("--baseline", default="baseline", help="design to diff against")
    p.add_argument("--dry-run", action="store_true")
    # inline knob overrides
    p.add_argument("--llm-model")
    p.add_argument("--num-ctx-l1", type=int)
    p.add_argument("--num-ctx-fanout", type=int)
    p.add_argument("--max-concurrency", type=int)
    p.add_argument("--min-subdivide-s", type=float)
    p.add_argument("--prompt-override")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)

    requested = [s for s in args.fixtures.split(",") if s]
    for s in requested:
        if s not in FIXTURE_SOURCES:
            raise SystemExit(f"unknown fixture {s!r}; known: {sorted(FIXTURE_SOURCES)}")

    if args.prepare_fixtures:
        # Setup is a standalone step: prepare and stop. Run the benchmark in a
        # separate invocation once fixtures exist.
        prepare_fixtures(requested or list(FIXTURE_SOURCES))
        return 0

    fixtures = requested or sorted(
        p.stem.removesuffix(".words") for p in FIXTURES_DIR.glob("*.words.json")
    )
    if not fixtures:
        raise SystemExit("no prepared fixtures. Run with --prepare-fixtures first.")

    design = resolve_design(args)
    log.info(
        "design %s | knobs=%s | prompt_override=%s | fixtures=%s | repeat=%d",
        design.name,
        design.knobs or "(defaults)",
        design.prompt_override or "none",
        ",".join(fixtures),
        args.repeat,
    )

    originals: dict[str, Any] = {}
    if design.prompt_override:
        originals = apply_prompt_override(design.prompt_override)

    runs: list[RunMetrics] = []
    try:
        for stem in fixtures:
            spans = load_spans(stem)
            for i in range(args.repeat):
                # Judge only the final repeat of each fixture (judging is
                # expensive and mechanical metrics are what --repeat averages).
                do_judge = (not args.no_judge) and (i == args.repeat - 1)
                log.info(
                    "run %s rep %d/%d%s",
                    stem,
                    i + 1,
                    args.repeat,
                    " (+judge)" if do_judge else "",
                )
                runs.append(
                    run_once(
                        design,
                        stem,
                        spans,
                        judge=do_judge,
                        judge_model=args.judge_model,
                        judge_num_ctx=args.judge_num_ctx,
                        judge_char_budget=args.judge_char_budget,
                    )
                )
    finally:
        if originals:
            restore_prompts(originals)

    run_id = f"{datetime.now(UTC):%Y%m%d-%H%M}-{uuid.uuid4().hex[:4]}"
    date = f"{datetime.now(UTC):%Y-%m-%d %H:%M}Z"

    total_mean, total_sd = _mean_sd([r.total_s for r in runs])
    calls = max((r.llm_calls_pipeline for r in runs), default=0)
    in_tok = round(statistics.fmean([r.input_tokens for r in runs])) if runs else 0
    out_tok = round(statistics.fmean([r.output_tokens for r in runs])) if runs else 0
    vram_deltas = [r.vram_delta_mb for r in runs if r.vram_delta_mb is not None]
    vram_delta = round(max(vram_deltas), 1) if vram_deltas else None
    judged = [r.judge for r in runs if r.judge is not None]
    j_avgs = [v for v in (judge_avg(j) for j in judged) if v is not None]
    j_avg = round(statistics.fmean(j_avgs), 2) if j_avgs else None

    # per-stage mean across all runs
    stage_keys = ["l1", "decide", "l2", "summarize", "l0"]
    stage_means = {
        k: round(
            statistics.fmean([r.stage_s.get(k, 0.0) for r in runs]) if runs else 0.0,
            1,
        )
        for k in stage_keys
    }
    stage_str = "/".join(f"{stage_means[k]:.0f}" for k in stage_keys)
    knob_str = ", ".join(f"{k}={v}" for k, v in design.knobs.items()) or "defaults"

    base_rec = find_baseline(args.baseline)
    verdict_bits: list[str] = []
    if base_rec and args.baseline != design.name:
        b_total = base_rec.get("total_s_mean")
        b_judge = base_rec.get("judge_avg")
        if isinstance(b_total, (int, float)) and b_total:
            verdict_bits.append(f"Δtotal {total_mean - b_total:+.1f}s")
        if isinstance(b_judge, (int, float)) and j_avg is not None:
            verdict_bits.append(f"Δjudge {j_avg - b_judge:+.2f}")
    verdict = "; ".join(verdict_bits) if verdict_bits else "—"

    row = (
        f"| `{run_id}` | {date} | `{design.name}` | {','.join(fixtures)} | "
        f"{knob_str} | {total_mean}±{total_sd} | {stage_str} | {calls} | "
        f"{in_tok} | {out_tok} | {vram_delta if vram_delta is not None else 'n/a'} | "
        f"{j_avg if j_avg is not None else 'n/a'} | {verdict} |"
    )

    judge_lines = ""
    for r in runs:
        if r.judge is None:
            continue
        jb = r.judge
        judge_lines += (
            f"  - **{r.fixture}** — bound={jb.boundary_coherence} "
            f"cover={jb.coverage_completeness} faith={jb.summary_faithfulness} "
            f"hier={jb.hierarchy_appropriateness} overall={jb.overall} "
            f"(avg {judge_avg(jb)}): {jb.rationale}\n"
        )

    env = {
        "git_sha": _git_sha(),
        "ollama_version": _ollama_version(),
        "gpu": _gpu_name(),
        "ollama_num_parallel": os.environ.get("OLLAMA_NUM_PARALLEL", "unset"),
        "python": sys.version.split()[0],
    }
    entry = (
        f"### `{run_id}` — `{design.name}`  ({date})\n\n"
        f"- **Design:** {design.description or '—'}\n"
        f"- **Knobs:** `{json.dumps(design.knobs)}`\n"
        f"- **Prompt override:** {design.prompt_override or 'none'}\n"
        f"- **Fixtures:** {', '.join(fixtures)} · **repeat:** {args.repeat}\n"
        f"- **Total:** {total_mean}±{total_sd}s · "
        f"**stages (l1/decide/l2/sum/l0):** {stage_str}s\n"
        f"- **LLM calls (pipeline):** {calls} · "
        f"**tokens in/out:** {in_tok}/{out_tok}\n"
        f"- **VRAM Δ:** "
        f"{f'{vram_delta} MB' if vram_delta is not None else 'n/a'} · "
        f"**ollama size:** "
        f"{f'{round(runs[0].ollama_size_mb)} MB' if runs and runs[0].ollama_size_mb else 'n/a'}\n"
        f"- **Tree:** L1={runs[0].n_l1 if runs else 0} "
        f"L2={runs[0].n_l2 if runs else 0} depth={runs[0].depth if runs else 0}\n"
        f"- **Judge ({args.judge_model}), avg {j_avg if j_avg is not None else 'n/a'}:**\n"
        f"{judge_lines or '  - (judging skipped)\n'}"
        f"- **Env:** `{json.dumps(env)}`\n"
        f"- **Verdict vs `{args.baseline}`:** {verdict}\n"
        f"- **Conclusion:** _(fill in: keep / discard / promote to a named design)_\n"
    )

    if args.dry_run:
        log.info("DRY RUN — not writing ledger or artifact")
        print("\nLEADERBOARD ROW:\n" + row)
        print("\nRUN ENTRY:\n" + entry)
        return 0

    RUNS_DIR.mkdir(exist_ok=True)
    artifact = {
        "run_id": run_id,
        "date": date,
        "design": design.name,
        "knobs": design.knobs,
        "prompt_override": design.prompt_override,
        "fixtures": fixtures,
        "repeat": args.repeat,
        "total_s_mean": total_mean,
        "total_s_sd": total_sd,
        "stage_s_mean": stage_means,
        "llm_calls_pipeline": calls,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "vram_delta_mb": vram_delta,
        "judge_avg": j_avg,
        "judge_model": args.judge_model,
        "env": env,
        "runs": [
            {
                "fixture": r.fixture,
                "total_s": r.total_s,
                "stage_s": r.stage_s,
                "llm_calls_pipeline": r.llm_calls_pipeline,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "vram_delta_mb": r.vram_delta_mb,
                "vram_peak_mb": r.vram_peak_mb,
                "ollama_size_mb": r.ollama_size_mb,
                "n_l1": r.n_l1,
                "n_l2": r.n_l2,
                "depth": r.depth,
                "judge": r.judge.model_dump() if r.judge else None,
                "tree": r.tree,
            }
            for r in runs
        ],
    }
    (RUNS_DIR / f"{run_id}.json").write_text(json.dumps(artifact, indent=2))
    _edit_ledger(row, entry)
    log.info("ledger updated; artifact -> runs/%s.json", run_id)
    print("\n" + row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
