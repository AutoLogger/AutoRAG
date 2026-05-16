---
name: autorag-agent-lab
description: Build, optimize, and compare the AutoRAG audio→topics langchain agent (the 5-stage L0/L1/L2 pipeline in src/autorag/agent.py), and keep a committed ledger of every design tried and how it scored. Runs real variants on cached fixtures via the bundled bench.py, captures mechanical metrics (per-stage wall time, LLM call count, tokens, VRAM) plus an LLM-judge quality score, and appends a reproducible entry to LEDGER.md. Use when the user asks to "optimize the topic agent", "compare LLM models for topic extraction", "tune num_ctx / NUM_PARALLEL / min_subdivide", "benchmark agent.py", "A/B the pipeline", "which model is best for the agent", "try a new prompt for the agent", or "record an agent experiment".
---

# AutoRAG Agent Lab

This skill makes agent tuning **measurable and cumulative**. Instead of eyeballing
one topic tree and guessing, you define a *design*, run it on a fixed fixture set,
capture objective + judged metrics, and append a row to a committed ledger so the
next session starts from what's already known. Structured like
`.claude/skills/autorag-puppeteer/SKILL.md`: numbered sections, concrete recipes,
verification, then quick-reference tables.

The agent under study is the single audio→topics pipeline in
`src/autorag/agent.py`. **`agent.py` is never edited by this skill** — knobs are
passed to `build_topic_runnable`; prompt variants are applied by monkey-patching
its module constants in the bench process.

## 0. Preamble — before the first run

- **Bundled runner.** Everything is driven by `bench.py` in this directory. Run it
  from the repo root in the project venv:
  ```
  uv run python .claude/skills/autorag-agent-lab/bench.py <args>
  ```
- **Prerequisites.**
  - `uv sync --all-extras` (the agent needs `[audio,diarize,rag]`; fixture prep
    needs Whisper).
  - Ollama running and reachable (`AUTORAG_OLLAMA_BASE_URL`, default
    `http://localhost:11434`), with **both** the design's `llm_model` *and* the
    `--judge-model` pulled (`ollama pull <model>`).
  - A CUDA GPU for meaningful VRAM/latency numbers. Without `nvidia-smi`, VRAM
    is recorded `n/a` (everything else still works); on CPU/Metal, timings are
    real but not comparable to GPU runs — note that in the conclusion.
- **What's committed vs. not.** `SKILL.md`, `bench.py`, `designs.json`,
  `LEDGER.md`, `prompts/` are committed — the ledger is the deliverable.
  `fixtures/` (cached word spans) and `runs/` (per-run JSON artifacts) are
  gitignored and regenerable.
- **Internal tooling.** This lives under `.claude/` — it is **not shipped in the
  wheel** and needs no `docs/` or `CHANGELOG` update (the CLAUDE.md "keep docs in
  sync" rule is about the SDK surface, not this skill).

## 1. Mental model — the agent and its knobs

Five stages (`src/autorag/agent.py`), Whisper excluded because the bench reuses
cached spans:

| # | Stage | Calls |
| - | ----- | ----- |
| 2 | L1 boundaries (one call) | 1 |
| 3a | Decide subdivide (per long L1, batched) | `N1_long` |
| 3b | L2 boundaries (per yes-L1, batched) | `N1_yes` |
| 4 | Summarize every L1 + L2 leaf (batched) | `K` |
| 5 | L0 aggregate (one call) | 1 |

Pipeline LLM calls ≈ `2 + N1_long + N1_yes + N1 + N2_total` (~20 for a 7-min
clip). The agent also fires **one** `keep_alive=0` eviction call in its
`finally`; the bench counts that separately so the leaderboard's
`LLM calls` column is the pipeline figure (`total − 1`).

Tunable knobs (kwargs of `build_topic_runnable`; a design sets any subset, rest
fall back to these defaults):

| Knob | Default | Trades |
| ---- | ------- | ------ |
| `llm_model` | `qwen2.5:14b-instruct-q8_0` | quality ↔ latency ↔ VRAM |
| `num_ctx_l1` | `8192` | L1 fidelity on >1 h audio ↔ one Stage 2→3a reload if ≠ fanout |
| `num_ctx_fanout` | `8192` | KV cache size; **must equal `num_ctx_l1` to keep the model warm** (Ollama reloads on any `num_ctx` change) |
| `max_concurrency` | `4` | batch-stage parallelism (needs `OLLAMA_NUM_PARALLEL≥4`) ↔ VRAM |
| `min_subdivide_duration_s` | `120.0` | L2 coverage ↔ call count (raise → fewer 3a/3b/4 calls) |
| `ollama_base_url` | env / localhost | target server |

Prompt constants are also "knobs" via a prompt override (section 4):
`_L1_SYS/_L1_HUMAN`, `_DECIDE_*`, `_L2_*`, `_NODE_SUM_*`, `_AGG_*`,
`_BOUNDARY_BLOCK_SECONDS`.

## 2. The experiment loop

Always follow this loop. It is what keeps the ledger trustworthy.

1. **State a hypothesis.** "qwen 7b is within 0.3 judge avg of 14b at half the
   VRAM." One sentence, falsifiable.
2. **Change one variable.** Add or edit exactly one design in `designs.json`
   (or pass one inline knob flag). Never change two knobs in one run — you
   won't know which moved the metric.
3. **Fix the fixture set.** Compare designs on the *same* `--fixtures`. Reuse
   cached spans so Whisper variance is out of the picture.
4. **Run** (section 4), with `--baseline` pointing at the design you're trying
   to beat.
5. **Judge** with a model held constant across the comparison (section 6).
6. **Read the verdict**, then **append a one-line conclusion** to the run entry
   in `LEDGER.md` (keep / discard / promote).
7. A win counts only if Δ exceeds run-to-run spread — use `--repeat` (section 5)
   when the delta is small.

## 3. One-time setup — prepare fixtures

Transcribe the repo's test audio once into cached word-span JSON (skips Whisper
on every subsequent run):

```
uv run python .claude/skills/autorag-agent-lab/bench.py --prepare-fixtures
```

Known stems: `fox-new` (smallest — use for smoke tests), `quin-rs-tut`,
`3b1b-llm`, `3b1b-llm2` (largest). Prepare a subset with
`--prepare-fixtures --fixtures fox-new,3b1b-llm`. Already-prepared fixtures are
skipped. `--prepare-fixtures` only prepares and exits; benchmark in a separate
invocation.

## 4. Running an experiment

```
# A registered design across two fixtures, judged, vs. the baseline:
uv run python .claude/skills/autorag-agent-lab/bench.py \
  --design qwen2.5-7b --fixtures fox-new,quin-rs-tut --baseline baseline

# Quick inline variant, no judge, dry run (prints the row, writes nothing):
uv run python .claude/skills/autorag-agent-lab/bench.py \
  --design baseline --num-ctx-l1 16384 --fixtures fox-new --no-judge --dry-run
```

`designs.json` schema — `name → { description, knobs, prompt_override }`:

```json
{
  "qwen2.5-7b": {
    "description": "Smaller model — expect lower VRAM + faster, watch quality.",
    "knobs": { "llm_model": "qwen2.5:7b-instruct-q8_0" },
    "prompt_override": null
  }
}
```

- `knobs` keys must be in the allowed set (section 1); an unknown key fails
  fast with a clear error.
- `prompt_override` is a path **relative to this skill dir**, e.g.
  `"prompts/terse_l1.py"`. Inline `--prompt-override` wins over the registry.
- Add new designs by editing `designs.json`. The `## Designs` table in
  `LEDGER.md` is regenerated from it on every run, so it never drifts.

**Prompt-override mechanism (no source edits).** A prompt-override file is a
Python module that redefines a subset of the agent's prompt constants. Before
`build_topic_runnable` is called, the bench imports it and `setattr`s each
matching name onto `autorag.agent`, restoring originals afterward. See
`prompts/terse_l1.py` for a worked example and the rules (keep `{template}`
vars; double literal braces `{{ }}`; the name must already exist on the agent).

## 5. Mechanical metrics — how each is captured

| Metric | Source | Caveat |
| ------ | ------ | ------ |
| Total wall (s) | `time.time()` around `runnable.invoke` | excludes Whisper (cached) |
| Per-stage wall (l1/decide/l2/sum/l0) | a `logging.Handler` scraping the agent's `Stage N done in …s` lines | sums ≈ total minus orchestration |
| LLM calls (pipeline) | LangChain `BaseCallbackHandler` counting start events, minus the 1 eviction call | callbacks propagate into the runnable via langchain's child-config contextvar |
| Tokens in/out | `usage_metadata` on each `AIMessage` (fallback `response_metadata` eval counts) | 0 if the model/provider omits usage |
| VRAM Δ / peak (MB) | background thread polling `nvidia-smi`; Δ = peak − pre-invoke baseline | `n/a` without `nvidia-smi`; baseline includes other GPU users |
| Ollama model size (MB) | `ollama ps` SIZE column, sampled | reported model+KV footprint, a coarse proxy |
| Tree shape | `#L1`, `#L2`, depth off the returned `TopicTree` | |

**Determinism caveat.** `temperature=0`, but Ollama is not bit-deterministic
across model loads. For small deltas use `--repeat K` (default 1): mechanical
metrics are averaged and the leaderboard shows `mean±sd`. Treat sd as the noise
floor — a "win" must clear it. Judging runs only on the final repeat of each
fixture (it's expensive and not what `--repeat` is measuring).

## 6. LLM-judge quality

After producing the tree, the bench feeds the block-formatted transcript
(`autorag.blocks.format_blocks`, the same view the agent's boundary stages see)
plus the produced tree to a judge model with structured output. Rubric, each
1–5: `boundary_coherence`, `coverage_completeness`, `summary_faithfulness`,
`hierarchy_appropriateness`, `overall`, plus a one-line `rationale`. The
leaderboard `Judge avg` is the mean of the four sub-dimensions.

- **Pick the judge deliberately.** Default `--judge-model
  qwen2.5:32b-instruct-q4_K_M` — a *different/larger* model than the typical
  subject reduces self-enhancement bias. Never judge a model with itself when
  comparing across models.
- **Scores are comparative, not absolute.** Only compare judge numbers produced
  by the *same* judge model. If you change `--judge-model`, you've started a new
  scale — note it and don't cross-compare old rows.
- Long transcripts are truncated to `--judge-char-budget` (default 48 000
  chars ≈ the 16 k `--judge-num-ctx`); the rationale will say if truncation
  likely hurt coverage scoring.
- Skip judging with `--no-judge` for fast mechanical-only sweeps (e.g. VRAM/
  latency knob scans where quality is expected unchanged).

## 7. The ledger

`LEDGER.md` has three bench-maintained regions (don't hand-edit inside the
`<!-- … -->` markers; the free-text `Conclusion:` line is yours and is
preserved on rewrite):

- **`## Designs`** — regenerated from `designs.json` every run; the registry of
  what each named design *is*.
- **`## Leaderboard`** — one row per run: id, date, design, fixtures, knobs,
  total±sd, per-stage, LLM calls, tokens, VRAM Δ, judge avg, verdict-vs-baseline.
- **`## Runs`** — append-only detail: full knobs JSON, prompt override, tree
  shape, per-fixture judge sub-scores + rationale, env snapshot
  (git SHA, ollama version, GPU, `OLLAMA_NUM_PARALLEL`, python), and your
  conclusion.

**Reproducing a past design:** read its `## Runs` entry — the knobs JSON +
prompt-override path + env fully specify it. Recreate the `designs.json` entry
(or pass the inline flags) and re-run on the same fixtures. Full per-node output
is in `runs/<id>.json` (gitignored) when it still exists locally.

## 8. Comparing designs

1. Open `LEDGER.md`, pick the current best row for your objective as the
   baseline (`--baseline <design>`; the bench diffs against that design's most
   recent `runs/*.json`).
2. Run the candidate on the **identical** `--fixtures`.
3. The `Verdict` column shows `Δtotal` and `Δjudge`. Decide on the explicit
   trade-off the hypothesis named (e.g. "−40% VRAM is worth −0.2 judge avg").
4. Only promote a win that clears run-to-run spread (`--repeat`). Record the
   decision in the run entry's `Conclusion:` line; if it's a keeper, add it as
   a named design in `designs.json` so it's reusable.

There is no single score — latency, VRAM, and quality are a Pareto surface.
State which corner the hypothesis targeted and judge against that.

## 9. Optimization playbook

Levers from `CLAUDE.md` "Ollama tuning", mapped to the metric each moves:

- **`num_ctx_l1 == num_ctx_fanout`** → keeps the model warm (zero mid-run
  reloads). Diverging them buys L1 fidelity on ≈1 h+ audio at the cost of one
  Stage 2→3a reload (visible as a jump in `l1`/`decide` stage time).
- **`OLLAMA_NUM_PARALLEL` (server env, not a knob)** → batch-stage latency.
  `≥4` lets Stage 3a/3b/4 actually parallelize (`max_concurrency=4`); `=1`
  serializes them but frees VRAM for a bigger model. Capture it in the env
  snapshot every run — it explains stage-time deltas the knobs don't.
- **`min_subdivide_duration_s`** → call count + L2 coverage. Raising it cuts
  3a/3b/4 calls (cheaper, faster) but flattens the hierarchy; the
  `no-subdivide` design (`1e9`) is the ablation that isolates the L2 layer's
  cost/quality contribution.
- **`llm_model`** → the big quality/VRAM/latency lever. Smaller q-levels and
  param counts drop VRAM and latency; judge avg tells you what it cost.
- **Prompt variants** → token cost and boundary/summary quality. The agent
  deliberately splits boundary detection from summarization (combined prompts
  bled content across boundaries); keep that separation when authoring
  overrides — test terser *boundary* prompts and faithfulness-tightened
  *summary* prompts independently.

Typical first experiments: `baseline` vs `qwen2.5-7b` (quality floor for the
cheap model); `baseline` vs `--num-ctx-l1 16384` on `3b1b-llm2` (does the long
clip's L1 improve?); `baseline` vs `no-subdivide` (is L2 earning its calls?).

## 10. Verification (smoke test of the skill itself)

```
# 1. fixture prep on the smallest clip
uv run python .claude/skills/autorag-agent-lab/bench.py \
  --prepare-fixtures --fixtures fox-new
ls .claude/skills/autorag-agent-lab/fixtures/fox-new.words.json

# 2. mechanical-only baseline run
uv run python .claude/skills/autorag-agent-lab/bench.py \
  --design baseline --fixtures fox-new --no-judge

# 3. with judge (judge model must differ from the subject)
uv run python .claude/skills/autorag-agent-lab/bench.py \
  --design baseline --fixtures fox-new --judge-model qwen2.5:32b-instruct-q4_K_M
```

Expect after step 2: a new `## Leaderboard` row, a `runs/<id>.json` artifact,
populated per-stage times and LLM call count (~20 for a multi-minute clip),
tokens > 0, VRAM Δ a positive MB figure (or `n/a` with a logged reason).
After step 3: judge sub-scores + rationale under `## Runs`. Static checks:
`uv run ruff check .claude/skills/autorag-agent-lab/bench.py` and
`uv run mypy .claude/skills/autorag-agent-lab/bench.py` are clean. Use
`--dry-run` to inspect the row/entry without touching the ledger.

## 11. Quick reference

**Knobs** — `llm_model`, `num_ctx_l1`, `num_ctx_fanout`, `max_concurrency`,
`min_subdivide_duration_s`, `ollama_base_url`. Defaults in section 1.

**bench.py flags**

| Flag | Meaning |
| ---- | ------- |
| `--prepare-fixtures` | transcribe fixtures once, then exit |
| `--design NAME` | design from `designs.json` (default `baseline`) |
| `--fixtures a,b` | fixture stems (default: all prepared) |
| `--repeat K` | repeat each fixture K times; report mean±sd |
| `--no-judge` | skip the LLM judge |
| `--judge-model M` | judge model (default `qwen2.5:32b-instruct-q4_K_M`) |
| `--judge-num-ctx` / `--judge-char-budget` | judge context / transcript cap |
| `--baseline NAME` | design to diff the verdict against |
| `--llm-model` / `--num-ctx-l1` / `--num-ctx-fanout` / `--max-concurrency` / `--min-subdivide-s` | inline knob overrides |
| `--prompt-override path` | prompt-override file (relative to skill dir) |
| `--dry-run` | print row + entry, write nothing |

**Metrics** — total s (mean±sd), per-stage s (l1/decide/l2/sum/l0), LLM calls
(pipeline), tokens in/out, VRAM Δ MB, ollama size MB, tree shape, judge avg
(+ per-dim). Mechanical = objective; judge = comparative within one judge model.

**Files** — `designs.json` (registry, committed) · `LEDGER.md` (results,
committed) · `prompts/` (overrides, committed) · `fixtures/`, `runs/`
(regenerable, gitignored).
