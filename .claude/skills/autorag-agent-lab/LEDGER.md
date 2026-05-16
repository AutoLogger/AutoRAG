# AutoRAG Agent Lab — Ledger

Append-only record of every topic-agent design that has been benchmarked, and
how it scored. **This file is committed** — it is the cumulative memory of what
has been tried. `bench.py` maintains the three managed regions below; do not
hand-edit inside the `<!-- … -->` markers (free-text conclusions go in the
`Conclusion:` line of a run entry, which bench.py leaves alone on rewrite).

- A **design** is a named bundle of knobs + an optional prompt override
  (`designs.json`). The `## Designs` table is regenerated from `designs.json`
  on every run, so it always reflects the current registry.
- A **run** is one design executed over a fixed fixture set. Mechanical
  metrics are objective; the judge score is **comparative, not absolute** —
  only compare judge scores produced by the *same* judge model.
- A win counts only when the delta vs. the baseline exceeds run-to-run spread
  (use `--repeat` to measure spread).

## Designs

<!-- DESIGNS:START -->
_No runs yet — this table is generated from `designs.json` on the first `bench.py` run._
<!-- DESIGNS:END -->

## Leaderboard

<!-- LEADERBOARD:START -->
| Run | Date | Design | Fixtures | Knobs | Total s | Stages (l1/3a/3b/sum/l0) s | LLM calls | In tok | Out tok | VRAM Δ MB | Judge avg | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
<!-- LEADERBOARD:ROWS -->
<!-- LEADERBOARD:END -->

## Runs

<!-- RUNS:START -->
<!-- RUNS:END -->
