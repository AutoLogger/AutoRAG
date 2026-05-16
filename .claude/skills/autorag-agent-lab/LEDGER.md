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

> **Provenance break — 2026-05-15 default flip.** The shipped default was
> refactored from `qwen2.5:14b-instruct-q8_0` to `gemma4:latest` with thinking
> disabled (`reasoning=False`), and `baseline` was redefined accordingly. The
> default judge also changed: `qwen2.5:32b-instruct-q4_K_M` →
> `batiai/qwen3.6-27b:q3` → `gemma4:26b`. The Leaderboard/Runs rows below are
> **preserved as immutable history but are NOT comparable to post-flip rows**:
> (a) `baseline` no longer means the qwen 14B config; (b) the old `gemma4` row
> ran with thinking **on** (Ollama default at the time) — the new
> `gemma4-thinking` design reproduces that exact condition as the
> apples-to-apples bridge to the new `reasoning=False` baseline; (c) those rows
> were judged by `batiai/qwen3.6-27b:q3`, a different scale than the new
> `gemma4:26b` judge. Re-run `baseline` (and a fresh comparison set) before
> drawing any post-flip conclusion; do not diff across this line.

## Designs

<!-- DESIGNS:START -->
| Design | Description | Knobs | Prompt override |
| --- | --- | --- | --- |
| `baseline` | Current shipped defaults (src/autorag/agent.py build_topic_runnable): gemma4:latest with thinking disabled. | llm_model=gemma4:latest, num_ctx_l1=8192, num_ctx_fanout=8192, max_concurrency=4, min_subdivide_duration_s=120.0, reasoning=False | — |
| `gemma4-thinking` | Ablation: gemma4:latest with thinking ON (reasoning=True). Quantifies the quality<->latency trade-off of the shipped reasoning=False default; the LEDGER's pre-flip gemma4 rows all ran with thinking on (Ollama default), so this is the apples-to-apples bridge to the new baseline. | reasoning=True | — |
| `gemma4-26b` | Bigger sibling gemma4:26b (25.8B Q4_K_M, ~17 GB) — quality-ceiling reference. Operator note: run `ollama serve` with OLLAMA_NUM_PARALLEL=1 before this design (a 17 GB model + 4 reserved slot KVs will not fit 24 GB; see CLAUDE.md "Ollama tuning"). max_concurrency=1 keeps the client from batching into slots the server no longer multiplexes. | llm_model=gemma4:26b, max_concurrency=1 | — |
| `granite4.1-8b` | Cross-model comparison: granite4.1:8b (8.8B Q4_K_M). The LEDGER's strongest non-qwen alternative under the old baseline (Pareto win there); kept as a standing comparison candidate against the gemma4 default. | llm_model=granite4.1:8b | — |
| `no-subdivide` | Ablation: never build L2 (min_subdivide_duration_s = inf). Isolates the cost/quality of the L2 layer. | min_subdivide_duration_s=1000000000.0 | — |
| `terse-l1` | Example prompt variant: a tighter L1 boundary system prompt (see prompts/terse_l1.py). | _(defaults)_ | prompts/terse_l1.py |
<!-- DESIGNS:END -->

## Leaderboard

<!-- LEADERBOARD:START -->
| Run | Date | Design | Fixtures | Knobs | Total s | Stages (l1/3a/3b/sum/l0) s | LLM calls | In tok | Out tok | VRAM Δ MB | Judge avg | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `20260516-0331-ae14` | 2026-05-16 03:31Z | `baseline` | fox-new,quin-rs-tut | llm_model=qwen2.5:14b-instruct-q8_0, num_ctx_l1=8192, num_ctx_fanout=8192, max_concurrency=4, min_subdivide_duration_s=120.0 | 50.5±15.6 | 12/2/2/31/4 | 17 | 8526 | 868 | 16420.0 | 3.88 | — |
| `20260516-0347-b3e9` | 2026-05-16 03:47Z | `gemma4` | fox-new,quin-rs-tut | llm_model=gemma4:latest | 91.9±37.1 | 58/2/4/24/4 | 13 | 7515 | 818 | 10969.0 | 5.0 | Δtotal +41.4s; Δjudge +1.12 |
| `20260516-0355-5ce7` | 2026-05-16 03:55Z | `granite4.1-8b` | fox-new,quin-rs-tut | llm_model=granite4.1:8b | 37.7±14.1 | 19/1/1/13/3 | 11 | 6634 | 886 | 10662.0 | 4.62 | Δtotal -12.8s; Δjudge +0.74 |
| `20260516-0605-aaf5` | 2026-05-16 06:05Z | `baseline` | fox-new,quin-rs-tut | llm_model=gemma4:latest, num_ctx_l1=8192, num_ctx_fanout=8192, max_concurrency=4, min_subdivide_duration_s=120.0, reasoning=False | 25.1±11.9 | 13/1/2/7/1 | 16 | 9645 | 900 | 186.0 | n/a | — |
| `20260516-0608-1083` | 2026-05-16 06:08Z | `baseline` | fox-new,quin-rs-tut | llm_model=gemma4:latest, num_ctx_l1=8192, num_ctx_fanout=8192, max_concurrency=4, min_subdivide_duration_s=120.0, reasoning=False | 16.8±2.2 | 5/1/2/7/1 | 16 | 9640 | 893 | 65.0 | n/a | — |
| `20260516-0631-7411` | 2026-05-16 06:31Z | `baseline` | fox-new,quin-rs-tut | llm_model=gemma4:latest, num_ctx_l1=8192, num_ctx_fanout=8192, max_concurrency=4, min_subdivide_duration_s=120.0, reasoning=False | 16.6±1.2 | 6/1/1/7/1 | 13 | 8470 | 833 | 12.0 | n/a | — |
| `20260516-0633-58a7` | 2026-05-16 06:33Z | `baseline` | fox-new,quin-rs-tut | llm_model=gemma4:latest, num_ctx_l1=16384, num_ctx_fanout=8192, max_concurrency=4, min_subdivide_duration_s=120.0, reasoning=False | 22.1±2.2 | 6/6/2/8/1 | 16 | 9645 | 902 | 226.0 | n/a | — |
| `20260516-0636-f838` | 2026-05-16 06:36Z | `baseline` | fox-new,quin-rs-tut | llm_model=gemma4:latest, num_ctx_l1=16384, num_ctx_fanout=8192, max_concurrency=4, min_subdivide_duration_s=120.0, reasoning=False | 21.7±2.2 | 6/5/2/8/1 | 16 | 9645 | 901 | 124.0 | n/a | — |
<!-- LEADERBOARD:ROWS -->
<!-- LEADERBOARD:END -->

## Runs

<!-- RUNS:START -->
### `20260516-0331-ae14` — `baseline`  (2026-05-16 03:31Z)

- **Design:** Current shipped defaults (src/autorag/agent.py build_topic_runnable).
- **Knobs:** `{"llm_model": "qwen2.5:14b-instruct-q8_0", "num_ctx_l1": 8192, "num_ctx_fanout": 8192, "max_concurrency": 4, "min_subdivide_duration_s": 120.0}`
- **Prompt override:** none
- **Fixtures:** fox-new, quin-rs-tut · **repeat:** 1
- **Total:** 50.5±15.6s · **stages (l1/decide/l2/sum/l0):** 12/2/2/31/4s
- **LLM calls (pipeline):** 17 · **tokens in/out:** 8526/868
- **VRAM Δ:** 16420.0 MB · **ollama size:** 17408 MB
- **Tree:** L1=7 L2=0 depth=2
- **Judge (batiai/qwen3.6-27b:q3), avg 3.88:**
  - **fox-new** — bound=4 cover=5 faith=3 hier=4 overall=4 (avg 4.0): The system accurately tiles the full recording and captures major thematic shifts, but repeatedly misattributes speakers in the summaries and artificially splits a continuous monologue at 06:29, reducing faithfulness and boundary precision.
  - **quin-rs-tut** — bound=5 cover=4 faith=3 hier=3 overall=3 (avg 3.75): Segment boundaries accurately align with natural topic shifts in the transcript, but summary faithfulness is undermined by clear hallucinations (e.g., 'Carrils' and 'iris-tail dragons' instead of 'Iron and Steel' and 'Roomington'), and the JSON structure exhibits overlapping sibling segments that violate proper tree hierarchy.
- **Env:** `{"git_sha": "b12bbd6", "ollama_version": "ollama version is 0.24.0", "gpu": "NVIDIA GeForce RTX 3090", "ollama_num_parallel": "unset", "python": "3.12.9"}`
- **Verdict vs `baseline`:** —
- **Conclusion:** KEEP as the ledger anchor. **Judge provenance (load-bearing for every future row):** the skill-default judges do not fit a 24 GB RTX 3090 — `qwen2.5:32b-q4_K_M` and Ollama's `qwen3.6:27b` (GGUF q4_K_M, 17.4 GB) both spill ~12% (2.5 GiB weights) to CPU because the in-container Ollama leaves only ~22.2 GiB usable (~1.9 GiB held by the devcontainer display stack) and Ollama offloads at whole-layer granularity; FA/q8_0-KV/num_ctx tuning bought only ~0.2 GiB (it's a weights wall, not KV). Resolved by `batiai/qwen3.6-27b:q3` (~13 GB community Q3 GGUF of the public Qwen3.6-27B weights): 65/65 layers on GPU, fully resident, 30 tok/s vs 12.4 offloaded. **All judge scores on this ledger are on the `batiai/qwen3.6-27b:q3` scale — do not cross-compare with any other judge.** Server env left at Ollama defaults (FA default-on, f16 KV) so the subject matches CLAUDE.md; `OLLAMA_NUM_PARALLEL` unset → server ran `Parallel:1`, so the batched stages (3a/3b/4) serialized despite `max_concurrency=4` — set `OLLAMA_NUM_PARALLEL≥4` to compare batch-stage latency fairly. Signal for future work: judge avg 3.88; recurring weaknesses worth a faithfulness-tightened summary-prompt variant — hallucinated proper nouns ('Carrils'/'iris-tail dragons' for 'Iron and Steel'/'Roomington'), speaker misattribution in summaries, occasional over-split of continuous monologue, and overlapping sibling segments on quin-rs-tut. Note: `min_subdivide_duration_s=120` triggered **no** L2 on either ~8-10 min clip (tree L2=0, 3a/3b ≈ 0s) — the `no-subdivide` ablation will be near-degenerate on these fixtures; add a longer clip (e.g. `3b1b-llm2`, 27 min) to exercise the L2 layer.

### `20260516-0347-b3e9` — `gemma4`  (2026-05-16 03:47Z)

- **Design:** Small model: gemma4:latest (8.0B Q4_K_M). Quality/VRAM/latency floor vs the 14b baseline; small enough to run OLLAMA_NUM_PARALLEL=4.
- **Knobs:** `{"llm_model": "gemma4:latest"}`
- **Prompt override:** none
- **Fixtures:** fox-new, quin-rs-tut · **repeat:** 1
- **Total:** 91.9±37.1s · **stages (l1/decide/l2/sum/l0):** 58/2/4/24/4s
- **LLM calls (pipeline):** 13 · **tokens in/out:** 7515/818
- **VRAM Δ:** 10969.0 MB · **ollama size:** 11264 MB
- **Tree:** L1=5 L2=0 depth=2
- **Judge (batiai/qwen3.6-27b:q3), avg 5.0:**
  - **fox-new** — bound=5 cover=5 faith=5 hier=5 overall=5 (avg 5.0): Segment boundaries align exactly with the host's explicit editorial transitions (e.g., at 03:20 and 05:07), and all titles/summaries tightly track the transcript content without hallucination. The flat L1 structure appropriately mirrors the broadcast's distinct thematic blocks, yielding a highly precise and faithful segmentation.
  - **quin-rs-tut** — bound=5 cover=5 faith=5 hier=5 overall=5 (avg 5.0): The segment boundaries align precisely with natural topic shifts (equipment, inventory, travel, dungeon navigation, and combat tips), and all summaries are tightly grounded in the corresponding transcript spans without hallucination. The L1/L2 nesting is logically warranted, cleanly separating preparation phases from combat execution, with the final L2 split accurately capturing the transition from survival advice to loot-collection efficiency.
- **Env:** `{"git_sha": "b12bbd6", "ollama_version": "ollama version is 0.24.0", "gpu": "NVIDIA GeForce RTX 3090", "ollama_num_parallel": "unset", "python": "3.12.9"}`
- **Verdict vs `baseline`:** Δtotal +41.4s; Δjudge +1.12
- **Conclusion:** KEEP as the **quality ceiling reference**, NOT viable as default (slowest). Judge 5.0/5.0 on every dimension, both fixtures — beats baseline by +1.12 and shows none of baseline's faithfulness faults (no hallucinated proper nouns, no speaker misattribution). But +41.4s slower, almost entirely the **l1 stage (58s vs 12s)**, which decomposes into: (a) a **cold-load confound** — `gemma4:latest` is 9.6 GB (≈2× a normal 8B) and was never loaded this session, so fox-new paid a true ~80s first-load tax while baseline's qwen2.5:14b only looked fast cold because it was OS-page-cache-warm from earlier runs (warm gemma4 l1 was 18.5s on quin-rs-tut, ≈1.6× baseline, not ≈5×); (b) L1 is a **single call** so `max_concurrency=4`/NP=4 can't parallelize it, and gemma4 even reserved 4× KV (`Parallel:4 KvSize:32768`) for zero L1 benefit. **Caveats:** n=1 (no `--repeat`) so the perfect 5.0 needs replication before trusting; latency Δ is **NP-confounded** — baseline ran server `Parallel:1`, gemma4 ran `Parallel:4` (the env-snapshot `ollama_num_parallel:"unset"` is **wrong for all rows** — bench reads its own process env, not the `ollama serve` env; ground truth is `ollama.log` load requests). Next: re-run with `--repeat 3` (discard cold pass) to get fair warm latency and confirm the 5.0 is not judge noise.

### `20260516-0355-5ce7` — `granite4.1-8b`  (2026-05-16 03:55Z)

- **Design:** Small model: granite4.1:8b (8.8B Q4_K_M). Quality/VRAM/latency floor vs the 14b baseline; small enough to run OLLAMA_NUM_PARALLEL=4.
- **Knobs:** `{"llm_model": "granite4.1:8b"}`
- **Prompt override:** none
- **Fixtures:** fox-new, quin-rs-tut · **repeat:** 1
- **Total:** 37.7±14.1s · **stages (l1/decide/l2/sum/l0):** 19/1/1/13/3s
- **LLM calls (pipeline):** 11 · **tokens in/out:** 6634/886
- **VRAM Δ:** 10662.0 MB · **ollama size:** 14336 MB
- **Tree:** L1=5 L2=2 depth=3
- **Judge (batiai/qwen3.6-27b:q3), avg 4.62:**
  - **fox-new** — bound=5 cover=5 faith=3 hier=5 overall=3 (avg 4.5): Segment boundaries and coverage are excellent, aligning precisely with conversational turns and topic shifts without gaps or overlaps. However, summary faithfulness is significantly compromised by consistent speaker-index mismatches and a critical semantic inversion in the final segment, where the Senator's defense of the Trump DOJ is mischaracterized as a criticism of it.
  - **quin-rs-tut** — bound=5 cover=5 faith=4 hier=5 overall=4 (avg 4.75): Segment boundaries align precisely with natural topic shifts and tile the recording completely without gaps, but the L1-2 title hallucinates a 'Carrils Dungeon' from a likely ASR mis-transcription of armor, slightly undermining faithfulness.
- **Env:** `{"git_sha": "b12bbd6", "ollama_version": "ollama version is 0.24.0", "gpu": "NVIDIA GeForce RTX 3090", "ollama_num_parallel": "unset", "python": "3.12.9"}`
- **Verdict vs `baseline`:** Δtotal -12.8s; Δjudge +0.74
- **Conclusion:** PROMISING — strongest candidate to **promote toward default**, but confirm before flipping. **Pareto win on all three axes vs the 14B baseline:** faster (37.7s vs 50.5, Δ−12.8s), ~35% less VRAM (10662 vs 16420 MB), AND higher quality (judge 4.62 vs 3.88, Δ+0.74). Notably the only design here to build an **L2 layer** (tree L1=5 L2=2 depth=3) at `min_subdivide_duration_s=120` where baseline & gemma4 stayed flat — it exercises the hierarchy these fixtures otherwise don't. Quality is strong-not-perfect: docked on **summary faithfulness** — a "critical semantic inversion" on fox-new (a Senator's *defense* of the DOJ summarized as *criticism*; faith=3/overall=3 there) + speaker-index mismatches, and the same ASR-driven 'Carrils Dungeon' hallucination on quin-rs-tut that baseline also hit (so partly an upstream Whisper artifact, not purely the LLM). **Caveats:** n=1 (no `--repeat`); latency Δ is **NP-confounded** — baseline ran server `Parallel:1`, this ran `Parallel:4` (env-snapshot `ollama_num_parallel:"unset"` is wrong for all rows; see baseline conclusion / `ollama.log`). granite's l1 is a single call (no NP benefit) at 19s vs baseline 12s, so its decode is marginally slower per-call — the total win comes from NP=4 + fewer/cheaper batch calls, so a fair test must **re-baseline at NP=4**. Next: `--repeat 3` + an NP=4 baseline row; if the win holds, promote `granite4.1-8b` to the recommended small-model default and test a faithfulness-tightened summary prompt against the semantic-inversion failure.

### `20260516-0605-aaf5` — `baseline`  (2026-05-16 06:05Z)

- **Design:** Current shipped defaults (src/autorag/agent.py build_topic_runnable): gemma4:latest with thinking disabled.
- **Knobs:** `{"llm_model": "gemma4:latest", "num_ctx_l1": 8192, "num_ctx_fanout": 8192, "max_concurrency": 4, "min_subdivide_duration_s": 120.0, "reasoning": false}`
- **Prompt override:** none
- **Fixtures:** fox-new, quin-rs-tut · **repeat:** 2
- **Total:** 25.1±11.9s · **stages (l1/decide/l2/sum/l0):** 13/1/2/7/1s
- **LLM calls (pipeline):** 16 · **tokens in/out:** 9645/900
- **VRAM Δ:** 186.0 MB · **ollama size:** 11264 MB
- **Tree:** L1=4 L2=2 depth=3
- **Judge (gemma4:26b), avg n/a:**
  - (judging skipped)
- **Env:** `{"git_sha": "b12bbd6", "ollama_version": "ollama version is 0.24.0", "gpu": "NVIDIA GeForce RTX 3090", "ollama_num_parallel": "1", "ollama_flash_attention": "Enabled(auto)", "ollama_kv_cache_type": "f16(default)", "python": "3.12.9"}`
- **Verdict vs `baseline`:** —
- **Conclusion:** **This is the FA-OFF arm of a flash-attention A/B (vs `20260516-0608-1083`, FA-ON).** ⚠️ **The Env `ollama_flash_attention: "Enabled(auto)"` is WRONG for this row.** This run's server was started with `OLLAMA_FLASH_ATTENTION=0` (verified: `/tmp/ollama-probe-fa0.log` logs `FlashAttention:Disabled` for the gemma4:latest load). The bench's auto-capture read the stale default `/tmp/ollama.log` (the *original* session server, FA auto-on) because `AUTORAG_OLLAMA_LOG` was not set for this invocation — the FA-ON row, run with `AUTORAG_OLLAMA_LOG` pointed at the live log, captured correctly. **Methodological lesson: the `ollama_flash_attention`/`kv_cache_type` env fields are only trustworthy when `AUTORAG_OLLAMA_LOG` points at the active server's log; otherwise treat them as unverified.** **Finding (FA OFF vs FA ON, gemma4:latest, num_ctx=8192, f16 KV both arms, NP=1):** turning flash attention OFF roughly *doubles* the L1 stage (13s vs 5s) and inflates total (25.1±11.9s vs 16.8±2.2s) with much wider variance; stages 3a/3b/sum/l0 are unchanged (1/2/7/1 both). The effect is **localized to L1** — the only long-context stage (full time-bucketed transcript at num_ctx=8192) — which is exactly where flash-attention prefill scales with sequence length; the short fan-out prompts see no FA effect. `ollama size` is identical (11264 MB) and tree shape identical (L1=4 L2=2 depth=3) → FA changes neither memory footprint nor output here; the VRAM Δ (186 vs 65) is residency-confound noise (warm-up loads), not an FA signal. **Caveats:** n=2 fixtures × repeat 2 (small); per-invoke `keep_alive=0` eviction means every fixture reloads cold in both arms (common-mode but the variance source); this isolates **flash attention alone** (KV held at f16) — the separate q8_0-KV axis (which *requires* FA) is unmeasured. **Decision: KEEP as the FA-OFF reference.** Do not promote to a named design — FA is a server-env setting, not a `build_topic_runnable` knob, so it cannot be a `designs.json` entry; the env snapshot + this conclusion are the record. **⚠ [REVISED 2026-05-16 — see `20260516-0636-f838`]:** the "FA roughly doubles L1" reading above does **not** hold. This row's ±11.9 total spread fails the skill's significance bar; the controlled num_ctx_l1=16384 FA A/B (`…-58a7`/`…-f838`, tight ±2.2) and the q8_0 run (`…-7411`) all show L1≈6s with **no FA effect**. The 13s L1 here was a cold-load artifact (model not resident for that invoke), not flash attention. Row kept as history; the FA-latency interpretation is **superseded**. The provenance lesson (env field mislabelled without `AUTORAG_OLLAMA_LOG`) still stands.

### `20260516-0608-1083` — `baseline`  (2026-05-16 06:08Z)

- **Design:** Current shipped defaults (src/autorag/agent.py build_topic_runnable): gemma4:latest with thinking disabled.
- **Knobs:** `{"llm_model": "gemma4:latest", "num_ctx_l1": 8192, "num_ctx_fanout": 8192, "max_concurrency": 4, "min_subdivide_duration_s": 120.0, "reasoning": false}`
- **Prompt override:** none
- **Fixtures:** fox-new, quin-rs-tut · **repeat:** 2
- **Total:** 16.8±2.2s · **stages (l1/decide/l2/sum/l0):** 5/1/2/7/1s
- **LLM calls (pipeline):** 16 · **tokens in/out:** 9640/893
- **VRAM Δ:** 65.0 MB · **ollama size:** 11264 MB
- **Tree:** L1=4 L2=2 depth=3
- **Judge (gemma4:26b), avg n/a:**
  - (judging skipped)
- **Env:** `{"git_sha": "b12bbd6", "ollama_version": "ollama version is 0.24.0", "gpu": "NVIDIA GeForce RTX 3090", "ollama_num_parallel": "1", "ollama_flash_attention": "Enabled", "ollama_kv_cache_type": "f16(default)", "python": "3.12.9"}`
- **Verdict vs `baseline`:** —
- **Conclusion:** **FA-ON arm of the flash-attention A/B** (paired with `20260516-0605-aaf5`, FA-OFF). Server `OLLAMA_FLASH_ATTENTION=1`; Env `ollama_flash_attention: "Enabled"` is correct here (`AUTORAG_OLLAMA_LOG` was pointed at the live `/tmp/ollama-probe-fa1.log`). vs the FA-OFF arm: **L1 5s vs 13s, total 16.8±2.2s vs 25.1±11.9s** — FA ~halves the long-context L1 stage and tightens variance ~5×; all other stages, `ollama size` (11264 MB), token counts (~9640) and tree shape (L1=4 L2=2 depth=3) are unchanged → pure latency win on long-context prefill, quality- and footprint-neutral. **Context (the headline qualitative finding):** Ollama 0.24 **auto-enables FA for gemma4 whenever `OLLAMA_FLASH_ATTENTION` is unset/default** (log: `Flash Attention was auto, set to enabled`); explicit `=0` is the only way to get `FlashAttention:Disabled`. So this FA-ON state *is* the normal/default operating condition for gemma4 (including every pre-flip `gemma4` LEDGER row — they already ran with FA auto-on); the FA-OFF arm is the `=0` counterfactual, ~50% slower on L1. **Implication for `start-ollama.sh`:** `OLLAMA_FLASH_ATTENTION=1` is, for gemma4, *redundant for enabling FA* (auto already does it) — its load-bearing role is unlocking the explicit `q8_0` KV cache (which requires FA and is a separate, larger-ctx/more-slots lever, not exercised at the agent's num_ctx=8192). **Decision: KEEP as the FA-ON reference; no named design (server-env, not a knob).** **⚠ [REVISED 2026-05-16 — see `20260516-0636-f838`]:** the "FA ~halves the long-context L1 stage" **latency claim is retracted** — the controlled num_ctx_l1=16384 FA A/B (`…-58a7`/`…-f838`) and the q8_0 null (`…-7411`) show FA on vs off is a wash for this workload at every tested ctx; the `…-aaf5` contrast was cold-load noise, not flash attention. The *qualitative* findings here STAND (Ollama auto-enables FA for gemma4 unless `=0`; `start-ollama.sh` FA=1 is redundant for gemma4) — and Exp A further showed its `q8_0` KV is likewise inert at the agent's ctx. Only the speed delta is withdrawn.

### `20260516-0631-7411` — `baseline`  (2026-05-16 06:31Z)

- **Design:** Current shipped defaults (src/autorag/agent.py build_topic_runnable): gemma4:latest with thinking disabled.
- **Knobs:** `{"llm_model": "gemma4:latest", "num_ctx_l1": 8192, "num_ctx_fanout": 8192, "max_concurrency": 4, "min_subdivide_duration_s": 120.0, "reasoning": false}`
- **Prompt override:** none
- **Fixtures:** fox-new, quin-rs-tut · **repeat:** 2
- **Total:** 16.6±1.2s · **stages (l1/decide/l2/sum/l0):** 6/1/1/7/1s
- **LLM calls (pipeline):** 13 · **tokens in/out:** 8470/833
- **VRAM Δ:** 12.0 MB · **ollama size:** 10240 MB
- **Tree:** L1=4 L2=2 depth=3
- **Judge (gemma4:26b), avg n/a:**
  - (judging skipped)
- **Env:** `{"git_sha": "b12bbd6", "ollama_version": "ollama version is 0.24.0", "gpu": "NVIDIA GeForce RTX 3090", "ollama_num_parallel": "1", "ollama_flash_attention": "Enabled", "ollama_kv_cache_type": "q8_0", "python": "3.12.9"}`
- **Verdict vs `baseline`:** —
- **Conclusion:** **Exp A — q8_0-KV vs f16-KV (FA on, num_ctx=8192, NP=1).** f16 anchor: `20260516-0608-1083` (same protocol). Total **16.6±1.2s vs 16.8±2.2s**, L1 6s vs 5s — **statistically indistinguishable** (Δ ≪ ±spread). `ollama size` 10240 vs 11264 MB looks smaller but is dominated by residency-sampling noise, not a real footprint win at this ctx. **q8_0 KV is a no-op for the agent at num_ctx=8192**: the KV cache is tiny next to the 9.6 GB weights, so halving KV precision moves neither latency nor effective footprint — the "weights wall, not KV" result from the `…-ae14` judge-provenance note, now confirmed for the *subject* too. **Implication:** `OLLAMA_KV_CACHE_TYPE=q8_0` in `start-ollama.sh` is inert at the agent's default operating point; its value is confined to large-ctx / NP≥4 / bigger-model (`gemma4:26b`) regimes. KEEP as the q8_0 reference; no named design (server-env, not a knob).

### `20260516-0633-58a7` — `baseline`  (2026-05-16 06:33Z)

- **Design:** Current shipped defaults (src/autorag/agent.py build_topic_runnable): gemma4:latest with thinking disabled.
- **Knobs:** `{"llm_model": "gemma4:latest", "num_ctx_l1": 16384, "num_ctx_fanout": 8192, "max_concurrency": 4, "min_subdivide_duration_s": 120.0, "reasoning": false}`
- **Prompt override:** none
- **Fixtures:** fox-new, quin-rs-tut · **repeat:** 2
- **Total:** 22.1±2.2s · **stages (l1/decide/l2/sum/l0):** 6/6/2/8/1s
- **LLM calls (pipeline):** 16 · **tokens in/out:** 9645/902
- **VRAM Δ:** 226.0 MB · **ollama size:** 11264 MB
- **Tree:** L1=4 L2=2 depth=3
- **Judge (gemma4:26b), avg n/a:**
  - (judging skipped)
- **Env:** `{"git_sha": "b12bbd6", "ollama_version": "ollama version is 0.24.0", "gpu": "NVIDIA GeForce RTX 3090", "ollama_num_parallel": "1", "ollama_flash_attention": "Disabled", "ollama_kv_cache_type": "f16(default)", "python": "3.12.9"}`
- **Verdict vs `baseline`:** —
- **Conclusion:** **Exp B1 — FA-OFF arm at num_ctx_l1=16384** (pair: `20260516-0636-f838`, FA-ON, same session/protocol). Env correctly captured (`Disabled`; `AUTORAG_OLLAMA_LOG` was set this time). Combined Exp B verdict lives in `…-f838`. Note `decide`=6s (vs ~1s at uniform num_ctx) is the **Stage 2→3a model reload** forced by num_ctx_l1(16384) ≠ num_ctx_fanout(8192) — common-mode with B2 so it cancels in the FA delta, but it quantifies the CLAUDE.md "one reload" cost at ≈+5s/run for gemma4 on these clips. KEEP as the FA-OFF@16384 reference; no named design (server-env, not a knob).

### `20260516-0636-f838` — `baseline`  (2026-05-16 06:36Z)

- **Design:** Current shipped defaults (src/autorag/agent.py build_topic_runnable): gemma4:latest with thinking disabled.
- **Knobs:** `{"llm_model": "gemma4:latest", "num_ctx_l1": 16384, "num_ctx_fanout": 8192, "max_concurrency": 4, "min_subdivide_duration_s": 120.0, "reasoning": false}`
- **Prompt override:** none
- **Fixtures:** fox-new, quin-rs-tut · **repeat:** 2
- **Total:** 21.7±2.2s · **stages (l1/decide/l2/sum/l0):** 6/5/2/8/1s
- **LLM calls (pipeline):** 16 · **tokens in/out:** 9645/901
- **VRAM Δ:** 124.0 MB · **ollama size:** 11264 MB
- **Tree:** L1=4 L2=2 depth=3
- **Judge (gemma4:26b), avg n/a:**
  - (judging skipped)
- **Env:** `{"git_sha": "b12bbd6", "ollama_version": "ollama version is 0.24.0", "gpu": "NVIDIA GeForce RTX 3090", "ollama_num_parallel": "1", "ollama_flash_attention": "Enabled", "ollama_kv_cache_type": "f16(default)", "python": "3.12.9"}`
- **Verdict vs `baseline`:** —
- **Conclusion:** **Exp B2 — FA-ON @ num_ctx_l1=16384; carries the combined Exp B verdict.** vs B1 `…-58a7` (FA-off, same session/protocol): total **21.7±2.2 vs 22.1±2.2s**, L1 **6s vs 6s** — **no measurable FA effect at 16384** (Δ ≪ ±spread). **This + Exp A (q8_0 null) REVISES the earlier `…-aaf5`/`…-1083` "FA ~halves L1" finding.** Full gemma4:latest L1 table: aaf5=13s (±11.9), 1083=5s, 7411=6s, 58a7=6s, f838=6s — every *controlled* run is L1≈5–6s with tight ±; only `…-aaf5` deviates and its ±11.9 spread fails the skill's "delta must exceed run-to-run spread" rule. **Corrected conclusion: FA on vs off is a *wash* for the gemma4 agent at every tested ctx (8192 & 16384), NP=1, f16 KV — the earlier 13→5s "FA halves L1" was a cold-load artifact of aaf5's single noisy arm, not flash attention.** Findings that STAND (log-derived, not latency-inferred): (1) Ollama 0.24 auto-enables FA for gemma4 unless `OLLAMA_FLASH_ATTENTION=0`; (2) no `GGML_ASSERT` on the safe `MultiUserCache:false` path; (3) `start-ollama.sh`'s `FLASH_ATTENTION=1` **and** `KV_CACHE_TYPE=q8_0` are both **inert at the agent's default operating point** — their value is confined to large-ctx / NP≥4 / `gemma4:26b` regimes. Independent clean finding: num_ctx_l1=16384 costs ≈+5s/run (uniform-8192 ≈16.7s → 16384 ≈21.9s) from the Stage 2→3a reload, FA-independent — so only raise num_ctx_l1 for ≈1 h+ audio that genuinely needs the L1 fidelity. KEEP as the FA-ON@16384 reference; no named design (server-env, not a knob).

<!-- RUNS:END -->
