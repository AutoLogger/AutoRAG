Audio pipeline design
=====================

The audio → topics pipeline lives in :mod:`autorag.agent`. It's
structured as five focused stages so each LLM call has one job and the
heavy stages share an identical prompt prefix for cache reuse.

Stages
------

::

    1. Whisper                              -> list[WordSpan]               1 call
    2. L1 boundaries  (single LLM call)     -> list[{s,e}]                  1 LLM
    3a Decide subdivide  (per long L1)      -> list[bool]                   N LLM
    3b L2 boundaries  (per yes-L1, batched) -> list[list[{s,e}]]            M LLM (M<=N)
    4. Summarize nodes  (per L1+L2, batched)-> {title,summary} per node     K LLM
    5. L0 aggregate                         -> {title, summary}             1 LLM

Total LLM calls per clip: roughly
``2 + N1_long + N1_yes + N1 + N2_total`` — about 20 calls for a
seven-minute clip.

Boundary calls receive the transcript as a time-bucketed view
(:func:`autorag.blocks.format_blocks`, 30-second windows — one
``MM:SS-MM:SS Speaker K: <words>`` line per turn instead of one
timestamped line per word, which keeps the boundary prompts compact).
They emit ``{s, e}`` as ``MM:SS`` strings copied straight from those
range markers; :func:`autorag.agent._parse_ts` converts them back to
float seconds before tiling — the model never does the arithmetic.
Per-node summary calls operate on the slice's plain text (no
timestamps) and emit ``{title, summary}``. The ``K = N1 + N2`` summary
calls share an identical prompt prefix so Ollama's per-slot prefix
cache pays once.

Final shape: ``{"topics": [L0]}`` with ``L0.children = [L1...]``,
each ``L1.children = [L2...]`` or ``[]``. The L0 root is the explicit
"what is this audio about" node.

Default LLM model: ``qwen2.5:14b-instruct-q8_0``. Override via the
``--llm-model`` flag on the CLI or the ``llm_model`` kwarg on the
SDK methods.

Whisper backend
---------------

:mod:`autorag.whisper_runner` runs whisperX — faster-whisper
(CTranslate2) for transcription plus a wav2vec2 forced-alignment pass
for frame-accurate word timestamps.

After each ``transcribe_segment`` call:

* The CTranslate2 model is removed from the module cache so Python GC
  can free VRAM.
* The wav2vec2 alignment model is offloaded to CPU via PyTorch
  ``.to("cpu")``; the next call restores it to CUDA.
* On a CUDA error, the runner falls back to CPU.

Diarization
-----------

:mod:`autorag.diarize` uses ``pyannote/speaker-diarization-3.1``,
which is HuggingFace-gated. ``HF_TOKEN`` must be set. Without it (or
on a load / runtime failure), every word is labelled ``"0"`` and the
agent logs a warning — output then matches pre-diarization behaviour.

Each :data:`~autorag.types.WordSpan` carries a ``speaker`` field
normalized to ``"0"``, ``"1"``, … in first-appearance order. Both
transcript views the agent feeds the LLM build on
:func:`autorag.blocks.group_by_speaker` to coalesce consecutive
same-speaker spans into turns: the boundary stages use
:func:`autorag.blocks.format_blocks` (``MM:SS-MM:SS Speaker K:
<words>``) and the per-node summary input uses ``Speaker N: <words>``,
so the LLM always sees explicit turn-taking.

After each ``_run_diarization`` call the pyannote pipeline is
offloaded to CPU and VRAM freed; ``_ensure_pipeline_on_cuda`` restores
it on the next call.

Why split boundaries from summaries
-----------------------------------

Earlier versions of the agent asked one LLM call to do "find the L1
sections AND title and summarize each one." That confused models on
long clips: section boundaries drifted as the model spent attention
on the prose. Splitting boundary detection (a constrained
``[{s, e}]`` output) from summarization (per-section ``{title,
summary}``) gives:

* One focused prompt per call (boundaries OR prose).
* A constant prompt prefix across the K summary calls, so the
  prefix-cache slot stays warm.
* Independent retry: a bad boundary call can be replayed without
  redoing all the summarization work.
