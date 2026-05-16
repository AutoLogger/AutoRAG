Ollama tuning
=============

The agent's batched stages parallelize requests to Ollama, so the
relevant server-side knob is ``OLLAMA_NUM_PARALLEL``. The right value
depends on whether you're tuning for parallelism or for a bigger
single-stream model.

Devcontainer defaults
---------------------

``.devcontainer/start-ollama.sh`` exports a tuned set before launching
``ollama serve`` (each overridable — an externally-set value wins):

.. code-block:: bash

    OLLAMA_FLASH_ATTENTION=1
    OLLAMA_KV_CACHE_TYPE=q8_0
    OLLAMA_NUM_PARALLEL=4
    OLLAMA_MAX_LOADED_MODELS=1

These are server-side only; the Python agent still sets ``num_ctx`` and
``keep_alive`` per request. Flash attention plus a ``q8_0`` KV cache
roughly halve per-slot KV VRAM at near-lossless quality and cut
attention memory bandwidth, so the four agent slots stay concurrent
*and* each call runs faster. ``MAX_LOADED_MODELS=1`` pins the single
agent LLM so it is never evicted to load a second model. The sections
below explain the reasoning behind each value.

The default agent LLM is ``gemma4:latest`` (8B Q4_K_M, ~9.6 GB), a
``thinking``-capable model. The agent disables thinking
(``reasoning=False``, sent to Ollama as ``think: false``) for all five
mechanical-JSON stages — that is a client-side per-request setting, not
a server env knob, but it is the dominant gemma4 latency lever, so it
is noted here for anyone tuning for speed. **Validation caveat:** the
agent-lab LEDGER's gemma4 rows were measured under Ollama's *default*
server env (flash attention default-on, f16 KV), **not** this tuned
``q8_0``-KV + explicit ``FLASH_ATTENTION=1`` + ``NUM_PARALLEL=4``
combination. Gemma-family models use interleaved sliding-window
attention, historically a sensitive pairing with flash attention in
llama.cpp / Ollama. The settings are sound and each is overridable;
re-run ``bench.py`` to confirm gemma4 quality holds under them.

``OLLAMA_NUM_PARALLEL``
-----------------------

* **≥ 4** for the agent's batched stages (Stage 3a "decide", Stage 3b
  L2 boundaries, Stage 4 per-node summaries). Required for
  ``Runnable.batch`` to actually parallelize.
* **= 1** for one-shot calls on a *bigger* model. Ollama pre-reserves
  all ``NUM_PARALLEL`` slots' KV cache at the configured ``num_ctx``,
  so 4 idle slots steal VRAM that the bigger model needs.

On a 24 GB GPU the default ``gemma4:latest`` (~9.6 GB) lands at
~11 GB total with the four slots at ``num_ctx=8192`` and the ``q8_0``
KV cache — full GPU offload with wide headroom. The
``NUM_PARALLEL=1`` case now applies to the bigger ``gemma4:26b`` (the
25.8B sibling, ~17 GB): a single stream gets the freed slot KV at
``num_ctx=8192`` with full offload, and a single-stream
``num_ctx=16384`` still fits. Verify with ``ollama ps`` after a load.

``OLLAMA_FLASH_ATTENTION`` and ``OLLAMA_MULTIUSER_CACHE``
---------------------------------------------------------

Flash attention is on by default (see *Devcontainer defaults*), which
also unlocks the ``q8_0`` KV cache. **Do not** combine
``OLLAMA_FLASH_ATTENTION=1`` with ``OLLAMA_MULTIUSER_CACHE=true`` and
concurrent slots — it triggers::

    GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers")

Because the devcontainer ships ``FLASH_ATTENTION=1`` with
``NUM_PARALLEL=4`` (concurrent slots), ``MULTIUSER_CACHE`` must stay
unset — ``start-ollama.sh`` deliberately omits it. The per-slot prefix
cache still works without it, which is what the agent's K identical
summary prompts benefit from.

Per-slot KV-cache sizing
------------------------

Every stage uses the same context size — ``num_ctx=8192`` — chosen to
fit the typical "4 slots × KV + ~9 GB model" budget on a 24 GB card.
With the default ``q8_0`` KV cache each slot's KV is roughly half its
f16 size, so that budget now has noticeably more headroom than the
original f16 sizing (which left it tight). A uniform ``num_ctx`` is
deliberate: Ollama reloads a model whenever ``num_ctx`` changes
between requests, so keeping it constant is what lets the model stay
resident across all five stages (see *Model residency during a run*
below).

``num_ctx_l1`` remains an overridable kwarg
(:func:`autorag.agent.build_topic_runnable` /
:meth:`autorag.core.AutoRAG.generate_topics`). The Stage 2 (L1) call
sees the *whole* time-bucketed transcript; on very long audio
(≈1 hr+) 8192 tokens can truncate it and degrade L1 boundary quality.
Raising ``num_ctx_l1`` back to e.g. ``16384`` fixes that, at the cost
of exactly one model reload at the Stage 2→3a boundary (the L1 call
then differs in ``num_ctx`` from the fan-out stages).

These values are conservative enough that bumping the LLM to the
bigger ``gemma4:26b`` (~17 GB) typically just needs ``NUM_PARALLEL=1``
and no other changes.

Model residency during a run
----------------------------

The topic agent keeps the LLM resident in VRAM for the whole run
instead of reloading it per stage. Two settings make that work:

* ``keep_alive="5m"`` on every chat client — long enough to span the
  sub-second gaps between stages, so Ollama never unloads mid-run. It
  doubles as a crash-safety fallback: if the run dies before the
  explicit eviction below, Ollama still unloads the model on its own
  after five idle minutes.
* a uniform ``num_ctx`` across all stages (see *Per-slot KV-cache
  sizing*) — without this the 16 K→8 K transition at the Stage 2→3a
  boundary would force a reload even with ``keep_alive`` set.

When the run finishes (or any stage raises), ``_build_tree`` issues
one throwaway ``keep_alive=0`` generation that evicts the model so it
doesn't squat VRAM during the downstream embed / ``/viz`` step. This
is the LLM analogue of the whisper / pyannote "offload to CPU after
use" idiom.

Because all stages now share one ``num_ctx`` and the model stays
warm, ``OLLAMA_NUM_PARALLEL`` ≥ 4 is unambiguously beneficial: the
batched stages parallelize and there is no per-stage reload cost to
trade off against.

Resolving the Ollama URL
------------------------

Both the agent (LLM chat) and :class:`autorag.embed.Embedder`
(embeddings) read ``AUTORAG_OLLAMA_BASE_URL`` (default
``http://localhost:11434``). The embedding model is separately
controlled with ``AUTORAG_EMBED_MODEL`` (default
``nomic-embed-text``).
