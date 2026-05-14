Ollama tuning
=============

The agent's batched stages parallelize requests to Ollama, so the
relevant server-side knob is ``OLLAMA_NUM_PARALLEL``. The right value
depends on whether you're tuning for parallelism or for a bigger
single-stream model.

``OLLAMA_NUM_PARALLEL``
-----------------------

* **≥ 4** for the agent's batched stages (Stage 3a "decide", Stage 3b
  L2 boundaries, Stage 4 per-node summaries). Required for
  ``Runnable.batch`` to actually parallelize.
* **= 1** for one-shot calls on a *bigger* model. Ollama pre-reserves
  all ``NUM_PARALLEL`` slots' KV cache at the configured ``num_ctx``,
  so 4 idle slots steal VRAM that the bigger model needs.

On a 24 GB GPU with ``NUM_PARALLEL=1``, you can run
``qwen2.5:14b-q8_0`` (~15 GB) with ``num_ctx=16384`` (~3 GB KV) and
full GPU offload. Bumping to ``num_ctx=32768`` pushes some layers onto
CPU. Verify with ``ollama ps`` after a load.

``OLLAMA_FLASH_ATTENTION`` and ``OLLAMA_MULTIUSER_CACHE``
---------------------------------------------------------

**Do not** combine ``OLLAMA_FLASH_ATTENTION=1`` with
``OLLAMA_MULTIUSER_CACHE=true`` and concurrent slots — it triggers::

    GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers")

Drop ``MULTIUSER_CACHE``. The per-slot prefix cache still works
without it, which is what the agent's K identical summary prompts
benefit from.

Per-slot KV-cache sizing
------------------------

The agent caps ``num_ctx`` to fit the typical "4 slots × KV +
~9 GB model" budget on a 24 GB card:

* **L1 call** — 16 K context.
* **Fan-out / summary calls** — 8 K context.

These values are conservative enough that bumping the LLM to
``qwen2.5:32b-q4_K_M`` typically just needs ``NUM_PARALLEL=1`` and no
other changes.

Resolving the Ollama URL
------------------------

Both the agent (LLM chat) and :class:`autorag.embed.Embedder`
(embeddings) read ``AUTORAG_OLLAMA_BASE_URL`` (default
``http://localhost:11434``). The embedding model is separately
controlled with ``AUTORAG_EMBED_MODEL`` (default
``nomic-embed-text``).
