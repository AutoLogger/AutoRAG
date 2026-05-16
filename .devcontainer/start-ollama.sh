#!/usr/bin/env bash
# Start the native, in-container Ollama server.
#
# The devcontainer has no systemd, so there is no `ollama.service`. This script
# is launched from devcontainer.json's postStartCommand and runs as the
# unprivileged `node` user (remoteUser) — no sudo, no root daemon, no `ollama`
# system user. Models live in $HOME/.ollama (bind-mounted to the host's
# ~/.ollama), so pulls persist across rebuilds and are shared with the host.
#
# Idempotent: a no-op if a server is already listening.
set -euo pipefail

OLLAMA_URL="${AUTORAG_OLLAMA_BASE_URL:-http://localhost:11434}"

if curl -fsS "${OLLAMA_URL}/api/version" >/dev/null 2>&1; then
  echo "ollama: already running at ${OLLAMA_URL}"
  exit 0
fi

# $HOME/.ollama is the bind mount; ensure it exists before serving.
mkdir -p "${HOME}/.ollama"

# --- Server-side performance tuning -----------------------------------------
# Read by `ollama serve` at startup (NOT by the Python agent, which sets
# num_ctx / keep_alive per request). All overridable: an externally-set value
# wins, matching OLLAMA_URL above.
#
#   FLASH_ATTENTION    fused attention kernel — faster, lower KV memory.
#   KV_CACHE_TYPE      q8_0 KV cache (needs flash attention): ~half the
#                      per-slot KV VRAM at near-lossless quality, plus less
#                      memory bandwidth per call. The default LLM is now
#                      `gemma4:latest` (8B Q4_K_M, ~9.6 GB) — much lighter
#                      than the old qwen2.5:14b-q8_0 (~15 GB) — so the 4
#                      agent slots + model land at ~11 GB total, leaving
#                      generous headroom on a 24 GB card.
#   NUM_PARALLEL       4 request slots — the agent's batched stages
#                      (3a decide / 3b L2 boundaries / 4 per-node summaries)
#                      need >=4 for Runnable.batch to actually run concurrently
#                      (see CLAUDE.md "Ollama tuning").
#   MAX_LOADED_MODELS  pin to 1: the agent uses a single LLM, so never evict
#                      it to load a second model.
#
# Caveat (validate before trusting): the agent-lab LEDGER's gemma4 rows were
# benchmarked under Ollama's *default* server env (flash attn default-on,
# f16 KV) — NOT this tuned q8_0-KV + explicit FA=1 + NP=4 combo. Gemma-family
# models use interleaved sliding-window attention, historically a sensitive
# pairing with flash attention in llama.cpp/Ollama. The settings are sound
# and each is `:=`-overridable; re-run bench.py to confirm gemma4 quality
# holds under them. The dominant gemma4 latency lever is client-side, not
# here: the agent disables thinking (`reasoning=False`) by default.
#
# Deliberately NOT setting OLLAMA_MULTIUSER_CACHE. Combining it with
# FLASH_ATTENTION=1 *and* concurrent slots trips
# `GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers")`.
# The per-slot prefix cache (what the K identical summary prompts rely on)
# works without it.
: "${OLLAMA_FLASH_ATTENTION:=1}"
: "${OLLAMA_KV_CACHE_TYPE:=q8_0}"
: "${OLLAMA_NUM_PARALLEL:=4}"
: "${OLLAMA_MAX_LOADED_MODELS:=1}"
export OLLAMA_FLASH_ATTENTION OLLAMA_KV_CACHE_TYPE OLLAMA_NUM_PARALLEL OLLAMA_MAX_LOADED_MODELS
echo "ollama: tuning FLASH_ATTENTION=${OLLAMA_FLASH_ATTENTION} KV_CACHE_TYPE=${OLLAMA_KV_CACHE_TYPE} NUM_PARALLEL=${OLLAMA_NUM_PARALLEL} MAX_LOADED_MODELS=${OLLAMA_MAX_LOADED_MODELS}"

# Detach from this shell so the server outlives the postStartCommand. PID 1 in
# the devcontainer is the keep-alive loop, which reaps the reparented process.
nohup ollama serve >/tmp/ollama.log 2>&1 &
disown

# Block until the API is up so dependent steps (and the user) see a ready
# server, then exit (the server keeps running, disowned).
for _ in $(seq 1 30); do
  if curl -fsS "${OLLAMA_URL}/api/version" >/dev/null 2>&1; then
    echo "ollama: ready at ${OLLAMA_URL} ($(ollama --version 2>/dev/null | tail -n1))"
    exit 0
  fi
  sleep 1
done

echo "ollama: failed to become ready within 30s; see /tmp/ollama.log" >&2
exit 1
