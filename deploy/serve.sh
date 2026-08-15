#!/bin/bash
# Qwen3.8-27B (NVFP4) on a single RTX 5090 -- recommended serving config.
#
# Every flag here is here for a measured or verified reason; see README.md.
set -euo pipefail

MODEL="${MODEL:-$HOME/models/Qwen3.8-27B-NVFP4}"
VENV="${VENV:-$HOME/qwen38/.venv}"
PORT="${PORT:-8000}"

# vLLM JIT-compiles a kernel during the dummy sampler run and shells out to
# `ninja`. Without the venv bin on PATH the engine dies inside
# _initialize_kv_caches with FileNotFoundError: 'ninja' -- an error that names
# neither the cause nor the fix.
export PATH="$VENV/bin:$HOME/.local/bin:$PATH"

exec "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name qwen38 \
  --port "$PORT" \
  \
  `# ~2x decode throughput. The single most important flag here.` \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  \
  `# Roughly doubles KV capacity. FlashAttention cannot serve FP8 KV on sm_120` \
  `# (it wants FA3/SM90 or FA4/SM100), so vLLM selects FlashInfer by itself.` \
  --kv-cache-dtype fp8 \
  \
  `# Concurrency here is bounded by MAMBA cache blocks, not KV: this model is a` \
  `# 3:1 linear-attention/full-attention hybrid and each decode sequence takes` \
  `# one Mamba block. Too high and the engine refuses to start with` \
  `# "max_num_seqs (N) exceeds available Mamba cache blocks".` \
  --max-num-seqs 8 \
  \
  `# 32k measured at 126,976 KV tokens baseline / 56,599 with MTP K3.` \
  `# 128k is reachable without MTP; see the KV table in README.md.` \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  \
  `# This model thinks by default at reasoning_effort=xhigh and can emit tens of` \
  `# thousands of reasoning tokens. Without a reasoning parser those land in` \
  `# content and break clients expecting an answer.` \
  --reasoning-parser qwen3 \
  \
  --no-enable-log-requests

# IMPORTANT, client side: send
#     "chat_template_kwargs": {"preserve_thinking": false}
# on every multi-turn request. The stock chat template otherwise renders an empty
# <think></think> on EVERY prior assistant turn, which truncates multi-turn
# agentic work and invalidates prefix caching. See README.md.
#
# Per-request thinking budget -- ONLY "xhigh" (default), "high" (alias for
# xhigh) and "low" do anything. "medium" is accepted by the validator but has NO
# branch in the template: it silently drops the reasoning instruction entirely,
# rendering a 60-char system prompt against 297 at default. It is not a middle
# setting, it is less steering than the default. Verified by rendering.
#     "chat_template_kwargs": {"reasoning_effort": "low"}
