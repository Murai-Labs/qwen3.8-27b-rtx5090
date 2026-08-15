#!/bin/bash
# Three levers, one control. Everything else held constant:
# fp8 KV (never override), MTP K3, 32k, max-num-seqs 8.
#
#   T0  control
#   T1  --mamba-ssm-cache-dtype float16
#       GDN recurrent state is ~151 MB/seq at fp32; at 8 seqs that is ~1.2 GB
#       and it is what bounds concurrency on this hybrid. Halving it is real
#       VRAM back -- IF quality holds.
#   T2  T1 + --enable-mamba-cache-stochastic-rounding
#       NVIDIA pairs fp16 state with stochastic rounding because per-token
#       recurrent quantization ACCUMULATES error across every token, unlike
#       attention KV where error stays per-token. So T1 without T2 is the
#       risky one and T2 is the intended configuration.
#   T3  --attention-backend TRITON_ATTN
#       FlashInfer's get_supported_kernel_block_sizes() returns [16,32,64]
#       unless a device-family-100 gate passes; sm_120 should fail it, yet our
#       logs show FlashInfer selected with a 1568-token block size. If
#       FlashInfer is genuinely constrained here, Triton is the correct backend.
MODEL="$HOME/models/Qwen3.8-27B-NVFP4"
PY="$HOME/qwen38/.venv/bin/python"
export PATH="$HOME/qwen38/.venv/bin:$HOME/.local/bin:$PATH"
PORT=8241

arm() { # $1=tag  rest=extra flags
  local tag="$1"; shift
  local log="$HOME/logs/lever_${tag}.log"
  pkill -f "vllm.entrypoints" 2>/dev/null; sleep 12
  read FREE TOTAL <<< $(nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits | tr ',' ' ')
  local UTIL=$(python3 -c "print(f'{max(0.80, ($FREE/$TOTAL)*0.97):.3f}')")
  echo "════════ $tag   $*"
  rm -f "$log"
  nohup "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name qwen38 --port $PORT \
    --max-model-len 32768 --gpu-memory-utilization "$UTIL" \
    --max-num-seqs 8 --kv-cache-dtype fp8 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --no-enable-log-requests "$@" > "$log" 2>&1 &
  for i in $(seq 1 360); do
    grep -q "Application startup complete" "$log" 2>/dev/null && break
    grep -qE "Engine core initialization failed" "$log" 2>/dev/null && {
      echo "   BOOT FAIL: $(grep -E 'ValueError:|RuntimeError:|NotImplementedError' "$log" | grep -v wait_for_engine | tail -1 | cut -c1-170)"; return 1; }
    sleep 5
  done
  for i in $(seq 1 60); do
    "$PY" -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/v1/models',timeout=5)" 2>/dev/null && break
    sleep 2
  done
  echo "   backend : $(grep -hoE 'Using [A-Z_]+ attention backend' "$log" | tail -1)"
  echo "   KV pool : $(grep -hE 'GPU KV cache size' "$log" | tail -1 | grep -oE '[0-9,]+ tokens')"
  echo "   maxconc : $(grep -hoE 'Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x' "$log" | tail -1)"
  # throughput
  "$PY" "$HOME/bench_fixed.py" --base-url "http://127.0.0.1:$PORT/v1" --model qwen38 \
    --prompt-tokens 256,2048 --concurrency 1 --max-tokens 200 --repeats 3 --warmup 1 \
    --label "$tag" --output "$HOME/results/lever_${tag}.json" 2>&1 | grep -E "^\s+[0-9]+\s+[0-9]"
  # correctness: needle + degeneration
  "$PY" "$HOME/longctx_test.py" --base-url "http://127.0.0.1:$PORT/v1" \
    --depths 4096,12000 --label "$tag" --out "$HOME/results/lever_lc_${tag}.json" 2>&1 | grep "d="
}

arm T0_control
arm T1_ssm_fp16      --mamba-ssm-cache-dtype float16
arm T2_ssm_fp16_sr   --mamba-ssm-cache-dtype float16 --enable-mamba-cache-stochastic-rounding
arm T3_triton        --attention-backend TRITON_ATTN
pkill -f "vllm.entrypoints" 2>/dev/null
echo "════════ levers done"
