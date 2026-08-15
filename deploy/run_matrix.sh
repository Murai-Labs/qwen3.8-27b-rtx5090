#!/bin/bash
# Boot + benchmark three arms in one long-lived process so WSL never idles out
# mid-run (it shut the VM down between calls three times otherwise).
#
# Arm A: baseline, no speculative decoding   <- the cheap baseline
# Arm B: + MTP K2
# Arm C: + MTP K3
#
# Everything else held constant. Attribution is only meaningful against A.

MODEL="$HOME/models/Qwen3.8-27B-NVFP4"
PY="$HOME/qwen38/.venv/bin/python"
BENCH="$HOME/bench_fixed.py"
OUT="$HOME/results"
export PATH="$HOME/qwen38/.venv/bin:$HOME/.local/bin:$PATH"
mkdir -p "$OUT" "$HOME/logs"

boot() {
  local name="$1"; shift
  local log="$HOME/logs/vllm_${name}.log"
  pkill -f "vllm.entrypoints" 2>/dev/null || true
  sleep 5
  rm -f "$log"
  nohup "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name qwen38 \
    --max-model-len 32768 --gpu-memory-utilization 0.90 \
    --max-num-seqs 64 --kv-cache-dtype fp8 \
    --port 8000 --no-enable-log-requests "$@" > "$log" 2>&1 &
  for i in $(seq 1 240); do
    if grep -q "Application startup complete" "$log" 2>/dev/null; then echo "  [$name] READY"; return 0; fi
    if grep -qE "Engine core initialization failed|CUDA out of memory" "$log" 2>/dev/null; then
      echo "  [$name] FAILED"; grep -E "ValueError|RuntimeError|Error:" "$log" | tail -3; return 1; fi
    sleep 5
  done
  echo "  [$name] TIMEOUT"; return 1
}

facts() {
  local log="$HOME/logs/vllm_$1.log"
  grep -hE "Available KV cache memory|GPU KV cache size|Maximum concurrency|speculative|Speculative" "$log" 2>/dev/null | tail -6
}

for arm in A B C; do
  case $arm in
    A) desc="baseline (no spec decode)"; extra=() ;;
    B) desc="MTP K2"; extra=(--speculative-config '{"method":"mtp","num_speculative_tokens":2}') ;;
    C) desc="MTP K3"; extra=(--speculative-config '{"method":"mtp","num_speculative_tokens":3}') ;;
  esac
  echo "================ ARM $arm : $desc ================"
  if boot "$arm" "${extra[@]}"; then
    facts "$arm"
    "$PY" "$BENCH" \
      --base-url http://127.0.0.1:8000/v1 --model qwen38 \
      --prompt-tokens 256,2048,8192 --concurrency 1,4 \
      --max-tokens 200 --repeats 3 --warmup 1 \
      --label "arm${arm}" --output "$OUT/arm${arm}.json" 2>&1 | tail -14
  else
    echo "  arm $arm skipped"
  fi
done

pkill -f "vllm.entrypoints" 2>/dev/null || true
echo "================ ALL ARMS DONE ================"
ls -l "$OUT"
