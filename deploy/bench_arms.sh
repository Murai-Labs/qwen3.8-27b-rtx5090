#!/bin/bash
# Benchmark the speculative-decoding arms against a common baseline.
#
#   Arm A: no speculative decoding   <- the baseline everything is measured against
#   Arm B: MTP K2
#   Arm C: MTP K3
#
# Everything else is held constant. Attribution is only meaningful against A.
#
# Two things this script does deliberately, both learned the hard way:
#   - it probes readiness with python urllib, not curl (a curl probe returned 000
#     against a healthy server here), and
#   - it NEVER tears down on a failed probe. An earlier version killed two
#     perfectly good servers that had logged "Application startup complete"
#     because the socket was not accepting yet. The benchmark is the verdict.
MODEL="${MODEL:-$HOME/models/Qwen3.8-27B-NVFP4}"
VENV="${VENV:-$HOME/qwen38/.venv}"
PY="$VENV/bin/python"
BENCH="${BENCH:-$HOME/bench_fixed.py}"
OUT="${OUT:-$HOME/results}"
export PATH="$VENV/bin:$HOME/.local/bin:$PATH"
mkdir -p "$OUT" "$HOME/logs"

probe() {
  "$PY" -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/v1/models',timeout=5)" 2>/dev/null
}

stop_all() {
  pkill -f "vllm.entrypoints" 2>/dev/null || true
  for i in $(seq 1 60); do
    ss -ltn 2>/dev/null | grep -q ":8000 " || return 0
    sleep 2
  done
}

for arm in A B C; do
  case $arm in
    A) desc="baseline"; spec=() ;;
    B) desc="MTP K2";   spec=(--speculative-config '{"method":"mtp","num_speculative_tokens":2}') ;;
    C) desc="MTP K3";   spec=(--speculative-config '{"method":"mtp","num_speculative_tokens":3}') ;;
  esac
  log="$HOME/logs/arm_${arm}.log"
  echo "================ ARM $arm : $desc ================"
  stop_all
  rm -f "$log"
  nohup "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name qwen38 \
    --max-model-len 32768 --gpu-memory-utilization 0.90 \
    --max-num-seqs 8 --kv-cache-dtype fp8 \
    --port 8000 --no-enable-log-requests "${spec[@]}" > "$log" 2>&1 &

  ok=0
  for i in $(seq 1 240); do
    grep -q "Application startup complete" "$log" 2>/dev/null && { ok=1; break; }
    grep -qE "Engine core initialization failed|CUDA out of memory" "$log" 2>/dev/null && {
      echo "  INIT FAILED"; grep -E "ValueError|RuntimeError" "$log" | tail -2; break; }
    sleep 5
  done
  [ "$ok" = "1" ] || { echo "  skipping arm $arm"; continue; }

  # generous socket wait -- readiness lags "startup complete" by a variable margin
  for i in $(seq 1 90); do probe && { echo "  SOCKET READY after $((i*2))s"; break; }; sleep 2; done
  probe || echo "  WARNING: probe still failing; benchmarking anyway"

  grep -hE "Available KV cache memory|GPU KV cache size|Maximum concurrency" "$log" | tail -3
  "$PY" "$BENCH" --base-url http://127.0.0.1:8000/v1 --model qwen38 \
    --prompt-tokens 256,2048,8192 --concurrency 1,4 \
    --max-tokens 200 --repeats 3 --warmup 1 \
    --label "arm${arm}" --output "$OUT/arm${arm}.json" 2>&1 | tail -12
done

stop_all
echo "================ DONE ================"
ls -l "$OUT"
