#!/bin/bash
# MTP arms, attempt 2. Previous attempt killed two perfectly healthy servers:
# both logged "Application startup complete" and were then torn down because a
# curl-based health probe returned 000 for 60s. curl 8.5.0 is installed and no
# proxy vars are set, so the probe -- not the server -- was at fault.
#
# Changes: probe with the venv python urllib (same path the benchmark uses, and
# proven working in arm A), and NEVER tear down on a failed probe -- warn and
# benchmark anyway, letting the benchmark be the real verdict.
MODEL="$HOME/models/Qwen3.8-27B-NVFP4"
PY="$HOME/qwen38/.venv/bin/python"
BENCH="$HOME/bench_fixed.py"
OUT="$HOME/results"
export PATH="$HOME/qwen38/.venv/bin:$HOME/.local/bin:$PATH"
mkdir -p "$OUT" "$HOME/logs"

probe() {
  "$PY" - <<'PY' 2>/dev/null
import sys, urllib.request
try:
    with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
PY
}

stop_all() {
  pkill -f "vllm.entrypoints" 2>/dev/null || true
  for i in $(seq 1 60); do
    ss -ltn 2>/dev/null | grep -q ":8000 " || return 0
    sleep 2
  done
}

for arm in B C; do
  case $arm in
    B) k=2 ;;
    C) k=3 ;;
  esac
  log="$HOME/logs/mtp2_${arm}.log"
  echo "================ ARM $arm : MTP K$k ================"
  stop_all
  rm -f "$log"
  nohup "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name qwen38 \
    --max-model-len 32768 --gpu-memory-utilization 0.90 \
    --max-num-seqs 8 --kv-cache-dtype fp8 --port 8000 --no-enable-log-requests \
    --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$k}" > "$log" 2>&1 &

  ok=0
  for i in $(seq 1 240); do
    if grep -q "Application startup complete" "$log" 2>/dev/null; then ok=1; break; fi
    if grep -qE "Engine core initialization failed|CUDA out of memory" "$log" 2>/dev/null; then
      echo "  INIT FAILED"; grep -E "ValueError|RuntimeError" "$log" | tail -2; break; fi
    sleep 5
  done
  [ "$ok" = "1" ] || { echo "  skipping arm $arm"; continue; }

  # give the socket a moment, then probe -- but do NOT kill on failure
  for i in $(seq 1 15); do
    if probe; then echo "  PROBE OK"; break; fi
    sleep 2
  done
  probe || echo "  WARNING: probe failed but server reports ready -- benchmarking anyway"

  grep -hE "Available KV cache memory|GPU KV cache size|Maximum concurrency" "$log" | tail -3
  "$PY" "$BENCH" --base-url http://127.0.0.1:8000/v1 --model qwen38 \
    --prompt-tokens 256,2048,8192 --concurrency 1,4 \
    --max-tokens 200 --repeats 3 --warmup 1 \
    --label "arm${arm}" --output "$OUT/arm${arm}.json" 2>&1 | tail -12
  echo "--- spec decode stats ---"
  grep -hiE "acceptance|accepted|draft acceptance|num_accepted" "$log" | tail -4
done

stop_all
echo "================ DONE ================"
ls -l "$OUT"
