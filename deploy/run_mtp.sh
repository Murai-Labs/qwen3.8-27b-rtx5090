#!/bin/bash
# Re-run the MTP arms only. Fixes from the first attempt:
#   1. wait for port 8000 to actually be FREE before launching the next server
#      (previous run reported "startup complete" then served nothing)
#   2. health-check /v1/models and require a real 200 before benchmarking
#   3. --max-num-seqs 8 -- MTP eats KV (127k -> 62k -> 39k tokens) and vLLM
#      warned max_num_scheduled_tokens fell to 2048 at max_num_seqs=64
MODEL="$HOME/models/Qwen3.8-27B-NVFP4"
PY="$HOME/qwen38/.venv/bin/python"
BENCH="$HOME/bench_fixed.py"
OUT="$HOME/results"
export PATH="$HOME/qwen38/.venv/bin:$HOME/.local/bin:$PATH"
mkdir -p "$OUT" "$HOME/logs"

stop_all() {
  pkill -f "vllm.entrypoints" 2>/dev/null || true
  for i in $(seq 1 60); do
    if ! ss -ltn 2>/dev/null | grep -q ":8000 "; then return 0; fi
    sleep 2
  done
  echo "  WARNING: port 8000 still bound after 120s"
}

boot() {
  local name="$1"; shift
  local log="$HOME/logs/mtp_${name}.log"
  stop_all
  rm -f "$log"
  nohup "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name qwen38 \
    --max-model-len 32768 --gpu-memory-utilization 0.90 \
    --max-num-seqs 8 --kv-cache-dtype fp8 \
    --port 8000 --no-enable-log-requests "$@" > "$log" 2>&1 &
  for i in $(seq 1 240); do
    if grep -q "Application startup complete" "$log" 2>/dev/null; then break; fi
    if grep -qE "Engine core initialization failed|CUDA out of memory" "$log" 2>/dev/null; then
      echo "  [$name] INIT FAILED"; grep -E "ValueError|RuntimeError" "$log" | tail -2; return 1; fi
    sleep 5
  done
  # real health check -- startup-complete alone proved insufficient last time
  for i in $(seq 1 30); do
    code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/v1/models 2>/dev/null)
    if [ "$code" = "200" ]; then echo "  [$name] SERVING (health 200)"; return 0; fi
    sleep 2
  done
  echo "  [$name] NOT SERVING (last health code: $code)"; return 1
}

for arm in B C; do
  case $arm in
    B) k=2 ;;
    C) k=3 ;;
  esac
  echo "================ ARM $arm : MTP K$k, max_num_seqs=8 ================"
  if boot "$arm" --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$k}"; then
    grep -hE "Available KV cache memory|GPU KV cache size|Maximum concurrency" "$HOME/logs/mtp_${arm}.log" | tail -3
    "$PY" "$BENCH" --base-url http://127.0.0.1:8000/v1 --model qwen38 \
      --prompt-tokens 256,2048,8192 --concurrency 1,4 \
      --max-tokens 200 --repeats 3 --warmup 1 \
      --label "arm${arm}" --output "$OUT/arm${arm}.json" 2>&1 | tail -12
    echo "--- spec decode acceptance ---"
    grep -hiE "acceptance|accepted|draft" "$HOME/logs/mtp_${arm}.log" | tail -5
  fi
done

stop_all
echo "================ MTP ARMS DONE ================"
ls -l "$OUT"
