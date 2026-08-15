#!/bin/bash
# Boot 2: same minimal config, but log to $HOME (survives WSL restarts, /tmp is
# cleaned on boot) and capture the real traceback from KV-cache init.
MODEL="$HOME/models/Qwen3.8-27B-NVFP4"
LOG="$HOME/logs/vllm_boot2.log"
PY="$HOME/qwen38/.venv/bin/python"
mkdir -p "$HOME/logs"

# vLLM JIT-compiles a kernel during the dummy sampler run and shells out to
# `ninja`. It is installed in the venv but the venv's bin is not on PATH when
# the interpreter is invoked by absolute path -- without this the engine dies
# with FileNotFoundError: 'ninja' during _initialize_kv_caches.
export PATH="$HOME/qwen38/.venv/bin:$HOME/.local/bin:$PATH"

pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 3
rm -f "$LOG"

nohup "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name qwen38 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 64 \
  --port 8000 \
  --no-enable-log-requests \
  > "$LOG" 2>&1 &
PID=$!
echo "pid=$PID log=$LOG"

for i in $(seq 1 180); do
  if grep -q "Application startup complete" "$LOG" 2>/dev/null; then echo "=== READY ==="; break; fi
  # only real fatals -- not transformers' [ERROR] docstring warnings
  if grep -qE "Engine core initialization failed|CUDA out of memory|^Traceback" "$LOG" 2>/dev/null; then echo "=== FAILED ==="; break; fi
  if ! kill -0 $PID 2>/dev/null; then echo "=== PROCESS EXITED ==="; break; fi
  sleep 5
done

echo "===== KV CACHE / MEMORY FACTS ====="
grep -iE "model weights took|Memory profiling|available KV cache|GPU KV cache size|maximum concurrency|block size|page size|free_memory|non_torch|torch activation" "$LOG" 2>/dev/null | head -20

echo "===== ROOT CAUSE ====="
awk '/Traceback \(most recent call last\)/{f=1} f{print}' "$LOG" 2>/dev/null | grep -vE "^\(APIServer" | head -40

echo "===== FINAL EXCEPTION LINES ====="
grep -E "Error|error|raise |assert" "$LOG" 2>/dev/null | tail -15
