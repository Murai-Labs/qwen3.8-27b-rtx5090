#!/bin/bash
# First boot: minimal flags. Goal is NOT speed -- it is to read which
# quantization kernels vLLM selects for the mixed FP8/NVFP4 checkpoint on sm_120.
MODEL="$HOME/models/Qwen3.8-27B-NVFP4"
LOG=/tmp/vllm_boot1.log
PY="$HOME/qwen38/.venv/bin/python"

pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 2
rm -f "$LOG"

VLLM_LOGGING_LEVEL=DEBUG nohup "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name qwen38 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --port 8000 \
  --no-enable-log-requests \
  > "$LOG" 2>&1 &

echo "pid=$! log=$LOG"
# wait up to 10 min for readiness or failure
for i in $(seq 1 200); do
  if grep -q "Application startup complete" "$LOG" 2>/dev/null; then echo "READY after ${i}x3s"; break; fi
  if grep -qE "Traceback|ERROR|error loading|ValueError|RuntimeError" "$LOG" 2>/dev/null; then echo "ERROR DETECTED after ${i}x3s"; break; fi
  sleep 3
done

echo "===== QUANTIZATION / KERNEL SELECTION ====="
grep -iE "quant|nvfp4|fp4|marlin|cutlass|compressed|scheme|w4a4|w8a8|fp8" "$LOG" | head -40
echo "===== ATTENTION / KV ====="
grep -iE "attention backend|kv cache|kv_cache|blocks|linear_attn|mamba|hybrid" "$LOG" | head -20
echo "===== ERRORS ====="
grep -iE "Traceback|ValueError|RuntimeError|not supported|fallback|falling back" "$LOG" | head -20
echo "===== TAIL ====="
tail -15 "$LOG"
