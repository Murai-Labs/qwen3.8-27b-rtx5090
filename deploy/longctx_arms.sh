#!/bin/bash
# 262,144-context arms.
#
#   L1  their recipe verbatim (4-bit KV, 5 GiB pin, gpu-util 0.98, NO MTP)
#   L2  L1 + MTP K3            <- the one that would beat them
#   L3  L2 but 3-bit KV        <- if MTP does not fit at 4-bit
#
# The question L2/L3 answer: the TurboQuant calibration log rejects MTP because
# it garbles output at 262144. We failed to reproduce that at 32768 with K3.
# This is where their claim gets its fair test -- same context they used.
MODEL="$HOME/models/Qwen3.8-27B-NVFP4"
PY="$HOME/qwen38/.venv/bin/python"
export PATH="$HOME/qwen38/.venv/bin:$HOME/.local/bin:$PATH"
PORT=8142
RES="$HOME/results"
mkdir -p "$RES" "$HOME/logs"

boot() {  # $1=name  rest=extra flags
  local name="$1"; shift
  local log="$HOME/logs/lc_${name}.log"
  pkill -f "vllm.entrypoints" 2>/dev/null; sleep 12
  rm -f "$log"
  nohup "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name qwen38 --port $PORT \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.94 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 512 \
    --no-enable-log-requests "$@" > "$log" 2>&1 &
  local ok=0
  for i in $(seq 1 300); do
    grep -q "Application startup complete" "$log" 2>/dev/null && { ok=1; break; }
    grep -qE "Engine core initialization failed|No valid attention backend|CUDA out of memory" "$log" 2>/dev/null && {
      echo "  [$name] INIT FAILED:"; grep -E "ValueError|RuntimeError|No valid attention" "$log" | tail -2 | cut -c1-200; return 1; }
    sleep 5
  done
  [ "$ok" = "1" ] || { echo "  [$name] TIMEOUT"; return 1; }
  for i in $(seq 1 60); do
    "$PY" -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/v1/models',timeout=5)" 2>/dev/null && { echo "  [$name] SERVING"; return 0; }
    sleep 2
  done
  echo "  [$name] listening but not serving"; return 1
}

facts() { grep -hE "Available KV cache memory|GPU KV cache size|Maximum concurrency|attention backend|Using .*backend" "$HOME/logs/lc_$1.log" 2>/dev/null | tail -4 | cut -c1-160; }

run_tests() {  # $1=label
  echo "  --- short-prompt garble check ---"
  BASE="http://127.0.0.1:$PORT/v1" "$PY" - <<'PYEOF'
import json,os,urllib.request
base=os.environ["BASE"]
for q in ["What is 2+2?","Name the capital of France in one word."]:
    b={"model":"qwen38","messages":[{"role":"user","content":q}],"max_tokens":200,
       "temperature":0,"chat_template_kwargs":{"preserve_thinking":False,"reasoning_effort":"low"}}
    try:
        r=urllib.request.Request(base+"/chat/completions",data=json.dumps(b).encode(),
                                 headers={"Content-Type":"application/json"})
        d=json.load(urllib.request.urlopen(r,timeout=300))
        m=d["choices"][0]["message"]; c=(m.get("content") or "")
        t=c.split(); w=1;cur=1
        for i in range(1,len(t)):
            cur=cur+1 if t[i]==t[i-1] else 1; w=max(w,cur)
        tail=c.rsplit("</think>",1)[-1].strip()[:60]
        print(f"    {q[:26]!r:30s} rep={w:2d} {'DEGENERATE' if w>8 else 'clean':11s} -> {tail!r}")
    except Exception as e:
        print(f"    {q[:26]!r:30s} ERROR {type(e).__name__}: {str(e)[:60]}")
PYEOF
  echo "  --- long-context recall + throughput ---"
  "$PY" "$HOME/longctx_test.py" --base-url "http://127.0.0.1:$PORT/v1" \
    --depths 8192,65536,131072 --label "$1" --out "$RES/longctx_$1.json" 2>&1 | tail -8
}

echo "================ L1: 4-bit KV, NO MTP (their recipe) ================"
if boot L1 --kv-cache-dtype turboquant_4bit_nc --kv-cache-memory-bytes 5368709120; then
  facts L1; run_tests l1-4bit-nospec
fi

echo "================ L2: 4-bit KV + MTP K3 ================"
if boot L2 --kv-cache-dtype turboquant_4bit_nc --kv-cache-memory-bytes 5368709120 \
     --speculative-config '{"method":"mtp","num_speculative_tokens":3}'; then
  facts L2; run_tests l2-4bit-mtp3
fi

echo "================ L3: 3-bit KV + MTP K3 ================"
if boot L3 --kv-cache-dtype turboquant_3bit_nc --kv-cache-memory-bytes 5368709120 \
     --speculative-config '{"method":"mtp","num_speculative_tokens":3}'; then
  facts L3; run_tests l3-3bit-mtp3
fi

pkill -f "vllm.entrypoints" 2>/dev/null
echo "================ LONG-CONTEXT ARMS DONE ================"
ls -l "$RES"/longctx_* 2>/dev/null
