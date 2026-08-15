# [Bug]: MTP speculative decoding produces repetition collapse with `turboquant_*` KV cache on sm120 (Qwen3.8-27B hybrid GDN)

### Summary

On an RTX 5090 (sm_120), `Qwen3.8-27B-NVFP4` (`model_type: qwen3_5`, hybrid Gated DeltaNet)
serves cleanly with MTP speculative decoding when the KV cache is `fp8`, and degenerates into
repetition collapse when the KV cache is `turboquant_4bit_nc` or `turboquant_3bit_nc`.

The failure is **silent** — the server boots cleanly, reports a healthy KV pool, answers short
prompts correctly, and only collapses on longer prompts. Output is a single token repeated
~230 times with the answer present in neither `content` nor `reasoning_content`.

### Reproduce

```bash
python -m vllm.entrypoints.openai.api_server \
  --model unsloth/Qwen3.8-27B-NVFP4 \
  --max-model-len 262144 --gpu-memory-utilization 0.93 \
  --max-num-seqs 4 --max-num-batched-tokens 2048 \
  --kv-cache-dtype turboquant_4bit_nc --kv-cache-memory-bytes 6442450944 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Send any prompt of roughly 8k tokens. Short prompts ("What is 2+2?") answer correctly, which
is what makes this easy to miss.

### Results

Held constant: model, MTP K3, `--max-num-seqs 4`. Varied: KV dtype, KV pin, and
`--max-num-batched-tokens`. Metric is the maximum count of a consecutively repeated token in
the response; needle recall on the same prompt.

| KV dtype | pin | mnbt | max consecutive repeat | needle |
|---|---|---|---|---|
| `fp8` | auto | 2048 | **1** | found |
| `turboquant_4bit_nc` | 6 GiB | 512 | 232 | not found |
| `turboquant_4bit_nc` | 6 GiB | 2048 | 230 | not found |
| `turboquant_4bit_nc` | 6 GiB | 4096 | 234 | not found |
| `turboquant_4bit_nc` | 5 GiB | 512 | 224 | not found |
| `turboquant_3bit_nc` | 4 GiB | 512 | 232 | not found |
| `turboquant_3bit_nc` | 5 GiB | 512 | 232 | not found |

12/12 runs with `turboquant_*` KV degenerate; every `fp8` run is clean.

`num_speculative_tokens: 1` with `turboquant_*` KV does not degenerate — it crashes the engine
(`HTTP 500` / `EngineDeadError`), 4/4 runs.

### Ruled out

- **`--max-num-batched-tokens` below the mamba block size** (#51562 / #51483). The attention
  block size here is 3120; the collapse is identical at mnbt 512, 2048 and 4096, including
  above the block size where that precondition does not hold.
- **KV pin size** — reproduced at 4, 5 and 6 GiB.
- **Memory pressure** — the server reports a healthy KV pool (310,789 tokens at the 6 GiB pin)
  and does not OOM.

### Not degenerate: `turboquant_*` KV *without* MTP

`turboquant_4bit_nc` with no `--speculative-config` answers correctly (needle recall passes at
8k, `max consecutive repeat = 1`). So neither MTP alone nor TurboQuant KV alone reproduces
this — only the combination.

### Possibly relevant

The checkpoint declares `kv_cache_quant_algo: FP8` and ships KV calibration scales;
overriding `--kv-cache-dtype` presumably discards them. The MTP draft head reads the same KV
as the target model, so draft/target divergence would compound. This is a hypothesis, not
something I have isolated.

Worth noting that llama.cpp reportedly runs MTP with `q4_0` KV on this model at 160k context
without degeneration, which suggests the interaction is specific to this path rather than
inherent to MTP with quantized KV.

### Environment

- RTX 5090, sm_120, driver 610.74, CUDA 13.0
- vLLM 0.27.1, torch 2.13.0+cu130, Python 3.12, WSL2 Ubuntu 24.04
- Model: `unsloth/Qwen3.8-27B-NVFP4` (compressed-tensors, mixed NVFP4 W4A4 + FP8)
- `Using TURBOQUANT attention backend out of potential backends: ['TURBOQUANT']`
- GDN prefill: `Using Triton/FLA GDN prefill kernel (requested=auto, head_k_dim=128)`

Reproduction scripts and raw per-probe JSON:
https://github.com/Murai-Labs/qwen3.8-27b-rtx5090
