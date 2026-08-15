# Qwen3.8-27B on a single RTX 5090

Serving **Qwen3.8-27B** (NVFP4) on one **RTX 5090** (Blackwell, sm_120, 32 GB) under vLLM
0.27.1 in WSL2, tuned for decode throughput without giving up quality.

Every number here was measured on this machine. Computed figures say so. Things we did not
measure say so.

## TL;DR

| | |
|---|---|
| **Decode** | **107.6 tok/s** @ p256, **97.3** @ p2048 — ~2× the no-speculation baseline |
| **Quality cost of that 2×** | **zero** — same 194/200 GSM8K items correct as without it, McNemar p=1.0 |
| **The flag that matters** | `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` |
| **The flag that breaks it** | any `--kv-cache-dtype` override. The checkpoint ships **fp8 KV calibration scales** |
| **Long context** | **124,326-token** recall verified (fp8 KV, no MTP); **~110k** ceiling with MTP |
| **Multi-turn** | send `"chat_template_kwargs": {"preserve_thinking": false}` |
| **Concurrency limit** | bounded by **Mamba cache blocks**, not KV — one per decode sequence |

## The recipe

```bash
python -m vllm.entrypoints.openai.api_server \
  --model unsloth/Qwen3.8-27B-NVFP4 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --kv-cache-dtype fp8 \
  --max-num-seqs 8 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3
```

Client side, on **every multi-turn request**:

```json
{"chat_template_kwargs": {"preserve_thinking": false}}
```

[`deploy/serve.sh`](deploy/serve.sh) carries the reasoning for each flag inline.

Three settings in priority order: **MTP K3** (~2× decode, free), **`preserve_thinking: false`**
(multi-turn correctness and prefix-cache stability), **fp8 KV** (~2× context — and do not
override it to anything else).

## Measured throughput

`bench/bench_fixed.py`, 3 repeats, medians. Every cell emitted exactly 200 tokens
(`min_tokens` + `ignore_eos` verified), random nonce so prefill is cold.

| prompt | conc | baseline | MTP K3 | Δ |
|---|---|---|---|---|
| 256 | 1 | 51.5 | **107.6** | **+108.8%** |
| 256 | 4 | 46.9 | **108.4** | +131.0% |
| 2048 | 1 | 52.5 | **97.3** | +85.2% |
| 2048 | 4 | 45.0 | **97.1** | +116.0% |
| 8192 | 1 | 52.0 | **98.2** | +88.7% |
| 8192 | 4 | 38.7 | **70.2** | +81.5% |

Baseline decode is flat across prompt length (51.5 / 52.5 / 52.0) — bandwidth-bound, which
the hybrid architecture explains below. Spread is tight at baseline (0.4–4.1) and much wider
with MTP (4.9–19.6), because acceptance depends on content: **a single MTP sample is not a
reliable number.**

Memory at `--gpu-memory-utilization 0.90`, 32k, fp8 KV: **4.47 GiB KV → 126,976 tokens**,
max concurrency 3.88×.

## Measured quality

`bench/quality_eval.py` + `bench/quality_compare.py`, GSM8K from the local HF cache, scored
exact-match. Paired design with **McNemar's exact test** — at n=200 two independent runs can
differ by 4 points with fully overlapping CIs and tell you nothing; the items that *flip* are
the signal.

| | baseline | MTP K3 |
|---|---|---|
| Accuracy | **97.0%** (95% CI 93.6–98.6) | **97.0%** |
| correct / incorrect / no_answer | 194 / 6 / **0** | 194 / 6 / **0** |
| Wall clock, same 200 items | **264 s** | **112 s** |

Paired: **both correct 194, only-baseline 0, only-MTP 0, both wrong 6. p = 1.0000.**

**Zero discordant items** — the same 194 right and the same 6 wrong, not merely the same
total. That is what correct speculative decoding should do: drafts are verified against the
target model and rejections resampled, so at `temperature=0` it is lossless. The 2.36×
wall-clock difference on a real workload also confirms the synthetic benchmark independently.

The 6 misses were read individually rather than trusted as a total; all are genuine
arithmetic errors, not extraction artifacts. Caveats: `temperature=0`, one task, n=200, and
equality of outcomes is not equality of tokens.

## The finding: never override KV dtype on this checkpoint

The checkpoint declares `kv_cache_quant_algo: FP8` and **ships calibration scales for it**
([SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B)).
Overriding `--kv-cache-dtype` discards those scales. Speculative decoding is maximally
exposed, because the MTP draft head reads the same KV as the target — divergence compounds
into repetition collapse.

| KV dtype | Calibrated | MTP result | runs |
|---|---|---|---|
| **fp8** | **yes** | **lossless** | 200 paired, 0 discordant |
| turboquant 4-bit | no | **garbled**, rep=224 | 2/2 |
| turboquant 3-bit | no | **garbled**, rep=232 | 4/4, two pin sizes |
| turboquant, MTP K1 | no | **engine crash** (HTTP 500) | 4/4 |

Degeneration means a single token repeated 224–234 times with the answer in neither `content`
nor `reasoning`. **These configs passed a short-prompt check moments before collapsing on an
8k prompt** — short probes are not evidence.

**The obvious alternative explanation was tested and eliminated.** vLLM
[#51562](https://github.com/vllm-project/vllm/issues/51562) reports that when
`--max-num-batched-tokens` falls below the mamba block size, a request can read another's
leftover recurrent state — silent garbage, not a crash. Every one of our garbling runs used
mnbt 512 (the TurboQuant recipe's value) and every clean run used the 2048 default, so this
was a live confound. Holding KV dtype and MTP fixed and varying only mnbt:

| mnbt | attention block size | result |
|---|---|---|
| 512 | 3120 | DEGENERATE rep=232, 2/2 |
| 2048 | 3120 | DEGENERATE rep=230, 2/2 |
| **4096** (above the block size) | 3120 | DEGENERATE rep=234, 2/2 |

Identical collapse at all three, including with mnbt **above** the block size where the
#51562 precondition does not hold. The KV dtype is the variable. Total evidence: **12/12
garbled runs** across 3 mnbt values, 2 KV widths and 2 pin sizes.

**Important counter-evidence:** [AtomicChat](https://huggingface.co/AtomicChat/Qwen3.8-27B-GGUF)
report MTP running fine with `q4_0` KV at 160k on **llama.cpp**. If quantized KV broke MTP
fundamentally, that should fail too. So this is more likely a **vLLM turboquant-path bug**
than an inherent incompatibility — worth reporting upstream. Either way, on vLLM today:
leave KV at fp8.

## Long context

| Config | KV pool | Result |
|---|---|---|
| **fp8 KV + MTP K3, 100k** | — | **4/4 needle depths PASS, rep=1 at every depth, up to an 85,440-token prompt** |
| fp8 KV, no MTP, 131k | 186,815 tok | needle recall PASS at a **124,326-token** prompt |
| fp8 KV + MTP | — | **~110k ceiling** (131k needs 4.88 GiB vs 4.34 available; 112k needs 4.25 vs 4.16) |
| turboquant 4-bit, no MTP, 262k | 306,325 tok | 8k PASS at 22.8 tok/s; **64k OOMs** |

**MTP does not garble at long context with fp8 KV.** The TurboQuant calibration log rejects
MTP partly on the grounds that it garbles at 262k *including* with fp8 KV. We tested exactly
that — MTP K3, fp8 KV, prompts from 7,898 to 85,440 tokens — and got clean recall with
`max_consecutive_repeat = 1` at every depth. Their observation is real (we reproduced it), but
it does not generalise to fp8 KV.

**262k is reachable in principle and not on a Windows desktop.** The
[TurboQuant recipe](https://github.com/ayayalar/Qwen3.8-27B-NVFP4-TurboQuant) reaches it with
`gpu-memory-utilization 0.98`; the Windows shell holds ~1.4 GiB of VRAM here, capping us at
~0.94. Activation headroom after weights and a 5 GiB KV pin is ~2.8 GiB for us against ~4.1
for them, and that gap OOMs at 64k prefill. A headless Linux box would likely reproduce it.

Weights are the other lever: NVFP4 is 22.1 GiB because it keeps attention and the last 8
layers' MLPs at FP8. A GGUF Q4_K of the same model is ~15.9 GiB — 6 GiB lighter, which is
worth ~190k tokens of fp8 KV, at the cost of FP4 tensor cores.

## Why this model is not a normal 27B

`model_type: qwen3_5`, and the architecture drives every sizing decision:

| Property | Value | Consequence |
|---|---|---|
| Layer layout | 16 × [3 × Gated DeltaNet → 1 × Gated Attention] | only **16 of 64** layers hold a growing KV cache |
| `head_dim` | **256** | unusually large; doubles per-layer KV vs 128 |
| KV heads | 4 (GQA 24/4) | — |
| **KV per token** | **32.8 KB** at fp8 | computed from config, confirmed by SGLang and by vLLM's allocation |
| MTP head | ships in-checkpoint (`model_mtp.safetensors`) | speculative decoding with no draft model |
| Context | 262,144 native | — |
| Modality | vision + video + text | vision tower left unquantized |

Because 48 of 64 layers use fixed-size recurrent state, **long context costs KV, not decode
speed** — which is why baseline decode is flat from 256 to 8192 tokens. It also means
**concurrency is bounded by Mamba state, not KV**: each decode sequence consumes one Mamba
block, and vLLM refuses to start if `max_num_seqs` exceeds the available blocks.

The checkpoint is mixed-precision by design: NVFP4 W4A4 on most MLPs, **FP8 W8A8** on all
attention projections, `lm_head`, and the MLPs of the **last 8 layers**; the vision tower is
unquantized. Native FP4 is verified engaged — the startup log selects
`CutlassNvFp4LinearKernel` / `CompressedTensorsW4A4Fp4`, with **no Marlin fallback**, so
[vLLM #47749](https://github.com/vllm-project/vllm/issues/47749) does not bite here. Read the
kernel selection directly; a silent fallback still serves tokens.

## The chat template poisons multi-turn conversations

The stock template renders an **empty `<think></think>` on every prior assistant turn**,
because `reasoning_content` falls back to `''` (clients replay only `content`) while
`preserve_thinking` defaults to true. A 6-turn conversation produces 6 empty blocks
(`bench/template_test.py`).

| Rendering | `<think>` tags | Empty blocks |
|---|---|---|
| default | 7 | **6** |
| `preserve_thinking=false` | 1 | **0** |
| [froggeric community template](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates), defaults | 1 | **0** |

Stock + the kwarg renders **byte-identically** to the community template, so no swap is
needed. Both keep the generation prompt intact — only the poisoned history goes.

**What this does not fix:** we originally framed the empty blocks as the *cause* of multi-turn
truncation. A third-party behavioural audit A/B tested both templates over the same 12-turn
session and found both truncate at the same turn, attributing it to context exhaustion from
the model's own 8–12k character answers — with `finish_reason: length` after ~2,900 of 32,768
tokens, which misleads you into raising `max_tokens` when the constraint is remaining context.
Set `preserve_thinking: false` for prompt hygiene and prefix-cache stability, not as a
truncation fix.

### `reasoning_effort=medium` is a silent no-op

The template validates `medium` then has branches only for `xhigh` and `low`, so it drops the
reasoning instruction entirely — **less** steering than the default, with no error:

| value | system prompt | instruction |
|---|---|---|
| unset / `xhigh` / `high` | 297 chars | xhigh (the default) |
| **`medium`** | **60 chars** | **none** |
| `low` | 226 chars | low |

## Optional: fp16 SSM state (+18.6% context, quality unproven)

`--mamba-ssm-cache-dtype float16` halves the GDN recurrent state, which is what bounds
concurrency on this hybrid:

| | fp32 SSM (default) | fp16 SSM |
|---|---|---|
| KV pool @32k | 83,409 tok | **98,934 tok (+18.6%)** |
| Max concurrency | 2.55x | **3.02x** |
| Decode p256 / p2048 | 103.8 / 91.1 | 99.0 / 96.6 (within spread) |
| GSM8K, n=200 paired | 97.0% | 96.5% |
| Paired vs control | — | both 193, **only-control 1, only-fp16 0**, p=1.0 |

vLLM warns that this overrides the model's declared `mamba_ssm_dtype='float32'`. One item
flipped, against fp16. **Contrast with MTP, which had zero discordant items across 200** —
that was a positive demonstration of losslessness; this is only an absence of detectable harm
at n=200. Recurrent-state error *accumulates across tokens* (unlike KV error, which stays
per-token), which is the shape early degradation would take. Offered as an option for
context-bound workloads, **not promoted to the default.**

`--enable-mamba-cache-stochastic-rounding` — the pairing NVIDIA recommends for fp16 state —
crashed the server ~32 s after startup here (`Cannot close a running event loop`), untested
further.

### `--attention-backend` is silently ignored

Both `--attention-backend TRITON_ATTN` and `VLLM_ATTENTION_BACKEND=TRITON_ATTN` were accepted
and had no effect; vLLM logs `potential backends: ['FLASHINFER', 'TRITON_ATTN']` and then
selects FlashInfer regardless. Same failure pattern as `reasoning_effort=medium`: a setting
accepted and then ignored. All measurements here are therefore on FlashInfer, which is what
this model actually gets on sm_120.

## Gotchas

| Symptom | Cause | Fix |
|---|---|---|
| `error: unrecognized arguments: --disable-log-requests` | renamed in 0.27.1 | `--no-enable-log-requests` |
| Engine dies in `_initialize_kv_caches`: `FileNotFoundError: 'ninja'` | vLLM JIT-compiles a kernel in the dummy sampler run and shells out to `ninja`; it is in the venv but not on `PATH` | `export PATH="$VENV/bin:$PATH"` |
| `max_num_seqs (256) exceeds available Mamba cache blocks (80)` | hybrid-model specific: one Mamba block per decode sequence | `--max-num-seqs` ≤ reported blocks |
| **Server logs `HTTP server started`, then refuses every connection** | **`networkingMode=mirrored` in `.wslconfig`.** The socket shows `TCP_LISTEN` in `/proc/net/tcp` and `connect()` still returns `ECONNREFUSED`, from WSL *and* Windows | **`networkingMode=nat` + `wsl --shutdown`.** Cost eight runs before being isolated; plain loopback and trivial servers work fine, which is what makes it so misleading |
| `gpu-memory-utilization 0.98` won't boot | Windows shell holds ~1.4 GiB VRAM (`explorer`, `SearchHost`, `dwm`) | compute it from live `nvidia-smi` free memory — it drifts as apps open |
| `--max-num-batched-tokens 16384` starves KV | larger batched-token budgets take activation memory out of KV | that value comes from DGX Spark recipes with 128 GB unified memory; it does not transfer to 32 GB |
| MTP silently downgrades cudagraphs | `FULL_AND_PIECEWISE is not supported with spec-decode for FlashInferBackend` | informational; our MTP numbers are on PIECEWISE |

## How we compare

Three independent points of reference, all on the same model:

| Setup | Context | MTP | Decode |
|---|---|---|---|
| 1× DGX Spark, NVFP4, vLLM ([Spark Arena](https://spark-arena.com), Saiyam Pathak) | — | ✗ | **11.48 tok/s** |
| 4× DGX Spark TP=4, FP8, vLLM ([Spark Arena](https://spark-arena.com), Drew Botwinick) | — | K3 | 39.17 tok/s |
| RTX 5090, NVFP4, vLLM ([TurboQuant](https://github.com/ayayalar/Qwen3.8-27B-NVFP4-TurboQuant)) | 262k | ✗ | ~55 tok/s |
| RTX 4090, GGUF Q4_K, llama.cpp ([AtomicChat](https://huggingface.co/AtomicChat/Qwen3.8-27B-GGUF)) | 160k | ✓ | ~60 tok/s |
| **This repo — RTX 5090, NVFP4, vLLM** | ~110k | **✓** | **97–108 tok/s** |

**~4.5× per node against a single DGX Spark**, comparing our baseline (51.5) to theirs
(11.48), both without speculative decoding — the defensible like-for-like number. The TurboQuant
figures (~55 single-stream, ~197 aggregate at 4-way) independently cross-validate our
baseline (51.5–52.5, 180.9) within ~7%, two testers on the same hardware.

Caveat on all cross-comparisons: different harnesses, output lengths and metric definitions.
On Spark Arena's own raw CSV, `t_s_req_mean` and `peak_ts_req_mean` differ by ~20% — the
"decode tok/s means three different things" trap. Treat orderings as solid and multiples as
approximate.

Worth stealing from the others: `--chunked-prefill-size 2048` (SGLang: 8192-token chunks stall
decode ~600 ms on hybrid GDN models; TurboQuant's 512 is on the other side of the optimum),
`--enable-prefix-caching` (we ran without it), and `--mamba-radix-cache-strategy
extra_buffer_lazy` (5 state slots per request → 4, "at no accuracy cost").

## Reproducing

```bash
# environment (WSL2 Ubuntu 24.04)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 && uv pip install vllm==0.27.1 pyarrow

# weights (~22 GB)
hf download unsloth/Qwen3.8-27B-NVFP4

bash deploy/serve.sh                       # serve
bash deploy/bench_arms.sh                  # throughput arms
python bench/quality_eval.py --task gsm8k --n 200 --label mine --out q.json
python bench/quality_compare.py a.json b.json   # paired McNemar
python bench/template_test.py              # empty think blocks
python bench/longctx_test.py --depths 4096,32768 --label mine
```

Copy weights onto ext4 rather than reading from `/mnt/<drive>`: drvfs measured 244 MB/s here,
~92 s of load time per start.

**Measurement methodology** — decode tok/s means at least three different things, and prefix
caching silently inflates prefill. The traps are documented in the companion repo's
[MEASUREMENT-NOTES.md](https://github.com/Murai-Labs/dgx-spark-x-2-upstream/blob/main/docs/MEASUREMENT-NOTES.md);
the harness here is the same one.

**Not measured:** concurrency above 4; MTP K2 (will not serve here, 4 attempts); quality
beyond GSM8K; whether MTP stays lossless at long context.

## Licence

MIT — see [LICENSE](LICENSE). Model weights are not included and carry their own licence
(Apache-2.0 for Qwen3.8-27B). Attribution for third-party data and findings is in
[NOTICE](NOTICE).
