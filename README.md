# Qwen3.8-27B on a single RTX 5090

Serving **Qwen3.8-27B** (NVFP4) on one **NVIDIA GeForce RTX 5090** (Blackwell, **sm_120**,
32 GB) under vLLM in WSL2, tuned for decode throughput without giving up quality.

Everything here was measured on the machine described in [Environment](#environment).
Where a number is computed rather than measured, it says so. Where something is not yet
measured, it says that too.

> **Status: work in progress.** The environment and architecture analysis below are
> verified. The throughput and quality tables are **not yet populated** — they are marked
> `PENDING` rather than filled with estimates.

---

## Why this model is not a normal 27B

`Qwen3.8-27B` reports `model_type: qwen3_5` and is **not** a standard dense transformer.
Read the config before sizing anything:

| Property | Value | Why it matters |
|---|---|---|
| Layers | 64 | — |
| `full_attention_interval` | **4** | 3× linear attention, then 1× full attention, repeating |
| Full-attention layers | **16 of 64** | Only these carry a growing KV cache |
| Linear-attention layers | **48 of 64** | Gated-DeltaNet style; fixed-size recurrent state |
| `head_dim` | **256** | Unusually large — doubles per-layer KV vs a 128 head_dim |
| KV heads | 4 (GQA 6:1) | — |
| `mtp_num_hidden_layers` | **1** | Ships an MTP head → speculative decoding available |
| Context | 262,144 | Extendable via YaRN |
| Modality | Vision + video + text | Vision tower is left unquantized |

**The practical consequence is the KV budget.** With only 16 full-attention layers:

```
KV bytes/token = 2 (K,V) x 16 layers x 4 kv_heads x 256 head_dim x bytes_per_elem
               = 32,768 bytes/token at FP8
               = 65,536 bytes/token at BF16
```

| Context | KV at FP8 | KV at BF16 |
|---|---|---|
| 32k | 1.07 GB | 2.15 GB |
| 128k | 4.29 GB | 8.59 GB |
| 262k (max) | 8.59 GB | 17.2 GB |

*(computed from the config, not yet measured against vLLM's actual allocation)*

Against 32.6 GB of VRAM minus ~22.5 GB of weights, **128k context is the interesting
target on a single 5090**. Had this been a conventional dense 64-layer model, KV would be
4× larger and 32k would have been the ceiling.

## The checkpoint is mixed-precision by design

`unsloth/Qwen3.8-27B-NVFP4` is `compressed-tensors` with `format: mixed-precision` —
it is **not** uniformly FP4:

| Group | Precision | Applies to |
|---|---|---|
| `group_0` | **FP8 W8A8** (channel weights, per-token dynamic acts) | all `self_attn` q/k/v/o projections, `linear_attn` in/out projections, `lm_head`, **and the MLPs of the last 8 layers (56–63)** |
| `group_1` | **NVFP4 W4A4** (`tensor_group`, group_size 16) | MLP gate/up/down of every other layer |
| ignored | unquantized | the entire vision tower (303 entries) |

Keeping the final 8 layers' MLPs at FP8 is a deliberate quality choice, and it is why the
checkpoint is 22.5 GB rather than the ~14 GB a uniform FP4 27B would be.

**This mixed shape is also the main risk.** vLLM has a history of mixed NVFP4 checkpoints
on sm_120 silently falling back to the Marlin W4A16 path instead of native FP4
([vllm#47749](https://github.com/vllm-project/vllm/issues/47749)), which still *runs* — just
slower. Verifying which kernels actually get selected is step one of this recipe, and it is
done by reading vLLM's startup log, not by inferring from throughput.

### The FP4 path does engage — verified

Read from the vLLM startup log at `VLLM_LOGGING_LEVEL=DEBUG`, not inferred from speed:

```
INFO   Using CutlassNvFp4LinearKernel for NVFP4 GEMM
DEBUG  Using scheme: CompressedTensorsW4A4Fp4  for ...mlp.gate_up_proj
DEBUG  Using scheme: CompressedTensorsW4A4Fp4  for ...mlp.down_proj
INFO   Selected CutlassFP8ScaledMMLinearKernel for CompressedTensorsW8A8Fp8
DEBUG  Using scheme: CompressedTensorsW8A8Fp8  for ...self_attn.qkv_proj
DEBUG  Using scheme: CompressedTensorsW8A8Fp8  for ...linear_attn.in_proj_qkvz
```

**No Marlin anywhere.** Native CUTLASS NVFP4 W4A4 on the MLPs and CUTLASS FP8 W8A8 on
attention — exactly the split the checkpoint declares. The #47749 fallback does not occur
with vLLM 0.27.1 + torch 2.13.0+cu130 on sm_120.

If you take one thing from this repo, take the method rather than the result: a fallback
still serves tokens, so it is invisible unless you read the kernel selection directly.

## Gotchas hit while bringing this up

Each of these cost a boot cycle. All are verified on this machine.

| Symptom | Cause | Fix |
|---|---|---|
| `error: unrecognized arguments: --disable-log-requests` | flag renamed in vLLM 0.27.1 | use `--no-enable-log-requests` |
| Engine dies in `_initialize_kv_caches` with `FileNotFoundError: 'ninja'` | vLLM JIT-compiles a kernel in the dummy sampler run and shells out to `ninja`; it is installed in the venv but the venv's `bin` is not on `PATH` when python is invoked by absolute path | `export PATH="$VENV/bin:$PATH"` before launching |
| `ValueError: max_num_seqs (256) exceeds available Mamba cache blocks (80)` | **hybrid-model specific** — each decode sequence consumes one Mamba cache block, so concurrency is bounded by Mamba state, not KV | `--max-num-seqs` ≤ the reported block count, or raise `--gpu-memory-utilization` |
| FlashAttention rejected, falls back to FlashInfer | `FP8 KV cache requires FA3 on SM90 or FA4 on SM100` — neither exists for sm_120 | expected; `TRITON_ATTN` is the other valid backend |
| Attention block size silently becomes 1568 tokens | vLLM pads it so the attention page size ≥ the Mamba page size, then pads Mamba by 0.13% to make them equal | informational, but it makes block-count arithmetic non-obvious |
| Server vanishes between commands | WSL2 shuts the VM down when idle, killing the server and clearing `/tmp` | log to `$HOME`, and run boot + benchmark inside **one** long-lived process |

## Measured memory (vLLM 0.27.1, `--gpu-memory-utilization 0.90`, 32k context, FP8 KV)

| Quantity | Value |
|---|---|
| Available KV cache memory | **4.56 GiB** |
| GPU KV cache size | **129,706 tokens** |
| Max concurrency at 32,768 tokens/request | **3.96x** |

That works out to ~37.8 KB/token actually allocated, against ~32.8 KB/token computed from
the config — the difference being Mamba state. The computed model above is a good estimate
but not a substitute for reading what vLLM reports.

## Environment

- **GPU:** NVIDIA GeForce RTX 5090, Blackwell **sm_120**, 32,607 MiB, driver 610.74
- **Host:** Windows 11 Pro 26200
- **Runtime:** WSL2, Ubuntu 24.04.3 LTS, kernel 6.6.87.2-microsoft-standard-WSL2, 30 GB RAM
- **vLLM 0.27.1**, **torch 2.13.0+cu130** (CUDA 13.0), Python 3.12
- torch arch list includes `sm_120` (FP4 kernels compiled in)
- **Model:** [`unsloth/Qwen3.8-27B-NVFP4`](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4),
  22.5 GB + 811 MB MTP head

`Qwen3_5ForConditionalGeneration` and `Qwen3_5MTP` are both registered in **stable vLLM
0.27.1** (not only on `main`), so no nightly wheel is required.

## Measurements

**PENDING** — nothing here yet. Planned matrix, using the harness in
[`bench/bench_fixed.py`](bench/bench_fixed.py):

- decode tok/s at prompt lengths 256 / 2048 / 8192, concurrency 1 / 4 / 8
- n=5 repeats, medians and spread, `min_tokens` + `ignore_eos` pinned, random nonce so
  prefill is cold
- KV dtype `auto` vs `fp8`
- MTP speculative decoding on vs off
- context ceiling actually achievable at each KV dtype

## The stock chat template poisons multi-turn conversations

Reported in the wild as multi-turn agentic work truncating (one report: 6 of 15 turns),
not present on Qwen3.6. Reproduced here from the template shipped with the model —
`bench/template_test.py` is the reproducer.

The assistant branch of `chat_template.jinja` reads:

```jinja
{%- set reasoning_content = '' %}
{%- if message.reasoning_content is string %}
    {%- set reasoning_content = message.reasoning_content %}
{%- endif %}
{%- if preserve_thinking is undefined or preserve_thinking is true
       or loop.index0 > ns.last_query_index %}
    {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
```

Two defaults collide:

1. `reasoning_content` falls back to `''` — and OpenAI-compatible clients replay only
   `content`, so it is empty for **every** historical turn.
2. `preserve_thinking is undefined` evaluates **true**, so the first branch always fires.

The result is an empty closed think block on every prior assistant turn. Rendering a
6-turn conversation the way an agent loop actually replays it:

| Rendering | `<think>` tags | Empty `<think>\n\n</think>` blocks |
|---|---|---|
| default (no kwargs) | 7 | **6** |
| `preserve_thinking=false` | 1 | **0** |
| `reasoning_effort=low` | 7 | **6** (does not help) |

Default:

```
<|im_start|>assistant\n<think>\n\n</think>\n\ndone step 1
<|im_start|>assistant\n<think>\n\n</think>\n\ndone step 2
```

With `preserve_thinking=false`:

```
<|im_start|>assistant\ndone step 1
<|im_start|>assistant\ndone step 2
```

**Both still end with the same generation prompt** (`<|im_start|>assistant\n<think>\n`), so
the live turn still thinks — only the poisoned history is removed.

Why this plausibly truncates: the model is handed six in-context demonstrations of "open
think, close it immediately, emit one short line", then asked to generate. There is a second
cost too — `reasoning_content` exists only on the newest turn, so the rendered history
*mutates* every round and prefix caching is invalidated.

### Two fixes, and they render identically

Compared head-to-head against
[froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)
on the same 6-turn replay (`bench/template_compare.py`):

| Template | kwargs | chars | `<think>` tags | Empty blocks |
|---|---|---|---|---|
| stock | defaults | 1005 | 7 | **6** |
| stock | `preserve_thinking=false` | 891 | 1 | **0** |
| froggeric | defaults | 891 | 1 | **0** |
| froggeric | `preserve_thinking=false` | 891 | 1 | **0** |

**The community template fixes this by default. Stock plus one kwarg produces identical
output.** So the cheapest fix is per-request, no template swap and no redeploy:

```json
{"chat_template_kwargs": {"preserve_thinking": false}}
```

The community template gets there differently — where the stock template emits a blank
block when `reasoning_content` is missing, it parses reasoning back out of inline
`</think>` markers in `content`. It also carries tool-argument handling and
`<|think_off|>` / `<|think_on|>` inline controls that this test does not exercise, so
"identical" holds for this conversation shape, not universally.

**Not yet verified here:** that either fix resolves the truncation *behaviourally*. The
above is prompt rendering only. `bench/multiturn_test.py` runs the behavioural check
(15-turn loop, measuring answer length after `</think>` per turn).

## Reference point: the same model on DGX Spark

[Weschera/Qwen3.8-27B-DGX-Spark-Quant-Ladder](https://github.com/Weschera/Qwen3.8-27B-DGX-Spark-Quant-Ladder)
ran a quant ladder for this model on **DGX Spark (GB10, sm_121)** on release day. Those are
**their** numbers on **their** hardware, quoted here as a cross-hardware reference — not
measured by us and not directly comparable to a 5090:

| Quant | Engine | Spec decode | Their tok/s (GB10) |
|---|---|---|---|
| NVFP4 | vLLM | MTP K3 | 23.7 |
| FP8 | vLLM | MTP K3 | 15.0 |
| UD-Q4_K_XL | llama.cpp | none | ~11.5 |
| BF16 | vLLM | MTP K3 | ~10 |

Their ordering (NVFP4 fastest) is the useful part. The absolute figures should differ
substantially on a 5090 — GB10 is a low-bandwidth unified-memory part, and decode on this
model is bandwidth-bound — which is exactly what the tables above are for.

Settings worth borrowing from their write-up, and why:

- **`--reasoning-parser qwen3`** — this model's default thinking effort is `xhigh` and it
  can emit tens of thousands of reasoning tokens before any answer. Without a reasoning
  parser those land in `content` and silently break clients that expect an answer.
  Per-request override: `{"chat_template_kwargs": {"reasoning_effort": "medium"}}`.
- **`--kv-cache-dtype fp8`** — roughly doubles KV capacity; used here too.
- **`--attention-backend triton_attn`** — they report FlashAttention failing on sm_121.
  On sm_120 we see the related constraint (FA3/FA4 required for FP8 KV), and vLLM selects
  FlashInfer on its own.
- **MTP K3 for code, K2 for chat** — they report position-1 acceptance 61–74%, position-2
  24–35%, position-3 only 6–22%, so the third token rarely pays for itself outside coding.

## Measurement methodology

Decode tok/s means at least three different things and published figures routinely mix
them. The traps — TTFT inclusion, prefix-cache inflation of prefill, output-length variance
destroying aggregates — are documented in the companion repo's
[MEASUREMENT-NOTES.md](https://github.com/Murai-Labs/dgx-spark-x-2-upstream/blob/main/docs/MEASUREMENT-NOTES.md),
and the harness here is the same one, unmodified.

## Reproducing

```bash
# 1. environment (WSL2 Ubuntu)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 && uv pip install vllm==0.27.1

# 2. weights
hf download unsloth/Qwen3.8-27B-NVFP4

# 3. serve  (see deploy/ for the exact scripts used)
bash deploy/vllm_boot1.sh
```

Copying the weights onto the WSL ext4 filesystem rather than reading them from `/mnt/<drive>`
is worth it if you restart the server often: drvfs measured **244 MB/s** here, which is ~92 s
of load time per start.

## Licence

MIT — see [LICENSE](LICENSE). Model weights are not included and carry their own licence
(Apache-2.0 for Qwen3.8-27B).
