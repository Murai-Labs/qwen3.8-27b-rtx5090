# Qwen3.8-27B on a single RTX 5090

Serving **Qwen3.8-27B** (NVFP4) on one **NVIDIA GeForce RTX 5090** (Blackwell, **sm_120**,
32 GB) under vLLM in WSL2, tuned for decode throughput without giving up quality.

Everything here was measured on the machine described in [Environment](#environment).
Where a number is computed rather than measured, it says so. Where something is not
measured, it says that too — including the arm that would not run.

## TL;DR

| | |
|---|---|
| **Decode** | **107.6 tok/s** at p256, **97.3** at p2048, single stream |
| **The flag that matters** | `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` — **~2× decode** |
| **Native FP4** | Verified engaged on sm_120 — `CutlassNvFp4LinearKernel`, no Marlin fallback |
| **Context** | 126,976 KV tokens at 32k/FP8 baseline; 56,599 with MTP K3 |
| **Concurrency limit** | Bounded by **Mamba cache blocks**, not KV — one per decode sequence |
| **Multi-turn** | Send `"chat_template_kwargs": {"preserve_thinking": false}` or agentic loops truncate |
| **Doesn't work** | MTP **K2** never serves on this build; K3 does |

The full recipe is in [The recipe](#the-recipe); the reasoning for each flag is inline in
[`deploy/serve.sh`](deploy/serve.sh).

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
| `--max-num-batched-tokens 16384` refuses to start: *"1.87 GiB KV cache is needed, which is larger than the available KV cache memory (1.42 GiB)"* | larger batched-token budgets consume activation memory, which comes straight out of KV. On 32 GB with MTP K3 it drops available KV from 3.00 GiB to 1.42 GiB — below one 32k request | this value is copied from DGX Spark recipes, where a node has **128 GB unified memory**. It does not transfer to a 32 GB card. See the sweep below |
| `--gpu-memory-utilization 0.98` refuses to boot: *"Free memory on device cuda:0 (30.2/31.84 GiB) on startup is less than desired GPU memory utilization (0.98, 31.21 GiB)"* | On a **Windows desktop** the shell holds ~1.4 GiB of VRAM (`explorer.exe`, `SearchHost`, `StartMenuExperienceHost`, `CrossDeviceResume`), so only ~30.2 of 31.84 GiB is free. Recipes quoting 0.98 assume a cleaner GPU | max safe utilization here is **0.948**; we use **0.94**. Worth checking `nvidia-smi` free memory rather than copying a number — this is the most machine-specific flag in any recipe |
| **Server logs `Application startup complete` and `HTTP server started`, then refuses every connection** | Specific to **port 8000** on this machine. `ss` shows `LISTEN 127.0.0.1:8000` owned by the live API-server pid, yet connections from the same WSL session — and from Windows — get `ECONNREFUSED`. Not a vLLM bug: a trivial python HTTP server on 127.0.0.1:8123 works fine, plain loopback is healthy, 8000 is not in a Windows reserved port range, and nothing is listening on 8000 Windows-side | **Use any other port.** 8137 worked first try and has been reliable since. This cost six benchmark runs before being isolated. Root cause not established; `networkingMode=mirrored` in `.wslconfig` is suspected but unproven |
| With MTP enabled, cudagraphs silently downgrade | *"CUDAGraphMode.FULL_AND_PIECEWISE is not supported with spec-decode for attention backend FlashInferBackend … setting cudagraph_mode=PIECEWISE"* | informational, but it means our MTP numbers are on PIECEWISE cudagraphs. `triton_attn` may support more — untested |

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

Harness: [`bench/bench_fixed.py`](bench/bench_fixed.py), unmodified. 3 repeats per cell,
medians reported. `min_tokens` + `ignore_eos` pinned so **every cell emitted exactly 200
tokens** — verified, so none of these numbers are confounded by output-length variance.
Random nonce per prompt, so prefill is cold, never prefix-cached.

Common flags: `--max-model-len 32768 --gpu-memory-utilization 0.90 --kv-cache-dtype fp8`.

### Decode throughput (tok/s)

| prompt | conc | A: baseline | C: MTP K3 | Δ |
|---|---|---|---|---|
| 256 | 1 | 51.5 | **107.6** | **+108.8%** |
| 256 | 4 | 46.9 | **108.4** | +131.0% |
| 2048 | 1 | 52.5 | **97.3** | +85.2% |
| 2048 | 4 | 45.0 | **97.1** | +116.0% |
| 8192 | 1 | 52.0 | **98.2** | +88.7% |
| 8192 | 4 | 38.7 | **70.2** | +81.5% |

**MTP speculative decoding roughly doubles decode throughput.** That is a much larger win
than the published per-position acceptance rates suggest, and it is the single most
important setting in this recipe.

Baseline decode is essentially flat across prompt length (51.5 / 52.5 / 52.0 at
concurrency 1) — the signature of bandwidth-bound decode. The hybrid architecture is why:
context costs KV, not decode speed.

**Spread matters here.** Baseline is tight (0.4–4.1 tok/s across repeats); MTP is much
noisier (4.9–19.6), because acceptance rate depends on what is being generated. The
+81–131% gains are far outside that spread, but a single MTP sample is not a reliable
number. Aggregate throughput at concurrency 4 is noisier still (±109 tok/s at p256) and
should not be quoted without its spread.

### What MTP costs you

| Config | KV memory | KV tokens | Max concurrency @32k |
|---|---|---|---|
| Baseline | 4.47 GiB | 126,976 | 3.88x |
| MTP K2 | 3.03 GiB | 63,351 | 1.93x |
| MTP K3 | 3.00 GiB | 56,599 | 1.73x |

MTP takes roughly **half your KV capacity**. On this card that is the real trade: double
the decode rate, half the context budget and concurrency. At 32k context it is clearly
worth it. If you need long context *and* many concurrent sequences, measure before
assuming.

vLLM also warns at K>1: *"Enabling num_speculative_tokens > 1 will run multiple times of
forward on same MTP layer, which may result in lower acceptance rate."*

### Not measured

- **Arm B (MTP K2) — would not serve, in four attempts.** The engine initialises
  cleanly (KV allocated: 3.03 GiB / 63,351 tokens) and the API server logs
  `Application startup complete`, but the socket never accepts connections. The final
  attempt ran K2 standalone with a 180-second socket wait and still got
  `Connection refused` on every request. **MTP K3 with otherwise identical flags serves
  fine**, so this is specific to `num_speculative_tokens: 2` on this build, not a
  startup race. Cause not investigated further. Worth knowing, because K2 is what the
  DGX Spark ladder recommends for chat — on this setup it is currently not an option.
- Concurrency above 4, and context above 32k.
- Quality. No perplexity or task evaluation was run, so nothing here says NVFP4 matches
  FP8 or BF16 on output quality — only that it is roughly twice as fast with MTP.

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

### What this does and does not cause

Our first framing here was that the empty blocks plausibly *cause* the multi-turn
truncation, on the reasoning that six in-context demonstrations of "open think, close it
immediately, emit one short line" teach the model to do the same. **That framing is
probably wrong**, and we never verified it behaviourally.

A third-party behavioural audit A/B tested the stock and community templates over the same
12-turn debugging session and reports: both templates truncate at **the same turn**, and
both run 8 agentic tool-use turns with zero aborts. Their conclusion is that the truncation
is **context exhaustion caused by the model's own verbosity** — it writes 8,000–12,000
character answers unprompted, and a 12-turn session reached ~46,800 characters by turn 6.

The dangerous part of their finding is the failure signature: turns return
`finish_reason: length` after only ~2,900 of a 32,768-token generation budget, which reads
as "raise `max_tokens`". Raising it does nothing, because the constraint was remaining
*context*, not the generation budget. The fix is context headroom plus an explicit brevity
instruction.

We have not reproduced that audit. What we independently verified is narrower and still
stands: the empty blocks are real, they are eliminated by either fix, and because
`reasoning_content` exists only on the newest turn the rendered history *mutates* every
round, which invalidates prefix caching. That last cost is real regardless of the
truncation question.

So: still set `preserve_thinking: false` — it removes real prompt garbage and restores
prefix-cache stability. Do **not** expect it to fix multi-turn truncation.

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

## Same GPU, same checkpoint: the TurboQuant recipe

[ayayalar/Qwen3.8-27B-NVFP4-TurboQuant](https://github.com/ayayalar/Qwen3.8-27B-NVFP4-TurboQuant)
is the closest possible reference: **same model** (`unsloth/Qwen3.8-27B-NVFP4`), **same GPU**
(RTX 5090, 32 GB), **same vLLM 0.27.1**. It targets the full 262,144-token context rather
than throughput, and its `CALIBRATION.md` is an unusually honest rejection log. Credit to
its author — several of the findings below are theirs, not ours.

**Independent cross-validation of our throughput.** Their no-MTP figures are ~55 tok/s
single-stream and ~197 aggregate at 4-way concurrency. Our arm A measured **51.5–52.5** and
**180.9**. Two testers, same hardware, within ~7%.

**Where their recipe goes further than ours:**

| Their choice | Ours | Effect |
|---|---|---|
| `--kv-cache-dtype turboquant_4bit_nc` | `fp8` | 4-bit KV halves the footprint, reaching the **full 262k context**. They report fp8 capping near ~227k |
| `--kv-cache-memory-bytes 5368709120` | auto-fit | auto-fit over-reserved **1.66×** (435k tokens), leaving ~0 MB for prefill → long requests OOM'd |
| `--gpu-memory-utilization 0.98` | 0.90 | they call 0.98 the practical ceiling on 32 GB — **but it will not boot on a Windows desktop**, see below |
| `--max-num-batched-tokens 512` | default | note this is the **opposite** of Spark Arena's 16384 — they optimise for context, Spark Arena for throughput. Neither transfers blindly |

### Their MTP warning — and why we still recommend it

Their calibration log rejects MTP outright:

> *"Garbles output with 4-bit KV — empty `content`, no tool calls, degeneration into token
> repetition. … Reproduced at 262144 with turboquant (both fp8 KV and 4-bit KV), with and
> without the KV pin. … it boots cleanly — that's the trap."*

We take that seriously, because MTP K3 is our headline recommendation and we had shipped it
without inspecting a single output. So we checked. **On this configuration it does not
reproduce:**

| Probe | `finish_reason` | Max consecutive repeated token | Output |
|---|---|---|---|
| `What is 2+2?` | stop | 1 | `4` |
| Capital of France | stop | 1 | `Paris` |
| One-line Python add | stop | 1 | coherent |

No repetition collapse, no empty content, no truncation.

**Scope, precisely:** they reproduced at `num_speculative_tokens: 2` with a 262,144 context;
we tested `num_speculative_tokens: 3` at 32,768 with short prompts, driver 610.74. Their
finding may well hold at long context or at K2 — we have not tested there, and notably
**K2 will not even serve on our machine**. Treat MTP as verified-clean only in the regime
measured here.

## Quality evaluation

`bench/quality_eval.py` + `bench/quality_compare.py`. Reads GSM8K and MMLU from the local
HF cache (no network, no `datasets` package), runs them against any OpenAI-compatible
endpoint, and scores exact-match.

Three design choices that are not optional for comparing quantizations:

1. **Paired, not independent.** Per-item results are saved and `quality_compare.py` runs
   **McNemar's exact test** on the items that flipped. At n=200 and ~85% accuracy the 95% CI
   on a single run is roughly ±5 points, so two independent runs can differ by 4 points with
   fully overlapping intervals and tell you nothing. The discordant pairs are the signal.
2. **`no_answer` is scored separately from `incorrect`.** A config that exhausts its
   generation budget mid-thinking is failing differently from one that reasons to a wrong
   conclusion — and this model thinks at `xhigh` by default. Collapsing them hides the
   mechanism.
3. **Config is recorded and enforced.** `reasoning_effort`, `max_tokens` and the task are
   written into the output, and the compare script **refuses** to compare runs that differ on
   any of them, since the delta would not be attributable to the quantization. It also
   rejects `reasoning_effort=medium` outright, that being a silent no-op.

The harness was verified offline before use: loaders are deterministic across runs (same
items every time, which is what makes pairing valid), the `</think>` stripping and answer
extraction pass a case table including unclosed think blocks, and McNemar correctly
recovers a synthetic 5-point effect at p=0.013.

### Result: GSM8K, NVFP4 + MTP K3

| | |
|---|---|
| Accuracy | **97.0%** (95% CI 93.6–98.6), n=200 |
| correct / incorrect / **no_answer** / error | 194 / 6 / **0** / 0 |
| `finish_reason` | `stop` on **all 200** |
| `think_closed` | True on all 200 |
| Median completion tokens | 242 |
| Wall clock | 112 s |

Config: `reasoning_effort=low`, `max_tokens=2048`, `temperature=0`,
`preserve_thinking=false`, fp8 KV, 32k context.

**Zero `no_answer` and zero truncation is the load-bearing part**, not the 97%. It is the
direct measurement that MTP is not garbling output on this configuration — a garbling
failure would show as near-zero accuracy with a high `no_answer` count, which is precisely
the failure mode reported against MTP elsewhere.

The 6 misses were inspected individually rather than trusted as an aggregate; all are
genuine arithmetic errors (48 vs 44, 105 vs 75, 600 vs 1000, 40 vs 35, 3.5 vs 3, 9860.78 vs
7400), not extraction artifacts.

Raw per-item results: [`bench/results/quality_gsm8k_nvfp4_mtp3.json`](bench/results/quality_gsm8k_nvfp4_mtp3.json).

### The paired control: MTP costs nothing

Same 200 items, same settings, the only difference being `--speculative-config`:

| | baseline (no spec decode) | MTP K3 |
|---|---|---|
| Accuracy | **97.0%** | **97.0%** |
| Median completion tokens | 236 | 242 |
| `no_answer` | 0 | 0 |
| Wall clock, same 200 items | **264 s** | **112 s** |

Paired table: **both correct 194, only-baseline 0, only-MTP 0, both wrong 6.**
McNemar exact two-sided **p = 1.0000**.

**Zero discordant items** — not just equal accuracy, but the *same* 194 right and the *same*
6 wrong. That is what correct speculative decoding should look like: MTP proposes draft
tokens, the target model verifies them, and rejections are resampled, so at `temperature=0`
the result is lossless. We measured the theory rather than assuming it.

So on this configuration **the ~2× speedup is free**. The 264 s → 112 s wall clock on an
identical real workload (2.36×) also confirms the synthetic decode benchmark independently.

Caveats worth keeping: this is `temperature=0` on one task at n=200, and equality of
*outcomes* is not equality of *tokens* — two runs can reach the same answer by different
text. And it says nothing about long context, which is exactly where the MTP garbling was
reported.

## Cross-hardware: the same checkpoint on DGX Spark (Spark Arena)

[**Spark Arena**](https://spark-arena.com) is a community LLM leaderboard for NVIDIA DGX
Spark, maintained by **Drew Botwinick**, **Eugene Rakhmatulin (eugr)** and **Raphael
Amorim**, built on three open-source tools they publish:
[spark-vllm-docker](https://github.com/eugr/spark-vllm-docker),
[llama-benchy](https://github.com/eugr/llama-benchy) and
[sparkrun](https://github.com/eugr/sparkrun). Every entry publishes its full recipe YAML
and raw benchmark CSV, which is what makes the comparison below possible at all — credit
to them for publishing reproducible configs rather than screenshots.

Two Qwen3.8-27B entries exist on their `tg128 (c1)` board. Snapshot of all 193 entries
retained at
[`bench/results/reference/spark-arena_tg128_c1.json`](bench/results/reference/spark-arena_tg128_c1.json)
(generated 2026-08-15, fetched from their public `/static/snapshot/test` endpoint):

| Rank | Entry | Runtime | Cluster | Spec decode | tok/s |
|---|---|---|---|---|---|
| 104 | `Qwen/Qwen3.8-27B-FP8` — submitted by Drew Botwinick | vLLM | **4 nodes** | MTP K3 | 39.17 |
| 190 | `unsloth/Qwen3.8-27B-NVFP4` — submitted by Saiyam Pathak | vLLM | **single** | **none** | 11.48 |

**Entry 190 is the same checkpoint this repo uses**, on one DGX Spark, and its recipe has
no `speculative_config` — so the like-for-like comparison against our baseline is:

| Setup | Spec decode | tok/s |
|---|---|---|
| 1× DGX Spark (GB10), NVFP4 | none | **11.48** |
| **1× RTX 5090, NVFP4 (this repo, arm A)** | none | **51.5–52.5** — **~4.5×** |
| 1× RTX 5090, NVFP4 (this repo, arm C) | MTP K3 | 97.3–107.6 |
| 4× DGX Spark, FP8 | MTP K3 | 39.17 |

**~4.5× per node with speculative decoding held off on both sides** is the defensible
number. The 8–9× figure against our MTP arm is not like-for-like and should not be quoted.

Two further caveats: their harness is `llama-benchy` generating 128 tokens, ours is
`bench_fixed.py` generating 200, and metric definitions differ — on the FP8 entry's raw
CSV, `t_s_req_mean` and `peak_ts_req_mean` differ by ~20%, which is exactly the
"decode tok/s means three different things" trap. Treat the ordering as solid and the
multiple as approximate.

Worth noting on their side: a 4-node TP=4 cluster reaches 39.17 tok/s single-stream, only
~3.4× a single node. Decode is bandwidth-bound, and sharding it across a network does not
fix bandwidth.

### Flags from their recipes worth stealing

Their `spark-vllm-docker` / `sparkrun` recipes set several things this repo did not:

| Flag | Why it matters here |
|---|---|
| `--max-num-batched-tokens 16384` | vLLM **explicitly warned us** that MTP dropped `max_num_scheduled_tokens` to 2048 and that this "may lead to suboptimal performance". We had not acted on it. |
| `--enable-prefix-caching` | Ours ran with `enable_prefix_caching=False`. Large for multi-turn — and it only pays if history is stable, which is what `preserve_thinking: false` restores. |
| `--load-format instanttensor` | Faster weight load; ours takes ~90 s per boot. |
| `--mm-encoder-tp-mode data` | Vision tower placement. |
| `--enable-auto-tool-choice --tool-call-parser qwen3_coder` | Required for structured tool calls. |
| `--reasoning-parser qwen3` | Already in our `serve.sh`. |

## Reference point: llama.cpp figures reported elsewhere

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
  Per-request override: `{"chat_template_kwargs": {"reasoning_effort": "low"}}` — **not
  `medium`**, see below.
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

## The recipe

Server — [`deploy/serve.sh`](deploy/serve.sh) has this with the reasoning for each flag
inline:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model $HOME/models/Qwen3.8-27B-NVFP4 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --kv-cache-dtype fp8 \
  --max-num-seqs 8 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3
```

Client — **send this on every multi-turn request**:

```json
{"chat_template_kwargs": {"preserve_thinking": false}}
```

Without it the stock template poisons the history with empty think blocks and multi-turn
agentic work truncates. Add `"reasoning_effort": "low"` to the same object to cap the
thinking budget; the default is `xhigh`. **Do not use `"medium"`** — see below.

### `reasoning_effort=medium` is a silent no-op

The template validates `medium` as a legal value, then has branches only for `xhigh` and
`low`. Setting it leaves the reasoning instruction empty — so it is not a middle setting,
it is *less* steering than the default, with no error to tell you. Verified by rendering:

| `reasoning_effort` | system prompt | instruction emitted |
|---|---|---|
| unset | 297 chars | **xhigh** (this is the default) |
| `xhigh` | 297 chars | xhigh |
| `high` | 297 chars | xhigh (aliased to `xhigh`) |
| **`medium`** | **60 chars** | **none — silently dropped** |
| `low` | 226 chars | low |
| anything else | raises | — |

An earlier version of this README recommended `medium`. That was wrong.

The three flags that matter most, in order: **MTP K3** (~2× decode), **`preserve_thinking:
false`** (multi-turn correctness), **`--kv-cache-dtype fp8`** (~2× context).

## Reproducing

```bash
# 1. environment (WSL2 Ubuntu 24.04)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 && uv pip install vllm==0.27.1

# 2. weights (~22 GB)
hf download unsloth/Qwen3.8-27B-NVFP4

# 3. serve
bash deploy/serve.sh

# 4. benchmark
python bench/bench_fixed.py --base-url http://127.0.0.1:8000/v1 --model qwen38 \
    --prompt-tokens 256,2048,8192 --concurrency 1,4 --max-tokens 200 --repeats 3 \
    --label mine --output results.json
python bench/summarize.py results.json

# 5. chat-template checks
python bench/template_test.py       # empty think blocks, stock template
python bench/template_compare.py    # stock vs community template
python bench/multiturn_test.py      # behavioural 15-turn truncation test
```

Copying the weights onto the WSL ext4 filesystem rather than reading them from `/mnt/<drive>`
is worth it if you restart the server often: drvfs measured **244 MB/s** here, which is ~92 s
of load time per start.

## Licence

MIT — see [LICENSE](LICENSE). Model weights are not included and carry their own licence
(Apache-2.0 for Qwen3.8-27B).
