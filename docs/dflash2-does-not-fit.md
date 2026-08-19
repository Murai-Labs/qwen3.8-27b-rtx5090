# DFlash 2 on this machine: four blockers, and the one that does not move

**Verdict: the recipe stays on MTP K3.** DFlash 2 cannot be served on a 32 GiB RTX 5090
alongside this 27B NVFP4 target. It is not a tuning problem — the drafter and the unquantized
LM head it requires cost more memory than the entire KV pool the published recipe runs on.

Everything below was measured on this machine on 2026-08-18. Logs are in
[`bench/results/dflash2-probes/`](../bench/results/dflash2-probes/).

## What DFlash 2 is

[Inco AI, August 2026](https://inco.ai/blog/dflash2/). A block-diffusion drafter: it predicts a
whole block of tokens in one pass, keeps the top-16 candidates per position, and walks a
coherent path through them with a learned selector, with two-tap dynamic convolutions to stop
the draft decaying toward the end of the block. Upstream claims, on `Qwen/Qwen3.8-27B`,
one H200, seven draft tokens:

| | MTP | DSpark | DFlash 2 |
|---|---:|---:|---:|
| GSM8K acceptance length | 5.02 | 4.36 | **5.46** |
| GSM8K decode, concurrency 1 | 178.5 tok/s | 185.3 | **236.1** |

Drafter: [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2),
2B parameters, BF16, **3.58 GiB** on disk. It ships no `embed_tokens` and no `lm_head` — it
borrows the target's.

Interface check before spending any GPU time — drafter against our NVFP4 target:
`hidden_size` 5120 = 5120, `num_target_layers` 64 = 64 layers, `vocab_size` 248320 = 248320,
`target_layer_ids [5, 19, 33, 47, 61]` all in range. Architecturally it fits.

## Blocker 1 — vLLM support is an open pull request

`pip install vllm` does not have it. Support is
[PR #52816](https://github.com/vllm-project/vllm/pull/52816), opened 2026-08-18, **still open**
at the time of this test, 11 files, all Python. Built from source into a *separate* venv so the
working 0.27.1 recipe stays intact:

```bash
git clone --filter=blob:none https://github.com/vllm-project/vllm.git
git fetch origin +refs/pull/52816/head:refs/heads/pr52816 && git checkout pr52816
uv pip install -r requirements/build/cuda.txt
VLLM_USE_PRECOMPILED=1 uv pip install --no-build-isolation -e .
```

Result: `vllm 0.26.1rc1.dev912+g19c935190`, torch 2.13.0+cu130, flashinfer 0.6.17,
`DFlash2DraftModel` registered. **Solved.**

The method string stays `dflash`; DFlash 2 is selected by the *checkpoint's* architecture, not
by the flag:

```json
{"method":"dflash","model":"<drafter>","num_speculative_tokens":7}
```

## Blocker 2 — the V2 model runner needs UVA, which vLLM disables under WSL2

DFlash 2's candidate selector lives only in the V2 model runner, so `use_v2_model_runner`
forces V2 (`vllm/config/vllm.py`, `_is_dflash2_draft`). V2 allocates UVA buffers and dies:

```
RuntimeError: UVA is not available          vllm/v1/worker/gpu/buffer_utils.py:47
```

`is_uva_available()` is just `is_pin_memory_available()`, and `vllm/platforms/cuda.py:290`
returns `envs.VLLM_WSL2_ENABLE_PIN_MEMORY` under WSL — **off by default**. This kernel supports
it; verified directly rather than assumed:

```
uname -r                                        -> 6.6.87.2-microsoft-standard-WSL2
torch.zeros(..., pin_memory=True).is_pinned()   -> True
get_accelerator_view_from_cpu_tensor(t)         -> cuda:0
```

`VLLM_WSL2_ENABLE_PIN_MEMORY=1`. **Solved.** Worth knowing for any V2-runner feature under WSL2,
not just this one.

## Blocker 3 — DFlash 2 refuses a quantized LM head, and ours is FP8

```
ValueError: DFlash2 requires an unquantized target LM head for candidate TopK.
            vllm/model_executor/models/qwen3_dflash2.py:322
```

The selector reads the target's top-K logits straight off the LM head
(`self.lm_head.quant_method.apply(...)`), and `unsloth/Qwen3.8-27B-NVFP4` quantizes it — the
first `config_groups` entry matches `re:.*lm_head`, and the checkpoint carries `lm_head.weight`
as `F8_E4M3 [248320, 5120]` with a BF16 `lm_head.weight_scale [248320, 1]`. The local `RadixArk`
NVFP4 conversion quantizes it too.

Public NVFP4 conversions that leave it alone do exist —
`gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` and `sakamakismile/Qwen3.8-27B-MTP-NVFP4` both
list `lm_head` in `ignore`. We did not switch to one: swapping the checkpoint changes every
weight at once, so the DFlash 2 number would stop being comparable to the MTP number this repo
already published, and the fp8 KV calibration scales this repo's central finding depends on
would come from a different calibration or, for the sakamakismile conversion
(`kv_cache_scheme: null`), not exist at all.

Instead, [`deploy/dequant_lm_head.py`](../deploy/dequant_lm_head.py) writes a derived checkpoint
with exactly one tensor changed: `lm_head.weight * lm_head.weight_scale -> BF16`, the scale
dropped, `lm_head` moved from the quantized targets into `ignore`. Cost on disk: **+1.18 GiB**
(22,568,192,096 -> 23,839,089,224 bytes). **Solved** — DFlash 2 loads and initialises against it.

## Blocker 4 — it does not fit, and this one does not move

Measured `Model loading took`:

| Config | Checkpoint | Weights resident |
|---|---|---:|
| **The published recipe** — MTP K3, vLLM 0.27.1 | stock NVFP4 | **22.13 GiB** |
| MTP K3, PR build | lm_head BF16 | **23.31 GiB** |
| DFlash 2, PR build | lm_head BF16 | **26.29 GiB** |

**+4.16 GiB over the recipe as shipped**, end to end. That is the number that decides this.

At the published recipe's own flags — `--gpu-memory-utilization 0.90 --max-num-seqs 8
--max-model-len 32768 --kv-cache-dtype fp8`:

```
Model loading took 26.29 GiB memory
Available KV cache memory: -0.56 GiB
ValueError: No available memory for the cache blocks.
```

**Negative.** Not "a small pool" — the engine is over budget before a single KV block exists.

Pushing every memory lever at once, well past anything servable — `--gpu-memory-utilization 0.94
--max-num-seqs 1 --max-model-len 4096 --max-num-batched-tokens 1024 --enforce-eager
--no-enable-prefix-caching --mamba-ssm-cache-dtype float16`:

```
Available KV cache memory: 0.61 GiB
ValueError: To serve at least one request with the model's max seq len (4096),
            0.89 GiB KV cache is needed, which is larger than the available 0.61 GiB
```

Still short, at a 4k context, with one sequence, with CUDA graphs off. An earlier run of the
same config read 0.9 GiB available / 4,134 tokens / concurrency 1.01× — the figure moves with
whatever the Windows desktop happens to be holding in VRAM, which is itself the point:
**DFlash 2 lands within a few hundred MiB of fitting, at a context four times too small to
serve.**

### The arithmetic, in one line

The +4.16 GiB splits as **+2.98 GiB** of drafter (26.29 − 23.31, both measured on the same
checkpoint) and **+1.18 GiB** of BF16 LM head (measured on disk).

The recipe as shipped has **2.94 GiB** of KV cache to give up — 55,606 tokens, measured on the
same idle machine minutes later, `deploy/serve.sh` unchanged, serving `"alive"` on a live
request. (4.45 GiB / 126,976 tokens is the *no-speculation* pool; MTP K3 already spends the
difference.)

**DFlash 2 asks for 4.16 GiB where 2.94 GiB exists.** Delete the entire KV cache and it still
does not fit. The drafter is bigger than the cache.

No flag closes that. It closes with a smaller target or a bigger card:

- a ~20.4 GiB NVFP4 conversion that already ships an unquantized LM head
  (`gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090`) is ~1.8 GiB lighter than our patched
  checkpoint and would plausibly leave a usable pool — at the cost of changing every weight, so
  its numbers would not be comparable to anything already published here. Untested;
- 48 GiB of VRAM makes the question disappear.

## Also observed: this vLLM build takes down the WSL2 VM on a cold compile

Independent of DFlash 2. On `main@9842d7014` + PR #52816, once the engine gets past KV
allocation into first-time compile/JIT, the whole distro dies — not an OOM-killed process, a
dead VM:

```
wsl.exe -d Ubuntu -- ...
Catastrophic failure
Error code: Wsl/Service/E_UNEXPECTED
```

5/5 reproducible. **The control that matters: it also happens with plain MTP K3, on the V1
runner, with `VLLM_WSL2_ENABLE_PIN_MEMORY=0`** — so it is neither DFlash 2, nor the V2 runner,
nor pinned memory. Capping `TORCHINDUCTOR_COMPILE_THREADS=4 MAX_JOBS=4` did not help. It is the
same failure mode `.wslconfig` in this workspace already documents from 2026-08-15 (kernel OOM
during first-time CUDA JIT, 11+ `cicc` processes at ~1.8 GiB RSS), except fatal to the VM rather
than to one process — the 48 GiB the VM was raised to is not enough for this build's cold
compile. vLLM 0.27.1 on the same machine does not do this.

Two consequences worth carrying forward: `/tmp` does not survive it, so benchmark logs must be
written outside tmpfs; and the ext4 journal does not always flush, so one probe's log was lost
entirely and **its numbers are not reported here.**

## What landed anyway

- `bench/arms.json` — `dflash2-k7` and `dflash2-k7-bf16kv` arms, wired and dry-run clean. They
  do not boot here; they are ready the day the target shrinks or the card grows.
- `bench/ratchet.py` — arm args now expand `$VARs`, so an arm can name a second checkpoint
  without hardcoding a home directory into `arms.json`.
- `deploy/dequant_lm_head.py` — the one-tensor checkpoint patch.

## Not measured

Acceptance length, throughput and quality for DFlash 2 — all three need an engine that boots.
No number in this document is a DFlash 2 performance number, and none of upstream's claims were
either confirmed or refuted here. The upstream fp8-KV question for non-causal drafters
([vLLM #41559](https://github.com/vllm-project/vllm/issues/41559), closed 2026-08-11 with
"support via FLASHINFER (CUTLASS) backend") was never reached either — the engine ran out of
memory first, so whether fp8 KV composes with DFlash on sm_120 is still open here.
