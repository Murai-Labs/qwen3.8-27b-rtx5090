# DFlash 2 works — on llama.cpp, and the win is a cheaper drafter, not a better one

DFlash 2 [could not be served under vLLM here](dflash2-does-not-fit.md): its BF16 drafter and the
unquantized LM head it forces cost more memory than the whole KV pool. The GGUF drafter is
**1.06 GiB instead of 3.58**, and that changes the answer completely.

**It is the fastest single-stream config this repo has measured: 149.5 tok/s at p256, against
107.6 for the published vLLM recipe — and it is measurably *more* faithful to unspeculated
decoding than MTP is.** It is not the recommended recipe yet; the reasons are at the bottom.

Measured 2026-08-19 on the RTX 5090, llama.cpp
[PR #27342](https://github.com/ggml-org/llama.cpp/pull/27342) (`build 10498`, sm_120), target
`ggml-org/Qwen3.8-27B-GGUF:Q4_K_M` (17.67 GiB), `-fa on -ctk q8_0 -ctv q8_0 -c 32768 -np 1`.
Throughput via this repo's own `bench/bench_fixed.py`, so these sit in the same table as the vLLM
numbers. Raw results in [`bench/results/llamacpp-dflash2/`](../bench/results/llamacpp-dflash2/).

## Throughput

7 repeats, medians, `±` is the range across repeats. Concurrency 1.

| arm | drafter | p256 | p2048 | p8192 |
|---|---|---:|---:|---:|
| no speculation | — | 66.8 ±0.3 | 66.1 ±1.4 | 64.6 ±2.0 |
| MTP K3, Q8_0 head | 2.95 GiB | 117.5 ±45.3 | 114.3 ±17.4 | 112.3 ±4.5 |
| MTP K3, Q4_0 head | 1.56 GiB | 126.1 ±22.5 | 122.4 ±13.8 | 116.1 ±18.8 |
| **DFlash 2, n=4** | **1.06 GiB** | **149.5 ±21.5** | **135.1 ±25.7** | **117.1 ±28.9** |
| DFlash 2, n=7 *(3 repeats)* | 1.06 GiB | 136.1 ±38.0 | 128.9 ±20.1 | 115.5 ±43.8 |
| *published vLLM recipe, NVFP4 + MTP K3* | — | *107.6* | *97.3* | *98.2* |

**`--spec-draft-n-max 4` beats the recommended 7** at p256 and p8192, independently reproducing
what a third party measured on a 3090 in the PR thread. The upstream default is not the right
setting on consumer NVIDIA.

**Read the spreads before the medians.** They are enormous — ±45.3 on one MTP cell against ±0.3
unspeculated — because acceptance depends on content, which this repo already documents. The
p256 gap is comfortably real; **the p8192 gap between DFlash 2 and MTP-Q4 (117.1 vs 116.1) is
not a result at all**, it is two overlapping ranges.

## Quality: DFlash 2 is the more faithful drafter

GSM8K n=200, `temperature=0`, `max_tokens=2048`, default (xhigh) reasoning, paired McNemar
against the *unspeculated* arm on the same engine and checkpoint.

| arm | accuracy | correct / incorrect / no_answer | discordant vs no-spec | p | gate |
|---|---:|---|---|---:|---|
| no speculation | 95.0% | 190 / 3 / 7 | — | — | control |
| **DFlash 2, n=4** | **94.5%** | — / — / 8 | **1** (0 gained, 1 lost) | 1.0000 | **PASS** |
| MTP K3 | 94.0% | 188 / 3 / 9 | **6** (2 gained, 4 lost) | 0.6875 | PASS |

Both pass. But **DFlash 2 moved exactly one item off the unspeculated baseline; MTP moved six.**
Speculative decoding is supposed to be distribution-preserving, so fewer discordant items is the
better result, and here the newer drafter is the cleaner one. Given a maintainer on the vLLM PR
saying he is *"not entirely convinced of the correctness of the probabilistic code"*, this was
the number most worth checking, and it came back clean.

Every arm's accuracy gap is `no_answer`, not wrong reasoning: all three sit at 3 incorrect, and
differ only in how often xhigh thinking exhausts the 2048-token budget.

## The mechanism: it is drafter cost, not draft quality

Server-reported mean accepted draft length, last 8 requests of each run:

| arm | drafter size | mean accepted length |
|---|---|---:|
| MTP K3, Q8_0 head | 2.95 GiB | 3.07–3.71 (**~3.41**) |
| DFlash 2, n=4 | 1.06 GiB | 2.54–3.55 (**~2.94**) |
| MTP K3, Q4_0 head | 1.56 GiB | 2.55–3.00 (**~2.74**) |

**MTP with its full-precision head drafts *better* than DFlash 2 here — and is still slower.**
Quantizing the MTP head Q8_0 → Q4_0 drops acceptance 3.41 → 2.74 and *raises* throughput
117.5 → 126.1. Draft quality went down, speed went up. On a bandwidth-bound single stream, what
the drafter costs to run dominates what it wins in acceptance.

That is also the honest framing of the comparison: at a matched drafter precision class
(Q4-ish), DFlash 2 wins on both axes — higher acceptance (2.94 vs 2.74) *and* a smaller drafter
(1.06 vs 1.56 GiB) — for **+18.6% / +10.4% / +0.9%** over MTP-Q4. Against the Q8_0 head it is
+27% / +18% / +4.3%, but part of that is simply comparing against a 2.8×-larger drafter.

**Upstream's headline claim does not reproduce here.** The model card reports DFlash 2 at 5.46
mean acceptance against MTP's 5.02 on an H200 in BF16. We measure ~2.9 against ~3.4 — the
ordering *inverts* — on a Q4_K_M target with quantized KV on consumer hardware. The speedup is
real; the reason given for it is not the reason it happens here.

## Why this is not the recipe yet

- **Concurrency 1 only.** Both engines have open concurrency bugs *right now*: an
  index-out-of-bounds crash at n=4 on sm_120 in the vLLM PR (our exact architecture), and a
  report of throughput collapsing to ~1 tok/s with multiple agents on llama.cpp. Measuring
  concurrency today would be measuring a known-broken path. The published vLLM recipe is
  validated to concurrency 4.
- **It is an unmerged PR**, one day old, on a fork of an engine this repo does not otherwise use.
- **The quality comparison against NVFP4 is missing, and could not be faked.** This repo's stored
  NVFP4 GSM8K runs used `reasoning_effort=low` with 0 `no_answer`; these ran at default xhigh.
  `quality_compare.py` refused the pairing outright — *"different reasoning_effort, the
  comparison would not be attributable to the quantization"*. So **whether Q4_K_M costs quality
  against NVFP4 is unmeasured**, and the 95.0% here must not be read against the README's 97.0%.
  Re-running at matched effort needs llama-server started with `--jinja` first, or
  `reasoning_effort` is silently ignored — the same accepted-and-ignored trap this repo has hit
  twice already.
- **Long context, multi-turn and the fp8-KV findings are all unretested** on this engine.

## What did land

The memory wall is gone: peak 24.1 GiB of 32, against a vLLM path that could not allocate a
single KV block. If the concurrency bugs close and the PRs merge, this is the strongest
candidate to replace the recipe — on evidence, not on the blog's numbers.

```bash
llama-server -m Qwen3.8-27B-Q4_K_M.gguf \
  -md Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --spec-type draft-dflash --spec-draft-n-max 4 \
  -ngl 99 -fa on -c 32768 -np 1 -ctk q8_0 -ctv q8_0
```
