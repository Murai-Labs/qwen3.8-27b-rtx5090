# Tuning the GDN decode kernel is not worth doing

**Verdict: negative. The hypothesis was tested and it failed.** Retuning vLLM's Gated
DeltaNet decode kernel is worth at most **0.26–0.85% end-to-end**, which is below the
noise floor of our own throughput harness. Do not spend time on it.

Reproduce with `python bench/kernel_gdn_decode.py --batches 1,4,8`; raw data in
`bench/results/kernel_gdn_sweep.json`.

## The hypothesis

48 of this model's 64 layers are Gated DeltaNet. Every one calls
`fused_recurrent_gated_delta_rule_packed_decode` once per decode step, and that wrapper
(`vllm/third_party/flash_linear_attention/ops/fused_recurrent.py:437-439` in vLLM 0.27.1)
hardcodes its launch parameters:

```python
BV         = min(triton.next_power_of_2(V), 32)   # 32 here
num_stages = 3
num_warps  = 1
```

No `@triton.autotune`, no per-device config. For this model's geometry
(`linear_num_key_heads=16`, `linear_num_value_heads=48`, `linear_key_head_dim=128`,
`linear_value_head_dim=128`) that gives `grid = (V/BV, B*HV) = (4, 48)` at batch 1:
**192 blocks of one warp each on a 170-SM GPU.**

That looks occupancy-starved, and there is precedent for the concern — vLLM's *Mamba2*
`selective_state_update` ships per-GPU tuned JSON for B200, GB200, H100, H200, MI300X and
RTX PRO 6000, with a `--save-configs` benchmark to generate more. There is no RTX 5090
entry. The GDN path has no config mechanism at all.

## The measurement

192 configs — `BV` × `num_warps` × `num_stages` × batch — each gated on numerical
agreement with the shipped config before its timing counts. All 192 passed
(max |diff| 3.8e-06, from reduction-order changes when `BV` splits `V` differently).

| batch | shipped | achieved BW | % of the 19.42 ms/token budget | best config | speedup | end-to-end ceiling |
|---|---|---|---|---|---|---|
| 1 | 10.0 µs/layer | 627 GB/s | **2.5%** | BV=16 w=4 s=2 | 1.17× | **0.36%** |
| 4 | 18.2 µs/layer | 1379 GB/s | **4.5%** | BV=32 w=2 s=4 | 1.06× | **0.26%** |
| 8 | 37.6 µs/layer | 1337 GB/s | **9.3%** | BV=32 w=8 s=3 | 1.12× | **0.85%** |

Budget is 19.42 ms/token, i.e. 1/51.5 tok/s, the measured no-speculation baseline.

## Why the hypothesis was wrong

Two things, and only the first was anticipated.

**The kernel is already efficient where it is big.** At batch 4 and 8 it sustains
1337–1462 GB/s against the RTX 5090's 1792 GB/s spec peak — **75–82%**. There is no
factor-of-two sitting there. Batch 1 *is* occupancy-limited at 627 GB/s (35% of peak),
exactly as the 192-blocks-of-one-warp reading suggested, but batch 1 is also where the
kernel costs least in absolute terms: 0.48 ms of a 19.42 ms budget. The regime where the
inefficiency is real is the regime where it does not matter.

**The kernel is simply not where decode time goes.** Recurrent state traffic is
2 × B × HV × V × K × 4 B = 6.3 MB per layer per step at batch 1, ~302 MB across all 48
layers. Weights are 23.7 GB — **78× more traffic**. Computed against the same 1792 GB/s,
weight reads alone account for ~13.2 ms of the 19.42 ms step. The GDN state is a rounding
error next to the weights, which is the same conclusion the README reaches from the other
direction when it observes that baseline decode is flat from 256 to 8192 tokens.

## What the sweep did establish

**vLLM's hardcoded values are defensible, and the `min(…, 32)` cap on `BV` is load-bearing.**
`BV=128` with 1–2 warps runs at **0.18–0.25×** — a 4–5× regression — because the `[BV, BK]`
fp32 state tile is 4096 floats, which at one warp is 128 registers per thread for that
tile alone. The cap is what prevents that. `num_warps=1` costs 6–17% against the
per-batch optimum and never falls off a cliff; for an untuned default serving every GDN
model on every GPU, that is a reasonable place to sit.

There is no single winner across batch sizes (BV=16/w=4 at batch 1, BV=32/w=8 at batch 8),
so even a correct fix would need the per-device *and* per-batch config machinery that the
Mamba2 path has and this one lacks — for under 1%.

## Limits of this result

- **Isolated-kernel measurement.** It bounds the win from above. It does not prove the
  end-to-end effect, and no server run was done, because a ≤0.85% ceiling cannot be
  resolved by a harness whose baseline decode spread is 0.4–4.1 tok/s.
- **Synthetic inputs** (`torch.randn`), not captured activations. Timing for this kernel
  is data-independent — no data-dependent branching outside the NULL-state early-out — so
  this should not matter, but it was not verified against real activations.
- **Batch 1/4/8 only**, at `max_num_seqs 8`. Prefill uses a different path entirely
  (`chunk_gated_delta_rule`, CuTeDSL or FlashInfer), which was not measured.
- The 1792 GB/s peak is a **spec-sheet figure**, not a measured achievable bandwidth, so
  the "% of peak" column is a lower bound on true efficiency.

## Where the headroom actually is

Weight traffic is ~68% of the decode step. That leaves exactly two levers with real room,
and the repo has already pulled one:

1. **Amortize the weight read** — speculative decoding. Already done: MTP K3 gives 2.09×,
   and 107.6 tok/s is *above* the 75.5 tok/s single-token weight-bandwidth ceiling, which
   is only possible because MTP verifies several tokens per weight pass.
2. **Reduce the bytes** — a lighter quantization. The README already notes GGUF Q4_K of
   this model is ~15.9 GiB against NVFP4's 22.1 GiB.

Kernel-level work on the recurrent state is not on that list.
