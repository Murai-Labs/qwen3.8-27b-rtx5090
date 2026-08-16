#!/usr/bin/env python3
"""Microbenchmark + launch-parameter sweep for vLLM's GDN decode kernel.

48 of this model's 64 layers are Gated DeltaNet, and every one of them calls
`fused_recurrent_gated_delta_rule_packed_decode` once per decode step. That
wrapper (vllm/third_party/flash_linear_attention/ops/fused_recurrent.py) hardcodes
its launch parameters:

    BV         = min(next_power_of_2(V), 32)   -> 32 here
    num_warps  = 1
    num_stages = 3

with no @triton.autotune and no per-device config file -- unlike vLLM's Mamba2
`selective_state_update`, which ships tuned JSON for B200/GB200/H100/H200/MI300X
and has no RTX 5090 entry either.

For this model (H=16, HV=48, K=128, V=128) that means grid = (V/BV, B*HV) =
(4, 48) at batch 1: 192 blocks of one warp each, on a 170-SM GPU. This script
measures whether that costs anything.

Two questions, in order:

  1. WHERE DOES THE TIME GO. Kernel time x 48 layers, against the measured
     19.4 ms/token decode budget (1/51.5 tok/s baseline). If the answer is
     0.1%, stop -- the kernel is not worth tuning and no sweep result matters.
  2. IS THE SHIPPED CONFIG OPTIMAL. Sweep BV x num_warps x num_stages, gated on
     numerical agreement with the shipped config.

Every candidate must pass the correctness gate before its timing is reported.
Changing BV changes the reduction split over V, so exact bitwise equality is not
expected; max |diff| against the shipped config is reported for every row and
anything above --atol is marked FAIL and excluded from the ranking.

Usage (inside the vLLM venv):
    python bench/kernel_gdn_decode.py --batches 1,4,8 --out bench/results/kernel_gdn.json
"""
import argparse
import itertools
import json
import sys
import time

import torch

try:
    import triton
    from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode_kernel as KERNEL,
    )
except ImportError as e:  # pragma: no cover - environment problem, not a bug
    sys.exit(f"import failed ({e}); run this with the vLLM venv python")

# Qwen3.8-27B GDN geometry, read from the checkpoint's config.json:
#   linear_num_key_heads=16, linear_num_value_heads=48,
#   linear_key_head_dim=128, linear_value_head_dim=128
H, HV, K, V = 16, 48, 128, 128
N_GDN_LAYERS = 48          # 16 blocks x [3 x GatedDeltaNet -> 1 x GatedAttention]
BASELINE_TOK_S = 51.5      # measured, README.md "Measured throughput", p256 c1

SHIPPED = {"BV": min(triton.next_power_of_2(V), 32), "num_warps": 1, "num_stages": 3}


def make_inputs(batch, state_dtype, io_dtype, device, seed=0):
    """Allocate one decode step's worth of GDN inputs.

    State index 0 is NULL_BLOCK_ID (the kernel early-returns on it), so the state
    pool holds batch+1 slots and sequences use indices 1..batch.
    """
    g = torch.Generator(device=device).manual_seed(seed)

    def rnd(*shape, dtype):
        return torch.randn(*shape, generator=g, device=device, dtype=torch.float32).to(dtype)

    qkv_dim = 2 * H * K + HV * V
    return {
        "mixed_qkv": rnd(batch, qkv_dim, dtype=io_dtype),
        "a": rnd(batch, HV, dtype=torch.float32),
        "b": rnd(batch, HV, dtype=torch.float32),
        "A_log": rnd(HV, dtype=torch.float32),
        "dt_bias": rnd(HV, dtype=torch.float32),
        "state": rnd(batch + 1, HV, V, K, dtype=state_dtype) * 0.1,
        "out": torch.zeros(batch, 1, HV, V, device=device, dtype=io_dtype),
        "idx": torch.arange(1, batch + 1, device=device, dtype=torch.int32),
    }


def launch(t, BV, num_warps, num_stages):
    """Call the raw Triton kernel, bypassing the wrapper's hardcoded launch params."""
    batch = t["mixed_qkv"].shape[0]
    grid = (triton.cdiv(V, BV), batch * HV)
    KERNEL[grid](
        mixed_qkv=t["mixed_qkv"], a=t["a"], b=t["b"],
        A_log=t["A_log"], dt_bias=t["dt_bias"],
        o=t["out"], h0=t["state"], ht=t["state"], ssm_state_indices=t["idx"],
        scale=K ** -0.5,
        stride_mixed_qkv_tok=t["mixed_qkv"].stride(0),
        stride_a_tok=t["a"].stride(0),
        stride_b_tok=t["b"].stride(0),
        stride_init_state_token=t["state"].stride(0),
        stride_final_state_token=t["state"].stride(0),
        stride_indices_seq=t["idx"].stride(0),
        H=H, HV=HV, K=K, V=V, BK=triton.next_power_of_2(K), BV=BV,
        SOFTPLUS_THRESHOLD=20.0, USE_QK_L2NORM_IN_KERNEL=True,
        num_warps=num_warps, num_stages=num_stages,
    )
    return grid


def run_once(batch, state_dtype, io_dtype, device, BV, num_warps, num_stages, seed=0):
    """One clean invocation from a freshly seeded state. Returns (out, final_state).

    The kernel updates state in place (h0 and ht alias), so every candidate must
    start from an identical state or the comparison is meaningless.
    """
    t = make_inputs(batch, state_dtype, io_dtype, device, seed=seed)
    launch(t, BV, num_warps, num_stages)
    torch.cuda.synchronize()
    return t["out"].clone(), t["state"].clone()


def bench(batch, state_dtype, io_dtype, device, BV, num_warps, num_stages):
    t = make_inputs(batch, state_dtype, io_dtype, device)
    launch(t, BV, num_warps, num_stages)          # warm/compile outside the timer
    torch.cuda.synchronize()
    ms = triton.testing.do_bench(
        lambda: launch(t, BV, num_warps, num_stages), warmup=25, rep=100
    )
    return ms


def state_bytes(batch, state_dtype):
    """Recurrent state traffic per layer per step: read h0 + write ht."""
    return 2 * batch * HV * V * K * torch.tensor([], dtype=state_dtype).element_size()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,4,8",
                    help="decode batch sizes; max_num_seqs is 8 in deploy/serve.sh")
    ap.add_argument("--state-dtype", default="float32", choices=["float32", "float16"],
                    help="float32 is the model default; float16 is --mamba-ssm-cache-dtype")
    ap.add_argument("--bv", default="16,32,64,128")
    ap.add_argument("--warps", default="1,2,4,8")
    ap.add_argument("--stages", default="1,2,3,4")
    ap.add_argument("--atol", type=float, default=2e-2,
                    help="max |diff| vs the shipped config allowed before a row FAILs")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("no CUDA device")
    device = "cuda"
    props = torch.cuda.get_device_properties(0)
    state_dtype = getattr(torch, a.state_dtype)
    io_dtype = torch.bfloat16

    batches = [int(x) for x in a.batches.split(",")]
    grid_bv = [int(x) for x in a.bv.split(",")]
    warps = [int(x) for x in a.warps.split(",")]
    stages = [int(x) for x in a.stages.split(",")]
    combos = list(itertools.product(grid_bv, warps, stages))
    total = len(batches) * len(combos)

    print(f"GPU        : {props.name}  sm_{props.major}{props.minor}  "
          f"{props.multi_processor_count} SMs", flush=True)
    print(f"torch      : {torch.__version__}   triton: {triton.__version__}", flush=True)
    print(f"geometry   : H={H} HV={HV} K={K} V={V}, {N_GDN_LAYERS} GDN layers", flush=True)
    print(f"state dtype: {a.state_dtype}   io dtype: bfloat16", flush=True)
    print(f"shipped    : BV={SHIPPED['BV']} num_warps={SHIPPED['num_warps']} "
          f"num_stages={SHIPPED['num_stages']}", flush=True)
    print(f"sweep      : {total} configs ({len(combos)} per batch size)\n", flush=True)

    results = []
    t_start = time.perf_counter()
    done = 0

    for batch in batches:
        # Reference = the shipped config, from a fixed seed. Everything is compared
        # against this, so a "faster" config that is wrong cannot win.
        ref_out, ref_state = run_once(batch, state_dtype, io_dtype, device,
                                      SHIPPED["BV"], SHIPPED["num_warps"],
                                      SHIPPED["num_stages"])
        ship_ms = bench(batch, state_dtype, io_dtype, device,
                        SHIPPED["BV"], SHIPPED["num_warps"], SHIPPED["num_stages"])
        traffic = state_bytes(batch, state_dtype)

        print(f"=== batch {batch} ===  shipped: {ship_ms * 1000:.1f} us/layer, "
              f"{ship_ms * N_GDN_LAYERS:.3f} ms/token over {N_GDN_LAYERS} layers "
              f"({ship_ms * N_GDN_LAYERS / (1000 / BASELINE_TOK_S) * 100:.1f}% of the "
              f"{1000 / BASELINE_TOK_S:.1f} ms/token budget)", flush=True)
        print(f"{'BV':>4} {'warps':>6} {'stages':>7} {'blocks':>7} {'us':>8} "
              f"{'GB/s':>7} {'vs ship':>8} {'maxdiff':>9}  status", flush=True)
        print("-" * 72, flush=True)

        for BV, nw, ns in combos:
            done += 1
            if done % 10 == 0:
                el = time.perf_counter() - t_start
                eta = el / done * (total - done)
                print(f"  [{done}/{total}] {el:.0f}s elapsed, ~{eta:.0f}s left", flush=True)
            row = {"batch": batch, "BV": BV, "num_warps": nw, "num_stages": ns,
                   "state_dtype": a.state_dtype,
                   "shipped": (BV, nw, ns) == (SHIPPED["BV"], SHIPPED["num_warps"],
                                               SHIPPED["num_stages"])}
            try:
                out, state = run_once(batch, state_dtype, io_dtype, device, BV, nw, ns)
                d_out = (out.float() - ref_out.float()).abs().max().item()
                d_state = (state.float() - ref_state.float()).abs().max().item()
                maxdiff = max(d_out, d_state)
                ok = maxdiff <= a.atol and not (torch.isnan(out).any() or
                                                torch.isnan(state).any())
                ms = bench(batch, state_dtype, io_dtype, device, BV, nw, ns)
                row.update({"ok": ok, "maxdiff": maxdiff, "us": ms * 1000,
                            "us_per_token": ms * N_GDN_LAYERS * 1000,
                            "gbps": traffic / (ms * 1e-3) / 1e9,
                            "speedup_vs_shipped": ship_ms / ms,
                            "blocks": triton.cdiv(V, BV) * batch * HV})
                status = "ok" if ok else f"FAIL diff>{a.atol}"
                mark = "  <- shipped" if row["shipped"] else ""
                print(f"{BV:>4} {nw:>6} {ns:>7} {row['blocks']:>7} {row['us']:>8.1f} "
                      f"{row['gbps']:>7.0f} {row['speedup_vs_shipped']:>7.2f}x "
                      f"{maxdiff:>9.2e}  {status}{mark}", flush=True)
            except Exception as e:
                row.update({"ok": False, "error": str(e)[:200]})
                print(f"{BV:>4} {nw:>6} {ns:>7} {'-':>7} {'-':>8} {'-':>7} {'-':>8} "
                      f"{'-':>9}  ERROR {str(e)[:60]}", flush=True)
            results.append(row)
        print(flush=True)

    print("=" * 72, flush=True)
    for batch in batches:
        rows = [r for r in results if r["batch"] == batch and r.get("ok")]
        if not rows:
            print(f"batch {batch}: no config passed the correctness gate", flush=True)
            continue
        best = min(rows, key=lambda r: r["us"])
        ship = next((r for r in results if r["batch"] == batch and r["shipped"]), None)
        print(f"batch {batch}: best BV={best['BV']} warps={best['num_warps']} "
              f"stages={best['num_stages']} -> {best['us']:.1f} us "
              f"({best['speedup_vs_shipped']:.2f}x vs shipped)", flush=True)
        if ship and ship.get("ok"):
            saved = (ship["us_per_token"] - best["us_per_token"]) / 1000
            budget = 1000 / BASELINE_TOK_S
            print(f"          per token over {N_GDN_LAYERS} layers: "
                  f"{ship['us_per_token'] / 1000:.3f} -> {best['us_per_token'] / 1000:.3f} ms "
                  f"(saves {saved:.3f} ms of a {budget:.1f} ms budget = "
                  f"{saved / budget * 100:.2f}% end-to-end IF decode is serial in this kernel)",
                  flush=True)

    print("\nCaveat: this is an isolated-kernel measurement. It bounds the win from "
          "above; it does not prove the end-to-end effect, which needs a server run.",
          flush=True)

    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"gpu": props.name, "sms": props.multi_processor_count,
                       "torch": torch.__version__, "triton": triton.__version__,
                       "geometry": {"H": H, "HV": HV, "K": K, "V": V,
                                    "n_gdn_layers": N_GDN_LAYERS},
                       "shipped": SHIPPED, "baseline_tok_s": BASELINE_TOK_S,
                       "rows": results}, fh, indent=2)
        print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
