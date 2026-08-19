#!/usr/bin/env python3
"""Write a copy of an NVFP4 checkpoint whose `lm_head` is BF16 instead of FP8.

Why this exists. DFlash 2's candidate selector reads the target model's top-K
logits directly off the LM head and refuses to run against a quantized one:

    ValueError: DFlash2 requires an unquantized target LM head for candidate TopK.
    (vllm/model_executor/models/qwen3_dflash2.py:322)

`unsloth/Qwen3.8-27B-NVFP4` quantizes `lm_head` to FP8 W8A8 -- it is matched by
`re:.*lm_head` in the first `config_groups` entry -- so DFlash 2 cannot serve it.

Public NVFP4 conversions of this model that leave `lm_head` alone do exist
(`gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090`,
`sakamakismile/Qwen3.8-27B-MTP-NVFP4`). We do not use one, because swapping the
checkpoint changes every weight at once: the DFlash 2 number would no longer be
comparable to the MTP number this repo already published, and the fp8 KV
calibration scales this checkpoint ships (README: "never override KV dtype")
would come from a different calibration or, for the sakamakismile conversion,
not exist at all. Dequantising one tensor changes one tensor.

What it does:
  lm_head.weight (F8_E4M3, [V, H]) * lm_head.weight_scale (BF16, [V, 1]) -> BF16
  drops lm_head.weight_scale
  moves `lm_head` out of the quantized targets and into `ignore` in config.json
  everything else is copied byte-for-byte

Cost: +1.27 GiB of weights on a 32 GiB card (FP8 1.27 GiB -> BF16 2.54 GiB).

Usage:
    python deploy/dequant_lm_head.py ~/models/Qwen3.8-27B-NVFP4 \
                                     ~/models/Qwen3.8-27B-NVFP4-lmheadbf16
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SHARD_BYTES = 8 * 1024**3  # keep peak RSS near this, not near the whole checkpoint


def log(msg):
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--weights-file", default="model.safetensors")
    a = ap.parse_args()
    src, dst = Path(a.src).expanduser(), Path(a.dst).expanduser()
    if not (src / a.weights_file).exists():
        sys.exit(f"no {a.weights_file} in {src}")
    dst.mkdir(parents=True, exist_ok=True)

    # ---- config.json: lm_head stops being a quantization target ---------------
    cfg = json.loads((src / "config.json").read_text())
    q = cfg["quantization_config"]
    moved = 0
    for gname, group in q.get("config_groups", {}).items():
        keep = [t for t in group.get("targets", []) if "lm_head" not in t]
        moved += len(group.get("targets", [])) - len(keep)
        group["targets"] = keep
    if moved == 0:
        log("NOTE: no lm_head target found in config_groups; nothing to move")
    q.setdefault("ignore", [])
    if "lm_head" not in q["ignore"]:
        q["ignore"].append("lm_head")
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    log(f"config.json: removed {moved} lm_head target(s), added lm_head to ignore")

    # ---- weights --------------------------------------------------------------
    f = safe_open(str(src / a.weights_file), framework="pt")
    names = list(f.keys())
    sizes = {}
    for n in names:
        sl = f.get_slice(n)
        shape = sl.get_shape()
        nel = 1
        for d in shape:
            nel *= d
        # F8/U8 are 1 byte, BF16 2, F32 4 -- only used to pick shard boundaries
        w = {"F8_E4M3": 1, "U8": 1, "BF16": 2, "F16": 2, "F32": 4}.get(sl.get_dtype(), 4)
        sizes[n] = nel * w
    # lm_head.weight doubles in width, weight_scale disappears
    sizes["lm_head.weight"] *= 2
    names = [n for n in names if n != "lm_head.weight_scale"]

    shards, cur, cur_bytes = [], [], 0
    for n in names:
        if cur and cur_bytes + sizes[n] > SHARD_BYTES:
            shards.append(cur)
            cur, cur_bytes = [], 0
        cur.append(n)
        cur_bytes += sizes[n]
    if cur:
        shards.append(cur)
    log(f"{len(names)} tensors -> {len(shards)} shard(s)")

    weight_map, total = {}, 0
    for i, shard in enumerate(shards, 1):
        fname = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        tensors = {}
        for n in shard:
            if n == "lm_head.weight":
                w = f.get_tensor(n)
                s = f.get_tensor("lm_head.weight_scale")
                t = (w.to(torch.float32) * s.to(torch.float32)).to(torch.bfloat16)
                log(f"  lm_head.weight {tuple(w.shape)} {w.dtype} -> {t.dtype} "
                    f"(scale {tuple(s.shape)}, |w|max {t.abs().max().item():.4f})")
                del w, s
            else:
                t = f.get_tensor(n)
            tensors[n] = t
            weight_map[n] = fname
            total += t.numel() * t.element_size()
        save_file(tensors, str(dst / fname), metadata={"format": "pt"})
        log(f"  wrote {fname}  ({len(shard)} tensors)")
        del tensors

    # ---- the MTP head and the small files ride along unchanged -----------------
    for extra in sorted(src.glob("*.safetensors")):
        if extra.name == a.weights_file:
            continue
        link = dst / extra.name
        if not link.exists():
            os.symlink(extra.resolve(), link)
        g = safe_open(str(extra), framework="pt")
        for n in g.keys():
            weight_map[n] = extra.name
            total += 0  # size not needed beyond the index metadata below
        log(f"  linked {extra.name}")

    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2))

    for small in src.iterdir():
        if small.suffix == ".safetensors" or small.name in (
                "config.json", "model.safetensors.index.json"):
            continue
        if small.is_file():
            shutil.copy2(small, dst / small.name)
    log(f"done -> {dst}")


if __name__ == "__main__":
    main()
