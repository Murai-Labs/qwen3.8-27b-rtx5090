#!/usr/bin/env python3
"""Turn bench_fixed.py arm outputs into the markdown tables used in the README.

Usage: python bench/summarize.py results/armA.json results/armB.json ...

Reports decode tok/s per (prompt_tokens, concurrency) cell for each arm, and the
delta of each arm against the first one given, which is treated as the baseline.
Prints spread so a difference can be told apart from noise.
"""
import json
import sys
from collections import OrderedDict

ARM_LABEL = {
    "armA": "A: baseline (no spec decode)",
    "armB": "B: MTP K2",
    "armC": "C: MTP K3",
}


def load(path):
    with open(path) as f:
        rows = json.load(f)
    if isinstance(rows, dict):
        rows = rows.get("rows", [rows])
    return rows


def key(r):
    return (int(r["prompt_tokens"]), int(r["concurrency"]))


def main(paths):
    arms = OrderedDict()
    for p in paths:
        rows = load(p)
        label = rows[0].get("label", p) if rows else p
        arms[label] = {key(r): r for r in rows}

    labels = list(arms)
    base = labels[0]
    cells = sorted({k for a in arms.values() for k in a})

    print("| prompt | conc | " + " | ".join(ARM_LABEL.get(l, l) for l in labels) +
          (" | vs baseline |" if len(labels) > 1 else " |"))
    print("|---|---|" + "---|" * (len(labels) + (1 if len(labels) > 1 else 0)))
    for c in cells:
        vals = []
        for l in labels:
            r = arms[l].get(c)
            vals.append(f"{r['decode_tok_s']:.1f}" if r else "—")
        line = f"| {c[0]} | {c[1]} | " + " | ".join(vals)
        if len(labels) > 1:
            b = arms[base].get(c)
            t = arms[labels[-1]].get(c)
            if b and t and b["decode_tok_s"]:
                line += f" | {100 * (t['decode_tok_s'] / b['decode_tok_s'] - 1):+.1f}% |"
            else:
                line += " | — |"
        else:
            line += " |"
        print(line)

    print()
    print("Spread (max-min decode tok/s across repeats) and output-length check:")
    for l in labels:
        for c in cells:
            r = arms[l].get(c)
            if not r:
                continue
            fixed = r.get("out_tok_min") == r.get("out_tok_max") == r.get("max_tokens")
            flag = "" if fixed else "  ** OUTPUT LENGTH VARIED - decode number is confounded **"
            print(f"  {l} p{c[0]}/c{c[1]}: spread {r.get('decode_spread', float('nan')):.1f}"
                  f"  out_tok {r.get('out_tok_min')}-{r.get('out_tok_max')}{flag}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
