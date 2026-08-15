#!/usr/bin/env python3
"""Paired comparison of two quality_eval.py runs, via McNemar's exact test.

Why paired: quantization effects are small. With n=200 and ~85% accuracy, the
95% CI on a single run is roughly +/-5 points, so two independent runs can
differ by 4 points with fully overlapping intervals and tell you nothing. But
the runs answered the SAME items, so the informative quantity is how many items
FLIPPED in each direction -- which is exactly McNemar's test.

Exact binomial is used rather than the chi-square approximation, because the
discordant counts are usually small, which is where the approximation is worst.

Usage:
  python quality_compare.py baseline.json candidate.json
"""
import json
import sys
from math import comb


def load(p):
    with open(p) as f:
        return json.load(f)


def mcnemar_exact_two_sided(b, c):
    """b = A right / B wrong, c = A wrong / B right.
    Under H0 each discordant pair is a fair coin flip."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main(pa, pb):
    A, B = load(pa), load(pb)

    if A["task"] != B["task"]:
        sys.exit(f"refusing: different tasks ({A['task']} vs {B['task']})")
    if A["reasoning_effort"] != B["reasoning_effort"]:
        sys.exit(f"refusing: different reasoning_effort "
                 f"({A['reasoning_effort']} vs {B['reasoning_effort']}) -- "
                 f"the comparison would not be attributable to the quantization")
    if A.get("max_tokens") != B.get("max_tokens"):
        sys.exit(f"refusing: different max_tokens ({A.get('max_tokens')} vs "
                 f"{B.get('max_tokens')}) -- affects the no_answer rate")

    ra = {r["id"]: r for r in A["results"]}
    rb = {r["id"]: r for r in B["results"]}
    ids = [i for i in ra if i in rb]
    if len(ids) != len(ra) or len(ids) != len(rb):
        print(f"warning: comparing on {len(ids)} shared items "
              f"({len(ra)} vs {len(rb)} present)")

    ok = lambda r: r["outcome"] == "correct"
    both = sum(1 for i in ids if ok(ra[i]) and ok(rb[i]))
    only_a = sum(1 for i in ids if ok(ra[i]) and not ok(rb[i]))
    only_b = sum(1 for i in ids if not ok(ra[i]) and ok(rb[i]))
    neither = sum(1 for i in ids if not ok(ra[i]) and not ok(rb[i]))

    n = len(ids)
    acc_a, acc_b = (both + only_a) / n, (both + only_b) / n
    p = mcnemar_exact_two_sided(only_a, only_b)

    print(f"task={A['task']}  n={n}  reasoning_effort={A['reasoning_effort']}  "
          f"max_tokens={A['max_tokens']}")
    print(f"\n  A = {A['label']:28s} accuracy {acc_a*100:5.1f}%")
    print(f"  B = {B['label']:28s} accuracy {acc_b*100:5.1f}%")
    print(f"  difference (B - A): {(acc_b-acc_a)*100:+.1f} points")

    print(f"\n  paired table")
    print(f"    both correct        {both:4d}")
    print(f"    only A correct      {only_a:4d}   <- B lost these")
    print(f"    only B correct      {only_b:4d}   <- B gained these")
    print(f"    both wrong          {neither:4d}")
    print(f"\n  McNemar exact two-sided p = {p:.4f}")
    if p < 0.05:
        better = B["label"] if only_b > only_a else A["label"]
        print(f"  -> significant at 0.05: {better} is better on this task")
    else:
        print(f"  -> NOT significant at 0.05. With {only_a + only_b} discordant items "
              f"this test cannot distinguish them;")
        print(f"     that is not evidence they are equivalent, only that this n is "
              f"too small to tell.")

    # failure-mode shift matters independently of accuracy
    print("\n  failure modes")
    for lbl, R in ((A["label"], A), (B["label"], B)):
        c = R["counts"]
        print(f"    {lbl:28s} no_answer={c['no_answer']:3d}  error={c['error']:3d}  "
              f"median_completion_tokens={R.get('median_completion_tokens')}")
    if A["counts"]["no_answer"] != B["counts"]["no_answer"]:
        print("    NOTE: no_answer rates differ -- one config is running out of "
              "generation budget mid-thinking more often. That is a separate defect "
              "from answering wrongly.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
