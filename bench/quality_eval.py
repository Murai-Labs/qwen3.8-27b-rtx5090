#!/usr/bin/env python3
"""Quality evaluation for quantized Qwen3.8-27B, against an OpenAI-compatible endpoint.

Designed for comparing QUANTIZATIONS, where the differences are small. Three
things follow from that and are not optional:

1. PER-ITEM results are saved, so two runs can be compared PAIRWISE
   (quality_compare.py runs McNemar's test). Comparing two bare accuracy
   numbers with overlapping confidence intervals will miss a real effect at
   these sample sizes.

2. Decoding is deterministic (temperature 0) and every knob that affects the
   answer -- notably reasoning_effort -- is recorded in the output file. A
   comparison across runs with different reasoning_effort is meaningless;
   the compare script refuses it.

3. "No answer" is scored SEPARATELY from "wrong answer". A quant that runs out
   of context mid-thinking and emits nothing is failing differently from one
   that reasons to a wrong conclusion, and this model thinks at xhigh by
   default. Collapsing them hides the mechanism.

Datasets are read from the local HF cache -- no network, no `datasets` package.

Usage:
  python quality_eval.py --task gsm8k --n 200 --label nvfp4-mtp3 --out q_nvfp4.json
  python quality_eval.py --task mmlu  --n 200 --label nvfp4-mtp3 --out m_nvfp4.json
"""
import argparse
import concurrent.futures as cf
import glob
import json
import os
import re
import sys
import time
import urllib.request

HUB = os.environ.get("HF_HUB_CACHE") or os.path.expanduser("~/.cache/huggingface/hub")


# ---------------------------------------------------------------- datasets

def _parquet(pattern):
    import pyarrow.parquet as pq
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no parquet found for {pattern}\n"
                         f"(looked under HF cache {HUB})")
    return pq.read_table(files[-1]).to_pylist()


def load_gsm8k(n, seed=0):
    rows = _parquet(os.path.join(HUB, "datasets--openai--gsm8k", "snapshots", "*",
                                 "main", "test-*.parquet"))
    items = []
    for i, r in enumerate(rows):
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        items.append({"id": f"gsm8k-{i}", "prompt": r["question"].strip(), "gold": gold})
    return _sample(items, n, seed)


def load_mmlu(n, seed=0):
    files = sorted(glob.glob(os.path.join(HUB, "datasets--cais--mmlu", "snapshots", "*",
                                          "*", "test-*.parquet")))
    if not files:
        raise SystemExit("no MMLU test parquets in cache")
    import pyarrow.parquet as pq
    items = []
    for f in files:
        subject = os.path.basename(os.path.dirname(f))
        for i, r in enumerate(pq.read_table(f).to_pylist()):
            letters = "ABCD"
            opts = "\n".join(f"{letters[j]}. {c}" for j, c in enumerate(r["choices"]))
            items.append({
                "id": f"mmlu-{subject}-{i}",
                "prompt": f"{r['question'].strip()}\n\n{opts}",
                "gold": letters[int(r["answer"])],
                "subject": subject,
            })
    return _sample(items, n, seed)


def _sample(items, n, seed):
    """Deterministic subset -- the SAME items every run, which is what makes
    the paired comparison valid."""
    if n <= 0 or n >= len(items):
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


TASKS = {
    "gsm8k": (load_gsm8k,
              "Solve the problem. Put your final numeric answer on the last line as: "
              "ANSWER: <number>"),
    "mmlu": (load_mmlu,
             "Answer the multiple-choice question. Put your choice on the last line as: "
             "ANSWER: <letter>"),
}


# ---------------------------------------------------------------- scoring

def strip_think(text):
    """Everything after the final </think> is the answer. A response that never
    closes its think block has no answer at all."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip(), True
    if "<think>" in text:
        return "", False      # opened thinking and never finished
    return text.strip(), True


def extract(task, answer_text):
    m = re.findall(r"ANSWER:\s*([A-Za-z0-9\.\-,/]+)", answer_text)
    cand = m[-1].strip().rstrip(".") if m else None
    if task == "gsm8k":
        if cand is None:
            nums = re.findall(r"-?\d[\d,]*\.?\d*", answer_text)
            cand = nums[-1] if nums else None
        if cand is None:
            return None
        cand = cand.replace(",", "").rstrip(".")
        try:
            f = float(cand)
            return str(int(f)) if f == int(f) else str(f)
        except ValueError:
            return None
    if cand and cand[0].upper() in "ABCD":
        return cand[0].upper()
    m2 = re.findall(r"\b([A-D])\b", answer_text)
    return m2[-1] if m2 else None


def norm_gold(task, gold):
    if task == "gsm8k":
        # strip separators the same way extract() does. The loader already does
        # this, but if the two ever normalise differently a CORRECT answer gets
        # scored wrong and the bug is invisible in the output.
        gold = str(gold).replace(",", "").strip()
        try:
            f = float(gold)
            return str(int(f)) if f == int(f) else str(f)
        except ValueError:
            return gold
    return gold


# ---------------------------------------------------------------- endpoint

def ask(base_url, model, system, prompt, max_tokens, reasoning_effort, timeout):
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 0,
    }
    kw = {"preserve_thinking": False}
    if reasoning_effort:
        kw["reasoning_effort"] = reasoning_effort
    body["chat_template_kwargs"] = kw
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    msg = d["choices"][0]["message"]
    raw = msg.get("content") or ""
    # if a reasoning parser is enabled the thinking is split out already
    if msg.get("reasoning_content") and not raw.strip():
        return "", d["choices"][0].get("finish_reason"), d.get("usage", {}), False
    ans, closed = strip_think(raw)
    return ans, d["choices"][0].get("finish_reason"), d.get("usage", {}), closed


# ---------------------------------------------------------------- main

def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen38")
    ap.add_argument("--task", choices=sorted(TASKS), required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--reasoning-effort", default=None,
                    help="xhigh|low. NOT 'medium' -- it is a silent no-op in this "
                         "model's template. Omit for the default (xhigh).")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--label", required=True, help="what is being evaluated, e.g. 'nvfp4-mtp3'")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.reasoning_effort == "medium":
        sys.exit("refusing: reasoning_effort=medium is a silent no-op in this template "
                 "(validated as legal, no branch). Use xhigh, low, or omit.")

    loader, system = TASKS[a.task]
    items = loader(a.n)
    print(f"{a.task}: {len(items)} items | label={a.label} | "
          f"reasoning_effort={a.reasoning_effort or 'default(xhigh)'}", flush=True)

    results = [None] * len(items)
    t0 = time.time()
    done = 0

    def run(idx):
        it = items[idx]
        try:
            ans, finish, usage, closed = ask(a.base_url, a.model, system, it["prompt"],
                                             a.max_tokens, a.reasoning_effort, a.timeout)
        except Exception as e:
            return idx, {"id": it["id"], "outcome": "error", "detail": str(e)[:120]}
        got = extract(a.task, ans) if ans else None
        gold = norm_gold(a.task, it["gold"])
        if got is None:
            outcome = "no_answer"
        elif got == gold:
            outcome = "correct"
        else:
            outcome = "incorrect"
        return idx, {"id": it["id"], "outcome": outcome, "got": got, "gold": gold,
                     "finish_reason": finish, "think_closed": closed,
                     "completion_tokens": (usage or {}).get("completion_tokens"),
                     "subject": it.get("subject")}

    with cf.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        for idx, res in ex.map(run, range(len(items))):
            results[idx] = res
            done += 1
            if done % 20 == 0 or done == len(items):
                el = time.time() - t0
                rate = done / el if el else 0
                eta = (len(items) - done) / rate if rate else 0
                ok = sum(1 for r in results if r and r["outcome"] == "correct")
                print(f"  {done}/{len(items)}  correct={ok}  "
                      f"{el:.0f}s elapsed, ETA {eta:.0f}s", flush=True)

    n = len(results)
    counts = {k: sum(1 for r in results if r["outcome"] == k)
              for k in ("correct", "incorrect", "no_answer", "error")}
    lo, hi = wilson(counts["correct"], n)
    toks = [r.get("completion_tokens") for r in results if r.get("completion_tokens")]
    out = {
        "label": a.label, "task": a.task, "n": n,
        "model": a.model, "base_url": a.base_url,
        "reasoning_effort": a.reasoning_effort or "default(xhigh)",
        "max_tokens": a.max_tokens, "temperature": 0,
        "preserve_thinking": False,
        "accuracy": counts["correct"] / n if n else 0,
        "accuracy_ci95": [lo, hi],
        "counts": counts,
        "median_completion_tokens": sorted(toks)[len(toks) // 2] if toks else None,
        "elapsed_s": round(time.time() - t0, 1),
        "results": results,
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)

    print(f"\n{a.label} / {a.task}: accuracy {out['accuracy']*100:.1f}% "
          f"(95% CI {lo*100:.1f}-{hi*100:.1f}), n={n}")
    print(f"  correct={counts['correct']} incorrect={counts['incorrect']} "
          f"no_answer={counts['no_answer']} error={counts['error']}")
    if counts["no_answer"]:
        print(f"  NOTE: {counts['no_answer']} items produced no answer -- check whether "
              f"thinking exhausted max_tokens ({a.max_tokens}); that is a different "
              f"failure from being wrong.")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
