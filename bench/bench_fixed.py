#!/usr/bin/env python3
"""Fixed-output benchmark for DeepSeek-V4 on vLLM.

Addresses three confounds in the ad-hoc harnesses:
  1. Output-length variance  -> fixed max_tokens + ignore_eos, so every request
     emits exactly the same number of tokens. (Observed elsewhere: 52.21 vs
     23.03 tok/s on identical inputs purely from 1980 vs 837 output tokens.)
  2. Prefix-cache inflation  -> a random nonce per request, so repeats cannot be
     served from cache and inflate prefill.
  3. Step-vs-token counting  -> completion token counts come from the server's
     own `usage.completion_tokens`, never from counting SSE deltas (spec decode
     emits several tokens per chunk, which undercounts by ~4x).

Reports median and spread across repeats so an effect can be distinguished from
noise rather than asserted from a single sample.
"""
import argparse, json, random, statistics, string, sys, time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def nonce(n=24):
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))


def make_prompt(target_tokens):
    # ~4 chars/token; nonce first so no two requests share a prefix.
    filler = ("The quick brown fox jumps over the lazy dog. " * ((target_tokens * 4) // 45 + 2))
    return f"[{nonce()}] {filler[: target_tokens * 4]}\n\nSummarise the above."


def one_request(base_url, model, prompt, max_tokens, timeout):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,      # force exactly max_tokens where supported
        "ignore_eos": True,            # do not stop early -> fixed output length
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first = None      # first content chunk
    t_last = None       # last content chunk
    n_chunks = 0
    usage = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except Exception:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            ch = (chunk.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            if delta.get("content") or delta.get("reasoning_content"):
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                t_last = now
                n_chunks += 1
    t1 = time.perf_counter()
    if usage is None:
        raise RuntimeError("server returned no usage block; cannot count tokens reliably")
    out_tok = usage.get("completion_tokens", 0)
    in_tok = usage.get("prompt_tokens", 0)
    if t_first is None:
        raise RuntimeError("no content chunks received")
    ttft = t_first - t0
    # Decode rate over the generation window (first->last content chunk). Do NOT
    # use total-minus-ttft: when the tail arrives in one burst that window
    # collapses toward zero and the rate explodes to ~1e11.
    window = t_last - t_first
    valid = (n_chunks >= 3) and (window >= 0.10)
    decode_tok_s = (out_tok / window) if valid else float("nan")
    return {"t0": t0, "t1": t1, "ttft": ttft, "out_tok": out_tok, "in_tok": in_tok,
            "n_chunks": n_chunks, "window": window, "valid": valid,
            "decode_tok_s": decode_tok_s, "prefill_tok_s": in_tok / max(ttft, 1e-9)}


def run_cell(base_url, model, prompt_tokens, conc, max_tokens, timeout):
    prompts = [make_prompt(prompt_tokens) for _ in range(conc)]
    with ThreadPoolExecutor(max_workers=conc) as ex:
        rs = list(ex.map(lambda p: one_request(base_url, model, p, max_tokens, timeout), prompts))
    total_out = sum(r["out_tok"] for r in rs)
    wall = max(r["t1"] for r in rs) - min(r["t0"] for r in rs)
    good = [r for r in rs if r["valid"]]
    return {
        "decode_tok_s": statistics.median(r["decode_tok_s"] for r in good) if good else float("nan"),
        "n_valid": len(good), "n_total": len(rs),
        "prefill_tok_s": statistics.median(r["prefill_tok_s"] for r in rs),
        "ttft_s": statistics.median(r["ttft"] for r in rs),
        "aggregate_tok_s": total_out / wall,
        "out_tok_min": min(r["out_tok"] for r in rs),
        "out_tok_max": max(r["out_tok"] for r in rs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--prompt-tokens", default="2048")
    ap.add_argument("--concurrency", default="1,4,8")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--label", default="run")
    ap.add_argument("--output", default=None)
    a = ap.parse_args()

    plens = [int(x) for x in a.prompt_tokens.split(",")]
    concs = [int(x) for x in a.concurrency.split(",")]

    print(f"warming ({a.warmup} requests)...", flush=True)
    for _ in range(a.warmup):
        try:
            one_request(a.base_url, a.model, make_prompt(512), a.max_tokens, a.timeout)
        except Exception as e:
            print("  warmup error:", e, flush=True)

    rows = []
    print(f"\n{'plen':>6} {'conc':>4} {'prefill':>9} {'decode':>8} {'±':>6} {'ttft':>7} {'aggregate':>10} {'±':>7}", flush=True)
    print("-" * 68, flush=True)
    for pl in plens:
        for c in concs:
            reps = []
            for _ in range(a.repeats):
                try:
                    reps.append(run_cell(a.base_url, a.model, pl, c, a.max_tokens, a.timeout))
                except Exception as e:
                    print(f"  cell p{pl} c{c} error: {e}", flush=True)
            if not reps:
                continue
            dec = [r["decode_tok_s"] for r in reps]
            agg = [r["aggregate_tok_s"] for r in reps]
            row = {
                "label": a.label, "prompt_tokens": pl, "concurrency": c,
                "max_tokens": a.max_tokens, "repeats": len(reps),
                "prefill_tok_s": statistics.median(r["prefill_tok_s"] for r in reps),
                "decode_tok_s": statistics.median(dec),
                "decode_spread": (max(dec) - min(dec)),
                "ttft_s": statistics.median(r["ttft_s"] for r in reps),
                "aggregate_tok_s": statistics.median(agg),
                "aggregate_spread": (max(agg) - min(agg)),
                "out_tok_min": min(r["out_tok_min"] for r in reps),
                "out_tok_max": max(r["out_tok_max"] for r in reps),
            }
            rows.append(row)
            print(f"{pl:>6} {c:>4} {row['prefill_tok_s']:>9.1f} {row['decode_tok_s']:>8.1f} "
                  f"{row['decode_spread']:>6.1f} {row['ttft_s']:>7.2f} "
                  f"{row['aggregate_tok_s']:>10.1f} {row['aggregate_spread']:>7.1f}", flush=True)

    if a.output:
        with open(a.output, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"\nwrote {a.output}", flush=True)
    bad = [r for r in rows if r["out_tok_min"] != r["out_tok_max"]]
    if bad:
        print(f"\nWARNING: output length varied in {len(bad)} cell(s) — "
              f"ignore_eos/min_tokens may be unsupported; decode numbers are then confounded.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
