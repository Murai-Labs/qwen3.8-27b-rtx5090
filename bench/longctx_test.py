#!/usr/bin/env python3
"""Long-context recall + throughput probe.

Two things are checked at each depth:

  1. RECALL -- a unique needle is planted at ~60% depth in NON-REPEATING filler.
     The filler is unique per line (a counter plus a hashed token), so the model
     cannot answer by pattern-matching a repeated blob, and a prefix cache
     cannot serve it. Scored by substring match on content AND reasoning: this
     model puts answers in `reasoning` on long prompts, and scoring `content`
     only makes a working model look broken.

  2. DEGENERATION -- the same repetition-collapse detector used for the short
     probes, because speculative decoding is reported to garble output at long
     context. A high accuracy number means nothing if the text is "a a a a".

Usage:
  python longctx_test.py --depths 8192,65536,131072,196608 --label mtp3-4bit
"""
import argparse
import hashlib
import json
import time
import urllib.request


def filler_line(i):
    h = hashlib.sha1(f"line-{i}".encode()).hexdigest()[:12]
    return (f"Record {i:07d}: inventory item {h} logged at station {i % 97} "
            f"with checksum {h[::-1]} and batch tag {i * 7919 % 100003}.")


def build_prompt(target_tokens, needle_id, depth_frac=0.6):
    """~4 chars/token. Needle planted at depth_frac through the filler."""
    target_chars = target_tokens * 4
    needle = (f"NEEDLE-{needle_id}: the authorised maintenance status is "
              f"VERIFIED-{needle_id}-OK.")
    lines, n = [], 0
    i = 0
    while n < target_chars:
        line = filler_line(i)
        lines.append(line)
        n += len(line) + 1
        i += 1
    pos = int(len(lines) * depth_frac)
    lines.insert(pos, needle)
    body = "\n".join(lines)
    q = (f"\n\nQuestion: find the line beginning with 'NEEDLE-{needle_id}'. "
         f"Reply with the authorised maintenance status exactly as written.")
    return body + q, f"VERIFIED-{needle_id}-OK"


def max_consecutive_repeat(text):
    t = text.split()
    if not t:
        return 0
    worst = cur = 1
    for i in range(1, len(t)):
        cur = cur + 1 if t[i] == t[i - 1] else 1
        worst = max(worst, cur)
    return worst


def ask(base_url, model, prompt, max_tokens, timeout, effort):
    body = {"model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"preserve_thinking": False}}
    if effort:
        body["chat_template_kwargs"]["reasoning_effort"] = effort
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    el = time.time() - t0
    m = d["choices"][0]["message"]
    return (m.get("content") or ""), (m.get("reasoning_content") or ""), \
        d["choices"][0].get("finish_reason"), d.get("usage", {}), el


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen38")
    ap.add_argument("--depths", default="8192,65536,131072,196608")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--reasoning-effort", default="low")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = []
    for d in [int(x) for x in a.depths.split(",")]:
        nid = f"{d:06d}"
        prompt, gold = build_prompt(d, nid)
        approx = len(prompt) // 4
        try:
            c, rc, fr, usage, el = ask(a.base_url, a.model, prompt,
                                       a.max_tokens, a.timeout, a.reasoning_effort)
        except Exception as e:
            print(f"  d={d:>7} ERROR {type(e).__name__}: {str(e)[:90]}")
            rows.append({"depth": d, "outcome": "error", "detail": str(e)[:120]})
            continue
        hay = c + "\n" + rc
        found = gold in hay
        rep = max_consecutive_repeat(c)
        pt = usage.get("prompt_tokens")
        ct = usage.get("completion_tokens")
        tps = (ct / el) if (ct and el) else 0
        status = "PASS" if found and rep <= 8 else ("DEGENERATE" if rep > 8 else "FAIL")
        rows.append({"depth": d, "approx_prompt_tokens": approx, "prompt_tokens": pt,
                     "completion_tokens": ct, "elapsed_s": round(el, 1),
                     "decode_tok_s": round(tps, 1), "found": found,
                     "max_consecutive_repeat": rep, "finish_reason": fr,
                     "answer_in": ("content" if gold in c else
                                   "reasoning" if gold in rc else "neither"),
                     "outcome": status})
        print(f"  d={d:>7}  prompt_tok={pt}  {status:10s} rep={rep:2d}  "
              f"{tps:5.1f} tok/s  {el:6.1f}s  answer_in={rows[-1]['answer_in']}")

    out = {"label": a.label, "model": a.model, "reasoning_effort": a.reasoning_effort,
           "max_tokens": a.max_tokens, "rows": rows}
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"  wrote {a.out}")
    ok = sum(1 for r in rows if r.get("outcome") == "PASS")
    print(f"\n{a.label}: {ok}/{len(rows)} depths PASS")


if __name__ == "__main__":
    main()
