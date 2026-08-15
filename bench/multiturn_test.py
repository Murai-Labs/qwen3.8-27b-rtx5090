#!/usr/bin/env python3
"""Behavioural test for the empty-think-block truncation on multi-turn work.

Runs a 15-turn agentic-style loop, feeding each assistant reply back into the
history the way a real agent loop does (content only, no reasoning_content).
Measures the length of the ANSWER (what follows </think>) per turn, which is
what actually truncates -- total completion_tokens can look healthy while the
answer is empty.

Run twice: stock defaults, then chat_template_kwargs={"preserve_thinking": false}.

Usage: python multiturn_test.py [base_url] [model] [turns]
"""
import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/v1"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen38"
TURNS = int(sys.argv[3]) if len(sys.argv) > 3 else 15
TRUNC_THRESHOLD = 20  # answer chars below this counts as truncated

TASKS = [
    "List three prime numbers greater than {n}.",
    "What is {n} squared? Answer with the number and one sentence of working.",
    "Name a country whose name has {n} letters, and its capital.",
    "Give a one-line Python function that returns the {n}th Fibonacci number.",
    "In one sentence, what happened in the year 19{n:02d}?",
]


def post(messages, kwargs):
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0,
    }
    if kwargs:
        body["chat_template_kwargs"] = kwargs
    req = urllib.request.Request(
        BASE.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def split_answer(raw):
    """Everything after the final </think> is the answer."""
    if "</think>" in raw:
        return raw.rsplit("</think>", 1)[1].strip()
    return raw.strip()


def run(label, kwargs):
    print(f"\n===== {label} =====")
    messages = [{"role": "system", "content": "You are a concise agent."}]
    truncated = 0
    for i in range(1, TURNS + 1):
        task = TASKS[(i - 1) % len(TASKS)].format(n=i)
        messages.append({"role": "user", "content": task})
        try:
            d = post(messages, kwargs)
        except Exception as e:
            print(f"  turn {i:2d}: REQUEST FAILED {type(e).__name__}: {e}")
            return None
        msg = d["choices"][0]["message"]
        raw = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
        answer = split_answer(msg.get("content") or "")
        if not answer and msg.get("reasoning_content"):
            answer = (msg.get("content") or "").strip()
        ct = d["usage"]["completion_tokens"]
        bad = len(answer) < TRUNC_THRESHOLD
        truncated += bad
        print(f"  turn {i:2d}: completion_tok={ct:4d}  answer_chars={len(answer):4d}"
              f"  {'<-- TRUNCATED' if bad else ''}  {answer[:60]!r}")
        # replay the way an agent loop does: content only
        messages.append({"role": "assistant", "content": answer or "(empty)"})
    print(f"  RESULT: {truncated}/{TURNS} turns truncated (answer < {TRUNC_THRESHOLD} chars)")
    return truncated


a = run("STOCK DEFAULTS", None)
b = run("preserve_thinking=false", {"preserve_thinking": False})
print("\n===== SUMMARY =====")
print(f"  stock defaults          : {a}/{TURNS} truncated")
print(f"  preserve_thinking=false : {b}/{TURNS} truncated")
