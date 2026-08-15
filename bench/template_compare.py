#!/usr/bin/env python3
"""Head-to-head: stock Qwen3.8 template vs froggeric community template.

Counts empty closed think blocks when a 6-turn agent conversation is replayed
the way an OpenAI-compatible client actually replays it (content only).
"""
import re
import sys

from jinja2 import Environment
from jinja2.exceptions import TemplateError

TEMPLATES = {
    "stock (shipped with model)": sys.argv[1],
    "froggeric community": sys.argv[2],
}


def make_env():
    def raise_exception(msg):
        raise TemplateError(msg)

    env = Environment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: "2026-08-14"
    return env


msgs = [{"role": "system", "content": "You are a coding agent."}]
for i in range(1, 7):
    msgs.append({"role": "user", "content": f"step {i}: read the file"})
    msgs.append({"role": "assistant", "content": f"done step {i}"})
msgs.append({"role": "user", "content": "now summarise"})

EMPTY_PATTERNS = [r"<think>\s*</think>", r"<think>\n\n</think>"]

print(f"{'template':30s} {'kwargs':26s} {'chars':>6s} {'<think>':>8s} {'EMPTY':>6s}")
print("-" * 82)
for name, path in TEMPLATES.items():
    try:
        src = open(path, encoding="utf-8").read()
        tpl = make_env().from_string(src)
    except Exception as e:
        print(f"{name:30s} LOAD ERROR: {type(e).__name__}: {e}")
        continue
    for klabel, kwargs in [("(defaults)", {}), ("preserve_thinking=false", {"preserve_thinking": False})]:
        try:
            out = tpl.render(messages=msgs, add_generation_prompt=True, **kwargs)
        except Exception as e:
            print(f"{name:30s} {klabel:26s} RENDER ERROR: {type(e).__name__}: {str(e)[:40]}")
            continue
        empty = sum(len(re.findall(p, out)) for p in EMPTY_PATTERNS[:1])
        n_think = len(re.findall(r"<think>", out))
        print(f"{name:30s} {klabel:26s} {len(out):6d} {n_think:8d} {empty:6d}")

print()
print("Sample assistant turn rendering (defaults):")
for name, path in TEMPLATES.items():
    try:
        tpl = make_env().from_string(open(path, encoding="utf-8").read())
        out = tpl.render(messages=msgs, add_generation_prompt=True)
        seg = out.split("<|im_start|>assistant")[1].split("<|im_end|>")[0]
        print(f"  {name:30s} {seg[:70]!r}")
    except Exception as e:
        print(f"  {name:30s} ERROR {type(e).__name__}")
