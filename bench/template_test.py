#!/usr/bin/env python3
"""Does the stock Qwen3.8 template inject empty <think></think> on every prior
assistant turn, and does preserve_thinking=false fix it without a template swap?

Renders a multi-turn agentic conversation the way an OpenAI-compatible client
replays it: assistant messages carry `content` only, no `reasoning_content`.
"""
import glob
import os
import re

from jinja2 import Environment
from jinja2.exceptions import TemplateError

path = glob.glob(os.path.expanduser(
    "~/models/Qwen3.8-27B-NVFP4/chat_template.jinja"))[0]
src = open(path, encoding="utf-8").read()


def raise_exception(msg):
    raise TemplateError(msg)


env = Environment(trim_blocks=True, lstrip_blocks=True)
env.globals["raise_exception"] = raise_exception
tpl = env.from_string(src)

# 6 turns, as replayed by a normal agent loop: no reasoning_content anywhere.
msgs = [{"role": "system", "content": "You are a coding agent."}]
for i in range(1, 7):
    msgs.append({"role": "user", "content": f"step {i}: read the file"})
    msgs.append({"role": "assistant", "content": f"done step {i}"})
msgs.append({"role": "user", "content": "now summarise"})

EMPTY = "<think>\n\n</think>"

for label, kwargs in [
    ("DEFAULT (no kwargs)", {}),
    ("preserve_thinking=false", {"preserve_thinking": False}),
    ("reasoning_effort=low", {"reasoning_effort": "low"}),
]:
    try:
        out = tpl.render(messages=msgs, add_generation_prompt=True, **kwargs)
    except Exception as e:
        print(f"{label:26s} -> RENDER ERROR: {type(e).__name__}: {e}")
        continue
    empties = out.count(EMPTY)
    total_think = len(re.findall(r"<think>", out))
    print(f"{label:26s} -> chars={len(out):6d}  <think> tags={total_think:2d}  "
          f"EMPTY closed blocks={empties}")

print()
print("=" * 70)
print("DEFAULT rendering, assistant turns only (first 3):")
out = tpl.render(messages=msgs, add_generation_prompt=True)
for seg in out.split("<|im_start|>assistant")[1:4]:
    print("  <|im_start|>assistant" + repr(seg.split("<|im_end|>")[0]))

print()
print("preserve_thinking=false, assistant turns only (first 3):")
out2 = tpl.render(messages=msgs, add_generation_prompt=True, preserve_thinking=False)
for seg in out2.split("<|im_start|>assistant")[1:4]:
    print("  <|im_start|>assistant" + repr(seg.split("<|im_end|>")[0]))

print()
print("tail of each (the generation prompt):")
print("  DEFAULT :", repr(out[-60:]))
print("  pt=false:", repr(out2[-60:]))
