# `bench_fixed.py` read ~2× too high whenever `--reasoning-parser` was on

Found by `bench/ratchet.py` on its first live run, because the ratchet benchmarks the
**recommended serving config** (`deploy/serve.sh`) while the original measurements used
`deploy/bench_arms.sh` — and those two differ by one flag that turns out to matter.

## The bug

`bench_fixed.py` decided a chunk was part of the generation window with:

```python
if delta.get("content") or delta.get("reasoning_content"):
```

**vLLM 0.27.1 names the field `reasoning`, not `reasoning_content`.** Verified directly off
the wire: the delta keys on a streamed chat completion from this build are
`{role, content, reasoning}`.

So with `--reasoning-parser qwen3` enabled, every thinking-phase chunk was invisible to the
harness. `t_first` therefore landed on the first *post-thinking* chunk, while `out_tok` still
came from `usage.completion_tokens` and counted **all** the tokens including the thinking
ones. The rate was a full token count divided by a window that excluded most of the
generation.

Worse, when thinking consumed the entire `max_tokens` budget there were no post-thinking
chunks at all, and the cell died with `no content chunks received` — which is how this
surfaced: a 32-token smoke test failed outright, and raising it to 200 produced a number
that was wrong rather than absent.

## Measured, one variable

Same live server, no speculative decoding, `--reasoning-parser qwen3` on, only the
delta-key fix between the two rows:

| | p256 c1 | p2048 c1 | ttft | prefill tok/s |
|---|---|---|---|---|
| before | **108.2** | — | 2.12 s | 144.8 |
| after | **49.8** | **51.6** | 0.07 s | 4610.1 |

The corrected 49.8 / 51.6 lands on the README's published baseline of 51.5 / 52.5. The ttft
and prefill figures move the same way and for the same reason: previously "time to first
token" was really "time to first token *after thinking finished*".

## What this does and does not invalidate

**The published numbers stand.** `deploy/bench_arms.sh` does not pass `--reasoning-parser`,
so on those runs every token arrived as `content`, the window was correct, and 51.5 / 107.6
are sound.

**What was never measured is the recommended config.** `deploy/serve.sh` *does* enable
`--reasoning-parser qwen3`, and until this fix any attempt to benchmark it would have
returned roughly double the true rate. The gap between "the config we benchmarked" and "the
config we recommend" was invisible precisely because the harness broke silently rather than
loudly on the second one.

The 108.2 reading is a coincidence of magnitude, not evidence that MTP does nothing — MTP
was measured on the parser-less config where the counting was correct.

## The fix

`bench_fixed.py` now also accepts `delta.get("reasoning")`. Nothing else changed.

Two notes for anyone porting this harness:

- The same field-name assumption exists wherever streamed deltas are inspected. `reasoning`
  vs `reasoning_content` is a vLLM-version-dependent detail, not a stable API.
- This is the second instance in this repo of the same failure shape — a setting accepted
  and then silently ignored (`reasoning_effort=medium`, `--attention-backend`). Here it was
  a field name silently absent. The lesson is the same: read what the server actually
  returned, do not assume the key you expected is the key you got.
