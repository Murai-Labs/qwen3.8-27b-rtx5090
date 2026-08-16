#!/usr/bin/env python3
"""One command per arm: boot, measure throughput + quality + long context, gate, record.

Why this exists. Every finding in README.md was hand-run once, by hand, and then
written up. That is how you discover things and it is not how you keep them: there
is no way to re-run last month's claim, and nothing stops a new lever from
silently regressing an old one. MLX improves `mlx.fast` steadily because each
change is a benchmarked unit that cannot regress unnoticed. This is that ratchet.

What it does per arm:
  1. boots the engine with that arm's flags (arms are DATA, in bench/arms.json)
  2. runs the suites the arm asks for, into bench/results/<arm>/
  3. gates quality against the control arm with McNemar's exact test
  4. appends a row to bench/results/leaderboard.jsonl and regenerates LEADERBOARD.md

It exits non-zero if a candidate is significantly WORSE than the control on
quality. A harness that records a regression as just another row is a log, not a
ratchet.

Three behaviours inherited from deploy/bench_arms.sh, each learned the hard way:
  - readiness is probed with urllib, not curl (a curl probe returned 000 against
    a healthy server here)
  - a failed probe NEVER tears the server down; the benchmark is the verdict
  - "Application startup complete" is not readiness; the socket lags it

Usage:
    python bench/ratchet.py --dry-run                 # validate wiring, boot nothing
    python bench/ratchet.py --arms mtp3               # one arm
    python bench/ratchet.py --all                     # every arm in arms.json
    python bench/ratchet.py --arms mtp3 --suites throughput
"""
import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "bench" / "results"
LEDGER = RESULTS / "leaderboard.jsonl"
MARKDOWN = REPO / "bench" / "LEADERBOARD.md"

sys.path.insert(0, str(REPO / "bench"))
from quality_compare import mcnemar_exact_two_sided  # noqa: E402  (repo-local)


# --------------------------------------------------------------------------- io

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def git_dirty():
    try:
        r = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                           capture_output=True, text=True, timeout=10)
        return bool(r.stdout.strip())
    except Exception:
        return None


# ---------------------------------------------------------------- engine control

class Engine:
    """Boot/teardown of one vLLM server. Context manager so a crashed suite still
    tears the engine down -- otherwise the next arm inherits a live port and
    measures the WRONG CONFIG, which is the most dangerous failure mode here."""

    def __init__(self, arm, cfg, python, model, logdir, boot_timeout=1200):
        self.arm, self.cfg, self.python, self.model = arm, cfg, python, model
        self.port = cfg["common"].get("port", 8000)
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.logfile = self.logdir / f"{arm}.log"
        self.boot_timeout = boot_timeout
        self.proc = None
        self.kv_lines = []

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}/v1"

    def _probe(self):
        code = ("import urllib.request;"
                f"urllib.request.urlopen('{self.base_url}/models',timeout=5)")
        return subprocess.run([self.python, "-c", code],
                              capture_output=True).returncode == 0

    def _kill_stale(self):
        subprocess.run(["pkill", "-f", "vllm.entrypoints"], capture_output=True)
        for _ in range(60):
            if not self._probe():
                return
            time.sleep(2)

    def argv(self):
        a = self.cfg["arms"][self.arm]
        return ([self.python, "-m", "vllm.entrypoints.openai.api_server",
                 "--model", self.model,
                 "--served-model-name", self.cfg["common"].get("served_model_name", "qwen38"),
                 "--port", str(self.port)]
                + list(self.cfg["common"]["args"]) + list(a["args"]))

    def __enter__(self):
        self._kill_stale()
        argv = self.argv()
        log(f"  boot: {' '.join(shlex.quote(x) for x in argv[3:])}")
        with open(self.logfile, "w") as fh:
            self.proc = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT,
                                         start_new_session=True)
        t0 = time.time()
        started = False
        while time.time() - t0 < self.boot_timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"engine exited rc={self.proc.returncode} during boot; "
                                   f"see {self.logfile}")
            txt = self.logfile.read_text(errors="replace")
            if "Application startup complete" in txt:
                started = True
                break
            for bad in ("Engine core initialization failed", "CUDA out of memory",
                        "ValueError:", "raise RuntimeError"):
                if bad in txt:
                    tail = "\n".join(txt.strip().splitlines()[-6:])
                    raise RuntimeError(f"engine boot failed ({bad}):\n{tail}")
            if int(time.time() - t0) % 60 < 5:
                log(f"    ...booting, {time.time() - t0:.0f}s")
            time.sleep(5)
        if not started:
            raise RuntimeError(f"engine did not start within {self.boot_timeout}s")

        # startup-complete precedes socket readiness by a variable margin
        for i in range(90):
            if self._probe():
                log(f"    socket ready after {time.time() - t0:.0f}s")
                break
            time.sleep(2)
        else:
            log("    WARNING: probe still failing; measuring anyway (never tear down "
                "on a failed probe -- the benchmark is the verdict)")

        txt = self.logfile.read_text(errors="replace")
        self.kv_lines = [l.strip() for l in txt.splitlines()
                         if any(k in l for k in ("Available KV cache memory",
                                                 "GPU KV cache size",
                                                 "Maximum concurrency"))][-3:]
        for l in self.kv_lines:
            log(f"    {l.split('] ')[-1]}")
        return self

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except Exception:
                self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except Exception:
                    self.proc.kill()
        self._kill_stale()
        return False


# ------------------------------------------------------------------------ suites

def run_suite(engine, name, spec, arm, python, outdir):
    """Run one measurement script against a live engine. Returns its parsed JSON."""
    out = outdir / f"{name}.json"
    argv = [python, str(REPO / spec["script"]),
            "--base-url", engine.base_url,
            "--model", engine.cfg["common"].get("served_model_name", "qwen38"),
            "--label", arm] + list(spec["args"])
    # bench_fixed.py spells its output flag --output; the others use --out
    argv += (["--output", str(out)] if "bench_fixed" in spec["script"] else ["--out", str(out)])

    log(f"  suite {name}: running (timeout {spec.get('timeout_s', 3600)}s)")
    t0 = time.time()
    r = subprocess.run(argv, cwd=REPO, timeout=spec.get("timeout_s", 3600))
    if r.returncode != 0:
        raise RuntimeError(f"suite {name} exited rc={r.returncode}")
    if not out.exists():
        raise RuntimeError(f"suite {name} wrote no output at {out}")
    with open(out) as fh:
        payload = json.load(fh)

    # A measurement script can exit 0 having measured NOTHING: bench_fixed.py
    # prints "cell ... error" per failed cell and still writes its (empty) rows
    # list. Recording that as a clean run is how a harness silently launders a
    # broken arm into a green leaderboard row -- caught by the smoke suite doing
    # exactly that on the first live run.
    if not payload:
        raise RuntimeError(f"suite {name} produced an EMPTY result set; the arm ran but "
                           f"measured nothing (see the suite output above)")
    if name in ("throughput", "smoke"):
        good = [r for r in payload if r.get("decode_tok_s") == r.get("decode_tok_s")]
        if not good:
            raise RuntimeError(f"suite {name}: no cell produced a finite decode rate")
    if name == "quality" and not payload.get("results"):
        raise RuntimeError(f"suite {name}: no per-item results")

    log(f"  suite {name}: done in {time.time() - t0:.0f}s -> {out.relative_to(REPO)}")
    return payload


def summarize(name, payload):
    """Pull the few numbers that belong on a leaderboard row."""
    if name == "throughput":
        cells = {f"p{r['prompt_tokens']}c{r['concurrency']}": round(r["decode_tok_s"], 1)
                 for r in payload}
        spread = {f"p{r['prompt_tokens']}c{r['concurrency']}": round(r["decode_spread"], 1)
                  for r in payload}
        return {"decode_tok_s": cells, "decode_spread": spread}
    if name == "quality":
        return {"accuracy": payload["accuracy"], "n": payload["n"],
                "counts": payload["counts"],
                "ci95": payload.get("accuracy_ci95")}
    if name == "longctx":
        rows = payload if isinstance(payload, list) else payload.get("rows", [])
        return {"depths": {str(r.get("depth")): r.get("outcome") for r in rows},
                "max_prompt_tokens": max((r.get("prompt_tokens") or 0 for r in rows),
                                         default=None),
                "worst_repeat": max((r.get("max_consecutive_repeat") or 0 for r in rows),
                                    default=None)}
    return {"raw": True}


# -------------------------------------------------------------------------- gate

def quality_gate(cand_path, ctrl_path):
    """McNemar's exact test, candidate vs control. Returns (verdict, detail).

    Verdicts:
      PASS      - no significant difference, i.e. no detected harm
      REGRESSED - control significantly better. This is what makes it a ratchet.
      IMPROVED  - candidate significantly better
      NO_CONTROL/SKIPPED
    """
    if not ctrl_path or not Path(ctrl_path).exists():
        return "NO_CONTROL", {}
    with open(cand_path) as f:
        B = json.load(f)
    with open(ctrl_path) as f:
        A = json.load(f)
    for k in ("task", "reasoning_effort", "max_tokens"):
        if A.get(k) != B.get(k):
            return "SKIPPED", {"reason": f"{k} differs ({A.get(k)} vs {B.get(k)}); "
                                         f"the comparison would not be attributable"}
    ra = {r["id"]: r for r in A["results"]}
    rb = {r["id"]: r for r in B["results"]}
    ids = [i for i in ra if i in rb]
    ok = lambda r: r["outcome"] == "correct"
    only_ctrl = sum(1 for i in ids if ok(ra[i]) and not ok(rb[i]))
    only_cand = sum(1 for i in ids if not ok(ra[i]) and ok(rb[i]))
    p = mcnemar_exact_two_sided(only_ctrl, only_cand)
    detail = {"n_shared": len(ids), "only_control": only_ctrl,
              "only_candidate": only_cand, "p": round(p, 4),
              "control_label": A["label"]}
    if p >= 0.05:
        return "PASS", detail
    return ("REGRESSED", detail) if only_ctrl > only_cand else ("IMPROVED", detail)


# ------------------------------------------------------------------- leaderboard

def _gate_floor(alpha=0.05, limit=64):
    """Smallest one-sided discordant count that trips the gate. Computed, not assumed."""
    for k in range(1, limit):
        if mcnemar_exact_two_sided(k, 0) < alpha:
            return k
    return None


def append_ledger(row):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as fh:
        fh.write(json.dumps(row) + "\n")


def render_markdown():
    if not LEDGER.exists():
        return
    rows = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    # Only rows carrying a real suite may set a leaderboard row. A smoke run (a
    # wiring check) or a failed arm must never blank out an arm's published
    # numbers just by being more recent -- which is what happened the first time
    # this ran live.
    REAL = ("throughput", "quality", "longctx")
    latest = {}
    for r in rows:
        if any(k in r for k in REAL) and not r.get("error"):
            latest[r["arm"]] = r      # last write per arm wins
    order = sorted(latest.values(), key=lambda r: r["date"])

    out = ["# Leaderboard",
           "",
           "Generated by `bench/ratchet.py`. One row per arm, most recent run.",
           "Full history, including superseded runs, is in "
           "`bench/results/leaderboard.jsonl`.",
           "",
           "`gate` is McNemar's exact test against the control arm: **PASS** means no "
           "detected quality change (which is not proof of equivalence, only that "
           "n was too small to tell), **REGRESSED** means the control was "
           "significantly better.",
           "",
           f"**Sensitivity floor: {_gate_floor()} discordant items.** A one-sided loss of "
           f"{_gate_floor()} items reaches p<0.05 and trips the gate; {_gate_floor() - 1} "
           "does not. This follows from the exact binomial and is independent of n, so a "
           "regression smaller than that is invisible here and a PASS is not a "
           "certificate. Verified by degrading a stored run item by item.",
           "",
           "| arm | date | commit | p256c1 | p2048c1 | p8192c1 | GSM8K | gate | p |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in order:
        t = (r.get("throughput") or {}).get("decode_tok_s", {})
        q = r.get("quality") or {}
        g = r.get("gate") or {}
        acc = f"{q['accuracy'] * 100:.1f}%" if q.get("accuracy") is not None else "—"
        gate = g.get("verdict", "—")
        pv = g.get("detail", {}).get("p")
        dirty = "*" if r.get("dirty") else ""
        out.append(f"| `{r['arm']}` | {r['date'][:10]} | `{r['commit']}`{dirty} | "
                   f"{t.get('p256c1', '—')} | {t.get('p2048c1', '—')} | "
                   f"{t.get('p8192c1', '—')} | {acc} | {gate} | "
                   f"{pv if pv is not None else '—'} |")
    out += ["", "`*` = the working tree was dirty when this row was recorded; the "
                "commit alone does not reproduce it.", ""]
    MARKDOWN.write_text("\n".join(out))
    log(f"wrote {MARKDOWN.relative_to(REPO)}")


# ---------------------------------------------------------------------- backfill

# The runs already in bench/results/ predate this harness. Seeding the ledger with
# them means the ratchet starts from what is already known instead of from empty.
# They are marked backfilled=true and carry the result file's mtime rather than a
# run date, because the original run date and commit were never recorded -- that
# is precisely the gap this harness closes going forward.
BACKFILL = {
    "baseline-nospec": {"throughput": "armA.json",
                        "quality": "quality_gsm8k_nvfp4_nospec.json"},
    "mtp3": {"throughput": "armC.json",
             "quality": "quality_gsm8k_nvfp4_mtp3.json",
             "longctx": "longctx_fp8-mtp3-100k.json"},
    "mtp3-ssmfp16": {"quality": "quality_gsm8k_ssmfp16.json"},
}


def backfill(cfg):
    ctrl = cfg.get("quality_control")
    ctrl_file = RESULTS / BACKFILL.get(ctrl, {}).get("quality", "")
    n = 0
    for arm, files in BACKFILL.items():
        row = {"arm": arm, "desc": cfg["arms"].get(arm, {}).get("desc", ""),
               "commit": "unknown", "dirty": None, "backfilled": True,
               "source_files": files}
        mtimes = []
        for suite, fname in files.items():
            p = RESULTS / fname
            if not p.exists():
                log(f"  backfill: {arm}/{suite}: {fname} missing, skipped")
                continue
            mtimes.append(p.stat().st_mtime)
            with open(p) as fh:
                row[suite] = summarize(suite, json.load(fh))
        if not mtimes:
            continue
        row["date"] = datetime.fromtimestamp(max(mtimes), timezone.utc).isoformat(
            timespec="seconds")
        if "quality" in files and arm != ctrl and ctrl_file.exists():
            v, d = quality_gate(RESULTS / files["quality"], ctrl_file)
            row["gate"] = {"verdict": v, "detail": d}
        append_ledger(row)
        log(f"  backfilled {arm}: {list(files)} "
            f"gate={row.get('gate', {}).get('verdict', '—')}")
        n += 1
    render_markdown()
    log(f"backfilled {n} arm(s) from existing result files")


# -------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "bench" / "arms.json"))
    ap.add_argument("--arms", default=None, help="comma-separated arm names")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--suites", default=None,
                    help="override which suites to run (default: each arm's own list)")
    ap.add_argument("--python", default=os.environ.get(
        "VENV_PY", str(Path.home() / "qwen38/.venv/bin/python")))
    ap.add_argument("--model", default=os.environ.get(
        "MODEL", str(Path.home() / "models/Qwen3.8-27B-NVFP4")))
    ap.add_argument("--logdir", default=str(Path.home() / "logs"))
    ap.add_argument("--dry-run", action="store_true",
                    help="validate config and print the exact commands; boot nothing")
    ap.add_argument("--render-only", action="store_true",
                    help="regenerate LEADERBOARD.md from the ledger and exit")
    ap.add_argument("--backfill", action="store_true",
                    help="seed the ledger from result files that predate this harness")
    a = ap.parse_args()

    if a.render_only:
        render_markdown()
        return 0

    with open(a.config) as fh:
        cfg = json.load(fh)

    if a.backfill:
        backfill(cfg)
        return 0

    if a.all:
        arms = list(cfg["arms"])
    elif a.arms:
        arms = [x.strip() for x in a.arms.split(",")]
    else:
        return ap.error("pass --arms or --all (or --dry-run to inspect)")
    unknown = [x for x in arms if x not in cfg["arms"]]
    if unknown:
        sys.exit(f"unknown arm(s): {unknown}; known: {list(cfg['arms'])}")

    # The control must run FIRST, or the first candidate has nothing to be gated
    # against and silently records NO_CONTROL.
    ctrl = cfg.get("quality_control")
    if ctrl in arms:
        arms = [ctrl] + [x for x in arms if x != ctrl]
    ctrl_quality = RESULTS / ctrl / "quality.json" if ctrl else None

    commit, dirty = git_commit(), git_dirty()
    log(f"repo {commit}{' (DIRTY)' if dirty else ''}   arms: {arms}")
    if ctrl and ctrl not in arms and not (ctrl_quality and ctrl_quality.exists()):
        log(f"NOTE: control '{ctrl}' is not in this run and has no stored quality.json; "
            f"quality gating will report NO_CONTROL.")

    if a.dry_run:
        for arm in arms:
            e = Engine(arm, cfg, a.python, a.model, a.logdir)
            suites = ([s.strip() for s in a.suites.split(",")] if a.suites
                      else cfg["arms"][arm]["suites"])
            print(f"\n=== {arm} — {cfg['arms'][arm]['desc']}")
            print("  " + " ".join(shlex.quote(x) for x in e.argv()))
            for s in suites:
                if s not in cfg["suites"]:
                    sys.exit(f"arm '{arm}' asks for unknown suite '{s}'")
                print(f"  suite {s}: {cfg['suites'][s]['script']} "
                      f"{' '.join(cfg['suites'][s]['args'])}")
        print(f"\ndry run OK: {len(arms)} arm(s), config valid, no engine booted.")
        return 0

    if not Path(a.python).exists():
        sys.exit(f"venv python not found: {a.python} (set VENV_PY)")

    failures, regressions = [], []
    for arm in arms:
        log(f"=== arm {arm} — {cfg['arms'][arm]['desc']}")
        outdir = RESULTS / arm
        outdir.mkdir(parents=True, exist_ok=True)
        suites = ([s.strip() for s in a.suites.split(",")] if a.suites
                  else cfg["arms"][arm]["suites"])
        row = {"arm": arm, "desc": cfg["arms"][arm]["desc"],
               "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "commit": commit, "dirty": dirty,
               "server_args": cfg["common"]["args"] + cfg["arms"][arm]["args"]}
        t0 = time.time()
        try:
            with Engine(arm, cfg, a.python, a.model, a.logdir) as eng:
                row["kv_log"] = eng.kv_lines
                for s in suites:
                    payload = run_suite(eng, s, cfg["suites"][s], arm, a.python, outdir)
                    row[s] = summarize(s, payload)
        except Exception as e:
            log(f"  ARM FAILED: {e}")
            row["error"] = str(e)[:500]
            failures.append(arm)
            append_ledger(row)
            continue

        if "quality" in suites:
            verdict, detail = quality_gate(outdir / "quality.json",
                                           ctrl_quality if arm != ctrl else None)
            row["gate"] = {"verdict": verdict, "detail": detail}
            log(f"  gate: {verdict} {detail}")
            if verdict == "REGRESSED":
                regressions.append(arm)

        row["elapsed_s"] = round(time.time() - t0, 1)
        append_ledger(row)
        log(f"  arm {arm} done in {row['elapsed_s']:.0f}s")

    render_markdown()

    if failures:
        log(f"FAILED arms: {failures}")
    if regressions:
        log(f"QUALITY REGRESSION in: {regressions} — not recording these as wins")
        return 2
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
