#!/usr/bin/env python3
"""tools/kolibri/decode_bench.py - an unprofiled, phase-separated decode benchmark.

Why this exists (docs/KOLIBRI_DECODE.md): the first decode investigation timed a profiled kernel trace
and compared whole-process walls that included model loading.  Neither is an unprofiled per-token
latency, and subtracting one from the other produced conclusions the data did not support.  This
harness measures only what the engine itself reports and what it prints per token:

  * `DONE <produced> <prompt_ids> <prompt_ms> <decode_ms> <finish> 0 0 <reuse> <hits> <lookups> <misses>
          0 0.0 <fresh> 0`   -- the engine's own request accounting (src/kolibri/kolibri_engine.cu)
  * one flushed `T <id>` line per generated token -- the interval series, captured off the pipe.

The engine's `decode_ms` starts AFTER the prompt has been consumed, so loading and prefill are outside
it by construction.  The T-to-T series is reported beside it so a single slow first token cannot hide
inside one average.

A candidate environment arm is proven live, not assumed live: `--require-marker` must match in the
candidate run and must NOT match in the baseline run.  A request that stops early (EOS), a run that
ends without DONE, a nonzero exit, an inactive marker and a run that outlives its deadline all fail
rather than being scored.

Diagnostic traces (`--trace`) are a separate, explicitly phase-tagged input.  They never produce the
speed number.  If the trace's forward count cannot be reconciled with the caller's phase metadata the
parser refuses instead of guessing which markers were decode.

Usage:
    .venv/bin/python tools/kolibri/decode_bench.py --bin build-mmq/strata-kolibri \
        --model /fast/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf \
        --tokens-file tools/kolibri/oracle_ids512.txt --n 128 --threads 8 \
        --extra "--prefill" --pairs 3 --out /tmp/kolibri-dec2
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

DONE_TOKENS = 16  # DONE + 15 fields, see the printf at the end of serve()


class BenchError(RuntimeError):
    """Any condition that makes a run or a parse unfit to be scored."""


# ----------------------------------------------------------------------------------------------------------
# Protocol parsing
# ----------------------------------------------------------------------------------------------------------
def _num(text: str, what: str) -> float:
    try:
        v = float(text)
    except ValueError:
        raise BenchError(f"{what}: not a number ({text!r})") from None
    if not math.isfinite(v):
        raise BenchError(f"{what}: not finite ({text!r})")
    return v


def _int(text: str, what: str) -> int:
    try:
        return int(text)
    except ValueError:
        raise BenchError(f"{what}: not an integer ({text!r})") from None


def parse_done(line: str) -> dict:
    """Parse the engine's DONE line.  Field order is the engine's contract, not a guess."""
    tok = line.split()
    if not tok or tok[0] != "DONE":
        raise BenchError(f"DONE: line does not start with DONE ({line!r})")
    if len(tok) != DONE_TOKENS:
        raise BenchError(f"DONE: expected {DONE_TOKENS} fields, got {len(tok)} ({line!r})")
    produced = _int(tok[1], "DONE produced")
    prompt_ids = _int(tok[2], "DONE prompt_ids")
    prompt_ms = _num(tok[3], "DONE prompt_ms")
    decode_ms = _num(tok[4], "DONE decode_ms")
    finish = tok[5]
    reused = _int(tok[8], "DONE reuse")
    hits = _int(tok[9], "DONE hits")
    lookups = _int(tok[10], "DONE lookups")
    misses = _int(tok[11], "DONE misses")
    fresh = _int(tok[14], "DONE fresh")
    if produced <= 0:
        raise BenchError(f"DONE: produced {produced} tokens")
    if prompt_ids <= 0:
        raise BenchError(f"DONE: prompt_ids {prompt_ids}")
    if decode_ms <= 0.0:
        raise BenchError(f"DONE: decode_ms {decode_ms} is not a positive duration")
    if prompt_ms <= 0.0:
        raise BenchError(f"DONE: prompt_ms {prompt_ms} is not a positive duration")
    if finish not in ("length", "stop", "cancel"):
        raise BenchError(f"DONE: unknown finish {finish!r}")
    if not 0 <= reused <= prompt_ids:
        raise BenchError(f"DONE: reuse {reused} outside 0..{prompt_ids}")
    if fresh != prompt_ids - reused:
        raise BenchError(f"DONE: fresh {fresh} != prompt_ids {prompt_ids} - reuse {reused}")
    if min(hits, misses, lookups) < 0:
        raise BenchError(f"DONE: negative cache counter (hits={hits} misses={misses} lookups={lookups})")
    if hits + misses != lookups:
        raise BenchError(f"DONE: hits {hits} + misses {misses} != lookups {lookups}")
    return {
        "produced": produced,
        "prompt_ids": prompt_ids,
        "prompt_ms": prompt_ms,
        "decode_ms": decode_ms,
        "finish": finish,
        "reused": reused,
        "fresh": fresh,
        "hits": hits,
        "lookups": lookups,
        "misses": misses,
        "tok_per_s": produced / decode_ms * 1e3,
    }


def parse_token_times(times_ns) -> dict:
    """Interval statistics from the `T` line timestamps (ns, monotonic, first one taken as 0).

    `first_ms` is the first token on its own; the steady figures exclude it.  Fewer than four intervals
    is too few to call a rate, so it is refused rather than averaged.
    """
    if not isinstance(times_ns, (list, tuple)) or any(not isinstance(t, int) for t in times_ns):
        raise BenchError("times: expected a list of integer nanosecond timestamps")
    if len(times_ns) < 5:
        raise BenchError(f"times: {len(times_ns)} timestamp(s) is fewer than the 5 needed for 4 intervals")
    for a, b in zip(times_ns, times_ns[1:]):
        if b <= a:
            raise BenchError(f"times: not strictly increasing ({a} -> {b})")
    steps = [(b - a) / 1e6 for a, b in zip(times_ns, times_ns[1:])]
    window = (times_ns[-1] - times_ns[0]) / 1e6
    steady_window = (times_ns[-1] - times_ns[1]) / 1e6
    return {
        "intervals": len(steps),
        "mean_ms": sum(steps) / len(steps),
        "median_ms": statistics.median(steps),
        "window_ms": window,
        "tok_per_s": len(steps) / window * 1e3,
        "first_ms": times_ns[0] / 1e6,  # the series' own origin, not a latency: see first_interval_ms
        "first_interval_ms": steps[0],
        "steady_tok_per_s": (len(steps) - 1) / steady_window * 1e3,
    }


def parse_info(line: str) -> dict:
    """The `INFO key=value ...` banner the serve loop prints before READY."""
    if not line.startswith("INFO "):
        raise BenchError(f"INFO: not an INFO line ({line!r})")
    out = {}
    for field in line[5:].split():
        if "=" not in field:
            raise BenchError(f"INFO: field without '=' ({field!r})")
        k, v = field.split("=", 1)
        try:
            out[k] = int(v)
        except ValueError:
            out[k] = v
    return out


# ----------------------------------------------------------------------------------------------------------
# Diagnostic kernel trace (never the speed number)
# ----------------------------------------------------------------------------------------------------------
TRACE_COLUMNS = {
    "name": "Kernel_Name",
    "start": "Start_Timestamp",
    "end": "End_Timestamp",
    "stream": "Stream_Id",
}


def parse_trace(path, *, prompt_forwards: int, generation_forwards: int, gap_us: int = 100,
                marker: str = "k_row<", columns=None) -> dict:
    """Attribute traced kernels to the generation phase of ONE request.

    Phase boundaries are declared by the caller, not inferred: the trace must contain exactly
    `prompt_forwards + generation_forwards` forward markers, one per forward (the per-forward embedding
    gather).  The decode window starts at the first generation forward's marker.  A mismatch is refused,
    because "the last 17 markers" is a guess that silently renames prompt work as decode work.

    Reported: the interval UNION of the kernels in the window (overlapping streams merged, so it is GPU
    time not wall time), the per-kernel-name totals, and the gaps between those intervals.  A gap is
    called a gap.  It is not called driver overhead: this data cannot tell host work from idle.
    """
    if prompt_forwards < 0 or generation_forwards < 1:
        raise BenchError("trace: need prompt_forwards >= 0 and generation_forwards >= 1")
    cols = dict(TRACE_COLUMNS)
    if columns:
        cols.update(columns)
    want = prompt_forwards + generation_forwards

    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append((str(r[cols["name"]]), int(r[cols["start"]]), int(r[cols["end"]]),
                             str(r.get(cols["stream"], ""))))
            except (KeyError, TypeError, ValueError) as e:
                raise BenchError(f"trace: unreadable row ({e}): {r!r}") from None
    rows.sort(key=lambda t: t[1])

    marks = [t for t in rows if marker in t[0]]
    if len(marks) != want:
        raise BenchError(
            f"trace: {len(marks)} forward marker(s) matching {marker!r}, but the caller declared "
            f"{prompt_forwards} prompt + {generation_forwards} generation = {want}. Phase boundaries could "
            f"not be established; refusing to attribute kernels to decode."
        )
    start = marks[prompt_forwards][1]
    window = [t for t in rows if t[1] >= start]
    if not window:
        raise BenchError("trace: the declared decode window holds no kernels")

    merged = []
    for _, s, e, _ in sorted(window, key=lambda t: t[1]):
        if e <= s:
            continue
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    kernel_ns = sum(e - s for s, e in merged)
    window_ns = max(t[2] for t in window) - start

    gaps, gap_ns = 0, 0
    for (_, a_end), (b_start, _) in zip(merged, merged[1:]):
        d = b_start - a_end
        if d > gap_us * 1e3:
            gaps += 1
            gap_ns += d

    names = {}
    for name, s, e, _ in window:
        if e > s:
            names[name] = names.get(name, 0.0) + (e - s) / 1e6

    return {
        "markers": len(marks),
        "prompt_forwards": prompt_forwards,
        "generation_forwards": generation_forwards,
        "intervals": generation_forwards,
        "window_ms": window_ns / 1e6,
        "kernel_ms": kernel_ns / 1e6,
        "coverage": kernel_ns / window_ns if window_ns > 0 else 0.0,
        "gaps_over": gaps,
        "gap_ms": gap_ns / 1e6,
        "names": dict(sorted(names.items(), key=lambda kv: -kv[1])),
    }


# ----------------------------------------------------------------------------------------------------------
# Pair planning and aggregation
# ----------------------------------------------------------------------------------------------------------
def plan_pairs(pairs: int):
    """Alternate which arm runs first, so a drifting machine does not favour one arm."""
    if pairs < 1:
        raise BenchError(f"pairs: need >= 1, got {pairs}")
    order = []
    for i in range(pairs):
        arms = ("baseline", "candidate") if i % 2 == 0 else ("candidate", "baseline")
        order.extend((arm, i) for arm in arms)
    return order


def _decode_ms(run) -> float:
    """decode_ms from a RunResult, a {done: {...}} record, or a flat one-field record."""
    if hasattr(run, "done"):
        return run.done["decode_ms"]
    if "done" in run:
        return run["done"]["decode_ms"]
    return run["decode_ms"]


def summarize(baseline, candidate) -> dict:
    if not baseline or not candidate:
        raise BenchError("summarize: an arm has no runs")
    if len(baseline) != len(candidate):
        raise BenchError(f"summarize: {len(baseline)} baseline run(s) vs {len(candidate)} candidate run(s)")
    b = statistics.median([_decode_ms(r) for r in baseline])
    c = statistics.median([_decode_ms(r) for r in candidate])
    return {
        "pairs": len(baseline),
        "baseline_median_ms": b,
        "candidate_median_ms": c,
        "reduction": (b - c) / b,
        "baseline_ms": [_decode_ms(r) for r in baseline],
        "candidate_ms": [_decode_ms(r) for r in candidate],
    }


def load_ids(path, expect: int | None = None) -> str:
    """Read a comma-separated id file and state how many ids it really holds."""
    text = Path(path).read_text().strip()
    if not text:
        raise BenchError(f"ids: {path} is empty")
    parts = [p.strip() for p in text.replace("\n", ",").split(",") if p.strip()]
    for p in parts:
        try:
            int(p)
        except ValueError:
            raise BenchError(f"ids: {path} holds a non-integer token ({p!r})") from None
    if expect is not None and len(parts) != expect:
        raise BenchError(f"ids: {path} holds {len(parts)} token(s), expected {expect}")
    return ",".join(parts)


def check_capacity(prompt_ids: int, n: int, ctx: int) -> None:
    """The engine refuses this too; assert it here so a truncated run is never scored."""
    if prompt_ids + n > ctx:
        raise BenchError(f"capacity: prompt {prompt_ids} + {n} new token(s) exceeds ctx {ctx}")


# ----------------------------------------------------------------------------------------------------------
# Engine process
# ----------------------------------------------------------------------------------------------------------
class RunResult:
    def __init__(self, done, token_times_ns, gen_to_first_t_ms, info, marker_seen, stdout, stderr, wall_s, pid,
                 load_before=None, load_after=None):
        self.done = done
        self.token_times_ns = token_times_ns
        self.gen_to_first_t_ms = gen_to_first_t_ms
        self.info = info
        self.marker_seen = marker_seen
        self.stdout = stdout
        self.stderr = stderr
        self.wall_s = wall_s
        self.pid = pid
        self.load_before = load_before
        self.load_after = load_after

    def intervals(self) -> dict:
        return parse_token_times(self.token_times_ns)

    def as_dict(self) -> dict:
        return {
            "done": self.done,
            "intervals": self.intervals(),
            "token_times_ns": self.token_times_ns,
            "gen_to_first_t_ms": self.gen_to_first_t_ms,
            "info": self.info,
            "marker_seen": self.marker_seen,
            "wall_s": self.wall_s,
            "load_before": self.load_before,
            "load_after": self.load_after,
        }


def _loadavg():
    try:
        return Path("/proc/loadavg").read_text().strip().split()[:3]
    except OSError:
        return None


def _write_log(log_dir, label, stdout, stderr) -> Path | None:
    if log_dir is None:
        return None
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    p = log_dir / f"{label}.log"
    with open(p, "w") as f:
        f.write("# stdout\n")
        f.writelines(stdout)
        f.write("# stderr\n")
        f.writelines(stderr)
    return p


def _tail(lines, n=12) -> str:
    return "".join(lines[-n:]).strip()


def run_request(cmd, *, tokens: str, n: int, timeout_s: float, require_marker: str | None = None,
                label: str = "run", log_dir=None, env=None, observe_marker: str | None = None) -> RunResult:
    """One serve request against an already-configured engine argv.  Everything is bounded."""
    if timeout_s <= 0:
        raise BenchError("timeout_s must be positive")
    child_env = dict(os.environ if env is None else env)
    load_before = _loadavg()
    t0 = time.monotonic()
    deadline = t0 + timeout_s
    try:
        p = subprocess.Popen(
            [str(c) for c in cmd],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
            env=child_env,
        )
    except (FileNotFoundError, PermissionError, OSError) as e:
        raise BenchError(f"cannot start {cmd[0]}: {e}") from None

    out_lines, err_lines, tokens_ns = [], [], []
    done_holder, info_holder, marker_holder = {}, {}, {"seen": False}
    ready, finished, eof = threading.Event(), threading.Event(), threading.Event()
    t_gen_sent = {"t": None}
    marker = require_marker or observe_marker
    pattern = re.compile(marker) if marker else None

    def scan(line: str) -> None:
        if pattern is not None and pattern.search(line):
            marker_holder["seen"] = True
        if line.startswith("READY"):
            ready.set()

    def read_stdout():
        for line in p.stdout:
            out_lines.append(line)
            scan(line)
            s = line.strip()
            if s.startswith("T "):
                tokens_ns.append(time.monotonic_ns())
            elif s.startswith("INFO ") and not info_holder:
                try:
                    info_holder.update(parse_info(s))
                except BenchError:
                    pass
            elif s.startswith("DONE"):
                done_holder["line"] = s
                finished.set()
        eof.set()

    def read_stderr():
        for line in p.stderr:
            err_lines.append(line)
            scan(line)

    th_out = threading.Thread(target=read_stdout, daemon=True)
    th_err = threading.Thread(target=read_stderr, daemon=True)
    th_out.start()
    th_err.start()

    def fail(msg: str) -> None:
        _write_log(log_dir, label, out_lines, err_lines)
        raise BenchError(f"{msg} [pid={p.pid}]")

    try:
        # READY, or the process ends first.
        while not ready.is_set():
            if time.monotonic() > deadline:
                fail(f"{label}: timed out after {timeout_s:.1f}s waiting for READY")
            if eof.is_set() and p.poll() is not None:
                fail(f"{label}: exit status {p.returncode} before READY; stderr: {_tail(err_lines)}")
            time.sleep(0.002)
        t_gen_sent["t"] = time.monotonic_ns()
        try:
            p.stdin.write(f"GEN {n} {tokens}\n")
            p.stdin.flush()
        except (BrokenPipeError, OSError):
            fail(f"{label}: the engine closed its input before GEN; stderr: {_tail(err_lines)}")

        while not finished.is_set():
            if time.monotonic() > deadline:
                fail(f"{label}: timed out after {timeout_s:.1f}s without DONE")
            if eof.is_set():
                rc_now = p.poll() if p.poll() is not None else p.wait(timeout=5)
                fail(f"{label}: ended without DONE after {len(tokens_ns)} T line(s), exit status {rc_now}; "
                     f"stderr: {_tail(err_lines)}")
            time.sleep(0.002)

        try:
            p.stdin.close()
        except OSError:
            pass
        while p.poll() is None:
            if time.monotonic() > deadline:
                fail(f"{label}: timed out after DONE waiting for exit")
            time.sleep(0.002)
        rc = p.returncode
    finally:
        if p.poll() is None:
            # SIGKILL the whole session: the engine may have spawned helpers, and only our group goes.
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        th_out.join(timeout=5)
        th_err.join(timeout=5)

    _write_log(log_dir, label, out_lines, err_lines)
    if rc != 0:
        raise BenchError(f"{label}: exit status {rc}; stderr: {_tail(err_lines)}")
    if "line" not in done_holder:
        raise BenchError(f"{label}: no DONE line")
    done = parse_done(done_holder["line"])
    if done["finish"] != "length":
        raise BenchError(f"{label}: request finished as {done['finish']!r} after {done['produced']} token(s); "
                         f"a short request is not a fixed-length decode")
    if done["produced"] != n:
        raise BenchError(f"{label}: asked for {n} token(s), engine produced {done['produced']}")
    if len(tokens_ns) != done["produced"]:
        raise BenchError(f"{label}: {len(tokens_ns)} T line(s) for {done['produced']} produced token(s)")
    if require_marker is not None and not marker_holder["seen"]:
        raise BenchError(f"{label}: the candidate marker {require_marker!r} never appeared; the arm is "
                         f"not proven active, so this run cannot be scored")

    times = [t - tokens_ns[0] for t in tokens_ns]
    first_ms = (tokens_ns[0] - t_gen_sent["t"]) / 1e6 if t_gen_sent["t"] else 0.0
    return RunResult(done, times, first_ms, info_holder, marker_holder["seen"], out_lines, err_lines,
                     time.monotonic() - t0, p.pid, load_before, _loadavg())


def run_pairs(cmd, *, tokens, n, pairs, timeout_s, candidate_env=None, require_marker=None,
              log_dir=None, order_log=None, label_prefix="pair") -> dict:
    """Alternating baseline/candidate runs in fresh engines, with the controls unset for the baseline."""
    candidate_env = dict(candidate_env or {})
    base_env = dict(os.environ)
    for k in candidate_env:
        base_env.pop(k, None)
    cand_env = dict(os.environ)
    cand_env.update(candidate_env)

    baseline, candidate, order = [], [], []
    for arm, idx in plan_pairs(pairs):
        label = f"{label_prefix}{idx}-{arm}"
        r = run_request(cmd, tokens=tokens, n=n, timeout_s=timeout_s,
                        require_marker=require_marker if arm == "candidate" else None,
                        observe_marker=require_marker if arm == "baseline" else None,
                        label=label, log_dir=log_dir, env=base_env if arm == "baseline" else cand_env)
        if arm == "baseline" and require_marker and r.marker_seen:
            raise BenchError(f"{label}: the baseline run also carries the candidate marker "
                             f"{require_marker!r}; the control arm is not a control")
        (baseline if arm == "baseline" else candidate).append(r)
        order.append(arm)
    if order_log is not None:
        Path(order_log).write_text(" ".join(order) + "\n")
    return {"baseline": baseline, "candidate": candidate, "summary": summarize(baseline, candidate),
            "order": order}


# ----------------------------------------------------------------------------------------------------------
# Recording: what the run was, so the number can be repeated or doubted
# ----------------------------------------------------------------------------------------------------------
WORKLOADS = ({"label": "512", "ids_expect": 512, "ctx": 8192, "n": 128},
             {"label": "8191", "ids_expect": 8191, "ctx": 12288, "n": 128})


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def environment_record(bin_path, model_path) -> dict:
    keep = sorted(k for k in os.environ if k.startswith("STRATA_") or k.startswith("HIP_"))
    rec = {
        "argv_env": {k: os.environ[k] for k in keep},
        "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES", ""),
        "cpu_count": os.cpu_count(),
        "threads_hint": os.environ.get("OMP_NUM_THREADS", ""),
        "binary": {"path": str(bin_path), "sha256": _sha256_file(bin_path), "size": os.path.getsize(bin_path)},
        "model": {"path": str(model_path), "size": os.path.getsize(model_path),
                  "mtime": os.path.getmtime(model_path)},
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        rec["loadavg"] = Path("/proc/loadavg").read_text().strip().split()[:3]
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(("MemTotal:", "MemAvailable:", "Cached:")):
                k, v = line.split(":", 1)
                rec[k] = v.strip()
    except OSError:
        rec["loadavg"] = None
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True, help="strata-kolibri (or another engine on the same serve protocol)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens-file", required=True, help="comma-separated prompt ids")
    ap.add_argument("--long-tokens-file", default=None,
                    help="8191-id prompt for the after-context workload (default: truncate --tokens-file)")
    ap.add_argument("--ctx", type=int, default=None, help="override the workload's context")
    ap.add_argument("--short-ctx", type=int, default=8192)
    ap.add_argument("--long-ctx", type=int, default=12288)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=3600.0, help="per-engine deadline, seconds")
    ap.add_argument("--out", required=True, help="directory for raw JSON and logs")
    ap.add_argument("--extra", default="", help="extra engine flags, identical for both arms")
    ap.add_argument("--candidate-env", action="append", default=[], metavar="NAME=VALUE",
                    help="candidate-arm environment; unset for the baseline arm (repeatable)")
    ap.add_argument("--require-marker", default=None,
                    help="regex that must appear in the candidate run and must not appear in the baseline")
    ap.add_argument("--workloads", default="short",
                    help="comma list of: short (512 ids, ctx 8192), long (8191 ids, ctx 12288), both")
    ap.add_argument("--trace", default=None,
                    help="optional rocprofv3 kernel CSV for a DIAGNOSTIC report (never the speed number)")
    ap.add_argument("--trace-prompt-forwards", type=int, default=None)
    ap.add_argument("--trace-generation-forwards", type=int, default=None)
    a = ap.parse_args(argv)

    os.environ["HIP_VISIBLE_DEVICES"] = "0"  # card0 only, for the child and for the record
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    extra = shlex.split(a.extra)
    cand_env = {}
    for kv in a.candidate_env:
        if "=" not in kv:
            raise BenchError(f"--candidate-env expects NAME=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        cand_env[k] = v

    ids_file = Path(a.tokens_file)
    short_tokens = load_ids(ids_file, expect=512)
    long_tokens = None
    if "long" in (("both",) if a.workloads == "both" else (a.workloads,))[0]:
        src = Path(a.long_tokens_file) if a.long_tokens_file else ids_file
        ids = load_ids(src).split(",")
        if len(ids) < 8191:
            raise BenchError(f"long workload: {src} holds {len(ids)} id(s), need 8191")
        long_tokens = ",".join(ids[:8191])

    want = "both" if a.workloads == "both" else a.workloads
    labels = ["short", "long"] if want == "both" else [want]
    if any(l not in ("short", "long") for l in labels):
        raise BenchError(f"--workloads: unknown workload(s) in {a.workloads!r}")

    record = {"environment": environment_record(a.bin, a.model), "threads": a.threads,
              "extra": extra, "candidate_env": cand_env, "require_marker": a.require_marker,
              "workloads": {}}

    for label in labels:
        if label == "short":
            tokens, expect, default_ctx = short_tokens, 512, a.short_ctx
        else:
            tokens, expect, default_ctx = long_tokens, 8191, a.long_ctx
        ctx = a.ctx if a.ctx is not None else default_ctx
        check_capacity(prompt_ids=expect, n=a.n, ctx=ctx)

        cmd = [a.bin, "--model", a.model, "--serve", "--ctx", str(ctx), "--cpu-threads", str(a.threads)]
        cmd += extra
        res = run_pairs(cmd, tokens=tokens, n=a.n, pairs=a.pairs, timeout_s=a.timeout,
                        candidate_env=cand_env, require_marker=a.require_marker,
                        log_dir=out / f"logs-{label}", order_log=out / f"order-{label}.txt",
                        label_prefix=f"{label}-pair")
        s = res["summary"]
        print(f"{label}: prompt {expect} id(s), ctx {ctx}, {a.n} new token(s), threads {a.threads}")
        print(f"  baseline  median {s['baseline_median_ms']:.1f} ms/request over {s['pairs']} run(s): "
              + ", ".join(f"{v:.1f}" for v in s["baseline_ms"]))
        print(f"  candidate median {s['candidate_median_ms']:.1f} ms/request over {s['pairs']} run(s): "
              + ", ".join(f"{v:.1f}" for v in s["candidate_ms"]))
        print(f"  reduction {s['reduction'] * 100:+.1f}%  (order: {' '.join(res['order'])})")
        for arm in ("baseline", "candidate"):
            for i, r in enumerate(res[arm]):
                it = r.intervals()
                print(f"    {arm}[{i}] decode {r.done['decode_ms']:.1f} ms  {r.done['tok_per_s']:.2f} tok/s  "
                      f"T-to-T median {it['median_ms']:.1f} ms  GEN->T1 {r.gen_to_first_t_ms / 1e3:.1f} s "
                      f"(prompt included)  hits {r.done['hits']}/{r.done['lookups']}")
        record["workloads"][label] = {
            "prompt_ids": expect, "ctx": ctx, "n": a.n,
            "summary": s,
            "order": res["order"],
            "runs": {arm: [r.as_dict() for r in res[arm]] for arm in ("baseline", "candidate")},
        }

    if a.trace:
        if a.trace_prompt_forwards is None or a.trace_generation_forwards is None:
            raise BenchError("--trace needs --trace-prompt-forwards and --trace-generation-forwards")
        t = parse_trace(a.trace, prompt_forwards=a.trace_prompt_forwards,
                        generation_forwards=a.trace_generation_forwards)
        print("diagnostic (not the speed number): " + json.dumps({k: t[k] for k in
              ("intervals", "window_ms", "kernel_ms", "coverage", "gaps_over", "gap_ms")}))
        record["trace"] = t

    (out / "decode_bench.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"raw: {out / 'decode_bench.json'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BenchError as e:
        print(f"decode_bench: {e}", file=sys.stderr)
        sys.exit(2)
