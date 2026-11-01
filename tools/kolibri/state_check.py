#!/usr/bin/env python3
"""State checks for a Kolibri arm over the engine's own serve protocol.

The accuracy policy requires correctness and repeatability checks that do not depend on any KL number: first use,
replay, prefix reuse, reset, cancellation, and a clean request after cancellation. The engine speaks a line protocol
on stdio (`src/kolibri/kolibri_engine.cu`, `serve()`): `READY`, then `GEN <max_new> [key=value ...] <ids>`, `PP`
progress lines, one `T <id>` per generated token, and `DONE <reason>`. `STOP` cancels the running request.

The checks, each on the same forced-greedy settings (`temperature=0 top_k=1`), so a difference is a difference in
state and not in sampling:

  first use   the first request in a fresh process; its ids are the baseline
  replay      the same request again in the same process  -> must equal first use
  reuse       the same 512 ids extended by 64             -> its own ids are recorded; the gate's reuse semantics
                                                             are checked separately by decode_gate.sh
  cancel      a long request, stopped after a few tokens
  after       a clean standard request after the cancelled one -> must equal first use
  reset       a DIFFERENT prefix, then the standard request again -> must equal first use

A non-zero exit means the checks failed. Nothing here says anything about numerical fidelity; it asks whether the
engine returns the same answer for the same request after every state transition it supports.

    state_check.py --bin PATH --model GGUF --ids FILE [--extend FILE] [--json OUT]
"""
import argparse
import json
import os
import queue
import threading
import subprocess
import sys
import time

DEFAULT_FORCE_KEYS = "temperature=0 top_k=1"
PER_REQUEST_S = float(os.environ.get("STATE_CHECK_REQUEST_S", "420"))   # one stalled request names itself


def say(msg):
    """Progress as it happens: a check that prints only at the end cannot be told apart from a hang."""
    print(msg, flush=True)


class Fail(RuntimeError):
    pass


class Engine:
    """Line protocol over stdio. A reader thread fills a queue: `select()` on a buffered TextIOWrapper is a trap,
    because a second line that arrived with the first sits in Python's buffer and never wakes `select`."""

    def __init__(self, argv, env, deadline=1800, stderr_path=None):
        self.deadline = deadline
        # stderr goes to a FILE, never a pipe: the engine writes hundreds of KB of load banners and trace lines, and
        # a pipe nobody drains fills at ~64 KB and blocks the engine mid-run. That deadlock is what made the first
        # version sit silently for half an hour.
        self.stderr_path = stderr_path
        err = open(stderr_path, "w", encoding="utf-8") if stderr_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                                     text=True, bufsize=1, env=env, start_new_session=True)
        self.q = queue.Queue()
        self.reader = threading.Thread(target=self._pump, daemon=True)
        self.reader.start()
        self.info = []
        while True:
            line = self._readline(900)
            if line.startswith("READY"):
                break
            if line.startswith("INFO"):
                self.info.append(line)
                continue
            raise Fail(f"engine did not say READY: {line!r}")

    def _pump(self):
        try:
            for line in self.proc.stdout:
                self.q.put(line.rstrip("\n"))
        except Exception:
            pass
        self.q.put(None)          # EOF marker

    def _readline(self, timeout):
        try:
            line = self.q.get(timeout=timeout)
        except queue.Empty:
            raise Fail(f"timeout waiting for a protocol line ({timeout:.0f}s)")
        if line is None:
            raise Fail(f"engine closed stdout (exit {self.proc.poll()})")
        return line

    def request(self, max_new, ids, keys=DEFAULT_FORCE_KEYS, stop_after=None, timeout=PER_REQUEST_S):
        """Send one GEN and return (ids, done_reason, _). Prints one line per event, so a stall is visible."""
        t0 = time.time()
        say(f"  -> GEN {max_new} prompt={len(ids)}" + (f" stop_after={stop_after}" if stop_after else ""))
        payload = f"GEN {max_new} {keys} " + ",".join(str(i) for i in ids) + "\n"
        self.proc.stdin.write(payload)
        self.proc.stdin.flush()
        got, stop_sent = [], False
        while True:
            line = self._readline(timeout)
            if line.startswith("T "):
                got.append(int(line[2:]))
                if stop_after is not None and not stop_sent and len(got) >= stop_after:
                    self.proc.stdin.write("STOP\n")
                    self.proc.stdin.flush()
                    stop_sent = True
                    say(f"     STOP sent after {len(got)} token(s)")
            elif line.startswith("DONE"):
                why = " ".join(line.split()[1:])
                say(f"  <- {len(got)} id(s) in {time.time() - t0:.1f}s | DONE {why}")
                return got, why, got
            elif line.startswith("ERR"):
                raise Fail(f"engine reported {line!r}")
            # PP / other progress lines are ignored on purpose

    def close(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()


def run_checks(argv, env, ids, extend_ids, log, stderr_path=None):
    eng = Engine(argv, env, stderr_path=stderr_path)
    try:
        base, why, _ = eng.request(8, ids)
        log["first_use"] = {"ids": base, "done": why}
        if not base:
            raise Fail("first use produced no tokens")

        again, why, _ = eng.request(8, ids)
        log["replay"] = {"ids": again, "done": why, "equals_first_use": again == base}

        reused, why, _ = eng.request(8, ids + extend_ids)
        log["reuse"] = {"ids": reused, "done": why, "prefix_len": len(ids) + len(extend_ids)}

        # cancellation: ask for many tokens, stop after a few
        cancelled, why_c, _ = eng.request(400, ids, stop_after=4)
        log["cancel"] = {"ids_written": len(cancelled), "done": why_c,
                         "stopped_early": len(cancelled) < 400}

        after, why, _ = eng.request(8, ids)
        log["after_cancel"] = {"ids": after, "done": why, "equals_first_use": after == base}

        # a different prefix, then the standard request again: the reset path
        eng.request(8, extend_ids + ids[:64])
        reset, why, _ = eng.request(8, ids)
        log["after_reset"] = {"ids": reset, "done": why, "equals_first_use": reset == base}
    finally:
        # nothing reads stderr here: it is already a file, and draining a pipe at this point is what deadlocked
        eng.close()

    ok = (log["replay"]["equals_first_use"] and log["after_cancel"]["equals_first_use"]
          and log["after_reset"]["equals_first_use"] and log["cancel"]["stopped_early"])
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ids", required=True, help="comma-separated ids file for the standard request")
    ap.add_argument("--extend", help="comma-separated ids appended for the prefix-reuse request")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--cpu-threads", type=int, default=8)
    ap.add_argument("--json", help="write the record here")
    ap.add_argument("--engine-stderr", help="where the engine's stderr goes (a file, never a pipe)")
    args = ap.parse_args(argv)

    ids = [int(x) for x in open(args.ids).read().strip().split(",") if x.strip()]
    ext = [int(x) for x in open(args.extend).read().strip().split(",") if x.strip()] if args.extend else ids[:64]

    env = {k: v for k, v in os.environ.items() if not k.startswith("STRATA_") and not k.startswith("KOLIBRI_")}
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    for k in ("STRATA_KOLIBRI_EXPERT_F16", "STRATA_KOLIBRI_PF_PARTIAL", "STRATA_KOLIBRI_DEC_HANDOFF",
              "STRATA_KOLIBRI_DEC_STREAM", "STRATA_KOLIBRI_DEC_HANDOFF_GRAPH", "STRATA_KOLIBRI_DEC_TOKEN_GRAPH",
              "STRATA_GEMM_DETERMINISTIC"):
        if k in os.environ:
            env[k] = os.environ[k]
    argv_full = [args.bin, "--serve", "--model", args.model, "--prefill", "--cpu-threads", str(args.cpu_threads),
                 "--ctx", str(args.ctx)]
    log = {"argv": argv_full, "ids_sha_len": [len(ids), len(ext)], "env": {k: v for k, v in env.items()
                                                                          if k.startswith("STRATA") or k == "CUDA_VISIBLE_DEVICES"}}
    try:
        ok = run_checks(argv_full, env, ids, ext, log,
                        stderr_path=args.engine_stderr or (args.json + ".engine.err" if args.json else None))
    except Fail as e:
        log["error"] = str(e)
        ok = False
    log["result"] = "PASS" if ok else "FAIL"
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=1)
    for k in ("first_use", "replay", "reuse", "cancel", "after_cancel", "after_reset"):
        if k in log:
            v = dict(log[k])
            v.pop("ids", None)
            print(f"  {k:14s} {v}")
    if "error" in log:
        print(f"  ERROR {log['error']}")
    print(f"  result: {log['result']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
