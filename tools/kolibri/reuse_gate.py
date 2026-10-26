#!/usr/bin/env python3
"""tools/kolibri/reuse_gate.py - the prefix-reuse gate.

Why this gate exists (measured 2026-10-25, Tesla T4 box): in serve(), `fresh` is a COUNT of new tokens
while both prefill loops compared it against a POSITION, so a request that EXTENDED the held prefix fed
the model nothing at all - a 580-token request reported `reused 516, read 64, prompt_ms 0.0` and answered
from KV that had never seen the new turn.  Every benchmark through this harness looked normal because with
reuse == 0 the arithmetic coincides with the right answer.

Two protocol facts this gate is built around, both measured rather than assumed:

  * `GEN 0` does not terminate - the engine emits nothing for a zero-token generation, so a gate that
    waits for its DONE hangs.  The prefix arm therefore uses `GEN 1` and carries the model's own token
    into the extension, exactly as an agent's next turn does.
  * `--prefill` and `--cpu-threads` are both required to be on the published path.  Without `--prefill`
    the engine reads one token per forward() (~93 ms/token measured, ~15x slower), and cpu_threads
    defaults to 0 (no CPU expert pool), so an unconfigured gate over a 3k prefix costs the better part
    of an hour instead of a minute.

Stages (one engine process; each stage starts from a cold state via a resetter request):

  1.  bitwise      cold one-shot read of `pre + own + tail` vs `pre` then extending with `own + tail`:
                   the 4 generated ids must be identical.  `pre` is 3,000 ids, well past the 513-row
                   sliding ring, so the ring has wrapped before the extension - extension-only reuse is
                   ring-safe by construction and this proves it rather than assuming it.
  1b. bitwise-1    same, with a ONE-token tail: nothing is left to prefill, and that must look correct
                   (this is what today's bug looks like) rather than be mistaken for it.  640-id prefix,
                   still past the ring, since the property is the ring, not the size.
  2. accounting    stage 1's reused arm must report reused == |pre| + 1, read == |tail|, prompt_ms > 0.
                   A prompt_ms of 0.0 with tokens read is the bug, so it fails the gate.
  3. negative      a prefix that diverges in its last id is not an extension: reused 0, everything read.

Exit: 0 all stages pass, 1 any stage fails, 2 the harness could not run.
"""
import argparse
import os
import queue
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

# A silent engine is a failure, not something to wait out.  Sized on the measured box: a 3,065-token
# chunked prefill is ~20 s at the published rate, so 180 s is ~9x worst case and the whole gate still
# finishes in a few minutes.  The first version had no bound at all and hung for the full timeout.
REQUEST_TIMEOUT_S = 180.0
READY_TIMEOUT_S = 300.0


class EngineHung(Exception):
    pass


class Engine:
    """One `--serve` process.  send() returns everything the engine said about one request."""

    def __init__(self, bin_, model, ctx, threads):
        self.p = subprocess.Popen(
            [bin_, "--model", model, "--serve", "--ctx", str(ctx), "--prefill",
             "--cpu-threads", str(threads)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        # A reader thread, not select(): select on a text-mode pipe waits on the OS fd while lines the
        # Python reader has already buffered sit in user space, so a ready answer reads as "silent".  The
        # first version of this gate timed out on a request the engine had answered in under a second.
        self.q = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        line = self._read(READY_TIMEOUT_S)
        for _ in range(64):                       # banner lines precede the handshake
            if line is None or line.startswith("READY"):
                break
            line = self._read(READY_TIMEOUT_S)
        if not line or not line.startswith("READY"):
            raise RuntimeError(f"engine did not say READY (got {line!r})")

    def _pump(self):
        for line in self.p.stdout:
            self.q.put(line.strip())
        self.q.put(None)                          # EOF: the engine is gone

    def _read(self, timeout=REQUEST_TIMEOUT_S):
        try:
            line = self.q.get(timeout=timeout)
        except queue.Empty:
            raise EngineHung(f"engine silent for {timeout:.0f} s")
        if line is None:
            raise EngineHung("engine closed its stdout")
        return line

    def send(self, ids, n):
        self.p.stdin.write(f"GEN {n} {','.join(str(i) for i in ids)}\n")
        self.p.stdin.flush()
        r = {"gen": [], "resume": None, "done": None, "pp": []}
        while True:
            line = self._read()
            if line.startswith("T "):
                r["gen"].append(int(line.split()[1]))
            elif line.startswith("PP "):
                r["pp"].append(line.split()[1:])
            elif line.startswith("RESUME "):       # reused tokens; NOT the end of the request
                r["resume"] = int(line.split()[1])
            elif line.startswith("DONE"):
                r["done"] = line.split()
                return r

    def send_cancel(self, ids, n):
        """Send a request, cancel it with STOP after its first token, and return the DONE fields.

        The point is the token the engine sampled last: it is pushed onto the held list but never fed, so a
        later request must not claim it as KV it built."""
        self.p.stdin.write(f"GEN {n} {','.join(str(i) for i in ids)}\n")
        self.p.stdin.flush()
        r = {"gen": [], "done": None}
        stopped = False
        while True:
            line = self._read()
            if line.startswith("T "):
                r["gen"].append(int(line.split()[1]))
                if not stopped:
                    self.p.stdin.write("STOP\n")
                    self.p.stdin.flush()
                    stopped = True
            elif line.startswith("RESUME "):
                pass
            elif line.startswith("DONE"):
                r["done"] = line.split()
                return r

    def resetter(self, token):
        """A one-token request nothing can extend, so the next request starts from a clean state.
        (GEN 0 would be cheaper but never answers - see the module docstring.)"""
        self.send([token], 1)

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=30)
        except Exception:
            self.p.kill()


def field(done, idx, cast):
    return cast(done[idx]) if done and len(done) > idx else None


def load(path):
    return [int(x) for x in open(path).read().replace(",", " ").split()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.environ.get("KOLIBRI_BIN", os.path.join(ROOT, "build", "strata-kolibri")))
    ap.add_argument("--model", default=os.environ.get("KOLIBRI_MODEL", ""))
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--cpu-threads", type=int, default=24)
    ap.add_argument("--ids3k", default=os.path.join(HERE, "reuse_ids3k.txt"))
    ap.add_argument("--tail64", default=os.path.join(HERE, "reuse_ids_tail64.txt"))
    a = ap.parse_args()
    if not a.model:
        print("FATAL: no model (set KOLIBRI_MODEL or --model)", file=sys.stderr)
        return 2

    pre, tail = load(a.ids3k), load(a.tail64)
    short_pre, short_tail = pre[:640], tail[:8]
    reset_token = 1 if pre[0] != 1 else 2
    results, rc = [], 0

    def record(ok, text):
        results.append((ok, text))

    eng = Engine(a.bin, a.model, a.ctx, a.cpu_threads)
    stage = "startup"
    try:
        # ---- stage 1 / 2: the agent's next turn, cold against reused ----------------------------
        # No warm-up and no size ordering: measured 2026-10-25, a 1,024-id request runs fine when it cannot
        # extend the held prefix (22.3 s, reused 0) and kills the engine in 1.1 s when it can.  The size was
        # never the problem - `n = fresh - 1 - reuse` goes negative in the reuse path, T = min(chunk, n) goes
        # negative with it, and (size_t)T * 4 wraps into a cudaMemcpy size the driver rejects (:3604).
        stage = "1. bitwise"
        own = eng.send(pre, 1)["gen"][0]          # the token the model itself writes after the prefix
        ext = pre + [own] + tail                  # what the next turn actually looks like
        t0 = time.time()
        eng.resetter(reset_token)
        cold = eng.send(ext, 4)
        cold_s, cold_ms = time.time() - t0, field(cold["done"], 3, float)
        stage = "1. bitwise (reused arm)"
        eng.resetter(reset_token)
        eng.send(pre, 1)                          # the previous turn: this is what `ext` extends
        reuse = eng.send(ext, 4)
        ok = cold["gen"] == reuse["gen"] and len(cold["gen"]) == 4
        record(ok, f"1. bitwise      {'PASS' if ok else 'FAIL'}  "
                   f"(cold: {cold['gen']} | reused: {reuse['gen']})")
        # the cold arm's own rate, so a gate that is slow for box reasons is visible rather than mysterious
        if cold_ms:
            record(True, f"    rate        cold read of {len(ext)} ids: {cold_ms / 1000.0:.1f} s "
                         f"({len(ext) / (cold_ms / 1000.0):.1f} tok/s)")

        stage = "2. accounting"

        reused_n = field(reuse["done"], 8, int)
        read_n = field(reuse["done"], 14, int)
        prompt_ms = field(reuse["done"], 3, float)
        # The counts must add up, and the read must cover the tail plus the token the previous request
        # produced (sampled, never fed, so it has no KV row) plus at most one CHUNK given back by the rewind
        # that makes a resumed read bitwise-equal to a cold one.  Asserting that relationship rather than a
        # fixed number is what keeps this stage honest as the rewind's granularity changes.
        ext_len = len(pre) + 1 + len(tail)
        tail_min = len(tail) + 1
        ok = (reused_n is not None and read_n is not None
              and reused_n + read_n == ext_len
              and read_n >= tail_min
              and read_n <= tail_min + 512
              and prompt_ms is not None and prompt_ms > 0.0)
        record(ok, f"2. accounting   {'PASS' if ok else 'FAIL'}  "
                   f"(reused {reused_n} + read {read_n} = {ext_len}; read between {tail_min} and {tail_min + 512}; "
                   f"prompt_ms {prompt_ms})")

        # ---- stage 1b: one new token - nothing to prefill, and that must be correct --------------
        stage = "1b. bitwise-1"
        eng.resetter(reset_token)
        own_s = eng.send(short_pre, 1)["gen"][0]
        ext_s = short_pre + [own_s] + short_tail[:1]
        eng.resetter(reset_token)
        cold_s = eng.send(ext_s, 4)
        eng.resetter(reset_token)
        eng.send(short_pre, 1)
        reuse_s = eng.send(ext_s, 4)
        read_s = field(reuse_s["done"], 14, int)
        ok = cold_s["gen"] == reuse_s["gen"] and len(cold_s["gen"]) == 4
        record(ok, f"1b. bitwise-1   {'PASS' if ok else 'FAIL'}  "
                   f"(cold: {cold_s['gen']} | reused: {reuse_s['gen']}, read {read_s})")

        # ---- stage 3: a diverging prefix is not an extension ------------------------------------
        stage = "3. negative"
        eng.resetter(reset_token)
        eng.send(short_pre + short_tail, 1)
        diverged = (short_pre + short_tail)[:-1] + [(short_tail[-1] ^ 7)]
        div = eng.send(diverged, 1)
        d_re, d_read = field(div["done"], 8, int), field(div["done"], 14, int)
        ok = d_re == 0 and d_read == len(diverged)
        record(ok, f"3. negative     {'PASS' if ok else 'FAIL'}  "
                   f"(reused {d_re}, read {d_read} of {len(diverged)})")
        # ---- stage 4: a cancelled request must not claim KV it never built -----------------------
        stage = "4. cancelled"
        eng.resetter(reset_token)
        eng.send(short_pre, 1)                      # the turn to be cancelled from
        eng.send_cancel(short_pre + short_tail, 64)  # cancelled after its first token
        # Now extend the cancelled prefix's prompt further: whatever the engine claims to reuse must be KV it
        # actually has, so the result must equal a cold read of this same prompt.
        ext4 = short_pre + short_tail + short_tail[:8]
        eng.resetter(reset_token)
        cold4 = eng.send(ext4, 4)
        eng.resetter(reset_token)
        eng.send(short_pre, 1)
        eng.send_cancel(short_pre + short_tail, 64)
        warm4 = eng.send(ext4, 4)
        re4 = field(warm4["done"], 8, int)
        ok = cold4["gen"] == warm4["gen"] and len(cold4["gen"]) == 4 and re4 is not None \
            and re4 <= len(short_pre + short_tail)
        record(ok, f"4. cancelled    {'PASS' if ok else 'FAIL'}  "
                   f"(after a STOP, reused {re4} of at most {len(short_pre + short_tail)}; "
                   f"cold {cold4['gen']} vs warm {warm4['gen']})")
    except EngineHung as e:
        record(False, f"{stage:14s} FAIL  ({e}) - the engine stopped answering; failing instead of hanging")
    finally:
        eng.close()

    for ok, text in results:
        print("  " + text)
        rc = rc or (0 if ok else 1)
    print("GATE PASS" if rc == 0 else "GATE FAIL")
    return rc


if __name__ == "__main__":
    sys.exit(main())
