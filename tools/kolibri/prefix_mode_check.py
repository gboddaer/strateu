#!/usr/bin/env python3
"""tools/kolibri/prefix_mode_check.py - the opt-in longest-common-prefix mode, in its own process.

STRATA_KOLIBRI_REUSE_PREFIX=1 reuses the longest common prefix instead of demanding that the whole held prefix
match, which is what a real client needs: it re-renders the conversation as text, and a BPE merge across the turn
boundary rewrites the prompt's last token or two.  This drives a prompt that agrees for 598 ids and then diverges,
so a strict-prefix engine would read all of it and a prefix-mode engine must reuse the 598.

It runs as its own process because a second engine cannot live alongside the 42.9 GiB one a gate already has
resident - doing this inside the gate made the engine die with "closed its stdout", which is why it lives here.

Exit: 0 pass, 1 fail, 2 could not run.
"""
import os
import queue
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


class Engine:
    def __init__(self, bin_, model):
        self.p = subprocess.Popen(
            [bin_, "--model", model, "--serve", "--ctx", "8192", "--prefill", "--cpu-threads", "24"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
            env=dict(os.environ, STRATA_KOLIBRI_REUSE_PREFIX="1"))
        self.q = queue.Queue()
        threading.Thread(target=lambda: [self.q.put(l.strip()) for l in self.p.stdout] and self.q.put(None),
                         daemon=True).start()
        line = self._read()
        while line and not line.startswith("READY"):
            line = self._read()
        if not line:
            raise SystemExit("engine did not say READY")

    def _read(self, t=600):
        try:
            return self.q.get(timeout=t)
        except Exception:
            return None

    def send(self, ids, n):
        self.p.stdin.write("GEN %d %s\n" % (n, ",".join(str(x) for x in ids)))
        self.p.stdin.flush()
        gen, done = [], None
        while True:
            x = self._read()
            if x is None:
                raise SystemExit("engine stopped answering")
            if x.startswith("T "):
                gen.append(int(x.split()[1]))
            elif x.startswith("DONE"):
                done = x.split()
                break
        return gen, done

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=30)
        except Exception:
            self.p.kill()


def main():
    bin_ = os.environ.get("KOLIBRI_BIN", os.path.join(ROOT, "build", "strata-kolibri"))
    model = os.environ.get("KOLIBRI_MODEL", "")
    if not model:
        print("FATAL: set KOLIBRI_MODEL", file=sys.stderr)
        return 2
    ids = [int(x) for x in open(os.path.join(HERE, "reuse_ids3k.txt")).read().replace(",", " ").split()]
    prev = ids[:600]
    nxt = ids[:598] + [1234, 5678] + ids[600:640]      # agrees on 598 ids, then diverges
    eng = Engine(bin_, model)
    try:
        eng.send([2], 1)
        cold, _ = eng.send(nxt, 4)
        eng.send([2], 1)
        eng.send(prev, 1)
        warm, done = eng.send(nxt, 4)
    finally:
        eng.close()
    reused = int(done[8])
    ok = reused > 0 and cold == warm and len(cold) == 4
    print("  prefix mode: reused %d of a 598-id common prefix, cold %s%s"
          % (reused, cold, "" if cold == warm else " != warm %s" % warm))
    print("  PREFIX PASS" if ok else "  PREFIX FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
