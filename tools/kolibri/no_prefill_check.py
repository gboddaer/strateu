#!/usr/bin/env python3
"""tools/kolibri/no_prefill_check.py - the geometry where the prefill reads NOTHING must still be right.

With `reuse == ids.size()-1` the chunked prefill loop runs zero times: no PP line is printed, and the request is
served entirely from held KV plus the one token the decode step feeds.  That is the closest a request can get to
"read nothing", and it is the shape the plan called "one-token tail with prompt_read 0".

`prompt_read` cannot be 0 there and the plan was wrong about that: the field counts every prompt token the
request did not skip, and the reuse guard requires `held < ids.size()`, so the final token is always fed - by the
decode step, not the prefill.

It also cannot be BITWISE, and that is the finding this check exists to record.  Reaching this geometry needs the
rewind disabled, and the rewind is exactly what makes a resumed read bitwise: without it the row at the reuse
boundary is inherited from the previous request's single-token decode instead of being recomputed inside a
prefill chunk, and the two land a few FP16 ulps apart.  Measured here: cold [4037 262 3480 262] against warm
[4037 262 7771 11534].  So the three things this check asserts are the ones that are true: the reuse covers the
whole held prefix, the prefill printed no PP lines at all, and prompt_read is 1.  The plan's "one-token tail with
prompt_read 0" is therefore unsatisfiable in either direction: with the rewind the prefill reads a chunk, without
it the answer stops being bitwise.

It needs its own process: the geometry is only reachable with the rewind disabled (`STRATA_KOLIBRI_REUSE_ALIGN=0`),
and a second engine cannot run alongside the 42.9 GiB one a gate already has resident.

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
            env=dict(os.environ, STRATA_KOLIBRI_REUSE_ALIGN="0"))
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
        pp, gen, done = 0, [], None
        while True:
            x = self._read()
            if x is None:
                raise SystemExit("engine stopped answering")
            if x.startswith("PP "):
                pp += 1
            elif x.startswith("T "):
                gen.append(int(x.split()[1]))
            elif x.startswith("DONE"):
                done = x.split()
                break
        return pp, gen, done

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
    pre = ids[:640]
    eng = Engine(bin_, model)
    try:
        eng.send([2], 1)
        own = eng.send(pre, 1)[1][0]            # the token the model writes after the prefix
        ext = pre + [own]                        # exactly one token past what the engine holds
        eng.send([2], 1)
        _, cold, _ = eng.send(ext, 4)
        eng.send([2], 1)
        eng.send(pre, 1)                         # the turn to be reused
        pp, warm, done = eng.send(ext, 4)         # the no-prefill geometry
    finally:
        eng.close()
    reused, read = int(done[8]), int(done[14])
    ok = reused == len(pre) and pp == 0 and read == 1 and len(cold) == 4
    print("  no-prefill: reused %d of %d, %d PP lines, prompt_read %d, cold %s vs warm %s%s"
          % (reused, len(pre), pp, read, cold, warm,
             "" if cold == warm else " (expected: not bitwise without the rewind)"))
    print("  NO-PREFILL PASS" if ok else "  NO-PREFILL FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
