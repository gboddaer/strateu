#!/usr/bin/env python3
"""tools/kolibri/plain_path_check.py - the --serve path WITHOUT --prefill must actually read the prompt.

Why this exists: every other stage of every gate passes --prefill or --tokens-file, so nothing in the suite could
see the one-token-per-forward() path.  It once fed NOTHING while still printing PP lines and claiming
prompt_read == ids.size() - the `forward(...)` call had been swallowed into a comment - and bench.py then printed a
fabricated prefill rate (640,000 tok/s) for work it never did.  This checks the three things that catch it:

  * the generated ids equal the CHUNKED path's for the same prompt (a different route, same answer),
  * the number of PP lines equals ids-1 (one per token actually read),
  * prompt_ms is positive.

Exit: 0 pass, 1 fail, 2 could not run.
"""
import os
import queue
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
N = 64


def run(bin_, model, prefill):
    ids = open(os.path.join(HERE, "oracle_ids512.txt")).read().replace(",", " ").split()[:N]
    argv = [bin_, "--model", model, "--serve", "--ctx", "8192", "--cpu-threads", "24"]
    if prefill:
        argv.append("--prefill")
    p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         text=True, bufsize=1)
    q = queue.Queue()
    threading.Thread(target=lambda: [q.put(l.strip()) for l in p.stdout] and q.put(None), daemon=True).start()

    def rd(t=600):
        try:
            return q.get(timeout=t)
        except Exception:
            return None

    line = rd()
    while line and not line.startswith("READY"):
        line = rd()
    if not line:
        raise SystemExit("engine did not say READY")
    p.stdin.write("GEN 4 %s\n" % ",".join(ids))
    p.stdin.flush()
    pp, gen, done = 0, [], None
    while True:
        x = rd()
        if x is None:
            break
        if x.startswith("PP "):
            pp += 1
        elif x.startswith("T "):
            gen.append(int(x.split()[1]))
        elif x.startswith("DONE"):
            done = x.split()
            break
    p.stdin.close()
    return pp, gen, done


def main():
    bin_ = os.environ.get("KOLIBRI_BIN", os.path.join(ROOT, "build", "strata-kolibri"))
    model = os.environ.get("KOLIBRI_MODEL", "")
    if not model:
        print("FATAL: set KOLIBRI_MODEL", file=sys.stderr)
        return 2
    pp_c, gen_c, _ = run(bin_, model, prefill=True)
    pp_p, gen_p, done_p = run(bin_, model, prefill=False)
    ms = float(done_p[3]) if done_p else 0.0
    ok = gen_p == gen_c and pp_p == N - 1 and ms > 0.0
    print("  plain path: %d PP lines for %d ids (chunked: %d), %.0f ms, gen %s%s"
          % (pp_p, N, pp_c, ms, gen_p, "" if gen_p == gen_c else " != chunked %s" % gen_c))
    print("  PLAIN PASS" if ok else "  PLAIN FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
