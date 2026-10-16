#!/usr/bin/env python3
"""tools/kolibri/serve_smoke.py - the Task 9 serve gates (plan docs/superpowers/plans/2026-10-16-kolibri1.md):
READY, GEN with a prompt's ids, T...DONE lines; a STOP mid-generation cancels; and the greedy text from --serve
equals the greedy ids from --tokens.  Drives the engine over a pipe like serve/server.py does."""
import subprocess, sys, time, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--bin", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--ids", required=True, help="comma-separated prompt ids")
ap.add_argument("--expect", required=True, help="the greedy ids --tokens produced, space-separated")
ap.add_argument("--cancel-after", type=int, default=10, help="T lines to take before STOP")
a = ap.parse_args()

p = subprocess.Popen([a.bin, "--model", a.model, "--serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     text=True, bufsize=1)
def rd():
    while True:
        l = p.stdout.readline()
        if not l: return None
        l = l.strip()
        if l.startswith("PP "): continue
        return l
def gen(max_new, ids, extra=""):
    p.stdin.write(f"GEN {max_new} {extra} {ids}\n"); p.stdin.flush()

t0 = time.time()
line = rd()
assert line and line.startswith("READY"), f"first line: {line!r}"
print(f"READY ok ({line}) in {time.time()-t0:.0f} s (load included)")

# gate 1: greedy from --serve == the greedy ids from --tokens
gen(8, a.ids)
toks = [rd() for _ in range(8)]
got = [int(t[2:]) for t in toks]
done = rd()
assert done.startswith("DONE"), f"expected DONE, got {done!r}"
want = [int(x) for x in a.expect.split()]
assert got == want, f"serve greedy {got} != --tokens {want}"
print(f"serve greedy == --tokens: {got}  ({done.split()[5]})")

# gate 2: STOP mid-generation cancels
gen(64, a.ids)
toks = [rd() for _ in range(a.cancel_after)]
p.stdin.write("STOP\n"); p.stdin.flush()
done = rd()
while done is not None and not done.startswith("DONE") and not done.startswith("T "):
    done = rd()
if done is not None and done.startswith("T "):   # drain the tokens emitted before the flag was seen
    n = 1
    while True:
        d = rd()
        if d is None or d.startswith("DONE"): done = d; break
        n += 1
    print(f"(STOP seen after {a.cancel_after} read, {a.cancel_after+n} emitted before the check)")
assert done is not None and done.startswith("DONE"), f"expected DONE after STOP, got {done!r}"
f = done.split()
assert f[5] == "cancel", f"finish: {f[5]}, expected cancel"
assert int(f[1]) < 64, f"produced {f[1]} of 64"
print(f"STOP cancel ok: produced {f[1]}, finish {f[5]}")
p.stdin.close(); p.wait(timeout=30)
print("serve_smoke OK")
