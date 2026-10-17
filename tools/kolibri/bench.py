#!/usr/bin/env python3
"""tools/kolibri/bench.py - the Task 10 bench (plan docs/superpowers/plans/2026-10-16-kolibri1.md):
prompt-read (512-token pp) and decode (128-token tg) tokens/s, short and after a 32K context.
Drives strata-kolibri --serve: the PP lines give the prompt rate, the DONE line the decode rate."""
import subprocess, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--bin", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--ids512", required=True, help="file: 512 prompt ids")
ap.add_argument("--ids32k", required=True, help="file: 32768 prompt ids")
ap.add_argument("--ctx", type=int, default=33024)
ap.add_argument("--tg", type=int, default=128)
a = ap.parse_args()

def run_case(ids: str, n: int, ctx: int, label: str):
    p = subprocess.Popen([a.bin, "--model", a.model, "--serve", "--ctx", str(ctx)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    def rd():
        while True:
            l = p.stdout.readline()
            if not l:
                return None
            l = l.strip()
            if l.startswith("PP ") or l.startswith("T "):
                continue
            return l
    ready = rd()
    assert ready and ready.startswith("READY"), f"expected READY, got {ready!r}"
    p.stdin.write(f"GEN {n} {ids}\n")
    p.stdin.flush()
    done = rd()
    assert done and done.startswith("DONE"), f"expected DONE, got {done!r}"
    f = done.split()
    produced, n_ids, prompt_ms, decode_ms = int(f[1]), int(f[2]), float(f[3]), float(f[4])
    print(f"{label}: pp {n_ids} tok in {prompt_ms:.0f} ms = {n_ids/max(1e-9,prompt_ms)*1000:.1f} tok/s; "
          f"tg {produced} tok in {decode_ms:.0f} ms = {produced/max(1e-9,decode_ms)*1000:.1f} tok/s")
    p.stdin.close()
    p.wait(timeout=60)

ids512 = open(a.ids512).read().strip()
ids32k = open(a.ids32k).read().strip()
run_case(ids512, a.tg, 8192, "short   ")
run_case(ids32k, a.tg, a.ctx, "after32K")
