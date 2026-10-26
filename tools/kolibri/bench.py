#!/usr/bin/env python3
"""tools/kolibri/bench.py - the Task 10 bench (plan docs/superpowers/plans/2026-10-16-kolibri1.md):
prompt-read (512-token pp) and decode (128-token tg) tokens/s, short and after a 32K context.
Drives strata-kolibri --serve: the PP lines give the prompt rate, the DONE line the decode rate."""
import subprocess, argparse, shlex

ap = argparse.ArgumentParser()
ap.add_argument("--bin", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--ids512", required=True, help="file: 512 prompt ids")
ap.add_argument("--ids32k", required=True, help="file: 32768 prompt ids")
ap.add_argument("--ctx", type=int, default=33024)
# The short case used to hard-code ctx=8192, which silently capped every prompt there at 8191 tokens ("prompt
# (8192) + max_new (8) exceeds the context (8192)").  Default unchanged, so no existing script moves.
ap.add_argument("--short-ctx", type=int, default=8192)
ap.add_argument("--tg", type=int, default=128)
ap.add_argument("--extra", default="", help="extra engine flags, e.g. '--prefill --counters'")
ap.add_argument("--case", choices=["short", "after", "both"], default="both", help="short = pp512/tg128 only (the plain path needs ~7 h for pp32768)")
a = ap.parse_args()

def run_case(ids: str, n: int, ctx: int, label: str):
    p = subprocess.Popen([a.bin, "--model", a.model, "--serve", "--ctx", str(ctx)] + shlex.split(a.extra),
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    def rd():
        while True:
            l = p.stdout.readline()
            if not l:
                return None
            l = l.strip()
            # RESUME <n> is the reused-token line and comes BEFORE the first PP; treating it as the end of the
            # request would end the run early.  bench.py drives cold requests (reuse 0) so it never sees one
            # today, but a harness that also drives warm ones must not trip on it.
            if l.startswith("PP ") or l.startswith("T ") or l.startswith("RESUME "):
                continue
            return l
    # The engine prints banner lines BEFORE the handshake: with --grouped the first line is
    # "kolibri: 25 / 50 layers captured as CUDA graphs (the rest use the mmvq loop)", so the assert below
    # rejected every graph-path run.  That is why the S0 flag table has no grouped-with-graphs row: the graph
    # path was never measurable through this harness.  Skip banner lines until READY arrives.
    ready = rd()
    for _ in range(64):
        if ready is None or ready.startswith("READY"):
            break
        ready = rd()
    assert ready and ready.startswith("READY"), f"expected READY, got {ready!r}"
    p.stdin.write(f"GEN {n} {ids}\n")
    p.stdin.flush()
    done = rd()
    assert done and done.startswith("DONE"), f"expected DONE, got {done!r}"
    f = done.split()
    produced, n_ids, prompt_ms, decode_ms = int(f[1]), int(f[2]), float(f[3]), float(f[4])
    # the prompt rate is over the tokens actually READ: with a reused prefix n_ids counts the whole prompt but
    # only n_ids - reused were prefilled, and rating the whole prompt against the tail's time would overstate it.
    reused = int(f[8]) if len(f) > 8 else 0
    read = max(1, n_ids - reused)
    print(f"{label}: pp {read} tok read (of {n_ids}) in {prompt_ms:.0f} ms = {read/max(1e-9,prompt_ms)*1000:.1f} tok/s; "
          f"tg {produced} tok in {decode_ms:.0f} ms = {produced/max(1e-9,decode_ms)*1000:.1f} tok/s")
    p.stdin.close()
    p.wait(timeout=60)

ids512 = open(a.ids512).read().strip()
ids32k = open(a.ids32k).read().strip()
if a.case in ("short", "both"):
    run_case(ids512, a.tg, a.short_ctx, "short   ")
if a.case in ("after", "both"):
    run_case(ids32k, a.tg, a.ctx, "after32K")
