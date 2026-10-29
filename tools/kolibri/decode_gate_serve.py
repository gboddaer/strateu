#!/usr/bin/env python3
"""Drive one serve process through a cold request and one that EXTENDS the held prefix.

The decode gate (tools/kolibri/decode_gate.sh) needs the reused-prefix path compared between two arms, and
that path only exists in serve mode: a second request whose ids extend the ids the engine already holds
skips the prefix.  The engine's STRATA_KOLIBRI_LOGITS_DUMP writes one record per request at its first decode
step, header included, so the two arms' files are comparable byte for byte.

Everything is bounded: a request that never reports DONE fails the run instead of hanging the gate.

    decode_gate_serve.py <engine> <model> <ids-file>
"""
import subprocess
import sys
import time

TIMEOUT_S = float(__import__("os").environ.get("KOLIBRI_DEC_GATE_TIMEOUT", "900"))


def main(argv):
    if len(argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    engine, model, ids_file = argv[1:]
    ids = [t for t in open(ids_file).read().strip().split(",") if t]
    if len(ids) < 512:
        print(f"decode_gate_serve: {ids_file} holds {len(ids)} id(s), need at least 512", file=sys.stderr)
        return 2

    p = subprocess.Popen(
        [engine, "--model", model, "--serve", "--ctx", "1024", "--prefill", "--cpu-threads", "8"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, start_new_session=True)
    deadline = time.monotonic() + TIMEOUT_S

    def line():
        while time.monotonic() < deadline:
            out = p.stdout.readline()
            if out == "":
                return None
            out = out.strip()
            if out.startswith(("READY", "T ", "PP ", "INFO", "RESUME", "kolibri:")):
                if out.startswith("READY"):
                    return out
                continue
            return out
        return None

    try:
        ready = line()
        if ready is None or not ready.startswith("READY"):
            print(f"decode_gate_serve: no READY (got {ready!r})", file=sys.stderr)
            return 1
        for label, toks in (("cold", ids), ("extend", ids + ids[:64])):
            p.stdin.write("GEN 4 " + ",".join(toks) + "\n")
            p.stdin.flush()
            got = line()
            if got is None or not got.startswith("DONE"):
                print(f"decode_gate_serve: {label}: expected DONE, got {got!r}", file=sys.stderr)
                return 1
            print(f"{label}: {got}")
        p.stdin.close()
        p.wait(timeout=120)
        return 0
    finally:
        if p.poll() is None:
            p.kill()
            p.wait(timeout=60)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
