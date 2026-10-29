#!/usr/bin/env python3
"""Bounded cold/extended-prefix serve requests for decode_gate.sh.

The engine runs in its own process group. A reader thread drains stdout into a deadline-aware queue;
stderr is inherited so activation markers remain in the gate's log. No blocking readline runs on the
supervisor thread, and successful DONE records do not hide a later engine failure.
"""
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time

from decode_bench import BenchError, parse_done

TIMEOUT_S = float(os.environ.get("KOLIBRI_DEC_GATE_TIMEOUT", "900"))


def drive(engine, model, ids, timeout_s):
    deadline = time.monotonic() + timeout_s
    proc = subprocess.Popen(
        [engine, "--model", model, "--serve", "--ctx", "1024", "--prefill", "--cpu-threads", "8"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, start_new_session=True)
    lines = queue.Queue()

    def read():
        try:
            for line in proc.stdout:
                lines.put(line.strip())
        finally:
            lines.put(None)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()

    def next_line():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise BenchError("serve deadline exceeded")
        try:
            line = lines.get(timeout=remaining)
        except queue.Empty:
            raise BenchError("serve deadline exceeded") from None
        if line is None:
            raise BenchError("engine closed stdout before request completed")
        return line

    try:
        while True:
            line = next_line()
            if line.startswith("READY "):
                break
            if line.startswith("ERR"):
                raise BenchError(line)
        for label, tokens in (("cold", ids), ("extend", ids + ids[:64])):
            proc.stdin.write("GEN 4 " + ",".join(tokens) + "\n")
            proc.stdin.flush()
            produced = 0
            while True:
                line = next_line()
                if line.startswith("T "):
                    int(line.split()[1])
                    produced += 1
                elif line.startswith("DONE "):
                    done = parse_done(line)
                    if done['produced'] != 4 or produced != 4 or done['finish'] != 'length':
                        raise BenchError(f"{label}: incomplete generation: {line}")
                    if done['prompt_ids'] != len(tokens):
                        raise BenchError(f"{label}: unexpected prompt length: {line}")
                    if label == 'cold' and done['reused'] != 0:
                        raise BenchError("cold request unexpectedly reused a prefix")
                    if label == 'extend' and not 0 < done['reused'] <= len(ids):
                        raise BenchError("extended request did not exercise valid prefix reuse")
                    print(f"{label}: {line}", flush=True)
                    break
                elif line.startswith("ERR"):
                    raise BenchError(line)
        proc.stdin.close()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise BenchError("serve deadline exceeded before exit")
        try:
            rc = proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            raise BenchError("serve deadline exceeded waiting for exit") from None
        if rc:
            raise BenchError(f"engine exit status {rc} after DONE")
    finally:
        # Kill only the group we created, including descendants whose parent may already have exited.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)
        reader.join(timeout=1)
        proc.stdout.close()
        if not proc.stdin.closed:
            proc.stdin.close()


def main(argv):
    if len(argv) != 4:
        print("usage: decode_gate_serve.py <engine> <model> <ids-file>", file=sys.stderr)
        return 2
    try:
        ids = Path(argv[3]).read_text().strip().split(',')
        if len(ids) != 512:
            raise BenchError(f"expected 512 prompt ids, got {len(ids)}")
        for token in ids:
            int(token)
        drive(argv[1], argv[2], ids, TIMEOUT_S)
    except (BenchError, OSError, ValueError) as exc:
        print(f"decode_gate_serve: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
