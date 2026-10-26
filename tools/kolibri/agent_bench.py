#!/usr/bin/env python3
"""tools/kolibri/agent_bench.py - what a coding agent's turns cost, with and without prefix reuse.

An agent's request is the previous context plus its own reply plus new content, so every turn extends the last.
This drives exactly that against one engine process, N turns, each adding `--chunk` fixture ids plus the tokens
the model itself writes, and reports per turn: prompt tokens, reused, read, prompt_ms and the rate over the
tokens actually read.  The total wall time is the figure that matters for a session.

    python tools/kolibri/agent_bench.py --bin BIN --model GGUF [--turns 6] [--chunk 384]

Rates are wall-clock over the engine's own prompt_ms, which is the same number the published figures use.  Run
it twice and say what the load was: a single run is not a rate.

Note the deliberate choice of fixture: real tokenized text (tools/kolibri/oracle_ids512.txt repeated), not ids
sampled at random.  A shuffled fixture makes the model degenerate to repeating one token, which changes the
number of tokens it writes per turn and therefore the shape of every later turn.
"""
import argparse
import os
import queue
import subprocess
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=33024)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=384, help="ids added per turn, before the model's own tokens")
    ap.add_argument("--gen", type=int, default=8, help="tokens the model writes per turn (a reply stub)")
    ap.add_argument("--cpu-threads", type=int, default=24)
    ap.add_argument("--ids", default=os.path.join(HERE, "oracle_ids512.txt"))
    ap.add_argument("--reuse", default=None, help="passed as STRATA_KOLIBRI_REUSE (0 forces the cold path)")
    a = ap.parse_args()

    base = [int(x) for x in open(a.ids).read().replace(",", " ").split()]
    env = dict(os.environ)
    if a.reuse is not None:
        env["STRATA_KOLIBRI_REUSE"] = a.reuse

    p = subprocess.Popen([a.bin, "--model", a.model, "--serve", "--ctx", str(a.ctx), "--prefill",
                          "--cpu-threads", str(a.cpu_threads)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         text=True, bufsize=1, env=env)
    q = queue.Queue()
    threading.Thread(target=lambda: [q.put(l.strip()) for l in p.stdout] and q.put(None), daemon=True).start()

    def rd(t=600):
        try:
            return q.get(timeout=t)
        except Exception:
            return None

    line = rd(600)
    while line and not line.startswith("READY"):
        line = rd(600)
    if not line or not line.startswith("READY"):
        raise SystemExit("engine did not say READY")

    prompt = []
    turns = []
    t_start = time.time()
    for turn in range(1, a.turns + 1):
        add = [base[(turn * a.chunk + i) % len(base)] for i in range(a.chunk)]
        prompt = prompt + add
        t0 = time.time()
        p.stdin.write("GEN %d %s\n" % (a.gen, ",".join(map(str, prompt))))
        p.stdin.flush()
        gen = []
        done = None
        while True:
            x = rd()
            if x is None:
                raise SystemExit("engine died on turn %d" % turn)
            if x.startswith("T "):
                gen.append(int(x.split()[1]))
            elif x.startswith("DONE"):
                done = x.split()
                break
        wall = time.time() - t0
        n_ids, prompt_ms = int(done[2]), float(done[3])
        reused = int(done[8]) if len(done) > 8 else 0
        read = max(1, n_ids - reused)
        turns.append((turn, n_ids, reused, read, prompt_ms, wall,
                      read / max(1e-9, prompt_ms) * 1000.0, done[5], len(gen)))
        prompt = prompt + gen          # the agent's next turn carries the reply it just wrote

    p.stdin.close()
    total = time.time() - t_start

    print(" turn  prompt  reused   read  prompt_ms   tok/s   wall_s  finish    gen")
    for t in turns:
        print("  %2d  %6d  %6d  %5d  %9.1f  %6.1f  %7.1f  %-8s %3d" %
              (t[0], t[1], t[2], t[3], t[4], t[6], t[5], t[7], t[8]))
    reused_sum = sum(t[2] for t in turns)
    read_sum = sum(t[3] for t in turns)
    print("  totals: %d prompt tokens, %d reused, %d read, wall %.1f s" %
          (sum(t[1] for t in turns), reused_sum, read_sum, total))
    print("  REUSE=%s  turns=%d  chunk=%d  cpu_threads=%d  load=%s" %
          (a.reuse if a.reuse is not None else "default(on)", a.turns, a.chunk, a.cpu_threads,
           "%.2f" % os.getloadavg()[0]))


if __name__ == "__main__":
    main()
