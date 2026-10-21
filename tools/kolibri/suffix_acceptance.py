#!/usr/bin/env python3
"""tools/kolibri/suffix_acceptance.py - task 13 (P4): the suffix drafter's ACCEPTANCE on the MODEL'S OWN
outputs, on the agent-turn workload.

The plan's contract for P4 is ">= 1.4x on agent-style traces, no regression on free-form text, verified output
token-identical to greedy".  The drafter's ceiling on *text* was measured with the flagship's model-free
simulator (tools/...: continue ~1.25 tok/step, copy 6.5-8.6) - but a simulator measures text, not what the
model actually writes.  This tool measures the real thing: it replays the agent turns against a running
Kolibri-1 server, takes the model's own completions, and for each turn computes what the suffix drafter would
have proposed and how much of it the model actually produced.

The drafter itself is C++ (strata::spec::SuffixDrafter), so the tool drives the tiny helper
`build/suffix_drafter_accept` (compiled from tools/kolibri/suffix_acceptance.cpp) with two id files.

Usage:
    # 1. the server (any kolibri serve):
    #    python -m serve.server --engine strata --config /tmp/kolibri_serve.json --port 8080
    # 2. the tool:
    python -u tools/kolibri/suffix_acceptance.py --url http://127.0.0.1:8080 --out /tmp/accept.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))


def post(url: str, messages: list[dict], max_tokens: int, temperature: float) -> dict:
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "top_p": 0.95,
            "top_k": 20, "seed": 1234}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--out", default="/tmp/accept.json")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--max-turns", type=int, default=0)
    ap.add_argument("--helper", default=str(ROOT / "build" / "suffix_drafter_accept"))
    ap.add_argument("--temperature", type=float, default=1.0)
    a = ap.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location("ab", str(ROOT / "tools" / "agent_turns_bench.py"))
    ab = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(ab)
    except SystemExit:
        pass
    work = ab.turns()
    if a.max_turns:
        work = work[:a.max_turns]

    import strata_tokenizer as ST
    tk = ST.Tokenizer.from_gguf("/data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf")

    rows = []
    for tag, msgs in work:
        t0 = time.time()
        r = post(a.url, msgs, a.max_new, a.temperature)
        dt = time.time() - t0
        text = r["choices"][0]["message"].get("content") or ""
        # the history the model saw (the prompt) and what it wrote
        prompt_text = "\n".join(m.get("content") or "" for m in msgs)
        p_ids = tk.encode(prompt_text)
        g_ids = tk.encode(text)
        if len(p_ids) < 8 or len(g_ids) < 4:
            print(f"{tag:16s} too short (prompt {len(p_ids)}, gen {len(g_ids)})", flush=True)
            continue
        hp = Path(f"/tmp/acc_{tag}_prompt.ids"); hg = Path(f"/tmp/acc_{tag}_gen.ids")
        hp.write_text(" ".join(map(str, p_ids)))
        hg.write_text(" ".join(map(str, g_ids)))
        out = subprocess.run([a.helper, str(hp), str(hg)], capture_output=True, text=True, timeout=600)
        parts = out.stdout.split()
        acc = float(parts[0]) if parts and parts[0] != "-" else 0.0
        prop = float(parts[1]) if len(parts) > 1 else 0.0
        tok_step = float(parts[2]) if len(parts) > 2 else 0.0
        rows.append({"turn": tag, "prompt_tokens": len(p_ids), "gen_tokens": len(g_ids), "seconds": dt,
                     "accept_per_proposal": acc, "proposal_rate": prop, "tokens_per_step": tok_step,
                     "decode_tok_s": len(g_ids) / dt if dt > 0 else 0.0})
        print(f"{tag:16s} prompt {len(p_ids):7d} gen {len(g_ids):5d}  {dt:6.1f}s  "
              f"accept/proposal {acc:5.2f}  proposals {prop:5.1%}  tok/step {tok_step:5.2f}", flush=True)

    if rows:
        med = sorted(r["tokens_per_step"] for r in rows)
        mid = med[len(med) // 2]
        mean_acc = sum(r["accept_per_proposal"] for r in rows) / len(rows)
        print(f"\nmedian tokens/step (the drafter's ceiling on the model's own outputs): {mid:.3f}")
        print(f"mean accept/proposal: {mean_acc:.3f}")
        print(f"the contract's 1.4x needs a median tokens/step >= 1.40 (a K+1 verify costing one step)")
    Path(a.out).write_text(json.dumps(rows, indent=1))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
