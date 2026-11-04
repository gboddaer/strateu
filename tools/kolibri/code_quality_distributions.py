#!/usr/bin/env python3
"""Compare complete next-token distributions on aligned code-quality histories.

Consumes code_quality_probe.py artifacts. Descriptive diagnostics only: there is
no universal fidelity threshold or model-quality acceptance decision here.
"""
import argparse
import gzip
import json
from pathlib import Path
import struct

import numpy as np
from code_quality_probe import post, tokenizer
from reuse_gate import Engine


def logsoftmax(x):
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("non-finite scores")
    y = x - np.max(x)
    return y - np.log(np.exp(y).sum())


def last_logits(path):
    data = Path(path).read_bytes()
    offset = 0
    result = None
    while offset < len(data):
        _, _, nv = struct.unpack_from("=iii", data, offset)
        offset += 12
        size = nv * 4
        if nv <= 0 or offset + size > len(data):
            raise ValueError("invalid logits dump")
        result = np.frombuffer(data[offset:offset + size], dtype=np.float32).copy()
        offset += size
    if result is None:
        raise ValueError("empty logits dump")
    return result


def common_prefix(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--llama", default="http://127.0.0.1:8200")
    args = ap.parse_args()
    pairs = Path(args.pairs)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    tok = tokenizer(args.tokenizer)
    points = []
    tags = ["insert-char", "line-edit", "raw-mode", "nonblocking-read", "notepad"]
    for tag in tags:
        ids = json.loads((pairs / (tag + "-prompt.json")).read_text())["ids"]
        points.append({"tag": tag + "-start", "ids": ids})
        a = json.loads((pairs / (tag + "-llama-response.json")).read_text())["tokens"]
        b = json.loads((pairs / (tag + "-strata-response.json")).read_text())["gen"]
        n = common_prefix(a, b)
        if tag in ("raw-mode", "notepad") and n < min(len(a), len(b)):
            points.append({"tag": tag + "-first-divergence", "ids": ids + a[:n],
                           "generated_prefix": tok.decode(a[:n])})
        marker = {"line-edit": "        col = lines[row].size()",
                  "notepad": "std::getline"}.get(tag)
        if marker:
            decoded = tok.decode(a)
            pos = decoded.index(marker)
            # Stop on a real boundary of the original generated ID sequence.
            k = 0
            while k < len(a) and len(tok.decode(a[:k + 1])) <= pos:
                k += 1
            points.append({"tag": tag + "-defect-prefix", "ids": ids + a[:k],
                           "generated_prefix": tok.decode(a[:k])})
    (out / "points.json").write_text(json.dumps(points, indent=2) + "\n")
    dump = out / "strata-logits.bin"
    eng = Engine(args.bin, args.model, 32768, 14,
                 {"STRATA_KOLIBRI_REUSE": "0", "STRATA_KOLIBRI_LOGITS_DUMP": str(dump)})
    rows = []
    first = None
    try:
        for point in points:
            ids = point["ids"]
            eng.send(ids, 1)
            logits = last_logits(dump)
            if first is None:
                first = logits.copy()
            response = post(args.llama + "/completion", {
                "prompt": ids, "n_predict": 1, "temperature": 0,
                "top_k": 1, "top_p": 1, "min_p": 0, "repeat_penalty": 1,
                "cache_prompt": False, "n_probs": len(logits),
                "post_sampling_probs": False, "return_tokens": True})
            with gzip.open(out / (point["tag"] + "-llama.json.gz"), "wt") as f:
                json.dump(response, f)
            entries = response["completion_probabilities"][0]["top_logprobs"]
            if len(entries) != len(logits) or {x["id"] for x in entries} != set(range(len(logits))):
                raise ValueError("llama response is not a complete vocabulary distribution")
            llama_lp = np.empty(len(logits), dtype=np.float64)
            for entry in entries:
                llama_lp[entry["id"]] = entry["logprob"]
            llama_lp = logsoftmax(llama_lp)
            strata_lp = logsoftmax(logits)
            p, q = np.exp(llama_lp), np.exp(strata_lp)
            itop, jtop = int(np.argmax(p)), int(np.argmax(q))
            row = {"tag": point["tag"], "history_tokens": len(ids), "vocab": len(logits),
                   "llama_top1": itop, "strata_top1": jtop,
                   "llama_top1_text": tok.decode([itop]), "strata_top1_text": tok.decode([jtop]),
                   "llama_probability_of_llama_top1": float(p[itop]),
                   "strata_probability_of_llama_top1": float(q[itop]),
                   "llama_probability_of_strata_top1": float(p[jtop]),
                   "strata_probability_of_strata_top1": float(q[jtop]),
                   "kl_llama_to_strata_nats": float(np.sum(p * (llama_lp - strata_lp))),
                   "kl_strata_to_llama_nats": float(np.sum(q * (strata_lp - llama_lp))),
                   "total_variation": float(np.abs(p - q).sum() / 2),
                   "abs_logprob_error_p95": float(np.percentile(np.abs(llama_lp - strata_lp), 95)),
                   "abs_logprob_error_max": float(np.max(np.abs(llama_lp - strata_lp)))}
            rows.append(row)
            (out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps(row), flush=True)
        eng.send(points[0]["ids"], 1)
        repeat = {"tag": points[0]["tag"],
                  "finite_bit_identical_cold_replay": first.tobytes() == last_logits(dump).tobytes()}
        (out / "repeat.json").write_text(json.dumps(repeat, indent=2) + "\n")
        print(json.dumps(repeat), flush=True)
    finally:
        eng.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
