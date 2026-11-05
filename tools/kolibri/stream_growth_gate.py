#!/usr/bin/env python3
"""Model/GPU regression: a short first request must not disable streamed prefill.

Demand streaming (no CPU fallback), disable KV reuse to isolate allocation, and
compare complete finite logits for identical long prompts across fresh processes.
The output directory must not already exist. This starts its own engine; do not
run alongside production workloads when collecting speed measurements.
"""
import argparse
import json
from pathlib import Path

from degeneration_gate import read_records
from reuse_gate import Engine, load


def compare_logits(reference, candidate, candidate_index=-1):
    """Compare every float byte, not just the winning token."""
    a, b = read_records(reference), read_records(candidate)
    if not a or not b:
        raise ValueError("missing logits record")
    return a[-1][2] == b[candidate_index][2]


def growth_sequences(ids):
    return (("short-long", [ids[:16], ids[:513], ids[:16], ids[:513]]),
            ("short-medium-long", [ids[:16], ids[:257], ids[:513], ids[:257], ids[:513]]))


def run_sequence(args, out, tag, prompts):
    dump = out / (tag + ".bin")
    env = {"STRATA_KOLIBRI_EXPERT_STREAM": "1",
           "STRATA_KOLIBRI_EXPERT_F16": "0",
           "STRATA_KOLIBRI_STREAM_DEV_ACT": "1",
           "STRATA_KOLIBRI_REUSE": "0",
           "STRATA_KOLIBRI_LOGITS_DUMP": str(dump)}
    engine = Engine(args.bin, args.model, 8192, args.cpu_threads, env)
    rows = []
    try:
        for ids in prompts:
            response = engine.send(ids, 1)
            rows.append({"prompt_ids": len(ids), "done": response["done"],
                         "generated_ids": response["gen"]})
            (out / (tag + ".json")).write_text(json.dumps(rows, indent=2) + "\n")
    finally:
        engine.close()
    records = read_records(dump)
    if len(records) != len(prompts):
        raise ValueError("expected one complete finite logit record per prompt")
    if any(len(record[2]) != 128000 * 4 for record in records):
        raise ValueError("expected the complete 128000-value Kolibri vocabulary")
    return dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cpu-threads", type=int, default=14)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    ids = load(Path(__file__).with_name("reuse_ids3k.txt"))
    if len(ids) < 513:
        raise ValueError("fixture must cover a full 512-token prefill chunk")
    (out / "inputs.json").write_text(json.dumps(
        {"fixture": ids[:513], "requests": [16, 257, 513],
         "reuse": False, "streaming": "required", "expert_f16": 0}, indent=2) + "\n")
    reference = run_sequence(args, out, "long-first", [ids[:513]])
    results = []
    for tag, prompts in growth_sequences(ids):
        candidate = run_sequence(args, out, tag, prompts)
        matches = [compare_logits(reference, candidate, i)
                   for i, prompt in enumerate(prompts) if len(prompt) == 513]
        results.append({"case": tag, "long_requests_compared": len(matches),
                        "finite_bitwise_equal": all(matches)})
        (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    for row in results:
        print(json.dumps(row))
    return 0 if all(row["finite_bitwise_equal"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
