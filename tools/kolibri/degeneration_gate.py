#!/usr/bin/env python3
"""Regression for reusing overwritten sliding-window KV. Requires a model/GPU.

Run with --bin and --model. Every comparison uses identical forced prompt IDs,
not independently generated histories. A resetter supplies the cold reference.
Full finite logits must match byte-for-byte when an unsafe rewind falls back.
"""
import argparse
import json
import math
from pathlib import Path
import struct

from reuse_gate import Engine, load


def read_records(path):
    data = Path(path).read_bytes()
    records = []
    offset = 0
    while offset < len(data):
        reuse, fresh, nv = struct.unpack_from("=iii", data, offset)
        offset += 12
        raw = data[offset:offset + nv * 4]
        if len(raw) != nv * 4:
            raise ValueError("truncated logits record")
        if not all(math.isfinite(x) for x in struct.unpack(f"={nv}f", raw)):
            raise ValueError("non-finite logits")
        records.append((reuse, fresh, raw))
        offset += nv * 4
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True, help="new artifact directory")
    ap.add_argument("--cpu-threads", type=int, default=14)
    args = ap.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    dump = out / "logits.bin"
    pre = load(Path(__file__).with_name("reuse_ids3k.txt"))
    tail = load(Path(__file__).with_name("reuse_ids_tail64.txt"))
    reset_token = 1 if pre[0] != 1 else 2
    eng = Engine(args.bin, args.model, 8192, args.cpu_threads,
                 {"STRATA_KOLIBRI_LOGITS_DUMP": str(dump)})
    rows = []
    try:
        # Earlier prompt ran beyond the 513-slot sliding ring. A later prompt
        # shares the first 45 IDs, whose cached rows are no longer available.
        target = pre[:45] + tail[:20]
        eng.resetter(reset_token)
        cold = eng.send(target, 1)
        cold_lg = read_records(dump)[-1][2]
        eng.resetter(reset_token)
        eng.send(pre[:640], 1)
        candidate = eng.send(target, 1)
        reuse, fresh, candidate_lg = read_records(dump)[-1]
        rows.append({"case": "short-common-prefix-after-ring-wrap",
                     "cold_ids": cold["gen"], "candidate_ids": candidate["gen"],
                     "resume": candidate["resume"], "reuse": reuse, "fresh": fresh,
                     "finite_bitwise_equal": cold_lg == candidate_lg})
        # Replay a long prompt: its early rows were overwritten even though all
        # its IDs match. Alignment must not rewind into the lost window either.
        target = pre[:640]
        eng.resetter(reset_token)
        cold = eng.send(target, 1)
        cold_lg = read_records(dump)[-1][2]
        candidate = eng.send(target, 1)
        reuse, fresh, candidate_lg = read_records(dump)[-1]
        rows.append({"case": "replay-after-ring-wrap",
                     "cold_ids": cold["gen"], "candidate_ids": candidate["gen"],
                     "resume": candidate["resume"], "reuse": reuse, "fresh": fresh,
                     "finite_bitwise_equal": cold_lg == candidate_lg})
        # A one-token previous request still has a decode-written first row;
        # cold prefill computes that row in a chunk, so alignment must include 0.
        eng.resetter(reset_token)
        own = eng.send(pre[:1], 1)["gen"][0]
        target = pre[:1] + [own] + tail[:8]
        eng.resetter(reset_token)
        cold = eng.send(target, 1)
        cold_lg = read_records(dump)[-1][2]
        eng.resetter(reset_token)
        eng.send(pre[:1], 1)
        candidate = eng.send(target, 1)
        reuse, fresh, candidate_lg = read_records(dump)[-1]
        rows.append({"case": "one-token-prefix-extension",
                     "cold_ids": cold["gen"], "candidate_ids": candidate["gen"],
                     "resume": candidate["resume"], "reuse": reuse, "fresh": fresh,
                     "finite_bitwise_equal": cold_lg == candidate_lg})
        # A partial first chunk must rewind to zero; skipping zero leaves the
        # old partial-chunk/decode rows in place instead of a cold computation.
        eng.resetter(reset_token)
        own = eng.send(pre[:257], 1)["gen"][0]
        target = pre[:257] + [own] + tail[:8]
        eng.resetter(reset_token)
        cold = eng.send(target, 1)
        cold_lg = read_records(dump)[-1][2]
        eng.resetter(reset_token)
        eng.send(pre[:257], 1)
        candidate = eng.send(target, 1)
        reuse, fresh, candidate_lg = read_records(dump)[-1]
        rows.append({"case": "partial-first-chunk-extension",
                     "cold_ids": cold["gen"], "candidate_ids": candidate["gen"],
                     "resume": candidate["resume"], "reuse": reuse, "fresh": fresh,
                     "finite_bitwise_equal": cold_lg == candidate_lg})
        # Preserve useful reuse while every sliding row is still present. The
        # 513-ID prefix rewinds to 512, the default prefill chunk boundary.
        eng.resetter(reset_token)
        own = eng.send(pre[:513], 1)["gen"][0]
        target = pre[:513] + [own] + tail[:8]
        eng.resetter(reset_token)
        cold = eng.send(target, 1)
        cold_lg = read_records(dump)[-1][2]
        eng.resetter(reset_token)
        eng.send(pre[:513], 1)
        candidate = eng.send(target, 1)
        reuse, fresh, candidate_lg = read_records(dump)[-1]
        rows.append({"case": "safe-extension-at-ring-capacity",
                     "cold_ids": cold["gen"], "candidate_ids": candidate["gen"],
                     "resume": candidate["resume"], "reuse": reuse, "fresh": fresh,
                     "finite_bitwise_equal": cold_lg == candidate_lg,
                     "retained_reuse": candidate["resume"] == 512})
        # At exactly 513 consumed positions no slot has been overwritten yet.
        target = pre[:513]
        eng.resetter(reset_token)
        cold = eng.send(target, 1)
        cold_lg = read_records(dump)[-1][2]
        candidate = eng.send(target, 1)
        reuse, fresh, candidate_lg = read_records(dump)[-1]
        rows.append({"case": "replay-at-ring-capacity",
                     "cold_ids": cold["gen"], "candidate_ids": candidate["gen"],
                     "resume": candidate["resume"], "reuse": reuse, "fresh": fresh,
                     "finite_bitwise_equal": cold_lg == candidate_lg,
                     "retained_reuse": candidate["resume"] == 512})
    finally:
        eng.close()
    (out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        print(json.dumps(row), flush=True)
    assert all(r["finite_bitwise_equal"] for r in rows), "reused logits differ from cold forced history"
    assert all(r.get("retained_reuse", True) for r in rows), "safe prefix reuse was lost"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
