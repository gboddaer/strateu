#!/usr/bin/env python3
"""Compare two `--dump-full-logits` files for a decode arm: bitwise, aligned, complete.

The engine writes one float32 record per decode step (the whole vocabulary), so two files are comparable
only if they have the same record count AND the same vocabulary — and the comparison is `memcmp`, not a
tolerance.  Two reasons for bitwise:

  * a scheduling-only arm runs the same kernels on the same values in the same order, so any difference is a
    bug, and a tolerance would hide it;
  * `-0.0` equals `0.0` and `1.0 + 1 ulp` equals `1.0` under any tolerance one would pick, so a tolerance can
    pass a real change.  `test_check_decode.py` pins both of those.

This is NOT a KL tool.  Comparing two *free-running* generations would compare token streams that already
diverged, so the caller must drive both arms over the SAME forced history (`--force`) and the same prompt.

    check_decode.py --baseline BASE.bin --candidate CAND.bin --vocab N --records M
"""
import argparse
import math
import struct
import sys
from array import array


class Mismatch(RuntimeError):
    """The two files are not the same logits."""


def _read(path, vocab, records):
    want = 4 * vocab * records
    size = path.stat().st_size
    if size != want:
        raise Mismatch(f"{path.name}: size {size} bytes, want {want} "
                       f"({records} record(s) x {vocab} float32); a truncated or extra record cannot be compared")
    a = array("f")
    with open(path, "rb") as f:
        a.frombytes(f.read())
    if sys.byteorder != "little":
        a.byteswap()
    return a


def compare(baseline, candidate, vocab, records, require_finite=True):
    """Both files must hold exactly `records` x `vocab` finite float32 and agree bit for bit."""
    if vocab <= 0:
        raise Mismatch(f"vocab must be positive, got {vocab}")
    if records <= 0:
        raise Mismatch(f"records must be positive, got {records}")
    a = _read(baseline, vocab, records)
    b = _read(candidate, vocab, records)

    if require_finite:
        for name, arr in (("baseline", a), ("candidate", b)):
            for i, v in enumerate(arr):
                if not math.isfinite(v):
                    raise Mismatch(f"{name}: value at record {i // vocab} index {i % vocab} is not finite ({v}); "
                                   f"a non-finite logit cannot be compared or averaged")

    worst, worst_at = 0.0, None
    for i in range(len(a)):
        if math.isnan(a[i]) or math.isnan(b[i]) or a[i] != b[i] or \
                struct.pack("<f", a[i]) != struct.pack("<f", b[i]):
            raise Mismatch(f"record {i // vocab} index {i % vocab} differs: {a[i]!r} against {b[i]!r} "
                           f"(bit patterns {struct.pack('<f', a[i]).hex()} / {struct.pack('<f', b[i]).hex()})")
        d = abs(a[i] - b[i])
        if d > worst:
            worst, worst_at = d, (i // vocab, i % vocab)

    return {"records": records, "values": len(a), "max_abs_diff": worst, "max_at": worst_at}


def compare_serve(baseline, candidate, vocab, records):
    """Compare headered [reuse, fresh, vocab]+float32 records; reject empty/truncated matched files."""
    if vocab <= 0 or records <= 0:
        raise Mismatch("vocab and record count must be positive")
    stride = 12 + 4 * vocab
    payloads = []
    for path in (baseline, candidate):
        raw = path.read_bytes()
        if len(raw) != stride * records:
            raise Mismatch(f"{path.name}: size {len(raw)}, expected {stride * records}")
        for rec in range(records):
            start = rec * stride
            reused, fresh, nv = struct.unpack_from('<iii', raw, start)
            if nv != vocab:
                raise Mismatch(f"{path.name}: record {rec} vocab {nv}, expected {vocab}")
            if reused < 0 or fresh < 0:
                raise Mismatch(f"{path.name}: invalid request header at record {rec}")
            for idx, (value,) in enumerate(struct.iter_unpack('<f', raw[start + 12:start + stride])):
                if not math.isfinite(value):
                    raise Mismatch(f"{path.name}: record {rec} index {idx} is not finite")
        payloads.append(raw)
    if payloads[0] != payloads[1]:
        raise Mismatch("headered serve records differ bitwise")
    return {'records': records, 'values': records * vocab, 'max_abs_diff': 0.0}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--vocab", type=int, required=True, help="float32 values per record (the model's n_vocab)")
    ap.add_argument("--records", type=int, required=True, help="records the caller expects in BOTH files")
    ap.add_argument("--serve-records", action="store_true", help="compare headered serve dump records")
    ap.add_argument("--allow-non-finite", action="store_true",
                    help="skip the finiteness gate (never use this for a correctness verdict)")
    a = ap.parse_args(argv)
    from pathlib import Path
    try:
        if a.serve_records:
            r = compare_serve(Path(a.baseline), Path(a.candidate), a.vocab, a.records)
        else:
            r = compare(Path(a.baseline), Path(a.candidate), a.vocab, a.records,
                        require_finite=not a.allow_non_finite)
    except (Mismatch, OSError) as e:
        print(f"FAIL {e}", flush=True)
        return 1
    print(f"PASS {r['records']} record(s) x {a.vocab} value(s) bitwise identical; "
          f"max |a-b| {r['max_abs_diff']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
