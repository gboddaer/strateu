#!/usr/bin/env python3
"""Full-logit comparison for the Kolibri EXPERT_F16 investigation: KL, error, ranking.

Both inputs are `--dump-full-logits` files taken on the SAME forced history: the engine writes one float32
record per decode position, the whole vocabulary, so record i of each file is the same model position. The
reference P is the arm that stays the accuracy control; the candidate Q is the arm under test. KL is reported
as KL(P||Q), one value per record, in float64 with a stable log-softmax.

What this does not do:

  * it does not accept or reject a candidate. Exit 0 means "comparable and finite", nothing more: the accuracy
    policy needs independent-reference and task-quality evidence, and no single KL number decides either;
  * it does not truncate to the shorter dump. Differing record counts are an invalid comparison, not a shorter
    one, because the extra records would otherwise vanish from the report;
  * it does not hide a worst case behind a mean. The worst step, the reference's winning margin there, and the
    candidate's rank for the reference winner are reported next to the percentiles.

    klcmp.py --ref BASE.bin --cand CAND.bin --vocab 128000 [--json]
"""
import argparse
import json
import os
import sys

import numpy as np

VOCAB = 128000


class InvalidInput(ValueError):
    """The dumps cannot be compared: wrong size, wrong shape, empty or contradictory arguments."""


class NonFiniteInput(ValueError):
    """A logit is NaN or infinite: a correctness failure, not a rounding difference."""


def load(path, vocab):
    """Read a dump as a contiguous float32 array of shape [records, vocab].

    The byte length must be a whole number of records; a truncated or padded file would otherwise lose or
    invent logits silently.
    """
    if not isinstance(vocab, int) or isinstance(vocab, bool) or vocab <= 0:
        raise InvalidInput(f"vocab must be a positive integer, got {vocab!r}")
    size = os.path.getsize(str(path))
    if size == 0:
        raise InvalidInput(f"{path}: empty dump; a comparison needs at least one record")
    if size % 4:
        raise InvalidInput(f"{path}: {size} bytes is not a whole number of float32 values; refusing to guess")
    values = size // 4
    if values % vocab:
        whole = values // vocab
        raise InvalidInput(
            f"{path}: {size} bytes is not a whole number of {vocab}-value records "
            f"({whole} whole record(s) plus {values % vocab} value(s)); refusing to truncate")
    raw = np.fromfile(str(path), dtype=np.dtype("<f4"))
    if raw.size != values:
        raise InvalidInput(f"{path}: read {raw.size} value(s) for a {size}-byte file; the file changed during the read")
    return raw.reshape(-1, vocab)


def _log_softmax(x):
    x = np.asarray(x, dtype=np.float64)
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))


def summary(ref, cand):
    """Compare two [records, vocab] arrays and return the reported quantities.

    `ref` and `cand` must already be loaded with the same vocab. Values are converted to float64 for the
    arithmetic; the bitwise check is done on the raw float32 bits so that -0.0 and 0.0 stay distinguishable.
    """
    a = np.asarray(ref)
    b = np.asarray(cand)
    if a.ndim != 2 or b.ndim != 2:
        raise InvalidInput(f"expected two 2-D arrays, got shapes {a.shape} and {b.shape}")
    if a.shape != b.shape:
        raise InvalidInput(f"record/vocab mismatch: {a.shape[0]}x{a.shape[1]} against {b.shape[0]}x{b.shape[1]}; "
                           f"refusing to compare the shorter part")
    records, vocab = a.shape

    finite_a = np.isfinite(a)
    finite_b = np.isfinite(b)
    if not (finite_a.all() and finite_b.all()):
        bad = int((~finite_a).sum() + (~finite_b).sum())
        first = int(np.argmin(finite_a & finite_b))
        raise NonFiniteInput(f"{bad} non-finite logit(s); first at record {first // vocab} "
                             f"index {first % vocab} ({float(a.flat[first])!r} / {float(b.flat[first])!r})")

    bits_a = a.astype(np.float32, copy=False).view(np.uint32)
    bits_b = b.astype(np.float32, copy=False).view(np.uint32)
    identical = bool(np.array_equal(bits_a, bits_b))
    differing = ~np.all(bits_a == bits_b, axis=1)

    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    delta = np.abs(a64 - b64)
    flat_worst = int(np.argmax(delta))
    max_abs_delta = float(delta.flat[flat_worst])

    lp = _log_softmax(a64)
    lq = _log_softmax(b64)
    kl = (np.exp(lp) * (lp - lq)).sum(axis=1)
    worst_step = int(np.argmax(kl))

    ref_winner = lp.argmax(axis=1)
    cand_winner = lq.argmax(axis=1)
    matching = int((ref_winner == cand_winner).sum())
    rows = np.arange(records)
    ref_top = a64[rows, ref_winner]
    # rank of the reference winner inside the candidate distribution, 1-based
    rank = 1 + (b64 > b64[rows, ref_winner][:, None]).sum(axis=1)
    ref_sorted = np.sort(a64, axis=1)
    margin = ref_sorted[:, -1] - ref_sorted[:, -2]

    return {
        "records": int(records),
        "vocab": int(vocab),
        "bytewise_identical": identical,
        "differing_records": int(differing.sum()),
        "max_abs_delta": max_abs_delta,
        "max_abs_delta_at": {"record": flat_worst // vocab, "index": flat_worst % vocab},
        "kl": kl,
        "kl_mean": float(kl.mean()),
        "kl_median": float(np.median(kl)),
        "kl_p95": float(np.percentile(kl, 95)),
        "kl_p99": float(np.percentile(kl, 99)),
        "kl_max": float(kl.max()),
        "kl_worst_step": worst_step,
        "top1_agreement": matching / records,
        "top1_matching": matching,
        "ref_margin": margin,
        "ref_margin_worst_step": float(margin[worst_step]),
        "ref_margin_min": float(margin.min()),
        "ref_margin_min_step": int(np.argmin(margin)),
        "ref_winner_rank": rank,
        "ref_winner_rank_worst": int(rank.max()),
        "ref_top": ref_top,
    }


def _json_ready(s):
    out = {}
    for k, v in s.items():
        if isinstance(v, np.ndarray):
            out[k] = [float(x) for x in v]
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v) if isinstance(v, np.floating) else int(v)
        else:
            out[k] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="reference dump P (the accuracy control)")
    ap.add_argument("--cand", required=True, help="candidate dump Q (the arm under test)")
    ap.add_argument("--vocab", type=int, default=VOCAB, help="float32 values per record (n_vocab)")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    args = ap.parse_args(argv)

    try:
        s = summary(load(args.ref, args.vocab), load(args.cand, args.vocab))
    except NonFiniteInput as e:
        print(f"FAIL non-finite: {e}", flush=True)
        return 1
    except (InvalidInput, OSError) as e:
        print(f"FATAL {e}", flush=True)
        return 2

    if args.json:
        print(json.dumps(_json_ready(s), allow_nan=False, indent=1), flush=True)
        return 0

    print(f"records {s['records']} x {s['vocab']} float32; bitwise identical: {s['bytewise_identical']}")
    print(f"differing records {s['differing_records']}/{s['records']}; max |a-b| {s['max_abs_delta']:.6g} "
          f"at record {s['max_abs_delta_at']['record']} index {s['max_abs_delta_at']['index']}")
    if s["bytewise_identical"]:
        print("KL 0 everywhere; top-1 agreement 1.0")
        return 0
    print(f"KL(P||Q)  mean {s['kl_mean']:.6f}  median {s['kl_median']:.6g}  p95 {s['kl_p95']:.6g}  "
          f"p99 {s['kl_p99']:.6g}  max {s['kl_max']:.6g} (step {s['kl_worst_step']})")
    print(f"top-1 agreement {s['top1_agreement']:.4f} ({s['top1_matching']}/{s['records']})")
    print(f"reference margin at the worst step {s['ref_margin_worst_step']:.6g}; "
          f"smallest margin {s['ref_margin_min']:.6g} at step {s['ref_margin_min_step']}")
    print(f"worst rank of the reference winner in the candidate {s['ref_winner_rank_worst']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
