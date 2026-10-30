#!/usr/bin/env python3
"""Qualification-grade comparison for Kolibri arms against an independent reference.

Why this exists next to `tools/kolibri/kl.py`: that script is a campaign diagnostic, and it cannot serve as the
qualification harness. It fixes `STRATA_DUMP_CTX=8192` for every executable, truncates comparisons with
`min(len(P), len(Q))`, keys its reference cache by filename, and deletes the engine's full dumps. Each of those
silently weakens a qualification result:

  * a fixed 8192 context cannot express the 8191-token KV boundary or the ~11000-token attention transition;
  * `min()` compares a prefix and calls it a comparison, so a truncated or extra record disappears;
  * a filename-keyed cache can hand back a dump produced under a different offload arrangement;
  * deleting the candidate's raw logits makes a reported number uncheckable afterwards.

So: every executable gets an explicit context and configuration, record counts must match exactly, reference dumps
are keyed by a hash of the configuration that produced them, and nothing is deleted.

Reported per arm, per prompt and per position: KL(reference||candidate) mean/median/p95/p99/worst, |dlogit|
percentiles and worst, top-1 agreement, the reference winner's rank in the candidate, and the reference margin at
every rank-1 change. A metric bundle is not an acceptance decision: `--limit` is compared only against limits that
were agreed before the results existed, and this tool never decides task quality.

    qualify.py --ref REF.bin --cand CAND.bin --vocab 128000 --records 9 [--json]
    qualify.py --cache-key '{"ctx": 12288, "ngl": 99, "cpu_moe": 999}'      # a reference dump's identity
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

VOCAB = 128000


class InvalidInput(ValueError):
    """The dumps cannot be compared as qualification evidence."""


class NonFiniteInput(ValueError):
    """A logit is NaN or infinite: a correctness failure, not a rounding difference."""


def load_exact(path, vocab, records):
    """Read a dump that must hold EXACTLY `records` x `vocab` float32; no prefix, no padding."""
    if records <= 0 or vocab <= 0:
        raise InvalidInput(f"records and vocab must be positive, got {records} / {vocab}")
    try:
        size = os.path.getsize(str(path))
    except OSError as e:
        raise InvalidInput(f"{path}: {e}") from e
    want = 4 * vocab * records
    if size != want:
        raise InvalidInput(
            f"{path}: {size} bytes, expected exactly {want} ({records} x {vocab} float32). "
            f"A shorter or longer dump is not the same history and is not truncated here")
    a = np.fromfile(str(path), dtype=np.dtype("<f4"))
    if a.size != records * vocab:
        raise InvalidInput(f"{path}: read {a.size} value(s) for a {size}-byte file")
    return a.reshape(records, vocab)


def _log_softmax(x):
    x = np.asarray(x, dtype=np.float64)
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))


def metrics(ref, cand):
    """KL, |dlogit| and ranking metrics for aligned [records, vocab] arrays."""
    a = np.asarray(ref, dtype=np.float64)
    b = np.asarray(cand, dtype=np.float64)
    if a.shape != b.shape:
        raise InvalidInput(f"shape mismatch {a.shape} against {b.shape}; a prefix comparison is not a comparison")
    for name, x in (("reference", a), ("candidate", b)):
        if not np.isfinite(x).all():
            bad = int((~np.isfinite(x)).sum())
            raise NonFiniteInput(f"{name}: {bad} non-finite logit(s)")
    lp, lq = _log_softmax(a), _log_softmax(b)
    kl = (np.exp(lp) * (lp - lq)).sum(axis=1)
    dl = np.abs(a - b)
    rows = np.arange(a.shape[0])
    ref_win = a.argmax(axis=1)
    cand_win = b.argmax(axis=1)
    rank = 1 + (b > b[rows, ref_win][:, None]).sum(axis=1)
    srt = np.sort(a, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    flips = np.where(ref_win != cand_win)[0]
    return {
        "records": int(a.shape[0]),
        "vocab": int(a.shape[1]),
        "kl_mean": float(kl.mean()),
        "kl_median": float(np.median(kl)),
        "kl_p95": float(np.percentile(kl, 95)),
        "kl_p99": float(np.percentile(kl, 99)),
        "kl_worst": float(kl.max()),
        "kl_worst_step": int(np.argmax(kl)),
        "dlogit_worst": float(dl.max()),
        "dlogit_p50": float(np.percentile(dl, 50)),
        "dlogit_p95": float(np.percentile(dl, 95)),
        "dlogit_p99": float(np.percentile(dl, 99)),
        "top1_agreement": float((ref_win == cand_win).mean()),
        "top1_flips": [int(i) for i in flips],
        "rank_worst": int(rank.max()),
        "margin_at_flips": [float(margin[i]) for i in flips],
        "margin_min": float(margin.min()),
        "per_step_kl": [float(v) for v in kl],
        "per_step_dlogit": [float(v) for v in dl.max(axis=1)],
    }


def check_limits(m, limit):
    """Compare a metric bundle with limits agreed in advance. Returns [(name, value, limit, ok)]."""
    out = []
    checks = [("kl_mean", "<="), ("kl_p99", "<="), ("dlogit_worst", "<="), ("top1_agreement", ">=")]
    for name, op in checks:
        if name not in limit:
            continue
        want = limit[name]
        got = m[name]
        ok = got <= want if op == "<=" else got >= want
        out.append((name, got, want, bool(ok)))
    if "margin_floor" in limit:
        floor = limit["margin_floor"]
        bad = [mv for mv in m["margin_at_flips"] if mv > floor]
        out.append(("rank-1 flips above margin floor", len(bad), 0, not bad))
    return out


def cache_key(config):
    """A stable identity for a reference dump: every setting that changes its numbers must be in it."""
    canon = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


def run(exe, argv, env, out_path, manifest_path, timeout=7200):
    """Run one executable with an EXPLICIT environment, keep its output and record what was run.

    `env` is the complete environment for the child: nothing is inherited, so a stray `STRATA_*` in the caller's
    shell cannot change what was measured.  The dump is never deleted.
    """
    for forbidden in ("CUDA_VISIBLE_DEVICES", "STRATA_KOLIBRI_EXPERT_F16", "STRATA_DUMP_CTX"):
        if forbidden in os.environ and forbidden not in env:
            raise InvalidInput(f"{forbidden} is set in this shell but not in the explicit environment; "
                               f"refusing to inherit a setting that changes the numbers")
    record = {"exe": str(exe), "argv": [str(a) for a in argv], "env": dict(sorted(env.items())),
              "out": str(out_path), "manifest": str(manifest_path)}
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, sort_keys=True)
    p = subprocess.run([str(exe)] + [str(a) for a in argv], env=dict(env), capture_output=True, text=True,
                       timeout=timeout)
    record["exit"] = p.returncode
    record["stdout_tail"] = p.stdout[-2000:]
    record["stderr_tail"] = p.stderr[-2000:]
    record["bytes"] = os.path.getsize(out_path) if os.path.exists(out_path) else None
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, sort_keys=True)
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref")
    ap.add_argument("--cand")
    ap.add_argument("--vocab", type=int, default=VOCAB)
    ap.add_argument("--records", type=int, default=9)
    ap.add_argument("--limit", help="JSON object of agreed limits, e.g. '{\"kl_mean\": 0.001}'")
    ap.add_argument("--cache-key", help="print the identity of a reference configuration and exit")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.cache_key:
        print(cache_key(json.loads(args.cache_key)))
        return 0
    if not (args.ref and args.cand):
        ap.error("--ref and --cand are required unless --cache-key is used")

    try:
        m = metrics(load_exact(args.ref, args.vocab, args.records),
                    load_exact(args.cand, args.vocab, args.records))
    except NonFiniteInput as e:
        print(f"FAIL non-finite: {e}", flush=True)
        return 1
    except (InvalidInput, OSError) as e:
        print(f"FATAL {e}", flush=True)
        return 2

    limit = json.loads(args.limit) if args.limit else None
    if limit:
        m["limits"] = [{"metric": n, "value": v, "limit": w, "ok": ok} for n, v, w, ok in check_limits(m, limit)]

    if args.json:
        print(json.dumps(m, indent=1))
        return 0
    print(f"records {m['records']} x {m['vocab']}")
    print(f"KL(P||Q)  mean {m['kl_mean']:.6g}  median {m['kl_median']:.6g}  p95 {m['kl_p95']:.6g}  "
          f"p99 {m['kl_p99']:.6g}  worst {m['kl_worst']:.6g} (step {m['kl_worst_step']})")
    print(f"|dlogit|  p50 {m['dlogit_p50']:.4g}  p95 {m['dlogit_p95']:.4g}  p99 {m['dlogit_p99']:.4g}  "
          f"worst {m['dlogit_worst']:.4g}")
    print(f"top-1 agreement {m['top1_agreement']:.4f}; rank-1 flips {m['top1_flips'] or 'none'}"
          f"{'' if not m['top1_flips'] else ' with reference margins ' + str([round(v, 4) for v in m['margin_at_flips']])}")
    print(f"worst rank of the reference winner {m['rank_worst']}; smallest reference margin {m['margin_min']:.4g}")
    if limit:
        for row in m["limits"]:
            print(f"  limit {row['metric']}: {row['value']:.6g} {row['limit']} -> {'PASS' if row['ok'] else 'FAIL'}")
        return 0 if all(r["ok"] for r in m["limits"]) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
