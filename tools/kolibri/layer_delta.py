#!/usr/bin/env python3
"""Per-layer residual comparison for the Kolibri EXPERT_F16 bisection.

Input files are `--dump-layers` dumps: per processed position, `n_layers` x `n_embd` float32 written in layer order
(`src/kolibri/kolibri_engine.cu`: `forward(..., layers.data())` then `n_embd` floats per layer). At the gate's shape
that is 50 x 2560 x 4 = 512,000 bytes per position, 4,608,000 for nine positions.

Reported per comparison: `||a-b|| / ||a||` for every (position, layer), with `a` the reference (the accuracy
control, whose own scale is the denominator); the maximum and the
median over positions for every layer, and over layers for every position; the first (position, layer) above a
relative threshold; and the number of layer outputs whose reference is exactly zero while the candidate is not, where the ratio is
undefined and an absolute difference is the only meaningful statement.

What this is for: choosing the layer and position that Task 4 instruments directly. A layer-by-layer residual
difference is NOT evidence that the divergence was produced in that layer - decode residual propagation, router
sensitivity and earlier KV all mix into it. A median alone would also hide a single bad position, so both are shown.

    layer_delta.py --a REF.layers --b CAND.layers --layers 50 --embd 2560 --rel 1e-3
"""
import argparse
import sys

import numpy as np

N_LAYER, N_EMBD = 50, 2560


class InvalidInput(ValueError):
    """The dumps cannot be compared: wrong size, wrong shape, empty or contradictory arguments."""


class NonFiniteInput(ValueError):
    """A residual value is NaN or infinite: a correctness failure, not a rounding difference."""


def load_layers(path, n_layer, n_embd):
    """Read a `--dump-layers` file as float32 [positions, layers, embedding]."""
    for name, v in (("layers", n_layer), ("embd", n_embd)):
        if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
            raise InvalidInput(f"{name} must be a positive integer, got {v!r}")
    stride = 4 * n_layer * n_embd
    try:
        import os
        size = os.path.getsize(str(path))
    except OSError as e:
        raise InvalidInput(f"{path}: {e}") from e
    if size == 0:
        raise InvalidInput(f"{path}: empty dump; a comparison needs at least one position")
    if size % stride:
        raise InvalidInput(
            f"{path}: {size} bytes is not a whole number of {n_layer}-layer x {n_embd}-value positions "
            f"({stride} bytes each); refusing to truncate")
    a = np.fromfile(str(path), dtype=np.dtype("<f4"))
    if a.size * 4 != size:
        raise InvalidInput(f"{path}: read {a.size} value(s) for a {size}-byte file")
    return a.reshape(-1, n_layer, n_embd)


def layer_metrics(ref, cand, rel_tol=1e-3):
    """Per-(position, layer) relative L2 difference, with its distribution and its first crossing."""
    a = np.asarray(ref, dtype=np.float64)
    b = np.asarray(cand, dtype=np.float64)
    if a.ndim != 3 or b.ndim != 3:
        raise InvalidInput(f"expected two 3-D arrays, got shapes {a.shape} and {b.shape}")
    if a.shape != b.shape:
        raise InvalidInput(f"shape mismatch: {a.shape} against {b.shape}; refusing to compare the shorter part")
    finite = np.isfinite(a).all() and np.isfinite(b).all()
    if not finite:
        bad = int((~np.isfinite(a)).sum() + (~np.isfinite(b)).sum())
        raise NonFiniteInput(f"{bad} non-finite residual value(s); a comparison cannot be taken over them")

    diff = np.linalg.norm(a - b, axis=2)
    nb = np.linalg.norm(a, axis=2)          # the reference's own scale: ||a-b||/||a||
    # 0/0 is not a divergence (report 0); 0 in the reference with a non-zero candidate has no ratio at all
    undefined = (nb == 0.0) & (diff != 0.0)
    rel = np.where(undefined, np.inf, diff / np.where(nb == 0.0, 1.0, nb))

    exceeding = np.argwhere(rel > rel_tol)
    first = tuple(int(x) for x in exceeding[0]) if len(exceeding) else None
    return {
        "positions": int(a.shape[0]),
        "layers": int(a.shape[1]),
        "embd": int(a.shape[2]),
        "rel": rel,
        "undefined": undefined,
        "undefined_count": int(undefined.sum()),
        "max_by_layer": rel.max(axis=0),
        "median_by_layer": np.median(rel, axis=0),
        "max_by_record": rel.max(axis=1),
        "median_by_record": np.median(rel, axis=1),
        "max_rel": float(rel.max()),
        "median_rel": float(np.median(rel)),
        "worst": tuple(int(x) for x in np.unravel_index(int(np.argmax(rel)), rel.shape)),
        "first_exceeding": first,
        "exceeding_count": int(len(exceeding)),
        "rel_tol": rel_tol,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="reference layer dump")
    ap.add_argument("--b", required=True, help="candidate layer dump")
    ap.add_argument("--layers", type=int, default=N_LAYER)
    ap.add_argument("--embd", type=int, default=N_EMBD)
    ap.add_argument("--rel", type=float, default=1e-3,
                    help="relative L2 threshold used to name the first divergence (navigation, not an "
                         "acceptance limit)")
    args = ap.parse_args(argv)

    try:
        m = layer_metrics(load_layers(args.a, args.layers, args.embd),
                          load_layers(args.b, args.layers, args.embd), rel_tol=args.rel)
    except NonFiniteInput as e:
        print(f"FAIL non-finite: {e}", flush=True)
        return 1
    except (InvalidInput, OSError) as e:
        print(f"FATAL {e}", flush=True)
        return 2

    print(f"positions {m['positions']} x layers {m['layers']} x embd {m['embd']}; "
          f"relative L2 ||a-b||/||b|| (b = reference)")
    print(f"overall  median {m['median_rel']:.6g}  max {m['max_rel']:.6g} at position {m['worst'][0]} "
          f"layer {m['worst'][1]}")
    print("per record (median, max, worst layer):")
    for p in range(m["positions"]):
        layer = int(np.argmax(m["rel"][p]))
        print(f"  rec {p:2d}  median {m['median_by_record'][p]:.6g}  max {m['max_by_record'][p]:.6g}  "
              f"at layer {layer}")
    print(f"per layer median over positions: " +
          " ".join(f"{v:.3g}" for v in m["median_by_layer"]))
    print(f"per layer max   over positions: " +
          " ".join(f"{v:.3g}" for v in m["max_by_layer"]))
    print(f"positions/layers above {m['rel_tol']:g}: {m['exceeding_count']}; "
          f"first {m['first_exceeding'] if m['first_exceeding'] else 'none'}; "
          f"layer outputs with a zero reference and a non-zero candidate: {m['undefined_count']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
