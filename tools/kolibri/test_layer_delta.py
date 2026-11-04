"""Tests for the per-layer residual comparison used to bisect the EXPERT_F16 divergence.

The dump the Kolibri CLI writes under `--dump-layers` is, per processed position, `n_layers` x `n_embd` float32 in
layer order: `src/kolibri/kolibri_engine.cu` calls `forward(..., layers.data())` and then writes `n_embd` floats per
layer, once per position. At `n_layers = 50`, `n_embd = 2560` that is 512,000 bytes per position, so the gate's nine
positions are 4,608,000 bytes.

The reader has to be exact about that layout, and the metrics have to keep a worst case visible: a median over nine
positions can hide the single position that carries the error, which is precisely the position a bisection needs.
"""
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer_delta  # noqa: E402

LAYERS, EMBD = 3, 4


def dump(path, rows):
    a = np.asarray(rows, dtype=np.float32)
    assert a.shape[1:] == (LAYERS, EMBD), a.shape
    path.write_bytes(a.tobytes(order="C"))
    return path


def test_load_uses_position_major_layer_minor_layout(tmp_path):
    rows = np.arange(2 * LAYERS * EMBD, dtype=np.float32).reshape(2, LAYERS, EMBD)
    p = dump(tmp_path / "a.layers", rows)
    got = layer_delta.load_layers(p, LAYERS, EMBD)
    assert got.shape == (2, LAYERS, EMBD)
    assert np.array_equal(got, rows)
    # layer 1 of position 0 starts one layer into the file, not one position in
    raw = np.fromfile(p, dtype=np.float32)
    assert np.array_equal(got[0, 1], raw[EMBD:2 * EMBD])
    assert np.array_equal(got[1, 0], raw[LAYERS * EMBD:LAYERS * EMBD + EMBD])


def test_identical_dumps_have_zero_delta_everywhere(tmp_path):
    rows = np.random.default_rng(0).standard_normal((2, LAYERS, EMBD)).astype(np.float32)
    a = dump(tmp_path / "a.layers", rows)
    b = dump(tmp_path / "b.layers", rows)
    m = layer_delta.layer_metrics(layer_delta.load_layers(a, LAYERS, EMBD),
                                  layer_delta.load_layers(b, LAYERS, EMBD))
    assert m["rel"].shape == (2, LAYERS)
    assert float(np.nanmax(m["rel"])) == 0.0
    assert m["first_exceeding"] is None


def test_a_difference_confined_to_one_layer_is_located(tmp_path):
    base = np.ones((1, LAYERS, EMBD), dtype=np.float32)
    other = base.copy()
    other[0, 2, :] = 2.0                      # only layer 2 differs, by 100% relative
    a = dump(tmp_path / "a.layers", base)
    b = dump(tmp_path / "b.layers", other)
    m = layer_delta.layer_metrics(layer_delta.load_layers(a, LAYERS, EMBD),
                                  layer_delta.load_layers(b, LAYERS, EMBD))
    assert m["rel"][0, 0] == 0.0 and m["rel"][0, 1] == 0.0
    assert m["rel"][0, 2] == pytest.approx(1.0)
    assert m["first_exceeding"] == (0, 2)


def test_earliest_divergence_is_record_major_then_layer(tmp_path):
    base = np.ones((3, LAYERS, EMBD), dtype=np.float32)
    other = base.copy()
    other[2, 0, :] = 5.0                     # later record, first layer
    other[1, 2, :] = 5.0                     # earlier record, last layer
    a = dump(tmp_path / "a.layers", base)
    b = dump(tmp_path / "b.layers", other)
    m = layer_delta.layer_metrics(layer_delta.load_layers(a, LAYERS, EMBD),
                                  layer_delta.load_layers(b, LAYERS, EMBD))
    assert m["first_exceeding"] == (1, 2)


def test_zero_reference_norm_is_reported_not_hidden(tmp_path):
    base = np.ones((1, LAYERS, EMBD), dtype=np.float32)
    base[0, 1, :] = 0.0                     # a layer whose reference output is exactly zero
    other = base.copy()
    other[0, 1, :] = 3.0
    a = dump(tmp_path / "a.layers", base)
    b = dump(tmp_path / "b.layers", other)
    m = layer_delta.layer_metrics(layer_delta.load_layers(a, LAYERS, EMBD),
                                  layer_delta.load_layers(b, LAYERS, EMBD))
    assert m["undefined"][0, 1]
    assert math.isinf(m["rel"][0, 1])
    # a zero against a zero is not a divergence at all
    base[0, 2, :] = 0.0
    other[0, 2, :] = 0.0
    a = dump(tmp_path / "a2.layers", base)
    b = dump(tmp_path / "b2.layers", other)
    m = layer_delta.layer_metrics(layer_delta.load_layers(a, LAYERS, EMBD),
                                  layer_delta.load_layers(b, LAYERS, EMBD))
    assert m["rel"][0, 2] == 0.0
    assert not m["undefined"][0, 2]


def test_one_outlier_record_survives_a_median(tmp_path):
    """The reason the report carries max and median: the median over records would hide position 4."""
    rows = np.random.default_rng(1).standard_normal((5, LAYERS, EMBD)).astype(np.float32)
    other = rows.copy()
    other[4, 1, :] += 50.0
    a = dump(tmp_path / "a.layers", rows)
    b = dump(tmp_path / "b.layers", other)
    m = layer_delta.layer_metrics(layer_delta.load_layers(a, LAYERS, EMBD),
                                  layer_delta.load_layers(b, LAYERS, EMBD))
    per_record_max = m["max_by_record"]
    assert per_record_max[4] > 10.0
    assert float(np.median(per_record_max)) < 1e-6
    # and the layer summary names the layer that carries it
    assert int(np.argmax(m["max_by_layer"])) == 1


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_values_are_rejected(tmp_path, bad):
    rows = np.zeros((1, LAYERS, EMBD), dtype=np.float32)
    other = rows.copy()
    other[0, 0, 0] = bad
    a = dump(tmp_path / "a.layers", rows)
    b = dump(tmp_path / "b.layers", other)
    with pytest.raises(layer_delta.NonFiniteInput):
        layer_delta.layer_metrics(layer_delta.load_layers(a, LAYERS, EMBD),
                                  layer_delta.load_layers(b, LAYERS, EMBD))


def test_misaligned_size_is_rejected_not_truncated(tmp_path):
    p = tmp_path / "a.layers"
    p.write_bytes(np.zeros((1, LAYERS, EMBD), dtype=np.float32).tobytes()[:-4])
    with pytest.raises(layer_delta.InvalidInput):
        layer_delta.load_layers(p, LAYERS, EMBD)


def test_empty_and_invalid_dimensions_are_rejected(tmp_path):
    p = tmp_path / "a.layers"
    p.write_bytes(b"")
    with pytest.raises(layer_delta.InvalidInput):
        layer_delta.load_layers(p, LAYERS, EMBD)
    good = dump(tmp_path / "good.layers", np.zeros((1, LAYERS, EMBD), dtype=np.float32))
    for layers, embd in ((0, EMBD), (LAYERS, 0), (-1, EMBD), (LAYERS, -4)):
        with pytest.raises(layer_delta.InvalidInput):
            layer_delta.load_layers(good, layers, embd)


def test_record_count_mismatch_is_rejected(tmp_path):
    a = dump(tmp_path / "a.layers", np.zeros((1, LAYERS, EMBD), dtype=np.float32))
    b = dump(tmp_path / "b.layers", np.zeros((2, LAYERS, EMBD), dtype=np.float32))
    with pytest.raises(layer_delta.InvalidInput):
        layer_delta.layer_metrics(layer_delta.load_layers(a, LAYERS, EMBD),
                                  layer_delta.load_layers(b, LAYERS, EMBD))


def test_relative_norm_uses_the_reference_as_the_denominator(tmp_path):
    """||a-b||/||a||, with a the reference: the reference's scale is the denominator, not the candidate's."""
    base = np.full((1, LAYERS, EMBD), 2.0, dtype=np.float32)
    other = base * 3.0
    a = dump(tmp_path / "a.layers", base)
    b = dump(tmp_path / "b.layers", other)
    m = layer_delta.layer_metrics(layer_delta.load_layers(a, LAYERS, EMBD),
                                  layer_delta.load_layers(b, LAYERS, EMBD))
    # ||2-6||/||2|| = 4/2 = 2
    assert m["rel"][0, 0] == pytest.approx(2.0)


def test_cli_exit_codes_and_report(tmp_path, capsys):
    a = dump(tmp_path / "a.layers", np.ones((1, LAYERS, EMBD), dtype=np.float32))
    b = dump(tmp_path / "b.layers", np.ones((1, LAYERS, EMBD), dtype=np.float32))
    rc = layer_delta.main(["--a", str(a), "--b", str(b), "--layers", str(LAYERS), "--embd", str(EMBD)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "median" in out and "max" in out
    # invalid shape -> 2, non-finite -> 1
    bad = tmp_path / "bad.layers"
    bad.write_bytes(b"\x00\x00\x00")
    assert layer_delta.main(["--a", str(bad), "--b", str(b), "--layers", str(LAYERS), "--embd", str(EMBD)]) == 2
    nan = np.ones((1, LAYERS, EMBD), dtype=np.float32)
    nan[0, 0, 0] = float("nan")
    nanp = dump(tmp_path / "nan.layers", np.nan_to_num(nan, nan=0.0))
    raw = np.fromfile(nanp, dtype=np.float32)
    raw[0] = float("nan")
    nanp.write_bytes(raw.tobytes())
    assert layer_delta.main(["--a", str(nanp), "--b", str(b), "--layers", str(LAYERS), "--embd", str(EMBD)]) == 1


def test_script_runs_as_a_program(tmp_path):
    a = dump(tmp_path / "a.layers", np.ones((1, LAYERS, EMBD), dtype=np.float32))
    script = Path(__file__).resolve().parent / "layer_delta.py"
    r = subprocess.run([sys.executable, str(script), "--a", str(a), "--b", str(a),
                        "--layers", str(LAYERS), "--embd", str(EMBD)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
