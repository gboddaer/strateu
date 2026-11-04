"""Tests for the full-logit comparator used by the EXPERT_F16 correctness investigation.

The comparator is an instrument, not an acceptance gate: a pass here says two dump files were read correctly and
compared the way the caller asked, not that the engine is accurate. The tests pin the parts that decide whether a
reported number means anything:

  * the hand-computed KL case from the plan, so the metric itself is checked against arithmetic, not against the
    implementation;
  * bytewise equality that distinguishes -0.0 from 0.0, because a tolerance-based compare cannot;
  * explicit rejection of non-finite values, wrong shapes, empty input and truncated bytes, so a comparison is
    never silently taken over the shorter of two dumps;
  * the reference winner's rank and margin, because a small mean KL over near-ties says less than one large step.
"""
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import klcmp  # noqa: E402

VOCAB = 8


def dump(path, rows, vocab=VOCAB):
    """Write a float32 dump and return its path."""
    a = np.asarray(rows, dtype=np.float32)
    assert a.shape == (len(rows), vocab), a.shape
    path.write_bytes(a.tobytes(order="C"))
    return path


def test_identical_finite_inputs_are_bitwise_and_zero_kl(tmp_path):
    rows = np.arange(3 * VOCAB, dtype=np.float32).reshape(3, VOCAB) / 7.0
    ref = dump(tmp_path / "ref.bin", rows)
    cand = dump(tmp_path / "cand.bin", rows)

    s = klcmp.summary(klcmp.load(ref, VOCAB), klcmp.load(cand, VOCAB))

    assert s["records"] == 3
    assert s["bytewise_identical"] is True
    assert s["differing_records"] == 0
    assert s["max_abs_delta"] == 0.0
    assert s["kl_max"] == 0.0
    assert s["top1_agreement"] == 1.0


def test_hand_computed_kl_matches_closed_form(tmp_path):
    """P=[0,0,0] against Q=[ln2,0,0] gives KL(P||Q) = 1/3 [ln(2/3) + 2 ln(4/3)] = 0.0566330..."""
    ref = dump(tmp_path / "ref.bin", [[0.0, 0.0, 0.0]], vocab=3)
    cand = dump(tmp_path / "cand.bin", [[math.log(2.0), 0.0, 0.0]], vocab=3)

    s = klcmp.summary(klcmp.load(ref, 3), klcmp.load(cand, 3))

    expected = (math.log(2.0 / 3.0) + 2.0 * math.log(4.0 / 3.0)) / 3.0
    assert expected == pytest.approx(0.0566330, abs=5e-7)
    assert s["kl_max"] == pytest.approx(0.0566330, abs=5e-7)
    assert s["kl_mean"] == pytest.approx(0.0566330, abs=5e-7)
    # the KL is directional: KL(Q||P) is a different number, so the argument order must be pinned
    assert s["kl"][0] != pytest.approx(
        (2.0 / 4.0) * math.log((2.0 / 4.0) / (1.0 / 3.0)) + 2 * (1.0 / 4.0) * math.log((1.0 / 4.0) / (1.0 / 3.0)),
        abs=1e-6)


def test_kl_is_reference_against_candidate_not_symmetric(tmp_path):
    """KL(P||Q) != KL(Q||P); swapping the arguments must change the number."""
    ref = dump(tmp_path / "ref.bin", [[0.0, 0.0, 0.0]], vocab=3)
    cand = dump(tmp_path / "cand.bin", [[math.log(2.0), 0.0, 0.0]], vocab=3)
    a = klcmp.summary(klcmp.load(ref, 3), klcmp.load(cand, 3))["kl"][0]
    b = klcmp.summary(klcmp.load(cand, 3), klcmp.load(ref, 3))["kl"][0]
    assert a != pytest.approx(b, abs=1e-9)


def test_top1_flip_is_reported_and_winner_rank_follows(tmp_path):
    ref = dump(tmp_path / "ref.bin", [[0.0, 0.0, 0.0]], vocab=3)
    cand = dump(tmp_path / "cand.bin", [[0.0, 0.0, 5.0]], vocab=3)

    s = klcmp.summary(klcmp.load(ref, 3), klcmp.load(cand, 3))

    assert s["top1_agreement"] == 0.0
    assert s["top1_matching"] == 0
    # the reference winner (index 0) is now ranked 2nd in the candidate (index 2 wins)
    assert s["ref_winner_rank"][0] == 2
    assert s["ref_winner_rank_worst"] == 2
    # a reference margin of 0.0 between two equal logits means any perturbation can flip it
    assert s["ref_margin"][0] == 0.0


def test_reference_margin_and_rank_are_the_references_own_quantities(tmp_path):
    """Margin is measured on the reference; rank of the reference winner is measured in the candidate."""
    ref = dump(tmp_path / "ref.bin", [[3.0, 1.0, 0.0]], vocab=3)
    cand = dump(tmp_path / "cand.bin", [[1.0, 3.0, 0.0]], vocab=3)

    s = klcmp.summary(klcmp.load(ref, 3), klcmp.load(cand, 3))

    assert s["ref_margin"][0] == pytest.approx(2.0)      # 3.0 - 1.0 on the reference
    assert s["ref_winner_rank"][0] == 2                  # index 0 is second in the candidate
    assert s["top1_agreement"] == 0.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_logits_are_rejected(tmp_path, bad):
    ref = dump(tmp_path / "ref.bin", [[0.0] * VOCAB])
    cand = dump(tmp_path / "cand.bin", [[0.0] * (VOCAB - 1) + [bad]])
    with pytest.raises(klcmp.NonFiniteInput):
        klcmp.summary(klcmp.load(ref, VOCAB), klcmp.load(cand, VOCAB))


def test_record_count_mismatch_is_rejected_not_truncated(tmp_path):
    ref = dump(tmp_path / "ref.bin", [[0.0] * VOCAB] * 2)
    cand = dump(tmp_path / "cand.bin", [[0.0] * VOCAB] * 3)
    with pytest.raises(klcmp.InvalidInput):
        klcmp.summary(klcmp.load(ref, VOCAB), klcmp.load(cand, VOCAB))


def test_empty_input_is_rejected(tmp_path):
    ref = tmp_path / "ref.bin"
    ref.write_bytes(b"")
    with pytest.raises(klcmp.InvalidInput):
        klcmp.load(ref, VOCAB)


def test_bytes_not_a_whole_number_of_records_are_rejected(tmp_path):
    full = dump(tmp_path / "ref.bin", [[0.0] * VOCAB] * 2)
    bad = tmp_path / "bad.bin"
    bad.write_bytes(full.read_bytes()[:-3])
    with pytest.raises(klcmp.InvalidInput):
        klcmp.load(bad, VOCAB)


def test_extra_bytes_are_rejected(tmp_path):
    """A dump with a trailing partial record must not be read as if it were whole records."""
    full = dump(tmp_path / "ref.bin", [[0.0] * VOCAB])
    bad = tmp_path / "bad.bin"
    bad.write_bytes(full.read_bytes() + b"\x00\x00")
    with pytest.raises(klcmp.InvalidInput):
        klcmp.load(bad, VOCAB)


@pytest.mark.parametrize("vocab", [0, -1])
def test_invalid_vocab_is_rejected(tmp_path, vocab):
    ref = dump(tmp_path / "ref.bin", [[0.0] * VOCAB])
    with pytest.raises(klcmp.InvalidInput):
        klcmp.load(ref, vocab)


def test_signed_zero_differs_bytewise_but_not_in_value(tmp_path):
    """-0.0 == 0.0 in value; a comparator that only subtracts would call these identical."""
    a = np.zeros(VOCAB, dtype=np.float32)
    b = a.copy()
    b[0] = -0.0
    ref = tmp_path / "ref.bin"
    cand = tmp_path / "cand.bin"
    ref.write_bytes(a.tobytes())
    cand.write_bytes(b.tobytes())

    s = klcmp.summary(klcmp.load(ref, VOCAB), klcmp.load(cand, VOCAB))

    assert s["bytewise_identical"] is False
    assert s["differing_records"] == 1
    assert s["max_abs_delta"] == 0.0
    assert np.signbit(klcmp.load(cand, VOCAB)[0][0])


def test_one_outlier_step_does_not_hide_in_the_median(tmp_path):
    """A comparator must report the worst step next to the median; a median-only report would hide this."""
    rows_ref = np.zeros((5, VOCAB), dtype=np.float32)
    rows_cand = np.zeros((5, VOCAB), dtype=np.float32)
    rows_ref[2] = np.linspace(-4.0, 4.0, VOCAB)
    rows_cand[2] = np.linspace(-4.0, 4.0, VOCAB)
    rows_cand[2][0] += 3.0
    ref = dump(tmp_path / "ref.bin", rows_ref)
    cand = dump(tmp_path / "cand.bin", rows_cand)

    s = klcmp.summary(klcmp.load(ref, VOCAB), klcmp.load(cand, VOCAB))

    assert s["kl_worst_step"] == 2
    assert s["kl_max"] > s["kl_median"]
    assert s["differing_records"] == 1


def test_summary_reports_percentiles_over_all_steps(tmp_path):
    ref_rows, cand_rows = [], []
    for i in range(1, 21):
        ref_rows.append(np.zeros(VOCAB, dtype=np.float32))
        c = np.zeros(VOCAB, dtype=np.float32)
        c[0] = math.log(1.0 + 0.01 * i)   # a graded, monotone perturbation
        cand_rows.append(c)
    ref = dump(tmp_path / "ref.bin", ref_rows)
    cand = dump(tmp_path / "cand.bin", cand_rows)

    s = klcmp.summary(klcmp.load(ref, VOCAB), klcmp.load(cand, VOCAB))

    assert s["records"] == 20
    assert len(s["kl"]) == 20
    assert s["kl_median"] <= s["kl_p95"] <= s["kl_p99"] <= s["kl_max"]
    assert s["kl_worst_step"] == 19


def test_cli_exit_zero_for_comparable_finite_dumps(tmp_path, capsys):
    a = dump(tmp_path / "a.bin", [[0.0] * VOCAB] * 2)
    b = dump(tmp_path / "b.bin", [[0.0] * VOCAB] * 2)
    assert klcmp.main(["--ref", str(a), "--cand", str(b), "--vocab", str(VOCAB)]) == 0
    out = capsys.readouterr().out
    assert "bitwise identical" in out


def test_cli_exit_one_for_non_finite_input(tmp_path):
    a = dump(tmp_path / "a.bin", [[0.0] * VOCAB])
    b = dump(tmp_path / "b.bin", [[0.0] * (VOCAB - 1) + [float("nan")]])
    assert klcmp.main(["--ref", str(a), "--cand", str(b), "--vocab", str(VOCAB)]) == 1


def test_cli_exit_two_for_shape_problems(tmp_path):
    a = dump(tmp_path / "a.bin", [[0.0] * VOCAB] * 2)
    b = dump(tmp_path / "b.bin", [[0.0] * VOCAB])
    assert klcmp.main(["--ref", str(a), "--cand", str(b), "--vocab", str(VOCAB)]) == 2
    assert klcmp.main(["--ref", str(a), "--cand", str(b), "--vocab", "0"]) == 2


def test_cli_json_output_round_trips(tmp_path, capsys):
    import json
    a = dump(tmp_path / "a.bin", [[0.0, 0.0, 0.0]], vocab=3)
    b = dump(tmp_path / "b.bin", [[math.log(2.0), 0.0, 0.0]], vocab=3)
    assert klcmp.main(["--ref", str(a), "--cand", str(b), "--vocab", "3", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["records"] == 1
    assert payload["kl_max"] == pytest.approx(0.0566330, abs=5e-7)
    assert payload["vocab"] == 3


def test_script_runs_as_a_program(tmp_path):
    """The CLI has to work when invoked the way the plan calls it, not only through main()."""
    a = dump(tmp_path / "a.bin", [[0.0] * VOCAB] * 2)
    script = Path(__file__).resolve().parent / "klcmp.py"
    r = subprocess.run([sys.executable, str(script), "--ref", str(a), "--cand", str(a),
                        "--vocab", str(VOCAB)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "bitwise identical" in r.stdout
