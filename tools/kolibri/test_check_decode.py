#!/usr/bin/env python3
"""CPU-only tests for tools/kolibri/check_decode.py.

    .venv/bin/python -m pytest tools/kolibri/test_check_decode.py -q
"""
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_decode as cd  # noqa: E402


def write_logits(path: Path, records):
    with open(path, "wb") as f:
        for rec in records:
            f.write(struct.pack(f"<{len(rec)}f", *rec))


def test_identical_files_pass(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    recs = [[0.0, 1.5, -2.25], [3.0, 4.0, 5.0]]
    write_logits(a, recs)
    write_logits(b, recs)
    r = cd.compare(a, b, vocab=3, records=2)
    assert r["records"] == 2
    assert r["values"] == 6
    assert r["max_abs_diff"] == 0.0


def test_one_changed_bit_fails(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    write_logits(a, [[1.0, 2.0, 3.0]])
    write_logits(b, [[1.0, 2.0, 3.0000002]])   # a strictly larger value, so it is not a float tie
    with pytest.raises(cd.Mismatch) as e:
        cd.compare(a, b, vocab=3, records=1)
    assert "record 0" in str(e.value)
    assert "index 2" in str(e.value)


def test_last_bit_only_difference_fails(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    with open(a, "wb") as f:
        f.write(struct.pack("<f", 1.0))
    with open(b, "wb") as f:
        # 1.0 with the lowest mantissa bit set: equal to any tolerance anyone would pick
        f.write(struct.pack("<f", struct.unpack("<I", struct.pack("<f", 1.0))[0] + 1 and 1.0000001192092896))
    with pytest.raises(cd.Mismatch):
        cd.compare(a, b, vocab=1, records=1)


def test_nan_and_inf_are_rejected(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    write_logits(a, [[float("nan"), 1.0]])
    write_logits(b, [[float("nan"), 1.0]])
    with pytest.raises(cd.Mismatch) as e:
        cd.compare(a, b, vocab=2, records=1)
    assert "finite" in str(e.value)

    write_logits(a, [[float("inf"), 1.0]])
    write_logits(b, [[float("inf"), 1.0]])
    with pytest.raises(cd.Mismatch):
        cd.compare(a, b, vocab=2, records=1)


def test_short_and_long_files_fail(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    write_logits(a, [[1.0, 2.0]])
    write_logits(b, [[1.0, 2.0], [3.0, 4.0]])       # one record too many
    with pytest.raises(cd.Mismatch) as e:
        cd.compare(a, b, vocab=2, records=1)
    assert "size" in str(e.value)
    with pytest.raises(cd.Mismatch):
        cd.compare(a, b, vocab=2, records=2)        # a size that does not match either file


def test_vocab_mismatch_fails(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    write_logits(a, [[1.0, 2.0, 3.0]])
    write_logits(b, [[1.0, 2.0, 3.0]])
    with pytest.raises(cd.Mismatch) as e:
        cd.compare(a, b, vocab=4, records=1)
    # the file does not hold a whole number of 4-value records: the shape the caller declared is wrong
    assert "size" in str(e.value) and "x 4 float32" in str(e.value)


def test_negative_zero_is_not_a_tie(tmp_path):
    # -0.0 == 0.0 numerically, so a tolerance comparison would call this equal; bitwise it is a difference.
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    write_logits(a, [[0.0]])
    write_logits(b, [[-0.0]])
    with pytest.raises(cd.Mismatch):
        cd.compare(a, b, vocab=1, records=1)


def test_reports_the_first_differing_record(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    write_logits(a, [[1.0], [2.0], [3.0]])
    write_logits(b, [[1.0], [2.5], [3.0]])
    with pytest.raises(cd.Mismatch) as e:
        cd.compare(a, b, vocab=1, records=3)
    assert "record 1" in str(e.value)


def test_cli_exit_codes(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    write_logits(a, [[1.0, 2.0]])
    write_logits(b, [[1.0, 2.0]])
    assert cd.main(["--baseline", str(a), "--candidate", str(b), "--vocab", "2", "--records", "1"]) == 0
    write_logits(b, [[1.0, 2.5]])
    assert cd.main(["--baseline", str(a), "--candidate", str(b), "--vocab", "2", "--records", "1"]) == 1
