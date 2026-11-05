"""CPU checks of the full-logit growth gate, not GPU/model qualification."""
from pathlib import Path
import struct
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stream_growth_gate
from stream_growth_gate import compare_logits


def record(path, values):
    path.write_bytes(struct.pack("=iii", 0, 513, len(values))
                     + struct.pack(f"={len(values)}f", *values))
    return path


def test_growth_plan_also_returns_to_smaller_requests():
    ids = list(range(513))
    plans = stream_growth_gate.growth_sequences(ids)
    assert [[len(p) for p in prompts] for _, prompts in plans] == [
        [16, 513, 16, 513], [16, 257, 513, 257, 513]]
    assert all(prompts[-1] == ids for _, prompts in plans)


def test_full_logit_gate_rejects_equal_winners_with_different_scores(tmp_path):
    a = record(tmp_path / "a", [10, 1])
    b = record(tmp_path / "b", [10, 2])
    assert not compare_logits(a, b)


def test_full_logit_gate_accepts_identical_finite_scores(tmp_path):
    a = record(tmp_path / "a", [10, 1])
    assert compare_logits(a, a)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_full_logit_gate_rejects_nonfinite(tmp_path, bad):
    a = record(tmp_path / "a", [10, 1])
    b = record(tmp_path / "b", [10, bad])
    with pytest.raises(ValueError, match="non-finite"):
        compare_logits(a, b)


def test_full_logit_gate_rejects_empty_and_truncated_dumps(tmp_path):
    empty = tmp_path / "empty"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="missing"):
        compare_logits(empty, empty)
    broken = tmp_path / "broken"
    broken.write_bytes(struct.pack("=iii", 0, 513, 2) + b"\0" * 4)
    with pytest.raises(ValueError, match="truncated"):
        compare_logits(broken, broken)
