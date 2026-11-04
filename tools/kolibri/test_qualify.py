"""Tests for the qualification harness.

The harness exists because the diagnostic script before it could silently weaken a result: a fixed context, a
`min()` truncation, a filename-keyed reference cache, and deletion of the raw dumps. Each of those is pinned here
by a test, so the harness cannot quietly regress into the thing it replaced.

The metrics themselves are pinned against arithmetic, not against the implementation: the same hand-computed KL
case the comparator uses, plus a top-1 flip whose reference margin has to come out as the reference's own number.
"""
import json
import math
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qualify  # noqa: E402

VOCAB = 8


def dump(path, rows, vocab=VOCAB):
    a = np.asarray(rows, dtype=np.float32)
    assert a.shape[1] == vocab
    path.write_bytes(a.tobytes(order="C"))
    return path


def test_exact_record_count_is_required_not_truncated(tmp_path):
    short = dump(tmp_path / "short.bin", [[0.0] * VOCAB] * 2)
    with pytest.raises(qualify.InvalidInput):
        qualify.load_exact(short, VOCAB, 3)
    long = dump(tmp_path / "long.bin", [[0.0] * VOCAB] * 4)
    with pytest.raises(qualify.InvalidInput):
        qualify.load_exact(long, VOCAB, 3)          # an extra record is not "close enough" either
    exact = dump(tmp_path / "exact.bin", [[0.0] * VOCAB] * 3)
    assert qualify.load_exact(exact, VOCAB, 3).shape == (3, VOCAB)


def test_hand_computed_kl(tmp_path):
    ref = dump(tmp_path / "ref.bin", [[0.0, 0.0, 0.0]], vocab=3)
    cand = dump(tmp_path / "cand.bin", [[math.log(2.0), 0.0, 0.0]], vocab=3)
    m = qualify.metrics(qualify.load_exact(ref, 3, 1), qualify.load_exact(cand, 3, 1))
    assert m["kl_mean"] == pytest.approx(0.0566330, abs=5e-7)
    assert m["kl_worst"] == m["kl_mean"]


def test_dlogit_percentiles_and_worst(tmp_path):
    ref = np.zeros((4, VOCAB), dtype=np.float32)
    cand = np.zeros((4, VOCAB), dtype=np.float32)
    cand[1, 0] = 5.0
    cand[2, 1] = 0.5
    a = dump(tmp_path / "a.bin", ref)
    b = dump(tmp_path / "b.bin", cand)
    m = qualify.metrics(qualify.load_exact(a, VOCAB, 4), qualify.load_exact(b, VOCAB, 4))
    assert m["dlogit_worst"] == pytest.approx(5.0)
    assert m["dlogit_p50"] == 0.0
    assert m["per_step_dlogit"][1] == pytest.approx(5.0)


def test_top1_flip_carries_the_reference_margin(tmp_path):
    ref = dump(tmp_path / "a.bin", [[3.0, 1.0, 0.0]], vocab=3)
    cand = dump(tmp_path / "b.bin", [[1.0, 3.0, 0.0]], vocab=3)
    m = qualify.metrics(qualify.load_exact(ref, 3, 1), qualify.load_exact(cand, 3, 1))
    assert m["top1_agreement"] == 0.0
    assert m["top1_flips"] == [0]
    assert m["margin_at_flips"][0] == pytest.approx(2.0)      # the reference's own 3.0 - 1.0
    assert m["rank_worst"] == 2


def test_non_finite_is_rejected(tmp_path):
    ref = dump(tmp_path / "a.bin", [[0.0] * VOCAB])
    nan = np.zeros((1, VOCAB), dtype=np.float32)
    nan[0, 0] = np.nan
    b = tmp_path / "b.bin"
    b.write_bytes(nan.tobytes())
    with pytest.raises(qualify.NonFiniteInput):
        qualify.metrics(qualify.load_exact(ref, VOCAB, 1), qualify.load_exact(b, VOCAB, 1))


def test_cache_key_changes_with_every_setting_that_changes_the_numbers():
    base = {"ctx": 12288, "ngl": 99, "cpu_moe": 999, "model": "kolibri", "kv_type": "f16"}
    k = qualify.cache_key(base)
    assert k == qualify.cache_key(dict(reversed(list(base.items()))))     # order must not matter
    for field, other in (("ctx", 14336), ("ngl", 0), ("cpu_moe", 0), ("kv_type", "f32"), ("model", "other")):
        changed = dict(base)
        changed[field] = other
        assert qualify.cache_key(changed) != k, field


def test_limits_are_applied_as_agreed():
    m = {"kl_mean": 0.0005, "kl_p99": 0.002, "dlogit_worst": 0.01, "top1_agreement": 1.0,
         "margin_at_flips": []}
    rows = qualify.check_limits(m, {"kl_mean": 0.001, "kl_p99": 0.005, "dlogit_worst": 0.05,
                                    "top1_agreement": 1.0, "margin_floor": 0.05})
    assert all(r[3] for r in rows)
    worse = dict(m, kl_mean=0.4, dlogit_worst=11.0)
    rows = qualify.check_limits(worse, {"kl_mean": 0.001, "dlogit_worst": 0.05})
    assert not all(r[3] for r in rows)


def test_margin_floor_rejects_a_flip_above_it():
    m = {"margin_at_flips": [0.01, 0.5], "kl_mean": 0.0, "kl_p99": 0.0, "dlogit_worst": 0.0,
         "top1_agreement": 0.9}
    rows = qualify.check_limits(m, {"margin_floor": 0.05})
    assert rows[0][0].startswith("rank-1 flips")
    assert rows[0][3] is False


def test_run_passes_only_the_explicit_environment_and_keeps_the_dump(tmp_path, monkeypatch):
    """A stray STRATA_* in the caller's shell must not reach the child, and the dump must survive."""
    script = tmp_path / "fake.sh"
    script.write_text("#!/bin/sh\necho \"$STRATA_KOLIBRI_EXPERT_F16\" > \"$1\"\nexit 0\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    out = tmp_path / "cand.bin"
    man = tmp_path / "manifest.json"
    monkeypatch.setenv("STRATA_KOLIBRI_EXPERT_F16", "24")
    with pytest.raises(qualify.InvalidInput):
        qualify.run(script, [out], {"PATH": os.environ.get("PATH", "/usr/bin")}, out, man)
    rec = qualify.run(script, [out], {"PATH": os.environ.get("PATH", "/usr/bin"),
                                      "STRATA_KOLIBRI_EXPERT_F16": "24"}, out, man)
    assert rec["exit"] == 0
    assert out.exists() and out.read_text().strip() == "24"
    saved = json.loads(man.read_text())
    assert saved["env"]["STRATA_KOLIBRI_EXPERT_F16"] == "24"
    assert saved["exit"] == 0
    assert "argv" in saved and "exe" in saved


def test_run_reports_a_failing_executable(tmp_path):
    script = tmp_path / "fail.sh"
    script.write_text("#!/bin/sh\nexit 3\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    rec = qualify.run(script, [], {"PATH": os.environ.get("PATH", "/usr/bin")},
                      tmp_path / "x.bin", tmp_path / "m.json")
    assert rec["exit"] == 3
    assert rec["bytes"] is None


def test_cli_exit_codes(tmp_path, capsys):
    a = dump(tmp_path / "a.bin", [[0.0] * VOCAB] * 2)
    b = dump(tmp_path / "b.bin", [[0.0] * VOCAB] * 2)
    assert qualify.main(["--ref", str(a), "--cand", str(b), "--vocab", str(VOCAB), "--records", "2"]) == 0
    assert qualify.main(["--ref", str(a), "--cand", str(b), "--vocab", str(VOCAB), "--records", "3"]) == 2
    nan = np.zeros((2, VOCAB), dtype=np.float32)
    nan[1, 1] = np.inf
    c = tmp_path / "c.bin"
    c.write_bytes(nan.tobytes())
    assert qualify.main(["--ref", str(a), "--cand", str(c), "--vocab", str(VOCAB), "--records", "2"]) == 1


def test_cli_limit_gives_a_verdict_only_when_limits_are_given(tmp_path):
    a = dump(tmp_path / "a.bin", [[0.0] * VOCAB] * 2)
    assert qualify.main(["--ref", str(a), "--cand", str(a), "--vocab", str(VOCAB), "--records", "2",
                         "--limit", '{"kl_mean": 0.001}']) == 0


def test_cache_key_cli(tmp_path, capsys):
    assert qualify.main(["--cache-key", '{"ctx": 12288, "ngl": 99, "cpu_moe": 999}']) == 0
    out = capsys.readouterr().out.strip()
    assert len(out) == 16 and all(c in "0123456789abcdef" for c in out)
