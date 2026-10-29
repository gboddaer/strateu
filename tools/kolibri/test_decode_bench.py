#!/usr/bin/env python3
"""CPU-only tests for tools/kolibri/decode_bench.py.

No GPU, no model, no engine binary: the parsers are pure and the harness is driven against small fake
children that speak (or deliberately break) the serve protocol.  Run:

    .venv/bin/python -m pytest tools/kolibri/test_decode_bench.py -q
"""
import csv
import os
import re
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import decode_bench as db  # noqa: E402

# The engine's line (src/kolibri/kolibri_engine.cu, the DONE printf at the end of serve()):
#   DONE produced prompt_ids prompt_ms decode_ms finish 0 0 reuse hits lookups misses 0 0.0 fresh 0
DONE_16 = "DONE 128 512 1234.5 8000.0 length 0 0 0 0 768 768 0 0.0 512 0"


# --------------------------------------------------------------------------------------------------------------
# DONE parsing
# --------------------------------------------------------------------------------------------------------------
def test_parse_done_valid_is_16_tok_s():
    d = db.parse_done(DONE_16)
    assert d["produced"] == 128
    assert d["prompt_ids"] == 512
    assert d["prompt_ms"] == pytest.approx(1234.5)
    assert d["decode_ms"] == pytest.approx(8000.0)
    assert d["finish"] == "length"
    assert d["reused"] == 0
    assert d["hits"] == 0
    assert d["lookups"] == 768
    assert d["misses"] == 768
    assert d["fresh"] == 512
    assert d["tok_per_s"] == pytest.approx(16.0)


def test_parse_done_reads_request_local_hit_counters():
    d = db.parse_done("DONE 64 512 800.0 4000.0 length 0 0 0 300 768 468 0 0.0 512 0")
    assert (d["hits"], d["lookups"], d["misses"]) == (300, 768, 468)
    assert d["tok_per_s"] == pytest.approx(16.0)


@pytest.mark.parametrize(
    "line",
    [
        "",  # nothing
        "DONE",  # no fields
        "DONE 128 512 1.0 8000.0 length 0 0 0 0 768 768 0 0.0 512",  # one field short
        "DONE 128 512 1.0 8000.0 length 0 0 0 0 768 768 0 0.0 512 0 extra",  # one field long
        "T 42",  # wrong verb
        "DONE 0 512 1.0 8000.0 length 0 0 0 0 0 0 0 0.0 512 0",  # no tokens produced
        "DONE 128 512 1.0 0.0 length 0 0 0 0 0 0 0 0.0 512 0",  # zero decode time
        "DONE 128 512 1.0 -8000.0 length 0 0 0 0 0 0 0 0.0 512 0",  # negative decode time
        "DONE 128 512 1.0 nan length 0 0 0 0 0 0 0 0.0 512 0",  # NaN decode time
        "DONE 128 512 nan 8000.0 length 0 0 0 0 0 0 0 0.0 512 0",  # NaN prompt time
        "DONE 128 512 1.0 inf length 0 0 0 0 0 0 0 0.0 512 0",  # infinite decode time
        "DONE -128 512 1.0 8000.0 length 0 0 0 0 0 0 0 0.0 512 0",  # negative token count
        "DONE 128 512 1.0 8000.0 length 0 0 0 0 700 600 0 0.0 512 0",  # hits+misses != lookups
        "DONE 128 512 1.0 8000.0 length 0 0 600 0 700 700 0 0.0 512 0",  # reuse > prompt ids
    ],
)
def test_parse_done_rejects(line):
    with pytest.raises(db.BenchError):
        db.parse_done(line)


def test_parse_done_accepts_a_reusing_request():
    d = db.parse_done("DONE 8 512 120.0 500.0 length 0 0 384 0 768 768 0 0.0 128 0")
    assert (d["reused"], d["fresh"], d["prompt_ids"]) == (384, 128, 512)
    assert d["tok_per_s"] == pytest.approx(16.0)


# --------------------------------------------------------------------------------------------------------------
# T-line intervals
# --------------------------------------------------------------------------------------------------------------
def test_parse_token_times_five_intervals():
    ts = [0, 60_000_000, 120_000_000, 180_000_000, 240_000_000, 300_000_000]
    s = db.parse_token_times(ts)
    assert s["intervals"] == 5
    assert s["mean_ms"] == pytest.approx(60.0)
    assert s["median_ms"] == pytest.approx(60.0)
    assert s["tok_per_s"] == pytest.approx(1e3 * 5 / 300.0)
    assert s["window_ms"] == pytest.approx(300.0)
    assert s["first_ms"] == pytest.approx(0.0)


def test_parse_token_times_first_token_is_separate_from_steady_state():
    # one slow first token (prompt tail + first decode), then three 50 ms steps
    ts = [0, 400_000_000, 450_000_000, 500_000_000, 550_000_000]
    s = db.parse_token_times(ts)
    assert s["intervals"] == 4
    assert s["first_ms"] == pytest.approx(0.0)
    assert s["first_interval_ms"] == pytest.approx(400.0)
    assert s["median_ms"] == pytest.approx(50.0)
    # the steady window excludes the first token: 3 intervals over the remaining 150 ms
    assert s["steady_tok_per_s"] == pytest.approx(1e3 * 3 / 150.0)


@pytest.mark.parametrize(
    "ts",
    [
        [],  # nothing
        [0],  # nothing after the first
        [0, 60_000_000, 120_000_000],  # only two intervals
        [0, 60_000_000, 60_000_000, 120_000_000, 180_000_000],  # not strictly increasing
        [0, 60_000_000, -120_000_000, 180_000_000, 240_000_000],  # backwards
        "0,60,120,180",  # not a list of ints
    ],
)
def test_parse_token_times_rejects(ts):
    with pytest.raises(db.BenchError):
        db.parse_token_times(ts)


# --------------------------------------------------------------------------------------------------------------
# INFO banner
# --------------------------------------------------------------------------------------------------------------
def test_parse_info():
    info = db.parse_info(
        "INFO engine=kolibri-1 expert_slots=0 expert_slots_per_layer=0 expert_cache_mib=0 "
        "expert_cache_full_mib=0 vram_free_mib=28123 kv=device cpu_threads=8 prefill=1 slots=0"
    )
    assert info["engine"] == "kolibri-1"
    assert info["cpu_threads"] == 8
    assert info["prefill"] == 1
    assert info["vram_free_mib"] == 28123


def test_parse_info_rejects_a_non_info_line():
    with pytest.raises(db.BenchError):
        db.parse_info("READY 8192 stop")


# --------------------------------------------------------------------------------------------------------------
# Kernel trace parsing (diagnostic only)
# --------------------------------------------------------------------------------------------------------------
TRACE_COLUMNS = ["Kernel_Name", "Stream_Id", "Start_Timestamp", "End_Timestamp"]


def write_trace(path: Path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(TRACE_COLUMNS)
        for name, stream, start, end in rows:
            w.writerow([name, stream, start, end])


def test_trace_splits_prompt_from_generation_and_merges_streams(tmp_path):
    # two prompt forwards and three generation forwards; one k_row< marker opens each forward.
    marker = "void (anonymous namespace)::k_row<12>"
    rows = []
    t = 1_000_000_000
    for fwd in range(5):
        rows.append((marker, 0, t, t + 500_000))
        # a kernel whose name carries a comma, on the same stream, and one on an overlapping stream
        rows.append(("mmq,kernel<8>", 0, t + 500_000, t + 1_500_000))
        rows.append(("attn", 1, t + 1_200_000, t + 2_000_000))
        t += 10_000_000
    p = tmp_path / "trace.csv"
    write_trace(p, rows)

    r = db.parse_trace(p, prompt_forwards=2, generation_forwards=3)
    # the decode window is the last three forwards: 3 x (marker + two overlapping kernels)
    assert r["intervals"] == 3
    assert r["markers"] == 5
    # one forward's kernel union: marker 0.5 ms then the two overlapping 1 ms kernels -> 0..2.0 ms
    assert r["kernel_ms"] == pytest.approx(2.0 * 3, abs=1e-6)
    assert r["window_ms"] == pytest.approx(22.0, abs=1e-6)
    assert r["names"]["mmq,kernel<8>"] == pytest.approx(3 * 1.0)
    assert r["names"]["attn"] == pytest.approx(3 * 0.8)


def test_trace_refuses_when_phase_boundaries_do_not_match(tmp_path):
    marker = "k_row<12>"
    p = tmp_path / "trace.csv"
    write_trace(p, [(marker, 0, i * 1_000_000, i * 1_000_000 + 100_000) for i in range(6)])
    with pytest.raises(db.BenchError):
        # 6 forwards in the file but the caller claims 2 prompt + 3 generation
        db.parse_trace(p, prompt_forwards=2, generation_forwards=3)


def test_trace_reports_gaps_without_naming_a_cause(tmp_path):
    marker = "k_row<12>"
    rows = [
        (marker, 0, 0, 1_000_000),  # 1 ms
        ("k2", 0, 20_000_000, 21_000_000),  # then a 19 ms gap
    ]
    p = tmp_path / "trace.csv"
    write_trace(p, rows)
    r = db.parse_trace(p, prompt_forwards=0, generation_forwards=1, gap_us=1_000)
    assert r["gaps_over"] == 1
    assert r["gap_ms"] == pytest.approx(19.0, abs=1e-6)
    assert r["kernel_ms"] == pytest.approx(2.0)
    # a gap is reported as a gap: no field claims to know what filled it
    assert not [k for k in r if "driver" in k or "host" in k]


def test_trace_ignores_rows_outside_the_decode_window(tmp_path):
    marker = "k_row<12>"
    rows = [
        ("early", 0, 0, 50_000_000),  # a huge prompt-phase kernel
        (marker, 0, 60_000_000, 60_100_000),  # the single prompt forward's marker
        (marker, 0, 100_000_000, 100_100_000),  # the generation forward's marker
        ("dec", 0, 100_200_000, 100_400_000),
    ]
    p = tmp_path / "trace.csv"
    write_trace(p, rows)
    r = db.parse_trace(p, prompt_forwards=1, generation_forwards=1)
    assert set(r["names"]) == {marker, "dec"}
    assert r["names"]["dec"] == pytest.approx(0.2)
    assert r["kernel_ms"] == pytest.approx(0.3)


# --------------------------------------------------------------------------------------------------------------
# Pair planning
# --------------------------------------------------------------------------------------------------------------
def test_plan_pairs_alternates_order():
    assert db.plan_pairs(3) == [
        ("baseline", 0),
        ("candidate", 0),
        ("candidate", 1),
        ("baseline", 1),
        ("baseline", 2),
        ("candidate", 2),
    ]


def test_plan_pairs_rejects_bad_counts():
    for n in (0, -1):
        with pytest.raises(db.BenchError):
            db.plan_pairs(n)


def test_summarize_reports_median_per_arm_and_reduction():
    base = [{"decode_ms": 8000.0}, {"decode_ms": 8200.0}, {"decode_ms": 8100.0}]
    cand = [{"decode_ms": 7600.0}, {"decode_ms": 7700.0}, {"decode_ms": 7650.0}]
    s = db.summarize(base, cand)
    assert s["baseline_median_ms"] == pytest.approx(8100.0)
    assert s["candidate_median_ms"] == pytest.approx(7650.0)
    assert s["reduction"] == pytest.approx((8100.0 - 7650.0) / 8100.0)
    assert s["pairs"] == 3


# --------------------------------------------------------------------------------------------------------------
# Harness behaviour against fake children
# --------------------------------------------------------------------------------------------------------------
def fake_child(tmp_path: Path, body: str, name: str = "fake_engine.py") -> Path:
    p = tmp_path / name
    p.write_text("import sys, time, signal\n" + textwrap.dedent(body))
    return p


def engine_cmd(script: Path) -> list:
    return [sys.executable, "-u", str(script)]


GOOD_CHILD = """
    def w(s):
        sys.stdout.write(s + "\\n"); sys.stdout.flush()
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    w("INFO engine=kolibri-1 cpu_threads=8 prefill=1 slots=0 vram_free_mib=100")
    w("banner: MARKER-ACTIVE")
    w("READY 8192 stop")
    for line in sys.stdin:
        if not line.startswith("GEN"):
            continue
        n = int(line.split()[1])
        for i in range(n):
            time.sleep(0.001)
            w("T %d" % i)
        w("DONE %d 512 100.0 %.1f length 0 0 0 0 0 0 0 0.0 512 0" % (n, n * 1.0))
        break
"""


def test_run_request_collects_tokens_and_done(tmp_path):
    script = fake_child(tmp_path, GOOD_CHILD)
    r = db.run_request(
        engine_cmd(script),
        tokens="1,2,3",
        n=8,
        timeout_s=30,
        require_marker="MARKER-ACTIVE",
    )
    assert r.done["produced"] == 8
    assert r.done["finish"] == "length"
    assert len(r.token_times_ns) == 8
    assert r.marker_seen
    # time to the first token is measured from the GEN write, in the same clock as the T timestamps
    assert 0.0 <= r.gen_to_first_t_ms < r.wall_s * 1e3
    assert r.load_before is not None


def test_run_request_fails_when_the_candidate_marker_never_appears(tmp_path):
    script = fake_child(tmp_path, GOOD_CHILD.replace("MARKER-ACTIVE", "SOMETHING-ELSE"))
    with pytest.raises(db.BenchError):
        db.run_request(engine_cmd(script), tokens="1,2,3", n=8, timeout_s=30, require_marker="MARKER-ACTIVE")


def test_run_request_fails_on_a_nonzero_exit(tmp_path):
    script = fake_child(
        tmp_path,
        """
        sys.stdout.write("READY 8192 stop\\n"); sys.stdout.flush()
        sys.stdin.readline()
        sys.stderr.write("boom\\n")
        sys.exit(3)
        """,
    )
    with pytest.raises(db.BenchError) as e:
        db.run_request(engine_cmd(script), tokens="1", n=4, timeout_s=30)
    assert "3" in str(e.value)


def test_run_request_fails_on_unexpected_eof_without_done(tmp_path):
    script = fake_child(
        tmp_path,
        """
        sys.stdout.write("READY 8192 stop\\n"); sys.stdout.flush()
        sys.stdin.readline()
        sys.stdout.write("T 5\\n"); sys.stdout.flush()
        """,
    )
    with pytest.raises(db.BenchError):
        db.run_request(engine_cmd(script), tokens="1", n=4, timeout_s=30)


def test_run_request_fails_on_an_early_eos_request(tmp_path):
    # a "stop" finish is not a fixed-length decode: it must not be scored as one
    script = fake_child(
        tmp_path,
        """
        def w(s):
            sys.stdout.write(s + "\\n"); sys.stdout.flush()
        w("READY 8192 stop")
        for line in sys.stdin:
            if line.startswith("GEN"):
                w("T 5")
                w("DONE 1 512 10.0 5.0 stop 0 0 0 0 0 0 0 0.0 512 0")
                break
        """,
    )
    with pytest.raises(db.BenchError) as e:
        db.run_request(engine_cmd(script), tokens="1", n=8, timeout_s=30)
    assert "stop" in str(e.value)


def test_run_request_times_out_and_kills_only_its_own_group(tmp_path):
    # a bystander process that is NOT in the engine's group must survive the timeout
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    try:
        script = fake_child(
            tmp_path,
            """
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            sys.stdout.write("READY 8192 stop\\n"); sys.stdout.flush()
            time.sleep(600)
            """,
        )
        t0 = time.monotonic()
        with pytest.raises(db.BenchError) as e:
            db.run_request(engine_cmd(script), tokens="1", n=4, timeout_s=2)
        elapsed = time.monotonic() - t0
        assert "timed out" in str(e.value)
        assert elapsed < 20
        assert bystander.poll() is None  # untouched
        # the harness must not leave its own child running: the recorded pid is gone
        pid = int(re.search(r"pid=(\d+)", str(e.value)).group(1))
        for _ in range(100):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            raise AssertionError(f"fake engine pid {pid} survived the timeout")
    finally:
        bystander.kill()
        bystander.wait(timeout=10)


def test_run_request_keeps_logs_on_failure(tmp_path):
    script = fake_child(
        tmp_path,
        """
        sys.stderr.write("load failed: no such tensor\\n"); sys.stderr.flush()
        sys.exit(1)
        """,
    )
    out = tmp_path / "run"
    with pytest.raises(db.BenchError) as e:
        db.run_request(engine_cmd(script), tokens="1", n=4, timeout_s=30, log_dir=out)
    assert "load failed" in str(e.value)
    logs = list(out.glob("*.log"))
    assert logs and any("load failed" in p.read_text() for p in logs)


def test_run_request_fails_on_a_missing_binary(tmp_path):
    with pytest.raises(db.BenchError):
        db.run_request([str(tmp_path / "nope")], tokens="1", n=4, timeout_s=10)


# --------------------------------------------------------------------------------------------------------------
# Paired runs
# --------------------------------------------------------------------------------------------------------------
PAIRED_CHILD = """
    slot = 0.0
    if "CANDIDATE=1" in __import__("os").environ:
        slot = 0.5
    def w(s):
        sys.stdout.write(s + "\\n"); sys.stdout.flush()
    w("READY 8192 stop")
    for line in sys.stdin:
        if line.startswith("GEN"):
            n = int(line.split()[1])
            extra = 0.0
            if __import__("os").environ.get("CANDIDATE") == "1":
                extra = 0.5
            for i in range(n):
                w("T %d" % i)
            w("DONE %d 512 100.0 %.1f length 0 0 0 0 0 0 0 0.0 512 0" % (n, n * 1.0 + extra))
            break
"""


def test_pair_run_unsets_the_candidate_controls_for_the_baseline(tmp_path):
    script = fake_child(tmp_path, PAIRED_CHILD)
    res = db.run_pairs(
        engine_cmd(script),
        tokens="1,2,3",
        n=4,
        pairs=2,
        timeout_s=30,
        candidate_env={"CANDIDATE": "1"},
        order_log=tmp_path / "order.txt",
    )
    assert len(res["baseline"]) == 2 and len(res["candidate"]) == 2
    # the candidate arm really ran: its decode_ms carries the +0.5 ms
    assert res["summary"]["candidate_median_ms"] > res["summary"]["baseline_median_ms"]
    assert (tmp_path / "order.txt").read_text().split() == ["baseline", "candidate", "candidate", "baseline"]


def test_pair_run_rejects_a_candidate_that_is_silently_inactive(tmp_path):
    script = fake_child(tmp_path, PAIRED_CHILD.replace('if "CANDIDATE=1" in __import__("os").environ:', "if False:"))
    with pytest.raises(db.BenchError):
        db.run_pairs(
            engine_cmd(script),
            tokens="1,2,3",
            n=4,
            pairs=1,
            timeout_s=30,
            candidate_env={"CANDIDATE": "1"},
            require_marker="MARKER-ACTIVE",
        )


def test_fixture_length_is_asserted(tmp_path):
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text(",".join(str(i) for i in range(10)))
    with pytest.raises(db.BenchError) as e:
        db.load_ids(ids_file, expect=512)
    assert "512" in str(e.value)
    assert db.load_ids(ids_file, expect=10) == ",".join(str(i) for i in range(10))


def test_context_capacity_is_asserted():
    with pytest.raises(db.BenchError):
        db.check_capacity(prompt_ids=8191, n=128, ctx=8192)
    db.check_capacity(prompt_ids=8191, n=128, ctx=12288)
