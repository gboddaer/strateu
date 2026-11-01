"""Harness tests for plan 2026-10-31-kolibri-gpu-streaming Task 1 (no GPU, no model, no engine).

What they protect: a capture that cannot be replayed is refused before a long replay starts; a timed run cannot
inherit another campaign's arm; a failed run is never a timing; every input a timing claim rests on is pinned by
hash; and the engine's cumulative prefill timer is read once instead of being summed.
"""
from __future__ import annotations

import hashlib
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import expert_stream_campaign as campaign


# --- a test-only capture writer, so a good file can be made bad on purpose ------------------------------------

def _record_payload(layer, chunk, top_k, selections, tokens, experts, stride, n_embd=256, n_ff=512,
                    d_type=14, blob_refs=()):
    gu_row = (n_embd // 256) * 144
    d_row = (n_ff // 256) * (210 if d_type == 14 else 144)
    act_bytes = (n_embd // 256) * 292
    h_bytes = (n_ff // 256) * 292
    payload = struct.pack("<iqiiiiQ", layer, chunk, top_k, selections, tokens, experts, stride)
    payload += struct.pack("<iiii", 12, d_type, 15, 15)
    payload += struct.pack("<qq", n_embd, n_ff)
    payload += struct.pack("<QQQQQQQ", gu_row, d_row, n_ff * gu_row, 2 * n_ff * gu_row,
                          2 * n_ff * gu_row + n_embd * d_row, act_bytes, h_bytes)
    payload += b"".join(struct.pack("<i", e) for e in [0] * selections)
    payload += bytes(tokens * act_bytes)
    payload += struct.pack("<I", len(blob_refs))
    for expert, sha in blob_refs:
        payload += struct.pack("<iQ", expert, stride) + sha
    payload += bytes(selections * n_embd * 4)
    return payload


def write_capture(directory, records, manifest=b'{"kind":"test"}', trailer=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out = bytearray(campaign.CAPTURE_MAGIC)
    out += struct.pack("<II", campaign.CAPTURE_VERSION, campaign.CAPTURE_ENDIAN)
    out += struct.pack("<Q", len(manifest)) + manifest + hashlib.sha256(manifest).digest()
    for record in records:
        payload = _record_payload(**record)
        out += struct.pack("<I", len(payload)) + payload + hashlib.sha256(payload).digest()
    if trailer:
        out += campaign.TRAILER_MAGIC
    (directory / "capture.bin").write_bytes(bytes(out))
    return directory


GOOD_RECORDS = [
    dict(layer=0, chunk=0, top_k=2, selections=4, tokens=2, experts=8, stride=254976),
    dict(layer=1, chunk=0, top_k=2, selections=6, tokens=3, experts=8, stride=254976),
]


def test_rejects_truncated_capture(tmp_path):
    good = write_capture(tmp_path / "good", GOOD_RECORDS)
    manifest, records = campaign.read_capture(good)
    assert manifest == '{"kind":"test"}'
    assert len(records) == 2

    # the trailer missing is a truncation, not an empty tail
    no_trailer = write_capture(tmp_path / "no-trailer", GOOD_RECORDS, trailer=False)
    with pytest.raises(campaign.CaptureError, match="trailer"):
        campaign.read_capture(no_trailer)

    # a record cut in half cannot be framed
    cut = write_capture(tmp_path / "cut", GOOD_RECORDS)
    data = (cut / "capture.bin").read_bytes()
    (cut / "capture.bin").write_bytes(data[:len(data) - 200])
    with pytest.raises(campaign.CaptureError):
        campaign.read_capture(cut)

    # a flipped payload byte fails the record's checksum even though the framing is intact
    flipped = write_capture(tmp_path / "flipped", GOOD_RECORDS)
    data = bytearray((flipped / "capture.bin").read_bytes())
    data[200] ^= 0x01
    (flipped / "capture.bin").write_bytes(bytes(data))
    with pytest.raises(campaign.CaptureError):
        campaign.read_capture(flipped)

    # a manifest whose checksum does not match is refused before any record is read
    bad_manifest = write_capture(tmp_path / "bad-manifest", GOOD_RECORDS)
    data = bytearray((bad_manifest / "capture.bin").read_bytes())
    data[25] ^= 0x01
    (bad_manifest / "capture.bin").write_bytes(bytes(data))
    with pytest.raises(campaign.CaptureError, match="manifest"):
        campaign.read_capture(bad_manifest)

    # and a file that is not a capture at all
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "capture.bin").write_bytes(b"not a capture, only a claim")
    with pytest.raises(campaign.CaptureError, match="magic"):
        campaign.read_capture(foreign)


def test_duplicate_captures_must_agree_on_every_output(tmp_path):
    first = write_capture(tmp_path / "a", GOOD_RECORDS)
    same = write_capture(tmp_path / "b", GOOD_RECORDS)
    campaign.compare_capture_outputs(first, same)   # identical work: no exception
    # a single differing output float between two runs of the same history is a refusal, not a tolerance
    other = dict(GOOD_RECORDS[1])
    other["experts"] = 8
    changed = write_capture(tmp_path / "c", [GOOD_RECORDS[0], other], manifest=b'{"kind":"test"}')
    data = bytearray((changed / "capture.bin").read_bytes())
    data[-1 - len(campaign.TRAILER_MAGIC)] ^= 0x01   # last byte of the last record's payload area
    (changed / "capture.bin").write_bytes(bytes(data))
    with pytest.raises(campaign.CaptureError):
        campaign.compare_capture_outputs(first, changed)
    # different record counts are not a comparison
    short = write_capture(tmp_path / "d", GOOD_RECORDS[:1])
    with pytest.raises(campaign.CaptureError, match="record count"):
        campaign.compare_capture_outputs(first, short)


def test_campaign_manifest_cli_pins_and_summarises(tmp_path, capsys):
    model = tmp_path / "kolibri.gguf"
    model.write_bytes(b"model")
    binary = tmp_path / "strata-kolibri"
    binary.write_bytes(b"elf")
    ids = tmp_path / "ids511.txt"
    ids.write_text("1 2 3\n")
    capture = write_capture(tmp_path / "cap", GOOD_RECORDS)
    out = tmp_path / "manifest.json"
    argv = ["--label", "arm-off", "--model", str(model), "--binary", str(binary), "--input", str(ids),
            "--flag", "STRATA_GEMM_DETERMINISTIC=1", "--arg=--ctx", "--arg=2048", "--arg=--prefill",
            "--manifest-capture", str(capture), "--manifest-out", str(out)]
    assert campaign.main(argv) == 0
    import json
    manifest = json.loads(out.read_text())
    assert manifest["label"] == "arm-off"
    assert manifest["argv"] == ["--ctx", "2048", "--prefill"]
    assert manifest["model"]["sha256"] == hashlib.sha256(b"model").hexdigest()
    assert manifest["captures"][0]["records"] == 2
    assert manifest["captures"][0]["selections"] == 10
    assert "gpu" in manifest and "host" in manifest


def test_no_inherited_experimental_flags(monkeypatch):
    monkeypatch.setenv("STRATA_KOLIBRI_EXPERT_F16", "28")
    monkeypatch.setenv("STRATA_KOLIBRI_PF_PARTIAL", "1")
    monkeypatch.setenv("STRATA_KOLIBRI_EXPERT_GPU", "1")
    monkeypatch.setenv("STRATA_KOLIBRI_KV_INT8", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    env = campaign.sanitized_environment()
    # nothing this campaign did not set survives
    assert not [k for k in env if k.startswith("STRATA_") and k not in campaign.DETERMINISTIC]
    assert env["STRATA_GEMM_DETERMINISTIC"] == "1"
    # and the process's non-experimental settings are kept
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    # the two arms are opted into explicitly, never inherited
    armed = campaign.sanitized_environment(overrides={campaign.ARM_FLAG: "1"})
    assert armed[campaign.ARM_FLAG] == "1"
    assert campaign.A1_FLAG not in armed
    # an explicit base (a subprocess environment) is sanitized the same way
    base = {"PATH": "/bin", "STRATA_KOLIBRI_EXPERT_STREAM": "1", "STRATA_KOLIBRI_STREAM_CAPTURE": "/tmp/x"}
    clean = campaign.sanitized_environment(base=base)
    assert clean == {"PATH": "/bin", "STRATA_GEMM_DETERMINISTIC": "1"}


def test_failure_is_not_a_timing():
    ok = SimpleNamespace(returncode=0, stdout="load: 37.3 s\nPFchunk 256/510 3886ms\nPFchunk 510/510 7754ms\ngen: 1 2\n")
    assert campaign.timing_of(ok) == 7754
    # a nonzero exit, a run with no timer and a run that stopped mid-prompt are all refusals
    with pytest.raises(campaign.CampaignError, match="failed run"):
        campaign.timing_of(SimpleNamespace(returncode=1, stdout="PFchunk 510/510 100ms"))
    with pytest.raises(campaign.CampaignError, match="no PFchunk"):
        campaign.timing_of(SimpleNamespace(returncode=0, stdout="load: 37.3 s\n"))
    with pytest.raises(campaign.CampaignError, match="not a complete timing"):
        campaign.timing_of(SimpleNamespace(returncode=0, stdout="PFchunk 256/510 3886ms"))
    assert campaign.timing_of(SimpleNamespace(returncode=0, stdout=b"PFchunk 510/510 8001ms\n")) == 8001


def test_manifest_pins_model_build_and_inputs(tmp_path):
    model = tmp_path / "kolibri.gguf"
    model.write_bytes(b"model bytes")
    binary = tmp_path / "strata-kolibri"
    binary.write_bytes(b"elf")
    ids = tmp_path / "ids512.txt"
    ids.write_text("1 2 3\n")
    manifest = campaign.run_manifest(model, binary, [ids], ["STRATA_KOLIBRI_EXPERT_STREAM=1"], "arm-on",
                                     argv=["--ctx", "2048", "--cpu-threads", "14", "--prefill"])
    assert manifest["argv"] == ["--ctx", "2048", "--cpu-threads", "14", "--prefill"]
    assert manifest["model"]["sha256"] == hashlib.sha256(b"model bytes").hexdigest()
    assert manifest["model"]["bytes"] == len(b"model bytes")
    assert manifest["binary"]["sha256"] == hashlib.sha256(b"elf").hexdigest()
    assert manifest["inputs"][0]["sha256"] == hashlib.sha256(b"1 2 3\n").hexdigest()
    assert manifest["flags"] == ["STRATA_KOLIBRI_EXPERT_STREAM=1"]
    assert manifest["environment"][campaign.ARM_FLAG] == "1"
    assert "gpu" in manifest and "host" in manifest
    # a missing input is a harness error, not a zero-length pin
    with pytest.raises(campaign.CampaignError, match="does not exist"):
        campaign.run_manifest(model, binary, [tmp_path / "missing.txt"], [], "arm-off")
    with pytest.raises(campaign.CampaignError, match="engine binary"):
        campaign.run_manifest(model, tmp_path / "no-binary", [ids], [], "arm-off")


def test_final_pfchunk_is_not_summed():
    text = "PFchunk 256/510 3886ms\nPFchunk 510/510 7754ms\n"
    chunks = campaign.pf_chunks(text)
    assert chunks == [(256, 510, 3886), (510, 510, 7754)]
    final = campaign.pf_final_ms(text)
    assert final == 7754
    assert final != sum(ms for _, _, ms in chunks)
    # three chunks: still the last value, not the total
    three = "PFchunk 170/510 2600ms\nPFchunk 340/510 5100ms\nPFchunk 510/510 7754ms\n"
    assert campaign.pf_final_ms(three) == 7754
    # an incomplete run has no final cumulative value at all
    with pytest.raises(campaign.CampaignError, match="not a complete timing"):
        campaign.pf_final_ms("PFchunk 170/510 2600ms\n")
