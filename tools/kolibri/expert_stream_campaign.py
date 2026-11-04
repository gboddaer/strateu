#!/usr/bin/env python3
"""Campaign harness for plan 2026-10-31-kolibri-gpu-streaming (streamed expert prefill).

Task 1's half: a sanitized run environment, the engine's prefill timer parsed ONCE (its final cumulative value, never
a sum), a manifest that pins every input a timing claim depends on, and an INDEPENDENT reader for the capture format
so a truncated or foreign capture is refused before a long replay starts.

Deliberately not phase2a_run: that harness sets a different experimental arm (KV int8), and reusing it would put an
unrequested override in the environment of every run here.

    python tools/kolibri/expert_stream_campaign.py --help
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------------------------------------------
# Run environment
# ---------------------------------------------------------------------------------------------------------------

#: Every knob this campaign sets itself.  Anything else named STRATA_* in the parent process is an inherited
#: experiment and is removed: a speed run must not silently carry another campaign's arm.
DETERMINISTIC = {"STRATA_GEMM_DETERMINISTIC": "1"}
ARM_FLAG = "STRATA_KOLIBRI_EXPERT_STREAM"
A1_FLAG = "STRATA_KOLIBRI_EXPERT_GPU"
CAPTURE_FLAG = "STRATA_KOLIBRI_STREAM_CAPTURE"


class CampaignError(RuntimeError):
    """A harness input or result that cannot be trusted as evidence."""


def sanitized_environment(base=None, overrides=None):
    """`base` (default os.environ) with every STRATA_* variable dropped, then the deterministic settings and
    `overrides` set explicitly.  The result is what a timed or qualified run gets."""
    env = {k: v for k, v in (os.environ if base is None else base).items() if not k.startswith("STRATA_")}
    env.update(DETERMINISTIC)
    env.update(overrides or {})
    return env


# ---------------------------------------------------------------------------------------------------------------
# The engine's prefill timer
# ---------------------------------------------------------------------------------------------------------------

_PFCHUNK = re.compile(r"^PFchunk (\d+)/(\d+) (\d+)ms\s*$", re.MULTILINE)


def pf_chunks(text):
    """Every `PFchunk done/total Nms` line, in file order.  The value is CUMULATIVE: the time when that chunk
    finished, not the time the chunk took."""
    return [(int(a), int(b), int(ms)) for a, b, ms in _PFCHUNK.findall(text)]


def pf_final_ms(text):
    """The final cumulative prefill time.  NEVER the sum of the lines: chunk k's line already contains chunks
    0..k-1, so summing them multiplies the answer by about (chunks+1)/2."""
    chunks = pf_chunks(text)
    if not chunks:
        raise CampaignError("no PFchunk line in the engine output: this run is not a prefill timing")
    last = chunks[-1]
    if last[0] != last[1]:
        raise CampaignError(f"the prefill stopped at {last[0]}/{last[1]} ids: this run is not a complete timing")
    return last[2]


def timing_of(result, label="run"):
    """The final cumulative prefill ms of a finished process, or a refusal.  A failed run, a killed run and a run
    that never reached the end of the prompt are all "not a timing" - never a slow timing."""
    if result.returncode != 0:
        raise CampaignError(f"{label}: exit status {result.returncode}; a failed run is not a timing")
    return pf_final_ms(result.stdout if isinstance(result.stdout, str) else result.stdout.decode("utf-8", "replace"))


# ---------------------------------------------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------------------------------------------

def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def pinned_file(path, what):
    """A file the manifest must pin by hash.  A missing input is a harness error, not a zero-length entry."""
    p = Path(path)
    if not p.is_file():
        raise CampaignError(f"{what}: {p} does not exist; refusing to pin a manifest without it")
    return {"path": str(p.resolve()), "bytes": p.stat().st_size, "sha256": sha256_file(p)}


def gpu_identity():
    """The card and driver a number was measured on, or an explicit "unknown" - never an assumed model."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return {"nvidia_smi": "unavailable"}
    lines = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    return {"nvidia_smi": lines if out.returncode == 0 else f"exit {out.returncode}"}


def run_manifest(model, binary, inputs, flags, label, argv=()):
    """Pins the model, the engine binary, every input history, the argv, the flags, the card and the host.  Two runs
    whose manifests differ in any pinned field are not comparable evidence."""
    return {
        "label": label,
        "argv": list(argv),
        "model": pinned_file(model, "model"),
        "binary": pinned_file(binary, "engine binary"),
        "inputs": [pinned_file(path, "input history") for path in inputs],
        "flags": list(flags),
        "environment": {k: v for k, v in sorted(sanitized_environment(overrides=flags_env(flags)).items())
                        if k.startswith(("STRATA_", "CUDA_"))},
        "gpu": gpu_identity(),
        "host": {"machine": os.uname().machine, "node": os.uname().nodename, "cpus": os.cpu_count()},
    }


def flags_env(flags):
    """A flag list maps to an environment the engine reads: the campaign passes arms as env pairs, not argv."""
    env = {}
    for flag in flags:
        if "=" in flag:
            key, value = flag.split("=", 1)
            if key.startswith("STRATA_"):
                env[key] = value
    return env


# ---------------------------------------------------------------------------------------------------------------
# Capture v1 (independent of the C++ reader: this is the harness's own check)
# ---------------------------------------------------------------------------------------------------------------

CAPTURE_MAGIC = b"KSTRCAP1"
TRAILER_MAGIC = b"KSTREND1"
CAPTURE_VERSION = 1
CAPTURE_ENDIAN = 0x01020304
#: layer i32, chunk i64, top_k/selections/tokens/experts i32 x4, weight_stride u64, then the format's 4 i32 +
#: 2 i64 + 7 u64.
PAYLOAD_FIXED = 4 + 8 + 4 * 4 + 8 + (4 * 4 + 2 * 8 + 7 * 8)
BLOB_REF_BYTES = 4 + 8 + 32


class CaptureError(CampaignError):
    """A capture that cannot be replayed as evidence: truncated, foreign, or internally inconsistent."""


def _fmt_geometry(payload):
    (layer, chunk, top_k, selections, tokens, experts, stride) = struct.unpack_from("<iqiiiiQ", payload, 0)
    (gu_type, d_type, gu_act, d_act) = struct.unpack_from("<iiii", payload, 36)
    (n_embd, n_ff) = struct.unpack_from("<qq", payload, 52)
    (gu_row, d_row, up_off, down_off, blob_bytes, act_bytes, h_bytes) = struct.unpack_from("<QQQQQQQ", payload, 68)
    return {
        "layer": layer, "chunk": chunk, "top_k": top_k, "selections": selections, "tokens": tokens,
        "experts": experts, "weight_stride": stride, "gu_type": gu_type, "d_type": d_type,
        "gu_act": gu_act, "d_act": d_act, "n_embd": n_embd, "n_ff": n_ff, "gu_row": gu_row, "d_row": d_row,
        "up_off": up_off, "down_off": down_off, "blob_bytes": blob_bytes, "act_bytes": act_bytes,
        "h_bytes": h_bytes,
    }


def expected_payload(geometry, blob_refs):
    """The payload length the record's own geometry implies.  Used to reject a mis-framed record before its
    contents are trusted, in the same spirit as the C++ reader's check."""
    sel, tokens, act_bytes = geometry["selections"], geometry["tokens"], geometry["act_bytes"]
    n_embd = geometry["n_embd"]
    if sel < 0 or tokens < 0 or geometry["top_k"] <= 0 or geometry["experts"] <= 0:
        raise CaptureError("impossible record geometry")
    if sel % geometry["top_k"] != 0 or sel // geometry["top_k"] != tokens:
        raise CaptureError("the record's selections are not tokens x top_k")
    if n_embd <= 0 or act_bytes != (n_embd // 256) * 292:
        raise CaptureError("the record's n_embd/act_bytes are not a supported Q8_K geometry")
    return (PAYLOAD_FIXED + sel * 4 + tokens * act_bytes + 4 + blob_refs * BLOB_REF_BYTES + sel * n_embd * 4)


def _scan(directory):
    """Streams and verifies a capture v1, yielding (geometry, payload) per record.  Streamed rather than slurped:
    the long capture's capture.bin is tens of GiB, and a comparison reads two of them at once."""
    path = Path(directory) / "capture.bin"
    try:
        f = open(path, "rb")
    except OSError as exc:
        raise CaptureError(f"cannot read {path}: {exc}") from exc
    with f:
        header = f.read(24)
        if len(header) < 24 or header[:8] != CAPTURE_MAGIC:
            raise CaptureError(f"{path}: not a v1 capture (bad magic)")
        version, endian = struct.unpack_from("<II", header, 8)
        if version != CAPTURE_VERSION:
            raise CaptureError(f"{path}: capture version {version}, expected {CAPTURE_VERSION}")
        if endian != CAPTURE_ENDIAN:
            raise CaptureError(f"{path}: unsupported byte order 0x{endian:08x}")
        (mlen,) = struct.unpack_from("<Q", header, 16)
        if mlen > (1 << 20):
            raise CaptureError(f"{path}: impossible manifest length {mlen}")
        manifest = f.read(mlen)
        digest = f.read(32)
        if len(manifest) != mlen or len(digest) != 32:
            raise CaptureError(f"{path}: truncated manifest")
        if hashlib.sha256(manifest).digest() != digest:
            raise CaptureError(f"{path}: the manifest checksum does not match")
        index = 0
        while True:
            tag = f.read(8)
            if len(tag) < 8:
                raise CaptureError(f"{path}: missing trailer (truncated capture)")
            if tag == TRAILER_MAGIC:
                if f.read(1):
                    raise CaptureError(f"{path}: bytes after the trailer")
                return
            f.seek(-8, 1)
            len_bytes = f.read(4)
            if len(len_bytes) != 4:
                raise CaptureError(f"{path}: truncated record header")
            (plen,) = struct.unpack("<I", len_bytes)
            payload = f.read(plen)
            recorded = f.read(32)
            if len(payload) != plen or len(recorded) != 32:
                raise CaptureError(f"{path}: record {index} is truncated")
            if hashlib.sha256(payload).digest() != recorded:
                raise CaptureError(f"{path}: record {index} fails its checksum")
            if len(payload) < PAYLOAD_FIXED:
                raise CaptureError(f"{path}: record {index} is shorter than its fixed header")
            geometry = _fmt_geometry(payload)
            refs_at = PAYLOAD_FIXED + geometry["selections"] * 4 + geometry["tokens"] * geometry["act_bytes"]
            (blob_refs,) = struct.unpack_from("<I", payload, refs_at)
            if plen != expected_payload(geometry, blob_refs):
                raise CaptureError(f"{path}: record {index} is not the length its geometry implies")
            geometry["blob_refs"] = blob_refs
            geometry["output_offset"] = refs_at + 4 + blob_refs * BLOB_REF_BYTES
            yield geometry, payload
            index += 1


def read_capture(directory, max_records=None):
    """Frames and verifies a capture v1 (header, manifest checksum, every record's framing and digest, the
    trailer) and returns the manifest text plus the records' geometry."""
    records = []
    for geometry, _payload in _scan(directory):
        records.append(geometry)
        if max_records is not None and len(records) >= max_records:
            break
    return _read_manifest(directory), records


def _read_manifest(directory):
    path = Path(directory) / "capture.bin"
    with open(path, "rb") as f:
        header = f.read(24)
        if len(header) < 24 or header[:8] != CAPTURE_MAGIC:
            raise CaptureError(f"{path}: not a v1 capture (bad magic)")
        (mlen,) = struct.unpack_from("<Q", header, 16)
        return f.read(mlen).decode("utf-8", "replace")


def compare_capture_outputs(first, second):
    """Two captures of the SAME forced history must have identical expert outputs, record by record.  This is the
    determinism check the plan requires before a capture is used as a control: the inputs may be re-derived, but a
    single differing output float means the run (not the replay) is not reproducible, and no timing from it counts.
    Returns the number of records compared."""
    a = _scan(first)
    b = _scan(second)
    n = 0
    while True:
        ra = next(a, None)
        rb = next(b, None)
        if ra is None and rb is None:
            return n
        if ra is None or rb is None:
            raise CaptureError(f"record count differs: {first} and {second} do not describe the same run")
        ga, pa = ra
        gb, pb = rb
        for key in ("layer", "chunk", "selections", "tokens", "n_embd"):
            if ga[key] != gb[key]:
                raise CaptureError(f"record {n}: {key} differs ({ga[key]} != {gb[key]})")
        off_a, off_b = ga["output_offset"], gb["output_offset"]
        if pa[off_a:] != pb[off_b:]:
            raise CaptureError(f"record {n} (layer {ga['layer']}, chunk {ga['chunk']}): the expert outputs differ "
                               f"between two captures of the same history")
        n += 1


def capture_summary(directory):
    """What the harness records about a capture: the manifest, the record count, the selections, the output
    bytes and the blob store's size.  Verified by _scan first, so a bad capture cannot be summarised."""
    records = [geometry for geometry, _payload in _scan(directory)]
    manifest = _read_manifest(directory)
    blobs = Path(directory) / "blobs"
    store = sum(p.stat().st_size for p in blobs.glob("*.bin")) if blobs.is_dir() else 0
    return {
        "dir": str(Path(directory).resolve()),
        "manifest": manifest,
        "records": len(records),
        "selections": sum(r["selections"] for r in records),
        "tokens": sum(r["tokens"] for r in records),
        "blob_refs": sum(r["blob_refs"] for r in records),
        "distinct_blobs": len(list(blobs.glob("*.bin"))) if blobs.is_dir() else 0,
        "blob_store_bytes": store,
        "output_bytes": sum(r["selections"] * r["n_embd"] * 4 for r in records),
        "chunk_geometry": sorted({(r["top_k"], r["selections"], r["tokens"]) for r in records}),
        "group_histogram": sorted({(r["blob_refs"], r["selections"]) for r in records}),
        "layers": sorted({r["layer"] for r in records}),
        "down_types": sorted({r["d_type"] for r in records}),
    }


def campaign_manifest(label, model, binary, inputs, flags, captures=(), argv=()):
    """The run manifest plus the verified summaries of every capture the run produced or replayed."""
    manifest = run_manifest(model, binary, inputs, flags, label, argv)
    manifest["captures"] = [capture_summary(d) for d in captures]
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="streamed expert prefill campaign harness")
    parser.add_argument("--capture", help="verify a capture v1 directory and print its summary")
    parser.add_argument("--capture-json", help="write the capture summary to this file")
    parser.add_argument("--manifest-out", help="write a campaign manifest to this file")
    parser.add_argument("--label", default="run", help="the manifest's label (arm-off, arm-on, replay)")
    parser.add_argument("--model", help="the GGUF the run used")
    parser.add_argument("--binary", help="the engine binary the run used")
    parser.add_argument("--input", action="append", default=[], help="an input history (repeatable)")
    parser.add_argument("--flag", action="append", default=[], help="a KEY=VALUE arm (repeatable)")
    parser.add_argument("--manifest-capture", action="append", default=[], help="a capture to summarise")
    parser.add_argument("--arg", action="append", default=[], help="an engine argv token (repeatable)")
    args = parser.parse_args(argv)
    if args.manifest_out:
        for required in ("model", "binary"):
            if not getattr(args, required):
                parser.error(f"--manifest-out needs --{required}")
        manifest = campaign_manifest(args.label, args.model, args.binary, args.input, args.flag,
                                     args.manifest_capture, args.arg)
        Path(args.manifest_out).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return 0
    if not args.capture:
        parser.print_help()
        return 2
    summary = capture_summary(args.capture)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.capture_json:
        Path(args.capture_json).write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
