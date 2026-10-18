#!/usr/bin/env python3
# tools/kolibri/mixed_type_reject_test.py - plan 2026-10-17-kolibri1-speed.md task 8 (S5): a crafted file with
# ffn_gate_exps.type != ffn_up_exps.type must be REJECTED at load, not silently repacked.
#
# Why it matters: the repack writes gate and up into ONE [gate|up] blob under a single layout (b.L.gu_type,
# taken from gate), and native_expert_layout is only told gate/down.  A mixed-type expert file would therefore
# have its up rows parsed with gate's type - silent numeric corruption, not a failure.  The engine rejects it
# (kolibri_engine.cu, the `gate->type != up->type` guard in pass 1).
#
# The file is header-only: the guard runs in pass 1, during the metadata scan, before any tensor bytes are
# read, so the crafted file carries the real KV section verbatim (4.8 MB) and three tensor infos with zero
# offsets and no data section.  That makes the test cheap enough to run on any box that has the model.
#
# Usage:  tools/kolibri/mixed_type_reject_test.py [--model GGUF] [--bin strata-kolibri] [--out crafted.gguf]
# Exit 0 = the engine rejected the file with the type-mismatch message.  Non-zero = the guard did not fire.
import argparse, os, struct, subprocess, sys

SZ = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}   # gguf scalar value types


def rd_u(f, fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


def skip_val(f, t):
    if t in SZ:
        f.seek(SZ[t], 1)
    elif t == 8:                                   # string
        f.seek(rd_u(f, "<Q"), 1)
    elif t == 9:                                   # array (may nest: sliding_window_pattern is array<array>)
        et = rd_u(f, "<I")
        for _ in range(rd_u(f, "<Q")):
            skip_val(f, et)
    else:
        raise SystemExit("unknown gguf value type %d" % t)


def scan_infos(f, n_tensors):
    """Walk the tensor-info section; return (bytes of the section, {name: (type_field_offset, dims, type)})."""
    start = f.tell()
    infos = {}
    for _ in range(n_tensors):
        nlen = rd_u(f, "<Q")
        name = f.read(nlen).decode()
        nd = rd_u(f, "<I")
        dims = [rd_u(f, "<q") for _ in range(nd)]
        ty_off = f.tell()
        ty = rd_u(f, "<I")
        f.seek(8, 1)                                   # offset field
        infos[name] = (ty_off, dims, ty)
    return f.read(start - f.tell()) if False else (start, f.tell()), infos


def craft(model, out):
    """Header + KV + ALL tensor infos verbatim, with layer 0's ffn_up_exps type flipped.  No data section:
    the guard runs in pass 1 during the metadata scan, before any tensor bytes are read.  All infos must be
    present because the engine validates the non-expert tensors (token_embd, output, attention, shexp)
    before it reaches the expert pass - a file with only the three expert tensors dies earlier with
    "missing token_embd.weight" (measured)."""
    with open(model, "rb") as f:
        assert f.read(4) == b"GGUF", "not a GGUF"
        ver = rd_u(f, "<I"); n_tensors = rd_u(f, "<Q"); n_kv = rd_u(f, "<Q")
        for _ in range(n_kv):
            f.seek(rd_u(f, "<Q"), 1)
            skip_val(f, rd_u(f, "<I"))
        kv_end = f.tell()
        (istart, iend), infos = scan_infos(f, n_tensors)
        _ = (istart, iend)
        f.seek(0)
        blob = f.read(iend)                            # magic + counts + KV + all tensor infos
    g = infos["blk.0.ffn_gate_exps.weight"]; u = infos["blk.0.ffn_up_exps.weight"]
    d = infos["blk.0.ffn_down_exps.weight"]
    print("real file: gate type %d, up type %d, down type %d (gate dims %s)" % (g[2], u[2], d[2], g[1]))
    if g[2] != u[2]:
        sys.exit("UNEXPECTED: the real file already has gate != up; the crafted case is not the interesting one")
    bad = 12 if g[2] != 12 else 14
    b = bytearray(blob)
    b[u[0]] = bad                                      # patch ffn_up_exps' type byte in place
    # The reader validates the data-section start against EOF at open ("GGUF: data section starts past EOF",
    # measured), so pad to the alignment boundary.  No tensor bytes follow: the guard fires in pass 1, during
    # the metadata scan, before any tensor data is read.
    align = 32
    pad = (-iend) % align
    with open(out, "wb") as f:
        f.write(b + bytes(pad))
    print("crafted %s (%d bytes, %d tensor infos, no data): gate %d, up %d -> %d, down %d"
          % (out, os.path.getsize(out), len(infos), g[2], u[2], bad, d[2]))
    return bad


def write_str(f, s):
    b = s.encode()
    f.write(struct.pack("<Q", len(b)) + b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf")
    ap.add_argument("--bin", default="build/strata-kolibri")
    ap.add_argument("--out", default="/tmp/kolibri_mixed_type.gguf")
    a = ap.parse_args()
    for p in (a.model, a.bin):
        if not os.path.exists(p):
            sys.exit("FATAL: missing %s" % p)

    craft(a.model, a.out)

    r = subprocess.run([a.bin, "--model", a.out, "--load-only"], capture_output=True, text=True, timeout=900)
    msg = (r.stderr or "") + (r.stdout or "")
    tail = "\n".join("  " + l for l in msg.strip().splitlines()[-6:])
    print("engine exit %d\n%s" % (r.returncode, tail))
    if r.returncode == 0:
        sys.exit("FAIL: the engine ACCEPTED a mixed-type expert file - the guard is not firing")
    if "ffn_up_exps" not in msg or "type" not in msg:
        sys.exit("FAIL: the engine rejected the file, but not by the type guard (expected a message naming "
                 "ffn_gate_exps/ffn_up_exps); it died earlier - check the message above")
    print("PASS: mixed-type expert file rejected at load by the type guard")


if __name__ == "__main__":
    main()
