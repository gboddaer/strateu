#!/usr/bin/env bash
# T13 (plan 2026-10-28-kolibri-r9700-speed.md): STRATA_KOLIBRI_RES_TOP=0 is a legal configuration - "keep the fp16
# expert cache but promote nobody" - and it used to die as `dequant launch: invalid configuration argument`.  The
# cause was in the promotion path, not the dequant: with a zero ceiling pf_res stays empty, so the resident GEMM
# block ran with nblk = 0 and launched zero-block kernels (100 times in a 512-token prefill).  The launch error is
# sticky, so it reported at the next layer's dequant instead of at its source.
#
# This test runs the exact arm twice: RES_TOP=0 must complete with a `gen:` line, and its ids must equal the
# RES_TOP=56 arm's - the guard must skip the launches without changing a number.
#
# Needs the 45 GB GGUF and a card that can hold a 512-token chunk of it, so it skips (77) without the model rather
# than pretending to have run.  0 = pass, 77 = skip, anything else = fail.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${KOLIBRI_BIN:-$ROOT/build/strata-kolibri}"
MODEL="${KOLIBRI_MODEL:-/fast/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf}"
IDS_FILE="$ROOT/tools/kolibri/oracle_ids512.txt"

[ -x "$BIN" ] || { echo "skip: no engine at $BIN"; exit 77; }
[ -e "$MODEL" ] || { echo "skip: no model at $MODEL"; exit 77; }
[ -e "$IDS_FILE" ] || { echo "skip: no id file at $IDS_FILE"; exit 77; }

IDS="$(cut -d, -f1-64 "$IDS_FILE")"

run_arm() {   # $1 = RES_TOP
    HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
    STRATA_KOLIBRI_RES_TOP="$1" STRATA_KOLIBRI_EXPERT_F16=24 \
        timeout 300 "$BIN" --model "$MODEL" --tokens "$IDS" --n 4 --ctx 8192 --cpu-threads 8 --prefill 2>/tmp/rt_zero_$1.err
}

out0="$(run_arm 0)"; rc0=$?
if [ "$rc0" -ne 0 ]; then
    echo "FAIL: RES_TOP=0 exited $rc0 (it must complete, not fail on a zero-block launch)"
    tail -5 /tmp/rt_zero_0.err
    exit 1
fi
gen0="$(printf '%s\n' "$out0" | grep '^gen:')"
[ -n "$gen0" ] || { echo "FAIL: RES_TOP=0 produced no 'gen:' line"; tail -5 /tmp/rt_zero_0.err; exit 1; }

out56="$(run_arm 56)"; rc56=$?
if [ "$rc56" -ne 0 ]; then
    echo "FAIL: the RES_TOP=56 control exited $rc56"; tail -5 /tmp/rt_zero_56.err; exit 1
fi
gen56="$(printf '%s\n' "$out56" | grep '^gen:')"
if [ "$gen0" != "$gen56" ]; then
    echo "FAIL: RES_TOP=0 and RES_TOP=56 disagree - the guard changed a number"
    echo "  0: $gen0"
    echo " 56: $gen56"
    exit 1
fi

echo "PASS: RES_TOP=0 completes and matches RES_TOP=56 bitwise ($gen0)"
exit 0
