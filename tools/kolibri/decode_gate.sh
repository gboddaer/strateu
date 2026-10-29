#!/usr/bin/env bash
# tools/kolibri/decode_gate.sh - the correctness gate for an opt-in DECODE arm.
#
# The arm this was written for is STRATA_KOLIBRI_DEC_HANDOFF (docs/KOLIBRI_DECODE.md), but the gate is
# arm-agnostic: it runs the SAME forced histories through the plain engine and through the arm and requires
# the full-logit dumps to be BITWISE identical, then requires six identical repeats of the arm.
#
# Why bitwise and not a tolerance or a KL: a scheduling arm runs the same kernels on the same values in the
# same order.  Any difference is a bug, and a tolerance would hide the differences that matter (see
# tools/kolibri/test_check_decode.py: -0.0 and a 1-ulp change both pass any tolerance one would pick).
#
# Why the forced histories: two free-running arms diverge at the first different token, and then compare two
# different token streams.  `--force` pins the history so record i of both files is the same position.
#
# Stages:
#   1. prompt lengths 16 / 511 / 512 / 513 -> covers the 256-token chunk rule at its boundary
#   2. a serve request that EXTENDS a held prefix (the reused-prefix path, one logits dump per request)
#   3. six repeats of the arm, requiring the arm's activation marker in every run
#   4. activation: the marker must NOT appear in any baseline run (an arm that is on by default, or that
#      leaks into the control, is not being measured)
#
# Everything is bounded: each engine run has a wall-clock limit and the script fails loudly instead of waiting.
#
#   HIP_VISIBLE_DEVICES=0 KOLIBRI_BIN=build-mmq/strata-kolibri bash tools/kolibri/decode_gate.sh check
#
# Environment:
#   KOLIBRI_BIN / KOLIBRI_MODEL / KOLIBRI_PYTHON   the engine, the GGUF, the interpreter (with numpy not needed)
#   KOLIBRI_DEC_ARM        default "" = the engine's default decode path.  The arms are ON by default, so the
#                          candidate needs no environment; pass assignments here to test another arm.
#   KOLIBRI_DEC_CONTROL    default the four arms explicitly OFF, so the control is the plain path.  Without this
#                          a default-on arm would be compared with itself and the marker check would fail.
#   KOLIBRI_DEC_MARKER     default "DEC_TOKEN_GRAPH replay active"
#   KOLIBRI_DEC_GATE_OUT   default /tmp/kolibri-decode-gate
#   KOLIBRI_DEC_GATE_REPEATS default 6
#   KOLIBRI_DEC_GATE_TIMEOUT default 900 (seconds per engine run)
set -u

BIN=${KOLIBRI_BIN:-build-mmq/strata-kolibri}
M=${KOLIBRI_MODEL:-/fast/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf}
PY=${KOLIBRI_PYTHON:-python3}
ARM=${KOLIBRI_DEC_ARM-}
CONTROL=${KOLIBRI_DEC_CONTROL-STRATA_KOLIBRI_DEC_HANDOFF=0 STRATA_KOLIBRI_DEC_STREAM=0 STRATA_KOLIBRI_DEC_HANDOFF_GRAPH=0 STRATA_KOLIBRI_DEC_TOKEN_GRAPH=0}
MARKER=${KOLIBRI_DEC_MARKER:-DEC_TOKEN_GRAPH replay active}
OUT=${KOLIBRI_DEC_GATE_OUT:-/tmp/kolibri-decode-gate}
REPEATS=${KOLIBRI_DEC_GATE_REPEATS:-6}
TMO=${KOLIBRI_DEC_GATE_TIMEOUT:-900}
IDS=tools/kolibri/oracle_ids512.txt
FORCE=${KOLIBRI_DEC_FORCE:-286,3515,310,262,14292,1343,1444,46}
# Expected geometry is independent of dump length. Kolibri-1 has 128000 logits per step.
VOCAB=${KOLIBRI_DEC_VOCAB:-128000}
RECORDS=$("$PY" - "$FORCE" <<'EOF'
import sys
forced = [int(x) for x in sys.argv[1].split(',') if x.strip()]
print(len(forced) + 1)
EOF
) || exit 2
COMMON=(--model "$M" --force "$FORCE" --n 8 --prefill --cpu-threads 8)

cmd=${1:-check}
if [ "$cmd" != "check" ]; then
    echo "usage: $0 check" >&2
    exit 2
fi
if [ ! -x "$BIN" ]; then echo "FATAL: no engine at $BIN" >&2; exit 2; fi
if [ ! -s "$IDS" ]; then echo "FATAL: no $IDS" >&2; exit 2; fi
mkdir -p "$OUT" || exit 2
rc=0
run() {   # run <seconds> <stdout file> <stderr file> <env assignments...> -- <args...>
    local t=$1 out=$2 err=$3; shift 3
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift
    ( env "${envs[@]}" timeout -k 5 "$t" "$BIN" "$@" >"$out" 2>"$err" )
}

marker_count() { grep -c -- "$MARKER" "$1" 2>/dev/null || true; }

# ---- 1. bitwise against the plain engine at the chunk boundary ----
for n in 16 511 512 513; do
    "$PY" - "$IDS" "$OUT/ids$n.txt" "$n" <<'EOF' || { echo "FATAL: cannot build ids$n" >&2; exit 2; }
import sys
ids = open(sys.argv[1]).read().strip().split(',')
n = int(sys.argv[3])
if len(ids) * (2 if n > 512 else 1) < n:
    sys.exit(1)
open(sys.argv[2], 'w').write(','.join((ids + ids)[:n]))
EOF
    rm -f "$OUT/base$n.bin" "$OUT/arm$n.bin"
    run "$TMO" "/dev/null" "$OUT/base$n.err" $CONTROL -- "${COMMON[@]}" --tokens-file "$OUT/ids$n.txt" --ctx 2048 \
        --dump-full-logits "$OUT/base$n.bin"
    b=$?
    # $ARM is deliberately unquoted: it holds one or more NAME=VALUE assignments ("A=1 B=1"), and word splitting
    # is what hands them to env as separate arguments.
    run "$TMO" "/dev/null" "$OUT/arm$n.err" $ARM -- "${COMMON[@]}" --tokens-file "$OUT/ids$n.txt" --ctx 2048 \
        --dump-full-logits "$OUT/arm$n.bin"
    a=$?
    if [ $b -ne 0 ] || [ $a -ne 0 ]; then
        echo "  1.$n  FAIL  engine exit $b (plain) / $a (arm); see $OUT/base$n.err $OUT/arm$n.err"
        rc=1
        continue
    fi
    if [ "$(marker_count "$OUT/arm$n.err")" = "0" ]; then
        echo "  1.$n  FAIL  the arm's marker '$MARKER' is missing: the run was not the arm"
        rc=1
        continue
    fi
    if [ "$(marker_count "$OUT/base$n.err")" != "0" ]; then
        echo "  1.$n  FAIL  the plain run also carries the arm's marker: the control is not a control"
        rc=1
        continue
    fi
    vocab=$VOCAB
    if [ "$(stat -c %s "$OUT/arm$n.bin")" != "$(stat -c %s "$OUT/base$n.bin")" ]; then
        echo "  1.$n  FAIL  dump sizes differ ($(stat -c %s "$OUT/base$n.bin") vs $(stat -c %s "$OUT/arm$n.bin"))"
        rc=1
        continue
    fi
    if "$PY" tools/kolibri/check_decode.py --baseline "$OUT/base$n.bin" --candidate "$OUT/arm$n.bin" \
            --vocab "$vocab" --records "$RECORDS" >"$OUT/check$n.txt" 2>&1; then
        echo "  1.$n  PASS  $(cat "$OUT/check$n.txt")"
    else
        echo "  1.$n  FAIL  $(cat "$OUT/check$n.txt")"
        rc=1
    fi
done

# ---- 2. the reused-prefix path: a cold request, then one that extends the held prefix ----
# STRATA_KOLIBRI_LOGITS_DUMP writes one [reuse][fresh][nv]+logits record per request at its first decode step,
# so the two requests' records are directly comparable between arms.
reuse_case() {   # reuse_case <arm-env-or-empty> <out.bin> <err> <log>
    rm -f "$2"
    # The driver owns its engine process group and enforces this deadline internally. An outer timeout
    # could terminate Python before its finally block reaps that separate group.
    ( env ${1:+$1} STRATA_KOLIBRI_LOGITS_DUMP="$2" KOLIBRI_DEC_GATE_TIMEOUT="$TMO" \
        "$PY" tools/kolibri/decode_gate_serve.py "$BIN" "$M" "$IDS" >"$4" 2>"$3" )
}
if [ -f tools/kolibri/decode_gate_serve.py ]; then
    rm -f "$OUT/reuse-base.bin" "$OUT/reuse-arm.bin"
    reuse_case "$CONTROL" "$OUT/reuse-base.bin" "$OUT/reuse-base.err" "$OUT/reuse-base.log"
    b=$?
    reuse_case "$ARM" "$OUT/reuse-arm.bin" "$OUT/reuse-arm.err" "$OUT/reuse-arm.log"
    a=$?
    if [ $b -ne 0 ] || [ $a -ne 0 ]; then
        echo "  2.reuse FAIL  serve exit $b (plain) / $a (arm); see $OUT/reuse-*.err"
        rc=1
    elif [ "$(marker_count "$OUT/reuse-arm.err")" = "0" ] || [ "$(marker_count "$OUT/reuse-base.err")" != "0" ]; then
        echo "  2.reuse FAIL  marker missing from the arm's log, or present in the plain run's"
        rc=1
    else
        if "$PY" tools/kolibri/check_decode.py --serve-records \
                --baseline "$OUT/reuse-base.bin" --candidate "$OUT/reuse-arm.bin" \
                --vocab "$VOCAB" --records 2 >"$OUT/reuse-check.txt" 2>&1; then
            echo "  2.reuse PASS  2 request record(s) (cold 512 ids, then +64) bitwise identical, headers included"
        else
            echo "  2.reuse FAIL  the reused-prefix records differ"
            tail -3 "$OUT/reuse-check.txt"
            rc=1
        fi
    fi
else
    echo "  2.reuse FAIL  tools/kolibri/decode_gate_serve.py is missing"
    rc=1
fi

# ---- 3. six repeats of the arm, with the marker required in every run ----
if env $ARM "$PY" tools/kolibri/check_repeat.py --runs "$REPEATS" --out "$OUT/repeat" \
        --require-marker "$MARKER" -- "$BIN" "${COMMON[@]}" --tokens-file "$IDS" --ctx 2048 \
        >"$OUT/repeat.txt" 2>&1; then
    echo "  3.repeat PASS  $(tail -1 "$OUT/repeat.txt")"
else
    echo "  3.repeat FAIL  $REPEATS run(s) of the arm are not bit-identical or the marker is missing"
    tail -3 "$OUT/repeat.txt" | sed 's/^/     /'
    rc=1
fi

if [ $rc -eq 0 ]; then
    echo "DECODE GATE PASS  arm '$ARM', marker '$MARKER', raw data in $OUT"
else
    echo "DECODE GATE FAIL  raw data in $OUT"
fi
exit $rc
