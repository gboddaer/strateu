#!/bin/bash
# tools/kolibri/pf_gate.sh - the prefill structural gate, in the order that actually catches breaks.
#
# WHY THIS EXISTS.  Six A2b refactors were landed and one was reverted; the one that was reverted (the PRE/POST split)
# had passed `PF_HASH` byte-identity and the oracle, and was still wrong: the 64-token prefill produced DIFFERENT
# GENERATED IDS and took twice as long.  `PF_HASH` hashes the expert leg's per-chunk output, so it is blind to the
# ORDER the halves run in, or to a region running twice.  The cheap test is the one that sees the schedule:
#
#   1. ids smoke   64-token prompt, 4 generated ids, ~1 s        <- catches duplicated / misordered work
#   2. PF_HASH     512-token prompt, 50 chunk-layer checksums     <- catches expert-leg value drift (needs determinism)
#   3. oracle      the plain path, bitwise vs the recorded ids    <- catches anything that moves the plain path
#   4. reuse       an extending request against a cold read        <- catches a resumed read that reads no tail,
#
# Usage:  bash tools/kolibri/pf_gate.sh            # gate the CURRENT tree against the recorded expectations
#         bash tools/kolibri/pf_gate.sh --record   # record this tree's outputs as the new expectation
# Env passes through, so the same script gates any A2b arm (e.g. STRATA_KOLIBRI_PF_PIPE=1).
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
M=${KOLIBRI_MODEL:-/data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf}
BIN=${KOLIBRI_BIN:-/tmp/kolibri_bench_node1.sh}
SMOKE=/tmp/pf_gate_smoke.txt
HASH=/tmp/pf_gate_hash.txt
EXP_SMOKE=${KOLIBRI_GATE_SMOKE:-/tmp/pf_gate_expect_smoke.txt}
EXP_HASH=${KOLIBRI_GATE_HASH:-/tmp/pf_gate_expect_hash.txt}
MODE=${1:-check}
MODE=${MODE#--}      # accept `--record` and `record` alike

# the shipped prefill config, with ATTN_FLASH=1 because PF_HASH needs a deterministic prefill
export STRATA_KOLIBRI_PF_STAGE=1 STRATA_KOLIBRI_EXPERT_F16=24 STRATA_KOLIBRI_ROUTER_TILE=1 STRATA_KOLIBRI_ROUTER_TK=3
export STRATA_KOLIBRI_ROUTER_TJ=16 STRATA_KOLIBRI_ATTN_MB=256 STRATA_KOLIBRI_RES_TOP=48 STRATA_KOLIBRI_ATTN_FLASH=1
export GGML_CUDA_DISABLE_GRAPHS=1

run_smoke() {   # 64-token prompt, 4 ids: the cheapest test that sees the schedule
    [ -s /tmp/ids64.txt ] || { echo "FATAL: /tmp/ids64.txt missing (64 prompt ids)"; return 2; }
    "$BIN" --model "$M" --tokens-file /tmp/ids64.txt --n 4 --ctx 8192 --prefill --cpu-threads 24 2>/dev/null \
        | grep '^gen:' > "$SMOKE"
    [ -s "$SMOKE" ] || { echo "FATAL: no gen: line (the prefill did not run)"; return 2; }
}
run_hash() {    # 512-token prompt: the per-chunk-layer expert-output checksums
    [ -s tools/kolibri/oracle_ids512.txt ] || { echo "FATAL: oracle_ids512.txt missing"; return 2; }
    env STRATA_KOLIBRI_PF_HASH=1 "$BIN" --model "$M" --tokens-file tools/kolibri/oracle_ids512.txt --n 4 \
        --ctx 8192 --prefill --cpu-threads 24 2>&1 | grep -a PFHASH > "$HASH"
    [ -s "$HASH" ] || { echo "FATAL: no PFHASH lines (the prefill did not run)"; return 2; }
}

# **BUILD FIRST, AND ABORT IF IT FAILS.**  A gate that does not build silently tests the PREVIOUS binary: this script
# reported GATE PASS while the tree did not compile, which is the same class of trap as the --cpu-threads one.
: "${KOLIBRI_CM:?set KOLIBRI_CM to the cmake binary}"
if ! "$KOLIBRI_CM" --build build -j 24 --target strata-kolibri >/tmp/pf_gate_build.txt 2>&1; then
    echo "  0. build       FAIL"; grep -E "error" /tmp/pf_gate_build.txt | head -5 | sed 's/^/     /'
    echo "GATE FAIL"; exit 2
fi
echo "  0. build       PASS"

rc=0
run_smoke || exit 2
if [ "$MODE" = record ]; then cp "$SMOKE" "$EXP_SMOKE"; else
    if cmp -s "$SMOKE" "$EXP_SMOKE"; then echo "  1. ids smoke   PASS  ($(cat "$SMOKE"))"
    else echo "  1. ids smoke   FAIL"; echo "     got:      $(cat "$SMOKE")"; echo "     expected: $(cat "$EXP_SMOKE" 2>/dev/null)"; rc=1; fi
fi

run_hash || exit 2
if [ "$MODE" = record ]; then cp "$HASH" "$EXP_HASH"; else
    if cmp -s "$HASH" "$EXP_HASH"; then echo "  2. PF_HASH     PASS  ($(wc -l < "$HASH") chunk-layers)"
    else echo "  2. PF_HASH     FAIL  (differs from $(wc -l < "$EXP_HASH" 2>/dev/null) recorded lines)"; rc=1; fi
fi

if bash tools/kolibri/oracle.sh check >/tmp/pf_gate_oracle.txt 2>&1; then
    echo "  3. oracle      PASS  ($(tail -1 /tmp/pf_gate_oracle.txt | cut -c1-60)...)"
else
    echo "  3. oracle      FAIL"; tail -3 /tmp/pf_gate_oracle.txt | sed 's/^/     /'; rc=1
fi

# The reuse stage costs about four minutes (a 3,072-id cold read plus the arms around it), which is why it
# is the last one: everything cheaper fails first.  It is skipped when REUSE_GATE=0, for the runs where only
# the cold path matters (an A/B on a prefill knob, say).
if [ "${REUSE_GATE:-1}" = 0 ]; then
    echo "  4. reuse       SKIP  (REUSE_GATE=0)"
elif KOLIBRI_BIN="$BIN" KOLIBRI_MODEL="$M" bash tools/kolibri/reuse_gate.sh check >/tmp/pf_gate_reuse.txt 2>&1; then
    echo "  4. reuse       PASS  ($(grep -c 'PASS' /tmp/pf_gate_reuse.txt) stage(s) green)"
else
    echo "  4. reuse       FAIL"; grep -E "FAIL|GATE" /tmp/pf_gate_reuse.txt | sed 's/^/     /'; rc=1
fi

[ "$MODE" = record ] && { echo "RECORDED ($EXP_SMOKE, $EXP_HASH)"; exit 0; }
[ $rc -eq 0 ] && echo "GATE PASS" || echo "GATE FAIL"
exit $rc
