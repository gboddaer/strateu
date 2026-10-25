#!/usr/bin/env bash
# tools/kolibri/reuse_gate.sh - the prefix-reuse gate (see reuse_gate.py for what each stage proves).
#
# Usage: tools/kolibri/reuse_gate.sh check
# Env:   KOLIBRI_BIN (engine or a numactl wrapper), KOLIBRI_MODEL, KOLIBRI_CTX (default 8192),
#        KOLIBRI_CPU_THREADS (default 24, the config the published figures use).
#
# ATTN_FLASH=1 is set here because the bitwise stages compare two different routes through the prefill
# (one cold read of 3,064 tokens against a 3,000-token read plus a 64-token extension); without a
# deterministic prefill the comparison would measure scheduling, not the reuse path.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ "${1:-check}" = "check" ] || { echo "usage: $0 check"; exit 2; }

: "${KOLIBRI_MODEL:?set KOLIBRI_MODEL to the Q4_K_M gguf}"
export ATTN_FLASH=1
exec python3 "$ROOT/tools/kolibri/reuse_gate.py" \
    --bin "${KOLIBRI_BIN:-$ROOT/build/strata-kolibri}" \
    --model "$KOLIBRI_MODEL" \
    --ctx "${KOLIBRI_CTX:-8192}" \
    --cpu-threads "${KOLIBRI_CPU_THREADS:-24}"
