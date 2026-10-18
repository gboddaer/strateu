#!/usr/bin/env bash
# tools/kolibri/oracle.sh - the plain-path oracle, PINNED.
#
# The standing rule of the speed plan (docs/superpowers/plans/2026-10-17-kolibri1-speed.md) is "the plain-path
# oracle re-anchored after each deliberate change".  It was not executable: the recorded oracle came from an
# ad-hoc command whose input was a shell variable holding an id string, nothing in the repo reproduces it, and
# `--tokens` (an id LIST) is not `--tokens-file` (a path) - passing the ids file to --tokens parses the path
# string as ids and generates a degenerate repeat, which reports a bitwise "difference" that has nothing to do
# with the code (measured 2026-10-18 while gating S2a).
#
# This pins every input: the model, the id file (committed at tools/kolibri/oracle_ids512.txt, 512 ids), the
# flag set (none - the plain path), --n 16, --ctx 8192, and the env.  Anyone can re-run it and get the same
# comparison.
#
# Usage:
#   tools/kolibri/oracle.sh record [OUTFILE]     # run and store the ids as the new oracle
#   tools/kolibri/oracle.sh check  [OUTFILE]     # run and compare against the stored oracle (exit 1 on diff)
#   KOLIBRI_BIN=... KOLIBRI_MODEL=... KOLIBRI_N=...  overrides
# Default OUTFILE /tmp/kolibri_oracle.txt.  Exit: 0 match (or record), 1 mismatch, 2 no oracle.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${KOLIBRI_BIN:-$ROOT/build/strata-kolibri}"
MODEL="${KOLIBRI_MODEL:-/data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf}"
IDS="$ROOT/tools/kolibri/oracle_ids512.txt"
N="${KOLIBRI_N:-16}"
CTX="${KOLIBRI_CTX:-8192}"
OUT="${2:-/tmp/kolibri_oracle.txt}"
MODE="${1:-check}"

for p in "$BIN" "$MODEL" "$IDS"; do [ -e "$p" ] || { echo "FATAL: missing $p"; exit 2; }; done

# The plain path: no --grouped, so e.graphs = grouped && ... is false and CUDA graphs are off; no slots, no
# pool, no --prefill.  STRATA_DUMP_CTX is the env the S0/S3 runs used; GGML_CUDA_DISABLE_GRAPHS is deliberately
# NOT set here (it belongs to the pool/slots runs, and setting it here would hide a graph-path regression).
ids=$(env STRATA_DUMP_CTX=8192 "$BIN" --model "$MODEL" --tokens-file "$IDS" --n "$N" --ctx "$CTX" 2>/dev/null | grep '^gen:')
[ -n "$ids" ] || { echo "FATAL: the engine produced no 'gen:' line"; exit 2; }

if [ "$MODE" = record ]; then
    mkdir -p "$(dirname "$OUT")"; printf '%s\n' "$ids" > "$OUT"
    echo "ORACLE RECORDED -> $OUT"
    echo "  $ids"
    echo "  bin md5 $(md5sum "$BIN" | cut -c1-8)  ids $(wc -c < "$IDS") B  n=$N ctx=$CTX"
    exit 0
fi
[ -s "$OUT" ] || { echo "FATAL: no oracle at $OUT (run: $0 record)"; exit 2; }
ref=$(cat "$OUT")
if [ "$ids" = "$ref" ]; then
    echo "BITWISE PASS: $ids"
    exit 0
fi
echo "BITWISE DIFF"
echo "  now: $ids"
echo "  ref: $ref"
exit 1
