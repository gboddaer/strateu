#!/usr/bin/env bash
# tools/kolibri/kill_gpu.sh - kill the gate's engine/bench processes and free the T4.
#
# Why a file: pkill/pgrep -f patterns match the *invoking shell's own command line*, so a one-liner
# like `pkill -f bench.py` from an interactive shell kills that shell (observed twice on 2026-10-18,
# exit 137).  Here the patterns are built at runtime from fragments, and $$ / $PPID are excluded.
set -uo pipefail
me=$$; parent=$PPID
killed=""
for pid in $(pgrep -f 'strata-kolibri' 2>/dev/null; pgrep -f 'bench[.]py' 2>/dev/null;
             pgrep -f 's3_run_gates[.]sh' 2>/dev/null; pgrep -f 's3_kv_gate[.]sh' 2>/dev/null;
             pgrep -f 's0_baseline[.]sh' 2>/dev/null; pgrep -f 'serve_smoke[.]py' 2>/dev/null); do
  [ "$pid" = "$me" ] && continue
  [ "$pid" = "$parent" ] && continue
  cmd=$(ps -o cmd= -p "$pid" 2>/dev/null)
  case "$cmd" in *kill_gpu*) continue;; esac
  kill -9 "$pid" 2>/dev/null && killed="$killed $pid"
done
echo "killed:${killed:- none}"
sleep 4
echo "GPU apps: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')"
