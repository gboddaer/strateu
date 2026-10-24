#!/usr/bin/env bash
# tools/kolibri/s0_numa_reset.sh - S0's NUMA-placement and reset_state measurements.
#
# Why: the T4 is on NUMA node 0, and node 1 was found nearly exhausted (463 MB free vs node 0's 140 GB)
# while the engine pins 42.9 GiB of expert blobs.  cudaHostAlloc placement therefore decides how much of
# the expert table the GPU reads cross-socket.  This measures tg/pp under four placement policies, and the
# per-request KV clear (reset_state) with two requests in one serve process.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$ROOT/build/strata-kolibri"
MODEL="${KOLIBRI_MODEL:-/data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf}"
IDS512="${KOLIBRI_IDS512:-/tmp/bench_ids512.txt}"
OUT="${1:-/tmp/kolibri_s0}"; mkdir -p "$OUT"
export STRATA_DUMP_CTX=8192 GGML_CUDA_DISABLE_GRAPHS=1
PY=python3; [ -x "$PY" ] || PY=python3
command -v numactl >/dev/null || { echo "FATAL: numactl missing"; exit 1; }

echo "== NUMA placement, plain path, pp512 + tg128 (ctx 8192); numactl wraps the engine via a shim =="
echo; echo "== NUMA placement, with numactl applied to the engine itself =="
run_numa() {
  local label="$1"; shift
  printf -- "--- %-22s " "$label"
  cat > "$OUT/shim.sh" <<EOF
#!/bin/sh
exec numactl $* $BIN --model $MODEL --serve --ctx 8192 --counters
EOF
  chmod +x "$OUT/shim.sh"
  timeout 900 "$PY" "$ROOT/tools/kolibri/bench.py" --bin "$OUT/shim.sh" --model "$MODEL" --ids512 "$IDS512" \
      --ids32k "$IDS512" --case short --extra="" --tg 128 2>>"$OUT/numa.log" \
    | sed 's/^short   : //' | tr -d '\n'
  echo ""
}
run_numa "default" ""
run_numa "interleave=all" "--interleave=all"
run_numa "membind=0 cpunode=0" "--membind=0 --cpunodebind=0"
run_numa "preferred=0" "--preferred=0"

echo; echo "== reset_state: two fresh requests in one serve process (--counters) =="
"$PY" - "$BIN" "$MODEL" "$IDS512" <<'PYEOF'
import subprocess, sys, os
bin_, model, idsf = sys.argv[1:4]
ids = open(idsf).read().strip()
env = dict(os.environ, STRATA_DUMP_CTX="8192", GGML_CUDA_DISABLE_GRAPHS="1")
p = subprocess.Popen([bin_, "--model", model, "--serve", "--ctx", "33024", "--counters"],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
def rd():
    while True:
        l = p.stdout.readline()
        if not l: return None
        l = l.strip()
        if l.startswith("PP ") or l.startswith("T ") or not l: continue
        return l
def ready():
    while True:
        l = rd()
        if l is None: raise SystemExit("serve died before READY")
        if l.startswith("READY"): return
        print(f"  (skipped pre-READY stdout: {l[:60]})")
ready()
for i in range(2):
    p.stdin.write(f"GEN 8 {ids}\n"); p.stdin.flush()
    d = rd()
    print(f"  request {i}: {d.split()[3]} ms prompt, {d.split()[4]} ms decode")
p.stdin.close(); p.wait(timeout=60)
for l in p.stderr.read().splitlines():
    if l.startswith("CNT serve"):
        print("  " + l.split("| reset=")[1].join(["reset=", ""]) + "  (full 33024-ctx KV clear)")
PYEOF
echo; echo "artifacts: $OUT/numa.log"
