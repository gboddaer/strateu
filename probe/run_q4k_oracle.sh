#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
GGML=${GGML:-/data2/develop/strateu-baseline/build/_deps/strata_llamacpp-src/ggml}
LIB=${LIB:-/data2/develop/strateu-publish/build-kolibri-cuda}
OUT=${OUT:-/data4/kolibri-perf0/a1-oracle}
mkdir -p "$OUT"
g++ -O2 -std=c++17 -mavx2 -mfma -mf16c -ffp-contract=off -I include -I "$GGML/include" -I "$GGML/src" probe/q4k_oneblock.cpp -o "$OUT/q4k_oneblock"
"$OUT/q4k_oneblock" | tee "$OUT/host.txt"
/usr/local/cuda/bin/nvcc -O3 -arch=sm_75 -std=c++17 -Xcompiler -mavx2,-mfma -I include -I "$GGML/include" -o "$OUT/q4k_parity" probe/q4k_parity.cu "$LIB/libstrata_kernels_cpu.a" "$LIB/ggml/src/libggml-cpu.a" "$LIB/ggml/src/libggml-base.a" -lpthread -ldl
for nb in 1 2 10 32; do "$OUT/q4k_parity" "$nb" | tee "$OUT/green-nb${nb}.txt"; done
sha256sum "$OUT/q4k_oneblock" "$OUT/q4k_parity" "$LIB/libstrata_kernels_cpu.a" > "$OUT/SHA256SUMS"
