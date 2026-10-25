#pragma once
// Kolibri 1 prompt attention: causal + sliding-window bounds for one query row, in SLICE coordinates.
//
// The chunked-prefill score buffer's columns are the gathered KV slice [first, first+P-1] (the sliding layers
// gather only the ring's last n_vis rows), while the softmax is handed a GLOBAL position pos0.  Query
// q = pos0 + t attends global [max(0, q-swa+1), q], i.e. slice [q-swa+1-first, q-first].  Using the global
// bounds against the slice-relative buffer - what the code did before the K1 fix - makes rows attend
// positions beyond q (future tokens) as soon as first > 0, i.e. for any prompt longer than n_swa.
//
// Lives in a header so tests/kolibri/window_test.cpp can exercise it on the host, including the worked
// case p0=1024, T=512, swa=513 that the gates cannot reach.
#include <cstdint>

// nvcc gives a plain host function implicit __host__ __device__ linkage when a kernel calls it; clang's HIP
// front end does not, and says so ("reference to __host__ function ... in __global__ function").  So the
// annotation is needed on both compilers, not just on CUDA.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_KOLIBRI_HD __host__ __device__
#else
#define STRATA_KOLIBRI_HD
#endif

namespace strata::kolibri {

STRATA_KOLIBRI_HD inline void attn_mask_bounds(int P, int pos0, int t, int64_t swa, int first, int* lo, int* n) {
    const int q = pos0 + t;
    *n = q - first + 1 < P ? q - first + 1 : P;          // causal, slice-relative
    if (*n < 0) *n = 0;
    *lo = swa > 0 ? (q + 1 - (int) swa - first) : 0;     // window start, slice-relative
    if (*lo < 0) *lo = 0;
    if (*lo > *n) *lo = *n;                              // fully masked row
}

}  // namespace strata::kolibri
