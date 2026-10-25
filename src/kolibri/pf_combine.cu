// src/kolibri/pf_combine.cu - the prefill MoE combiner.  See pf_combine.hpp for the contract.
//
// WHY THE PER-TOKEN ROW LOOKUP IS INDEXED AND NOT APPENDED.  The obvious gather is "each thread that finds a
// row for this token atomically appends it to a shared list".  That makes the ORDER of the list - and therefore
// the order of the float additions - depend on thread scheduling.  The campaign's determinism gate is two runs
// producing IDENTICAL ids, so a schedule-dependent summation order is not a style problem, it is a gate
// failure waiting to happen (and case (e) of the unit test catches it).  Indexing the shared slot by the pick
// number instead (`s_row[sel - t*K] = j`) is a plain write with no ordering question: each (t,k) appears at most
// once per chunk-layer, which is the same invariant the PF_HASH probe's `dupslot=0` discriminator watches.  It
// also fixes the summation order at ascending k, matching the reference sum term for term.
//
// WHY THE SCAN IS NOT IN THE COLUMN LOOP.  Scanning `rr_sel` once per column would cost rrows iterations per
// thread per column-chunk: at Tc=512, N=2560, rrows=3072 that is ~4e9 integer compares per chunk-layer, about
// 1 ms of the 4.7 ms `post` leg this change is trying to shorten.  Scanning once per block into shared memory
// costs rrows/blockDim iterations per thread instead - ~15.7M compares per chunk-layer, noise.

#include "pf_combine.hpp"

namespace {

/// One row per pick of this token.  Dynamic shared memory, K ints per block.
__global__ void k_pf_combine(float* moe, const float* sy, const float* p, const float* rr,
                             const int* rr_sel, const float* wts, int rrows, int K, int N) {
    const int t = blockIdx.x;
    const int i = blockIdx.y * blockDim.x + threadIdx.x;

    extern __shared__ int s_row[];  // s_row[k] = the block row holding pick (t,k), or -1
    for (int k = threadIdx.x; k < K; k += (int)blockDim.x) s_row[k] = -1;
    __syncthreads();

    for (int j = threadIdx.x; j < rrows; j += (int)blockDim.x) {
        const int sel = rr_sel[j];
        if (sel >= 0 && sel / K == t) s_row[sel - t * K] = j;
    }
    __syncthreads();

    if (i >= N) return;  // both __syncthreads() above are reached by every thread before this point
    const int base = t * K;
    float acc = sy[(size_t)t * N + i] + p[(size_t)t * N + i];
    for (int k = 0; k < K; ++k) {
        const int j = s_row[k];
        if (j >= 0) acc += wts[base + k] * rr[(size_t)j * N + i];
    }
    moe[(size_t)t * N + i] = acc;
}

}  // namespace

namespace strata {
namespace kolibri {

void pf_combine_rows(float* moe, const float* sy, const float* p, const float* rr, const int* rr_sel,
                     const float* wts, int Tc, int rrows, int K, int N, cudaStream_t st) {
    const int B = 256;
    dim3 grid((unsigned)Tc, (unsigned)((N + B - 1) / B));
    k_pf_combine<<<grid, B, (size_t)K * sizeof(int), st>>>(moe, sy, p, rr, rr_sel, wts, rrows, K, N);
}

}  // namespace kolibri
}  // namespace strata
