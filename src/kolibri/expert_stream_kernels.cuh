// src/kolibri/expert_stream_kernels.cuh - plan 2026-10-31-kolibri-gpu-streaming, Task 2.
//
// The expert products of the streamed prefill, on the GPU, in the pool's arithmetic.
//
// WHAT MUST NOT MOVE.  The CPU pool computes these products with src/kernels/cpu/kq_avx2.cpp (Q4_K) and ggml-cpu's
// vec_dot (Q6_K).  Its arithmetic is a contract, not a detail: eight int32 lanes per row, each accumulating the
// four-value groups at 64j+4k and 64j+32+4k scaled by scales[2j] and scales[2j+1]; one FMA per block of
// d * sumi (and of the minimum term, with dmin = -y.d * xdmin); and a fixed final tree,
// ((a0+a4)+(a2+a6)) + ((a1+a5)+(a3+a7)), plus ((m0+m2)+(m1+m3)) for the minima.  src/kolibri/kq_a1.cuh already holds
// that order, one token at a time.  This file keeps it exactly and adds a token dimension: a row's weight bytes are
// read and decoded once for kTileTokens tokens instead of once per token, and the four-value integer groups use
// __dp4a.  The integer part is exact in int32, so grouping four of those products into one instruction cannot
// change a bit; only the per-block float FMA sequence and the final tree can, and those are untouched per token.
//
// Refusals are explicit: a geometry this path does not support returns non-zero and launches nothing.
#pragma once

#include "kq_a1.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>

namespace strata::kolibri::stream {

/// Output rows per CTA and tokens per row: the tile the plan fixes before any tuning.
constexpr int kTileRows = 16;
constexpr int kTileTokens = 4;

/// STRATA_KOLIBRI_STREAM_ROWS: rows per CTA (8, 16 or 32; the CTA is rows/4 warps of four rows each).  16 is the
/// shipped tile.  Fewer rows cost weight re-reads per token and buy CTAs: a 1,024-row gate/up launch at eight tokens
/// per group is 64 CTAs at 16 rows, against 40 SMs that hold seven CTAs each, so the launch is CTA-starved and eight
/// rows is the obvious thing to try.  More rows do the opposite.  No arithmetic changes, so every setting writes the
/// same bits.
/// STRATA_KOLIBRI_STREAM_DEV_ACT: compute the SwiGLU and the Q8_K quantization of the hidden row on the device
/// instead of on the host, which removes the gate/up D2H, the host stall behind it and the hidden-row H2D from every
/// batch.  The device reproduces this host's own expf bit for bit (tools/kolibri/expf_bit_probe.cu: 4,294,876,429 of
/// 4,294,967,296 float patterns identical, the rest in the denormal tail and NaN payloads), so the bytes are the
/// pool's.  0 or unset = the host bridge.
inline bool es_device_activation() {
    const char* v = std::getenv("STRATA_KOLIBRI_STREAM_DEV_ACT");
    return v != nullptr && v[0] == '1';
}

inline int es_row_tile() {
    const char* v = std::getenv("STRATA_KOLIBRI_STREAM_ROWS");
    const int r = v != nullptr ? std::atoi(v) : 0;
    return (r == 8 || r == 32) ? r : 16;
}

/// STRATA_KOLIBRI_STREAM_SMEM: stage the tile's Q8_K activation blocks in shared memory once per CTA per block.
///
/// The sixteen rows of a CTA read the SAME 292 activation bytes for a given block and token, four tokens at a time,
/// against 128 bytes of weight codes: without staging those bytes cross L1 sixteen times per block.  With staging
/// they are copied once, 73 ints per token over the CTA's 128 threads, and read from shared memory by every row.
/// Only the address a load reads from changes - the values, the per-token accumulation order and the final tree are
/// the kernel's own - so the two modes are bit-identical by construction and the kernel test compares both against
/// the pool.  0 or unset = read the activations from global memory (the previous behaviour).
inline bool es_smem_staging() {
    const char* v = std::getenv("STRATA_KOLIBRI_STREAM_SMEM");
    return v != nullptr && v[0] == '1';
}

/// STRATA_KOLIBRI_STREAM_GU: run the gate/up product through es_q4k_gu_rows - one thread holds the gate row and the
/// up row of the same k, so a token's activation bytes are loaded once for four dp4a instead of twice for two - rather
/// than es_q4k_rows over the whole [gate | up] table.  On by default: measured 1.033x on a 511-token prompt and
/// 1.017x on an 8,190-token one (docs/experiments/kolibri-stream-kernels-layout.md).  0 = the whole-table launch.
/// Both write the same bits: the kernel test runs the two entries on one blob and compares them.
inline bool es_gu_pairing() {
    const char* v = std::getenv("STRATA_KOLIBRI_STREAM_GU");
    return v == nullptr || v[0] != '0';
}

/// STRATA_KOLIBRI_STREAM_PLANES: hold each Q6_K down blob a second time in the plane-split layout (q6k_plane_tile)
/// and compute the down product from there.  1 = planes, 0 or unset = the block-interleaved blob as it arrives.
/// The copy is byte-neutral but costs its size in device memory per slot; both launch the same arithmetic.
inline bool es_plane_layout() {
    const char* v = std::getenv("STRATA_KOLIBRI_STREAM_PLANES");
    return v != nullptr && v[0] == '1';
}


const char* es_unsupported(int ggml_weight_type, int nb, int rows, const void* w, const void* acts);

/// out[token * out_stride + row] = w_row . acts[token], for rows [0, rows) and tokens [0, nt).
/// `w` points at `rows` rows of `nb` blocks.  `acts` points at a CONTIGUOUS device buffer of `nt` quantized
/// activation rows at `act_stride` bytes each, gathered in the caller's selection order.

/// Returns 0 on success, non-zero when refused (nothing was launched and nothing was written).
int es_q4k_rows(const uint8_t* w, int nb, int rows, const uint8_t* const* acts, int nt, float* out, int out_stride,
                cudaStream_t st);
int es_q6k_rows(const uint8_t* w, int nb, int rows, const uint8_t* const* acts, int nt, float* out, int out_stride,
                cudaStream_t st);

}  // namespace strata::kolibri::stream

namespace strata::kolibri::stream {
namespace detail {

using a1::kQ4KBlock;
using a1::kQ6KBlock;
using a1::kQ8KBlock;
using a1::kQK;
using a1::tree8;

/// Four bytes at an arbitrary alignment (the block layouts put them at 4k inside 32- and 64-byte fields).
__device__ __forceinline__ int ld4(const void* p) {
    int v;
    __builtin_memcpy(&v, p, 4);
    return v;
}
__device__ __forceinline__ float h2f_at(const uint8_t* p) {
    uint16_t h;
    __builtin_memcpy(&h, p, 2);
    return __half2float(*(__half*) &h);
}

/// The eight 6-bit scales and the eight 6-bit minima of a Q4_K block, exactly as ggml's AVX2 kernel unmasks them.
__device__ __forceinline__ void q4k_scales_mins(const uint8_t* xb, int* s, int* mn) {
    int j;
    for (j = 0; j < 4; ++j) {
        s[j] = xb[4 + j] & 63;
        mn[j] = xb[8 + j] & 63;
    }
    for (j = 4; j < 8; ++j) {
        s[j] = (xb[8 + j] & 0x0F) | ((xb[j] >> 6) << 4);
        mn[j] = (xb[8 + j] >> 4) | ((xb[4 + j] >> 6) << 4);
    }
}

// ---- the host's expf and its Q8_K quantizer, so the SwiGLU can be computed here without moving a bit --------------
//
// This is glibc 2.36's expf as this host runs it: the -mfma variant, whose disassembly keeps two fusions the C source
// does not show (kd = fma(InvLn2N, xd, SHIFT) and r = fma(InvLn2N, xd, -k)).  tools/kolibri/expf_bit_probe.cu
// compares it against the host's expf over every one of the 2^32 float patterns: 4,294,876,429 identical, and the
// 90,853 that differ are all where exp underflows to a denormal float (|x| >= 103.2789) - a gate/up product that
// large means the layer has already failed - plus the NaN payloads, which the two ISAs canonicalise differently.
static __device__ __constant__ unsigned long long kExpfTab[32] = {
    0x3ff0000000000000ull, 0x3fefd9b0d3158574ull, 0x3fefb5586cf9890full, 0x3fef9301d0125b51ull,
    0x3fef72b83c7d517bull, 0x3fef54873168b9aaull, 0x3fef387a6e756238ull, 0x3fef1e9df51fdee1ull,
    0x3fef06fe0a31b715ull, 0x3feef1a7373aa9cbull, 0x3feedea64c123422ull, 0x3feece086061892dull,
    0x3feebfdad5362a27ull, 0x3feeb42b569d4f82ull, 0x3feeab07dd485429ull, 0x3feea47eb03a5585ull,
    0x3feea09e667f3bcdull, 0x3fee9f75e8ec5f74ull, 0x3feea11473eb0187ull, 0x3feea589994cce13ull,
    0x3feeace5422aa0dbull, 0x3feeb737b0cdc5e5ull, 0x3feec49182a3f090ull, 0x3feed503b23e255dull,
    0x3feee89f995ad3adull, 0x3feeff76f2fb5e47ull, 0x3fef199bdd85529cull, 0x3fef3720dcef9069ull,
    0x3fef5818dcfba487ull, 0x3fef7c97337b9b5full, 0x3fefa4afa2a490daull, 0x3fefd0765b6e4540ull};
static __device__ __constant__ double kExpfShift = 0x1.8p+52;
static __device__ __constant__ double kExpfInvLn2N = 0x1.71547652b82fep+0 * 32.0;
static __device__ __constant__ double kExpfC0 = 0x1.c6af84b912394p-5 / 32.0 / 32.0 / 32.0;
static __device__ __constant__ double kExpfC1 = 0x1.ebfce50fac4f3p-3 / 32.0 / 32.0;
static __device__ __constant__ double kExpfC2 = 0x1.62e42ff0c52d6p-1 / 32.0;

__device__ __forceinline__ float expf_glibc(float x) {
    const uint32_t ux = __float_as_uint(x);
    const uint32_t abstop = (ux >> 20) & 0x7ff;
    if (abstop > 0x42a) {                      // > top12(88.0f), as the binary compares it
        if (ux == 0xff800000u) return 0.0f;    // -inf
        if (abstop > 0x7f7) return x + x;      // |x| >= inf, or nan
        if (x > 0x1.62e42ep6f) return __int_as_float(0x7f800000);   // log(0x1p128): overflow
        if (x < -0x1.9d1d9ep6f) return 0.0f;                        // log(0x1p-149): underflow
    }
    const double xd = (double) x;
    const double kd = __fma_rn(kExpfInvLn2N, xd, kExpfShift);
    const unsigned long long ki = __double_as_longlong(kd);
    const double k = __dadd_rn(kd, -kExpfShift);
    const double r = __fma_rn(kExpfInvLn2N, xd, -k);
    unsigned long long t = kExpfTab[ki & 31];
    t += ki << (52 - 5);
    const double s = __longlong_as_double(t);
    const double r2 = __dmul_rn(r, r);
    const double z1 = __fma_rn(kExpfC0, r, kExpfC1);
    const double y1 = __fma_rn(kExpfC2, r, 1.0);
    const double y2 = __fma_rn(z1, r2, y1);
    return (float) __dmul_rn(y2, s);
}

/// The pool's SwiGLU expression, in its own precision and order: (g / (1 + expf(-g))) * u, three rounded float
/// operations and no fusion (src/kolibri/expert_stream.cu and native_gu_rows write it the same way).
__device__ __forceinline__ float swiglu_glibc(float g, float u) {
    return __fmul_rn(__fdiv_rn(g, __fadd_rn(1.f, expf_glibc(-g))), u);
}

/// One Q8_K block of 256 values from the pool's quantizer (src/kernels/cpu/iq_avx2.cpp q8k_quant_avx2), operation
/// for operation: the block max, the FIRST value of that magnitude as mx, iscale = -127/mx, the magic-add rounding,
/// the low byte sign-extended, the sixteen 16-value sums, and d = 1/iscale.  A block of zeros writes d = 0 and zero
/// codes and leaves the sums alone, which is what the reference does.
__device__ __forceinline__ void q8k_block_glibc(const float* __restrict__ x, uint8_t* __restrict__ out) {
    float m = 0.f;
    for (int j = 0; j < kQK; ++j) m = fmaxf(m, fabsf(x[j]));
    float* d = (float*) out;
    int8_t* qs = (int8_t*) (out + 4);
    int16_t* bs = (int16_t*) (out + 4 + kQK);
    if (m == 0.f) {
        *d = 0.f;
        for (int j = 0; j < kQK; ++j) qs[j] = 0;
        return;   // the sums are left as they were, as the reference leaves them
    }
    // the first value whose magnitude is the block's max, in ascending order - the reference takes that one
    int first = -1;
    for (int j = 0; j < kQK; ++j)
        if (fabsf(x[j]) == m) {
            first = j;
            break;
        }
    const float mx = x[first];
    const float iscale = __fdiv_rn(-127.f, mx);
    for (int j = 0; j < kQK; ++j) {
        const float p = __fmul_rn(iscale, x[j]);
        const float t = __fadd_rn(p, 12582912.f);
        int v = (__float_as_int(t) & 0x007fffff) - 0x00400000;
        v = v < 127 ? v : 127;
        qs[j] = (int8_t) (v & 0xff);
    }
    for (int g = 0; g < 16; ++g) {
        int s = 0;
        for (int j = 0; j < 16; ++j) s += (int) qs[g * 16 + j];
        bs[g] = (int16_t) s;
    }
    *d = __fdiv_rn(1.f, iscale);
}

/// Q4_K x Q8_K, four tokens per row.  Thread = one of the row's eight integer lanes; the four token groups of a
/// warp are the same eight lanes, so every token keeps the CPU's own accumulation sequence in its own registers.
///
/// SMEM replaces the early return with an `active` predicate: staging needs every thread of the CTA at
/// __syncthreads(), so a thread whose row is past `rows` still copies and still synchronises, and only its own
/// weight loads and its store are skipped.  `sact` is one byte (not one tile) where SMEM is off, so the two
/// instantiations use the same registers and occupancy and the A/B measures the staging and nothing else.
template <int ROWS, bool SMEM>
__global__ void q4k_tile(const uint8_t* __restrict__ w, const uint8_t* __restrict__ acts, int act_stride, int nb,
                         int rows, int nt, float* __restrict__ out, int out_stride) {
    __shared__ uint8_t sact[SMEM ? kTileTokens * kQ8KBlock : 1];
    const int warp = (int) (threadIdx.x >> 5);
    const int lane = (int) (threadIdx.x & 31);
    const int row = blockIdx.x * ROWS + warp * 4 + (lane >> 3);
    const int k = lane & (a1::kLanesPerRow - 1);
    const bool active = row < rows;   // a row's eight lanes share the row, so a group leaves or stays together
    const unsigned gmask = 0xffu << (8 * (lane >> 3));
    const int t0 = (int) blockIdx.y * kTileTokens;
    const int ntok_here = (nt - t0) < kTileTokens ? (nt - t0) : kTileTokens;   // >= 1: the grid spans ceil(nt / 4)

    const uint8_t* xrow = w + (size_t) row * (size_t) nb * kQ4KBlock;
    const uint8_t* yb[kTileTokens];
    float acc[kTileTokens], am[kTileTokens];
#pragma unroll
    for (int t = 0; t < kTileTokens; ++t) {
        yb[t] = (t0 + t < nt) ? acts + (size_t) (t0 + t) * (size_t) act_stride : nullptr;
        acc[t] = 0.f;
        am[t] = 0.f;
    }

    for (int i = 0; i < nb; ++i) {
        const uint8_t* yptr[kTileTokens];
#pragma unroll
        for (int t = 0; t < kTileTokens; ++t)
            yptr[t] = yb[t] ? yb[t] + (size_t) i * kQ8KBlock : nullptr;
        if constexpr (SMEM) {
            // one copy per CTA per block: 73 ints per token, one per thread of the 128, so a 4-token tile is one
            // pass.  The tokens past nt are not there and are not copied; the last tile's copy stays in bounds.
            const int nw = ntok_here * (kQ8KBlock / 4);
            for (int wd = (int) threadIdx.x; wd < nw; wd += (int) blockDim.x) {
                const int t = wd / (kQ8KBlock / 4);
                const int o = (wd - t * (kQ8KBlock / 4)) * 4;
                *reinterpret_cast<int*>(sact + t * kQ8KBlock + o) = *reinterpret_cast<const int*>(yptr[t] + o);
            }
            __syncthreads();
#pragma unroll
            for (int t = 0; t < kTileTokens; ++t) yptr[t] = yb[t] ? sact + t * kQ8KBlock : nullptr;
        }
        if (active) {
        const uint8_t* xb = xrow + (size_t) i * kQ4KBlock;
        const float xd = h2f_at(xb), xdmin = h2f_at(xb + 2);
        int s[8], mn[8];
        q4k_scales_mins(xb, s, mn);
        const uint8_t* qs = xb + 16;

        int sumi[kTileTokens], prod[kTileTokens];
#pragma unroll
        for (int t = 0; t < kTileTokens; ++t) {
            sumi[t] = 0;
            prod[t] = 0;
        }

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q = ld4(qs + 32 * j + 4 * k);
            const int lo4 = q & 0x0f0f0f0f;              // nibbles of values 64j + 4k .. +3
            const int hi4 = (q >> 4) & 0x0f0f0f0f;       // nibbles of values 64j + 32 + 4k .. +3
#pragma unroll
            for (int t = 0; t < kTileTokens; ++t) {
                if (!yptr[t]) continue;
                const int8_t* yq = (const int8_t*) (yptr[t] + 4);
                sumi[t] += s[2 * j] * __dp4a(lo4, ld4(yq + 64 * j + 4 * k), 0);
                sumi[t] += s[2 * j + 1] * __dp4a(hi4, ld4(yq + 64 * j + 32 + 4 * k), 0);
            }
        }
        if (k < 4) {
#pragma unroll
            for (int t = 0; t < kTileTokens; ++t) {
                if (!yptr[t]) continue;
                const int16_t* ys = (const int16_t*) (yptr[t] + 4 + a1::kQK);
                prod[t] = mn[2 * k] * (int(ys[4 * k]) + int(ys[4 * k + 1])) +
                          mn[2 * k + 1] * (int(ys[4 * k + 2]) + int(ys[4 * k + 3]));
            }
        }
#pragma unroll
        for (int t = 0; t < kTileTokens; ++t) {
            if (!yptr[t]) continue;
            const float yd = *(const float*) yptr[t];
            acc[t] = __fmaf_rn(xd * yd, (float) sumi[t], acc[t]);
            if (k < 4) am[t] = __fmaf_rn(-(yd * xdmin), (float) prod[t], am[t]);   // the CPU's dmin sign
        }
        }
        if constexpr (SMEM) __syncthreads();   // the next block's copy overwrites sact
    }

#pragma unroll
    for (int t = 0; t < kTileTokens; ++t) {
        if (!yb[t] || !active) continue;
        float a[8], m[4];
#pragma unroll
        for (int j = 0; j < 8; ++j) a[j] = __shfl_sync(gmask, acc[t], j, a1::kLanesPerRow);
#pragma unroll
        for (int j = 0; j < 4; ++j) m[j] = __shfl_sync(gmask, am[t], j, a1::kLanesPerRow);
        if (k == 0) out[(size_t) (t0 + t) * out_stride + row] = __fadd_rn(tree8(a), (m[0] + m[2]) + (m[1] + m[3]));
    }
}

/// Q6_K x Q8_K, four tokens per row: A1's per-lane order, per block, per token.  SMEM as in q4k_tile.
template <int ROWS, bool SMEM>
__global__ void q6k_tile(const uint8_t* __restrict__ w, const uint8_t* __restrict__ acts, int act_stride, int nb,
                         int rows, int nt, float* __restrict__ out, int out_stride) {
    __shared__ uint8_t sact[SMEM ? kTileTokens * kQ8KBlock : 1];
    const int warp = (int) (threadIdx.x >> 5);
    const int lane = (int) (threadIdx.x & 31);
    const int row = blockIdx.x * ROWS + warp * 4 + (lane >> 3);
    const int k = lane & (a1::kLanesPerRow - 1);
    const bool active = row < rows;   // a row's eight lanes share the row, so a group leaves or stays together
    const unsigned gmask = 0xffu << (8 * (lane >> 3));
    const int t0 = (int) blockIdx.y * kTileTokens;
    const int ntok_here = (nt - t0) < kTileTokens ? (nt - t0) : kTileTokens;

    const uint8_t* xrow = w + (size_t) row * (size_t) nb * kQ6KBlock;
    const uint8_t* yb[kTileTokens];
    float acc[kTileTokens];
#pragma unroll
    for (int t = 0; t < kTileTokens; ++t) {
        yb[t] = (t0 + t < nt) ? acts + (size_t) (t0 + t) * (size_t) act_stride : nullptr;
        acc[t] = 0.f;
    }

    for (int i = 0; i < nb; ++i) {
        const uint8_t* yptr[kTileTokens];
#pragma unroll
        for (int t = 0; t < kTileTokens; ++t)
            yptr[t] = yb[t] ? yb[t] + (size_t) i * kQ8KBlock : nullptr;
        if constexpr (SMEM) {
            const int nw = ntok_here * (kQ8KBlock / 4);
            for (int wd = (int) threadIdx.x; wd < nw; wd += (int) blockDim.x) {
                const int t = wd / (kQ8KBlock / 4);
                const int o = (wd - t * (kQ8KBlock / 4)) * 4;
                *reinterpret_cast<int*>(sact + t * kQ8KBlock + o) = *reinterpret_cast<const int*>(yptr[t] + o);
            }
            __syncthreads();
#pragma unroll
            for (int t = 0; t < kTileTokens; ++t) yptr[t] = yb[t] ? sact + t * kQ8KBlock : nullptr;
        }
        if (active) {
        const uint8_t* xb = xrow + (size_t) i * kQ6KBlock;
        const uint8_t* ql = xb;
        const uint8_t* qh = xb + 128;
        const int8_t* sc = (const int8_t*) (xb + 192);
        const float xd = h2f_at(xb + 208);

        int raw[kTileTokens];
#pragma unroll
        for (int t = 0; t < kTileTokens; ++t) raw[t] = 0;

#pragma unroll
        for (int j = 0; j < 2; ++j) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int s = int(sc[8 * j + 2 * c + (k >> 2)]);
                const int byte_base = 64 * j + (c & 1) * 32;
                const int qlb = ld4(ql + byte_base + 4 * k);
                const int qhb = ld4(qh + 32 * j + 4 * k);
                const int low = (c < 2) ? (qlb & 0x0f0f0f0f) : ((qlb >> 4) & 0x0f0f0f0f);
                const int high = ((qhb >> (2 * c)) & 0x03030303) << 4;
                const int w4 = low | high;
#pragma unroll
                for (int t = 0; t < kTileTokens; ++t) {
                    if (!yptr[t]) continue;
                    const int8_t* yq = (const int8_t*) (yptr[t] + 4);
                    raw[t] += s * __dp4a(w4, ld4(yq + 128 * j + 32 * c + 4 * k), 0);
                }
            }
        }
#pragma unroll
        for (int t = 0; t < kTileTokens; ++t) {
            if (!yptr[t]) continue;
            const uint8_t* yp = yptr[t];
            const float yd = *(const float*) yp;
            const int16_t* ys = (const int16_t*) (yp + 4 + a1::kQK);
            // the offset removal, distributed exactly as ggml's AVX2 kernel distributes it
            const int corr = 32 * (int(ys[2 * k]) * int(sc[2 * k]) + int(ys[2 * k + 1]) * int(sc[2 * k + 1]));
            acc[t] = __fmaf_rn(xd * yd, float(raw[t] - corr), acc[t]);
        }
        }
        if constexpr (SMEM) __syncthreads();
    }

#pragma unroll
    for (int t = 0; t < kTileTokens; ++t) {
        if (!yb[t] || !active) continue;
        float a[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) a[j] = __shfl_sync(gmask, acc[t], j, a1::kLanesPerRow);
        if (k == 0) out[(size_t) (t0 + t) * out_stride + row] = tree8(a);
    }
}

/// Q4_K gate/up pair: one thread holds the gate row and the up row of the SAME k, so the activation bytes that both
/// rows multiply are loaded once and used for four dp4a instead of two.  The blob keeps its [gate rows | up rows]
/// order - no weight byte moves - the kernel only maps a tile's rows to `kk` and `rows + kk`.  Each row keeps its own
/// accumulation sequence, FMA and tree, so the outputs are bit-identical to q4k_tile's for the same rows.
template <int ROWS, bool SMEM>
__global__ void q4k_gu_tile(const uint8_t* __restrict__ w, const uint8_t* __restrict__ acts, int act_stride, int nb,
                            int rows, int nt, float* __restrict__ out, int out_stride) {
    __shared__ uint8_t sact[SMEM ? kTileTokens * kQ8KBlock : 1];
    const int warp = (int) (threadIdx.x >> 5);
    const int lane = (int) (threadIdx.x & 31);
    const int kk = blockIdx.x * ROWS + warp * 4 + (lane >> 3);
    const int k = lane & (a1::kLanesPerRow - 1);
    const bool active = kk < rows;
    const unsigned gmask = 0xffu << (8 * (lane >> 3));
    const int t0 = (int) blockIdx.y * kTileTokens;
    const int ntok_here = (nt - t0) < kTileTokens ? (nt - t0) : kTileTokens;

    const size_t rstride = (size_t) nb * kQ4KBlock;
    const uint8_t* xg = w + (size_t) kk * rstride;
    const uint8_t* xu = w + (size_t) (rows + kk) * rstride;
    const uint8_t* yb[kTileTokens];
    float accg[kTileTokens], amg[kTileTokens], accu[kTileTokens], amu[kTileTokens];
#pragma unroll
    for (int t = 0; t < kTileTokens; ++t) {
        yb[t] = (t0 + t < nt) ? acts + (size_t) (t0 + t) * (size_t) act_stride : nullptr;
        accg[t] = 0.f;
        amg[t] = 0.f;
        accu[t] = 0.f;
        amu[t] = 0.f;
    }

    for (int i = 0; i < nb; ++i) {
        const uint8_t* yptr[kTileTokens];
#pragma unroll
        for (int t = 0; t < kTileTokens; ++t)
            yptr[t] = yb[t] ? yb[t] + (size_t) i * kQ8KBlock : nullptr;
        if constexpr (SMEM) {
            const int nw = ntok_here * (kQ8KBlock / 4);
            for (int wd = (int) threadIdx.x; wd < nw; wd += (int) blockDim.x) {
                const int t = wd / (kQ8KBlock / 4);
                const int o = (wd - t * (kQ8KBlock / 4)) * 4;
                *reinterpret_cast<int*>(sact + t * kQ8KBlock + o) = *reinterpret_cast<const int*>(yptr[t] + o);
            }
            __syncthreads();
#pragma unroll
            for (int t = 0; t < kTileTokens; ++t) yptr[t] = yb[t] ? sact + t * kQ8KBlock : nullptr;
        }
        if (active) {
        const uint8_t* xbg = xg + (size_t) i * kQ4KBlock;
        const uint8_t* xbu = xu + (size_t) i * kQ4KBlock;
        const float xdg = h2f_at(xbg), xdming = h2f_at(xbg + 2);
        const float xdu = h2f_at(xbu), xdminu = h2f_at(xbu + 2);
        int sg[8], mng[8], su[8], mnu[8];
        q4k_scales_mins(xbg, sg, mng);
        q4k_scales_mins(xbu, su, mnu);
        const uint8_t* qsg = xbg + 16;
        const uint8_t* qsu = xbu + 16;

        int sumig[kTileTokens], prodg[kTileTokens], sumiu[kTileTokens], produ[kTileTokens];
#pragma unroll
        for (int t = 0; t < kTileTokens; ++t) {
            sumig[t] = sumiu[t] = 0;
            prodg[t] = produ[t] = 0;
        }

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int qg = ld4(qsg + 32 * j + 4 * k);
            const int qu = ld4(qsu + 32 * j + 4 * k);
            const int log = qg & 0x0f0f0f0f, hig = (qg >> 4) & 0x0f0f0f0f;
            const int lou = qu & 0x0f0f0f0f, hiu = (qu >> 4) & 0x0f0f0f0f;
#pragma unroll
            for (int t = 0; t < kTileTokens; ++t) {
                if (!yptr[t]) continue;
                const int8_t* yq = (const int8_t*) (yptr[t] + 4);
                const int a1 = ld4(yq + 64 * j + 4 * k);          // one activation load, two rows
                const int a2 = ld4(yq + 64 * j + 32 + 4 * k);
                sumig[t] += sg[2 * j] * __dp4a(log, a1, 0);
                sumig[t] += sg[2 * j + 1] * __dp4a(hig, a2, 0);
                sumiu[t] += su[2 * j] * __dp4a(lou, a1, 0);
                sumiu[t] += su[2 * j + 1] * __dp4a(hiu, a2, 0);
            }
        }
        if (k < 4) {
#pragma unroll
            for (int t = 0; t < kTileTokens; ++t) {
                if (!yptr[t]) continue;
                const int16_t* ys = (const int16_t*) (yptr[t] + 4 + a1::kQK);
                const int s0 = int(ys[4 * k]), s1 = int(ys[4 * k + 1]);
                const int s2 = int(ys[4 * k + 2]), s3 = int(ys[4 * k + 3]);
                prodg[t] = mng[2 * k] * (s0 + s1) + mng[2 * k + 1] * (s2 + s3);
                produ[t] = mnu[2 * k] * (s0 + s1) + mnu[2 * k + 1] * (s2 + s3);
            }
        }
#pragma unroll
        for (int t = 0; t < kTileTokens; ++t) {
            if (!yptr[t]) continue;
            const float yd = *(const float*) yptr[t];
            accg[t] = __fmaf_rn(xdg * yd, (float) sumig[t], accg[t]);
            accu[t] = __fmaf_rn(xdu * yd, (float) sumiu[t], accu[t]);
            if (k < 4) {
                amg[t] = __fmaf_rn(-(yd * xdming), (float) prodg[t], amg[t]);
                amu[t] = __fmaf_rn(-(yd * xdminu), (float) produ[t], amu[t]);
            }
        }
        }
        if constexpr (SMEM) __syncthreads();
    }

#pragma unroll
    for (int t = 0; t < kTileTokens; ++t) {
        if (!yb[t] || !active) continue;
        float ag[8], mg[4], au[8], mu[4];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            ag[j] = __shfl_sync(gmask, accg[t], j, a1::kLanesPerRow);
            au[j] = __shfl_sync(gmask, accu[t], j, a1::kLanesPerRow);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            mg[j] = __shfl_sync(gmask, amg[t], j, a1::kLanesPerRow);
            mu[j] = __shfl_sync(gmask, amu[t], j, a1::kLanesPerRow);
        }
        if (k == 0) {
            const size_t o = (size_t) (t0 + t) * out_stride;
            out[o + kk] = __fadd_rn(tree8(ag), (mg[0] + mg[2]) + (mg[1] + mg[3]));
            out[o + rows + kk] = __fadd_rn(tree8(au), (mu[0] + mu[2]) + (mu[1] + mu[3]));
        }
    }
}

/// Q6_K from the PLANE-SPLIT layout: the same 210 bytes per block - ql 128, qh 64, the 16 scale bytes, the 2-byte d -
/// held as four planes of one expert's rows instead of inside each block.  The block-interleaved form puts block i at
/// 210 * i, so its ql/qh start 2 bytes further out every block and every ld4 in q6k_tile is a 2-byte-aligned load.
/// In the planes, block i's ql is at 128 * i (128-aligned), its qh at 64 * i (64-aligned), its scales at 16 * i and
/// its d at 2 * i.  No byte moves between planes that changes a value and the per-row arithmetic is q6k_tile's own, so
/// the outputs are bit-identical; the kernel test compares this entry against es_q6k_rows on the packed copy.
template <int ROWS, bool SMEM>
__global__ void q6k_plane_tile(const uint8_t* __restrict__ w, const uint8_t* __restrict__ acts, int act_stride,
                               int nb, int rows, int nt, float* __restrict__ out, int out_stride) {
    __shared__ uint8_t sact[SMEM ? kTileTokens * kQ8KBlock : 1];
    const int warp = (int) (threadIdx.x >> 5);
    const int lane = (int) (threadIdx.x & 31);
    const int row = blockIdx.x * ROWS + warp * 4 + (lane >> 3);
    const int k = lane & (a1::kLanesPerRow - 1);
    const bool active = row < rows;
    const unsigned gmask = 0xffu << (8 * (lane >> 3));
    const int t0 = (int) blockIdx.y * kTileTokens;
    const int ntok_here = (nt - t0) < kTileTokens ? (nt - t0) : kTileTokens;

    const long long nb_all = (long long) rows * (long long) nb;   // blocks of this expert's region
    const uint8_t* qh_all = w + (size_t) nb_all * 128;
    const uint8_t* sc_all = qh_all + (size_t) nb_all * 64;
    const uint8_t* d_all = sc_all + (size_t) nb_all * 16;
    const long long rbase = (long long) row * (long long) nb;

    const uint8_t* yb[kTileTokens];
    float acc[kTileTokens];
#pragma unroll
    for (int t = 0; t < kTileTokens; ++t) {
        yb[t] = (t0 + t < nt) ? acts + (size_t) (t0 + t) * (size_t) act_stride : nullptr;
        acc[t] = 0.f;
    }

    for (int i = 0; i < nb; ++i) {
        const uint8_t* yptr[kTileTokens];
#pragma unroll
        for (int t = 0; t < kTileTokens; ++t)
            yptr[t] = yb[t] ? yb[t] + (size_t) i * kQ8KBlock : nullptr;
        if constexpr (SMEM) {
            const int nw = ntok_here * (kQ8KBlock / 4);
            for (int wd = (int) threadIdx.x; wd < nw; wd += (int) blockDim.x) {
                const int t = wd / (kQ8KBlock / 4);
                const int o = (wd - t * (kQ8KBlock / 4)) * 4;
                *reinterpret_cast<int*>(sact + t * kQ8KBlock + o) = *reinterpret_cast<const int*>(yptr[t] + o);
            }
            __syncthreads();
#pragma unroll
            for (int t = 0; t < kTileTokens; ++t) yptr[t] = yb[t] ? sact + t * kQ8KBlock : nullptr;
        }
        if (active) {
        const long long ib = rbase + i;
        const uint8_t* ql = w + (size_t) ib * 128;
        const uint8_t* qh = qh_all + (size_t) ib * 64;
        const int8_t* sc = (const int8_t*) (sc_all + (size_t) ib * 16);
        const float xd = h2f_at(d_all + (size_t) ib * 2);

        int raw[kTileTokens];
#pragma unroll
        for (int t = 0; t < kTileTokens; ++t) raw[t] = 0;

#pragma unroll
        for (int j = 0; j < 2; ++j) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int s = int(sc[8 * j + 2 * c + (k >> 2)]);
                const int byte_base = 64 * j + (c & 1) * 32;
                const int qlb = ld4(ql + byte_base + 4 * k);
                const int qhb = ld4(qh + 32 * j + 4 * k);
                const int low = (c < 2) ? (qlb & 0x0f0f0f0f) : ((qlb >> 4) & 0x0f0f0f0f);
                const int high = ((qhb >> (2 * c)) & 0x03030303) << 4;
                const int w4 = low | high;
#pragma unroll
                for (int t = 0; t < kTileTokens; ++t) {
                    if (!yptr[t]) continue;
                    const int8_t* yq = (const int8_t*) (yptr[t] + 4);
                    raw[t] += s * __dp4a(w4, ld4(yq + 128 * j + 32 * c + 4 * k), 0);
                }
            }
        }
#pragma unroll
        for (int t = 0; t < kTileTokens; ++t) {
            if (!yptr[t]) continue;
            const uint8_t* yp = yptr[t];
            const float yd = *(const float*) yp;
            const int16_t* ys = (const int16_t*) (yp + 4 + a1::kQK);
            const int corr = 32 * (int(ys[2 * k]) * int(sc[2 * k]) + int(ys[2 * k + 1]) * int(sc[2 * k + 1]));
            acc[t] = __fmaf_rn(xd * yd, float(raw[t] - corr), acc[t]);
        }
        }
        if constexpr (SMEM) __syncthreads();
    }

#pragma unroll
    for (int t = 0; t < kTileTokens; ++t) {
        if (!yb[t] || !active) continue;
        float a[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) a[j] = __shfl_sync(gmask, acc[t], j, a1::kLanesPerRow);
        if (k == 0) out[(size_t) (t0 + t) * out_stride + row] = tree8(a);
    }
}

/// The host's SwiGLU + Q8_K quantization for a batch of rows: one CTA per (row, block of 256 values), 256 threads.
/// Every value goes through swiglu_glibc and q8k_block_glibc, i.e. the same float operations in the same order as the
/// host bridge, so the bytes it writes are the bytes the pool would have quantized.
static __global__ void swiglu_quant_kernel(const float* __restrict__ gu, int n_ff, int rows,
                                           uint8_t* __restrict__ hidden) {   // static: this header is in several TUs
    __shared__ float v[kQK];
    __shared__ int8_t q[kQK];
    __shared__ float s_warp[8];
    __shared__ float s_amax;
    __shared__ int s_first;
    __shared__ float s_iscale;
    __shared__ int s_bsum[16];
    const int b = (int) blockIdx.x, row = (int) blockIdx.y, j = (int) threadIdx.x;
    const int nb = n_ff / kQK;
    const float* grow = gu + (size_t) row * 2 * (size_t) n_ff;
    const int idx = b * kQK + j;
    v[j] = swiglu_glibc(grow[idx], grow[n_ff + idx]);
    if (j < 16) s_bsum[j] = 0;
    __syncthreads();

    float mv = fabsf(v[j]);
    for (int off = 16; off > 0; off >>= 1) mv = fmaxf(mv, __shfl_xor_sync(0xffffffffu, mv, off));
    if ((j & 31) == 0) s_warp[j >> 5] = mv;
    if (j == 0) s_first = 0x7fffffff;
    __syncthreads();
    if (j == 0) {
        float m = s_warp[0];
        for (int w = 1; w < 8; ++w) m = fmaxf(m, s_warp[w]);
        s_amax = m;
        s_iscale = 0.f;
    }
    __syncthreads();
    uint8_t* blk = hidden + (size_t) row * (size_t) nb * kQ8KBlock + (size_t) b * kQ8KBlock;
    if (s_amax == 0.f) {   // the reference writes d = 0 and zero codes, and leaves the sums as they were
        if (j == 0) *(float*) blk = 0.f;
        q[j] = 0;
    } else {
        if (fabsf(v[j]) == s_amax) atomicMin(&s_first, j);   // the first value of that magnitude, as the reference
        __syncthreads();
        if (j == 0) s_iscale = __fdiv_rn(-127.f, v[s_first]);
        __syncthreads();
        const float p = __fmul_rn(s_iscale, v[j]);
        const float t = __fadd_rn(p, 12582912.f);
        int iv = (__float_as_int(t) & 0x007fffff) - 0x00400000;
        iv = iv < 127 ? iv : 127;
        const int8_t c = (int8_t) (iv & 0xff);
        q[j] = c;
        atomicAdd(&s_bsum[j >> 4], (int) c);
        if (j == 0) *(float*) blk = __fdiv_rn(1.f, s_iscale);
    }
    __syncthreads();
    if (j < kQK / 4) {   // four codes at a time, coalesced
        int packed;
        __builtin_memcpy(&packed, &q[4 * j], 4);
        __builtin_memcpy(blk + 4 + 4 * j, &packed, 4);
    }
    if (j < 16 && s_amax != 0.f) {   // a zero block leaves the sums as they were, like the reference
        int16_t sum = (int16_t) s_bsum[j];
        __builtin_memcpy(blk + 4 + kQK + 2 * j, &sum, 2);
    }
}

/// The packing half of the plane layout: one thread per Q6_K block, 105 16-bit words, same bytes, four destinations.
/// Byte-neutral (128 + 64 + 16 + 2 = 210), so the PCIe traffic is unchanged.  `static`: this header is included by
/// more than one translation unit of the library (as kq_a1.cuh is), and a non-static __global__ would be a duplicate.
static __global__ void q6k_pack_planes(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, int nb, int rows) {
    const long long n = (long long) rows * (long long) nb;
    const long long idx = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const uint16_t* s16 = (const uint16_t*) (src + (size_t) idx * 210);
    uint16_t* ql = (uint16_t*) (dst + (size_t) idx * 128);
    uint16_t* qh = (uint16_t*) (dst + (size_t) n * 128 + (size_t) idx * 64);
    uint16_t* sc = (uint16_t*) (dst + (size_t) n * 192 + (size_t) idx * 16);
    uint16_t* dd = (uint16_t*) (dst + (size_t) n * 208 + (size_t) idx * 2);
#pragma unroll
    for (int k = 0; k < 64; ++k) ql[k] = s16[k];
#pragma unroll
    for (int k = 0; k < 32; ++k) qh[k] = s16[64 + k];
#pragma unroll
    for (int k = 0; k < 8; ++k) sc[k] = s16[96 + k];
    dd[0] = s16[104];
}

}  // namespace detail

namespace detail {

/// Token tiles are a GRID dimension, not a host loop: with rows/16 CTAs alone a 1,024-row gate/up launch fills only
/// 1.6 CTAs per SM on this card and measures 230 GFLOP/s (measured, kbench); the token tiles are what gives the card
/// enough work in flight.  The row tile reaches the kernel as a template argument - its CTA is rows/4 warps - so its
/// dispatch lives here once instead of at every launch site.
template <int ROWS>
inline int launch_rows_t(int type, const uint8_t* w, int nb, int rows, const uint8_t* acts, int act_stride, int nt,
                         float* out, int out_stride, cudaStream_t st) {
    const dim3 grid((unsigned) ((rows + ROWS - 1) / ROWS), (unsigned) ((nt + kTileTokens - 1) / kTileTokens));
    const int threads = (ROWS / 4) * 32;
    if (type == 12) {
        if (es_smem_staging()) {
            q4k_tile<ROWS, true><<<grid, threads, 0, st>>>(w, acts, act_stride, nb, rows, nt, out, out_stride);
        } else {
            q4k_tile<ROWS, false><<<grid, threads, 0, st>>>(w, acts, act_stride, nb, rows, nt, out, out_stride);
        }
    } else {
        if (es_smem_staging()) {
            q6k_tile<ROWS, true><<<grid, threads, 0, st>>>(w, acts, act_stride, nb, rows, nt, out, out_stride);
        } else {
            q6k_tile<ROWS, false><<<grid, threads, 0, st>>>(w, acts, act_stride, nb, rows, nt, out, out_stride);
        }
    }
    return 0;
}

inline int launch_rows(int type, const uint8_t* w, int nb, int rows, const uint8_t* acts, int act_stride, int nt,
                       float* out, int out_stride, cudaStream_t st) {
    switch (es_row_tile()) {
        case 8: return launch_rows_t<8>(type, w, nb, rows, acts, act_stride, nt, out, out_stride, st);
        case 32: return launch_rows_t<32>(type, w, nb, rows, acts, act_stride, nt, out, out_stride, st);
        default: return launch_rows_t<kTileRows>(type, w, nb, rows, acts, act_stride, nt, out, out_stride, st);
    }
}

template <int ROWS>
inline int launch_gu_t(const uint8_t* w, int nb, int rows, const uint8_t* acts, int act_stride, int nt, float* out,
                       int out_stride, cudaStream_t st) {
    const dim3 grid((unsigned) ((rows + ROWS - 1) / ROWS), (unsigned) ((nt + kTileTokens - 1) / kTileTokens));
    const int threads = (ROWS / 4) * 32;
    if (es_smem_staging()) {
        q4k_gu_tile<ROWS, true><<<grid, threads, 0, st>>>(w, acts, act_stride, nb, rows, nt, out, out_stride);
    } else {
        q4k_gu_tile<ROWS, false><<<grid, threads, 0, st>>>(w, acts, act_stride, nb, rows, nt, out, out_stride);
    }
    return 0;
}

inline int launch_gu(const uint8_t* w, int nb, int rows, const uint8_t* acts, int act_stride, int nt, float* out,
                     int out_stride, cudaStream_t st) {
    switch (es_row_tile()) {
        case 8: return launch_gu_t<8>(w, nb, rows, acts, act_stride, nt, out, out_stride, st);
        case 32: return launch_gu_t<32>(w, nb, rows, acts, act_stride, nt, out, out_stride, st);
        default: return launch_gu_t<kTileRows>(w, nb, rows, acts, act_stride, nt, out, out_stride, st);
    }
}

template <int ROWS>
inline int launch_plane_t(const uint8_t* w, int nb, int rows, const uint8_t* acts, int act_stride, int nt, float* out,
                          int out_stride, cudaStream_t st) {
    const dim3 grid((unsigned) ((rows + ROWS - 1) / ROWS), (unsigned) ((nt + kTileTokens - 1) / kTileTokens));
    const int threads = (ROWS / 4) * 32;
    if (es_smem_staging()) {
        q6k_plane_tile<ROWS, true><<<grid, threads, 0, st>>>(w, acts, act_stride, nb, rows, nt, out, out_stride);
    } else {
        q6k_plane_tile<ROWS, false><<<grid, threads, 0, st>>>(w, acts, act_stride, nb, rows, nt, out, out_stride);
    }
    return 0;
}

inline int launch_plane(const uint8_t* w, int nb, int rows, const uint8_t* acts, int act_stride, int nt, float* out,
                        int out_stride, cudaStream_t st) {
    switch (es_row_tile()) {
        case 8: return launch_plane_t<8>(w, nb, rows, acts, act_stride, nt, out, out_stride, st);
        case 32: return launch_plane_t<32>(w, nb, rows, acts, act_stride, nt, out, out_stride, st);
        default: return launch_plane_t<kTileRows>(w, nb, rows, acts, act_stride, nt, out, out_stride, st);
    }
}

}  // namespace detail

/// Why this geometry cannot run here, or nullptr when it can.  Checked before anything is allocated or launched.
inline const char* es_unsupported(int ggml_weight_type, int nb, int rows, const void* w, const void* acts) {
    if (w == nullptr) return "null weights";
    if (acts == nullptr) return "null activations";
    if (ggml_weight_type != 12 && ggml_weight_type != 14) return "only Q4_K (12) and Q6_K (14) expert rows";
    if (nb < 1 || nb > 1024) return "super-block count outside 1..1024";
    if (rows < 1 || rows > (1 << 20)) return "row count outside 1..1048576";
    return nullptr;
}

/// out[token * out_stride + row] = w_row . acts[token], for rows [0, rows) and tokens [0, nt).
inline int es_launch(int type, const uint8_t* w, int nb, int rows, const uint8_t* acts, int act_stride, int nt,
                     float* out, int out_stride, cudaStream_t st) {
    if (es_unsupported(type, nb, rows, w, acts) != nullptr) return 1;
    if (nt < 1 || nt > 1024) return 1;
    if (act_stride < (int) (nb * detail::kQ8KBlock)) return 1;
    if (out == nullptr || out_stride < rows) return 1;
    // Token tiles are a GRID dimension, not a host loop: with rows/16 CTAs alone a 1,024-row gate/up launch fills
    // only 1.6 CTAs per SM on this card and measures 230 GFLOP/s (measured, kbench); the token tiles are what
    // gives the card enough work in flight.
    return detail::launch_rows(type, w, nb, rows, acts, act_stride, nt, out, out_stride, st);
}

inline int es_q4k_rows(const uint8_t* w, int nb, int rows, const uint8_t* acts, int act_stride, int nt, float* out,
                       int out_stride, cudaStream_t st) {
    return es_launch(12, w, nb, rows, acts, act_stride, nt, out, out_stride, st);
}

inline int es_q6k_rows(const uint8_t* w, int nb, int rows, const uint8_t* acts, int act_stride, int nt, float* out,
                       int out_stride, cudaStream_t st) {
    return es_launch(14, w, nb, rows, acts, act_stride, nt, out, out_stride, st);
}

/// The gate/up pair of one expert in a single launch: `w` holds [gate rows | up rows] at `rows` rows each, the blob's
/// own order.  out[token * out_stride + k] is the gate product and out[token * out_stride + rows + k] the up one,
/// which is exactly what es_q4k_rows(2 * rows) writes - this entry only loads each token's activation bytes once for
/// both rows.  Bit-identical to that call; the kernel test proves it against it.
/// The plane-split Q6_K entry: `w` is one expert's packed region (see q6k_plane_tile), nb / rows its geometry, and
/// the output is es_q6k_rows' output for the same rows, bit for bit.
inline int es_q6k_plane_rows(const uint8_t* w, int nb, int rows, const uint8_t* acts, int act_stride, int nt,
                             float* out, int out_stride, cudaStream_t st) {
    if (es_unsupported(14, nb, rows, w, acts) != nullptr) return 1;
    if (nt < 1 || nt > 1024) return 1;
    if (act_stride < (int) (nb * detail::kQ8KBlock)) return 1;
    if (out == nullptr || out_stride < rows) return 1;
    return detail::launch_plane(w, nb, rows, acts, act_stride, nt, out, out_stride, st);
}

/// hidden[token][block] = Q8_K of the SwiGLU of gu[token][gate, up], bit for bit what the host bridge writes.
inline int es_swiglu_quant(const float* gu, int n_ff, int rows, int h_bytes, uint8_t* hidden, cudaStream_t st) {
    if (gu == nullptr || hidden == nullptr) return 1;
    if (rows < 1 || n_ff < detail::kQK || n_ff % detail::kQK != 0) return 1;
    if (h_bytes < (n_ff / detail::kQK) * detail::kQ8KBlock) return 1;
    const dim3 grid((unsigned) (n_ff / detail::kQK), (unsigned) rows);
    detail::swiglu_quant_kernel<<<grid, (unsigned) detail::kQK, 0, st>>>(gu, n_ff, rows, hidden);
    return 0;
}

/// Packs one expert's block-interleaved Q6_K rows into the planes q6k_plane_tile reads.  Same bytes in, same bytes
/// out; the caller copies src to the device first (the transfer is unchanged) and reads dst in the down launch.
inline int es_q6k_pack_planes(const uint8_t* src, uint8_t* dst, int nb, int rows, cudaStream_t st) {
    if (src == nullptr || dst == nullptr || nb < 1 || rows < 1) return 1;
    const long long n = (long long) rows * (long long) nb;
    const int threads = 128;
    const long long blocks = (n + threads - 1) / threads;
    if (blocks > 0x7fffffffLL) return 1;
    detail::q6k_pack_planes<<<(unsigned) blocks, threads, 0, st>>>(src, dst, nb, rows);
    return 0;
}

inline int es_q4k_gu_rows(const uint8_t* w, int nb, int rows, const uint8_t* acts, int act_stride, int nt, float* out,
                          int out_stride, cudaStream_t st) {
    if (es_unsupported(12, nb, rows, w, acts) != nullptr) return 1;
    if (nt < 1 || nt > 1024) return 1;
    if (act_stride < (int) (nb * detail::kQ8KBlock)) return 1;
    if (out == nullptr || out_stride < 2 * rows) return 1;
    return detail::launch_gu(w, nb, rows, acts, act_stride, nt, out, out_stride, st);
}

}  // namespace strata::kolibri::stream
