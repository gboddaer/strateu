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
/// than es_q4k_rows over the whole [gate | up] table.  1 = paired, 0 or unset = the whole-table launch.  Both write
/// the same bits: the kernel test runs the two entries on one blob and compares them.
inline bool es_gu_pairing() {
    const char* v = std::getenv("STRATA_KOLIBRI_STREAM_GU");
    return v != nullptr && v[0] == '1';
}

/// STRATA_KOLIBRI_STREAM_PLANES: hold each Q6_K down blob a second time in the plane-split layout (q6k_plane_tile)
/// and compute the down product from there.  1 = planes, 0 or unset = the block-interleaved blob as it arrives.
/// The copy is byte-neutral but costs its size in device memory per slot; both launch the same arithmetic.
inline bool es_plane_layout() {
    const char* v = std::getenv("STRATA_KOLIBRI_STREAM_PLANES");
    return v != nullptr && v[0] == '1';
}

/// Why this geometry cannot run here, or nullptr when it can.  Checked before anything is allocated or launched.
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

/// Q4_K x Q8_K, four tokens per row.  Thread = one of the row's eight integer lanes; the four token groups of a
/// warp are the same eight lanes, so every token keeps the CPU's own accumulation sequence in its own registers.
///
/// SMEM replaces the early return with an `active` predicate: staging needs every thread of the CTA at
/// __syncthreads(), so a thread whose row is past `rows` still copies and still synchronises, and only its own
/// weight loads and its store are skipped.  `sact` is one byte (not one tile) where SMEM is off, so the two
/// instantiations use the same registers and occupancy and the A/B measures the staging and nothing else.
template <bool SMEM>
__global__ void q4k_tile(const uint8_t* __restrict__ w, const uint8_t* __restrict__ acts, int act_stride, int nb,
                         int rows, int nt, float* __restrict__ out, int out_stride) {
    __shared__ uint8_t sact[SMEM ? kTileTokens * kQ8KBlock : 1];
    const int warp = (int) (threadIdx.x >> 5);
    const int lane = (int) (threadIdx.x & 31);
    const int row = blockIdx.x * kTileRows + warp * 4 + (lane >> 3);
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
template <bool SMEM>
__global__ void q6k_tile(const uint8_t* __restrict__ w, const uint8_t* __restrict__ acts, int act_stride, int nb,
                         int rows, int nt, float* __restrict__ out, int out_stride) {
    __shared__ uint8_t sact[SMEM ? kTileTokens * kQ8KBlock : 1];
    const int warp = (int) (threadIdx.x >> 5);
    const int lane = (int) (threadIdx.x & 31);
    const int row = blockIdx.x * kTileRows + warp * 4 + (lane >> 3);
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
template <bool SMEM>
__global__ void q4k_gu_tile(const uint8_t* __restrict__ w, const uint8_t* __restrict__ acts, int act_stride, int nb,
                            int rows, int nt, float* __restrict__ out, int out_stride) {
    __shared__ uint8_t sact[SMEM ? kTileTokens * kQ8KBlock : 1];
    const int warp = (int) (threadIdx.x >> 5);
    const int lane = (int) (threadIdx.x & 31);
    const int kk = blockIdx.x * kTileRows + warp * 4 + (lane >> 3);
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
template <bool SMEM>
__global__ void q6k_plane_tile(const uint8_t* __restrict__ w, const uint8_t* __restrict__ acts, int act_stride,
                               int nb, int rows, int nt, float* __restrict__ out, int out_stride) {
    __shared__ uint8_t sact[SMEM ? kTileTokens * kQ8KBlock : 1];
    const int warp = (int) (threadIdx.x >> 5);
    const int lane = (int) (threadIdx.x & 31);
    const int row = blockIdx.x * kTileRows + warp * 4 + (lane >> 3);
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
    const dim3 grid((unsigned) ((rows + kTileRows - 1) / kTileRows), (unsigned) ((nt + kTileTokens - 1) / kTileTokens));
    if (type == 12) {
        if (es_smem_staging()) {
            detail::q4k_tile<true><<<grid, 128, 0, st>>>(w, acts, act_stride, nb, rows, nt, out, out_stride);
        } else {
            detail::q4k_tile<false><<<grid, 128, 0, st>>>(w, acts, act_stride, nb, rows, nt, out, out_stride);
        }
    } else {
        if (es_smem_staging()) {
            detail::q6k_tile<true><<<grid, 128, 0, st>>>(w, acts, act_stride, nb, rows, nt, out, out_stride);
        } else {
            detail::q6k_tile<false><<<grid, 128, 0, st>>>(w, acts, act_stride, nb, rows, nt, out, out_stride);
        }
    }
    return 0;
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
    const dim3 grid((unsigned) ((rows + kTileRows - 1) / kTileRows),
                    (unsigned) ((nt + kTileTokens - 1) / kTileTokens));
    if (es_smem_staging()) {
        detail::q6k_plane_tile<true><<<grid, 128, 0, st>>>(w, acts, act_stride, nb, rows, nt, out, out_stride);
    } else {
        detail::q6k_plane_tile<false><<<grid, 128, 0, st>>>(w, acts, act_stride, nb, rows, nt, out, out_stride);
    }
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
    const dim3 grid((unsigned) ((rows + kTileRows - 1) / kTileRows),
                    (unsigned) ((nt + kTileTokens - 1) / kTileTokens));
    if (es_smem_staging()) {
        detail::q4k_gu_tile<true><<<grid, 128, 0, st>>>(w, acts, act_stride, nb, rows, nt, out, out_stride);
    } else {
        detail::q4k_gu_tile<false><<<grid, 128, 0, st>>>(w, acts, act_stride, nb, rows, nt, out, out_stride);
    }
    return 0;
}

}  // namespace strata::kolibri::stream
