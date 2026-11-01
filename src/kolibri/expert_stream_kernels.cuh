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

namespace strata::kolibri::stream {

/// Output rows per CTA and tokens per row: the tile the plan fixes before any tuning.
constexpr int kTileRows = 16;
constexpr int kTileTokens = 4;

/// Why this geometry cannot run here, or nullptr when it can.  Checked before anything is allocated or launched.
const char* es_unsupported(int ggml_weight_type, int nb, int rows, const void* w, const void* acts);

/// out[token * out_stride + row] = w_row . acts[token], for rows [0, rows) and tokens [0, nt).
/// `w` points at `rows` rows of `nb` blocks; `acts` is a HOST array of DEVICE pointers to `nb` Q8_K blocks each.
/// The launcher unpacks at most kTileTokens of those pointers into kernel arguments: a kernel must never read a
/// host array, and a device pointer array would cost an upload per launch.
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
__global__ void q4k_tile(const uint8_t* __restrict__ w, const uint8_t* __restrict__ a0, const uint8_t* __restrict__ a1,
                         const uint8_t* __restrict__ a2, const uint8_t* __restrict__ a3, int nb, int rows, int tn,
                         float* __restrict__ out, int out_stride) {
    const int warp = (int) (threadIdx.x >> 5);
    const int lane = (int) (threadIdx.x & 31);
    const int row = blockIdx.x * kTileRows + warp * 4 + (lane >> 3);
    const int k = lane & (a1::kLanesPerRow - 1);
    if (row >= rows) return;   // a row's eight lanes share the row, so a group leaves or stays together
    const unsigned gmask = 0xffu << (8 * (lane >> 3));

    const uint8_t* xrow = w + (size_t) row * (size_t) nb * kQ4KBlock;
    const uint8_t* yb[kTileTokens] = {a0, a1, a2, a3};
    float acc[kTileTokens], am[kTileTokens];
#pragma unroll
    for (int t = 0; t < kTileTokens; ++t) {
        if (t >= tn) yb[t] = nullptr;
        acc[t] = 0.f;
        am[t] = 0.f;
    }

    for (int i = 0; i < nb; ++i) {
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
        const uint8_t* yptr[kTileTokens];
#pragma unroll
        for (int t = 0; t < kTileTokens; ++t)
            yptr[t] = yb[t] ? yb[t] + (size_t) i * kQ8KBlock : nullptr;

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

#pragma unroll
    for (int t = 0; t < kTileTokens; ++t) {
        if (!yb[t]) continue;
        float a[8], m[4];
#pragma unroll
        for (int j = 0; j < 8; ++j) a[j] = __shfl_sync(gmask, acc[t], j, a1::kLanesPerRow);
#pragma unroll
        for (int j = 0; j < 4; ++j) m[j] = __shfl_sync(gmask, am[t], j, a1::kLanesPerRow);
        if (k == 0) out[(size_t) t * out_stride + row] = __fadd_rn(tree8(a), (m[0] + m[2]) + (m[1] + m[3]));
    }
}

/// Q6_K x Q8_K, four tokens per row: A1's per-lane order, per block, per token.
__global__ void q6k_tile(const uint8_t* __restrict__ w, const uint8_t* __restrict__ a0, const uint8_t* __restrict__ a1,
                         const uint8_t* __restrict__ a2, const uint8_t* __restrict__ a3, int nb, int rows, int tn,
                         float* __restrict__ out, int out_stride) {
    const int warp = (int) (threadIdx.x >> 5);
    const int lane = (int) (threadIdx.x & 31);
    const int row = blockIdx.x * kTileRows + warp * 4 + (lane >> 3);
    const int k = lane & (a1::kLanesPerRow - 1);
    if (row >= rows) return;   // a row's eight lanes share the row, so a group leaves or stays together
    const unsigned gmask = 0xffu << (8 * (lane >> 3));

    const uint8_t* xrow = w + (size_t) row * (size_t) nb * kQ6KBlock;
    const uint8_t* yb[kTileTokens] = {a0, a1, a2, a3};
    float acc[kTileTokens];
#pragma unroll
    for (int t = 0; t < kTileTokens; ++t) {
        if (t >= tn) yb[t] = nullptr;
        acc[t] = 0.f;
    }

    for (int i = 0; i < nb; ++i) {
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
                    if (!yb[t]) continue;
                    const int8_t* yq = (const int8_t*) (yb[t] + (size_t) i * kQ8KBlock + 4);
                    raw[t] += s * __dp4a(w4, ld4(yq + 128 * j + 32 * c + 4 * k), 0);
                }
            }
        }
#pragma unroll
        for (int t = 0; t < kTileTokens; ++t) {
            if (!yb[t]) continue;
            const uint8_t* yp = yb[t] + (size_t) i * kQ8KBlock;
            const float yd = *(const float*) yp;
            const int16_t* ys = (const int16_t*) (yp + 4 + a1::kQK);
            // the offset removal, distributed exactly as ggml's AVX2 kernel distributes it
            const int corr = 32 * (int(ys[2 * k]) * int(sc[2 * k]) + int(ys[2 * k + 1]) * int(sc[2 * k + 1]));
            acc[t] = __fmaf_rn(xd * yd, float(raw[t] - corr), acc[t]);
        }
    }

#pragma unroll
    for (int t = 0; t < kTileTokens; ++t) {
        if (!yb[t]) continue;
        float a[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) a[j] = __shfl_sync(gmask, acc[t], j, a1::kLanesPerRow);
        if (k == 0) out[(size_t) t * out_stride + row] = tree8(a);
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
inline int es_launch(int type, const uint8_t* w, int nb, int rows, const uint8_t* const* acts, int nt, float* out,
                     int out_stride, cudaStream_t st) {
    if (es_unsupported(type, nb, rows, w, acts) != nullptr) return 1;
    if (nt < 1 || nt > 1024) return 1;
    if (out == nullptr || out_stride < rows) return 1;
    const dim3 grid((unsigned) ((rows + kTileRows - 1) / kTileRows));
    for (int t0 = 0; t0 < nt; t0 += kTileTokens) {
        const int tn = (nt - t0 < kTileTokens) ? nt - t0 : kTileTokens;
        const uint8_t* p[4] = {nullptr, nullptr, nullptr, nullptr};
        for (int t = 0; t < tn; ++t) p[t] = acts[t0 + t];
        float* o = out + (size_t) t0 * (size_t) out_stride;
        if (type == 12) {
            detail::q4k_tile<<<grid, 128, 0, st>>>(w, p[0], p[1], p[2], p[3], nb, rows, tn, o, out_stride);
        } else {
            detail::q6k_tile<<<grid, 128, 0, st>>>(w, p[0], p[1], p[2], p[3], nb, rows, tn, o, out_stride);
        }
    }
    return 0;
}

inline int es_q4k_rows(const uint8_t* w, int nb, int rows, const uint8_t* const* acts, int nt, float* out,
                       int out_stride, cudaStream_t st) {
    return es_launch(12, w, nb, rows, acts, nt, out, out_stride, st);
}

inline int es_q6k_rows(const uint8_t* w, int nb, int rows, const uint8_t* const* acts, int nt, float* out,
                       int out_stride, cudaStream_t st) {
    return es_launch(14, w, nb, rows, acts, nt, out, out_stride, st);
}

}  // namespace strata::kolibri::stream
