// A1: the pool's Q4_K x Q8_K expert dot on the GPU, reproducing Strata's AVX2 summation order EXACTLY.
//
// The order is not a detail here, it is the contract. src/kernels/cpu/kq_avx2.cpp's q4k_dot keeps eight int32
// lanes: lane k accumulates the four-value groups at 64j + 4k and 64j + 32 + 4k, scaled by scales[2j] and
// scales[2j+1] respectively; each block adds d * lane and -dmin * minimum-lane as an FMA; and the eight lanes are
// summed with ((a0+a4)+(a2+a6)) + ((a1+a5)+(a3+a7)), against (m0+m2)+(m1+m3) for the minima. probe/q4k_multiblock.cpp
// pins every one of those steps against the real kernel, and this file reproduces them.
//
// Why the sign matters, in one line: the CPU computes dmin = -y.d * xdmin. A port that uses +y.d * xdmin still
// looks plausible, still produces finite numbers, and is wrong by up to 1.05 in the logits - which is how the first
// attempt failed.
//
// Parallel shape: one warp handles four rows; the eight lanes of each row own one integer lane of the dot, so every
// lane accumulates in exactly the scalar order and no cross-lane reduction reorders anything. The final tree is
// performed by lane 0 from its eight lane values (shuffle width 8 keeps it inside the row).
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace strata::kolibri::a1 {

constexpr int kQK = 256;          // values per super-block
constexpr int kQ4KBlock = 144;    // bytes: d, dmin, 12 scale bytes, 128 nibbles
constexpr int kQ8KBlock = 292;    // bytes: d(float), 256 int8, 16 int16 sums
constexpr int kQ6KBlock = 210;    // bytes: ql[128], qh[64], scales[16] int8, d (fp16)
constexpr int kLanesPerRow = 8;   // one thread per integer lane
constexpr int kRowsPerWarp = 4;   // 32 / 8

/// Is this geometry something this kernel may compute? Refuse loudly rather than compute something else.
inline const char* unsupported(int nb, int64_t rows, const void* w, const void* act) {
    if (w == nullptr || act == nullptr) return "null weights or activations";
    if (nb < 1) return "nb must be at least one super-block";
    if (nb > 1024) return "nb beyond 1024 super-blocks (262144 values) is not supported";
    if (rows < 1) return "rows must be positive";
    if (rows > (int64_t)1 << 20) return "rows beyond 1048576 are not supported";
    return nullptr;
}

/// The down projection is Q6_K on half of Kolibri's layers: same guards, different block size.
inline const char* unsupported_q6k(int nb, int64_t rows, const void* w, const void* act) {
    return unsupported(nb, rows, w, act);
}

__device__ __forceinline__ float tree8(const float* a) {
    return ((a[0] + a[4]) + (a[2] + a[6])) + ((a[1] + a[5]) + (a[3] + a[7]));
}

/// One warp per four rows. `w` holds `rows` contiguous rows of `nb` Q4_K blocks; `act` holds `nb` Q8_K blocks
/// shared by every row (the activations of the token being multiplied). Results are written per row by lane 0.
__global__ void kq_a1_dot_rows(const uint8_t* __restrict__ w, const uint8_t* __restrict__ act, int nb, int rows,
                              float* __restrict__ out) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int row = warp * kRowsPerWarp + (lane >> 3);      // four rows per warp
    const int k = lane & (kLanesPerRow - 1);                // this thread's integer lane
    if (row >= rows) return;

    const uint8_t* xrow = w + (size_t) row * nb * kQ4KBlock;
    float acc = 0.f;     // lane k's accumulation across blocks
    float am = 0.f;      // lane k's minimum correction (only k < 4 uses it)

    for (int i = 0; i < nb; ++i) {
        const uint8_t* xb = xrow + (size_t) i * kQ4KBlock;
        uint16_t hd, hm;
        __builtin_memcpy(&hd, xb, 2);
        __builtin_memcpy(&hm, xb + 2, 2);
        const float xd = __half2float(*(__half*) &hd), xdmin = __half2float(*(__half*) &hm);

        int s[8], mn[8];
        for (int j = 0; j < 4; ++j) { s[j] = xb[4 + j] & 63; mn[j] = xb[8 + j] & 63; }
        for (int j = 4; j < 8; ++j) {
            s[j] = (xb[8 + j] & 0x0F) | ((xb[j] >> 6) << 4);
            mn[j] = (xb[8 + j] >> 4) | ((xb[4 + j] >> 6) << 4);
        }

        const uint8_t* qs = xb + 16;
        const uint8_t* yb = act + (size_t) i * kQ8KBlock;
        const float yd = *(const float*) yb;
        const int8_t* yq = (const int8_t*) (yb + 4);
        const int16_t* ysums = (const int16_t*) (yb + 4 + kQK);

        // lane k: the four-value group at 64j + 4k, scaled by s[2j], and the one at 64j + 32 + 4k, scaled by s[2j+1]
        int sumi = 0;
        for (int j = 0; j < 4; ++j) {
            for (int half = 0; half < 2; ++half) {
                const int base = 64 * j + 32 * half;
                const int sc = s[2 * j + half];
                int v = 0;
                for (int t = 0; t < 4; ++t) {
                    const int idx = base + 4 * k + t;
                    const int byte = qs[32 * j + ((idx & 63) & 31)];
                    const int nib = ((idx & 63) < 32) ? (byte & 0xF) : (byte >> 4);
                    v += nib * int(yq[idx]);
                }
                sumi += sc * v;     // integer: exact, so the order inside a lane cannot matter
            }
        }
        acc = __fmaf_rn(xd * yd, (float) sumi, acc);
        if (k < 4) {
            const int prod = mn[2 * k] * (int(ysums[4 * k]) + int(ysums[4 * k + 1])) +
                             mn[2 * k + 1] * (int(ysums[4 * k + 2]) + int(ysums[4 * k + 3]));
            // the CPU's sign: dmin = -y.d * xdmin
            am = __fmaf_rn(-(yd * xdmin), (float) prod, am);
        }
    }

    // The final tree, in the CPU's order, inside the row's eight lanes.
    float a[8], m[4];
    for (int j = 0; j < 8; ++j) a[j] = __shfl_sync(0xffffffffu, acc, j, kLanesPerRow);
    for (int j = 0; j < 4; ++j) m[j] = __shfl_sync(0xffffffffu, am, j, kLanesPerRow);
    if (k == 0) out[row] = __fadd_rn(tree8(a), (m[0] + m[2]) + (m[1] + m[3]));
}

/// Q6_K x Q8_K, reproducing ggml's AVX2 branch (quants.c: ggml_vec_dot_q6_K_q8_K) exactly.
///
/// Two details make this different from the Q4_K order above.  There is no minimum term: the six-bit weights
/// are offset by 32 and the offset is removed as an integer sum, `sumi -= 32 * (scales * bsums)`.  And that
/// subtraction is applied per lane from the sub-block PAIR 2k, 2k+1 - a distribution that does not follow the
/// values each lane actually holds.  The integer total is identical either way, but the float accumulation is
/// per lane, so the distribution decides the rounding.  Reproducing it is the whole point of this file.
__global__ void kq_a1_dot_rows_q6k(const uint8_t* __restrict__ w, const uint8_t* __restrict__ act, int nb, int rows,
                                   float* __restrict__ out) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int row = warp * kRowsPerWarp + (lane >> 3);
    const int k = lane & (kLanesPerRow - 1);
    if (row >= rows) return;

    const uint8_t* xrow = w + (size_t) row * nb * kQ6KBlock;
    float acc = 0.f;
    for (int i = 0; i < nb; ++i) {
        const uint8_t* xb = xrow + (size_t) i * kQ6KBlock;
        const uint8_t* ql = xb;              // 128 bytes of paired 4-bit halves
        const uint8_t* qh = xb + 128;        // 64 bytes of the two high bits
        const int8_t* sc = (const int8_t*) (xb + 192);
        uint16_t hd;
        __builtin_memcpy(&hd, xb + 208, 2);  // the super-block scale is at the END of a Q6_K block
        const float xd = __half2float(*(__half*) &hd);

        const uint8_t* yb = act + (size_t) i * kQ8KBlock;
        const float yd = *(const float*) yb;
        const int8_t* yq = (const int8_t*) (yb + 4);
        const int16_t* ysums = (const int16_t*) (yb + 4 + kQK);

        int raw = 0;
        for (int j = 0; j < 2; ++j) {
            for (int c = 0; c < 4; ++c) {
                // A 32-value chunk carries TWO scales: ggml's get_scale_shuffle expands each index into the pair
                // (2n, 2n+1) of the 16-value sub-blocks, so the first 16 values use scales[8j+2c] and the second
                // 16 use scales[8j+2c+1]. Lane k holds values 4k..4k+3, so lanes 0-3 take the first, 4-7 the second.
                const int s = int(sc[8 * j + 2 * c + (k >> 2)]);
                const int byte_base = 64 * j + (c & 1) * 32;   // each 128-group consumes 64 ql bytes
                int v4 = 0;
                for (int t = 0; t < 4; ++t) {
                    const int l = 4 * k + t;
                    const int b = ql[byte_base + l];
                    const int low = (c < 2) ? (b & 0xF) : ((b >> 4) & 0xF);
                    const int high = ((qh[32 * j + l] >> (2 * c)) & 3) << 4;
                    v4 += (low | high) * int(yq[128 * j + 32 * c + l]);
                }
                raw += s * v4;
            }
        }
        // the offset removal, distributed exactly as ggml's AVX2 kernel distributes it
        const int corr = 32 * (int(ysums[2 * k]) * int(sc[2 * k]) + int(ysums[2 * k + 1]) * int(sc[2 * k + 1]));
        acc = __fmaf_rn(xd * yd, float(raw - corr), acc);
    }

    float a[8];
    for (int j = 0; j < 8; ++j) a[j] = __shfl_sync(0xffffffffu, acc, j, kLanesPerRow);
    if (k == 0) out[row] = tree8(a);
}

}  // namespace strata::kolibri::a1
