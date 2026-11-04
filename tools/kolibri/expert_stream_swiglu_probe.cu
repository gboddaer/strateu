// Is the host's SwiGLU + Q8_K quantization reproducible on the device, bit for bit?
//
// The streamed expert leg currently finishes each group on the host: the gate/up floats come back (D2H), the host
// computes the pool's SwiGLU expression, quantizes the hidden row with native_quant_h, and sends it back (H2D). Those
// two round trips sit inside the per-batch critical path. Doing both on the device would remove them, but only if the
// device writes the same bytes.
//
// This probe answers that first, on random rows with the model's own width (n_ff = 512, Q8_K):
//   * host:  ff = (g / (1 + std::exp(-g))) * u   then  q8k_quant_avx2(ff)          <- the reference bytes
//   * dev A: the same expression evaluated in DOUBLE with exp(double), then the quantizer below
//   * dev B: the same expression in FLOAT with __expf, then the quantizer below
// It reports, per variant, how many rows differ from the reference and how many values do; and separately how many
// of the SwiGLU floats themselves differ, which is where a mismatch has to come from.
//
// Build: cmake --build build-sram --target kolibri_expert_stream_swiglu_probe -j16
// Run:   ./build-sram/kolibri_expert_stream_swiglu_probe [rows]
#include "strata/kernels/cpu/iq_avx2.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "ggml.h"

namespace {

constexpr int kN = 512;   // Kolibri's n_ff: one hidden row, two Q8_K blocks
constexpr int kQK = 256;

struct Q8K {
    float d;
    int8_t qs[kQK];
    int16_t bsums[16];
};

static_assert(sizeof(Q8K) == 292, "the Q8_K block is 292 bytes");

/// The reference quantizer's arithmetic, one value at a time (src/kernels/cpu/iq_avx2.cpp q8k_quant_avx2): the magic
/// add rounds to nearest-even without a libm call, the low byte is sign-extended, the block's d is 1 / iscale.
__device__ __forceinline__ void quant_block(const float* __restrict__ x, Q8K* out) {
    float m = 0.f;
    for (int j = 0; j < kQK; ++j) m = fmaxf(m, fabsf(x[j]));
    if (m == 0.f) {
        out->d = 0.f;
        for (int j = 0; j < kQK; ++j) out->qs[j] = 0;
        for (int j = 0; j < 16; ++j) out->bsums[j] = 0;
        return;
    }
    float mx = 0.f;
    for (int j = 0; j < kQK; ++j)
        if (fabsf(x[j]) == m) {
            mx = x[j];
            break;
        }
    const float iscale = __fdiv_rn(-127.f, mx);
    for (int j = 0; j < kQK; ++j) {
        const float p = __fmul_rn(iscale, x[j]);
        const float t = __fadd_rn(p, 12582912.f);
        int v = ((__float_as_int(t) & 0x007fffff) - 0x00400000);
        v = v < 127 ? v : 127;
        out->qs[j] = (int8_t) (v & 0xff);
    }
    for (int g = 0; g < 16; ++g) {
        int s = 0;
        for (int j = 0; j < 16; ++j) s += (int) out->qs[g * 16 + j];
        out->bsums[g] = (int16_t) s;
    }
    out->d = __fdiv_rn(1.f, iscale);
}

/// One row of n_ff values, block by block: the SwiGLU value, then the Q8_K block it belongs to.
template <int VARIANT>
__device__ __forceinline__ float swiglu_value(float g, float u) {
    if constexpr (VARIANT == 0) {   // double throughout, exp(double)
        const double gd = (double) g;
        return (float) ((gd / (1.0 + exp((double) (-g)))) * (double) u);
    } else if constexpr (VARIANT == 1) {   // the host's own precision and operations: float, expf, separate rounding
        return __fmul_rn(__fdiv_rn(g, __fadd_rn(1.f, expf(-g))), u);
    } else {   // the fast intrinsic, for contrast
        return __fmul_rn(__fdiv_rn(g, __fadd_rn(1.f, __expf(-g))), u);
    }
}

template <int VARIANT>
__global__ void swiglu_quant(const float* __restrict__ gu, int n_ff, int rows, Q8K* out, float* ff_out) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* row = gu + (size_t) r * 2 * n_ff;
    const int nb = n_ff / kQK;
    for (int b = 0; b < nb; ++b) {
        float x[kQK];
        for (int j = 0; j < kQK; ++j) {
            const int idx = b * kQK + j;
            const float v = swiglu_value<VARIANT>(row[idx], row[n_ff + idx]);
            x[j] = v;
            if (ff_out != nullptr) ff_out[(size_t) r * n_ff + idx] = v;
        }
        quant_block(x, out + (size_t) r * nb + b);
    }
}

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

/// One distribution's worth of rows through all three paths.
void run(const char* name, std::normal_distribution<float>& nd, std::mt19937& rng, int rows) {
    const int nb = kN / kQK;
    std::vector<float> gu((size_t) rows * 2 * kN);
    for (auto& v : gu) v = nd(rng);

    // the host's path: pool expression in double, then the reference quantizer
    std::vector<Q8K> want((size_t) rows * nb);
    std::vector<float> ff_host((size_t) rows * kN);
    for (int r = 0; r < rows; ++r) {
        const float* row = gu.data() + (size_t) r * 2 * kN;
        for (int j = 0; j < kN; ++j) {
            const float g = row[j], u = row[kN + j];
            ff_host[(size_t) r * kN + j] = (g / (1.f + std::exp(-g))) * u;
        }
        strata::kernels::cpu::q8k_quant_avx2(ff_host.data() + (size_t) r * kN, want.data() + (size_t) r * nb, kN);
    }

    float *dgu = nullptr, *dff = nullptr;
    Q8K* dout = nullptr;
    check(cudaMalloc(&dgu, gu.size() * sizeof(float)), "gu");
    check(cudaMalloc(&dout, want.size() * sizeof(Q8K)), "out");
    check(cudaMalloc(&dff, ff_host.size() * sizeof(float)), "ff");
    check(cudaMemcpy(dgu, gu.data(), gu.size() * sizeof(float), cudaMemcpyHostToDevice), "copy gu");

    const int threads = 128, blocks = (rows + threads - 1) / threads;
    for (int variant = 0; variant < 3; ++variant) {
        check(cudaMemset(dff, 0, ff_host.size() * sizeof(float)), "clear ff");
        if (variant == 0) {
            swiglu_quant<0><<<blocks, threads>>>(dgu, kN, rows, dout, dff);
        } else if (variant == 1) {
            swiglu_quant<1><<<blocks, threads>>>(dgu, kN, rows, dout, dff);
        } else {
            swiglu_quant<2><<<blocks, threads>>>(dgu, kN, rows, dout, dff);
        }
        check(cudaDeviceSynchronize(), "kernel");
        std::vector<Q8K> got(want.size());
        std::vector<float> ff_dev(ff_host.size());
        check(cudaMemcpy(got.data(), dout, got.size() * sizeof(Q8K), cudaMemcpyDeviceToHost), "copy out");
        check(cudaMemcpy(ff_dev.data(), dff, ff_dev.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy ff");

        if (variant == 1 && rows > 0) {   // one row's first values, so "1 bit" and "wrong value" are told apart
            for (int j = 0; j < 4; ++j) {
                uint32_t hb, db;
                std::memcpy(&hb, &ff_host[j], 4);
                std::memcpy(&db, &ff_dev[j], 4);
                std::printf("    row0[%d]: host %.9g (0x%08x)  device %.9g (0x%08x)  %s\n", j, (double) ff_host[j], hb,
                            (double) ff_dev[j], db, hb == db ? "same" : "DIFFERENT");
            }
        }
        long long ff_diff = 0, v_diff = 0, rows_diff = 0;
        for (size_t i = 0; i < ff_host.size(); ++i) ff_diff += std::memcmp(&ff_host[i], &ff_dev[i], 4) != 0;
        for (int r = 0; r < rows; ++r) {
            bool any = false;
            for (int b = 0; b < nb; ++b) {
                const Q8K& a = want[(size_t) r * nb + b];
                const Q8K& c = got[(size_t) r * nb + b];
                any = any || std::memcmp(&a, &c, sizeof(Q8K)) != 0;
                if (std::memcmp(&a.d, &c.d, 4) != 0) ++v_diff;
                if (std::memcmp(a.qs, c.qs, kQK) != 0) {
                    for (int j = 0; j < kQK; ++j) v_diff += a.qs[j] != c.qs[j];
                }
                if (std::memcmp(a.bsums, c.bsums, sizeof(a.bsums)) != 0) ++v_diff;
            }
            rows_diff += any;
        }
        std::printf("%-9s %-22s rows %6d | swiglu floats differ %8lld (%.2e) | quantized bytes differ %8lld | "
                    "rows with any byte difference %6lld (%.2e)\n",
                    name,
                    variant == 0 ? "double exp" : (variant == 1 ? "float expf" : "float __expf"), rows, ff_diff,
                    (double) ff_diff / (double) ff_host.size(), v_diff, rows_diff,
                    (double) rows_diff / (double) rows);
    }
    cudaFree(dgu);
    cudaFree(dff);
    cudaFree(dout);
}

}  // namespace

int main(int argc, char** argv) {
    const int rows = argc > 1 ? std::atoi(argv[1]) : 20000;
    std::mt19937 rng(2024);
    std::printf("SwiGLU + Q8_K: device against the host's own bytes, n_ff = %d, %d row(s) per distribution\n", kN, rows);
    for (float sigma : {1.0f, 3.0f, 10.0f}) {
        std::normal_distribution<float> nd(0.f, sigma);
        char name[32];
        std::snprintf(name, sizeof name, "sigma=%.0f", (double) sigma);
        run(name, nd, rng, rows);
    }
    std::normal_distribution<float> wide(0.f, 3.f);
    run("sigma=3", wide, rng, rows);
    return 0;
}
