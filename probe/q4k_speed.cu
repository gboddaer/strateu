// Ceiling probe: how fast can this card do the pool's Q4_K x Q8_K dot at Kolibri's shape?
// One thread per weight row, four tokens at once (the CPU's kq256_gu_rows reuses decoded weights across MAXT=4
// tokens; this kernel does the same). dp4a for the int8 products; per 32-value sub-block the exact int32 sum is
// scaled by that sub-block's 6-bit scale, accumulated per block in fp32 -- the same *structure* as the CPU kernel,
// though not yet bit-identical to it (the AVX2 summation order is a separate question).
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <chrono>
#include <thread>
#include "strata/kernels/cpu/kq_avx2.hpp"
#include "ggml-cpu.h"
using namespace strata::kernels::cpu;

static uint16_t f2h(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    const uint16_t sign = uint16_t((x >> 16) & 0x8000u); const uint32_t mag = x & 0x7fffffffu;
    if (mag > 0x7f800000u) return uint16_t(sign | 0x7c00u | 0x200u);
    if (mag >= 0x47800000u) return uint16_t(sign | 0x7c00u);
    const uint32_t e8 = mag >> 23, mant = mag & 0x7fffffu;
    if (mag < 0x38800000u) return uint16_t(sign);
    int e16 = int(e8) - 112;
    uint32_t m = (mant >> 13) + ((((mant >> 12) & 1u) && (((mant & 0xfffu) != 0) || ((mant >> 13) & 1u))) ? 1u : 0u);
    if (m == 0x400u) { m = 0; ++e16; }
    if (e16 >= 31) return uint16_t(sign | 0x7c00u);
    return uint16_t(sign | (uint32_t(e16) << 10) | m);
}
static uint64_t rs = 0x9E3779B97F4A7C15ull;
static uint32_t rnd() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return uint32_t(rs >> 32); }
static void make_q4k(uint8_t* b) {
    for (int i = 0; i < 144; ++i) b[i] = uint8_t(rnd());
    const float d = 0.004f + float(rnd() % 1000) * 2e-5f, dmin = -(0.004f + float(rnd() % 1000) * 2e-5f);
    const uint16_t hd = f2h(d), hm = f2h(dmin);
    std::memcpy(b, &hd, 2); std::memcpy(b + 2, &hm, 2);
    for (int s = 0; s < 12; ++s) b[4 + s] = uint8_t(rnd() & 0x3f);
}

// rows per thread = 1; tokens per thread = 4. `yq[t]` points at that token's 256 int8 values for this block.
__global__ void k_dot4(const uint8_t* __restrict__ w, const int8_t* const* __restrict__ q8, const float* const* dy,
                       int nb, float* out, int rows, int ld) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const uint8_t* xrow = w + (size_t) r * nb * 144;
    float acc[4] = {0,0,0,0};
    for (int t = 0; t < 4; ++t) {
        for (int i = 0; i < nb; ++i) {
            const uint8_t* xb = xrow + (size_t) i * 144;
            uint16_t hd, hm; std::memcpy(&hd, xb, 2); std::memcpy(&hm, xb + 2, 2);
            const float xd = __half2float(*(__half*)&hd), xdmin = __half2float(*(__half*)&hm);
            int sc[8], mn[8];
            for (int j = 0; j < 8; ++j) {
                if (j < 4) { sc[j] = xb[4 + j] & 63; mn[j] = xb[8 + j] & 63; }
                else { sc[j] = (xb[8 + j] & 0x0F) | ((xb[j] >> 6) << 4); mn[j] = (xb[8 + j] >> 4) | ((xb[4 + j] >> 6) << 4); }
            }
            const uint8_t* qs = xb + 16;
            const int8_t* yv = q8[t] + (size_t) i * 256;
            const float* dd = dy[t] + (size_t) i;
            float sum = 0.f;
            for (int j = 0; j < 8; ++j) {             // 32-value sub-block, one scale
                const int g = j >> 1, hi = j & 1;
                int s = 0;
                for (int l = 0; l < 8; ++l) {         // 8 bytes = 16 nibbles... take 4 at a time via dp4a
                    const int b0 = qs[32 * g + 4 * l], b1 = qs[32 * g + 4 * l + 1];
                    const int b2 = qs[32 * g + 4 * l + 2], b3 = qs[32 * g + 4 * l + 3];
                    uint32_t nib;
                    if (hi) nib = ((uint32_t)(b0 >> 4)) | ((uint32_t)(b1 >> 4) << 8) | ((uint32_t)(b2 >> 4) << 16) | ((uint32_t)(b3 >> 4) << 24);
                    else    nib = ((uint32_t)(b0 & 15)) | ((uint32_t)(b1 & 15) << 8) | ((uint32_t)(b2 & 15) << 16) | ((uint32_t)(b3 & 15) << 24);
                    const int8_t* y4 = yv + 64 * g + 32 * hi + 4 * l;
                    uint32_t q4;
                    std::memcpy(&q4, y4, 4);
                    s = __dp4a((int)nib, (int)q4, s);
                }
                sum = fmaf(xd * dd[0], (float)(s * sc[j]), sum);
            }
            // the min term: bsums . mins, per block
            const int16_t* bs = (const int16_t*)((const uint8_t*)q8[t] + (size_t) i * 292 + 260);
            int sumi = 0;
            for (int j = 0; j < 16; ++j) sumi += int(bs[j]) * mn[j / 2];
            sum = fmaf(-(xdmin * dd[0]), (float)sumi, sum);
            acc[t] += sum;
        }
        out[(size_t) t * ld + r] = acc[t];
    }
}

int main() {
    const int rows = 512, n = 2560, nt = 4, nb = n / 256;
    ggml_cpu_init();
    const ggml_type_traits_cpu* t8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    const size_t act_bytes = 10 * 292, gu_row = 10 * 144;
    std::vector<uint8_t> w(size_t(rows) * gu_row);
    for (size_t i = 0; i < w.size() / 144; ++i) make_q4k(&w[i * 144]);
    std::vector<float> xv(size_t(nt) * n);
    for (auto& v : xv) v = -1.f + 2.f * float(rnd() % 20001) / 10000.f;
    std::vector<uint8_t> act(size_t(nt) * act_bytes);
    std::vector<const void*> actp(nt);
    for (int t = 0; t < nt; ++t) { t8->from_float(xv.data() + size_t(t) * n, act.data() + size_t(t) * act_bytes, n); actp[t] = act.data() + size_t(t) * act_bytes; }
    std::vector<float> ref(size_t(nt) * rows), outp(nt);
    std::vector<float*> refp(nt);
    for (int t = 0; t < nt; ++t) refp[t] = ref.data() + size_t(t) * rows;
    // CPU: 1 thread, and 13 threads (each takes a slice of the rows) for the like-for-like ceiling
    auto run_cpu = [&](int threads) {
        auto t0 = std::chrono::steady_clock::now();
        std::vector<std::thread> th;
        for (int k = 0; k < threads; ++k) {
            th.emplace_back([&, k]() {
                const int r0 = rows * k / threads, r1 = rows * (k + 1) / threads;
                std::vector<float*> o(nt);
                for (int t = 0; t < nt; ++t) o[t] = ref.data() + size_t(t) * rows;
                kq256_rows(12, w.data(), gu_row, n, actp.data(), nt, o.data(), r0, r1);
            });
        }
        for (auto& x : th) x.join();
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    };
    const double cpu1 = run_cpu(1), cpu13 = run_cpu(13);

    // GPU
    uint8_t* dw; int8_t* dq[4]; float* ddy[4]; float* dout;
    cudaMalloc(&dw, w.size()); cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice);
    const int8_t* hq[4]; float* hdy[4];
    for (int t = 0; t < nt; ++t) {
        int8_t* q; float* dy;
        cudaMalloc(&q, 10 * 256); cudaMemcpy(q, act.data() + size_t(t) * act_bytes + 4, 10 * 256, cudaMemcpyHostToDevice);
        const float* src = (const float*)(act.data() + size_t(t) * act_bytes);
        std::vector<float> dvals(10); for (int i = 0; i < 10; ++i) dvals[i] = *(const float*)(act.data() + size_t(t) * act_bytes + i * 292);
        cudaMalloc(&dy, 10 * 4); cudaMemcpy(dy, dvals.data(), 40, cudaMemcpyHostToDevice);
        dq[t] = q; ddy[t] = dy; hq[t] = q; hdy[t] = dy;
    }
    const int8_t** dqp; float** ddyp;
    cudaMalloc(&dqp, 4 * 8); cudaMalloc(&ddyp, 4 * 8);
    cudaMemcpy(dqp, hq, 32, cudaMemcpyHostToDevice); cudaMemcpy(ddyp, hdy, 32, cudaMemcpyHostToDevice);
    cudaMalloc(&dout, size_t(nt) * rows * 4);
    float best = 1e9f;
    for (int it = 0; it < 20; ++it) {
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a);
        k_dot4<<<(rows + 63) / 64, 64>>>(dw, dqp, ddyp, nb, dout, rows, rows);
        cudaEventRecord(b); cudaEventSynchronize(b);
        float ms = 0; cudaEventElapsedTime(&ms, a, b); if (ms < best) best = ms;
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
    std::printf("CPU 1 thread: %.2f ms | CPU 13 threads: %.2f ms | GPU dp4a: %.3f ms\n", cpu1, cpu13, best);
    std::printf("same 512 rows x 4 tokens: GPU is %.1fx vs 1 thread, %.1fx vs 13 threads\n", cpu1 / best, cpu13 / best);
    const double flop = 2.0 * rows * nt * n;
    std::printf("effective: CPU13 %.1f GFLOP/s, GPU %.1f GFLOP/s (2 FLOP per MAC)\n", flop / (cpu13 / 1000) / 1e9, flop / (best / 1000) / 1e9);
    (void)dout;
    return 0;
}
