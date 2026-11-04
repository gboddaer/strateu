// Does the A1 kernel in src/kolibri reproduce Strata's AVX2 q4k_dot exactly?
// Reference: strata::kernels::cpu::kq256_rows(...) -> dot_rows -> q4k_dot (src/kernels/cpu/kq_avx2.cpp).
// The kernel under test is src/kolibri/kq_a1.cuh; this file contains no copy of it.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "strata/kernels/cpu/kq_avx2.hpp"
#include "ggml-cpu.h"
#include "../src/kolibri/kq_a1.cuh"

using namespace strata::kernels::cpu;
static constexpr int QK_K = 256;

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
static uint64_t rs = 0x243F6A8885A308D3ull;
static uint32_t rnd() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return uint32_t(rs >> 32); }

static void make_q4k(uint8_t* b) {
    for (int i = 0; i < 144; ++i) b[i] = uint8_t(rnd());
    const float d = 0.004f + float(rnd() % 1000) * 2e-5f;
    const float dmin = (rnd() & 1 ? -1.f : 1.f) * (0.004f + float(rnd() % 1000) * 2e-5f);
    const uint16_t hd = f2h(d), hm = f2h(dmin);
    std::memcpy(b, &hd, 2); std::memcpy(b + 2, &hm, 2);
}

int main(int argc, char** argv) {
    const int nb = argc > 1 ? std::stoi(argv[1]) : 10;
    if (nb < 1 || nb > 64) return 2;
    const size_t act_bytes = size_t(nb) * 292;
    const size_t gu_row = size_t(nb) * 144;
    const int rows = 512, n = nb * QK_K, nt = 4;

    // Shape guards must refuse rather than compute something else.
    int guard_fail = 0;
    if (strata::kolibri::a1::unsupported(0, rows, (void*) 1, (void*) 1) == nullptr) ++guard_fail;
    if (strata::kolibri::a1::unsupported(nb, rows, nullptr, (void*) 1) == nullptr) ++guard_fail;
    if (strata::kolibri::a1::unsupported(2048, rows, (void*) 1, (void*) 1) == nullptr) ++guard_fail;
    if (strata::kolibri::a1::unsupported(nb, rows, (void*) 1, (void*) 1) != nullptr) ++guard_fail;
    if (guard_fail) { std::printf("shape guards: FAIL (%d)\n", guard_fail); return 1; }

    std::vector<uint8_t> w(size_t(rows) * nb * 144);
    for (size_t i = 0; i < w.size() / 144; ++i) make_q4k(&w[i * 144]);
    std::vector<uint8_t> act(size_t(nt) * act_bytes);
    std::vector<const void*> actp(nt);
    std::vector<float> xv(size_t(nt) * n);
    const ggml_type_traits_cpu* t8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    if (!t8 || !t8->from_float) { std::printf("no Q8_K quantiser in this ggml\n"); return 2; }
    for (auto& v : xv) v = -1.f + 2.f * float(rnd() % 20001) / 10000.f;
    for (int t = 0; t < nt; ++t) {
        t8->from_float(xv.data() + size_t(t) * n, act.data() + size_t(t) * act_bytes, n);
        actp[t] = act.data() + size_t(t) * act_bytes;
    }

    std::vector<float> ref(size_t(nt) * rows), got(rows);
    std::vector<float*> outp(nt);
    for (int t = 0; t < nt; ++t) outp[t] = ref.data() + size_t(t) * rows;
    const auto c0 = std::chrono::steady_clock::now();
    kq256_rows(12, w.data(), gu_row, n, actp.data(), nt, outp.data(), 0, rows);
    const double cpu_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count();

    uint8_t *dw, *da; float* dout;
    cudaMalloc(&dw, w.size()); cudaMalloc(&da, act.size()); cudaMalloc(&dout, size_t(rows) * 4);
    cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice);

    long long match = 0, total = 0; double worst = 0; double gpu_ms = 0;
    for (int t = 0; t < nt; ++t) {
        cudaMemcpy(da, act.data() + size_t(t) * act_bytes, act_bytes, cudaMemcpyHostToDevice);
        const int threads = 128, rows_per_block = (threads / 32) * strata::kolibri::a1::kRowsPerWarp;
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);
        strata::kolibri::a1::kq_a1_dot_rows<<<(rows + rows_per_block - 1) / rows_per_block, threads>>>(
            dw, da, nb, rows, dout);
        cudaEventRecord(e1);
        if (cudaEventSynchronize(e1) != cudaSuccess || cudaGetLastError() != cudaSuccess) return 2;
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1); gpu_ms += ms;
        cudaMemcpy(got.data(), dout, size_t(rows) * 4, cudaMemcpyDeviceToHost);
        for (int r = 0; r < rows; ++r) {
            ++total;
            if (!std::isfinite(got[r])) { std::printf("non-finite at row %d\n", r); return 1; }
            if (std::memcmp(&got[r], &outp[t][r], 4) == 0) ++match;
            else worst = std::fmax(worst, std::fabs(double(got[r]) - double(outp[t][r])));
        }
    }
    gpu_ms /= nt;
    std::printf("nb=%-3d rows=%d tokens=%d: %lld/%lld bit-identical with kq256_rows (%.2f%%), worst |d| %.9g\n",
                nb, rows, nt, match, total, 100.0 * double(match) / double(total), worst);
    std::printf("time: CPU %.3f ms, GPU %.4f ms per launch (informational, not a benchmark)\n", cpu_ms, gpu_ms);
    cudaFree(dw); cudaFree(da); cudaFree(dout);
    const bool ok = match == total;
    std::printf("A1 kernel exactness: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
