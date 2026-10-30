// Q6_K x Q8_K: does src/kolibri/kq_a1.cuh reproduce ggml's AVX2 order exactly?
// Reference: the same function the CPU pool calls for the down projection, taken from ggml's type traits.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "ggml-cpu.h"
#include "../src/kolibri/kq_a1.cuh"

static constexpr int QK_K = 256;
static constexpr int Q6K = 210;
extern "C" void ggml_vec_dot_q6_K_q8_K(int n, float* s, size_t bs, const void* vx, size_t bx,
                                       const void* vy, size_t by, int nrc);

static uint64_t rs = 0x9E3779B97F4A7C15ull;
static uint32_t rnd() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return uint32_t(rs >> 32); }

static uint16_t f2h(float f) {
    __half h = __float2half(f);
    uint16_t u; std::memcpy(&u, &h, 2); return u;
}

// A valid Q6_K block: ql[128], qh[64], scales[16] (int8, including negatives), d (fp16) at the end.
static void make_q6k(uint8_t* b) {
    for (int i = 0; i < 128; ++i) b[i] = uint8_t(rnd());
    for (int i = 0; i < 64; ++i) b[128 + i] = uint8_t(rnd());
    for (int i = 0; i < 16; ++i) b[192 + i] = uint8_t(rnd());   // full int8 range, sign included
    const uint16_t d = f2h(0.002f + float(rnd() % 1000) * 2e-5f);
    std::memcpy(b + 208, &d, 2);
}

int main(int argc, char** argv) {
    ggml_cpu_init();
    const int nb = argc > 1 ? std::stoi(argv[1]) : 10;
    if (nb < 1 || nb > 64) return 2;
    const int rows = 512, n = nb * QK_K, nt = 4;
    const size_t row_bytes = size_t(nb) * Q6K, act_bytes = size_t(nb) * 292;

    int guard_fail = 0;
    if (strata::kolibri::a1::unsupported_q6k(0, rows, (void*) 1, (void*) 1) == nullptr) ++guard_fail;
    if (strata::kolibri::a1::unsupported_q6k(nb, rows, nullptr, (void*) 1) == nullptr) ++guard_fail;
    if (strata::kolibri::a1::unsupported_q6k(nb, rows, (void*) 1, (void*) 1) != nullptr) ++guard_fail;
    if (guard_fail) { std::printf("shape guards: FAIL (%d)\n", guard_fail); return 1; }

    const ggml_type_traits_cpu* t8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    if (!t8 || !t8->from_float) { std::printf("ggml lacks the Q8_K quantiser\n"); return 2; }

    std::vector<uint8_t> w(size_t(rows) * row_bytes);
    for (size_t i = 0; i < w.size() / Q6K; ++i) make_q6k(&w[i * Q6K]);
    std::vector<uint8_t> act(size_t(nt) * act_bytes);
    std::vector<const void*> actp(nt);
    std::vector<float> xv(size_t(nt) * n);
    for (auto& v : xv) v = -1.f + 2.f * float(rnd() % 20001) / 10000.f;
    for (int t = 0; t < nt; ++t) {
        t8->from_float(xv.data() + size_t(t) * n, act.data() + size_t(t) * act_bytes, n);
        actp[t] = act.data() + size_t(t) * act_bytes;
    }

    std::vector<float> ref(size_t(nt) * rows), got(rows);
    for (int t = 0; t < nt; ++t) {
        float* o = ref.data() + size_t(t) * rows;
        for (int r = 0; r < rows; ++r)              // the AVX2 function the pool's trait points at
            ggml_vec_dot_q6_K_q8_K(n, &o[r], 0, w.data() + size_t(r) * row_bytes, 0, actp[t], 0, 1);
    }

    uint8_t *dw, *da; float* dout;
    cudaMalloc(&dw, w.size()); cudaMalloc(&da, act.size()); cudaMalloc(&dout, size_t(rows) * 4);
    cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice);

    long long match = 0, total = 0; double worst = 0;
    for (int t = 0; t < nt; ++t) {
        cudaMemcpy(da, act.data() + size_t(t) * act_bytes, act_bytes, cudaMemcpyHostToDevice);
        const int threads = 128, rows_per_block = (threads / 32) * strata::kolibri::a1::kRowsPerWarp;
        strata::kolibri::a1::kq_a1_dot_rows_q6k<<<(rows + rows_per_block - 1) / rows_per_block, threads>>>(
            dw, da, nb, rows, dout);
        if (cudaDeviceSynchronize() != cudaSuccess || cudaGetLastError() != cudaSuccess) return 2;
        cudaMemcpy(got.data(), dout, size_t(rows) * 4, cudaMemcpyDeviceToHost);
        const float* o = ref.data() + size_t(t) * rows;
        for (int r = 0; r < rows; ++r) {
            ++total;
            if (!std::isfinite(got[r])) { std::printf("non-finite at row %d\n", r); return 1; }
            if (std::memcmp(&got[r], &o[r], 4) == 0) ++match;
            else worst = std::fmax(worst, std::fabs(double(got[r]) - double(o[r])));
        }
    }
    std::printf("Q6_K nb=%-3d rows=%d tokens=%d: %lld/%lld bit-identical with ggml AVX2 (%.2f%%), worst |d| %.9g\n",
                nb, rows, nt, match, total, 100.0 * double(match) / double(total), worst);
    cudaFree(dw); cudaFree(da); cudaFree(dout);
    const bool ok = match == total;
    std::printf("Q6_K exactness: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
