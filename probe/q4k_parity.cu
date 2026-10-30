// Bit-exact Q4_K x Q8_K port probe: the pool's AVX2 kernel versus a CUDA emulation of its exact order.
// Reference: strata::kernels::cpu::kq256_rows(12, ...) -> dot_rows -> q4k_dot (src/kernels/cpu/kq_avx2.cpp:85).
// The CUDA kernel reproduces q4k_dot's structure: per 256-block exact int32 per 32-value sub-block (lane),
// acc[lane] += d * (float)sumi[lane] as an fma, then hsum_float_8's tree and the acc_m tree.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "strata/kernels/cpu/kq_avx2.hpp"
#include "ggml-cpu.h"

using namespace strata::kernels::cpu;
static constexpr int QK_K = 256, QK8 = 256;

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

// A valid Q4_K super-block: d, dmin (fp16), 12 scale bytes, 128 nibble bytes.
static void make_q4k(uint8_t* b) {
    for (int i = 0; i < 144; ++i) b[i] = uint8_t(rnd());
    const float d = 0.004f + float(rnd() % 1000) * 2e-5f;
    const float dmin = -(0.004f + float(rnd() % 1000) * 2e-5f);
    const uint16_t hd = f2h(d), hm = f2h(dmin);
    std::memcpy(b, &hd, 2); std::memcpy(b + 2, &hm, 2);
    for (int s = 0; s < 12; ++s) b[4 + s] = uint8_t(rnd() & 0x3f);
}

__device__ __forceinline__ float hsum8(float a0,float a1,float a2,float a3,float a4,float a5,float a6,float a7) {
    // hsum_float_8's tree: (hi128 + lo128) elementwise, then pairwise within 4 lanes
    const float b0 = a4 + a0, b1 = a5 + a1, b2 = a6 + a2, b3 = a7 + a3;   // _mm256_extractf128 + add
    const float c0 = b2 + b0, c1 = b3 + b1;                                 // _mm_add_ps(res, _mm_movehl_ps(res,res))
    return c1 + c0;                                                         // _mm_add_ss(res, _mm_movehdup_ps(res))
}

// One row x one token. `w` = the row's nb super-blocks; `a` = the token's Q8_K blocks.
__global__ void k_q4k_dot(const uint8_t* __restrict__ w, const uint8_t* __restrict__ a, int nb, float* out, int rows) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const uint8_t* x = w + (size_t) r * nb * 144;
    float acc[8] = {0,0,0,0,0,0,0,0};
    float am[4] = {0,0,0,0};
    for (int i = 0; i < nb; ++i) {
        const uint8_t* xb = x + (size_t) i * 144;
        uint16_t hd, hm; std::memcpy(&hd, xb, 2); std::memcpy(&hm, xb + 2, 2);
        const float xd = __half2float(*(__half*)&hd), xdmin = __half2float(*(__half*)&hm);
        // 6-bit scales/mins, same unpacking as scale_min_k4 / the AVX2 kmask dance
        int sc[8], mn[8];
        for (int j = 0; j < 8; ++j) {
            if (j < 4) { sc[j] = xb[4 + j] & 63; mn[j] = xb[4 + 4 + j] & 63; }
            else { sc[j] = (xb[4 + j + 4] & 0x0F) | ((xb[4 + j - 4] >> 6) << 4);
                   mn[j] = (xb[4 + j + 4] >> 4) | ((xb[4 + j] >> 6) << 4); }
        }
        const uint8_t* qs = xb + 16;
        // Q8_K block: float d; int8 qs[256]; int16 bsums[16]
        const float yd = *(const float*)(a + (size_t) i * 292);
        const int8_t* yq = (const int8_t*)(a + (size_t) i * 292 + 4);
        const int16_t* ybs = (const int16_t*)(a + (size_t) i * 292 + 4 + 256);
        const float d = yd * xd, dmin = -yd * xdmin;
        // acc_m: 4 lanes over pairs of sub-blocks (mins . bsums)
        for (int lane = 0; lane < 4; ++lane) {
            const int p = mn[2*lane] * int(ybs[2*lane]) + mn[2*lane+1] * int(ybs[2*lane+1]);
            am[lane] = fmaf(dmin, (float)p, am[lane]);
        }
        for (int j = 0; j < 8; ++j) {                 // sub-block of 32 values -> lane j
            // ggml's q4_K layout: for 64-value group g = j/2, byte qs[32g + l] holds value (64g + l) in its low
            // nibble and value (64g + 32 + l) in its high nibble. Sub-block j is the (j%2)-th half of group g.
            const int g = j >> 1, hi = j & 1;
            int32_t s = 0;
            for (int l = 0; l < 32; ++l) {
                const int b = qs[32 * g + l];
                const int nib = hi ? (b >> 4) : (b & 0xF);
                s += nib * int(yq[64 * g + 32 * hi + l]);
            }
            acc[j] = fmaf(d, (float)(s * sc[j]), acc[j]);
        }
    }
    float m0 = am[0], m1 = am[1], m2 = am[2], m3 = am[3];
    const float mh0 = m2 + m0, mh1 = m3 + m1;          // _mm_add_ps(m, movehl(m,m))
    const float msum = mh1 + mh0;                       // _mm_add_ss(m, movehdup(m))
    out[r] = hsum8(acc[0],acc[1],acc[2],acc[3],acc[4],acc[5],acc[6],acc[7]) + msum;
}

int main() {
    ggml_cpu_init();
    const ggml_type_traits_cpu* t8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    if (!t8 || !t8->from_float) { std::printf("no Q8_K quantiser in this ggml\n"); return 2; }
    const size_t act_bytes = 10 * 292;          // 2560 values as Q8_K: 10 blocks of 292 B
    const size_t gu_row = 10 * 144;             // 2560 values as Q4_K: 10 blocks of 144 B
    const int rows = 512, n = 2560, nt = 4, nb = n / QK_K;
    std::printf("geometry: gu_row %zu act_bytes %zu\n", gu_row, act_bytes); std::fflush(stdout);

    std::vector<uint8_t> w(size_t(rows) * nb * 144);
    for (size_t i = 0; i < w.size() / 144; ++i) make_q4k(&w[i * 144]);
    std::printf("stage: blob ok\n"); std::fflush(stdout);

    std::vector<float> xv(size_t(nt) * n);
    for (auto& v : xv) v = -1.f + 2.f * float(rnd() % 20001) / 10000.f;
    std::vector<uint8_t> act(size_t(nt) * act_bytes);
    std::printf("stage: xv ok\n"); std::fflush(stdout);
    std::vector<const void*> actp(nt);
    for (int t = 0; t < nt; ++t) { std::printf("stage: quantise t=%d\n", t); std::fflush(stdout); t8->from_float(xv.data() + size_t(t) * n, act.data() + size_t(t) * act_bytes, n); actp[t] = act.data() + size_t(t) * act_bytes; }

    std::vector<float> ref(size_t(nt) * rows);
    std::vector<float*> outp(nt);
    for (int t = 0; t < nt; ++t) outp[t] = ref.data() + size_t(t) * rows;
    std::printf("stage: before kq256_rows\n"); std::fflush(stdout);
    auto t0 = std::chrono::steady_clock::now();
    kq256_rows(12, w.data(), gu_row, n, actp.data(), nt, outp.data(), 0, rows);
    auto t1 = std::chrono::steady_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::printf("stage: ref done\n"); std::fflush(stdout);
    // GPU
    uint8_t *dw, *da; float* dout;
    cudaMalloc(&dw, w.size()); cudaMalloc(&da, act.size()); cudaMalloc(&dout, size_t(rows) * 4);
    cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice);
    std::vector<float> got; got.resize((size_t)rows);
    long long match = 0, total = 0; double worst = 0; int worst_r = -1;
    double gpu_ms = 0;
    for (int t = 0; t < nt; ++t) {
        cudaMemcpy(da, act.data() + size_t(t) * act_bytes, act_bytes, cudaMemcpyHostToDevice);
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0);
        k_q4k_dot<<<(rows + 127) / 128, 128>>>(dw, da, nb, dout, rows);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1); gpu_ms += ms;
        cudaMemcpy(got.data(), dout, size_t(rows) * 4, cudaMemcpyDeviceToHost);
        for (int r = 0; r < rows; ++r) {
            ++total;
            if (std::memcmp(&got[r], &outp[t][r], 4) == 0) ++match;
            else { const double d = std::fabs(double(got[r]) - double(outp[t][r])); if (d > worst) { worst = d; worst_r = r; } }
        }
    }
    std::printf("bitwise: %lld of %lld match (%.2f%%), worst |diff| %.6g at row %d\n", match, total,
                100.0 * match / total, worst, worst_r);
    std::printf("time: CPU kq256_rows %.2f ms for %d rows x %d tokens (%.1f GFLOP/s); GPU %.3f ms\n",
                cpu_ms, rows, nt, 2.0 * rows * nt * n / (cpu_ms / 1000) / 1e9, gpu_ms);
    return 0;
}
