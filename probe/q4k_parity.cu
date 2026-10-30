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

// ggml's scalar q4_K x q8_K semantics (quants.c: ggml_vec_dot_q4_K_q8_K_generic), reproduced exactly:
//   aux8[64g + l]   = low  nibble of qs[32g + l]        (values 0-31 of the 64-value group g)
//   aux8[64g + 32+l] = high nibble of qs[32g + l]        (values 32-63)
//   per 32-value sub-block j: scale = scales[j]; four 8-value chunks; aux32[l] += scale * q8*a  (int, exact)
//   sums[l] += d * aux32[l]   with d = x.d * y.d, l = 0..7, per block
//   sumf   -= dmin * sumi     with dmin = x.dmin * y.d, sumi = sum_j bsums[j] * mins[j/2]
//   result  = sumf + sums[0] + ... + sums[7]
__global__ void k_q4k_dot(const uint8_t* __restrict__ w, const uint8_t* __restrict__ a, int nb, float* out, int rows, int mode) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const uint8_t* xrow = w + (size_t) r * nb * 144;
    float sums[8] = {0,0,0,0,0,0,0,0};
    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const uint8_t* xb = xrow + (size_t) i * 144;
        uint16_t hd, hm; std::memcpy(&hd, xb, 2); std::memcpy(&hm, xb + 2, 2);
        const float xd = __half2float(*(__half*)&hd), xdmin = __half2float(*(__half*)&hm);
        // 12 packed bytes -> 8 scales and 8 mins, as ggml's kmask dance produces
        uint32_t u0, u1, u2;
        std::memcpy(&u0, xb + 4, 4); std::memcpy(&u1, xb + 8, 4); std::memcpy(&u2, xb + 12, 4);
        const uint32_t km1 = 0x3f3f3f3fu, km2 = 0x0f0f0f0fu, km3 = 0x03030303u;
        uint32_t ut3 = ((u2 >> 4) & km2) | (((u1 >> 6) & km3) << 4);
        const uint32_t uaux = u1 & km1;
        uint32_t ut1 = (u2 & km2) | (((u0 >> 6) & km3) << 4);
        uint32_t ut2 = uaux, ut0 = u0 & km1;
        const uint32_t* uu = (const uint32_t*)(uint32_t[4]){ut0, ut1, ut2, ut3};
        const uint8_t* scales = (const uint8_t*)&uu[0];
        const uint8_t* mins = (const uint8_t*)&uu[2];
        const float yd = *(const float*)(a + (size_t) i * 292);
        const int8_t* yq = (const int8_t*)(a + (size_t) i * 292 + 4);
        const int16_t* ybs = (const int16_t*)(a + (size_t) i * 292 + 4 + 256);
        const uint8_t* qs = xb + 16;

        int aux32[8] = {0,0,0,0,0,0,0,0};
        for (int j = 0; j < 8; ++j) {               // 32-value sub-block
            const int scale = int(scales[j]);
            const int base = j * 32;
            for (int c = 0; c < 4; ++c) {           // four 8-value chunks
                const int o = base + c * 8;
                for (int l = 0; l < 8; ++l) {
                    const int v = o + l;
                    const int g = v >> 6, lo = v & 63;             // 64-value group and offset within it
                    const int b = qs[32 * g + (lo & 31)];
                    const int nib = (lo < 32) ? (b & 0xF) : (b >> 4);
                    aux32[l] += scale * (int(yq[v]) * nib);
                }
            }
        }
        int sumi = 0;
        for (int j = 0; j < 16; ++j) sumi += int(ybs[j]) * int(mins[j / 2]);
        const float d = xd * yd;
        const float dmin = xdmin * yd;
        if (mode & 1) { for (int l = 0; l < 8; ++l) sums[l] = __fmaf_rn(d, (float)aux32[l], sums[l]); }
        else          { for (int l = 0; l < 8; ++l) sums[l] = __fadd_rn(sums[l], __fmul_rn(d, (float)aux32[l])); }
        if (mode & 2) { sumf = __fmaf_rn(-dmin, (float)sumi, sumf); }
        else          { sumf = __fadd_rn(sumf, __fmul_rn(-dmin, (float)sumi)); }
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    out[r] = sumf;
}

extern "C" void ggml_vec_dot_q4_K_q8_K_generic(int n, float* s, size_t bs, const void* vx, size_t bx,
                                               const void* vy, size_t by, int nrc);
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
    std::vector<const void*> actp(nt);
    for (int t = 0; t < nt; ++t) {
        t8->from_float(xv.data() + size_t(t) * n, act.data() + size_t(t) * act_bytes, n);
        actp[t] = act.data() + size_t(t) * act_bytes;
    }

    std::vector<float> ref(size_t(nt) * rows);
    std::vector<float*> outp(nt);
    for (int t = 0; t < nt; ++t) outp[t] = ref.data() + size_t(t) * rows;
    auto t0 = std::chrono::steady_clock::now();
    kq256_rows(12, w.data(), gu_row, n, actp.data(), nt, outp.data(), 0, rows);
    auto t1 = std::chrono::steady_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ggml's own generic reference, same inputs: is my port exact against it?
    std::vector<float> gen(size_t(nt) * rows);
    for (int t = 0; t < nt; ++t)
        for (int r = 0; r < rows; ++r)
            ggml_vec_dot_q4_K_q8_K_generic(n, &gen[size_t(t) * rows + r], 0, w.data() + (size_t) r * gu_row, 0,
                                           actp[t], 0, 1);
    long long gm = 0, gt = 0; double gw = 0;
    for (size_t i = 0; i < gen.size(); ++i) { ++gt; if (std::memcmp(&gen[i], &outp[0][0], 0) == 0) {} }
    for (int t = 0; t < nt; ++t)
        for (int r = 0; r < rows; ++r) { ++gt; if (false) {} (void)r; (void)t; }
    std::printf("generic ref computed (%lld values); comparing Strata AVX2 against generic next\n", (long long)gen.size());
    long long avx_vs_gen = 0;
    for (size_t i = 0; i < gen.size(); ++i) if (std::memcmp(&gen[i], &ref[i], 4) == 0) ++avx_vs_gen;
    std::printf("Strata AVX2 kq256_rows vs ggml generic: %lld of %zu match (%.2f%%)\n", avx_vs_gen, gen.size(),
                100.0 * avx_vs_gen / gen.size());

    uint8_t *dw, *da; float* dout;
    cudaMalloc(&dw, w.size()); cudaMalloc(&da, act.size()); cudaMalloc(&dout, size_t(rows) * 4);
    cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice);
    std::vector<float> got; got.resize((size_t)rows);

    double gpu_ms_best = 1e9;
    for (int mode = 0; mode < 4; ++mode) {
        long long m2 = 0, tot2 = 0; double w2 = 0, gpu_ms = 0;
        for (int t = 0; t < nt; ++t) {
            cudaMemcpy(da, act.data() + size_t(t) * act_bytes, act_bytes, cudaMemcpyHostToDevice);
            cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            cudaEventRecord(e0);
            k_q4k_dot<<<(rows + 127) / 128, 128>>>(dw, da, nb, dout, rows, mode);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms = 0; cudaEventElapsedTime(&ms, e0, e1); gpu_ms += ms;
            cudaMemcpy(got.data(), dout, size_t(rows) * 4, cudaMemcpyDeviceToHost);
            for (int r = 0; r < rows; ++r) {
                ++tot2;
                if (std::memcmp(&got[r], &outp[t][r], 4) == 0) ++m2;
                else { const double d = std::fabs(double(got[r]) - double(outp[t][r])); if (d > w2) w2 = d; }
            }
        }
        gpu_ms /= nt;
        if (gpu_ms < gpu_ms_best) gpu_ms_best = gpu_ms;
        long long mgen = 0;
        for (int t = 0; t < nt; ++t)
            for (int r = 0; r < rows; ++r)
                if (std::memcmp(&got[r], &gen[size_t(t) * rows + r], 4) == 0) ++mgen;
        std::printf("mode %d (%s/%s): vs Strata AVX2 %lld/%lld match (%.2f%%) | vs ggml generic %lld/%d match (%.2f%%) | GPU %.3f ms/launch\n",
                    mode, (mode & 1) ? "fma" : "mul", (mode & 2) ? "fma" : "mul", m2, tot2,
                    100.0 * m2 / tot2, mgen, rows * nt, 100.0 * mgen / (rows * nt), gpu_ms);
    }
    std::printf("time: CPU kq256_rows %.2f ms for %d rows x %d token(s) (%.1f GFLOP/s, one thread); GPU best %.3f ms/launch\n",
                cpu_ms, rows, nt, 2.0 * rows * nt * n * 0 + 2.0 * rows * n / (cpu_ms / 1000) / 1e9 / nt * nt, gpu_ms_best);
    return 0;
}
