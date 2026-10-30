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

// A valid Q4_K super-block: d, dmin (fp16), 12 scale bytes, 128 nibble bytes.
static void make_q4k(uint8_t* b) {
    for (int i = 0; i < 144; ++i) b[i] = uint8_t(rnd());
    const float d = 0.004f + float(rnd() % 1000) * 2e-5f;
    const float dmin = (rnd() & 1 ? -1.f : 1.f) * (0.004f + float(rnd() % 1000) * 2e-5f);
    const uint16_t hd = f2h(d), hm = f2h(dmin);
    std::memcpy(b, &hd, 2); std::memcpy(b + 2, &hm, 2);
    // Keep all bits of the random packed scales: high bits carry scales/mins 4..7.
}

// ggml's scalar q4_K x q8_K semantics (quants.c: ggml_vec_dot_q4_K_q8_K_generic), reproduced exactly:
//   aux8[64g + l]   = low  nibble of qs[32g + l]        (values 0-31 of the 64-value group g)
//   aux8[64g + 32+l] = high nibble of qs[32g + l]        (values 32-63)
//   per 32-value sub-block j: scale = scales[j]; four 8-value chunks; aux32[l] += scale * q8*a  (int, exact)
//   sums[l] += d * aux32[l]   with d = x.d * y.d, l = 0..7, per block
//   sumf   -= dmin * sumi     with dmin = x.dmin * y.d, sumi = sum_j bsums[j] * mins[j/2]
//   result  = sumf + sums[0] + ... + sums[7]
// Strata's AVX2 q4k_dot, reproduced exactly (src/kernels/cpu/kq_avx2.cpp:85-145):
//   s[0..7], mn[0..7] from the 12 packed bytes, as ggml's kmask unpacking produces them
//   sumi[k] (int32, 8 lanes) = sum over the four 64-value groups j of
//        s[2j]   * sum of the 4 values at 64j + 4k
//      + s[2j+1] * sum of the 4 values at 64j + 32 + 4k
//   acc[k] = fma(d, (float)sumi[k], acc[k])                        per block, d = x.d * y.d
//   am[i]  = fma(-x.dmin*y.d, (float)(mn[2i]*(B[4i]+B[4i+1]) + mn[2i+1]*(B[4i+2]+B[4i+3])), am[i])
//   result = hsum_float_8(acc) + ((am0+am2) + (am1+am3))
// with hsum_float_8(x) = ((x0+x4)+(x2+x6)) + ((x1+x5)+(x3+x7)).
__global__ void k_q4k_dot(const uint8_t* __restrict__ w, const uint8_t* __restrict__ a, int nb, float* out, int rows,
                          int mode) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const uint8_t* xrow = w + (size_t) r * nb * 144;
    float acc[8] = {0,0,0,0,0,0,0,0};
    float am[4] = {0,0,0,0};
    for (int i = 0; i < nb; ++i) {
        const uint8_t* xb = xrow + (size_t) i * 144;
        uint16_t hd, hm; std::memcpy(&hd, xb, 2); std::memcpy(&hm, xb + 2, 2);
        const float xd = __half2float(*(__half*)&hd), xdmin = __half2float(*(__half*)&hm);
        int s[8], mn[8];
        for (int j = 0; j < 4; ++j) { s[j] = xb[4 + j] & 63; mn[j] = xb[8 + j] & 63; }
        for (int j = 4; j < 8; ++j) {
            s[j] = (xb[8 + j] & 0x0F) | ((xb[j] >> 6) << 4);
            mn[j] = (xb[8 + j] >> 4) | ((xb[4 + j] >> 6) << 4);
        }
        const uint8_t* qs = xb + 16;
        const float yd = *(const float*)(a + (size_t) i * 292);
        const int8_t* yq = (const int8_t*)(a + (size_t) i * 292 + 4);
        const int16_t* ybs = (const int16_t*)(a + (size_t) i * 292 + 4 + 256);
        int32_t sumi[8] = {0,0,0,0,0,0,0,0};
        for (int j = 0; j < 4; ++j) {
            for (int half = 0; half < 2; ++half) {
                const int base = 64 * j + 32 * half, sc = s[2 * j + half];
                for (int k = 0; k < 8; ++k) {
                    int v = 0;
                    for (int t = 0; t < 4; ++t) {
                        const int idx = base + 4 * k + t;              // value index in the 256-block
                        const int byte = qs[32 * j + ((idx & 63) & 31)];   // one byte holds the low and high nibble pair
                        const int nib = ((idx & 63) < 32) ? (byte & 0xF) : (byte >> 4);
                        v += nib * int(yq[idx]);
                    }
                    sumi[k] += sc * v;                             // integer: exact, order-free
                }
            }
        }
        const float d = xd * yd, dmin = -yd * xdmin;
        for (int k = 0; k < 8; ++k) {
            if (mode & 1) acc[k] = __fmaf_rn(d, (float)sumi[k], acc[k]);
            else          acc[k] = __fadd_rn(acc[k], __fmul_rn(d, (float)sumi[k]));
        }
        for (int q = 0; q < 4; ++q) {
            const int prod = mn[2*q] * (int(ybs[4*q]) + int(ybs[4*q+1])) +
                             mn[2*q+1] * (int(ybs[4*q+2]) + int(ybs[4*q+3]));
            if (mode & 2) am[q] = __fmaf_rn(dmin, (float)prod, am[q]);
            else          am[q] = __fadd_rn(am[q], __fmul_rn(dmin, (float)prod));
        }
    }
    const float hsum = __fadd_rn(__fadd_rn(__fadd_rn(acc[0], acc[4]), __fadd_rn(acc[2], acc[6])),
                                 __fadd_rn(__fadd_rn(acc[1], acc[5]), __fadd_rn(acc[3], acc[7])));
    const float msum = __fadd_rn(__fadd_rn(am[0], am[2]), __fadd_rn(am[1], am[3]));
    out[r] = __fadd_rn(hsum, msum);
}

extern "C" void ggml_vec_dot_q4_K_q8_K_generic(int n, float* s, size_t bs, const void* vx, size_t bx,
                                               const void* vy, size_t by, int nrc);

int main(int argc, char** argv) {
    ggml_cpu_init();
    const ggml_type_traits_cpu* t8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    if (!t8 || !t8->from_float) { std::printf("no Q8_K quantiser in this ggml\n"); return 2; }
    const int nb = argc > 1 ? std::stoi(argv[1]) : 10;
    if (nb < 1 || nb > 64) return 2;
    const size_t act_bytes = size_t(nb) * 292;
    const size_t gu_row = size_t(nb) * 144;
    const int rows = 512, n = nb * QK_K, nt = 4;
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
    bool exact = false;
    for (int mode = 0; mode < 4; ++mode) {
        long long m2 = 0, tot2 = 0, mgen = 0; double w2 = 0, gpu_ms = 0;
        for (int t = 0; t < nt; ++t) {
            cudaMemcpy(da, act.data() + size_t(t) * act_bytes, act_bytes, cudaMemcpyHostToDevice);
            cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            cudaEventRecord(e0);
            k_q4k_dot<<<(rows + 127) / 128, 128>>>(dw, da, nb, dout, rows, mode);
            cudaEventRecord(e1);
            if (cudaEventSynchronize(e1) != cudaSuccess || cudaGetLastError() != cudaSuccess) return 2;
            float ms = 0; cudaEventElapsedTime(&ms, e0, e1); gpu_ms += ms;
            cudaMemcpy(got.data(), dout, size_t(rows) * 4, cudaMemcpyDeviceToHost);
            for (int r = 0; r < rows; ++r) {
                ++tot2;
                if (std::memcmp(&got[r], &gen[size_t(t) * rows + r], 4) == 0) ++mgen;
                if (std::memcmp(&got[r], &outp[t][r], 4) == 0) ++m2;
                else { const double d = std::fabs(double(got[r]) - double(outp[t][r])); if (d > w2) w2 = d; }
            }
        }
        gpu_ms /= nt;
        if (gpu_ms < gpu_ms_best) gpu_ms_best = gpu_ms;
        if (mode == 3) exact = m2 == tot2;
        std::printf("worst |dlogit|: %.9g\n", w2);
        std::printf("mode %d (%s/%s): vs Strata AVX2 %lld/%lld match (%.2f%%) | vs ggml generic %lld/%d match (%.2f%%) | GPU %.3f ms/launch\n",
                    mode, (mode & 1) ? "fma" : "mul", (mode & 2) ? "fma" : "mul", m2, tot2,
                    100.0 * m2 / tot2, mgen, rows * nt, 100.0 * mgen / (rows * nt), gpu_ms);
    }
    std::printf("time: CPU kq256_rows %.2f ms for %d rows x %d token(s) (%.1f GFLOP/s, one thread); GPU best %.3f ms/launch\n",
                cpu_ms, rows, nt, 2.0 * rows * nt * n * 0 + 2.0 * rows * n / (cpu_ms / 1000) / 1e9 / nt * nt, gpu_ms_best);
    std::printf("FMA/FMA exactness: %s\n", exact ? "PASS" : "FAIL");
    return exact ? 0 : 1;
}
