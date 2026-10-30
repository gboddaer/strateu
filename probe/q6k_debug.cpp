// One-block Q6_K debug: is my decode + scale mapping the same as ggml's dequantisation, and is the dot the same?
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_fp16.h>
#include "ggml-cpu.h"
#include "ggml.h"

extern "C" void ggml_vec_dot_q6_K_q8_K(int n, float* s, size_t bs, const void* vx, size_t bx,
                                       const void* vy, size_t by, int nrc);
static uint64_t rs = 0x9E3779B97F4A7C15ull;
static uint32_t rnd() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return uint32_t(rs >> 32); }

int main() {
    ggml_cpu_init();
    const int n = 256;
    std::vector<uint8_t> w(210), act(292);
    for (int i = 0; i < 208; ++i) w[i] = uint8_t(rnd());
    __half h = __float2half(0.01f);
    std::memcpy(&w[208], &h, 2);
    const auto* t8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    const auto* t6 = ggml_get_type_traits_cpu(GGML_TYPE_Q6_K);
    std::vector<float> xv(n);
    for (auto& v : xv) v = -1.f + 2.f * float(rnd() % 20001) / 10000.f;
    t8->from_float(xv.data(), act.data(), n);
    float ref = 0; ggml_vec_dot_q6_K_q8_K(n, &ref, 0, w.data(), 0, act.data(), 0, 1);

    std::vector<float> dq(n);
    ggml_get_type_traits(GGML_TYPE_Q6_K)->to_float(w.data(), dq.data(), n);
    const auto* yb = (const uint8_t*) act.data();
    const float yd = *(const float*) yb;
    const int8_t* yq = (const int8_t*) (yb + 4);
    const int16_t* ys = (const int16_t*) (yb + 4 + 256);

    const uint8_t* ql = w.data(); const uint8_t* qh = w.data() + 128;
    const int8_t* sc = (const int8_t*) (w.data() + 192);
    const float xd = __half2float(h);

    // 1. decode + scale, value by value
    int bad = 0;
    long long raw = 0;
    for (int j = 0; j < 2; ++j)
        for (int c = 0; c < 4; ++c)
            for (int l = 0; l < 32; ++l) {
                const int b = ql[64 * j + (c & 1) * 32 + l];
                const int low = (c < 2) ? (b & 0xF) : ((b >> 4) & 0xF);
                const int high = ((qh[32 * j + l] >> (2 * c)) & 3) << 4;
                const int q6 = (low | high) - 32;
                const int s = int(sc[8 * j + 2 * c + (l >= 16)]);
                const int idx = 128 * j + 32 * c + l;
                raw += (long long) s * q6 * int(yq[idx]);
                const double want = double(dq[idx]) / (double(xd) * double(s));
                if (std::fabs(double(q6) - want) > 1e-2) {
                    if (bad < 6) std::printf("idx %3d: mine %d, ggml %.4f (scale %d)\n", idx, q6, want, s);
                    ++bad;
                }
            }
    // 2. the dot, in float64, from ggml's own dequantised weights
    double exact = 0;
    for (int i = 0; i < n; ++i) exact += double(dq[i]) * double(yq[i]) * double(yd);
    std::printf("decode mismatches: %d of 256\nref(AVX2) = %.9g\nexact(float64) = %.9g\ndiff = %.6g\n",
                bad, ref, exact, double(ref) - exact);
    return 0;
}
