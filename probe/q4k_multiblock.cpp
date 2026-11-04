// Multi-block oracle: instrument the REAL AVX2 source at its FMA boundaries and compare, for every block,
// the eight integer lanes, the four minimum-correction lanes, both scale factors, and the final result bit
// for bit. Host-only, so it runs while the GPU is busy. It is the fast loop for A1: a mapping change costs a
// seconds-long host build instead of a CUDA rebuild.
//
// No copied AVX2 body: the intrinsic FMA is wrapped, kq_avx2.cpp is compiled in, and only the boundary values
// are recorded. That is what caught the missing minus sign in the minimum correction.
#include <immintrin.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cstring>

struct Call { float d; float v[8]; };
static Call g_hi[64], g_lo[64];
static int g_nhi = 0, g_nlo = 0;

static __m256 capture256(__m256 d, __m256 v, __m256 a) {
    if (g_nhi < 64) { g_hi[g_nhi].d = _mm_cvtss_f32(_mm256_castps256_ps128(d));
                      _mm256_storeu_ps(g_hi[g_nhi].v, v); ++g_nhi; }
    return _mm256_fmadd_ps(d, v, a);
}
static __m128 capture128(__m128 d, __m128 v, __m128 a) {
    if (g_nlo < 64) { g_lo[g_nlo].d = _mm_cvtss_f32(d);
                      _mm_storeu_ps(g_lo[g_nlo].v, v); ++g_nlo; }
    return _mm_fmadd_ps(d, v, a);
}
#define _mm256_fmadd_ps capture256
#define _mm_fmadd_ps capture128
#include "../src/kernels/cpu/kq_avx2.cpp"
#undef _mm256_fmadd_ps
#undef _mm_fmadd_ps

static uint64_t rng = 0x243f6a8885a308d3ULL;
static uint32_t random32() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return uint32_t(rng >> 32); }
static float tree8(const float* a) { return ((a[0]+a[4])+(a[2]+a[6]))+((a[1]+a[5])+(a[3]+a[7])); }
static float tree4(const float* a) { return (a[0]+a[2])+(a[1]+a[3]); }
static bool same(float a, float b) { return std::memcmp(&a, &b, 4) == 0; }

// Q8_K sums exactly as ggml does it, so the oracle does not depend on the quantiser.
static void fill_q8(block_q8_K& y) {
    y.d = (int(random32() % 20001) - 10000) * 1e-5f;
    for (auto& v : y.qs) v = int8_t(random32());
    for (int j = 0; j < 16; ++j) { int s = 0; for (int k = 0; k < 16; ++k) s += y.qs[16 * j + k]; y.bsums[j] = int16_t(s); }
}
static void make_q4k(block_q4_K& x) {
    auto* b = reinterpret_cast<uint8_t*>(&x);
    for (int i = 0; i < 144; ++i) b[i] = uint8_t(random32());
    // finite, positive, normal FP16: keeps the scale arithmetic in range and reproducible
    uint16_t hd = uint16_t(0x2000 + random32() % 0x1800), hm = uint16_t(0x2000 + random32() % 0x1800);
    std::memcpy(b, &hd, 2); std::memcpy(b + 2, &hm, 2);
}

static int g_quiet = 0;

// One trial at a block count: does the real kernel equal the scalar re-derivation in every lane and at the end?
static bool check(int nb, bool mutate_sign) {
    std::vector<block_q4_K> x(static_cast<std::size_t>(nb));
    std::vector<block_q8_K> y(static_cast<std::size_t>(nb));
    for (int i = 0; i < nb; ++i) { make_q4k(x[size_t(i)]); fill_q8(y[size_t(i)]); }
    std::vector<const void*> act(static_cast<std::size_t>(nb));
    for (int i = 0; i < nb; ++i) act[size_t(i)] = &y[size_t(i)];

    g_nhi = g_nlo = 0;
    float actual = 0.f;
    strata::kernels::cpu::q4k_dot(x.data(), nb, act.data(), 1, &actual);
    if (g_nhi != nb || g_nlo != nb) { std::printf("capture count %d/%d does not match nb %d\n", g_nhi, g_nlo, nb); return false; }

    float acc[8] = {0,0,0,0,0,0,0,0}, am[4] = {0,0,0,0};
    for (int i = 0; i < nb; ++i) {
        const block_q4_K& xb = x[size_t(i)];
        const block_q8_K& yb = y[size_t(i)];
        const auto* sb = reinterpret_cast<const uint8_t*>(&xb);
        uint16_t hd, hm;
        std::memcpy(&hd, sb, 2); std::memcpy(&hm, sb + 2, 2);
        const float xd = _cvtsh_ss(hd), xm = _cvtsh_ss(hm);
        int s[8], m[8];
        for (int j = 0; j < 4; ++j) { s[j] = xb.scales[j] & 63; m[j] = xb.scales[j + 4] & 63; }
        for (int j = 4; j < 8; ++j) {
            s[j] = (xb.scales[j + 4] & 15) | ((xb.scales[j - 4] >> 6) << 4);
            m[j] = (xb.scales[j + 4] >> 4) | ((xb.scales[j] >> 6) << 4);
        }
        int sumi[8] = {}, prod[4] = {};
        for (int j = 0; j < 8; ++j)
            for (int k = 0; k < 32; ++k) {
                const int q = (xb.qs[32 * (j / 2) + k] >> (4 * (j % 2))) & 15;
                sumi[k / 4] += s[j] * q * int(yb.qs[32 * j + k]);
            }
        for (int k = 0; k < 4; ++k)
            prod[k] = m[2 * k] * (int(yb.bsums[4 * k]) + yb.bsums[4 * k + 1]) +
                      m[2 * k + 1] * (int(yb.bsums[4 * k + 2]) + yb.bsums[4 * k + 3]);

        // every integer lane, exactly
        for (int k = 0; k < 8; ++k) if (int(g_hi[i].v[k]) != sumi[k]) { if (!g_quiet) std::printf("block %d lane %d: %g vs %d\n", i, k, g_hi[i].v[k], sumi[k]); return false; }
        for (int k = 0; k < 4; ++k) if (int(g_lo[i].v[k]) != prod[k]) { if (!g_quiet) std::printf("block %d min lane %d: %g vs %d\n", i, k, g_lo[i].v[k], prod[k]); return false; }
        // both scale factors, bit for bit
        if (!same(g_hi[i].d, yb.d * xd)) { if (!g_quiet) std::printf("block %d: d differs\n", i); return false; }
        const float want_dmin = mutate_sign ? (yb.d * xm) : (-yb.d * xm);
        if (!same(g_lo[i].d, want_dmin)) { if (!g_quiet) std::printf("block %d: dmin sign/value differs\n", i); return false; }

        for (int k = 0; k < 8; ++k) acc[k] = std::fma(yb.d * xd, float(sumi[k]), acc[k]);
        for (int k = 0; k < 4; ++k) am[k] = std::fma(-yb.d * xm, float(prod[k]), am[k]);
    }
    return same(tree8(acc) + tree4(am), actual);
}

int main(int argc, char** argv) {
    const bool mutation = argc > 1 && std::strcmp(argv[1], "--mutation") == 0;
    const int trials = 2048;
    int failures = 0;
    for (int nb : {1, 2, 10, 32}) {
        int bad = 0;
        for (int t = 0; t < trials; ++t) if (!check(nb, false)) ++bad;
        std::printf("nb=%-3d %4d trials: %s\n", nb, trials, bad ? "FAIL" : "100% bit-identical (lanes, scales, result)");
        failures += bad;
    }
    // The oracle must still catch the defect it was built for: the positive minimum correction.
    g_quiet = 1;
    int caught = 0;
    for (int t = 0; t < 64; ++t) if (!check(4, true)) ++caught;
    g_quiet = 0;
    // …and it must not fire on the correct sign, or it would "detect" everything.
    g_quiet = 1;
    int false_alarm = 0;
    for (int t = 0; t < 64; ++t) if (!check(4, false)) ++false_alarm;
    g_quiet = 0;
    std::printf("sign mutation: %d/64 detected; correct sign: %d/64 clean\n", caught, 64 - false_alarm);
    if (caught != 64 || false_alarm != 0) { std::printf("the oracle's discrimination is broken\n"); return 2; }
    if (mutation) return 0;
    return failures ? 1 : 0;
}
