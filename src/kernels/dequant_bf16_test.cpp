// src/kernels/dequant_bf16_test.cpp - plan v0.3 P5: the device dequantizers against the artifact's validated CPU
// dequantizers (`strata/artifact/dequant.hpp`), on one real tensor of every type the model uses.
//
//     dequant_bf16_test SHARD1.gguf [SHARD2.gguf]
//
// For each type: the first 4 rows of the first tensor of that type, FP32 path compared with a relative tolerance of
// 1e-6 (the products are the reference's, reordered at most), BF16 path within half a BF16 ulp of the CPU value.
#include "strata/artifact/dequant.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/dequant_bf16.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

bool cpu_block(int type, const uint8_t* b, float* out) {
    using namespace strata;
    switch (type) {
    case 2: dequantize_q4_0(b, out); return true;
    case 6: dequantize_q5_0(b, out); return true;
    case 7: dequantize_q5_1(b, out); return true;
    case 8: dequantize_q8_0(b, out); return true;
    case 11: dequantize_q3_K(b, out); return true;
    case 12: dequantize_q4_K(b, out); return true;
    case 13: dequantize_q5_K(b, out); return true;
    case 14: dequantize_q6_K(b, out); return true;
    case 20: dequantize_iq4_nl(b, out); return true;
    case 23: dequantize_iq4_xs(b, out); return true;
    case 42: dequantize_q2_0(b, out); return true;
    default: return false;
    }
}

float bf16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// ---- host round-to-nearest-even float32 -> float16 -------------------------------------------------------------
// Written from the IEEE 754 rule, not from the device code: the point of the fp16 conversion test is to have a
// reference that does not share an implementation with `__float2half_rn`. `rne_shift` rounds a fixed-point value
// right by `shift` bits, half to even.
uint32_t rne_shift(uint64_t v, int shift) {
    if (shift <= 0) return (uint32_t) (v << (unsigned) (-shift));
    const uint64_t half = 1ull << (shift - 1), mask = (1ull << shift) - 1;
    const uint64_t rem = v & mask;
    uint64_t q = v >> shift;
    if (rem > half || (rem == half && (q & 1u))) ++q;
    return (uint32_t) q;
}

uint16_t f32_to_f16_rne(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint16_t sign = (uint16_t) ((x >> 16) & 0x8000u);
    const uint32_t mag = x & 0x7fffffffu;
    if (mag > 0x7f800000u) return (uint16_t) (sign | 0x7c00u | 0x200u);   // NaN stays NaN
    if (mag >= 0x47800000u) return (uint16_t) (sign | 0x7c00u);          // >= 65536 overflows
    const uint32_t e8 = mag >> 23, mant = mag & 0x7fffffu;
    if (mag < 0x38800000u) {                                             // below 2^-14: subnormal or zero
        const uint64_t sig = e8 ? (0x800000ull | mant) : (uint64_t) mant;
        const int exp2 = e8 ? (int) e8 - 150 : -149;                     // value = sig * 2^exp2
        const uint32_t n = rne_shift(sig, -(exp2 + 24));                 // in units of 2^-24
        return (uint16_t) (sign | n);                                    // 0x0400 is the smallest normal
    }
    int e16 = (int) e8 - 112;                                            // the result's fp16 exponent
    uint32_t m = rne_shift(mant, 13);                                    // 23 -> 10 fraction bits, half to even
    if (m == 0x400u) { m = 0; ++e16; }
    if (e16 >= 31) return (uint16_t) (sign | 0x7c00u);
    return (uint16_t) (sign | ((uint32_t) e16 << 10) | m);
}

float f16_to_f32(uint16_t h) {                                           // exact; used to count inexact conversions
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    if (e == 0 && m) {                                                   // subnormal: m * 2^-24, exact in fp32
        const float v = (float) m * 5.9604644775390625e-8f;
        return (h & 0x8000u) ? -v : v;
    }
    uint32_t bits;
    if (e == 0) bits = sign;                                             // +-0
    else if (e == 31) bits = sign | 0x7f800000u | (m << 13);             // inf / NaN
    else bits = sign | ((e - 15 + 127) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// splitmix64: the same fixtures on every platform and compiler
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        uint64_t z = (s += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }
    uint8_t u8() { return (uint8_t) (next() >> 56); }
};

void put_u16(uint8_t* p, uint16_t v) {
    p[0] = (uint8_t) (v & 0xffu);
    p[1] = (uint8_t) (v >> 8);
}

}  // namespace

// The selftest: synthetic Q4_K and Q6_K blocks, the artifact's independent CPU decoder as the reference, and the
// host RNE conversion above. Exit 0 = every covered case agreed, 1 = an assertion failed. It cannot establish
// end-to-end fidelity; it can establish that the conversion does what it claims on covered inputs, and that it was
// actually exercised (rounded, subnormal and zero cases are counted, and zero coverage is a failure).
int selftest() {
    struct Expected {
        float in;
        uint16_t bits;
        const char* why;
    };
    // the halfway cases are the ones a truncating or a round-half-away conversion gets wrong
    const Expected table[] = {
        {0.0f, 0x0000u, "+0"}, {1.0f, 0x3c00u, "1"}, {65504.0f, 0x7bffu, "max finite"},
        {65519.0f, 0x7bffu, "below the overflow tie"}, {65520.0f, 0x7c00u, "overflow tie rounds to inf"},
        {65536.0f, 0x7c00u, "overflow"},
        {5.9604644775390625e-8f, 0x0001u, "smallest subnormal"},
        {2.9802322387695312e-8f, 0x0000u, "half of the smallest subnormal ties to zero"},
        {8.9406967163085938e-8f, 0x0002u, "3/2 of the smallest subnormal ties to even"},
        {6.103515625e-5f, 0x0400u, "smallest normal"},
        {1.0009765625f, 0x3c01u, "one ulp above 1"},
        {1.00048828125f, 0x3c00u, "halfway to one ulp above 1 ties to even"},
        {1.00146484375f, 0x3c02u, "halfway between one and two ulp above 1 ties to even"},
    };
    int fails = 0;
    int table_bad = 0;
    for (const auto& e : table) {
        const uint16_t got = f32_to_f16_rne(e.in);
        if (got != e.bits) {
            ++table_bad;
            std::printf("  host RNE: %.9g -> 0x%04x, expected 0x%04x (%s)\n", (double) e.in, got, e.bits, e.why);
        }
    }
    if (f32_to_f16_rne(-0.0f) != 0x8000u) { ++table_bad; std::printf("  host RNE: -0 lost its sign\n"); }
    if (table_bad) { ++fails; std::printf("selftest: host RNE conversion table FAILED (%d case(s))\n", table_bad); }
    else std::printf("selftest: host RNE conversion table ok (%zu cases, halves and subnormals included)\n",
                     sizeof(table) / sizeof(table[0]));

    struct Fixture {
        const char* name;
        int type;
        int rows, cols;
        float d, dmin;   // written into the fp16 scale fields; 0.0f gives exact zeros
        bool quants_zero;
        uint64_t seed;
    };
    // The scales are deliberately NOT powers of two: with d a power of two every product is exact in fp16 and the
    // rounding path is never taken. 2^-20 pushes the products into the fp16 subnormal range.
    const Fixture fixtures[] = {
        {"Q4_K normal", 12, 4, 256, 0.0001234f, 0.0000371f, false, 11},
        {"Q4_K zeros", 12, 2, 256, 0.0f, 0.0f, true, 12},
        {"Q4_K subnormal", 12, 2, 256, 9.5367431640625e-7f, 0.0f, false, 13},
        {"Q6_K normal", 14, 4, 256, 0.0001234f, 0.0f, false, 21},
        {"Q6_K zeros", 14, 2, 256, 0.0f, 0.0f, true, 22},
        {"Q6_K subnormal", 14, 2, 256, 9.5367431640625e-7f, 0.0f, false, 23},
        {"Q4_K two blocks per row", 12, 3, 512, 0.0001234f, 0.0000371f, false, 31},
        {"Q6_K two blocks per row", 14, 3, 512, 0.0001234f, 0.0f, false, 41},
    };
    long long total_rounded = 0, total_subnormal = 0, total_values = 0;

    for (const auto& fx : fixtures) {
        const int be = 256, bb = fx.type == 12 ? 144 : 210;
        const int64_t row_bytes = (int64_t) fx.cols / be * bb;
        std::vector<uint8_t> host((size_t) (fx.rows * row_bytes));
        Rng rng(fx.seed);
        for (int64_t r = 0; r < fx.rows; ++r) {
            for (int64_t b = 0; b < fx.cols / be; ++b) {
                uint8_t* blk = host.data() + r * row_bytes + b * bb;
                for (int i = 0; i < bb; ++i) blk[i] = rng.u8();
                if (fx.quants_zero) {
                    const int from = fx.type == 12 ? 16 : 0, to = fx.type == 12 ? 144 : 192;   // qs / ql+qh
                    for (int i = from; i < to; ++i) blk[i] = 0;
                }
                put_u16(blk + (fx.type == 12 ? 0 : 208), f32_to_f16_rne(fx.d));
                if (fx.type == 12) put_u16(blk + 2, f32_to_f16_rne(fx.dmin));
            }
        }
        std::vector<float> ref((size_t) (fx.rows * fx.cols));
        for (int64_t r = 0; r < fx.rows; ++r)
            for (int64_t b = 0; b < fx.cols / be; ++b) {
                const uint8_t* blk = host.data() + r * row_bytes + b * bb;
                strata::dequantize_q4_K(blk, ref.data() + r * fx.cols + b * be);
                if (fx.type == 14) strata::dequantize_q6_K(blk, ref.data() + r * fx.cols + b * be);
            }

        void* d_blocks = nullptr;
        float* d_f = nullptr;
        uint16_t* d_h = nullptr;
        if (cudaMalloc(&d_blocks, host.size()) != cudaSuccess ||
            cudaMalloc(&d_f, ref.size() * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&d_h, ref.size() * sizeof(uint16_t)) != cudaSuccess) {
            std::printf("selftest: %s FAILED - cudaMalloc\n", fx.name);
            ++fails;
            continue;
        }
        cudaMemcpy(d_blocks, host.data(), host.size(), cudaMemcpyHostToDevice);
        strata::kernels::dequant_f32(fx.type, d_blocks, 0, fx.rows, fx.cols, d_f, nullptr);
        strata::kernels::dequant_f16(fx.type, d_blocks, 0, fx.rows, fx.cols, d_h, nullptr);
        std::vector<float> gf(ref.size());
        std::vector<uint16_t> gh(ref.size());
        cudaMemcpy(gf.data(), d_f, ref.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(gh.data(), d_h, ref.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
        cudaFree(d_blocks);
        cudaFree(d_f);
        cudaFree(d_h);

        long long f32_bad = 0, h_vs_gpu_bad = 0, h_vs_cpu_bad = 0, rounded = 0, subnormal = 0, zeros = 0;
        double worst_rel = 0, worst_ulp = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            const double e = std::fabs((double) gf[i] - ref[i]);
            const double rel = e / (std::fabs((double) ref[i]) + 1e-30);
            if (std::fabs((double) gf[i] - ref[i]) > 1e-6 * std::fabs((double) ref[i]) + 1e-30) ++f32_bad;
            worst_rel = (std::max)(worst_rel, rel);
            const uint16_t want_gpu = f32_to_f16_rne(gf[i]);
            const uint16_t want_cpu = f32_to_f16_rne(ref[i]);
            if (gh[i] != want_gpu) ++h_vs_gpu_bad;
            if (gh[i] != want_cpu) ++h_vs_cpu_bad;
            const double ulp = std::fabs((double) (int) gh[i] - (double) (int) want_gpu);
            worst_ulp = (std::max)(worst_ulp, ulp);
            if (f16_to_f32(want_cpu) != ref[i]) ++rounded;                  // the conversion was not exact
            if ((want_cpu & 0x7c00u) == 0 && (want_cpu & 0x3ffu) != 0) ++subnormal;
            if (ref[i] == 0.0f) ++zeros;
        }
        total_rounded += rounded;
        total_subnormal += subnormal;
        total_values += (long long) ref.size();
        const bool ok = f32_bad == 0 && h_vs_gpu_bad == 0;
        if (!ok) ++fails;
        std::printf("selftest: %-26s %d x %d  f32 vs CPU %s (%lld off, worst rel %.2e);  "
                    "f16 vs RNE(GPU f32) %s (%lld off, worst %g ulp);  f16 vs RNE(CPU f32) %lld off;  "
                    "rounded %lld, subnormal %lld, zeros %lld\n",
                    fx.name, fx.rows, fx.cols, f32_bad ? "FAIL" : "ok", f32_bad, worst_rel,
                    h_vs_gpu_bad ? "FAIL" : "ok", h_vs_gpu_bad, worst_ulp, h_vs_cpu_bad, rounded, subnormal,
                    zeros);
        if (fx.quants_zero && fx.d == 0.0f && zeros != (long long) ref.size()) {
            std::printf("selftest: %s FAILED - a zero block did not produce all zeros\n", fx.name);
            ++fails;
        }
    }

    // Row slice and output stride: the caller's row0 and ld must be honoured, and the padding must stay untouched.
    {
        const int type = 14, rows = 4, cols = 256, bb = 210;   // Q6_K: d at 208
        const int64_t row_bytes = (int64_t) cols / 256 * bb;
        std::vector<uint8_t> host((size_t) (rows * row_bytes));
        Rng rng(77);
        for (auto& b : host) b = rng.u8();
        for (int r = 0; r < rows; ++r) put_u16(host.data() + r * row_bytes + 208, f32_to_f16_rne(0.0001234f));
        std::vector<float> ref((size_t) (rows * cols));
        for (int r = 0; r < rows; ++r) strata::dequantize_q6_K(host.data() + r * row_bytes, ref.data() + r * cols);

        const int ld = cols + 8;                       // a stride of whole 16-byte units, with 8 values of padding
        void* d_blocks = nullptr;
        uint16_t* d_h = nullptr;
        cudaMalloc(&d_blocks, host.size());
        cudaMalloc(&d_h, (size_t) (rows * ld) * sizeof(uint16_t));
        cudaMemcpy(d_blocks, host.data(), host.size(), cudaMemcpyHostToDevice);
        std::vector<uint16_t> sentinel((size_t) (rows * ld), 0xabcdu);
        cudaMemcpy(d_h, sentinel.data(), sentinel.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

        // slice: rows 1..2 only, from block row 1
        strata::kernels::dequant_f16(type, d_blocks, 1, 2, cols, d_h, nullptr);
        std::vector<uint16_t> slice((size_t) (2 * cols));
        cudaMemcpy(slice.data(), d_h, slice.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
        long long slice_bad = 0;
        for (size_t i = 0; i < slice.size(); ++i)
            if (slice[i] != f32_to_f16_rne(ref[cols + i])) ++slice_bad;   // ref row 1 onward, contiguous
        std::printf("selftest: %-26s row slice row0=1 rows=2  %s (%lld off)\n", "Q6_K", slice_bad ? "FAIL" : "ok",
                    slice_bad);
        if (slice_bad) ++fails;

        // stride: the whole tensor at ld, padding must remain the sentinel
        cudaMemcpy(d_h, sentinel.data(), sentinel.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
        const bool launched = strata::kernels::dequant_f16_ld(type, d_blocks, 0, rows, cols, ld, d_h, nullptr);
        std::vector<uint16_t> strided(sentinel.size());
        cudaMemcpy(strided.data(), d_h, strided.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
        long long stride_bad = 0, pad_bad = 0;
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < ld; ++c) {
                const uint16_t got = strided[(size_t) r * ld + c];
                if (c < cols) {
                    if (got != f32_to_f16_rne(ref[(size_t) r * cols + c])) ++stride_bad;
                } else if (got != 0xabcdu) {
                    ++pad_bad;
                }
            }
        cudaFree(d_blocks);
        cudaFree(d_h);
        const bool stride_ok = launched && stride_bad == 0 && pad_bad == 0;
        if (!stride_ok) ++fails;
        std::printf("selftest: %-26s ld=cols+8           %s (launched %d, %lld off, %lld padding value(s) written)\n",
                    "Q6_K", stride_ok ? "ok" : "FAIL", (int) launched, stride_bad, pad_bad);
    }

    // A type this test claims to cover must be supported: a missing registration is a failure, never a skip.
    for (int t : {12, 14})
        if (!strata::kernels::dequant_bf16_supported(t)) {
            std::printf("selftest: type %d is not registered with the device dequantizers - the coverage this test "
                        "claims is not there\n", t);
            ++fails;
        }

    if (total_rounded == 0 || total_subnormal == 0) {
        std::printf("selftest: the fixtures exercised no %s case, so passing would mean nothing\n",
                    total_rounded == 0 ? "rounding" : "subnormal");
        ++fails;
    }
    std::printf("selftest: %lld value(s) covered, %lld rounded, %lld subnormal -> %s\n", total_values, total_rounded,
                total_subnormal, fails ? "FAILED" : "OK");
    return fails ? 1 : 0;
}

int main(int argc, char** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "--selftest") == 0) return selftest();
    if (argc < 2) {
        std::fprintf(stderr, "usage: dequant_bf16_test SHARD.gguf [SHARD2.gguf]\n"
                             "       dequant_bf16_test --selftest\n");
        return 2;
    }
    std::map<int, bool> done;
    int fails = 0, checked = 0;
    for (int a = 1; a < argc; ++a) {
        strata::GgufFile gguf(argv[a]);
        for (const auto& t : gguf.tensors()) {
            const int type = (int) t.type;
            if (done.count(type) || !strata::kernels::dequant_bf16_supported(type) || t.shape.size() != 2) continue;
            int be = 0, bb = 0;
            if (!strata::block_geometry(t.type, be, bb)) continue;
            const int64_t cols = (int64_t) t.shape[0];
            const int64_t rows = (std::min<int64_t>)(4, (int64_t) t.shape[1]);
            const int64_t row_bytes = cols / be * bb;
            const uint8_t* host = (const uint8_t*) gguf.tensor_data(t);
            void* d_blocks = nullptr;
            float* d_f = nullptr;
            uint16_t* d_h = nullptr;
            cudaMalloc(&d_blocks, (size_t) (rows * row_bytes));
            cudaMalloc(&d_f, (size_t) (rows * cols) * 4);
            cudaMalloc(&d_h, (size_t) (rows * cols) * 2);
            cudaMemcpy(d_blocks, host, (size_t) (rows * row_bytes), cudaMemcpyHostToDevice);
            strata::kernels::dequant_f32(type, d_blocks, 0, rows, cols, d_f, nullptr);
            strata::kernels::dequant_bf16(type, d_blocks, 0, rows, cols, d_h, nullptr);
            std::vector<float> gf((size_t) (rows * cols));
            std::vector<uint16_t> gh((size_t) (rows * cols));
            cudaMemcpy(gf.data(), d_f, gf.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(gh.data(), d_h, gh.size() * 2, cudaMemcpyDeviceToHost);
            cudaFree(d_blocks);
            cudaFree(d_f);
            cudaFree(d_h);
            std::vector<float> ref((size_t) (rows * cols));
            for (int64_t r = 0; r < rows; ++r)
                for (int64_t b = 0; b < cols / be; ++b)
                    cpu_block(type, host + r * row_bytes + b * bb, ref.data() + r * cols + b * be);
            int64_t bad_f = 0, bad_h = 0;
            double worst = 0;
            for (size_t i = 0; i < ref.size(); ++i) {
                const double e = std::fabs((double) gf[i] - ref[i]);
                const double tol = 1e-6 * std::fabs((double) ref[i]) + 1e-30;
                if (e > tol) ++bad_f;
                worst = (std::max)(worst, e / (std::fabs((double) ref[i]) + 1e-30));
                const double eh = std::fabs((double) bf16_to_f32(gh[i]) - ref[i]);
                if (eh > std::ldexp(std::fabs((double) ref[i]), -8) + 1e-30) ++bad_h;
            }
            ++checked;
            done[type] = true;
            const bool ok = bad_f == 0 && bad_h == 0;
            if (!ok) ++fails;
            std::printf("type %2d  %-40s %lld x %lld: f32 %s (%lld off, worst rel %.2e), bf16 %s (%lld off)\n", type,
                        t.name.c_str(), (long long) rows, (long long) cols, bad_f ? "FAIL" : "ok", (long long) bad_f,
                        worst, bad_h ? "FAIL" : "ok", (long long) bad_h);
        }
    }
    std::printf("dequant_bf16_test: %d types checked, %s\n", checked, fails ? "FAILED" : "OK");
    return fails ? 1 : 0;
}
