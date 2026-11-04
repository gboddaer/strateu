// tests/kolibri/f16_expert_gemm_parity.cu - the FP16 expert GEMM measured in three separate steps.
//
// WHAT THIS ANSWERS.  `STRATA_KOLIBRI_EXPERT_F16` turns an expert's Q4_K/Q6_K blocks into FP16 on the device
// (`dequant_f16`) and multiplies them with FP16 activations through `Gemm::f16_batched`
// (`cublasGemmBatchedEx`, A = W CUDA_R_16F, B = X CUDA_R_16F, C = Y CUDA_R_32F, compute CUBLAS_COMPUTE_32F,
// CUBLAS_GEMM_DEFAULT).  The engine-vs-engine KL of 0.400 (max |dlogit| 11.08) has to be either that arithmetic or
// something else, and this program separates the two components that the arithmetic contributes:
//
//   (a) Y_a = float64 dot of the UNROUNDED operands: CPU-dequantized f32 weights and the f32 activations
//   (b) Y_b = float64 dot of the operands actually sent: the FP16 weights and the FP16 activations, widened exactly
//   (c) Y_c = what the GPU produced
//
//   operand rounding  = ||Y_b - Y_a||   (inherent to storing weights and activations as FP16; reported, not gated)
//   accumulation      = ||Y_c - Y_b||   (order and f32 rounding inside cuBLAS; gated by gamma_K * sum|w*x|,
//                                        gamma_K = K*u/(1-K*u), u = 2^-24 for fp32)
//
// Both are reported per element against that element's own sum |w*x|, so a row that cancels heavily cannot hide
// behind a large-magnitude neighbour.  A pass says the FP16 expert GEMM behaves like fp32 accumulation over the
// rounded operands on these shapes; it says nothing about 50 layers of composition, routing, or KV state.
//
// MUTATIONS.  `--mutate swap|offset|rows` deliberately breaks the wiring to show the gate detects it.  They are
// flags rather than edits so the evidence can be re-produced:
//   swap   - two experts' weight pointers exchanged in the device array (a wrong-expert pointer)
//   offset - the Q6_K blob handed over shifted by one super-block (a corrupted block offset)
//   rows   - the reference covers T-1 rows while the GPU is asked for T (a missing/padded row)
//
// Exit 0 = gate passed, 1 = failed, 2 = the harness itself could not run (no GPU, no GEMM, non-finite fixture).
//
//   f16_expert_gemm_parity [--embd 2560] [--ff 512] [--rows 32] [--batch 3] [--mutate NAME]

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "strata/artifact/dequant.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/prefill/gemm.hpp"

namespace {

constexpr int kQ4K = 12;
constexpr int kQ6K = 14;

void check(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return;
    std::fprintf(stderr, "FATAL %s: %s\n", what, cudaGetErrorString(e));
    std::exit(2);
}

/// IEEE half, round-to-nearest-even, as bits.  Host-only, so the reference does not share the device's conversion.
uint16_t f2h(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint16_t sign = uint16_t((x >> 16) & 0x8000u);
    const uint32_t mag = x & 0x7fffffffu;
    if (mag > 0x7f800000u) return uint16_t(sign | 0x7c00u | 0x200u);
    if (mag >= 0x47800000u) return uint16_t(sign | 0x7c00u);
    const uint32_t e8 = mag >> 23, mant = mag & 0x7fffffu;
    if (mag < 0x38800000u) {                       // subnormal / zero
        const uint64_t sig = e8 ? (0x800000ull | mant) : uint64_t(mant);
        const int exp2 = e8 ? int(e8) - 150 : -149;
        const int sh = -(exp2 + 24);
        uint64_t q = sh > 0 ? (sig >> sh) : (sig << -sh);
        if (sh > 0) {
            const uint64_t half = 1ull << (sh - 1), rem = sig & ((1ull << sh) - 1);
            if (rem > half || (rem == half && (q & 1u))) ++q;
        }
        return uint16_t(sign | uint32_t(q));
    }
    int e16 = int(e8) - 112;
    uint32_t m = (mant >> 13) + (((mant >> 12) & 1u) && (((mant & 0xfffu) != 0) || ((mant >> 13) & 1u)) ? 1u : 0u);
    if (m == 0x400u) { m = 0; ++e16; }
    if (e16 >= 31) return uint16_t(sign | 0x7c00u);
    return uint16_t(sign | (uint32_t(e16) << 10) | m);
}

float h2f(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    if (e == 0 && m) {
        const float v = float(m) * 5.9604644775390625e-8f;
        return (h & 0x8000u) ? -v : v;
    }
    uint32_t bits;
    if (e == 0) bits = sign;
    else if (e == 31) bits = sign | 0x7f800000u | (m << 13);
    else bits = sign | ((e - 15 + 127) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

struct Shape {
    const char* name;
    int type;
    int64_t N, K;      // N = output rows of the weight (rows of W), K = reduction length
};

int64_t block_bytes(int type) { return type == kQ4K ? 144 : 210; }

/// A valid, finite block set: random payload, the scale fields overwritten with bounded values.
std::vector<uint8_t> make_blob(int type, int64_t N, int64_t K, uint64_t& rng) {
    const int64_t nblk = N * K / 256;
    std::vector<uint8_t> b(size_t(nblk * block_bytes(type)));
    auto next = [&rng] { rng = rng * 6364136223846793005ULL + 1442695040888963407ULL; return uint32_t(rng >> 33); };
    for (auto& v : b) v = uint8_t(next());
    for (int64_t blk = 0; blk < nblk; ++blk) {
        uint8_t* p = b.data() + blk * block_bytes(type);
        const float d = 0.004f + float(next() % 1000u) * 2e-5f;      // bounded, non-power-of-two
        if (type == kQ4K) {
            const float dmin = -(0.004f + float(next() % 1000u) * 2e-5f);
            const uint16_t hd = f2h(d), hm = f2h(dmin);
            std::memcpy(p, &hd, 2);
            std::memcpy(p + 2, &hm, 2);
            for (int s = 0; s < 12; ++s) p[4 + s] = uint8_t(next() & 0x3fu);      // 6-bit sub-scales
        } else {
            const uint16_t hd = f2h(d);
            std::memcpy(p + 208, &hd, 2);                                        // Q6_K: d is LAST
            for (int s = 0; s < 16; ++s) p[192 + s] = uint8_t(int(next() % 65u) - 32);   // 8-bit signed scales
        }
    }
    return b;
}

/// CPU dequantization of one whole matrix, row-major [N, K], through the artifact's decoder.
std::vector<float> cpu_dequant(int type, const std::vector<uint8_t>& blob, int64_t N, int64_t K) {
    std::vector<float> w(size_t(N * K));
    const int64_t bb = block_bytes(type);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t blk = 0; blk < K / 256; ++blk) {
            const uint8_t* p = blob.data() + (n * (K / 256) + blk) * bb;
            float* out = w.data() + n * K + blk * 256;
            if (type == kQ4K) strata::dequantize_q4_K(p, out);
            else strata::dequantize_q6_K(p, out);
        }
    return w;
}

struct Proj {
    Shape sh;
    std::vector<uint8_t> blob;
    std::vector<float> w32;        // CPU dequantized, unrounded
    std::vector<float> x32;        // the activations before rounding
    std::vector<double> ya, yb;    // (a) and (b), per expert, T x N
    std::vector<float> yc;         // (c), per expert, T x N
};

struct Result {
    int64_t T = 0, N = 0, K = 0;
    double max_operand_rel = 0, max_accum_abs = 0, max_accum_rel_to_bound = 0, max_y = 0;
    long long weight_bad = 0;     // the FP16 weights the GPU used are not the ones the reference computed
    double worst_sum_abs = 0;
    long long finite_ok = 1;
    long long wrong = 0;          // failures of the gate
};

/// gamma_K for fp32 accumulation: K*u/(1-K*u), u = 2^-24.
double gamma_k(int64_t K) {
    const double u = 5.9604644775390625e-8;      // 2^-24
    const double ku = double(K) * u;
    return ku / (1.0 - ku);
}

Result run_proj(const Shape& sh, int64_t T, int batch, const char* mutate) {
    Result r;
    r.T = T; r.N = sh.N; r.K = sh.K;
    uint64_t rng = 0x243F6A8885A308D3ULL ^ uint64_t(sh.N * 131 + sh.K * 7 + T * 17 + batch);
    Proj p;
    p.sh = sh;
    std::vector<std::vector<uint8_t>> hw(batch);
    for (int b = 0; b < batch; ++b) hw[b] = make_blob(sh.type, sh.N, sh.K, rng);
    std::vector<std::vector<float>> w(batch), x(batch);
    for (int b = 0; b < batch; ++b) {
        w[b] = cpu_dequant(sh.type, hw[b], sh.N, sh.K);
        x[b].resize(size_t(T * sh.K));
        for (auto& v : x[b]) {
            rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
            v = float(int64_t(rng >> 40) % 20001 - 10000) / 10000.0f;    // [-1, 1) with non-dyadic values
        }
    }

    // device buffers
    std::vector<void*> dW(batch), dWf16(batch), dX(batch), dY(batch);
    std::vector<uint16_t*> hXh(batch);
    for (int b = 0; b < batch; ++b) {
        const size_t wb = size_t(sh.N * sh.K / 256) * size_t(block_bytes(sh.type));
        const bool shift = std::strcmp(mutate, "offset") == 0;      // Q6_K read one super-block late
        check(cudaMalloc(&dW[b], wb + (shift ? size_t(block_bytes(sh.type)) : 0)), "malloc W");
        check(cudaMemset(dW[b], 0, wb + (shift ? size_t(block_bytes(sh.type)) : 0)), "zero W");
        check(cudaMemcpy(dW[b], hw[b].data(), wb, cudaMemcpyHostToDevice), "H2D W");
        // the GPU is pointed one super-block late while the reference keeps the true blob: an offset error that
        // cancels out if both sides are shifted, which is why the shift is applied to the pointer only
        void* dWp = (uint8_t*)dW[b] + (shift ? size_t(block_bytes(sh.type)) : 0);
        check(cudaMalloc(&dWf16[b], size_t(sh.N * sh.K) * 2), "malloc Wf16");
        hXh[b] = new uint16_t[size_t(T * sh.K)];
        for (int64_t i = 0; i < T * sh.K; ++i) hXh[b][i] = f2h(x[b][size_t(i)]);
        check(cudaMalloc(&dX[b], size_t(T * sh.K) * 2), "malloc X");
        check(cudaMemcpy(dX[b], hXh[b], size_t(T * sh.K) * 2, cudaMemcpyHostToDevice), "H2D X");
        check(cudaMalloc(&dY[b], size_t(T * sh.N) * 4), "malloc Y");
        check(cudaMemset(dY[b], 0, size_t(T * sh.N) * 4), "zero Y");
        // the engine's own conversion of the same blocks into FP16
        strata::kernels::dequant_f16(sh.type, dWp, 0, sh.N, sh.K, (uint16_t*)dWf16[b], nullptr);
    }
    check(cudaDeviceSynchronize(), "convert sync");

    // (b) needs the FP16 weights the device actually produced, not the host's idea of them
    std::vector<std::vector<uint16_t>> w16(batch);
    for (int b = 0; b < batch; ++b) {
        w16[b].resize(size_t(sh.N * sh.K));
        check(cudaMemcpy(w16[b].data(), dWf16[b], size_t(sh.N * sh.K) * 2, cudaMemcpyDeviceToHost), "D2H Wf16");
    }

    // THE WIRING CHECK.  The (b)/(c) comparison above derives its reference from the weights the device produced, so
    // by itself it is blind to a wrong-expert pointer, a shifted block offset or a wrong conversion: those move both
    // sides together.  Compare the device's FP16 bits against the host's RNE conversion of the CPU-dequantized block -
    // this is what notices that the GPU is multiplying numbers the reference never saw.
    for (int b = 0; b < batch; ++b)
        for (int64_t i = 0; i < sh.N * sh.K; ++i)
            if (w16[b][size_t(i)] != f2h(w[b][size_t(i)])) ++r.weight_bad;

    std::vector<const uint16_t*> hWa(batch), hXa(batch);
    std::vector<float*> hYa(batch);
    for (int b = 0; b < batch; ++b) {
        hWa[b] = (const uint16_t*)dWf16[b];
        hXa[b] = (const uint16_t*)dX[b];
        hYa[b] = (float*)dY[b];
    }
    if (std::strcmp(mutate, "swap") == 0 && batch >= 2)
        std::swap(hWa[0], hWa[1]);                 // two experts' weights exchanged: a wrong-expert pointer

    const uint16_t* const* dWa; const uint16_t* const* dXa; float* const* dYa;
    check(cudaMalloc((void**)&dWa, size_t(batch) * 8), "malloc Wa");
    check(cudaMalloc((void**)&dXa, size_t(batch) * 8), "malloc Xa");
    check(cudaMalloc((void**)&dYa, size_t(batch) * 8), "malloc Ya");
    check(cudaMemcpy((void*)dWa, hWa.data(), size_t(batch) * 8, cudaMemcpyHostToDevice), "H2D Wa");
    check(cudaMemcpy((void*)dXa, hXa.data(), size_t(batch) * 8, cudaMemcpyHostToDevice), "H2D Xa");
    check(cudaMemcpy((void*)dYa, hYa.data(), size_t(batch) * 8, cudaMemcpyHostToDevice), "H2D Ya");

    strata::prefill::Gemm gemm;
    {
        std::string err;
        if (!gemm.init(nullptr, sh.N * sh.K, err)) {
            std::fprintf(stderr, "FATAL gemm.init: %s\n", err.c_str());
            std::exit(2);
        }
    }
    gemm.f16_batched(dWa, dXa, dYa, T, sh.N, sh.K, batch);
    check(cudaGetLastError(), "f16_batched");
    check(cudaDeviceSynchronize(), "gemm sync");

    // the three arms
    const int64_t ref_rows = std::strcmp(mutate, "rows") == 0 ? T - 1 : T;   // a missing/padded row
    const double g = gamma_k(sh.K);
    double sum_abs = 0;
    std::vector<float> yc(size_t(T * sh.N));
    for (int b = 0; b < batch; ++b) {
        check(cudaMemcpy(yc.data(), dY[b], size_t(T * sh.N) * 4, cudaMemcpyDeviceToHost), "D2H Y");
        for (int64_t t = 0; t < T; ++t)
            for (int64_t n = 0; n < sh.N; ++n) {
                double a = 0.0, bb = 0.0;
                for (int64_t k = 0; k < sh.K; ++k) {
                    const int64_t idx = n * sh.K + k;
                    const double xf = double(x[b][size_t(t * sh.K + k)]);
                    a += double(w[b][size_t(idx)]) * xf;
                    bb += double(h2f(w16[b][size_t(idx)])) * double(h2f(hXh[b][size_t(t * sh.K + k)]));
                }
                double sa = 0.0;
                for (int64_t k = 0; k < sh.K; ++k) {
                    const int64_t idx = n * sh.K + k;
                    sa += std::fabs(double(h2f(w16[b][size_t(idx)])) * double(h2f(hXh[b][size_t(t * sh.K + k)])));
                }
                const double c = double(yc[size_t(t * sh.N + n)]);
                if (!std::isfinite(c) || !std::isfinite(a) || !std::isfinite(bb)) { r.finite_ok = 0; continue; }
                if (t < ref_rows) {      // rows at or past ref_rows are the "missing" row
                    const double operand = std::fabs(bb - a);
                    const double accum = std::fabs(c - bb);
                    r.max_operand_rel = std::max(r.max_operand_rel, operand / (sa > 0 ? sa : 1e-30));
                    r.max_accum_abs = std::max(r.max_accum_abs, accum);
                    sum_abs = std::max(sum_abs, sa);
                    r.worst_sum_abs = std::max(r.worst_sum_abs, sa);
                    r.max_y = std::max(r.max_y, std::fabs(bb));
                    const double bound = g * (sa > 0 ? sa : 1e-30);
                    r.max_accum_rel_to_bound = std::max(r.max_accum_rel_to_bound, accum / bound);
                    if (!(accum <= bound * 4.0)) ++r.wrong;    // 4x headroom over the order-independent bound
                } else {
                    ++r.wrong;                                 // this row had no reference at all
                }
            }
    }
    if (!r.finite_ok) ++r.wrong;
    if (r.weight_bad) r.wrong += r.weight_bad;
    std::printf("  %-4s N=%4lld K=%4lld T=%2lld batch=%d  max|Y| %.3f  max sum|w*x| %.3f  "
                "operand-rounding %.3e (rel to sum|w*x|)  accumulation %.3e (%.2fx of gamma_K*sum|w*x|)  "
                "weights not the reference's: %lld\n",
                sh.name, (long long)sh.N, (long long)sh.K, (long long)T, batch, r.max_y, r.worst_sum_abs,
                r.max_operand_rel, r.max_accum_abs, r.max_accum_rel_to_bound, r.weight_bad);
    for (int b = 0; b < batch; ++b) { delete[] hXh[b]; cudaFree(dW[b]); cudaFree(dWf16[b]); cudaFree(dX[b]); cudaFree(dY[b]); }
    cudaFree((void*)dWa); cudaFree((void*)dXa); cudaFree((void*)dYa);
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    int64_t embd = 0, ff = 0, T = 0;      // 0 = the synthetic shapes, so the registered ctest stays cheap
    int batch = 0;
    const char* mutate = "";
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--embd") embd = std::atoll(argv[i + 1]);
        else if (a == "--ff") ff = std::atoll(argv[i + 1]);
        else if (a == "--rows") T = std::atoll(argv[i + 1]);
        else if (a == "--batch") batch = std::atoi(argv[i + 1]);
        else if (a == "--mutate") mutate = argv[i + 1];
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    const bool toy = embd <= 0 || ff <= 0;
    if (toy) { embd = 256; ff = 256; }
    const int64_t e = toy ? 256 : embd;      // K must stay a multiple of 256 (one super-block)
    const int64_t f = toy ? 256 : ff;
    const Shape shapes[2] = {{"gate", kQ4K, f, e}, {"down", kQ6K, e, f}};
    std::vector<int64_t> rows = T > 0 ? std::vector<int64_t>{T} : std::vector<int64_t>{1, 31, 32, 33};
    std::vector<int> batches = batch > 0 ? std::vector<int>{batch} : std::vector<int>{1, 3};

    std::printf("f16_expert_gemm_parity: %s shapes (embd %lld, ff %lld), cuBLAS contract: "
                "cublasGemmBatchedEx, A=W CUDA_R_16F, B=X CUDA_R_16F, C=Y CUDA_R_32F, "
                "compute CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT; gate: accumulation <= 4 * gamma_K * sum|w*x|, "
                "gamma_K = K*2^-24/(1-K*2^-24)\n", toy ? "toy" : "model", (long long)e, (long long)f);
    if (*mutate) std::printf("  MUTATION '%s' active: the gate is expected to FAIL\n", mutate);

    Result total;
    long long wrong = 0;
    int runs = 0;
    for (const Shape& sh : shapes)
        for (int64_t t : rows)
            for (int b : batches) {
                const Result r = run_proj(sh, t, b, mutate);
                wrong += r.wrong;
                ++runs;
            }
    std::printf("f16_expert_gemm_parity: %d comparison(s), %lld failing value(s) -> %s\n", runs, wrong,
                wrong ? "FAIL" : "PASS");
    return wrong ? 1 : 0;
}
