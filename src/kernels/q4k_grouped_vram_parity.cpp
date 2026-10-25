// src/kernels/q4k_grouped_vram_parity.cpp - campaign II stage G2: the parity gate for the grouped
// Q4_K-from-VRAM GEMM (src/kolibri/q4k_grouped.cu), written and shown FAILING before that kernel exists.
//
// WHAT IS BEING PROVED.  Stage G3 will compute, for `batch` experts at once,
//
//     Y_i[T,N] = X_i[T,K] . W_i[N,K]^T        W_i = Q4_K blocks read straight from VRAM
//
// The reference is the path the engine already runs and trusts: dequant_f16 the same Q4_K blocks to FP16, then
// Gemm::f16_batched over the same X into the same shape.  Identical weights, identical activations, identical
// shapes - the only difference is whether the weights reach the tensor cores as Q4_K read in place or as FP16
// materialised first.  Anything the new kernel gets wrong shows up here and nowhere later.
//
// THE GATE.  Bitwise where the two can be made to agree; otherwise bounded by BOTH a relative error on the
// product and a KL between the softmaxed rows, because the campaign's accuracy claims are KL claims and a GEMM
// that is "close" in max-abs but shifts the argmax is not close enough.  Thresholds are printed with the result
// so a reader can see how much headroom was left.
//
// Convention: CTest-registered, built like mmvq_il_parity (CMakeLists.txt:855).  Exit 0 = PASS, 1 = FAIL.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "kolibri/q4k_grouped.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/prefill/gemm.hpp"

namespace {

void check(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return;
    std::fprintf(stderr, "FATAL %s: %s\n", what, cudaGetErrorString(e));
    std::exit(2);
}

// Q4_K: 144 bytes per 256 values -> d (half), dmin (half), 12 sub-scale/min bytes, 128 bytes of nibbles.
constexpr int kQ4K = 12;
constexpr int64_t kBlkVals = 256;
constexpr int64_t kBlkBytes = 144;

/// IEEE half, round-to-nearest-even, as bits.  Hand-rolled so the test stays a host .cpp (no CUDA headers in
/// the host TU beyond the runtime API).
uint16_t f2h(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = int32_t((x >> 23) & 0xFF) - 127 + 15;
    uint32_t man = x & 0x7FFFFFu;
    if (exp <= 0) return uint16_t(sign);                       // flush subnormals to zero: scales here are ~1e-2
    if (exp >= 31) return uint16_t(sign | 0x7C00u);
    exp <<= 10;
    const uint32_t round = (man >> 13) & 1u;                   // round-to-nearest-even on the 13 dropped bits
    man = (man >> 13) + round;
    if (man > 0x3FFu) { man = 0; exp += 1u << 10; }
    return uint16_t(sign | uint32_t(exp) | (man & 0x3FFu));
}

/// A deterministic, VALID Q4_K blob: sane positive scales (random nibbles under a random-but-bounded scale),
/// so dequantization produces finite values of a realistic magnitude.  Random bytes would put NaN/Inf in the
/// half fields and the reference GEMM would produce garbage that neither path can be blamed for.
std::vector<uint8_t> make_blob(int64_t rows, int64_t cols, uint64_t& rng) {
    std::vector<uint8_t> b(size_t(rows) * (cols / kBlkVals) * kBlkBytes);
    for (size_t i = 0; i < b.size(); ++i) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        b[i] = uint8_t(rng >> 33);
    }
    // repair the 16-byte header of every block: d, dmin, then 12 sub-scale bytes kept in a sane range
    for (size_t blk = 0; blk < b.size() / size_t(kBlkBytes); ++blk) {
        uint8_t* p = b.data() + blk * size_t(kBlkBytes);
        const float d = 0.004f + float((rng >> 40) % 1000u) * 2e-5f;
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        const float dmin = -(0.004f + float((rng >> 40) % 1000u) * 2e-5f);
        const uint16_t hd = f2h(d), hm = f2h(dmin);
        std::memcpy(p, &hd, 2);
        std::memcpy(p + 2, &hm, 2);
        for (int s = 0; s < 12; ++s) p[4 + s] = uint8_t((rng >> (s % 32)) & 0x3Fu);   // 6-bit sub-scales
    }
    return b;
}

/// mean KL(softmax(ref) || softmax(new)) over the T rows - the same quantity the campaign quotes for the model.
double row_kl(const float* ref, const float* newv, int64_t n) {
    double mr = 0.0, mn = 0.0;
    for (int64_t i = 0; i < n; ++i) { mr += ref[i]; mn += newv[i]; }
    mr /= double(n); mn /= double(n);
    double lr = 0.0, ln = 0.0;
    for (int64_t i = 0; i < n; ++i) { lr += std::exp(ref[i] - mr); ln += std::exp(newv[i] - mn); }
    lr = std::log(lr); ln = std::log(ln);
    double kl = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double p = std::exp(ref[i] - mr - lr);
        const double q = std::exp(newv[i] - mn - ln);
        if (p > 0.0 && q > 0.0) kl += p * std::log(p / q);
    }
    return kl;
}

}  // namespace

int main() {
    // The staged prefill's own expert shapes (docs/KOLIBRI1.md: n_embd 2560, FFN 512), T rows per expert and
    // `batch` experts per group - the down projection, which is the leg that costs the most bytes.
    const int64_t T = 4, N = 2560, K = 512;
    const int batch = 8;
    const int64_t wbytes = (N * K / kBlkVals) * kBlkBytes;
    std::printf("q4k_grouped_vram_parity: T=%lld N=%lld K=%lld batch=%d (Q4_K, type %d)\n",
                (long long)T, (long long)N, (long long)K, batch, kQ4K);

    uint64_t rng = 0x9E3779B97F4A7C15ULL;
    std::vector<std::vector<uint8_t>> hw(batch);
    std::vector<std::vector<uint16_t>> hx(batch);
    for (int b = 0; b < batch; ++b) {
        hw[b] = make_blob(N, K, rng);
        hx[b].resize(size_t(T) * K);
        for (auto& v : hx[b]) {
            rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
            const float f = float((rng >> 33) % 20000u) / 10000.0f - 1.0f;   // [-1, 1)
            v = f2h(f);
        }
    }

    std::vector<void*> dW(batch), dX(batch), dYref(batch), dYnew(batch);
    std::vector<const uint8_t*> hWarr(batch);
    std::vector<const uint16_t*> hXarr(batch);
    std::vector<float*> hYref(batch), hYnew(batch);
    for (int b = 0; b < batch; ++b) {
        check(cudaMalloc(&dW[b], size_t(wbytes)), "malloc W");
        check(cudaMemcpy(dW[b], hw[b].data(), size_t(wbytes), cudaMemcpyHostToDevice), "H2D W");
        check(cudaMalloc(&dX[b], size_t(T) * K * 2), "malloc X");
        check(cudaMemcpy(dX[b], hx[b].data(), size_t(T) * K * 2, cudaMemcpyHostToDevice), "H2D X");
        check(cudaMalloc(&dYref[b], size_t(T) * N * 4), "malloc Yref");
        check(cudaMemset(dYref[b], 0, size_t(T) * N * 4), "zero Yref");
        check(cudaMalloc(&dYnew[b], size_t(T) * N * 4), "malloc Ynew");
        // deliberately NOT zeroed: a stub that writes nothing must be caught, and a zeroed buffer would let a
        // silent no-show look like a small-but-nonzero error instead of a total one.
        hWarr[b] = (const uint8_t*)dW[b];
        hXarr[b] = (const uint16_t*)dX[b];
        hYref[b] = (float*)dYref[b];
        hYnew[b] = (float*)dYnew[b];
    }
    const uint8_t* const* dWarr;
    const uint16_t* const* dXarr;
    float* const* dYrefArr;
    float* const* dYnewArr;
    check(cudaMalloc((void**)&dWarr, size_t(batch) * 8), "malloc Warr"),
    check(cudaMalloc((void**)&dXarr, size_t(batch) * 8), "malloc Xarr"),
    check(cudaMalloc((void**)&dYrefArr, size_t(batch) * 8), "malloc YrefArr"),
    check(cudaMalloc((void**)&dYnewArr, size_t(batch) * 8), "malloc YnewArr"),
    check(cudaMemcpy((void*)dWarr, hWarr.data(), size_t(batch) * 8, cudaMemcpyHostToDevice), "H2D Warr"),
    check(cudaMemcpy((void*)dXarr, hXarr.data(), size_t(batch) * 8, cudaMemcpyHostToDevice), "H2D Xarr"),
    check(cudaMemcpy((void*)dYrefArr, hYref.data(), size_t(batch) * 8, cudaMemcpyHostToDevice), "H2D YrefArr"),
    check(cudaMemcpy((void*)dYnewArr, hYnew.data(), size_t(batch) * 8, cudaMemcpyHostToDevice), "H2D YnewArr");

    // ---- the reference arm: dequant_f16 the Q4_K blocks, then the engine's own batched FP16 GEMM ----
    std::vector<void*> dWf16(batch);
    for (int b = 0; b < batch; ++b) {
        check(cudaMalloc(&dWf16[b], size_t(N) * K * 2), "malloc Wf16");
        strata::kernels::dequant_f16(kQ4K, dW[b], 0, N, K, (uint16_t*)dWf16[b], nullptr);
    }
    strata::prefill::Gemm gemm;
    // Gemm needs init before any product (kolibri_engine.cu:3907 does the same); without it cublas returns
    // CUBLAS_STATUS_INVALID_VALUE and the reference arm silently produces nothing - which would make this test
    // fail for the wrong reason, the one failure mode a parity gate must never have.
    {
        std::string err;
        if (!gemm.init(nullptr, N * K, err)) {
            std::fprintf(stderr, "FATAL gemm.init: %s\n", err.c_str());
            return 2;
        }
    }
    // the pointer arrays f16_batched wants are the FP16 weights; X and Y are shared with the new kernel
    std::vector<const uint16_t*> hWf16arr(batch);
    for (int b = 0; b < batch; ++b) hWf16arr[b] = (const uint16_t*)dWf16[b];
    const uint16_t* const* dWf16Arr;
    check(cudaMalloc((void**)&dWf16Arr, size_t(batch) * 8), "malloc Wf16Arr"),
    check(cudaMemcpy((void*)dWf16Arr, hWf16arr.data(), size_t(batch) * 8, cudaMemcpyHostToDevice), "H2D Wf16Arr"),
    gemm.f16_batched(dWf16Arr, dXarr, dYrefArr, T, N, K, batch);
    check(cudaGetLastError(), "reference f16_batched");
    check(cudaDeviceSynchronize(), "reference sync");
    // A reference arm that silently produces zeros would let anything pass later.  Prove it computed something.
    {
        std::vector<float> probe(size_t(T) * N);
        check(cudaMemcpy(probe.data(), dYref[0], size_t(T) * N * 4, cudaMemcpyDeviceToHost), "D2H probe");
        double s = 0.0;
        for (auto v : probe) s += std::fabs(double(v));
        if (!(s > 0.0) || !std::isfinite(s)) {
            std::fprintf(stderr, "FATAL reference arm produced %g (zeros or non-finite) - gate is not testing\n", s);
            return 2;
        }
        std::printf("  reference arm   : ok (sum |Y_0| = %.3f)\n", s);
    }

    // ---- the arm under test ----
    const int rc = strata::kolibri::q4k_grouped_gemm_vram(dWarr, dXarr, dYnewArr, T, N, K, batch, nullptr);
    check(cudaDeviceSynchronize(), "kernel sync");
    if (rc != 0) {
        std::printf("FAIL q4k_grouped_gemm_vram returned %d (kernel not implemented - G2 red phase)\n", rc);
        return 1;
    }

    // ---- compare ----
    std::vector<float> ref(size_t(T) * N), newv(size_t(T) * N);
    double max_abs = 0.0, max_ref = 0.0, kl = 0.0;
    int64_t bitwise = 0, total = 0;
    for (int b = 0; b < batch; ++b) {
        check(cudaMemcpy(ref.data(), dYref[b], size_t(T) * N * 4, cudaMemcpyDeviceToHost), "D2H ref");
        check(cudaMemcpy(newv.data(), dYnew[b], size_t(T) * N * 4, cudaMemcpyDeviceToHost), "D2H new");
        for (int64_t i = 0; i < T * N; ++i) {
            ++total;
            uint32_t a, c;
            std::memcpy(&a, &ref[i], 4);
            std::memcpy(&c, &newv[i], 4);
            if (a == c) ++bitwise;
            const double d = std::fabs(double(ref[i]) - double(newv[i]));
            if (d > max_abs) max_abs = d;
            if (std::fabs(double(ref[i])) > max_ref) max_ref = std::fabs(double(ref[i]));
        }
        for (int64_t t = 0; t < T; ++t) kl += row_kl(ref.data() + t * N, newv.data() + t * N, N);
        kl /= double(T * batch);
    }
    const double rel = max_ref > 0.0 ? max_abs / max_ref : max_abs;
    std::printf("  bitwise equal : %lld / %lld\n", (long long)bitwise, (long long)total);
    std::printf("  max abs diff  : %.6e (max |ref| %.3f) -> rel %.3e\n", max_abs, max_ref, rel);
    std::printf("  mean row KL   : %.6e\n", kl);

    // Thresholds: fp16 accumulation over K=512 with Q4_K weights.  Stated, not discovered after the fact.
    const double rel_gate = 2e-3, kl_gate = 1e-4;
    int fails = 0;
    if (bitwise != total) {
        if (rel > rel_gate) { std::printf("FAIL rel %.3e > gate %.3e\n", rel, rel_gate); ++fails; }
        if (kl > kl_gate)   { std::printf("FAIL KL %.3e > gate %.3e\n", kl, kl_gate); ++fails; }
        if (!fails) std::printf("  (not bitwise: accumulation order differs; both bounds hold)\n");
    }
    if (fails) { std::printf("PARITY FAIL\n"); return 1; }
    std::printf("PARITY PASS\n");
    return 0;
}
