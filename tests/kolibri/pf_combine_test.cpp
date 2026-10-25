// tests/kolibri/pf_combine_test.cpp - the unit test for the prefill MoE combiner (plan
// docs/superpowers/plans/2026-10-24-kolibri-pf-partial-sums.md Task 1).
//
// WHAT IT PINS.  Under STRATA_KOLIBRI_PF_PARTIAL the prefill stops round-tripping resident-expert rows through
// host RAM: the CPU pool hands over one weighted partial per token, the resident rows stay on device in block
// order with a `-1`-padded selection map, and ONE kernel folds shared-expert row + CPU partial + weighted
// resident rows into the MoE output.  The bookkeeping is the defect class this codebase has been bitten by
// twice (the P1 `pf_res_slot` and same-slot bugs), so the mapping is tested here before it is wired in.
//
// The reference is the engine's own definition (kolibri_engine.cu:1086 `k_moe_sum_rows`):
//     moe[t*N+i] = sy[t*N+i] + sum_k wts[t*K+k] * gout[(t*K+k)*N+i]
// with the router weights left un-renormalised (sigmoid, docs/KOLIBRI1.md).  The combiner adds the CPU partial
// once and the resident rows in BLOCK order, so the additions arrive in a different order than the reference -
// which is why cases (b) and (c) are tolerance-gated and (a) and (d) are bitwise.
//
// `wts` is drawn in [0,1] on purpose: that is the range the engine actually passes (sigmoid router weights), and
// it keeps the float accumulation error of a 6-term sum of [-100,100] rows (~2e-4) inside the 1e-3 gate.  With
// weights also at [-100,100] the products reach 1e4 and the float ulp alone (0.008) exceeds the gate, which
// would measure the format, not the kernel.
//
// It needs a GPU and nothing else: no engine internals, no GGUF.

#include "pf_combine.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

int g_fails = 0;
int g_checks = 0;

void check(bool ok, const char* what) {
    ++g_checks;
    if (!ok) {
        ++g_fails;
        std::printf("  FAIL: %s\n", what);
    }
}

void cuda_ok(cudaError_t e, const char* what) {
    ++g_checks;
    if (e != cudaSuccess) {
        ++g_fails;
        std::printf("  FAIL: cuda %s: %s\n", what, cudaGetErrorString(e));
    }
}

// deterministic, no RNG dependency: two runs of the test must see identical inputs
float frand(uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return (float)((s >> 8) & 0xFFFFFF) / (float)0xFFFFFF;  // [0,1)
}
/// value in [-mag, mag]
float frand_sym(uint32_t& s, float mag) { return (frand(s) * 2.f - 1.f) * mag; }

struct Dev {
    float* moe = nullptr;
    float* sy = nullptr;
    float* p = nullptr;
    float* rr = nullptr;
    int* rr_sel = nullptr;
    float* wts = nullptr;

    void alloc(int Tc, int N, int rrows, int K) {
        cuda_ok(cudaMalloc((void**)&moe, (size_t)Tc * N * sizeof(float)), "malloc moe");
        cuda_ok(cudaMalloc((void**)&sy, (size_t)Tc * N * sizeof(float)), "malloc sy");
        cuda_ok(cudaMalloc((void**)&p, (size_t)Tc * N * sizeof(float)), "malloc p");
        cuda_ok(cudaMalloc((void**)&rr, (size_t)rrows * N * sizeof(float)), "malloc rr");
        cuda_ok(cudaMalloc((void**)&rr_sel, (size_t)(rrows ? rrows : 1) * sizeof(int)), "malloc rr_sel");
        cuda_ok(cudaMalloc((void**)&wts, (size_t)Tc * K * sizeof(float)), "malloc wts");
    }
    void free() {
        ::cudaFree(moe);
        ::cudaFree(sy);
        ::cudaFree(p);
        ::cudaFree(rr);
        ::cudaFree(rr_sel);
        ::cudaFree(wts);
        *this = Dev{};
    }
};

/// max |a-b| over n floats, and the largest |b| (to tell a real mismatch from float accumulation)
double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0.0;
    for (size_t i = 0; i < a.size(); ++i) d = std::max(d, (double)std::fabs(a[i] - b[i]));
    return d;
}

bool bitwise_equal(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    return std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

const int N = 2560;  // the model's hidden size
const int K = 6;     // experts per token
const int Tc = 32;   // chunk tokens

}  // namespace

int main() {
    if (cudaFree(0) != cudaSuccess) {
        std::printf("SKIP: no CUDA device\n");
        return 0;
    }

    // ---------------------------------------------------------------------------------------------
    // (a) no residents: rrows = 0.  The combine must reduce to sy + p, and - because that is the same
    //     two-term sum in the same order - it must be BITWISE, not merely close.
    // ---------------------------------------------------------------------------------------------
    std::printf("case a: no residents (rrows = 0) -> bitwise sy + p\n");
    {
        Dev d;
        d.alloc(Tc, N, 1, K);
        std::vector<float> sy((size_t)Tc * N), p((size_t)Tc * N), wts((size_t)Tc * K);
        uint32_t s = 12345u;
        for (auto& v : sy) v = frand_sym(s, 100.f);
        for (auto& v : p) v = frand_sym(s, 100.f);
        for (auto& v : wts) v = frand(s);
        std::vector<int> sel(1, -1);
        std::vector<float> ref((size_t)Tc * N);
        for (int t = 0; t < Tc; ++t)
            for (int i = 0; i < N; ++i) ref[(size_t)t * N + i] = sy[(size_t)t * N + i] + p[(size_t)t * N + i];

        cudaMemcpy(d.sy, sy.data(), sy.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d.p, p.data(), p.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d.wts, wts.data(), wts.size() * 4, cudaMemcpyHostToDevice);
        strata::kolibri::pf_combine_rows(d.moe, d.sy, d.p, d.rr, d.rr_sel, d.wts, Tc, 0, K, N, 0);
        std::vector<float> got((size_t)Tc * N);
        cudaMemcpy(got.data(), d.moe, got.size() * 4, cudaMemcpyDeviceToHost);
        check(bitwise_equal(got, ref), "case a: moe is bitwise sy + p");
        if (!bitwise_equal(got, ref)) std::printf("    max abs diff %g\n", max_abs_diff(got, ref));
        d.free();
    }

    // ---------------------------------------------------------------------------------------------
    // (b) no misses: p is zero and all K picks per token are resident rows.  Block order reorders the
    //     sum, so this is tolerance-gated (1e-3 max-abs), not bitwise.
    // ---------------------------------------------------------------------------------------------
    std::printf("case b: no misses (p = 0, all K picks resident) -> within 1e-3\n");
    {
        Dev d;
        const int rrows = Tc * K;
        d.alloc(Tc, N, rrows, K);
        std::vector<float> sy((size_t)Tc * N), p((size_t)Tc * N, 0.f), wts((size_t)Tc * K);
        std::vector<float> rr((size_t)rrows * N);
        std::vector<int> rr_sel(rrows);
        std::vector<int> row_of(Tc * K, -1);
        uint32_t s = 999u;
        for (auto& v : sy) v = frand_sym(s, 100.f);
        for (auto& v : wts) v = frand(s);
        for (auto& v : rr) v = frand_sym(s, 100.f);
        // block order: ascending j maps to ascending (t,k)
        for (int j = 0; j < rrows; ++j) {
            rr_sel[j] = j;
            row_of[j] = j;
        }
        // the kernel weights the rows itself from wts[rr_sel[j]], so rr carries the raw rows
        cudaMemcpy(d.sy, sy.data(), sy.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d.p, p.data(), p.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d.rr, rr.data(), rr.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d.wts, wts.data(), wts.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d.rr_sel, rr_sel.data(), rr_sel.size() * 4, cudaMemcpyHostToDevice);
        strata::kolibri::pf_combine_rows(d.moe, d.sy, d.p, d.rr, d.rr_sel, d.wts, Tc, rrows, K, N, 0);
        std::vector<float> got((size_t)Tc * N);
        cudaMemcpy(got.data(), d.moe, got.size() * 4, cudaMemcpyDeviceToHost);

        std::vector<float> ref((size_t)Tc * N);
        for (int t = 0; t < Tc; ++t)
            for (int i = 0; i < N; ++i) {
                double acc = sy[(size_t)t * N + i];
                for (int k = 0; k < K; ++k) {
                    const int j = row_of[t * K + k];
                    acc += (double)wts[t * K + k] * (double)rr[(size_t)j * N + i];
                }
                ref[(size_t)t * N + i] = (float)acc;
            }
        const double diff = max_abs_diff(got, ref);
        check(diff <= 1e-3, "case b: within 1e-3 of sy + sum_k wts*rr");
        std::printf("    max abs diff %.3e\n", diff);
        d.free();
    }

    // ---------------------------------------------------------------------------------------------
    // (c) mixed: half the picks resident, the other half already folded into the CPU partial.
    // ---------------------------------------------------------------------------------------------
    std::printf("case c: mixed (3 of 6 picks resident) -> within 1e-3\n");
    {
        Dev d;
        // the partial already carries the miss picks pre-weighted, so build it on the host the way
        // the engine's pool pass will: p[t,i] = sum over miss k of wts[t,k] * miss_row[t,k,i]
        const int kres = 3;
        const int rrows = Tc * kres;
        d.alloc(Tc, N, rrows, K);
        std::vector<float> sy((size_t)Tc * N), p((size_t)Tc * N, 0.f), wts((size_t)Tc * K);
        std::vector<float> rr((size_t)rrows * N);
        std::vector<int> rr_sel(rrows, -1);
        std::vector<int> row_of(Tc * K, -1);
        uint32_t s = 4242u;
        for (auto& v : sy) v = frand_sym(s, 100.f);
        for (auto& v : wts) v = frand(s);
        for (auto& v : rr) v = frand_sym(s, 100.f);
        std::vector<float> miss_rows((size_t)Tc * K * N, 0.f);
        for (int t = 0; t < Tc; ++t) {
            for (int k = 0; k < K; ++k) {
                if (k < kres) {  // resident: block row j = t*kres + k
                    const int j = t * kres + k;
                    rr_sel[j] = t * K + k;
                    row_of[t * K + k] = j;
                } else {  // miss: folded into p
                    for (int i = 0; i < N; ++i)
                        miss_rows[(size_t)(t * K + k) * N + i] = frand_sym(s, 100.f);
                }
            }
        }
        for (int t = 0; t < Tc; ++t)
            for (int k = kres; k < K; ++k)
                for (int i = 0; i < N; ++i)
                    p[(size_t)t * N + i] += wts[t * K + k] * miss_rows[(size_t)(t * K + k) * N + i];

        cudaMemcpy(d.sy, sy.data(), sy.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d.p, p.data(), p.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d.rr, rr.data(), rr.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d.wts, wts.data(), wts.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d.rr_sel, rr_sel.data(), rr_sel.size() * 4, cudaMemcpyHostToDevice);
        strata::kolibri::pf_combine_rows(d.moe, d.sy, d.p, d.rr, d.rr_sel, d.wts, Tc, rrows, K, N, 0);
        std::vector<float> got((size_t)Tc * N);
        cudaMemcpy(got.data(), d.moe, got.size() * 4, cudaMemcpyDeviceToHost);

        std::vector<float> ref((size_t)Tc * N);
        for (int t = 0; t < Tc; ++t)
            for (int i = 0; i < N; ++i) {
                double acc = sy[(size_t)t * N + i] + p[(size_t)t * N + i];
                for (int k = 0; k < kres; ++k) {
                    const int j = row_of[t * K + k];
                    acc += (double)wts[t * K + k] * (double)rr[(size_t)j * N + i];
                }
                ref[(size_t)t * N + i] = (float)acc;
            }
        const double diff = max_abs_diff(got, ref);
        check(diff <= 1e-3, "case c: within 1e-3 of sy + p + resident weighted rows");
        std::printf("    max abs diff %.3e\n", diff);
        d.free();
    }

    // ---------------------------------------------------------------------------------------------
    // (d) padding: `-1` selection rows must contribute EXACTLY zero.  Same data, one block padded with
    //     trailing -1 rows, must come out BITWISE identical to the unpadded run.
    // ---------------------------------------------------------------------------------------------
    std::printf("case d: -1 padding contributes exactly zero -> bitwise vs truncated\n");
    {
        Dev d;
        const int real = Tc * K;
        const int pad = 7;
        const int rrows = real + pad;
        d.alloc(Tc, N, rrows, K);
        std::vector<float> sy((size_t)Tc * N), p((size_t)Tc * N, 0.f), wts((size_t)Tc * K);
        std::vector<float> rr((size_t)rrows * N);
        std::vector<int> rr_sel(rrows, -1);
        uint32_t s = 777u;
        for (auto& v : sy) v = frand_sym(s, 100.f);
        for (auto& v : wts) v = frand(s);
        for (auto& v : rr) v = frand_sym(s, 100.f);  // padded rows hold real garbage on purpose
        for (int j = 0; j < real; ++j) rr_sel[j] = j;  // trailing `pad` entries stay -1
        cudaMemcpy(d.sy, sy.data(), sy.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d.p, p.data(), p.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d.rr, rr.data(), rr.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d.wts, wts.data(), wts.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d.rr_sel, rr_sel.data(), rr_sel.size() * 4, cudaMemcpyHostToDevice);
        strata::kolibri::pf_combine_rows(d.moe, d.sy, d.p, d.rr, d.rr_sel, d.wts, Tc, rrows, K, N, 0);
        std::vector<float> padded((size_t)Tc * N);
        cudaMemcpy(padded.data(), d.moe, padded.size() * 4, cudaMemcpyDeviceToHost);

        // the same block truncated: rrows = real, rr_sel unchanged for the first `real` rows
        cudaMemset(d.moe, 0, (size_t)Tc * N * 4);
        strata::kolibri::pf_combine_rows(d.moe, d.sy, d.p, d.rr, d.rr_sel, d.wts, Tc, real, K, N, 0);
        std::vector<float> trunc((size_t)Tc * N);
        cudaMemcpy(trunc.data(), d.moe, trunc.size() * 4, cudaMemcpyDeviceToHost);
        check(bitwise_equal(padded, trunc), "case d: padded and truncated outputs are bitwise equal");
        if (!bitwise_equal(padded, trunc)) std::printf("    max abs diff %g\n", max_abs_diff(padded, trunc));
        d.free();
    }

    // ---------------------------------------------------------------------------------------------
    // (e) determinism: two launches over identical inputs give bitwise-identical output.  The prefill
    //     gate (two runs, identical ids) rests on this.
    // ---------------------------------------------------------------------------------------------
    std::printf("case e: two launches -> bitwise identical\n");
    {
        Dev d;
        const int rrows = Tc * K;
        d.alloc(Tc, N, rrows, K);
        std::vector<float> sy((size_t)Tc * N), p((size_t)Tc * N), wts((size_t)Tc * K);
        std::vector<float> rr((size_t)rrows * N);
        std::vector<int> rr_sel(rrows);
        uint32_t s = 31337u;
        for (auto& v : sy) v = frand_sym(s, 100.f);
        for (auto& v : p) v = frand_sym(s, 100.f);
        for (auto& v : wts) v = frand(s);
        for (auto& v : rr) v = frand_sym(s, 100.f);
        for (int j = 0; j < rrows; ++j) rr_sel[j] = (j % 4 == 3) ? -1 : j;  // mix of residents and padding
        cudaMemcpy(d.sy, sy.data(), sy.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d.p, p.data(), p.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d.rr, rr.data(), rr.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d.wts, wts.data(), wts.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d.rr_sel, rr_sel.data(), rr_sel.size() * 4, cudaMemcpyHostToDevice);
        std::vector<float> r1((size_t)Tc * N), r2((size_t)Tc * N);
        strata::kolibri::pf_combine_rows(d.moe, d.sy, d.p, d.rr, d.rr_sel, d.wts, Tc, rrows, K, N, 0);
        cudaMemcpy(r1.data(), d.moe, r1.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemset(d.moe, 0, (size_t)Tc * N * 4);
        strata::kolibri::pf_combine_rows(d.moe, d.sy, d.p, d.rr, d.rr_sel, d.wts, Tc, rrows, K, N, 0);
        cudaMemcpy(r2.data(), d.moe, r2.size() * 4, cudaMemcpyDeviceToHost);
        check(bitwise_equal(r1, r2), "case e: two launches are bitwise identical");
        d.free();
    }

    std::printf("\n%d checks, %d failures\n", g_checks, g_fails);
    return g_fails ? 1 : 0;
}
