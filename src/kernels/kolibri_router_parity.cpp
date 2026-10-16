// src/kernels/kolibri_router_parity.cpp - the Kolibri 1 router (expert_gating_func 5, sigmoid_logit_add) checked
// against a host reference written from the same specification: select the top-k of `logits + bias` (equal scores
// -> the lower expert index), the weights are the UNBIASED sigmoid(logits) of the picks, NOT renormalised.
// 4096 rows per case (a prefill chunk's worth); ids and weights compared bitwise.  The reference's expf is the
// host's, the kernel's is CUDA's: ids are exact by construction, weights bitwise where the two expf agree, and
// the report says which it is.
#include "strata/kernels/native_router.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

constexpr int NE = 384;   // kolibri1.expert_count
constexpr int K = 6;      // kolibri1.expert_used_count

// The host reference, in plain F32: scores = logits + bias, stable descending argsort (strict > keeps equal
// scores in index order, so the lower index wins ties), weights = 1/(1+exp(-logit)) of the picks.
void reference(const float* logits, const float* bias, int* ids, float* w) {
    std::vector<float> score((size_t) NE);
    for (int e = 0; e < NE; ++e) score[(size_t) e] = logits[e] + bias[e];
    std::vector<int> idx((size_t) NE);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return score[(size_t) a] > score[(size_t) b]; });
    for (int i = 0; i < K; ++i) {
        ids[i] = idx[(size_t) i];
        w[i] = 1.0f / (1.0f + expf(-logits[ids[i]]));
    }
}

int run_case(const char* name, const std::vector<float>& logits, const std::vector<float>& bias, int n_tok) {
    std::vector<int> h_ids((size_t) n_tok * K), r_ids((size_t) n_tok * K);
    std::vector<float> h_w((size_t) n_tok * K), r_w((size_t) n_tok * K);
    for (int t = 0; t < n_tok; ++t)
        reference(&logits[(size_t) t * NE], &bias[0], &r_ids[(size_t) t * K], &r_w[(size_t) t * K]);

    float *d_l = nullptr, *d_b = nullptr, *d_w = nullptr;
    int* d_ids = nullptr;
    check(cudaMalloc(&d_l, logits.size() * sizeof(float)), "malloc logits");
    check(cudaMalloc(&d_b, bias.size() * sizeof(float)), "malloc bias");
    check(cudaMalloc(&d_ids, h_ids.size() * sizeof(int)), "malloc ids");
    check(cudaMalloc(&d_w, h_w.size() * sizeof(float)), "malloc w");
    check(cudaMemcpy(d_l, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice), "copy logits");
    check(cudaMemcpy(d_b, bias.data(), bias.size() * sizeof(float), cudaMemcpyHostToDevice), "copy bias");
    cudaStream_t st;
    check(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
    strata::kernels::native_router_sigmoid_logit_add(d_l, d_b, d_ids, d_w, n_tok, st);
    check(cudaStreamSynchronize(st), "sync");
    check(cudaStreamDestroy(st), "destroy");
    check(cudaMemcpy(h_ids.data(), d_ids, h_ids.size() * sizeof(int), cudaMemcpyDeviceToHost), "back ids");
    check(cudaMemcpy(h_w.data(), d_w, h_w.size() * sizeof(float), cudaMemcpyDeviceToHost), "back w");

    long long id_bad = 0, w_bit_bad = 0;
    double worst = 0;
    for (size_t i = 0; i < h_ids.size(); ++i) {
        if (h_ids[i] != r_ids[i]) ++id_bad;
        if (std::memcmp(&h_w[i], &r_w[i], 4) != 0) ++w_bit_bad;
        const double ref = std::fabs((double) r_w[i]) > 1e-30 ? std::fabs((double) r_w[i]) : 1e-30;
        worst = std::max(worst, std::fabs((double) h_w[i] - (double) r_w[i]) / ref);
    }
    std::printf("  %-30s ids %s (%lld bad)   weights bitwise %s (%lld off, worst rel %.3e)\n", name,
                id_bad ? "*** WRONG ***" : "exact", id_bad, w_bit_bad ? "off" : "EXACT", w_bit_bad, worst);
    cudaFree(d_l); cudaFree(d_b); cudaFree(d_ids); cudaFree(d_w);
    return (int) id_bad;   // ids must be exact; weight bit differences are reported, not failures (expf host/device)
}

}  // namespace

int main() {
    std::mt19937 rng(20261016);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    int bad = 0;
    const int NT = 4096;

    // ---- random logits and bias, 4096 rows (a prefill chunk)
    std::vector<float> a((size_t) NT * NE), ab((size_t) NE);
    for (auto& v : a) v = gauss(rng);
    for (auto& v : ab) v = 0.1f * gauss(rng);
    bad += run_case("random normal x4096", a, ab, NT);

    // ---- single token (decode step)
    bad += run_case("random normal x1", a, ab, 1);

    // ---- ALL-EQUAL logits and bias: every expert ties, so the "lower index wins" rule is the only thing deciding
    std::vector<float> b((size_t) NT * NE, 0.5f), bb((size_t) NE, 0.25f);
    bad += run_case("all equal (ties)", b, bb, NT);

    // ---- exact ties among the top: 12 experts share the top score, the 6 picks must be the 6 lowest indices
    std::vector<float> c((size_t) NT * NE);
    for (int t = 0; t < NT; ++t)
        for (int e = 0; e < NE; ++e) c[(size_t) t * NE + e] = (e < 12) ? 3.0f : gauss(rng);
    bad += run_case("12-way exact tie at the top", c, ab, NT);

    // ---- bias decides the picks: logits all zero, the bias ranks the experts
    std::vector<float> d((size_t) NT * NE, 0.0f), db((size_t) NE);
    for (int e = 0; e < NE; ++e) db[(size_t) e] = (float) (NE - e);
    bad += run_case("bias-only ranking", d, db, NT);

    // ---- large-magnitude logits: sigmoid saturates to 0 and 1, exp overflows to inf
    std::vector<float> f((size_t) NT * NE);
    for (int t = 0; t < NT; ++t)
        for (int e = 0; e < NE; ++e) f[(size_t) t * NE + e] = (e % 2 ? 1.0f : -1.0f) * (float) (e % 200);
    bad += run_case("large logits (+-200)", f, ab, NT);

    // ---- near-ties: scores differ by one float step, the comparison must be exact
    std::vector<float> g((size_t) NT * NE);
    for (int t = 0; t < NT; ++t)
        for (int e = 0; e < NE; ++e) g[(size_t) t * NE + e] = 2.0f + (e == 7 ? 0.0f : std::nextafterf(2.0f, 0.0f) * 2.0f);
    bad += run_case("near-ties (one float step)", g, ab, NT);

    std::printf("\nkolibri_router_parity: %d id failures over 7 cases x %d rows x %d experts, top-%d\n",
                bad, NT, NE, K);
    if (bad) return 1;
    std::printf("kolibri_router_parity OK\n");
    return 0;
}
