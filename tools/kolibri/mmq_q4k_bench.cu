// tools/kolibri/mmq_q4k_bench.cu - Kolibri's prompt expert product through strata::prefill::mmq, at Kolibri's shapes.
//
//     STRATA_MMQ_KQ=1 mmq_q4k_bench [rows_per_expert ...]
//
// The plan (2026-10-28-kolibri-prefill-gap.md, P4) needs one number before any engine wiring: what the grouped Q4_K
// product delivers on this card, in GFLOP/s, at the rows per expert a chunk actually produces.
//
// Shapes are Kolibri-1's: gate and up are two [512 x 2560] Q4_K matrices per expert (they sit contiguously in the
// group buffer, so the product is one [1024 x 2560]), and down is [2560 x 512].  Rows per expert from the counters
// that motivated this: a 512-token chunk gives 1 535.8 rows per layer-chunk over ~158 touched experts, i.e. ~10;
// 160 and 626 bracket the 8K and 32K cases.
//
// The number to beat is the CPU pool's: 316-323 GFLOP/s at 7.0 GB/s (docs/AMD_R9700_PORT.md §P0/P2).
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace mmq = strata::prefill::mmq;

static void ck(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
        std::exit(1);
    }
}
static constexpr int kN = 16;   // experts per group (kGatherGroupMax)

/// Valid synthetic Q4_K: the two f16 scales of every 144-byte superblock overwritten with a small positive value, the
/// rest random.  The product's arithmetic is real; only the numbers are noise, which is what a rate measurement needs.
static std::vector<uint8_t> weights_q4k(int64_t rows, int64_t cols, int experts, std::mt19937& rng) {
    const size_t per = mmq::matrix_bytes(GGML_TYPE_Q4_K, rows, cols);
    std::vector<uint8_t> w((size_t) experts * per + 4096);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& b : w) b = (uint8_t) byte(rng);
    const size_t blocks = (size_t) experts * per / 144;   // Q4_K: 144 bytes per 256 values
    for (size_t i = 0; i < blocks; ++i) {
        const uint16_t d = 0x2e66, dmin = 0x2c66;   // ~0.05 and ~0.03 in f16
        std::memcpy(&w[i * 144 + 0], &d, 2);
        std::memcpy(&w[i * 144 + 2], &dmin, 2);
    }
    return w;
}

static float bench_once(mmq::Context& ctx, const mmq::Product& p, int reps) {
    hipEvent_t a, b;
    ck(hipEventCreate(&a), "event");
    ck(hipEventCreate(&b), "event");
    for (int i = 0; i < 2; ++i) ctx.run(p, nullptr);
    ck(hipDeviceSynchronize(), "sync");
    ck(hipEventRecord(a), "rec");
    for (int i = 0; i < reps; ++i) ctx.run(p, nullptr);
    ck(hipEventRecord(b), "rec");
    ck(hipEventSynchronize(b), "sync");
    float ms = 0.f;
    ck(hipEventElapsedTime(&ms, a, b), "elapsed");
    ck(hipEventDestroy(a), "destroy");
    ck(hipEventDestroy(b), "destroy");
    return ms / (float) reps;
}

int main(int argc, char** argv) {
    if (!mmq::built()) {
        std::printf("mmq: not built (this build has no STRATA_PREFILL_MMQ)\n");
        return 2;
    }
    if (!mmq::supported(GGML_TYPE_Q4_K)) {
        std::printf("mmq: Q4_K unsupported here; on HIP that needs STRATA_MMQ_KQ=1 at run time\n");
        return 2;
    }
    std::vector<int> rows_per_expert;
    for (int i = 1; i < argc; ++i) rows_per_expert.push_back(std::atoi(argv[i]));
    if (rows_per_expert.empty()) rows_per_expert = {10, 160, 626};

    const int64_t n_embd = 2560, n_ff = 512;      // Kolibri-1
    const int64_t gu_rows = 2 * n_ff, gu_cols = n_embd;   // gate and up contiguous
    const int64_t d_rows = n_embd, d_cols = n_ff;

    std::mt19937 rng(12345);
    // one group's worth of weights: experts contiguous, each [gu_rows x gu_cols] then [d_rows x d_cols]
    const size_t gu_bytes = mmq::matrix_bytes(GGML_TYPE_Q4_K, gu_rows, gu_cols);
    const size_t d_bytes = mmq::matrix_bytes(GGML_TYPE_Q4_K, d_rows, d_cols);
    auto wgu = weights_q4k(gu_rows, gu_cols, kN, rng);
    auto wd = weights_q4k(d_rows, d_cols, kN, rng);

    void *d_gu = nullptr, *d_d = nullptr;
    ck(hipMalloc(&d_gu, wgu.size()), "malloc gu");
    ck(hipMalloc(&d_d, wd.size()), "malloc d");
    ck(hipMemcpy(d_gu, wgu.data(), wgu.size(), hipMemcpyHostToDevice), "copy gu");
    ck(hipMemcpy(d_d, wd.data(), wd.size(), hipMemcpyHostToDevice), "copy d");

    const int64_t max_rows = *std::max_element(rows_per_expert.begin(), rows_per_expert.end());
    const int64_t total_rows = (int64_t) kN * max_rows;
    float* d_x = nullptr;
    void* d_xq = nullptr;
    float* d_dst = nullptr;
    int32_t* d_ids = nullptr;
    int32_t* d_bounds = nullptr;
    ck(hipMalloc(&d_x, (size_t) total_rows * n_embd * 4), "malloc x");
    ck(hipMalloc(&d_xq, mmq::q8_bytes(total_rows, n_embd) + 4096), "malloc xq");
    ck(hipMalloc(&d_dst, (size_t) kN * max_rows * std::max(gu_rows, d_rows) * 4), "malloc dst");
    ck(hipMalloc(&d_ids, (size_t) kN * max_rows * 4), "malloc ids");
    ck(hipMalloc(&d_bounds, (size_t) (kN + 1) * 4), "malloc bounds");
    {
        std::vector<float> x((size_t) total_rows * n_embd);
        for (size_t i = 0; i < x.size(); ++i) x[i] = (float) ((int) (i * 2654435761u % 2000) - 1000) / 1000.f;
        ck(hipMemcpy(d_x, x.data(), x.size() * 4, hipMemcpyHostToDevice), "copy x");
        std::vector<int32_t> ids((size_t) kN * max_rows), bounds((size_t) kN + 1);
        for (size_t i = 0; i < ids.size(); ++i) ids[i] = (int32_t) i;
        for (int e = 0; e <= kN; ++e) bounds[(size_t) e] = (int32_t) e * (int32_t) max_rows;
        ck(hipMemcpy(d_ids, ids.data(), ids.size() * 4, hipMemcpyHostToDevice), "copy ids");
        ck(hipMemcpy(d_bounds, bounds.data(), bounds.size() * 4, hipMemcpyHostToDevice), "copy bounds");
    }
    mmq::quantize(d_x, nullptr, d_xq, GGML_TYPE_Q4_K, n_embd, n_embd, total_rows, nullptr);
    ck(hipGetLastError(), "quantize");
    ck(hipDeviceSynchronize(), "sync");

    std::printf("mmq Q4_K on Kolibri shapes: %d experts a group, %lld x %lld (gate+up) and %lld x %lld (down)\n",
                kN, (long long) gu_rows, (long long) gu_cols, (long long) d_rows, (long long) d_cols);
    std::printf("%-8s %-10s %-12s %-12s %-12s\n", "rows/ex", "group rows", "gate+up s", "gate+up GF/s", "down GF/s");
    mmq::Context ctx;
    for (int rpe : rows_per_expert) {
        const int64_t rows = (int64_t) kN * rpe;
        std::vector<int32_t> bounds((size_t) kN + 1);
        for (int e = 0; e <= kN; ++e) bounds[(size_t) e] = (int32_t) e * rpe;
        ck(hipMemcpy(d_bounds, bounds.data(), bounds.size() * 4, hipMemcpyHostToDevice), "copy bounds");

        mmq::Product p{};
        p.type = GGML_TYPE_Q4_K;
        p.n = kN;
        p.xq = d_xq;
        p.bounds = d_bounds;
        p.ids = d_ids;
        p.total_rows = rows;
        p.max_rows = rpe;
        p.ld_dst = gu_rows;

        p.w = d_gu;
        p.w_rows = gu_rows;
        p.w_cols = gu_cols;
        p.expert_bytes = gu_bytes;
        p.dst = d_dst;
        const float ms_gu = bench_once(ctx, p, 20);

        p.w = d_d;
        p.w_rows = d_rows;
        p.w_cols = d_cols;
        p.expert_bytes = d_bytes;
        p.ld_dst = d_rows;
        const float ms_d = bench_once(ctx, p, 20);

        const double fl_gu = 2.0 * (double) rows * (double) gu_rows * (double) gu_cols / 1e9;
        const double fl_d = 2.0 * (double) rows * (double) d_rows * (double) d_cols / 1e9;
        std::printf("%-8d %-10lld %-12.4f %-12.1f %-12.1f\n", rpe, (long long) rows, ms_gu, fl_gu / (ms_gu * 1e-3),
                    fl_d / (ms_d * 1e-3));
        std::fflush(stdout);
    }
    std::printf("\nthe CPU pool this replaces: 316-323 GFLOP/s at 7.0 GB/s (docs/AMD_R9700_PORT.md P0/P2)\n");
    return 0;
}
