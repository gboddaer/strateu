// tools/kolibri/mmq_pipeline_probe.cu - the engine's MMQ orchestration, standalone, so a hang is a 5-second experiment.
//
// The engine's first wiring of P4 (plan 2026-10-28-kolibri-prefill-gap.md) compiled and then hung with no output.  This
// probe replays the same sequence on synthetic Q4_K weights and a permutation `ids` table, with switches that take
// one step out at a time, so the hanging step is named instead of guessed:
//
//     STRATA_MMQ_KQ=1 mmq_pipeline_probe [--rows R] [--groups G] [--no-gather] [--no-down] [--identity-ids]
//                                       [--dst-host] [--no-quant-ids]
//
// The default is the engine's shape: R = 10 rows per expert, 16 experts a group, gate+up as one [1024 x 2560], down
// [2560 x 512], activations quantized in expert order through a row->token table, the down product writing rows to a
// permutation of slots (dst row = ids[row], ld_dst = 2560).
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
static void step(const char* s) {
    std::printf("  step: %s\n", s);
    std::fflush(stdout);
}

static std::vector<uint8_t> weights_q4k(int64_t rows, int64_t cols, int experts, std::mt19937& rng) {
    const size_t per = mmq::matrix_bytes(GGML_TYPE_Q4_K, rows, cols);
    std::vector<uint8_t> w((size_t) experts * per + 4096);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& b : w) b = (uint8_t) byte(rng);
    const size_t blocks = (size_t) experts * per / 144;
    for (size_t i = 0; i < blocks; ++i) {
        const uint16_t d = 0x2e66, dmin = 0x2c66;
        std::memcpy(&w[i * 144 + 0], &d, 2);
        std::memcpy(&w[i * 144 + 2], &dmin, 2);
    }
    return w;
}

int main(int argc, char** argv) {
    int R = 10, groups = 1;
    bool no_gather = false, no_down = false, identity_ids = false, dst_host = false, no_quant_ids = false;
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (!std::strcmp(a, "--rows") && i + 1 < argc) R = std::atoi(argv[++i]);
        else if (!std::strcmp(a, "--groups") && i + 1 < argc) groups = std::atoi(argv[++i]);
        else if (!std::strcmp(a, "--no-gather")) no_gather = true;
        else if (!std::strcmp(a, "--no-down")) no_down = true;
        else if (!std::strcmp(a, "--identity-ids")) identity_ids = true;
        else if (!std::strcmp(a, "--dst-host")) dst_host = true;
        else if (!std::strcmp(a, "--no-quant-ids")) no_quant_ids = true;
    }
    if (!mmq::built() || !mmq::supported(GGML_TYPE_Q4_K)) {
        std::printf("mmq: not built / Q4_K unsupported (needs STRATA_MMQ_KQ=1)\n");
        return 2;
    }

    const int64_t N = 2560, FF = 512;
    const int kN = 16, NG = std::max(1, groups);
    const int64_t rows = (int64_t) NG * kN * R;

    std::mt19937 rng(7);
    const size_t gu_half = mmq::matrix_bytes(GGML_TYPE_Q4_K, FF, N);
    const size_t d_sz = mmq::matrix_bytes(GGML_TYPE_Q4_K, N, FF);
    auto wgu = weights_q4k(FF, N, kN, rng);   // one gate matrix per expert slot of a group
    auto wd = weights_q4k(N, FF, kN, rng);

    void *gu_stage = nullptr, *d_stage = nullptr;
    ck(hipMalloc(&gu_stage, 2 * gu_half * kN + 4096), "malloc gu_stage");
    ck(hipMalloc(&d_stage, d_sz * kN + 4096), "malloc d_stage");
    // gather_native_group expects one blob per expert: gate at +0, up at +up_off, down at +down_off
    std::vector<uint8_t> blobs((size_t) kN * (2 * gu_half + d_sz) + 4096);
    for (int z = 0; z < kN; ++z) {
        const size_t base = (size_t) z * (2 * gu_half + d_sz);
        std::memcpy(&blobs[base], &wgu[(size_t) z * gu_half], gu_half);
        std::memcpy(&blobs[base + gu_half], &wgu[(size_t) z * gu_half], gu_half);
        std::memcpy(&blobs[base + 2 * gu_half], &wd[(size_t) z * d_sz], d_sz);
    }
    void* d_blobs = nullptr;
    ck(hipMalloc(&d_blobs, blobs.size()), "malloc blobs");
    ck(hipMemcpy(d_blobs, blobs.data(), blobs.size(), hipMemcpyHostToDevice), "copy blobs");

    float* d_x = nullptr;
    void *d_xq = nullptr, *d_xqh = nullptr;
    float *d_gu = nullptr, *d_h = nullptr, *d_dst = nullptr;
    int32_t *d_row_tok = nullptr, *d_row_sel = nullptr, *d_bounds = nullptr;
    ck(hipMalloc(&d_x, (size_t) rows * N * 4), "malloc x");
    ck(hipMalloc(&d_xq, mmq::q8_bytes(rows, N) + 4096), "malloc xq");
    ck(hipMalloc(&d_xqh, mmq::q8_bytes(rows, FF) + 4096), "malloc xqh");
    ck(hipMalloc(&d_gu, (size_t) rows * 2 * FF * 4), "malloc gu");
    ck(hipMalloc(&d_h, (size_t) rows * FF * 4), "malloc h");
    float* host_dst = nullptr;
    if (dst_host) {
        ck(hipHostAlloc(&host_dst, (size_t) rows * N * 4, hipHostAllocDefault), "hostalloc dst");
        d_dst = host_dst;
    } else {
        ck(hipMalloc(&d_dst, (size_t) rows * N * 4), "malloc dst");
    }
    ck(hipMalloc(&d_row_tok, (size_t) rows * 4), "malloc row_tok");
    ck(hipMalloc(&d_row_sel, (size_t) rows * 4), "malloc row_sel");
    ck(hipMalloc(&d_bounds, (size_t) (NG * kN + 1) * 4), "malloc bounds");

    {
        std::vector<float> x((size_t) rows * N);
        for (size_t i = 0; i < x.size(); ++i) x[i] = (float) ((int) (i * 2654435761u % 2000) - 1000) / 1000.f;
        ck(hipMemcpy(d_x, x.data(), x.size() * 4, hipMemcpyHostToDevice), "copy x");
        std::vector<int32_t> tok((size_t) rows), sel((size_t) rows), bnd((size_t) NG * kN + 1);
        for (size_t i = 0; i < tok.size(); ++i) {
            tok[i] = (int32_t) (i / 6);                     // 6 picks a token, as Kolibri's top-6 does
            sel[i] = identity_ids ? (int32_t) i : (int32_t) (i * 7 % rows);
        }
        for (int g = 0; g <= NG * kN; ++g) bnd[(size_t) g] = (int32_t) ((int64_t) g * R);
        ck(hipMemcpy(d_row_tok, tok.data(), tok.size() * 4, hipMemcpyHostToDevice), "copy tok");
        ck(hipMemcpy(d_row_sel, sel.data(), sel.size() * 4, hipMemcpyHostToDevice), "copy sel");
        ck(hipMemcpy(d_bounds, bnd.data(), bnd.size() * 4, hipMemcpyHostToDevice), "copy bounds");
    }
    std::printf("probe: rows %lld (R %d x %d experts x %d groups), ids %s, dst %s\n", (long long) rows, R, kN, NG,
                identity_ids ? "identity" : "permutation", dst_host ? "pinned host" : "device");
    std::fflush(stdout);

    step("quantize(ids)");
    mmq::quantize(d_x, no_quant_ids ? nullptr : d_row_tok, d_xq, GGML_TYPE_Q4_K, N, N, rows, nullptr);
    ck(hipGetLastError(), "quantize");
    ck(hipDeviceSynchronize(), "sync quantize");

    mmq::Context ctx;
    for (int g = 0; g < NG; ++g) {
        step("gather_native_group");
        if (!no_gather) {
            mmq::GatherGroup grp;
            grp.first = g * kN;
            grp.n = kN;
            for (int q = 0; q < kN; ++q) grp.blob[q] = (const uint8_t*) d_blobs + (size_t) q * (2 * gu_half + d_sz);
            const bool ok = mmq::gather_native_group(grp, gu_half, gu_half, 2 * gu_half, d_sz, gu_stage, 2 * gu_half,
                                                     d_stage, d_sz, nullptr);
            std::printf("  gather returned %d\n", (int) ok);
            std::fflush(stdout);
            if (!ok) return 3;
            ck(hipGetLastError(), "gather");
            ck(hipDeviceSynchronize(), "sync gather");
        }
        step("gate+up product");
        mmq::Product p{};
        p.type = GGML_TYPE_Q4_K;
        p.n = kN;
        p.xq = d_xq;
        p.bounds = d_bounds + (size_t) g * kN;
        p.ids = d_row_sel;
        p.total_rows = rows;
        p.max_rows = R;
        p.w = gu_stage;
        p.w_rows = 2 * FF;
        p.w_cols = N;
        p.expert_bytes = 2 * gu_half;
        p.dst = d_gu;
        p.ld_dst = 2 * FF;
        ctx.run(p, nullptr);
        ck(hipGetLastError(), "gu product");
        ck(hipDeviceSynchronize(), "sync gu");
        step("swiglu");
        mmq::swiglu(d_gu, d_h, rows, FF, false, nullptr);
        ck(hipGetLastError(), "swiglu");
        ck(hipDeviceSynchronize(), "sync swiglu");
        step("quantize hidden (no ids)");
        mmq::quantize(d_h, nullptr, d_xqh, GGML_TYPE_Q4_K, FF, FF, rows, nullptr);
        ck(hipGetLastError(), "quantize h");
        ck(hipDeviceSynchronize(), "sync quantize h");
        if (!no_down) {
            step("down product -> dst via ids");
            mmq::Product pd = p;
            pd.xq = d_xqh;
            pd.w = d_stage;
            pd.w_rows = N;
            pd.w_cols = FF;
            pd.expert_bytes = d_sz;
            pd.dst = d_dst;
            pd.ld_dst = N;
            ctx.run(pd, nullptr);
            ck(hipGetLastError(), "down product");
            ck(hipDeviceSynchronize(), "sync down");
        }
    }
    std::printf("PROBE OK: the engine's sequence completes\n");
    return 0;
}
