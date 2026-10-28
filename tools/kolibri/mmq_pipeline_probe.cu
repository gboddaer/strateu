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


// ---- a scalar Q4_K reference (the engine's deq32, host side): one row of a matrix, `cols` values -------------------
static float h2f_(uint16_t h) {
    const uint32_t s = (uint32_t) (h >> 15) << 31, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e == 0) { const float v = std::ldexp((float) m, -24); return s ? -v : v; }
    const uint32_t f = e == 31 ? (s | 0x7f800000u | (m << 13)) : (s | ((e + 112) << 23) | (m << 13));
    float out; std::memcpy(&out, &f, 4); return out;
}
static uint16_t ld16_(const uint8_t* p) { return (uint16_t) (p[0] | (p[1] << 8)); }
static void scale_min_k4_(int j, const uint8_t* q, int& d, int& m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
static void deq_row_q4k(const uint8_t* matrix, int64_t row, int64_t cols, float* out) {
    const size_t row_bytes = (size_t) (cols / 256) * 144;
    const uint8_t* r = matrix + (size_t) row * row_bytes;
    for (int64_t c = 0; c < cols / 32; ++c) {
        const uint8_t* b = r + 144 * (c >> 3);
        const int j = (int) (c & 7);
        const float d = h2f_(ld16_(b)), dmin = h2f_(ld16_(b + 2));
        int sc, mn;
        scale_min_k4_(j, b + 4, sc, mn);
        const float d1 = d * (float) sc, m1 = dmin * (float) mn;
        const uint8_t* qs = b + 16 + 32 * (j >> 1);
        for (int l = 0; l < 32; ++l)
            out[c * 32 + l] = d1 * (float) ((j & 1) ? (qs[l] >> 4) : (qs[l] & 0xF)) - m1;
    }
}

// ---- parity: one group's product against the reference, and the three layout hypotheses --------------------------
static int parity_mode(int R, int kN) {
    const int64_t N = 2560, FF = 512;
    const int64_t rows = (int64_t) kN * R;
    std::mt19937 rng(11);
    auto wgu = weights_q4k(FF, N, kN, rng);   // gate only, one matrix per expert
    auto wd = weights_q4k(N, FF, kN, rng);
    const size_t gu_half = (size_t) (FF / 256) * 144 * 1;   // one matrix's bytes per expert: rows * (cols/256) * 144
    const size_t gu_bytes = (size_t) FF * (N / 256) * 144;
    const size_t d_bytes = (size_t) N * (FF / 256) * 144;
    (void) gu_half;

    // blobs: gate at +0, up at +gu_bytes, down at +2*gu_bytes
    std::vector<uint8_t> blobs((size_t) kN * (2 * gu_bytes + d_bytes) + 4096);
    for (int z = 0; z < kN; ++z) {
        const size_t base = (size_t) z * (2 * gu_bytes + d_bytes);
        std::memcpy(&blobs[base], &wgu[(size_t) z * gu_bytes], gu_bytes);                 // gate
        std::memcpy(&blobs[base + gu_bytes], &wgu[(size_t) z * gu_bytes], gu_bytes);      // up (same matrix: a test)
        std::memcpy(&blobs[base + 2 * gu_bytes], &wd[(size_t) z * d_bytes], d_bytes);
    }
    void* d_blobs = nullptr;
    ck(hipMalloc(&d_blobs, blobs.size()), "malloc blobs");
    ck(hipMemcpy(d_blobs, blobs.data(), blobs.size(), hipMemcpyHostToDevice), "copy blobs");

    void *gu_stage = nullptr, *d_stage = nullptr, *d_xq = nullptr;
    float *d_gu = nullptr, *d_dst = nullptr, *d_x = nullptr;
    int32_t *d_tok = nullptr, *d_sel = nullptr, *d_bounds = nullptr;
    ck(hipMalloc(&gu_stage, 2 * gu_bytes * kN + 4096), "malloc gu_stage");
    ck(hipMalloc(&d_stage, d_bytes * kN + 4096), "malloc d_stage");
    ck(hipMalloc(&d_xq, mmq::q8_bytes(rows, N) + 4096), "malloc xq");
    ck(hipMalloc(&d_gu, (size_t) rows * 2 * FF * 4), "malloc gu");
    ck(hipMalloc(&d_dst, (size_t) rows * N * 4), "malloc dst");
    ck(hipMalloc(&d_x, (size_t) rows * N * 4), "malloc x");
    ck(hipMalloc(&d_tok, (size_t) rows * 4), "malloc tok");
    ck(hipMalloc(&d_sel, (size_t) rows * 4), "malloc sel");
    ck(hipMalloc(&d_bounds, (size_t) (kN + 1) * 4), "malloc bounds");
    std::vector<float> x((size_t) rows * N, 0.f);
    for (int64_t r = 0; r < rows; ++r) x[(size_t) r * N] = 1.f;   // delta: out[row][c] must be W[c][0] of that row's expert
    ck(hipMemcpy(d_x, x.data(), x.size() * 4, hipMemcpyHostToDevice), "copy x");
    std::vector<int32_t> tok((size_t) rows), sel((size_t) rows), bnd((size_t) kN + 1);
    for (size_t i = 0; i < tok.size(); ++i) { tok[i] = (int32_t) (i / 2); sel[i] = (int32_t) (rows - 1 - i); }  // a swap: tests the ids base
    for (int g = 0; g <= kN; ++g) bnd[(size_t) g] = (int32_t) ((int64_t) g * R);
    ck(hipMemcpy(d_tok, tok.data(), tok.size() * 4, hipMemcpyHostToDevice), "copy tok");
    ck(hipMemcpy(d_sel, sel.data(), sel.size() * 4, hipMemcpyHostToDevice), "copy sel");
    ck(hipMemcpy(d_bounds, bnd.data(), bnd.size() * 4, hipMemcpyHostToDevice), "copy bounds");

    mmq::quantize(d_x, d_tok, d_xq, GGML_TYPE_Q4_K, N, N, rows, nullptr);
    mmq::GatherGroup grp;
    grp.first = 0;
    grp.n = kN;
    for (int q = 0; q < kN; ++q) grp.blob[q] = (const uint8_t*) d_blobs + (size_t) q * (2 * gu_bytes + d_bytes);
    if (!mmq::gather_native_group(grp, gu_bytes, gu_bytes, 2 * gu_bytes, d_bytes, gu_stage, 2 * gu_bytes, d_stage,
                                  d_bytes, nullptr)) {
        std::printf("gather refused\n");
        return 3;
    }
    mmq::Context ctx;
    mmq::Product p{};
    p.type = GGML_TYPE_Q4_K;
    p.n = kN;
    p.xq = d_xq;
    p.bounds = d_bounds;
    p.ids = d_sel;
    p.total_rows = rows;
    p.max_rows = R;
    p.w = gu_stage;
    p.w_rows = 2 * FF;
    p.w_cols = N;
    // the product's per-expert stride: test one matrix (gu_bytes) against the gather's two (2*gu_bytes)
    p.expert_bytes = std::getenv("PROBE_STRIDE_HALF") ? gu_bytes : 2 * gu_bytes;
    p.dst = d_gu;
    p.ld_dst = 2 * FF;
    ck(hipMemset(d_gu, 0, (size_t) rows * 2 * FF * 4), "zero dst");   // does the product ACCUMULATE into dst (fixup)?
    ctx.run(p, nullptr);
    ck(hipDeviceSynchronize(), "sync");

    std::vector<float> gu((size_t) rows * 2 * FF);
    ck(hipMemcpy(gu.data(), d_gu, gu.size() * 4, hipMemcpyDeviceToHost), "copy gu");
    // reference for row 0, columns 0, 1 and FF (gate vs up), and column FF+1
    std::vector<float> wr((size_t) std::max(N, FF)), gater((size_t) N), upr((size_t) N);
    const int64_t cols_gate[4] = {0, 1, 7, (int64_t) FF - 1};
    {   // dump the staged slots and the product's output rows: does slot 0 hold expert 0's gate, and is out row 0 its result?
        std::vector<uint8_t> st((size_t) kN * 2 * gu_bytes);
        ck(hipMemcpy(st.data(), gu_stage, st.size(), hipMemcpyDeviceToHost), "copy staging");
        const uint64_t* w0 = (const uint64_t*) st.data();
        const uint64_t* w1 = (const uint64_t*) (st.data() + 2 * gu_bytes);
        const uint64_t* b0 = (const uint64_t*) blobs.data();
        const uint64_t* b1 = (const uint64_t*) (blobs.data() + 2 * gu_bytes + d_bytes);
        std::printf("staging slot0[0..1]=%016llx %016llx  blob0[0..1]=%016llx %016llx  slot1==blob1? %d  slot0==blob0? %d\n",
                    (unsigned long long) w0[0], (unsigned long long) w0[1],
                    (unsigned long long) b0[0], (unsigned long long) b0[1],
                    std::memcmp(w1, b1, 64) == 0, std::memcmp(w0, b0, 64) == 0);
        std::printf("out rows 0..3 col0: %.6g %.6g %.6g %.6g  (ref row0 = 2.4435)\n",
                    (double) gu[0], (double) gu[(size_t) 1 * 2 * FF], (double) gu[(size_t) 2 * 2 * FF],
                    (double) gu[(size_t) 3 * 2 * FF]);
    }
    std::printf("row 0 reference (fp32 dequant x fp32 x) against MMQ's gathered gate/up block:\n");
    std::printf("%-8s %-14s %-14s %-14s\n", "col", "MMQ out", "ref gate", "ref up");
    for (int64_t ci = 0; ci < 4; ++ci) {
        const int64_t c = cols_gate[ci];
        // gate row c of expert 0
        auto dot = [&](const uint8_t* mat, int64_t row, int64_t col) {
            deq_row_q4k(mat, row, N, wr.data());
            (void) col;
            return wr[0];   // the delta activation picks out column 0 of row `row`
        };
        const float rg = dot(&blobs[0], c, 0);
        const float ru = dot(&blobs[gu_bytes], c, 0);
        std::printf("%-8lld %-14.5g %-14.5g %-14.5g\n", (long long) c, gu[(size_t) 0 * 2 * FF + c], rg, ru);
    }
    // ids base: sel[0] = rows-1, so dst row rows-1 should hold row 0's product
    std::printf("ids check: ids[0]=%d, MMQ row0 == dst row %d ? (compare col 0: %.6g vs %.6g)\n", sel[0], rows - 1,
                gu[0], gu[(size_t) (rows - 1) * 2 * FF]);
    return 0;
}

int main(int argc, char** argv) {
    int R = 10, groups = 1;
    bool parity = false, no_gather = false, no_down = false, identity_ids = false, dst_host = false, no_quant_ids = false;
    int parity_experts = 1;
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (!std::strcmp(a, "--parity")) parity = true;
        else if (!std::strcmp(a, "--parity-experts") && i + 1 < argc) parity_experts = std::atoi(argv[++i]);
        if (!std::strcmp(a, "--rows") && i + 1 < argc) R = std::atoi(argv[++i]);
        else if (!std::strcmp(a, "--groups") && i + 1 < argc) groups = std::atoi(argv[++i]);
        else if (!std::strcmp(a, "--no-gather")) no_gather = true;
        else if (!std::strcmp(a, "--no-down")) no_down = true;
        else if (!std::strcmp(a, "--identity-ids")) identity_ids = true;
        else if (!std::strcmp(a, "--dst-host")) dst_host = true;
        else if (!std::strcmp(a, "--no-quant-ids")) no_quant_ids = true;
    }
    if (parity) return parity_mode(2, parity_experts);
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
    const size_t gu_half_bytes_ref = mmq::matrix_bytes(GGML_TYPE_Q4_K, FF, N);
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
    // Independent timing of the number the P4 kill criterion rests on (mmq_q4k_bench reports the same shape from a
    // different harness): one group of 16 experts, R rows each, gate+up as one [2*FF x N] product.
    {
        mmq::Product tp{};
        tp.type = GGML_TYPE_Q4_K;
        tp.n = kN;
        tp.xq = d_xq;
        tp.bounds = d_bounds;
        tp.ids = d_row_sel;
        tp.total_rows = rows;
        tp.max_rows = R;
        tp.w = gu_stage;
        tp.w_rows = 2 * FF;
        tp.w_cols = N;
        tp.expert_bytes = 2 * gu_half_bytes_ref;
        tp.dst = d_gu;
        tp.ld_dst = 2 * FF;
        for (int i = 0; i < 3; ++i) ctx.run(tp, nullptr);
        ck(hipDeviceSynchronize(), "sync warm");
        hipEvent_t e0, e1;
        ck(hipEventCreate(&e0), "event");
        ck(hipEventCreate(&e1), "event");
        const int reps = 20;
        ck(hipEventRecord(e0), "rec");
        for (int i = 0; i < reps; ++i) ctx.run(tp, nullptr);
        ck(hipEventRecord(e1), "rec");
        ck(hipEventSynchronize(e1), "sync");
        float ms = 0.f;
        ck(hipEventElapsedTime(&ms, e0, e1), "elapsed");
        ms /= (float) reps;
        const double fl = 2.0 * (double) rows * 2.0 * FF * (double) N / 1e9;
        std::printf("PROBE TIMING: rows %lld, gate+up %.4f ms -> %.0f GFLOP/s (mmq_q4k_bench reports the same shape "
                    "from its own harness)\n", (long long) rows, ms, fl / (ms * 1e-3));
        ck(hipEventDestroy(e0), "d");
        ck(hipEventDestroy(e1), "d");
    }
    std::printf("PROBE OK: the engine's sequence completes\n");
    return 0;
}
