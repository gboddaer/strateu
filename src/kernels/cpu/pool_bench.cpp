// src/kernels/cpu/pool_bench.cpp - plan 2026-10-17-kolibri1-speed.md task 6 (P1a): the CPU expert pool's
// measured ceiling, measured BEFORE any pool code is changed.
//
// Why: the port's performance target was set from an arithmetic estimate (findings K24) built on a wrong
// expert width, and the plan de-scoped the pool on the strength of it.  The rule adopted after that (goal
// level): every derived number is cross-checked against >= 2 independent sources before it gates a decision.
// This tool is the measurement side of that rule.
//
// It drives the kernels the pool actually runs - strata::kernels::cpu::native_gu_rows / native_down_rows,
// ggml-cpu's own vec_dot over the GGUF weight form - on the real Kolibri-1 geometry (n_embd 2560, n_ff 512,
// Q4_K gate/up + Q4_K down = a 2.21 MB blob), and reports:
//
//   [1] one expert blob, single core: GFLOP/s and GB/s.
//   [2] the same over 1 / 6 / 12 / 24 / 48 pool workers: the scaling curve, and the implied tok/s at
//       300 expert-blobs/token (6 routed x 50 layers = 663 MB, measured 660 MB by the engine's --counters).
//       Run under `numactl --cpunodebind=N --membind=N` for the one-socket numbers and under plain
//       `numactl --interleave=all` for the two-socket numbers; the delta is the NUMA item's size.
//   [3] the per-layer barrier: ExpertPool::run_split_multi_native with the SAME jobs, so the pool's own
//       per-phase counters (ms_multi_gu / ms_multi_q / ms_multi_down, multi_bytes) attribute the time to the
//       work or to the synchronisation - the question pool.hpp:198 says cannot be answered without the split.
//   [4] the result-return path priced from the S0 byte counts (1.8 MB D2H + 7.7 MB H2D per token), which is
//       what the pinned ring buffer (K15/K18) removes; it must be counted before any ceiling is quoted.
//
// Kolibri's engine uses its own ~20-line shard fan-out (kolibri_engine.cu:92, where the K7
// min(cpu_threads, K) cap lives), NOT this class; [2]/[3] are therefore also the reference numbers for the
// task-7 port onto ExpertPool.
//
// Usage: pool_bench [--threads 1,6,12,24,48] [--experts 6] [--reps 20]

#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if __has_include("ggml.h")
#include "ggml.h"
#define PB_HAVE_GGML 1
#endif

using namespace strata::kernels::cpu;
using Clock = std::chrono::steady_clock;
static double ms_since(const Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static std::vector<int> split_ints(const std::string& s) {
    std::vector<int> v;
    size_t p = 0;
    while (p <= s.size()) {
        size_t q = s.find(',', p);
        int x = std::atoi(s.substr(p, q == std::string::npos ? std::string::npos : q - p).c_str());
        if (x > 0) v.push_back(x);
        if (q == std::string::npos) break;
        p = q + 1;
    }
    return v;
}

int main(int argc, char** argv) {
    // Kolibri-1 geometry, cross-checked three ways: load_test.cpp:78 asserts n_ff == 512; the blob arithmetic
    // (512*(1440+1440) + 2560*288 = 2.21 MB x 19200 = 42.5 GB, matching the loader's 42.90 GiB); and the
    // published 3.46 B active.  Q4_K gate/up + Q4_K down.
    const int64_t n_embd = 2560, n_ff = 512;
    int gu_type = 12, d_type = 12;   // GGML_TYPE_Q4_K
#if PB_HAVE_GGML
    gu_type = d_type = GGML_TYPE_Q4_K;
#endif
    int reps = 20, n_experts = 6, blob_pool = 1024, rows = 1;
    std::vector<int> thread_list = {1, 6, 12, 24, 48};
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (a == "--experts" && i + 1 < argc) n_experts = std::atoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) { thread_list = split_ints(argv[++i]); }
        else if (a == "--blob-pool" && i + 1 < argc) { blob_pool = std::atoi(argv[++i]); }
        else if (a == "--rows" && i + 1 < argc) { rows = std::atoi(argv[++i]); }
    }
    if (!native_experts_available()) { std::fprintf(stderr, "pool_bench: no ggml-cpu native expert path in this build\n"); return 1; }
    NativeFmt f;
    std::string err;
    if (!native_fmt(gu_type, d_type, n_embd, n_ff, f, err)) {
        std::fprintf(stderr, "pool_bench: native_fmt(gu=%d d=%d %lld %lld): %s\n", gu_type, d_type,
                     (long long)n_embd, (long long)n_ff, err.c_str());
        return 1;
    }
    const size_t blob_bytes = f.bytes;
    // gate dot + up dot + silu*mul, then the down projection.
    const double flop_per = 2.0 * 2.0 * (double)n_ff * (double)n_embd + 2.0 * 2.0 * (double)n_embd * (double)n_ff;
    std::printf("== pool_bench  blob %.3f MB (gu_row %zu x %lld, up +%zu, down_row %zu x %lld at +%zu)\n",
                blob_bytes / 1e6, f.gu_row, (long long)n_ff, f.up_off, f.d_row, (long long)n_embd, f.down_off);
    std::printf("   %.1f MFLOP per expert-token;  %.4f bytes/FLOP;  %d experts/token x 50 layers = %d blobs\n",
                flop_per / 1e6, blob_bytes / flop_per, n_experts, n_experts * 50);

    std::vector<uint8_t> blob(blob_bytes);
    for (size_t i = 0; i < blob_bytes; ++i) blob[i] = (uint8_t)(i * 37 + 11);
    // A pool of DISTINCT blobs, cycled so every layer reads a cold set.  Without this the working set is
    // n_experts x 2.21 MB = 13.3 MB, which fits the 60 MB L3 of the E5-2690 v4, and the bench measures L3
    // bandwidth (94.8 GB/s, above one socket's 68 GB/s theoretical - the tell) instead of DRAM.
    std::vector<std::vector<uint8_t>> pool_blobs((size_t)blob_pool, blob);
    for (int b = 0; b < blob_pool; ++b)
        for (size_t i = 0; i < blob_bytes; i += 64) pool_blobs[b][i] = (uint8_t)(b * 131 + i);
    const double pool_mb = (double)blob_pool * blob_bytes / 1e6;
    std::vector<float> x((size_t)n_embd, 0.001f), h((size_t)n_ff, 0.001f);
    std::vector<uint8_t> act(f.act_bytes), hq(f.h_bytes);
    native_quant_act(f, x.data(), act.data());
    native_quant_h(f, h.data(), hq.data());
    std::vector<float> ff((size_t)rows * n_ff), out((size_t)rows * n_embd, 0.f);

    // [1] single core, direct kernels (no pool at all): the pure per-blob ceiling.  --rows N reproduces the
    // PREFILL shape: N activation rows sharing one cold blob (measured in-situ: ~20 rows/expert/chunk).  The
    // in-situ pool leg runs these same kernels at 9.5 GFLOP/s/core; this isolates kernel shape from call pattern.
    {
        std::vector<const void*> aps((size_t)rows, act.data());
        std::vector<const void*> hps((size_t)rows, hq.data());
        std::vector<float*> fps((size_t)rows), ops((size_t)rows);
        for (int r = 0; r < rows; ++r) {
            fps[r] = ff.data() + (size_t) r * n_ff;
            ops[r] = out.data() + (size_t) r * n_embd;
        }
        auto t0 = Clock::now();
        for (int r = 0; r < reps; ++r) {
            native_gu_rows(f, blob.data(), aps.data(), rows, fps.data(), 0, (int)n_ff);
            native_down_rows(f, blob.data(), hps.data(), rows, ops.data(), 0, (int)n_embd);
        }
        const double ms = ms_since(t0) / reps;
        std::printf("\n[1] single core, direct kernels, %d rows/blob: %.3f ms/blob-call -> %.1f GFLOP/s, %.1f GB/s (%.3f ms/row)\n",
                    rows, ms, rows * flop_per / (ms * 1e-3) / 1e9, blob_bytes / (ms * 1e-3) / 1e9, ms / rows);
    }

    // [2]+[3] through ExpertPool on the native path: n_experts jobs of one token each, per layer.
    std::printf("\n[2] ExpertPool::run_split_multi_native, %d experts per call (= one token's routed set):\n", n_experts);
    std::printf("    %8s %11s %11s %10s %10s %12s %12s\n", "workers", "ms/layer", "ms/token", "tok/s", "GB/s",
                "gu ms/lay", "dn ms/lay");
    std::printf("    blob pool: %d distinct blobs = %.0f MB (L3 is 60 MB; a set must be cold)\n", blob_pool, pool_mb);
    for (int T : thread_list) {
        std::vector<std::vector<float>> outs_h((size_t)n_experts, std::vector<float>((size_t)n_embd, 0.f));
        std::vector<std::vector<float>> outs((size_t)n_experts, std::vector<float>((size_t)n_embd, 0.f));
        std::vector<ExpertJobMulti> jobs((size_t)n_experts);
        for (int e = 0; e < n_experts; ++e) {
            jobs[e].blob = pool_blobs[(size_t)e % pool_blobs.size()].data();
            jobs[e].nt = 1;
            jobs[e].nact[0] = act.data();
            jobs[e].out[0] = outs[e].data();
        }
        ExpertPool pool(T, true, true, PoolAffinity::All);
        // warm the pool (first call pays thread creation)
        pool.run_split_multi_native(f, jobs.data(), n_experts);
        pool.ms_multi_gu = pool.ms_multi_q = pool.ms_multi_down = 0.0;
        pool.multi_bytes = 0;
        const int layers = std::max(1, reps);
        auto t0 = Clock::now();
        for (int r = 0; r < layers; ++r) {
            // rotate the blob set every layer so each layer's 6 blobs are cold in L3
            for (int e = 0; e < n_experts; ++e)
                jobs[e].blob = pool_blobs[((size_t)r * n_experts + e) % pool_blobs.size()].data();
            pool.run_split_multi_native(f, jobs.data(), n_experts);
        }
        const double ms_layer = ms_since(t0) / layers;
        const double ms_tok = ms_layer * 50.0;                       // 50 MoE layers per token
        const double gbs = (double)n_experts * blob_bytes / (ms_layer * 1e-3) / 1e9;
        std::printf("    %8d %11.3f %11.2f %10.1f %10.1f %12.1f %12.1f\n", pool.workers(), ms_layer, ms_tok,
                    ms_tok > 0 ? 1000.0 / ms_tok : 0.0, gbs,
                    pool.ms_multi_gu / layers, pool.ms_multi_down / layers);
        std::printf("             phases per layer: gu %.3f ms  quantize %.3f ms  down %.3f ms  (barrier+rest %.3f ms = %.1f%%);"
                    " streamed %.1f MB/layer\n", pool.ms_multi_gu / layers,
                    pool.ms_multi_q / layers, pool.ms_multi_down / layers,
                    ms_layer - (pool.ms_multi_gu + pool.ms_multi_q + pool.ms_multi_down) / layers,
                    100.0 * (ms_layer - (pool.ms_multi_gu + pool.ms_multi_q + pool.ms_multi_down) / layers) / ms_layer,
                    pool.multi_bytes / 1e6 / layers);
    }

    // [4] the result-return path, priced (S0 measured bytes; the pinned ring buffer removes it).
    {
        const double d2h = 1.8, h2d = 7.7, link = 12.0;   // MB/token, PCIe 3.0 x16 effective GB/s
        const double ms = (d2h + h2d) / 1000.0 / link * 1000.0;   // MB -> GB, / GB/s, s -> ms
        std::printf("\n[4] result-return path (S0 measured): %.1f MB D2H + %.1f MB H2D per token over %.0f GB/s"
                    " = %.2f ms/token\n    -> add this to any pool ceiling above; the pinned ring buffer (K15/K18)"
                    " removes it.\n", d2h, h2d, link, ms);
    }
    return 0;
}
