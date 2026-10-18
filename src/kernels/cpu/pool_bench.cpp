// src/kernels/cpu/pool_bench.cpp - plan 2026-10-17-kolibri1-speed.md task 6 (P1a): the CPU expert pool's
// measured ceiling, BEFORE any pool code is changed.
//
// Why this exists: the port's performance target was set from an arithmetic estimate (findings K24) that used a
// wrong expert width, and the plan then de-scoped the pool on the strength of it.  The rule adopted after that
// (goal-level): every derived number is cross-checked against >= 2 independent sources before it gates a
// decision.  This tool is the measurement side of that rule for the pool.
//
// It measures, on the box that will run the model, with the kernels the pool actually calls
// (strata::kernels::cpu::native_gu_rows / native_down_rows - ggml-cpu's own vec_dot over the GGUF weight form):
//
//   1. one expert blob (the real geometry: n_embd 2560, n_ff 512, Q4_K gate/up + Q4_K down = 2.21 MB):
//      GFLOP/s and GB/s, single core.
//   2. the same blob set scaled over 1 / 6 / 12 / 24 / 48 threads: the scaling curve, so the socket-crossing
//      penalty is visible (run under `numactl --cpunodebind=N --membind=N` for the one-socket numbers).
//   3. the per-layer host barrier: Pool::run across N shards with a no-op body, i.e. the floor cost the pool pays
//      50 times per token before any expert math.
//   4. the result-return path: the per-layer D2H/H2D bytes the current design moves (measured in S0: 1.8 MB D2H +
//      7.7 MB H2D per token) priced at the link rate, which is what the pinned-ring-buffer change (K15/K18) buys.
//
// Usage:  pool_bench [--threads 1,6,12,24,48] [--reps N]
//
// STATUS (2026-10-18): DRAFT, not built, not wired into CMake.  The body below drives native_gu_rows /
// native_down_rows directly, which is right for [1] and [2] (the per-blob kernel ceiling).  For [3] and [4] it
// must NOT re-implement the pool: the real API is `ExpertPool` (include/strata/kernels/cpu/pool.hpp:150) taking
// `ExpertJob*` / `ExpertJobMulti*` job arrays - the lambda the engine passes at kolibri_engine.cu:744 is the
// engine's own wrapper, not the pool's signature.  The pool also already measures what [3] needs:
//   - ms_multi_gu / ms_multi_q / ms_multi_down  (per-phase ms, accumulated over its lifetime)
//   - multi_bytes                               (bytes streamed)
//   - a three-phase barrier breakdown (park / drain / re-park) documented at pool.hpp:198, which is exactly the
//     "is the pool slow at the WORK or at the SYNCHRONISATION" question task 6 has to answer.
//   - run_split / run_split_multi_native        (the row-split path that streams every expert at all threads,
//     which is the fix for the K7 6-worker cap) - so the bench should compare `run` vs `run_split` directly.
// So: [1]/[2] as written (fix the `&ff.data()` temporaries -> named float* locals), [3] via ExpertPool's own
// counters with run vs run_split, [4] as arithmetic from the S0 byte counts.
// The numbers go into docs/superpowers/plans/2026-10-17-kolibri1-speed-ledger.md (task 6 row).

#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#if __has_include("ggml.h")
#include "ggml.h"
#define PB_HAVE_GGML 1
#endif

using Clock = std::chrono::steady_clock;
static double ms_since(const Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

int main(int argc, char** argv) {
    // Kolibri-1 geometry, cross-checked three ways (load_test.cpp asserts n_ff == 512; the blob arithmetic;
    // the published 3.46 B active).  Q4_K gate/up + Q4_K down.
    const int64_t n_embd = 2560, n_ff = 512;
    int gu_type = 12, d_type = 12;  // GGML_TYPE_Q4_K
#if PB_HAVE_GGML
    gu_type = d_type = GGML_TYPE_Q4_K;
#endif
    int reps = 20;
    std::vector<int> thread_list = {1, 6, 12, 24, 48};

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) {
            thread_list.clear();
            for (auto& tok : [] (const std::string& s) {
                std::vector<int> v; size_t p = 0;
                while (p <= s.size()) { size_t q = s.find(',', p); v.push_back(std::atoi(s.substr(p, q == std::string::npos ? std::string::npos : q - p).c_str())); if (q == std::string::npos) break; p = q + 1; }
                return v;
            }(argv[i + 1])) if (tok > 0) thread_list.push_back(tok);
            ++i;
        }
    }

    if (!strata::kernels::cpu::native_experts_available()) {
        std::fprintf(stderr, "pool_bench: this build has no ggml-cpu native expert path\n");
        return 1;
    }
    strata::kernels::cpu::NativeFmt f;
    std::string err;
    if (!strata::kernels::cpu::native_fmt(gu_type, d_type, n_embd, n_ff, f, err)) {
        std::fprintf(stderr, "pool_bench: native_fmt(gu=%d, d=%d, %lld, %lld) failed: %s\n", gu_type, d_type,
                     (long long)n_embd, (long long)n_ff, err.c_str());
        return 1;
    }
    const size_t blob_bytes = f.bytes;
    // FLOPs for one expert-token: gate row-dot + up row-dot + silu*mul, then the down projection.
    //   gu: 2 * n_ff * n_embd MACs -> 2 FLOP each;  d: 2 * n_embd * n_ff MACs.
    const double flop_per_expert = 2.0 * 2.0 * (double)n_ff * (double)n_embd + 2.0 * 2.0 * (double)n_embd * (double)n_ff;

    std::printf("== pool_bench: expert blob %.3f MB (gu_row %zu B x %lld rows, up at +%zu, down_row %zu B x %lld rows at +%zu)\n",
                blob_bytes / 1e6, f.gu_row, (long long)n_ff, f.up_off, f.d_row, (long long)n_embd, f.down_off);
    std::printf("   blob FLOP/expert-token = %.1f MFLOP; bytes/FLOP = %.4f\n", flop_per_expert / 1e6,
                blob_bytes / flop_per_expert);

    // one blob, filled with a deterministic pattern (the dot cost is data-independent for these formats)
    std::vector<uint8_t> blob(blob_bytes);
    for (size_t i = 0; i < blob_bytes; ++i) blob[i] = (uint8_t)(i * 37 + 11);
    std::vector<float> x((size_t)n_embd, 0.001f), ff((size_t)n_ff), out((size_t)n_embd, 0.f);
    std::vector<uint8_t> act(f.act_bytes), hq(f.h_bytes);
    strata::kernels::cpu::native_quant_act(f, x.data(), act.data());
    std::vector<float> h((size_t)n_ff, 0.001f);
    strata::kernels::cpu::native_quant_h(f, h.data(), hq.data());
    const void* const actp = act.data();
    const void* const hqp = hq.data();

    auto one_expert = [&](void) {
        strata::kernels::cpu::native_gu_rows(f, blob.data(), &actp, 1, &ff.data(), 0, (int)n_ff);
        strata::kernels::cpu::native_down_rows(f, blob.data(), &hqp, 1, &out.data(), 0, (int)n_embd);
    };

    // ---- 1. single core -------------------------------------------------
    {
        auto t0 = Clock::now();
        for (int r = 0; r < reps; ++r) one_expert();
        const double ms = ms_since(t0) / reps;
        std::printf("\n[1] single core, one expert blob: %.3f ms  ->  %.1f GFLOP/s, %.1f GB/s\n", ms,
                    flop_per_expert / (ms * 1e-6) / 1000.0, blob_bytes / (ms * 1e-3) / 1e9);
    }

    // ---- 2. thread scaling (each worker on its own blob copy) -----------
    std::printf("\n[2] thread scaling (each worker computes its own expert, blob copies are separate):\n");
    std::printf("    %8s %12s %14s %14s %18s\n", "threads", "ms/expert", "GFLOP/s", "GB/s", "tok/s @663MB/tok");
    for (int T : thread_list) {
        std::vector<std::vector<uint8_t>> copies((size_t)T, blob);
        const int per = std::max(1, reps);
        auto t0 = Clock::now();
        std::vector<std::thread> th;
        for (int i = 0; i < T; ++i) {
            th.emplace_back([&, i]() {
                float ffl[(size_t)n_ff], outl[(size_t)n_embd];
                const void* ap = act.data();
                const void* hp = hq.data();
                for (int r = 0; r < per; ++r) {
                    strata::kernels::cpu::native_gu_rows(f, copies[i].data(), &ap, 1, &ffl, 0, (int)n_ff);
                    strata::kernels::cpu::native_down_rows(f, copies[i].data(), &hp, 1, &outl, 0, (int)n_embd);
                }
            });
        }
        for (auto& t : th) t.join();
        const double ms = ms_since(t0) / ((double)per * T);   // wall ms per expert, aggregated
        const double gf = flop_per_expert / (ms * 1e-6) / 1000.0;
        const double gbs = blob_bytes / (ms * 1e-3) / 1e9;
        // a token touches 6 routed experts x 50 layers = 300 expert-blobs = 663 MB (measured by --counters: 660)
        const double tok_ms = 300.0 * ms;
        std::printf("    %8d %12.3f %14.1f %14.1f %18.1f\n", T, ms, gf, gbs, tok_ms > 0 ? 1000.0 / tok_ms : 0.0);
    }

    // ---- 3. the per-layer host barrier ---------------------------------
    {
        strata::kernels::cpu::Pool pool;
        for (int T : {6, 12, 24, 48}) {
            pool.start(T);
            volatile int sink = 0;
            auto t0 = Clock::now();
            for (int r = 0; r < 200; ++r) pool.run([&](int shard) { sink += shard; }, T);
            const double us = ms_since(t0) * 1000.0 / 200.0;
            std::printf("\n[3] Pool::run barrier, %d shards, empty body: %.1f us per layer  ->  %.2f ms per token "
                        "(x50 layers)\n", T, us, us * 50.0 / 1000.0);
            pool.stop();
        }
    }

    // ---- 4. the result-return path (K15/K18) ---------------------------
    {
        const double d2h_mb = 1.8, h2d_mb = 7.7;   // S0 measured, per token, pool path
        for (double link : {12.0}) {               // PCIe 3.0 x16 effective GB/s on this box
            const double ms = (d2h_mb + h2d_mb) / link * 1000.0;
            std::printf("\n[4] result-return path: %.1f MB D2H + %.1f MB H2D per token over %.0f GB/s = %.2f ms/token\n"
                        "    -> that is %.0f%% of the pool's token time at the measured ceiling; the pinned ring\n"
                        "       buffer (K15/K18) removes it, so it must be counted before the ceiling is quoted.\n",
                        d2h_mb, h2d_mb, link, ms, 0.0);
        }
    }
    return 0;
}
