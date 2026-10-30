// src/kolibri/kolibri_engine.cu - the engine path for Kolibri 1 (GGUF architecture kolibri1): a driver of its
// own beside Flash-Next's, on the src/qwen35 template (plan docs/superpowers/plans/2026-10-16-kolibri1.md).
// It grows task by task: T3 the stub, T6 the reference forward (kolibriref.cu), T7 this file - the weight load,
// the expert repack and the hybrid placement; T8 the layer driver and serve protocol.  `strata` switches to it
// from the file's architecture at the top of its main(); strata-kolibri wraps the same entry point.
//
// The model (docs/KOLIBRI1.md): 50 layers, every layer MoE (384 routed experts top-6 + 1 ungated shared,
// FFN 512, n_embd 2560); the router selects on `logits + expert_bias` and weights by the unbiased
// `sigmoid(logits)`, no renorm (expert_gating_func 5); GQA 48/4 x 128 with q/k RMSNorm; per-layer
// four sliding-window (513, RoPE) to one full-attention (NoPE); sandwich norms around both branches.
// No GDN, no hyper-connections, no indexer, no MTP, no PLE.
//
// The weights, as the plan's Global Constraints fix them: the dense tensors (attention, router, shared expert,
// norms, embeddings) sit on the GPU in their GGUF types (the MMVQ kernels read them as is); the 19,200 routed
// experts are repacked once at load into native [gate|up|down] blobs (native_expert_layout) in a pinned host
// table - the GPU is the expert cache, the model is bigger than any PC's VRAM (Q4_K_M: 42.6 GiB of experts,
// 3.5 GiB of dense).  --load-only runs just the load and reports.
#include "strata/kolibri/engine.hpp"
#include "strata/kolibri/decode_wait.hpp"
#include "pf_combine.hpp"   // G1: the prefill MoE combiner (src/kolibri/pf_combine.cu)

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/family_layout.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/dequant_bf16.hpp"   // S7 dense-fp16 cache: dequant_f16 (Q4_K blocks -> FP16 bits on device)
#include "strata/kernels/kolibri_window.hpp"
#include "strata/kernels/kolibri_hot_policy.hpp"   // hot_due / hot_round: the residency round's arithmetic (host-only)
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/elementwise.hpp"   // doorbell_publish: the decode handoff's publication node
#if defined(STRATA_KOLIBRI_MMQ)
#include "strata/prefill/moe_mmq.hpp"   // P4: the prompt path's per-group Q4_K product (parity-gated, ledger §P4)
#endif
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"   // P1b: the library's ExpertPool (row-split, affinity-aware) for the CPU experts
#include "ggml.h"
#include "ggml-cpu.h"
#include <atomic>
#include <charconv>
#include <new>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <functional>
#include "strata/kernels/native_router.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "strata/prefill/gemm.hpp"
#include <cuda_fp16.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#define KC(x)                                                                                                             \
    do {                                                                                                                  \
        cudaError_t e_ = (x);                                                                                             \
        if (e_ != cudaSuccess) {                                                                                          \
            std::fprintf(stderr, "strata-kolibri: %s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__,         \
                         __LINE__);                                                                                       \
            std::exit(1);                                                                                                 \
        }                                                                                                                 \
    } while (0)

namespace {

size_t row_bytes(int type, int64_t n) {
    int be = 0, bb = 0;
    if (!strata::block_geometry((uint32_t) type, be, bb)) return 0;
    return (size_t) (n / be) * (size_t) bb;
}
size_t lcm(size_t a, size_t b) { size_t x = a, y = b; while (y) { size_t t = x % y; x = y; y = t; } return a / x * b; }

struct Dense {
    const uint8_t* d = nullptr;   // device
    int type = -1;
    std::vector<int64_t> shape;
    size_t row_bytes = 0;
    // S7 dense-fp16 cache (STRATA_KOLIBRI_DENSE_F16=1): the Q4_K blocks dequantized to FP16 ONCE at load, so the
    // prefill GEMMs run cublas fp16/fp32-acc (Gemm::f16) with no per-call dequant.  Measured: gemm.native
    // dequantizes the whole weight into scratch on EVERY call (gemm.hpp:5,43) - the shared-expert FFN's 7.5 ms
    // per launch is that dequant, 553 GFLOP at ~6% of the T4.  Layout: GGUF shape[0]=K, shape[1]=N, so the
    // device blocks are N rows of K values = exactly the [N,K] row-major W that Gemm::f16 takes.
    uint16_t* f16 = nullptr;
};

// The dense-fp16 gate, read once.  Absent/0 = the plain path (the oracle's path) untouched.
static inline bool dense_f16_on() {
    static const bool on = std::getenv("STRATA_KOLIBRI_DENSE_F16") != nullptr &&
                           std::string(std::getenv("STRATA_KOLIBRI_DENSE_F16")) != "0";
    return on;
}

// An env gate that reads the VALUE (0 = off): the plain presence test (getenv != nullptr) turns the feature ON
// for an empty or "0" value - which silently voided an A/B (both arms on the same path).
static inline bool env_on(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != 0 && std::atoi(v) != 0;
}

// S8b/B1: the vectorized F32 decode gemv (the router).  Declared here because `gemv` (the dispatcher) is far above the
// per-feature gates; bitwise by construction, env-gated until measured.
// A2b: the two-chunk prefill wavefront.  Step 1 allocates the second activation set and changes no scheduling.
static inline bool pf_pipe_on() { return env_on("STRATA_KOLIBRI_PF_PIPE"); }

// G1 (plan docs/superpowers/plans/2026-10-24-kolibri-pf-partial-sums.md): the prefill partial-sum path.  Opt-in;
// unset, every byte of the default path is what it was.  Refused together with PF_PIPE at startup (see the check
// in the config print): the resident rows live in one Scratch-wide buffer (s.res_ey) and the partial ring is
// per-ActSet, so two chunks in flight would share them - and PF_PIPE is exactly the two-chunks-in-flight mode.
static inline bool pf_partial_on() { return env_on("STRATA_KOLIBRI_PF_PARTIAL"); }

static inline bool dec_gemv_v4_on() {
    // ON by default: bitwise (oracle BITWISE PASS with it enabled, and the router feeds the ids), two-run measured at
    // tg 26.7 vs 25.1/25.2 for the scalar read.  STRATA_KOLIBRI_DEC_GEMV_V4=0 is the A/B arm.
    const char* v = std::getenv("STRATA_KOLIBRI_DEC_GEMV_V4");
    return v == nullptr || (v[0] != 0 && std::atoi(v) != 0);
}

// The same value-parse for a gate that is ON unless the env says otherwise (an unset gate must not silently
// disable a default: "STRATA_KOLIBRI_PF_PIN=0" is the A/B arm, an absent variable is the shipped behaviour).
static inline bool env_on_default_on(const char* name) {
    const char* v = std::getenv(name);
    return v == nullptr || (v[0] != 0 && std::atoi(v) != 0);
}

// S7 expert fp16 cache: experts per layer to keep VRAM-resident (STRATA_KOLIBRI_EXPERT_F16=N).  0 = off.
static inline int expert_f16_n() {
    static const int n = std::getenv("STRATA_KOLIBRI_EXPERT_F16") ? std::atoi(std::getenv("STRATA_KOLIBRI_EXPERT_F16")) : 0;
    return n;
}

// S7 prong A: the prefill scores-buffer cap, in floats (STRATA_KOLIBRI_ATTN_MB, MiB; default 48 = the old
// kAttScoreFloats 12<<20).  The tq loop batches this many score elements at once; larger caps (e.g. 384) cut
// the K/V re-read amplification and give cublas real M at deep prefixes.  Gate-off (default) = bitwise today.
// P1b: the tiled path stores softmax probabilities as fp16 before the PV product.  At 32K+ keys each probability is
// ~1e-5, where fp16 keeps barely three digits, so the PV sum loses precision that grows with the key count - measured
// against the llama.cpp reference, the layer-1 relative divergence rises 0.0017 -> 0.020 -> 0.037 for 512 -> 8K -> 98.6K
// tokens.  Scaling by a power of two before the store (and folding 1/S into the PV GEMM's alpha) preserves the fp16
// mantissa at no cost: no extra pass, no extra memory.  Default 1 = today's behaviour, bit for bit.
static inline float pv_scale() {
    static const float s = [] {
        const char* v = std::getenv("STRATA_KOLIBRI_PV_SCALE");
        const float f = v ? (float) std::atof(v) : 1.0f;
        return f > 0.f ? f : 1.0f;
    }();
    return s;
}

static inline int64_t attn_cap_floats() {
    static const int64_t mb = [] {
        const char* v = std::getenv("STRATA_KOLIBRI_ATTN_MB");
        int64_t m = v ? (int64_t) std::atoi(v) : (int64_t) 48;
        return m < 1 ? (int64_t) 48 : m;
    }();
    return mb * ((1 << 20) / 4);   // floats; 48 MiB = 12,582,912 floats
}

// S7 prong-B pricing: split the tq loop into scores / softmax / PV wall segments (env-gated; the syncs
// serialize the stream, so measurement-only runs only).
static inline bool attn_phases_on() {
    static const bool on = env_on("STRATA_KOLIBRI_ATTN_PHASES");
    return on;
}

// S7 prong B-lite: batch the tq loop by head-group at M = the whole chunk (T queries per GEMM) instead of
// tq queries per GEMM.  cublas ran M=42 at 0.8%% of tensor peak (phase timers: scores 133.5 s of the 166 s
// attention term at 32K); M=512 shapes are real tensor tiles.  The cap-sized buffers hold per-slot slices
// [slot][T][ldp] (Bh reusable slots, consumed in stream order); the softmax rows and per-item PV shapes are
// unchanged - only the cublas M/batch, the softmax grid and the slice bases change.  Gate-off = exact current
// path.  Needs one head's whole-chunk scores to fit the cap: T*ldp <= cap floats.
static inline bool attn_kg_on() {
    static const bool on = env_on("STRATA_KOLIBRI_ATTN_KG");
    return on;
}

// S7 prong B: the prompt scores via k_attn_scores_simt (deterministic fp32 SIMT tiling, fp16 operands converted
// at the smem tiles, k=0..127 fmaf chain in order) instead of the cublas scores GEMM.  Same [H][nt][ldp] att_s
// layout, so softmax16 and the PV GEMM are untouched.  Gate-off = exact cublas path.
static inline bool attn_simt_on() {
    static const bool on = env_on("STRATA_KOLIBRI_ATTN_SIMT");
    return on;
}

// S7 prong B (full flash): fuse the prefill tq loop into ONE k_attn_flash launch per layer (scores+softmax+PV
// in-kernel; no fp32 score/p16 round trips).  Gate-off = exact current path.  Guards at dispatch: the kolibri
// shape the kernel hard-codes (H 48 / KV 4 / D 128); any other shape falls through to the cublas path.
static inline bool attn_flash_on() {
    static const bool on = env_on("STRATA_KOLIBRI_ATTN_FLASH");
    return on;
}

// S7 cycle-2 instrument: dump one layer's att_m (the prefill's attention output) to a binary file, so the
// FIRST NaN cell (row, head, dim) can be read.  Env STRATA_KOLIBRI_ATTN_DUMP=<path>; the layer index comes
// from STRATA_KOLIBRI_ATTN_DUMP_LAYER (default 0).  Measurement-only: the copy is after the layer's
// attention and does not touch the numerics.
static inline const char* attn_dump_path() {
    static const char* p = std::getenv("STRATA_KOLIBRI_ATTN_DUMP");
    return p;
}
static inline int attn_dump_layer() {
    static const int l = std::getenv("STRATA_KOLIBRI_ATTN_DUMP_LAYER")
                             ? std::atoi(std::getenv("STRATA_KOLIBRI_ATTN_DUMP_LAYER")) : 0;
    return l;
}

// Reuse-gate instrument: dump the logits of the LAST prompt position.  Measurement-only and off unless
// STRATA_KOLIBRI_LOGITS_DUMP is set: the copy is the one the sampler already takes, so the numerics are
// untouched.  This exists because a bitwise comparison of generated ids cannot say WHY two arms differ -
// greedy sampling turns a 1e-6 logit difference into a different token and hides the scale of the cause.
static inline const char* reuse_gate_dump_path() {
    static const char* p = std::getenv("STRATA_KOLIBRI_LOGITS_DUMP");
    return p;
}

// Same instrument for the FINAL HIDDEN STATE at the generating position: the logits are one gemv away from
// it, so if the hidden states of two arms agree while their logits differ the divergence is the output
// projection, and if the hidden states already differ it is inside the stack.  Measurement-only.
static inline const char* reuse_gate_hidden_path() {
    static const char* p = std::getenv("STRATA_KOLIBRI_HIDDEN_DUMP");
    return p;
}

// And for the K/V themselves, at the layer named by STRATA_KOLIBRI_ATTN_DUMP_LAYER.  Comparing the two arms'
// bytes says whether the state handed over by the prefix request differs at all, and if it does, the first
// differing row is the position the reuse path got wrong.  Measurement-only.
static inline const char* reuse_gate_kv_path() {
    static const char* p = std::getenv("STRATA_KOLIBRI_KV_DUMP");
    return p;
}

// The K/V buffers are laid out [kvh][stride][D], where `stride` is swa for a sliding layer and ctx for a
// full-attention one.  A plain copy of rows*HKV*D halves is only correct when stride == rows; for a
// full-attention layer it runs past the first head's rows into unrelated positions and every row index after
// that is meaningless.  Copy head by head, and record the stride so the parser can be sure.
static inline bool reuse_gate_copy_kv(const __half* src, int64_t stride, int64_t n_rows, int HKV, int D,
                                      std::vector<__half>& out) {
    out.resize((size_t) n_rows * (size_t) HKV * (size_t) D);
    for (int h = 0; h < HKV; ++h) {
        if (cudaMemcpy(out.data() + (size_t) h * (size_t) n_rows * (size_t) D,
                       src + (size_t) h * (size_t) stride * (size_t) D,
                       (size_t) n_rows * (size_t) D * sizeof(__half),
                       cudaMemcpyDeviceToHost) != cudaSuccess)
            return false;
    }
    return true;
}



struct LayerBlobs {
    strata::kernels::NativeExpertLayout L;
    size_t stride = 0;              // blob slot bytes, block-aligned (the lcm of the types' super-blocks)
    uint8_t* table = nullptr;       // pinned host: n_expert slots + 1 zero slot (the no-pool overflow path)
    uint8_t* dev = nullptr;         // --expert-slots: device table, slots slots + 1 zero slot
    bool native_ok = false;         // native_expert_supported for this layer's (gu, d) at these dims
    // The CPU pool's OWN capability.  native_ok is ANDed with the GPU's native_expert_supported, so gating the
    // pool branch on it excludes exactly the layers the GPU path is worst at: measured, the pool ran on 25 of 50
    // layers per token and the GPU paid PCIe (mmvq, 43 ms/token, 75% of GPU busy) on the other 25.  The pool's
    // native_fmt supports the Q6_K-down layers the GPU grouped kernel cannot, so the pool branch must key on
    // this, not on native_ok.
    bool pool_ok = false;
    strata::kernels::cpu::NativeFmt fmt;  // the CPU pool's view of the same blob (native_fmt at load)
    // S7 expert fp16 cache (STRATA_KOLIBRI_EXPERT_F16=N): the first N experts of this layer dequantized to FP16
    // ONCE at load, device-resident.  Per expert the layout is [gate (n_ff x n_embd) | up | down (n_embd x n_ff)],
    // the same slice order as the blob, so a slice is one dequant_f16 of the blob's ggml block rows.  Routed
    // experts in [0, n_cached) can run their GEMMs as Gemm::f16 (cublas fp16/fp32-acc) with no per-call dequant
    // and no PCIe blob read; the rest stay on the pool / pinned-table path.  Increment A: cache only, no routing.
    uint16_t* f16 = nullptr;
    int n_cached = 0;
    // S7 increment B: slot -> expert id (-1 = free) and last-touch chunk, for LRU promotion.  The coverage probe
    // showed the hot experts are scattered across the id space (top-40 by frequency = 82.8% of touches), so the
    // cache must hold the HOT experts, not ids 0..N-1: a touched expert that misses is promoted into a free slot,
    // evicting the least-recently-touched, and pf_touch's LPT order (biggest-first) promotes the chunk's biggest
    // experts first.
    std::vector<int32_t> f16_slot_id;
    std::vector<uint64_t> f16_slot_last;
};

/// A persistent worker pool: run(f, n) splits [0, n) over n shards across the pool (caller runs shard 0) and
/// returns when all shards are done.  --cpu-threads sizes it; the CPU expert pool is its user.
struct Pool {
    int n_threads = 0;
    bool pin = false;               // STRATA_KOLIBRI_PF_PIN=0 reproduces the unpinned A/B arm
    std::vector<int> cores;         // one pinnable physical core per worker lane
    std::vector<std::thread> workers;
    std::mutex mu;
    std::condition_variable work_cv, done_cv;
    std::function<void(int)> fn;
    unsigned long gen = 0;          // one generation per run(): a worker consumes each generation once
    std::atomic<int> left{0};
    int n_shards = 0;
    bool quit = false;
    void start(int t, const std::vector<int>& lane_cores) {
        n_threads = t;
        // **THE PREFILL LANES HAD NO AFFINITY AT ALL.**  The library ExpertPool pins every worker to a physical
        // core; this pool - the one S7's prefill fan-out actually uses - spawned plain threads.  Under
        // `numactl --cpunodebind=1` this node's cpuset is 28 logical CPUs = 14 physical cores x 2, so 24
        // unpinned lanes float over 14 cores: lanes share cores, and the per-chunk barrier waits for whichever
        // lane loses the SMT toss.  Measured on the 32K prefill (chunks=4000, 568.77 GB): shard bytes PERFECTLY
        // equal (min = max = 28.21 GB) with a **2.0x shard-busy spread** (max 49,709 ms / mean 34,785 ms) and
        // 10.7 GB/s, against the library pool's 34.0 GB/s on the SAME kernels on the SAME node (pool_bench:
        // 12 workers 34.5, 24 workers 34.0, 48 collapse to 2.8).  Equal bytes and unequal time is the signature
        // of a placement problem, not a kernel one, so pin.
        cores = pin ? lane_cores : std::vector<int>();
        // **DERIVE THE MAP BEFORE PINNING, NEVER AFTER.**  `physical_cores()` reads the CALLING thread's affinity,
        // so once the host lane is pinned to one core the map comes back EMPTY - a probe run in that order
        // printed "pinned 0 worker lanes" while looking successful.  The map is therefore computed at startup
        // (before any pinning) and passed in; an empty map here is a bug, not a quiet fallback to the old
        // unpinned behaviour, so say so.
        if (pin && cores.empty())
            std::fprintf(stderr,
                         "strata-kolibri: prefill pool: PINNING REQUESTED BUT THE CORE MAP IS EMPTY - the lanes run"
                         " unpinned (the caller derived the topology after it was pinned)\n");
        for (int i = 1; i < t; ++i) workers.emplace_back([this, i] {
            if (pin) {
                const size_t idx = (size_t) (i - 1);
                const int core = idx < cores.size() ? cores[idx] : -1;
                if (core >= 0) strata::kernels::cpu::pin_current_thread(core);
            }
            unsigned long my_gen = 0;
            for (;;) {
                std::unique_lock<std::mutex> lk(mu);
                work_cv.wait(lk, [&] { return quit || gen > my_gen; });
                if (quit) return;
                my_gen = gen;
                const int n = n_shards;   // a run with fewer shards than workers leaves the idle ones waiting
                if (i >= n) continue;
                lk.unlock();
                fn(i);
                lk.lock();
                if (--left == 0) done_cv.notify_one();
            }
        });
    }
    ~Pool() { { std::lock_guard<std::mutex> lk(mu); quit = true; } work_cv.notify_all(); for (auto& w : workers) w.join(); }
    void run(std::function<void(int)> f, int n) {
        if (n <= 1) { f(0); return; }
        { std::lock_guard<std::mutex> lk(mu); fn = std::move(f); n_shards = n; left = n - 1; ++gen; }
        work_cv.notify_all();
        fn(0);
        std::unique_lock<std::mutex> lk(mu);
        done_cv.wait(lk, [&] { return left.load() == 0; });
    }
};

struct Engine {
    strata::core::ModelGeometry g;
    int64_t top_k = 6;
    float eps = 1e-6f, rope_base = 1e4f;
    int cpu_threads = 0;            // --cpu-threads: routed experts on the CPU pool (ggml-cpu vec_dot, no PCIe)
    int pf_lanes = 0;               // the prefill fan-out's lane count (host lane + one lane per pinnable core)
    std::vector<int> pf_cores;      // the pinnable physical cores, derived ONCE at startup before any pinning
    int64_t pf_total = 0;           // the prefill's total prompt length (the POOL_ALL rule keys on it)
    int slots = 0;                  // --expert-slots: experts per layer resident in VRAM (the GPU expert cache)
    // EXPERT-CACHE FITTING AND HOTNESS RESIDENCY (opt-in; with the knobs off every field below is unused and the
    // cache is allocated by expert id exactly as before).  --expert-slots auto fits the slot count to the VRAM
    // that is still free once the dense weights, the KV and the scratch are allocated (the Qwen engine's method);
    // --expert-slots-hot picks WHICH experts occupy those slots by how often the router picks them instead of by
    // id, which is what caps the hit rate at slots/n_expert today; --expert-slots-profile saves and seeds that set.
    int vram_reserve_mib = 700;     // --vram-reserve-mib: room kept free for the CUDA runtime (the Qwen default)
    int hot_on = 0;                 // --expert-slots-hot or STRATA_KOLIBRI_SLOTS_HOT=1
    int hot_min_picks = 8192;       // re-select only after this many routed picks (no per-token churn)
    long long hot_picks_at_last = 0;
    // THE ROUND: the Qwen3.8-Flash-Next engine's adaptive tier (include/strata/kernels/kolibri_hot_policy.hpp carries
    // the rules and the tests).  A round runs every --expert-slots-reselect-every POSITIONS (32; 0 = only after a
    // request), moves at most --expert-slots-swaps experts mid-request (96) or --expert-slots-swaps-final after a
    // request (2048), admits a candidate only when it beats the occupant by --expert-slots-min-gain (1.5), and
    // multiplies the score it sorts on by --expert-slots-decay (0.7) after every round.  The clock of 2026-10-27 is
    // gone: a wall clock moved a different number of positions on every box, so two runs of one prompt disagreed.
    int hot_every = 32;
    long long hot_pos_at_last = 0;
    int hot_swaps = 96;
    int hot_swaps_final = 2048;
    double hot_min_gain = 1.5;
    double hot_decay_factor = 0.7;
    long long hot_promotions = 0;   // blobs re-copied into the slots (each one is n_layers x stride of PCIe traffic)
    std::string hot_profile;        // --expert-slots-profile PATH: the learned hot set, read at load and saved after
    // --expert-slots-per-layer (DEFAULT ON since 2026-10-27; --no-expert-slots-per-layer or
    // STRATA_KOLIBRI_SLOTS_PER_LAYER=0 restores one global set): ONE residency map per layer instead of one set shared by
    // all 50.  The storage was already per layer (`e.blobs`, one cudaMalloc each); only `slot_of` and the histogram
    // forced one set everywhere.  Measured on this box (findings-log, "0.5"): the same 75 slots/layer cover 29.9% as one
    // global set and 69.3% as 75 per-layer sets; on the repeat prompt one set went 34.8% -> 34.5% where per layer went
    // 34.8% -> 81.5% -> 89.4%.  With the knob off, slot_of and hot_cnt are n_expert long and every index below is `id`,
    // so the off path is byte-identical to the release before this.  It only acts when the hot path is on
    // (`--expert-slots-hot`, which the server config passes); with the by-id cache nothing below is touched.
    int per_layer = 1;
    // expert index into slot_of / hot_cnt: the layer's own row when per_layer, the single global row otherwise.
    size_t hot_idx(int64_t l, int id) const {
        return per_layer ? (size_t) (l * (int64_t) g.n_expert + id) : (size_t) id;
    }
    std::vector<int> slot_of;       // (per_layer ? layer*n_expert : n_expert) entries: expert -> slot, -1 = not resident
                                    // (hot mode; empty in the id path)
    mutable std::vector<long long> hot_cnt;  // (per_layer ? layer*n_expert : n_expert) routed-pick histogram: counted
                                             // from layer_step, whose Engine is const (it reads the layer, it does not
                                             // own the cache)
    mutable std::vector<double> hot_score;   // the same shape, +1.0 per routed pick and *decay once per round: the score
                                             // the swap policy sorts on.  hot_cnt stays the LIFETIME count that
                                             // hot_profile_save writes, so a saved profile means what it always did.
    int grouped = 0;                // --grouped: native_expert_grouped for the routed experts (one launch per layer)
    int graphs = 0;                 // --grouped + graphs (default on with --grouped): per-layer CUDA graphs
    int prefill = 0;                // --prefill: chunked prompt processing (GEMM projections, FP16 attention)
    // --counters: measurement only, no numerics effect.  The speed plan's gates need the numbers the engine
    // actually moves: slot hits/misses (the expert bytes over PCIe per token), the D2H/H2D traffic per layer,
    // and the per-request KV clear cost.
    struct Counters {
        long long tok = 0;            // decode tokens measured (forward() calls)
        long long slot_hits = 0;      // routed picks served from VRAM
        long long slot_miss = 0;      // routed picks that read the pinned table / the CPU pool
        long long slot_bytes = 0;     // expert bytes pulled over PCIe by those misses
        long long d2h_n = 0, d2h_bytes = 0;
        long long h2d_n = 0, h2d_bytes = 0;
        // K29: decode residency - picks served from the fp16 cache against picks made.
        long long dec_res_hits = 0, dec_res_picks = 0;
        long long reset_ms = 0, reset_n = 0;
        // Which expert branch each layer-step actually took.  The nsys trace showed GPU mmvq at 575 launches/token
        // (= 48 layers' gate+up for all 6 picks) WHILE the pool branch ran 26x/token - i.e. both sides computing
        // the routed experts.  The branch counts settle which layers go where; they are the input to task 9's
        // placement, and they make every future placement change readable from CNT without a profiler.
        long long br_pool = 0, br_gpu = 0, br_slots = 0, br_grouped = 0;
        // S7 prong-B pricing (STRATA_KOLIBRI_ATTN_PHASES): the prefilm tq loop's three segments, ns, all layers.
        long long attn_sc_ns = 0, attn_sm_ns = 0, attn_pv_ns = 0;
        // S6: the decode's per-layer segments, ns (env-gated by STRATA_KOLIBRI_DEC_PHASES): the pre-expert
        // chain, the ids+h D2H wait, the pool call, and the post-expert chain - the decode's 47 ms/token split.
        long long dec_pre_ns = 0, dec_d2h_ns = 0, dec_pool_ns = 0, dec_post_ns = 0, dec_layers = 0;
        // S7: the PREFILL's per-layer segments, ns (the ATTN_PHASES gate): dense GEMMs, attention, the expert
        // leg (GPU grouped + CPU pool), and the layer's other kernels - the pp512 breakdown.
        long long pf_dense_ns = 0, pf_attn_ns = 0, pf_exp_ns = 0, pf_other_ns = 0, pf_layers = 0;
        // The tail marks: the region between pf_other (the pool drain) and the next layer had NO mark, which is where
        // pp512's ~1670 ms of unattributed time lives - the scatter + H2D, the shared expert, and the MoE sum + residual.
        long long pf_post_ns = 0, pf_shx_ns = 0, pf_tail_ns = 0;
    };
    mutable Counters cnt;   // layer_step takes the Engine as const (like `mutable Pool pool`)
    bool counters = false;
    // S7: the chunked prefill's pool leg had NO counter at all (the CNT pool line reads e.epool, which the
    // prefill path does not use), so a 13.9 tok/s pp was unreadable - grouped bytes, achieved rate and the
    // pool/GPU split were all invisible.  These are that leg's counters.
    long long pf_pool_ms = 0, pf_pool_bytes = 0, pf_chunks = 0, pf_experts = 0;
    // THE POOL'S REAL WORK IS ROWS, NOT BYTES.  Removing the fp16 cache made the pool leg 3.6x slower (126.6 s ->
    // 454.0 s at 32K) while its bytes rose only 1.33x, so the leg is arithmetic-bound and every byte-rate quoted
    // for it (10.7 GB/s "against a 34 GB/s roof") was comparing the wrong quantity.  Count the rows the pool
    // actually computes and print GFLOP/s - the CPU pool's measured roof is 226 GFLOP/s.
    std::atomic<long long> pf_rows{0};
    // Per-shard bytes for the prefill leg.  The leg calls the engine's ~20-line shard fan-out (e.pool.run), the
    // construct whose decode-side defect was "worker i runs exactly shard i"; if the work concentrates in a few
    // shards, that is the whole 0.06 GB/s/core story.  Each shard writes only its own slot, so no atomics needed.
    std::vector<long long> pf_shard_bytes = std::vector<long long>(64, 0);   // sized at declaration: the load path holds a const Engine&
    std::vector<long long> pf_shard_ns = std::vector<long long>(64, 0);      // per-shard BUSY time: window/max-busy = the true imbalance
    // S7: coverage of a hypothetical prefill slot cache, measured on the chunk's real id set.  The wiring design
    // needs "what fraction of a chunk's picks land under S", and the grouped leg never copies ids to the host,
    // so this is the only way to price it.  Gated on --counters: it adds a 12 KB D2H, so it must never run in a
    // timed run that did not ask for counters.
    long long pf_hit_entries = 0, pf_entry_n = 0, pf_uniq_sum = 0, pf_cov_chunks = 0;
    long long pf_f16_promotes = 0;   // S7 increment B: fp16 cache promotions (LRU churn measure)
    // G4 prerequisite: the fill gate ("fill bytes per prompt <= 2.5 GB, from 7.9") had NO counter behind it - the
    // engine counted promotions but never the bytes they moved, so the gate could not be measured, only argued.
    // This counts bytes actually written into the resident slots (gate + up + down planes, FP16).
    long long pf_fill_bytes = 0;
    // S7 expert-cache coverage probe (STRATA_KOLIBRI_EXPERT_HIST=1): per-layer counts of routed touches per expert
    // id, over the whole run.  The fp16 VRAM cache fits ~2036 of 19,200 experts (10.6%); whether it pays is
    // decided by the touch SKEW - top-K coverage of this histogram is the cache's hit-rate curve.
    bool hist_on = false;
    std::vector<std::vector<uint64_t>> touch_hist;   // [layer][expert] touch counts
    long long pf_stage_chunks = 0, pf_stage_bytes = 0;   // S7 staging: chunks served from VRAM, bytes staged
    // S7 phase timers: the pool leg is 1.76 GFLOP/s/core against pool_bench's 43.7 on the SAME kernels (25x),
    // so the term is inside this loop.  Name it: gu rows / h-quantize / down rows, per 64-row batch.
    std::atomic<long long> pf_gu_ns{0}, pf_qh_ns{0}, pf_dn_ns{0};
    cublasHandle_t blas = nullptr;
    strata::prefill::Gemm gemm;
    std::vector<unsigned long long> gexec;   // cudaGraphExec_t per layer (0 = the layer is not capturable)
    mutable Pool pool;              // the worker pool behind the CPU expert path
    // P1b (plan 2026-10-17-kolibri1-speed.md task 7): the CPU expert path on the library's ExpertPool, which
    // splits every expert's rows across ALL workers.  The Pool above assigns one expert per shard, so with
    // K = 6 routed experts it caps at 6 threads (the K7 min(cpu_threads, K)) and streams each 2.21 MB blob at
    // one core's bandwidth.  Kept as a pointer: ExpertPool is not default-constructible-then-assigned.
    mutable std::unique_ptr<strata::kernels::cpu::ExpertPool> epool;
    std::map<std::string, Dense> dense;
    std::map<std::string, float*> f32;
#if defined(STRATA_KOLIBRI_MMQ)
    mutable std::unique_ptr<strata::prefill::mmq::Context> mmq_ctx;
#endif
    std::vector<LayerBlobs> blobs;   // per layer
};

// ------------------------------------------------------------------ the forward kernels (the kolibriref set, plain:
// exactness first; the native grouped-expert path over the blob table arrives with the placement work in T8+)
enum : int { T_F32 = 0, T_F16 = 1, T_Q8_0 = 8, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14, T_BF16 = 30 };

__device__ __forceinline__ float h2f(uint16_t h) {
    const uint32_t s = (uint32_t) (h >> 15) << 31, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e == 0) { const float v = ldexpf((float) m, -24); return s ? -v : v; }
    const uint32_t f = e == 31 ? (s | 0x7f800000u | (m << 13)) : (s | ((e + 112) << 23) | (m << 13));
    return __uint_as_float(f);
}
__device__ __forceinline__ uint16_t ld16(const uint8_t* p) { return (uint16_t) (p[0] | (p[1] << 8)); }
__device__ __forceinline__ void scale_min_k4(int j, const uint8_t* q, int& d, int& m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
// T4: packed (32-bit, branch-free) Q4_K/Q6_K nibble extraction.  Same float expression per element as the scalar
// form, so the two are bitwise identical by construction; mirrored from STRATA_KOLIBRI_PACKED_DEQ at startup, the
// way g_attn_coalesced and g_rope_f64 are.
__device__ int g_packed_deq = 0;
template <int T> __device__ void deq32(const uint8_t* row, int c, float* out) {
    if constexpr (T == T_F32) { const float* r = (const float*) row + 32 * c; for (int l = 0; l < 32; ++l) out[l] = r[l]; }
    else if constexpr (T == T_F16) { const uint8_t* r = row + 64 * c; for (int l = 0; l < 32; ++l) out[l] = h2f(ld16(r + 2 * l)); }
    else if constexpr (T == T_BF16) { const uint8_t* r = row + 64 * c; for (int l = 0; l < 32; ++l) out[l] = __uint_as_float((uint32_t) ld16(r + 2 * l) << 16); }
    else if constexpr (T == T_Q8_0) { const uint8_t* b = row + 34 * c; const float d = h2f(ld16(b)); const int8_t* q = (const int8_t*) (b + 2); for (int l = 0; l < 32; ++l) out[l] = d * (float) q[l]; }
    else if constexpr (T == T_Q4_K) {
        const uint8_t* b = row + 144 * (c >> 3); const int j = c & 7;
        const float d = h2f(ld16(b)), dmin = h2f(ld16(b + 2));
        int sc, m; scale_min_k4(j, b + 4, sc, m);
        const float d1 = d * (float) sc, m1 = dmin * (float) m;
        const uint8_t* qs = b + 16 + 32 * (j >> 1);
        // T4: the packed form loads 32 bytes as 8 words and selects the nibble by a uniform (hoisted) flag instead of
        // a branch per element.  Measured on the R9700: +21 % on a [2560 x 2560] Q4_K GEMV, bitwise identical
        // (tools/kolibri/grid_shape_bench.cu, docs/AMD_R9700_PORT.md §T2).
        if (g_packed_deq) {
            const uint32_t* w32 = (const uint32_t*) qs;
            const bool hi = (j & 1) != 0;
            #pragma unroll
            for (int w = 0; w < 8; ++w) {
                const uint32_t v = w32[w];
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const uint32_t byte = (v >> (8 * k)) & 0xFFu;
                    out[w * 4 + k] = d1 * (float) (hi ? (int) (byte >> 4) : (int) (byte & 0xF)) - m1;
                }
            }
        } else {
            for (int l = 0; l < 32; ++l) out[l] = d1 * (float) ((j & 1) ? (qs[l] >> 4) : (qs[l] & 0xF)) - m1;
        }
    } else if constexpr (T == T_Q5_K) {
        const uint8_t* b = row + 176 * (c >> 3); const int j = c & 7;
        const float d = h2f(ld16(b)), dmin = h2f(ld16(b + 2));
        int sc, m; scale_min_k4(j, b + 4, sc, m);
        const float d1 = d * (float) sc, m1 = dmin * (float) m;
        const uint8_t* qh = b + 16; const uint8_t* qs = b + 48 + 32 * (j >> 1);
        for (int l = 0; l < 32; ++l) { const int q = ((j & 1) ? (qs[l] >> 4) : (qs[l] & 0xF)) + ((qh[l] >> j) & 1 ? 16 : 0); out[l] = d1 * (float) q - m1; }
    } else if constexpr (T == T_Q6_K) {
        const uint8_t* b = row + 210 * (c >> 3); const int j = c & 7, n = j >> 2, r = j & 3;
        const uint8_t* ql = b + 64 * n; const uint8_t* qh = b + 128 + 32 * n;
        const int8_t* sc = (const int8_t*) (b + 192) + 8 * n;
        const float d = h2f(ld16(b + 208));
        if (g_packed_deq) {
            // T4: the r branch hoisted out of the loop and the bytes read four at a time (the half of ql and the
            // nibble are both fixed once r is), with the per-element float expression unchanged - bitwise against the
            // scalar form, which the bench's parity arm asserts element by element.
            const uint32_t* qhw = (const uint32_t*) qh;
            // The two independent facts about r: the ql half (r odd takes ql + 32) and the nibble (r >= 2 takes the
            // high one).  Conflating them is the bug the parity arm caught on its first run.
            const uint32_t* src = (r & 1) ? ((const uint32_t*) ql + 8) : (const uint32_t*) ql;
            const bool upper = (r >= 2);
            const int sh = 2 * r;
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                const uint32_t lw = src[w], hw = qhw[w];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int l = w * 4 + k;
                    const int qlb = (int) ((lw >> (8 * k)) & 0xFFu), qhb = (int) ((hw >> (8 * k)) & 0xFFu);
                    const int q = (upper ? (qlb >> 4) : (qlb & 0xF)) | (((qhb >> sh) & 3) << 4);
                    out[l] = d * (float) sc[(l / 16) + 2 * r] * (float) (q - 32);
                }
            }
        } else
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16; int q;
            if (r == 0) q = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
            else if (r == 1) q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
            else if (r == 2) q = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
            else q = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
            out[l] = d * (float) sc[is + 2 * r] * (float) (q - 32);
        }
    }
}
__device__ float block_sum(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if ((int) threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s]; __syncthreads(); }
    const float r = sh[0];
    __syncthreads();
    return r;
}
constexpr int kThreads = 128;
// K25: the single-pass / two-pass seam, raisable for testing.  The accuracy gate for the tiled online softmax is
// the SAME prompt at the SAME position through both paths, which needs the threshold to be a knob, not a
// constant; production default is 11000 (44 KB dynamic + 2.5 KB static, under the 48 KB T4 per-block cap).
static int g_two_pass_max = 11000;
cudaStream_t g_st = nullptr;   // the router kernels require a created stream (kolibriref does the same)
// S6: the decode's coalesced score dot flag, mirrored to the device (the kernel body is device code).
__device__ bool g_attn_coalesced = false;
__device__ bool g_attn_pv4 = false;   // S8b: the PV 4-chain variant (host mirrors it at startup)
/// S6: the decode's dense gemv from the PRE-DEQUANTIZED fp16 weights (STRATA_KOLIBRI_DENSE_F16): no per-token
/// Q4_K dequant (the decode's pre-expert chain measured 0.9 ms/layer = 66%% of the decode's wall, at 36 GB/s
/// effective = 6%% of the T4's bandwidth - the dequant, not the bandwidth).  The x vector is converted to fp16
/// once per call site; the dot accumulates in fp32.
/// The row body both F16 grids share (one function, two grids: the per-row arithmetic cannot drift between them).
__device__ __forceinline__ float row_dot_f16(const __half* row, const __half* x, int n_in, float* sh) {
    float acc = 0.f;
    for (int c = threadIdx.x * 8; c < n_in; c += blockDim.x * 8) {
        const uint4 wv = *reinterpret_cast<const uint4*>(row + c);
        const uint4 xv = *reinterpret_cast<const uint4*>(x + c);
        const __half* wh = reinterpret_cast<const __half*>(&wv);
        const __half* xh = reinterpret_cast<const __half*>(&xv);
#pragma unroll
        for (int l = 0; l < 8; ++l) acc = fmaf(__half2float(wh[l]), __half2float(xh[l]), acc);
    }
    return block_sum(acc, sh);
}
__global__ void k_gemv_f16(const __half* __restrict__ W, const __half* __restrict__ x, int n_in, float* y, int n_out) {
    __shared__ float sh[kThreads];
    const int o = blockIdx.x;
    y[o] = row_dot_f16(W + (size_t) o * (size_t) n_in, x, n_in, sh);
}

// T3 (plan 2026-10-28-kolibri-r9700-speed.md): the same row under a capped grid.  A GEMV with a large row count
// pays for one block per row - 128 000 single-row blocks at the LM head's shape - and above kGemvGsMinOut rows a
// grid-stride form over the SAME rows is faster: measured on the R9700 at n_out = 128 000, 1.902 ms against
// 2.120 ms, +10.3 % (tools/kolibri/grid_shape_bench.cu, docs/AMD_R9700_PORT.md §T2).  Below that it is SLOWER -
// n_out = 2 560 measured 33.6 us against 26.1 us, -29 %, because 256 blocks leave this card's 32 CUs under-fed on a
// 3.7 MB weight.  So the cap is gated on the row count, never applied blanket: shipped shapes (2 560) take the
// block-per-row form whatever the knob says, and the shipped output is byte-identical by construction.  The row
// body is shared with the block-per-row kernel (one function, two grids), so the arithmetic order per row cannot
// drift between them.
constexpr int kGemvGsMinOut = 16384;   // the measured crossover band, not a guess: 2 560 loses, 128 000 wins
constexpr int kGemvGsBlocks = 1024;
static inline bool gemv_gs_on() {
    static const bool v = env_on("STRATA_KOLIBRI_GEMV_GS");
    return v;
}
static void gemv_gs_note(int n_out) {
    static bool noted = false;
    if (noted) return;
    noted = true;
    std::fprintf(stderr, "strata-kolibri: STRATA_KOLIBRI_GEMV_GS: grid-stride GEMV, %d row(s) over %d blocks\n", n_out,
                 kGemvGsBlocks);
}

__global__ void k_gemv_f16_gs(const __half* __restrict__ W, const __half* __restrict__ x, int n_in, float* y,
                              int n_out) {
    __shared__ float sh[kThreads];
    for (int o = blockIdx.x; o < n_out; o += gridDim.x) y[o] = row_dot_f16(W + (size_t) o * (size_t) n_in, x, n_in, sh);
}

/// The quantised-row body both `k_gemv` grids share, for the same reason.
template <int T>
__device__ __forceinline__ float row_dot_q(const uint8_t* row, const float* xs, int n_in) {
    float acc = 0.f, v[32];
    for (int c = threadIdx.x; c < n_in / 32; c += blockDim.x) {
        deq32<T>(row, c, v);
        const float* xc = xs + 32 * c;
        for (int l = 0; l < 32; ++l) acc += v[l] * xc[l];
    }
    return acc;
}
template <int T>
__global__ void k_gemv_gs(const uint8_t* W, size_t row_bytes, int n_in, const float* x, float* y, int n_out) {
    __shared__ float sh[kThreads];
    for (int o = blockIdx.x; o < n_out; o += gridDim.x)
        y[o] = block_sum(row_dot_q<T>(W + (size_t) o * row_bytes, x, n_in), sh);
}

template <int T>
__global__ void k_gemv(const uint8_t* W, size_t row_bytes, size_t expert_bytes, const int* ids, int n_in, const float* x,
                       size_t x_stride, float* y, int n_out) {
    __shared__ float sh[kThreads];
    const int o = blockIdx.x, sel = blockIdx.y;
    const int e = ids ? ids[sel] : 0;
    const uint8_t* row = W + (size_t) e * expert_bytes + (size_t) o * row_bytes;
    const float* xs = x + (size_t) sel * x_stride;
    const float acc = block_sum(row_dot_q<T>(row, xs, n_in), sh);
    if (threadIdx.x == 0) y[(size_t) sel * n_out + o] = acc;
}
// S8b/B1: the DECODE's F32 dense gemv - the router, 3.75 MiB/layer read on every token - runs at 57 GB/s (69 us a call,
// 3.44 ms/token) against a 0.62 ms bandwidth floor, because k_gemv<T_F32> reads each 32-float chunk as 32 separate
// 4-byte loads.  This reads the row as eight 16-byte float4 loads instead and keeps EVERYTHING else identical: the
// chunk order (thread tid walks c = tid, tid+128, ... over n_in/32 chunks), the l = 0..31 ascending accumulation, and
// the same 128-lane block_sum tree - so it is bitwise by construction, which the oracle checks.
__global__ void k_gemv_f32_v4(const float* __restrict__ W, const float* __restrict__ x, int n_in, float* y, int n_out) {
    __shared__ float sh[kThreads];
    const int o = blockIdx.x;
    const float* row = W + (size_t) o * (size_t) n_in;
    float acc = 0.f;
    for (int c = threadIdx.x; c < n_in / 32; c += blockDim.x) {
        const float4* r4 = reinterpret_cast<const float4*>(row + 32 * c);
        const float4* x4 = reinterpret_cast<const float4*>(x + 32 * c);
        float v[32];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const float4 a = r4[k], b = x4[k];
            v[4 * k + 0] = a.x * b.x;
            v[4 * k + 1] = a.y * b.y;
            v[4 * k + 2] = a.z * b.z;
            v[4 * k + 3] = a.w * b.w;
        }
        for (int l = 0; l < 32; ++l) acc += v[l];   // same ascending order as k_gemv's acc += v[l] * xc[l]
    }
    acc = block_sum(acc, sh);
    if (threadIdx.x == 0) y[o] = acc;
}

/// S7: the prefill router GEMM, tiled over 32 tokens per block.  nsys priced k_gemv<T_F32> at 19 ms x 100 =
/// 1.94 s of the pp512 wall: 196k tiny blocks (one per expert x token), each reading the 10 KB row, launch-
/// and latency-bound at 53 GFLOP/s.  Here one block per (expert, 32-token tile) loads the row ONCE and reuses
/// it across 32 tokens.  BITWISE-identical to k_gemv<T_F32> by construction: with n_in/32 = 80 chunks and 256
/// threads, k_gemv's thread tid sums exactly chunk tid (one iteration), l = 0..31 sequential, then block_sum
/// over 256 lanes (lanes 80..255 contribute 0) - this kernel reproduces that per-token order exactly.
// S7: the token tile per thread is a template parameter (TJ): the per-token sum is UNCHANGED (the same chunk
// order, the same l order, the same 128-lane block_sum tree), so every tile size is bitwise-identical to
// k_gemv<T_F32>.  The tile size trades ILP against parallelism: TJ=32 gives each thread 32 accumulators (a
// long dependent FMA chain, 80 of 128 lanes active) - the nsys trace priced the router at 20.85 ms per call
// (31% of the 4K prefill), ~390x off the bandwidth roof, i.e. latency-bound; a smaller tile gives more CTAs
// (more SM occupancy) and a shorter dependency chain.
template <int TT, int TJ> __global__ void k_router_tile(const uint8_t* W, const float* x, size_t x_stride, float* y,
                                                        int n_in, int T, int n_out) {
    __shared__ float sh[kThreads];
    const int o = blockIdx.x, t0 = blockIdx.y * TJ;
    const uint8_t* row = W + (size_t) o * (size_t) (n_in * 4);   // T_F32 row, as bytes for deq32
    float acc[TJ];
    #pragma unroll
    for (int j = 0; j < TJ; ++j) acc[j] = 0.f;
    const int c = threadIdx.x;   // n_in/32 = 80 chunks: lanes 0..79 active, the rest idle (acc 0) - as in k_gemv
    if (c < n_in / 32) {
        float v[32];
        deq32<TT>(row, c, v);
        for (int j = 0; j < TJ; ++j) {
            const float* xc = x + (size_t) (t0 + j) * x_stride + 32 * c;
            for (int l = 0; l < 32; ++l) acc[j] += v[l] * xc[l];
        }
    }
    #pragma unroll
    for (int j = 0; j < TJ; ++j) {
        const float a = block_sum(acc[j], sh);
        if (threadIdx.x == 0 && t0 + j < T) y[(size_t) (t0 + j) * n_out + o] = a;
    }
}

// K31: THE ROUTER RE-READS ITS ACTIVATION TILE ONCE PER EXPERT.  nsys priced it at 35% of the 32K prefill's GPU busy
// (60.7 s, 158 us/call) and the trace says it is bound on ~1.6 GB per chunk-layer of x reads: 384 blocks each pull the
// same T-tile (TJ tokens x n_in floats).  A TJ sweep from 8 to 384 changed nothing, which is what a pure x-traffic
// bound looks like.  This variant puts TK experts in ONE block: their dequantized weight rows live in registers and the
// activation tile is read ONCE and accumulated into all TK accumulators.  x traffic / TK, W traffic unchanged.
// BITWISE: each output's accumulation order (chunk c ascending, then l ascending) and the block_sum tree over 128
// threads are exactly k_router_tile's - only the block that computes each output changes.
template <int TT, int TJ, int TK>
__global__ void k_router_tile_k(const uint8_t* W, const float* x, size_t x_stride, float* y, int n_in, int T,
                                int n_out) {
    __shared__ float sh[kThreads];
    const int o0 = blockIdx.x * TK, t0 = blockIdx.y * TJ;
    float acc[TK][TJ];
#pragma unroll
    for (int k = 0; k < TK; ++k)
#pragma unroll
        for (int j = 0; j < TJ; ++j) acc[k][j] = 0.f;
    const int c = threadIdx.x;   // n_in/32 chunks: lanes 0..79 active, the rest contribute 0 - as in k_router_tile
    if (c < n_in / 32) {
        float v[TK][32];
#pragma unroll
        for (int k = 0; k < TK; ++k) {
            if (o0 + k >= n_out) continue;
            const uint8_t* row = W + (size_t) (o0 + k) * (size_t) (n_in * 4);
            deq32<TT>(row, c, v[k]);
        }
        for (int j = 0; j < TJ; ++j) {
            const float* xc = x + (size_t) (t0 + j) * x_stride + 32 * c;
            float xr[32];
#pragma unroll
            for (int l = 0; l < 32; ++l) xr[l] = xc[l];       // the ONE read this expert group shares
#pragma unroll
            for (int k = 0; k < TK; ++k)
#pragma unroll
                for (int l = 0; l < 32; ++l) acc[k][j] += v[k][l] * xr[l];
        }
    }
#pragma unroll
    for (int k = 0; k < TK; ++k)
        for (int j = 0; j < TJ; ++j) {
            const float a = block_sum(acc[k][j], sh);
            if (threadIdx.x == 0 && t0 + j < T && o0 + k < n_out) y[(size_t) (t0 + j) * n_out + (o0 + k)] = a;
        }
}

template <int T> __device__ void k_row_body(const uint8_t* W, size_t row_bytes, int row, int n, float* out) {
    const uint8_t* r = W + (size_t) row * row_bytes;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < n / 32; c += gridDim.x * blockDim.x) deq32<T>(r, c, out + 32 * c);
}
template <int T> __global__ void k_row(const uint8_t* W, size_t row_bytes, int row, int n, float* out) {
    k_row_body<T>(W, row_bytes, row, n, out);
}
/// graph decode: the row (token id) from device memory
template <int T> __global__ void k_row_p(const uint8_t* W, size_t row_bytes, const int* row_p, int n, float* out) {
    k_row_body<T>(W, row_bytes, *row_p, n, out);
}

__global__ void k_rms(const float* x, const float* w, float* y, int n, float eps) {
    __shared__ float sh[kThreads];
    const float* xg = x + (size_t) blockIdx.x * n;
    float* yg = y + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xg[i] * xg[i];
    s = block_sum(s, sh);
    const float r = 1.0f / sqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yg[i] = xg[i] * r * (w ? w[i] : 1.0f);
}
__device__ __forceinline__ float silu(float v) { return v / (1.0f + expf(-v)); }
__global__ void k_swiglu(const float* g, const float* u, float* h, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = silu(g[i]) * u[i];
}
__global__ void k_add(float* x, const float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}
// Diagnostic snapshot on the same compute queue; --dump-layers must not add per-layer host drains.
__global__ void k_layer_snapshot(const float* src, float* dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}
// S6 (per-layer round-trip removal, the cheap half): rmsnorm followed by the residual add in ONE launch.  The decode's
// chain is launch-latency bound - ~15 launches a layer at ~44 us of exposed latency each, because the host stalls on
// the per-layer D2H - so removing a launch is worth about as much as removing its arithmetic (which is trivial here).
// SAME block reduce and SAME per-element expression as k_rms + k_add, so the result is bitwise.
__global__ void k_rms_add(const float* xin, const float* w, float* acc, int n, float eps) {
    __shared__ float sh[kThreads];
    const float* xg = xin + (size_t) blockIdx.x * n;
    float* ag = acc + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xg[i] * xg[i];
    s = block_sum(s, sh);
    const float r = 1.0f / sqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) ag[i] += xg[i] * r * (w ? w[i] : 1.0f);
}
static inline dim3 grid1(int64_t n, int block = kThreads) { return dim3((unsigned) ((n + block - 1) / block)); }
__device__ float block_max(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    const float r = sh[0];
    // THE BARRIER IS LOAD-BEARING, and it is why this file was non-deterministic: the shared array is reused by the next
    // helper (block_sum in k_attn_softmax16).  Without a barrier here, a thread that finishes this read can overwrite
    // sh[0] in block_sum while another thread is still reading the max, so the softmax denominator/maximum changed from
    // run to run.  compute-sanitizer racecheck pinned 50 hazards to k_attn_softmax16 before this line existed;
    // block_sum already had the equivalent barrier, which is what made the asymmetry so easy to miss.
    __syncthreads();
    return r;
}
__global__ void k_pos_add(int* pos) { *pos += 1; }
// S7 increment B: gather token rows of an FP16 matrix by index (the resident-expert GEMMs take the gathered
// activation rows of one expert contiguously).  out[i] = x[tok[i / N] * N + i % N].
__global__ void k_gather_f16(const __half* x, const int32_t* tok, __half* out, int N_) {
    int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t t = tok[i / N_];
    out[i] = t >= 0 ? x[(size_t) t * (size_t) N_ + (int)(i % N_)] : __half(0.f);   // tok -1 = padded row
}

__global__ void k_to_f16(const float* x, __half* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}
/// S7 prong B: the prompt scores as a deterministic fp32 SIMT-tiled kernel (main's block_scores_simt structure,
/// ported to kolibri's fp16 KV and per-head scores).  grid (k-tiles 64, q-tiles 32, head-groups of 4 within one
/// kvh); 256 threads; each thread owns 2 queries x 4 heads x TN keys; the k = 0..127 fmaf chain runs IN ORDER in
/// every tiling (main's bitwise-tiling property).  fp16 q/K are converted to fp32 at the smem tile load, so the
/// FMA operands are exactly the fp16 cache bits.  The epilogue skips masked cells (key >= pos0+q+1, or below the
/// sliding window) - the softmax only reads [lo, n).  Replaces ONLY the cublas scores GEMM call (gate:
/// STRATA_KOLIBRI_ATTN_SIMT); softmax16 and the PV GEMM are untouched.
template <int QT, int NB, int KC, int TN>
__global__ void __launch_bounds__(256) k_attn_scores_simt(const __half* __restrict__ q16, const __half* __restrict__ kcache,
                                                          int H, int p0, int t0, int h0, int nt, int P, int D, int ldp,
                                                          int64_t swa, int first, float* __restrict__ att_s) {
    constexpr int ROWS = QT * 4, QSB = ROWS + 4, KSB = NB + 4, CX = NB / TN;   // h0 = kvh*G; the CTA's 4 heads = h0 + blockIdx.z*4 + (r&3)
    __shared__ __align__(16) float Qs[KC][QSB];
    __shared__ __align__(16) float Ks[KC][KSB];
    const int q0 = blockIdx.y * QT, b0 = blockIdx.x * NB;
    const int hbase = h0 + (int) blockIdx.z * 4;
    const int tid = threadIdx.x, tx = tid % CX, ty = tid / CX;
    float acc[8][TN] = {};
    for (int d0 = 0; d0 < D; d0 += KC) {
        for (int idx = tid; idx < ROWS * (KC / 8); idx += 256) {   // the q rows: one uint4 (8 halves) -> 8 smem floats
            const int r = idx / (KC / 8), u = idx % (KC / 8);
            uint4 v = make_uint4(0, 0, 0, 0);
            if (q0 + (r >> 2) < nt)
                v = *reinterpret_cast<const uint4*>(q16 + ((size_t) (t0 + q0 + (r >> 2)) * H + hbase + (size_t) (r & 3)) * D + d0 + u * 8);
            const __half* hh = reinterpret_cast<const __half*>(&v);
#pragma unroll
            for (int j = 0; j < 8; ++j) Qs[u * 8 + j][r] = __half2float(hh[j]);
        }
        for (int idx = tid; idx < NB * (KC / 8); idx += 256) {     // the key rows, same conversion
            const int c = idx / (KC / 8), u = idx % (KC / 8);
            uint4 v = make_uint4(0, 0, 0, 0);
            if (b0 + c < P)
                v = *reinterpret_cast<const uint4*>(kcache + (size_t) (b0 + c) * D + d0 + u * 8);
            const __half* hh = reinterpret_cast<const __half*>(&v);
#pragma unroll
            for (int j = 0; j < 8; ++j) Ks[u * 8 + j][c] = __half2float(hh[j]);
        }
        __syncthreads();
#pragma unroll 8
        for (int k = 0; k < KC; ++k) {
            float qr[8], kr[TN];
            *reinterpret_cast<float4*>(&qr[0]) = *reinterpret_cast<const float4*>(&Qs[k][ty * 8]);
            *reinterpret_cast<float4*>(&qr[4]) = *reinterpret_cast<const float4*>(&Qs[k][ty * 8 + 4]);
#pragma unroll
            for (int t = 0; t < TN; t += 4)
                *reinterpret_cast<float4*>(&kr[t]) = *reinterpret_cast<const float4*>(&Ks[k][tx * TN + t]);
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(qr[i], kr[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int q = q0 + ty * 2 + (i >> 2);
        if (q >= nt) continue;
        int lo, n;
        strata::kolibri::attn_mask_bounds(P, p0 + t0, q, swa, first, &lo, &n);   // per-row [lo, pos0+q+1)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int b = b0 + tx * TN + j;
            if (b >= n || b < lo) continue;
            att_s[((size_t) (h0 + blockIdx.z * 4 + (i & 3)) * (size_t) nt + (size_t) q) * ldp + b] = acc[i][j];
        }
    }
}

/// S7 prong B (full flash): the prefill's scores + softmax + PV in ONE kernel - no fp32 score or p16 round
/// trips.  Grid (ceil(T/QT), HKV); one CTA = QT consecutive queries x all G=12 heads of one kvh, so every
/// K/V key tile is loaded to shared memory ONCE and shared across the heads (the GQA traffic lever; live
/// CTAs walk the same stream, so DRAM sees ~one copy).  8 warps: warp w owns query q0+w; its G rows keep
/// the online-softmax max/sum and the PV dim-slice accumulator in REGISTERS across the whole k-loop
/// (lane = 4 dims - the dot partials' dims and the acc's dims are the same slice).  Per dot: the lane
/// partial (4 FMA, in order) + the 5-step shfl_xor butterfly (deterministic sum order); scale before max;
/// w = exp(s*scale - m); acc/l rescaled per step; final normalize by the sum -> att_m.  Mask per row via
/// the same helper the softmax uses ([lo, pos0+q+1), slice-relative rows, so one row-indexing covers both
/// the full cache and the gathered ring).  D is hard-coded 128 (the dispatch guards it); env
/// STRATA_KOLIBRI_ATTN_FLASH=1.
__global__ void __launch_bounds__(256) k_attn_flash(const __half* __restrict__ q16, const __half* __restrict__ kcache,
                                                    const __half* __restrict__ vcache, float* __restrict__ att_m,
                                                    int H, long long kstride, int T, int P, int p0, int64_t swa,
                                                    int first, float scale) {
    constexpr int QT = 8, GW = 12, KCK = 32;
    __shared__ __align__(16) __half Ks[KCK][128];
    __shared__ __align__(16) __half Vs[KCK][128];   // 16 KB: 4 CTAs/SM on sm_75 (v1 held a 24.6 KB Q tile too:
    // 1 CTA/SM, 8 warps, and the ~200-cycle dependent chains (5 chained shuffles + expf x2) per (row, key) did
    // not hide - measured 2.3x WORSE than the cublas path at 32K; the q slices now live in registers per lane.
    const int q0 = blockIdx.x * QT, kvh = blockIdx.y;
    const int hbase = kvh * GW;
    const __half* kh0 = kcache + (size_t) kvh * (size_t) kstride * 128;
    const __half* vh0 = vcache + (size_t) kvh * (size_t) kstride * 128;
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    const int dl = lane * 4;   // this lane's D-slice (the dot partials' dims and the PV acc's dims)
    // the lane's q slices, loaded once: 12 heads x 4 dims (uint2 per row, 16-B safe at dl*2 multiple)
    float qsl[GW][4];
    bool warp_live = q0 + warp < T;
    if (warp_live) {
#pragma unroll
        for (int h = 0; h < GW; ++h) {
            const __half* qh4 = q16 + ((size_t) (q0 + warp) * H + hbase + (size_t) h) * 128 + dl;
            const uint2 v = *reinterpret_cast<const uint2*>(qh4);
            const __half* hh = reinterpret_cast<const __half*>(&v);
            qsl[h][0] = __half2float(hh[0]); qsl[h][1] = __half2float(hh[1]);
            qsl[h][2] = __half2float(hh[2]); qsl[h][3] = __half2float(hh[3]);
        }
    }
    int lo9, nmax;             // the block's max causal bound: the last valid warp row's
    strata::kolibri::attn_mask_bounds(P, p0, min(q0 + 7, T - 1), swa, first, &lo9, &nmax);
    int lo_d, nn;              // this warp's row's bounds (slice-relative rows, ring and cache alike)
    strata::kolibri::attn_mask_bounds(P, p0, q0 + warp, swa, first, &lo_d, &nn);
    float m[GW], l[GW], acc[GW][4];
#pragma unroll
    for (int h = 0; h < GW; ++h) {
        m[h] = -INFINITY; l[h] = 0.f;
        acc[h][0] = acc[h][1] = acc[h][2] = acc[h][3] = 0.f;
    }
    for (int kt = 0; kt < nmax; kt += KCK) {
        const int kn = min(KCK, nmax - kt);
        for (int idx = threadIdx.x; idx < kn * 16; idx += 256) {
            const int kr = idx / 16, u = idx % 16;
            *reinterpret_cast<uint4*>(&Ks[kr][u * 8]) =
                *reinterpret_cast<const uint4*>(kh0 + (size_t) (kt + kr) * 128 + u * 8);
            *reinterpret_cast<uint4*>(&Vs[kr][u * 8]) =
                *reinterpret_cast<const uint4*>(vh0 + (size_t) (kt + kr) * 128 + u * 8);
        }
        __syncthreads();
        if (warp_live && kt < nn) {
            const int knw = min(KCK, nn - kt);
            // the k loop OUTER, the 12 heads unrolled INSIDE: one small hot body (fits the instruction cache;
            // the h-outer layout was 12 copies of the k-loop body that thrashed it every tile), K/V smem
            // reads hoisted per k, and the 12 independent shuffle chains interleaved for ILP
            for (int k = 0; k < knw; ++k) {
                if (kt + k < lo_d) continue;   // the sliding window's start (0 or 1 at this chunk size)
                const __half* kh4 = &Ks[k][dl];
                const float k0f = __half2float(kh4[0]), k1f = __half2float(kh4[1]),
                            k2f = __half2float(kh4[2]), k3f = __half2float(kh4[3]);
                const __half* vh4 = &Vs[k][dl];
                const float v0f = __half2float(vh4[0]), v1f = __half2float(vh4[1]),
                            v2f = __half2float(vh4[2]), v3f = __half2float(vh4[3]);
#pragma unroll
                for (int h = 0; h < GW; ++h) {
                    float p = qsl[h][0] * k0f;
                    p = fmaf(qsl[h][1], k1f, p);
                    p = fmaf(qsl[h][2], k2f, p);
                    p = fmaf(qsl[h][3], k3f, p);
                    p += __shfl_xor_sync(0xffffffffu, p, 16);
                    p += __shfl_xor_sync(0xffffffffu, p, 8);
                    p += __shfl_xor_sync(0xffffffffu, p, 4);
                    p += __shfl_xor_sync(0xffffffffu, p, 2);
                    p += __shfl_xor_sync(0xffffffffu, p, 1);
                    const float s = p * scale;
                    const float mn = fmaxf(m[h], s);
                    const float rsv = expf(m[h] - mn);
                    const float ce = expf(s - mn);
                    acc[h][0] = fmaf(ce, v0f, acc[h][0] * rsv);
                    acc[h][1] = fmaf(ce, v1f, acc[h][1] * rsv);
                    acc[h][2] = fmaf(ce, v2f, acc[h][2] * rsv);
                    acc[h][3] = fmaf(ce, v3f, acc[h][3] * rsv);
                    m[h] = mn;
                    l[h] = fmaf(l[h], rsv, ce);
                }
            }
        }
        __syncthreads();
    }
    if (warp_live) {
#pragma unroll
        for (int h = 0; h < GW; ++h) {
            const float il = 1.f / l[h];   // l >= 1 for any row with a visible key (its own position)
            float* o = att_m + ((size_t) (q0 + warp) * H + hbase + (size_t) h) * 128 + dl;
            o[0] = acc[h][0] * il; o[1] = acc[h][1] * il; o[2] = acc[h][2] * il; o[3] = acc[h][3] * il;
        }
    }
}

// S7 prong B (mma): the prefill's fused attention on tensor cores - mma.sync.m16n8k8 f16 (Turing) with fp32
// accumulation and NO fp32 score/p16 round trips.  Grid (ceil(T/16), HKV); block 128 = 4 warps; warp w owns
// the d-slice [32w, 32w+32) of the output and 3 of the 12 heads; the block = 16 consecutive queries of one
// kvh.  Score partials per 24-key tile cross warps through shared memory and sum in a fixed order
// (deterministic); the online softmax (m/l per row-pair in registers, the frag-max gathered with two XOR
// shuffles inside the 4-lane group) rescales the register accumulators; p.v is mma'd straight from the
// softmaxed weights (packed per row-pair in registers) against the shared V tile - the two keys' halves
// picked strided and packed per lane, exactly as the ported reference does on its [key][dim] V layout.
// Fragment maps (the PTX ISA, verified against the working reference): a = {row gid, the k-pair 2*tig..+1}
// & {row gid+8, ...}; b = {k 2*tig..+1} x {n gid}; c = {row gid: n 2tig/+1; row gid+8: n 2tig/+1}.  The K/V
// tiles are [key][dim] rows padded to 136 halves so the 272-B row steps walk the 32 smem banks.
// env STRATA_KOLIBRI_ATTN_MMA=1.
//
// PTX-only.  mma.sync.aligned.m16n8k8 is an NVIDIA tensor-core instruction, so the helper, the kernel
// that uses it and its launch are compiled for CUDA only.  On HIP the flag reads as off rather than
// silently selecting a different kernel: the path is opt-in and off by default, so the AMD default path
// loses nothing, and porting it to RDNA4 is a recorded deferral rather than a gap in what ships.
#if defined(__HIPCC__)
#define STRATA_KOLIBRI_MMA_AVAILABLE 0
#else
#define STRATA_KOLIBRI_MMA_AVAILABLE 1
__device__ __forceinline__ void mma168(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(b[0]));
}

static inline bool attn_mma_on() {
    static const bool on = env_on("STRATA_KOLIBRI_ATTN_MMA");
    return on;
}
#endif

#if STRATA_KOLIBRI_MMA_AVAILABLE
__global__ void __launch_bounds__(128) k_attn_flash_mma(const __half* __restrict__ q16,
                                                        const __half* __restrict__ kcache,
                                                        const __half* __restrict__ vcache,
                                                        float* __restrict__ att_m, int H, long long kstride,
                                                        int T, int P, int p0, int64_t swa, int first,
                                                        float scale) {
    constexpr int Q16 = 16, KCK = 24, HS = 3, KP = 136;
    __shared__ __align__(16) __half Ks[KCK][KP];   // [key][dim]
    __shared__ __align__(16) __half Vs[KCK][KP];   // [key][dim]
    __shared__ float part[4][HS][16][KCK];         // the d-slice partials, then the sums (the w=0 slot)
    const int q0 = blockIdx.x * Q16, kvh = blockIdx.y / 4, hgrp = (int) (blockIdx.y % 4) * 3;
    // all 4 warps handle the SAME 3 heads and split the 128 dims (each warp = 32 dims): the score partials
    // then reduce per HEAD across the warps (before, the head depended on the warp and the reduction added
    // different heads' partials - the first wrong cell was dim 32 = warp 1's slice, and the mma output there
    // was exactly 0).
    const __half* kh0 = kcache + (size_t) kvh * (size_t) kstride * 128;
    const __half* vh0 = vcache + (size_t) kvh * (size_t) kstride * 128;
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    const int gid = lane / 4, tig = lane % 4;
    int lo9, nmax;   // the block's last valid row's bounds
    strata::kolibri::attn_mask_bounds(P, p0, min(q0 + 15, T - 1), swa, first, &lo9, &nmax);
    int lo_r, nn, lo_8, nn8;   // this lane's two rows (gid, gid+8)
    strata::kolibri::attn_mask_bounds(P, p0, min(q0 + gid, T - 1), swa, first, &lo_r, &nn);
    strata::kolibri::attn_mask_bounds(P, p0, min(q0 + gid + 8, T - 1), swa, first, &lo_8, &nn8);
    const bool live0 = q0 + gid < T, live8 = q0 + gid + 8 < T;
    uint32_t qa[HS][4][2];   // the A-frags: 3 heads x 4 dim-tiles x {row gid, row gid+8} half2s
#pragma unroll
    for (int hs = 0; hs < HS; ++hs) {
        const __half* qh = q16 + ((size_t) (q0 + gid) * H + kvh * 12 + hgrp + hs) * 128 + warp * 32 + tig * 2;
        const __half* qh8 = q16 + ((size_t) (q0 + gid + 8) * H + kvh * 12 + hgrp + hs) * 128 + warp * 32 + tig * 2;
#pragma unroll
        for (int ds = 0; ds < 4; ++ds) {
            qa[hs][ds][0] = live0 ? *reinterpret_cast<const uint32_t*>(qh + ds * 8) : 0u;
            qa[hs][ds][1] = live8 ? *reinterpret_cast<const uint32_t*>(qh8 + ds * 8) : 0u;
        }
    }
    float m[HS][2], l[HS][2], acc[HS][4][4];   // the acc: c0/c1 = row gid (dims 2tig/+1), c2/c3 = gid+8
#pragma unroll
    for (int hs = 0; hs < HS; ++hs) {
        m[hs][0] = m[hs][1] = -INFINITY; l[hs][0] = l[hs][1] = 0.f;
#pragma unroll
        for (int ds = 0; ds < 4; ++ds)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[hs][ds][c] = 0.f;
    }
    for (int kt = 0; kt < nmax; kt += KCK) {
        const int kn = min(KCK, nmax - kt);
        for (int idx = threadIdx.x; idx < KCK * 16; idx += 128) {
            const int kr = idx / 16, u = idx % 16;
            if (kr < kn && kt + kr < nmax) {
                *reinterpret_cast<uint4*>(&Ks[kr][u * 8]) =
                    *reinterpret_cast<const uint4*>(kh0 + (size_t) (kt + kr) * 128 + u * 8);
                *reinterpret_cast<uint4*>(&Vs[kr][u * 8]) =
                    *reinterpret_cast<const uint4*>(vh0 + (size_t) (kt + kr) * 128 + u * 8);
            } else {   // the rows past the tile: ZERO, never stale smem - the PV mma multiplies them by the
                *reinterpret_cast<uint4*>(&Ks[kr][u * 8]) = make_uint4(0, 0, 0, 0);   // softmax weights, and
                *reinterpret_cast<uint4*>(&Vs[kr][u * 8]) = make_uint4(0, 0, 0, 0);   // 0 * NaN = NaN
            }
        }
        __syncthreads();
#pragma unroll
        for (int hs = 0; hs < HS; ++hs) {
#pragma unroll
            for (int kf = 0; kf < 3; ++kf) {
                float sc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int ds = 0; ds < 4; ++ds) {
                    const uint32_t kb = *reinterpret_cast<const uint32_t*>(&Ks[kf*8 + gid][warp * 32 + ds*8 + tig*2]);
                    mma168(sc, qa[hs][ds], &kb);
                }
                part[warp][hs][gid][kf*8 + tig*2] = sc[0];
                part[warp][hs][gid][kf*8 + tig*2 + 1] = sc[1];
                part[warp][hs][gid + 8][kf*8 + tig*2] = sc[2];
                part[warp][hs][gid + 8][kf*8 + tig*2 + 1] = sc[3];
            }
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < 4 * HS * 16 * KCK; idx += 128) {
            const int g3 = idx % (HS * 16 * KCK);
            const int hs = g3 / (16 * KCK), r2 = (g3 % (16 * KCK)) / KCK, k2 = g3 % KCK;
            const float s = part[0][hs][r2][k2] + part[1][hs][r2][k2] + part[2][hs][r2][k2] + part[3][hs][r2][k2];
            part[0][hs][r2][k2] = s;
        }
        __syncthreads();
#pragma unroll
        for (int hs = 0; hs < HS; ++hs) {
#pragma unroll
            for (int kf = 0; kf < 3; ++kf) {
                const float s00 = part[0][hs][gid][kf*8 + tig*2] * scale;
                const float s01 = part[0][hs][gid][kf*8 + tig*2 + 1] * scale;
                const float s10 = part[0][hs][gid + 8][kf*8 + tig*2] * scale;
                const float s11 = part[0][hs][gid + 8][kf*8 + tig*2 + 1] * scale;
                const int kg0 = kt + kf*8 + tig*2, kg1 = kg0 + 1;
                const float w00 = (live0 && kg0 >= lo_r && kg0 < nn) ? s00 : -INFINITY;
                const float w01 = (live0 && kg1 >= lo_r && kg1 < nn) ? s01 : -INFINITY;
                const float w10 = (live8 && kg0 >= lo_8 && kg0 < nn8) ? s10 : -INFINITY;
                const float w11 = (live8 && kg1 >= lo_8 && kg1 < nn8) ? s11 : -INFINITY;
                float mx0 = fmaxf(w00, w01), mx8 = fmaxf(w10, w11);
                mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
                mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
                mx8 = fmaxf(mx8, __shfl_xor_sync(0xffffffffu, mx8, 1));
                mx8 = fmaxf(mx8, __shfl_xor_sync(0xffffffffu, mx8, 2));
                const float mn0 = fmaxf(m[hs][0], mx0), mn8 = fmaxf(m[hs][1], mx8);
                const float rs0 = (mn0 > -INFINITY) ? expf(m[hs][0] - mn0) : 0.0f;
                const float rs8 = (mn8 > -INFINITY) ? expf(m[hs][1] - mn8) : 0.0f;
                const float c00 = (kg0 >= lo_r && kg0 < nn) ? expf(w00 - mn0) : 0.0f;
                const float c01 = (kg1 >= lo_r && kg1 < nn) ? expf(w01 - mn0) : 0.0f;
                const float c10 = (kg0 >= lo_8 && kg0 < nn8) ? expf(w10 - mn8) : 0.0f;
                const float c11 = (kg1 >= lo_8 && kg1 < nn8) ? expf(w11 - mn8) : 0.0f;
                float ls0 = c00 + c01, ls8 = c10 + c11;
                ls0 += __shfl_xor_sync(0xffffffffu, ls0, 1);
                ls0 += __shfl_xor_sync(0xffffffffu, ls0, 2);
                ls8 += __shfl_xor_sync(0xffffffffu, ls8, 1);
                ls8 += __shfl_xor_sync(0xffffffffu, ls8, 2);
                l[hs][0] = fmaf(l[hs][0], rs0, ls0);
                l[hs][1] = fmaf(l[hs][1], rs8, ls8);
#pragma unroll
                for (int ds = 0; ds < 4; ++ds) {
                    acc[hs][ds][0] *= rs0; acc[hs][ds][1] *= rs0;
                    acc[hs][ds][2] *= rs8; acc[hs][ds][3] *= rs8;
                }
                const __half2 wa0 = __floats2half2_rn(c00, c01), wa8 = __floats2half2_rn(c10, c11);
                const uint32_t aa[2] = {*reinterpret_cast<const uint32_t*>(&wa0),
                                        *reinterpret_cast<const uint32_t*>(&wa8)};
                m[hs][0] = mn0; m[hs][1] = mn8;
#pragma unroll
                for (int ds = 0; ds < 4; ++ds) {
                    const __half2 vb = __halves2half2(*(const __half*)&Vs[kf*8 + 2*tig][warp * 32 + ds*8 + gid],
                                                      *(const __half*)&Vs[kf*8 + 2*tig + 1][warp * 32 + ds*8 + gid]);
                    const uint32_t ab = *reinterpret_cast<const uint32_t*>(&vb);
                    mma168(acc[hs][ds], aa, &ab);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int hs = 0; hs < HS; ++hs) {
        const float i0 = l[hs][0] > 0.0f ? 1.0f / l[hs][0] : 0.0f;
        const float i8 = l[hs][1] > 0.0f ? 1.0f / l[hs][1] : 0.0f;
        const int h = kvh * 12 + hgrp + hs;
#pragma unroll
        for (int ds = 0; ds < 4; ++ds) {
            const int d = warp * 32 + ds * 8 + tig * 2;
            if (live0)
                *reinterpret_cast<float2*>(&att_m[(((size_t) (q0 + gid)) * H + h) * 128 + d]) =
                    make_float2(acc[hs][ds][0] * i0, acc[hs][ds][1] * i0);
            if (live8)
                *reinterpret_cast<float2*>(&att_m[(((size_t) (q0 + gid + 8)) * H + h) * 128 + d]) =
                    make_float2(acc[hs][ds][2] * i8, acc[hs][ds][3] * i8);
        }
    }
}
#endif  // STRATA_KOLIBRI_MMA_AVAILABLE

/// S3: the KV cache is FP16 at the source, so a full-attention layer feeds the score/PV GEMMs straight from the

/// cache (rows 0..P-1 of a ctx-row buffer) with NO per-chunk conversion.  Only the sliding ring wraps, and its
/// gather is a half-to-half copy of at most n_swa cells (2 KB per head-cell, not 4).  The old kernels converted
/// the WHOLE prefix per chunk -- O(prefix^2) in KV traffic, the dominant prefill term (K19).
__global__ void k_kv_ring(const __half* kc, const __half* vc, __half* k16, __half* v16, int64_t swa, int n_vis,
                          int64_t first, int HD, int HKV) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t per = (int64_t) n_vis * HD;
    if (i >= (int64_t) HKV * per) return;
    const int64_t kvh = i / per, rest = i % per;
    const int64_t c = rest / HD, d = rest % HD;
    const int64_t slot = (first + c) % swa;
    k16[i] = kc[((size_t) kvh * swa + slot) * HD + d];
    v16[i] = vc[((size_t) kvh * swa + slot) * HD + d];
}
/// The prompt's causal softmax (window-masked for sliding layers: cells before pos0+t+1-swa are zero): row (h, t)
/// of S (ld floats, P cells) -> softmax(S * scale) over [lo, pos0+t+1), FP16 out.  grid (T, H), block 256.
__global__ void k_attn_softmax16(const float* S, __half* S16, int ld, int T, int P, int pos0, float scale,
                                 int64_t swa, int first, float scl) {
    __shared__ float sh[kThreads];
    const int t = blockIdx.x, h = blockIdx.y;
    const float* row = S + ((size_t) h * T + t) * ld;
    __half* out = S16 + ((size_t) h * T + t) * ld;
    int n, lo;
    strata::kolibri::attn_mask_bounds(P, pos0, t, swa, first, &lo, &n);
    float mx = -INFINITY;
    for (int p = threadIdx.x + lo; p < n; p += blockDim.x) mx = fmaxf(mx, row[p] * scale);
    mx = block_max(mx, sh);
    float sum = 0.f;
    for (int p = threadIdx.x + lo; p < n; p += blockDim.x) sum += expf(row[p] * scale - mx);
    sum = block_sum(sum, sh);
    const float inv = 1.0f / sum;
    for (int p = threadIdx.x; p < P; p += blockDim.x)
        out[p] = __float2half(p >= lo && p < n ? expf(row[p] * scale - mx) * inv * scl : 0.f);
}
/// rms over T rows: row t of x -> row t of out.  grid (T), block kThreads.
__global__ void k_rms_rows(const float* x, const float* w, float* out, int n, float eps) {
    __shared__ float sh[kThreads];
    const float* src = x + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += src[i] * src[i];
    s = block_sum(s, sh);
    const float r = 1.0f / sqrtf(s / (float) n + eps);
    float* dst = out + (size_t) blockIdx.x * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i] * r * w[i];
}
__global__ void k_add_rows(float* x, const float* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}
__global__ void k_swiglu_rows(const float* g, const float* u, float* out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (g[i] / (1.f + expf(-g[i]))) * u[i];
}
/// moe[t] = sy[t] + sum_k wts[t*K+k] * gout[(t*K+k)]: grid (T, ceil(N/B)), block B.
__global__ void k_moe_sum_rows(float* moe, const float* gout, const float* wts, const float* sy, int K, int N) {
    const int t = blockIdx.x;
    const int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float acc = sy[(size_t) t * N + i];
    for (int k = 0; k < K; ++k) acc += wts[(size_t) t * K + k] * gout[((size_t) t * K + k) * (size_t) N + i];
    moe[(size_t) t * N + i] = acc;
}
/// token embedding rows: row toks[t] of W -> row t of out.  grid (T, 4), block 128.
template <int TY> __global__ void k_row_multi(const uint8_t* W, size_t row_bytes, const int* toks, int n, float* out) {
    const uint8_t* r = W + (size_t) toks[blockIdx.x] * row_bytes;
    float* dst = out + (size_t) blockIdx.x * n;
    for (int c = blockIdx.y * blockDim.x + threadIdx.x; c < n / 32; c += 4 * blockDim.x) deq32<TY>(r, c, dst + 32 * c);
}
/// multi-token q/k/v prep: block (head, t); pos = p0 + t.  q rows T x H x D, k/v rows T x HKV x D.

/// A2b/needle: the RoPE angle.  This target is built with `--use_fast_math`, so `cosf`/`sinf` become `__cosf`/`__sinf`,
/// whose argument reduction degrades as |theta| grows - and theta = pos * base^(-2i/D) is ~1e5 radians at position 1e5.
/// That is invisible at 32K and material at 98K, which is exactly where the 100K depth-90 needle parts from llama.cpp
/// (measured: step 10 top-1 logit 14.30 vs the reference's 16.79 on token 122).  The gate computes the angle in double,
/// the way the reference does, so the hypothesis can be tested end-to-end before it is believed.
__device__ int g_rope_f64 = 0;
__device__ inline void rope_cs_sn(int64_t pos, float base, int pp, int D, float& cs, float& sn) {
    if (g_rope_f64) {
        const double th = (double) pos * pow((double) base, -2.0 * (double) pp / (double) D);
        cs = (float) cos(th); sn = (float) sin(th);
    } else {
        // the ORIGINAL expression, character for character: the gate-off arm must stay bitwise identical (PF_HASH)
        const float theta = (float) pos * powf(base, -2.0f * (float) pp / (float) D);
        cs = cosf(theta); sn = sinf(theta);
    }
}

__global__ void k_attn_prep_multi(const float* qx, const float* kx, const float* vx, const float* qw, const float* kw,
                                  float* q, __half* kc, __half* vc, int H, int HKV, int D, int64_t p0, int64_t swa,
                                  int ctx, float base, float eps, bool rope) {
    const int t = blockIdx.y;
    const int head = blockIdx.x;
    const bool isq = head < H;
    const int kvh = head - H;
    const float* src = isq ? qx + ((size_t) t * H + head) * D : kx + ((size_t) t * HKV + kvh) * D;
    const int pos = (int) (p0 + t);
    const int64_t rows = rope ? swa : ctx;
    const int64_t slot = rope ? (int64_t) pos % swa : (int64_t) pos;
    float s = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) s += src[i] * src[i];
    __shared__ float sh[kThreads];
    s = block_sum(s, sh);
    const float r = 1.0f / sqrtf(s / (float) D + eps);
    __shared__ float buf[512];
    for (int i = threadIdx.x; i < D; i += blockDim.x) buf[i] = src[i] * r * (isq ? qw[i] : kw[i]);
    __syncthreads();
    float* dstq = isq ? q + ((size_t) t * H + head) * D : nullptr;
    __half* dstk = isq ? nullptr : kc + ((size_t) kvh * rows + slot) * D;
    const int half = D / 2;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = buf[i];
        if (rope) {
            const int pp = i < half ? i : i - half;
            float cs, sn;
            rope_cs_sn(pos, base, pp, D, cs, sn);
            v = i < half ? buf[i] * cs - buf[i + half] * sn : buf[i - half] * sn + buf[i] * cs;
        }
        if (isq) dstq[i] = v; else dstk[i] = v;
    }
    if (!isq)
        for (int i = threadIdx.x; i < D; i += blockDim.x)
            vc[((size_t) kvh * rows + slot) * D + i] = vx[((size_t) t * HKV + kvh) * D + i];
}
/// The routed entries (K picks for one token, entry e = k) grouped per expert on the device: one group per
/// expert used (ascending id), its blob address, its entries in entry order (q36_engine.cu's k_group).
__global__ void k_group(const int32_t* ids, int n_ent, int K_, int n_expert, const uint8_t* blobs,
                        unsigned long long blob, unsigned long long* grp_ptr, int32_t* grp_start, int32_t* n_groups,
                        int32_t* ent_dst, int32_t* ent_tok) {
    __shared__ int cnt[512], off[512];
    for (int x = threadIdx.x; x < n_expert; x += blockDim.x) cnt[x] = 0;
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x) atomicAdd(&cnt[ids[e]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int g = 0, pos = 0;
        for (int x = 0; x < n_expert; ++x) {
            off[x] = pos;
            if (cnt[x] == 0) continue;
            grp_ptr[g] = (unsigned long long) (blobs + (size_t) x * blob);
            grp_start[g] = pos;
            pos += cnt[x];
            ++g;
        }
        grp_start[g] = pos;
        *n_groups = g;
        for (int e = 0; e < n_ent; ++e) {
            const int q = off[ids[e]]++;
            ent_dst[q] = e;
            ent_tok[q] = e / K_;
        }
    }
}

__global__ void k_moe_sum(float* moe, const float* y, const float* w, int k, const float* ys, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = ys[i];
    for (int s = 0; s < k; ++s) acc += w[s] * y[(size_t) s * n + i];
    moe[i] = acc;
}
// the slots path's MoE sum: each selection's row is the sum of its GPU-slot row and its overflow row (the CPU
// pool's or the pinned table's; the side that did not compute a selection leaves its row zero).
__global__ void k_moe_sum2(float* moe, const float* y, const float* yc, const float* w, int k, const float* ys, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = ys[i];
    for (int s = 0; s < k; ++s) acc += w[s] * (y[(size_t) s * n + i] + yc[(size_t) s * n + i]);
    moe[i] = acc;
}
__device__ void k_attn_prep_body(const float* qx, const float* kx, const float* vx, const float* qw, const float* kw,
                            float* q, __half* kc, __half* vc, int H, int D, int pos, int64_t swa, int ctx, float base,
                            float eps, bool rope) {
    __shared__ float sh[kThreads];
    __shared__ float buf[512];
    const int head = blockIdx.x;
    const bool isq = head < H;
    const int kvh = head - H;
    const float* src = isq ? qx + (size_t) head * D : kx + (size_t) kvh * D;
    const int64_t rows = rope ? swa : ctx;
    const int64_t slot = rope ? (int64_t) pos % swa : (int64_t) pos;
    float s = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) s += src[i] * src[i];
    s = block_sum(s, sh);
    const float r = 1.0f / sqrtf(s / (float) D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x) buf[i] = src[i] * r * (isq ? qw[i] : kw[i]);
    __syncthreads();
    float* dstq = isq ? q + (size_t) head * D : nullptr;
    __half* dstk = isq ? nullptr : kc + ((size_t) kvh * rows + slot) * D;
    const int half = D / 2;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = buf[i];
        if (rope) {
            const int p = i < half ? i : i - half;
            float cs, sn;
            rope_cs_sn(pos, base, p, D, cs, sn);
            v = i < half ? buf[i] * cs - buf[i + half] * sn : buf[i - half] * sn + buf[i] * cs;
        }
        if (isq) dstq[i] = v; else dstk[i] = v;
    }
    if (!isq)
        for (int i = threadIdx.x; i < D; i += blockDim.x)
            vc[((size_t) kvh * rows + slot) * D + i] = vx[(size_t) kvh * D + i];
}
__global__ void k_attn_prep(const float* qx, const float* kx, const float* vx, const float* qw, const float* kw,
                            float* q, __half* kc, __half* vc, int H, int D, int pos, int64_t swa, int ctx, float base,
                            float eps, bool rope) {
    k_attn_prep_body(qx, kx, vx, qw, kw, q, kc, vc, H, D, pos, swa, ctx, base, eps, rope);
}
/// graph decode: the position from device memory
__global__ void k_attn_prep_p(const float* qx, const float* kx, const float* vx, const float* qw, const float* kw,
                              float* q, __half* kc, __half* vc, int H, int D, const int* pos_p, int64_t swa, int ctx,
                              float base, float eps, bool rope) {
    k_attn_prep_body(qx, kx, vx, qw, kw, q, kc, vc, H, D, *pos_p, swa, ctx, base, eps, rope);
}

// S6: the decode's single-pass score dot, coalesced (STRATA_KOLIBRI_ATTN_COALESCED): a NUMERICS change
// K27: the transposed-K mirror (STRATA_KOLIBRI_ATTN_KT).  Unlike ATTN_COALESCED this is NOT a numerics change - the
// mirror holds the same values and the dot keeps the scalar d order - it only makes the warp's K loads contiguous.
// Allocated for the SLIDING layers only (26 MB total): those are the 39 layers that dominate short-context decode,
// and their window is fixed at n_swa rows, so the mirror is small and cheap to rebuild every token.
// S8b: the attention's PV pass is 43% of the decode's GPU time and each thread walks ~513 positions in ONE dependent
// FMA chain.  Four interleaved accumulators (p = 4n+0..3) give the scheduler four independent chains over the SAME
// products and the same V mirror; only the ORDER of the four partial sums changes, so this is a NUMERICS change -
// env-gated and KL-gated (STRATA_KOLIBRI_ATTN_PV4).
__host__ __device__ inline bool attn_pv4_on() {
    // __CUDA_ARCH__ is nvcc's device-pass macro; HIP's device pass defines __HIP_DEVICE_COMPILE__
    // instead, and without it the device pass falls into the host branch and calls getenv().
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return g_attn_pv4;   // the host sets it once at startup, like g_attn_coalesced
#else
    // ON by default: four interleaved PV chains are worth +13% decode (tg 30.1/30.3 against 26.7/26.7 - the tg-short
    // floor of 30, reached) and cost nothing measurable in accuracy: KL vs llama.cpp per prompt 0.00016-0.00740
    // (baseline 0.00004-0.01045), worst answer-step 0.0147 against the 0.02 gate, and the plain-path oracle still
    // prints BITWISE PASS because no generated id moves.  STRATA_KOLIBRI_ATTN_PV4=0 is the A/B arm.
    const char* v = std::getenv("STRATA_KOLIBRI_ATTN_PV4");
    return v == nullptr || (v[0] != 0 && std::atoi(v) != 0);
#endif
}

static inline bool attn_kt_on() {
    // ON by default: unlike ATTN_COALESCED this is not a numerics change - the mirror holds the same values and the
    // dot keeps the scalar d order - and the oracle prints BITWISE PASS with it enabled (measured on this card).
    // STRATA_KOLIBRI_ATTN_KT=0 reproduces the pre-K27 path for an A/B.
    const char* v = std::getenv("STRATA_KOLIBRI_ATTN_KT");
    return v == nullptr || (v[0] != 0 && std::atoi(v) != 0);
}

__host__ __device__ inline bool attn_coalesced_on() {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return g_attn_coalesced;   // the host sets it once at startup
#else
    static const bool on = env_on("STRATA_KOLIBRI_ATTN_COALESCED");
    return on;
#endif
}

// K27: THE TRANSPOSED K MIRROR.  kt[(kvh*D + d)*rows + slot] = kc[(kvh*rows + slot)*D + d].  The score dot then
// reads kt[d*rows + slot]: the 32 lanes of a warp own 32 CONSECUTIVE positions, so at each d their halves are
// contiguous - a coalesced warp load - while every lane keeps the untouched scalar d = 0..127 chain.  That is why
// this can be bitwise where the warp-cooperative dot (a shuffle tree) is not, and build/attn_k_parity measured the
// pass 12.5x faster this way (0.2097 -> 0.0167 ms at H=48, 256 threads, sw 513).
__global__ void k_kt_window(const __half* kc, const __half* vc, __half* kt, __half* vt, int64_t rows, int n_vis,
                            int64_t first, int64_t swa, int D, int HKV, bool sliding) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t per = (int64_t) n_vis * D;
    if (i >= (int64_t) HKV * per) return;
    const int64_t kvh = i / per, rest = i % per;
    const int64_t p = rest / D, d = rest % D;
    const int64_t gp = first + p, slot = sliding ? gp % swa : gp;
    const int64_t src = ((size_t) kvh * rows + slot) * D + d, dst = ((size_t) kvh * D + d) * rows + slot;
    kt[dst] = kc[src];
    vt[dst] = vc[src];   // V gets the same mirror: its PV accumulation walks positions at stride D, uncoalesced too
}

/// STRATA_KOLIBRI_DEC_HANDOFF_GRAPH: the STEADY (incremental) form of the kt window with the position from device
/// memory - one row appended per token, at slot ((sliding) ? pos % swa : pos).  Bitwise the eager `k_kt_window` for
/// the same position: same index arithmetic, same stores.  The first decode token of a request takes the EAGER
/// whole-window form instead, so a captured graph only ever holds this steady shape.
__global__ void k_kt_window_dp(const __half* kc, const __half* vc, __half* kt, __half* vt, int64_t rows,
                               const int* pos_p, int64_t swa, int D, int HKV, bool sliding) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t pos = *pos_p;
    const int64_t per = (int64_t) D;
    if (i >= (int64_t) HKV * per) return;
    const int64_t kvh = i / per, d = i % per;
    const int64_t slot = sliding ? pos % swa : pos;
    const int64_t src = ((size_t) kvh * rows + slot) * D + d, dst = ((size_t) kvh * D + d) * rows + slot;
    kt[dst] = kc[src];
    vt[dst] = vc[src];
}

__device__ void k_attn_body(const float* q, const __half* kc, const __half* vc, float* out, int H, int HKV, int D, int pos,
                       int64_t swa, int ctx, float scale, bool sliding, bool force_2pass = false, int bench_mode = 0,
                       int two_pass_max = 11000, const __half* ktc = nullptr, const __half* vtc = nullptr) {
    extern __shared__ float sc[];
    __shared__ float sh[512];   // sized for the widest block the two-pass path launches (512 = 4 slices x D)
    const int h = blockIdx.x, kvh = h / (H / HKV);
    const int64_t rows = sliding ? swa : ctx;
    const int n_pos = sliding ? (pos + 1 < (int) swa ? pos + 1 : (int) swa) : pos + 1;
    const int64_t first = sliding ? (int64_t) pos + 1 - n_pos : 0;
    const float* qh = q + (size_t) h * D;
    // K25: q staged in shared memory for the warp-cooperative score dot.  The scalar loop read qh[d] and
    // kp[lane][d] with d fixed, so the 32 lanes touched addresses 256 B apart - every warp instruction was 32
    // separate sectors, fully uncoalesced, and that pass is 91% of the kernel (measured: 10.98 of 12.05 ms).
    // The staging (load + syncthreads) lives in the two-pass branch only: staging it for every caller cost the
    // single-pass path 0.305 -> 0.395 ms/launch, ~1 ms/token on short decode, for a buffer it never reads.
    __shared__ float qsm[128];
    // the scores live in dynamic shared memory up to the 48 KB a T4-class block gets; past that (a 32K full-
    // attention layer) the kernel runs two passes and recomputes each score - slower, and it works
    if (!force_2pass && n_pos <= two_pass_max) {   // 11000*4 = 44 KB dynamic + 2.5 KB static sh[]/qsm[] < the 48 KB T4 cap
    // S6: the single-pass score dot, warp-cooperative and COALESCED (the same pattern the two-pass path
    // uses for D == 128): the scalar form had every thread read a whole 128-dim K row (256 B), so a warp
    // instruction touched 32 separate sectors - the decode's attention ran at ~10 GFLOP/s (the T4's fp32 peak
    // is 32.5 TFLOPS) and cost 14.7 ms/token at a 512-token context.  Here lane l reads 8 contiguous bytes of
    // one K row (a warp instruction covers 256 B) and the dot finishes with a warp shuffle reduce.
    // The summation order changes (the shuffle tree vs the scalar loop), so this is a NUMERICS change: it is
    // env-gated (STRATA_KOLIBRI_ATTN_COALESCED) and KL-gated, and the plain path stays bitwise without it.
    const bool coal = D == 128 && attn_coalesced_on();
    if (coal) {
        if (threadIdx.x < D) qsm[threadIdx.x] = qh[threadIdx.x];
        __syncthreads();
        const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nwarp = (int) blockDim.x >> 5;
        const __half2* kbase = reinterpret_cast<const __half2*>(kc) + (size_t) kvh * rows * (D / 2);
        for (int row = wid; row < n_pos; row += nwarp) {
            const int64_t gp = first + row, slot = sliding ? gp % swa : gp;
            const __half2* kp2 = kbase + slot * (D / 2) + lane * 2;
            const float2 a = __half22float2(kp2[0]);
            const float2 b = __half22float2(kp2[1]);
            float part = qsm[lane * 4] * a.x + qsm[lane * 4 + 1] * a.y
                       + qsm[lane * 4 + 2] * b.x + qsm[lane * 4 + 3] * b.y;
            for (int off = 16; off > 0; off >>= 1) part += __shfl_down_sync(0xffffffffu, part, off);
            if (lane == 0) sc[row] = part * scale;
        }
    } else {
    for (int p = threadIdx.x; p < n_pos; p += blockDim.x) {
        const int64_t gp = first + p, slot = sliding ? gp % swa : gp;
        float s = 0.f;
        if (ktc != nullptr) {   // K27: the transposed mirror - same expression, same d order, coalesced addresses
            const __half* ktbase = ktc + (size_t) kvh * D * rows;
            for (int i = 0; i < D; ++i) s += qh[i] * __half2float(ktbase[(size_t) i * rows + slot]);
        } else {
            const __half* kp = kc + ((size_t) kvh * rows + slot) * D;
            for (int i = 0; i < D; ++i) s += qh[i] * __half2float(kp[i]);
        }
        sc[p] = s * scale;
    }
    }
    __syncthreads();
    float mx = -INFINITY;
    for (int p = threadIdx.x; p < n_pos; p += blockDim.x) mx = fmaxf(mx, sc[p]);
    sh[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if ((int) threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]); __syncthreads(); }
    mx = sh[0];
    __syncthreads();
    float sum = 0.f;
    for (int p = threadIdx.x; p < n_pos; p += blockDim.x) { sc[p] = expf(sc[p] - mx); sum += sc[p]; }
    sum = block_sum(sum, sh);
    const bool pv4 = attn_pv4_on() && vtc != nullptr && D == 128;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float acc = 0.f;
        if (pv4) {
            // four interleaved chains over the SAME products, then one fixed reduction - only the order changes
            const __half* vtbase = vtc + (size_t) kvh * D * rows;
            float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
            int p = 0;
            for (; p + 3 < n_pos; p += 4) {
                const int64_t s0 = sliding ? (first + p) % swa : (first + p);
                const int64_t s1 = sliding ? (first + p + 1) % swa : (first + p + 1);
                const int64_t s2 = sliding ? (first + p + 2) % swa : (first + p + 2);
                const int64_t s3 = sliding ? (first + p + 3) % swa : (first + p + 3);
                a0 += sc[p] * __half2float(vtbase[(size_t) i * rows + s0]);
                a1 += sc[p + 1] * __half2float(vtbase[(size_t) i * rows + s1]);
                a2 += sc[p + 2] * __half2float(vtbase[(size_t) i * rows + s2]);
                a3 += sc[p + 3] * __half2float(vtbase[(size_t) i * rows + s3]);
            }
            for (; p < n_pos; ++p) {
                const int64_t gp = first + p, slot = sliding ? gp % swa : gp;
                a0 += sc[p] * __half2float(vtbase[(size_t) i * rows + slot]);
            }
            acc = (a0 + a1) + (a2 + a3);
        } else if (vtc != nullptr) {   // K27: the transposed V mirror - same p order, contiguous addresses
            const __half* vtbase = vtc + (size_t) kvh * D * rows;
            for (int p = 0; p < n_pos; ++p) {
                const int64_t gp = first + p, slot = sliding ? gp % swa : gp;
                acc += sc[p] * __half2float(vtbase[(size_t) i * rows + slot]);
            }
        } else {
            for (int p = 0; p < n_pos; ++p) {
                const int64_t gp = first + p, slot = sliding ? gp % swa : gp;
                acc += sc[p] * __half2float(vc[((size_t) kvh * rows + slot) * D + i]);
            }
        }
        out[(size_t) h * D + i] = acc / sum;
    }
    } else {
        // K25: the old two-pass path recomputed every score INSIDE the per-dimension loop, i.e. D times per
        // position per block: n_pos * D * D FMAs.  At 32K that is ~50 GFLOP/token of pure recompute, which is
        // why long-context decode sat at 0.6 tok/s while the KV read alone allows ~400.  Tiled online softmax:
        // each score is computed EXACTLY once per block, kept in a 4096-entry shared tile, and the running max,
        // denominator and accumulator are rescaled tile by tile.  Same max, same weights, same result as the
        // two-pass form; the only numerics change is the rescaling order.
        const int TILE = 4096;                    // 16 KB of dynamic shared memory, under the 48 KB T4 cap
        // Split-K over positions: D == 128 dims, and a two-pass block runs 512 threads, so four threads own each
        // dimension and walk a quarter of the tile's positions each.  The old mapping (one thread per dimension,
        // 128 threads) left a single serial load-FMA chain of n_pos iterations per thread - 263 ns per position
        // measured, pure latency, no instruction-level parallelism.  Four slices times four unrolled accumulators
        // below gives 16 independent load streams per dimension.
        const int di = threadIdx.x % D;
        const int slice = threadIdx.x / D;
        const int nslice = blockDim.x / D;
        float m = -INFINITY, l = 0.f, acc = 0.f;
        for (int64_t t0 = 0; t0 < n_pos; t0 += TILE) {
            const int nt = (int) (n_pos - t0 < TILE ? n_pos - t0 : TILE);
            // Warp-cooperative score: 32 lanes x 4 halfs = the 128-dim row, so lane l reads 8 contiguous bytes
            // and a warp instruction covers 256 B of one K row - coalesced, 8x fewer load instructions, and the
            // dot finishes with a warp shuffle reduce.  Requires D == 128 (the artifact's head_dim); the scalar
            // path stays for any other width.
            if (bench_mode == 2) {
                for (int p = threadIdx.x; p < nt; p += blockDim.x) sc[p] = 1.f;
            } else if (D == 128) {
                if (threadIdx.x < D) qsm[threadIdx.x] = qh[threadIdx.x];
                __syncthreads();
                const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nwarp = (int) blockDim.x >> 5;
                const __half2* kbase = reinterpret_cast<const __half2*>(kc) + (size_t) kvh * rows * (D / 2);
                for (int row = wid; row < nt; row += nwarp) {
                    const int64_t gp = first + t0 + row, slot = sliding ? gp % swa : gp;
                    const __half2* kp2 = kbase + slot * (D / 2) + lane * 2;
                    const float2 a = __half22float2(kp2[0]);
                    const float2 b = __half22float2(kp2[1]);
                    float part = qsm[lane * 4] * a.x + qsm[lane * 4 + 1] * a.y
                               + qsm[lane * 4 + 2] * b.x + qsm[lane * 4 + 3] * b.y;
                    for (int off = 16; off > 0; off >>= 1) part += __shfl_down_sync(0xffffffffu, part, off);
                    if (lane == 0) sc[row] = part * scale;
                }
                __syncthreads();
            } else {
                for (int p = threadIdx.x; p < nt; p += blockDim.x) {
                    const int64_t gp = first + t0 + p, slot = sliding ? gp % swa : gp;
                    const __half* kp = kc + ((size_t) kvh * rows + slot) * D;
                    float s = 0.f;
                    for (int d = 0; d < D; ++d) s += qh[d] * __half2float(kp[d]);
                    sc[p] = s * scale;
                }
                __syncthreads();
            }
            float tm = -INFINITY;
            for (int p = threadIdx.x; p < nt; p += blockDim.x) tm = fmaxf(tm, sc[p]);
            sh[threadIdx.x] = tm;
            __syncthreads();
            for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) { if ((int) threadIdx.x < s2) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s2]); __syncthreads(); }
            const float m_new = fmaxf(m, sh[0]);
            const float resc = (m == -INFINITY) ? 0.f : expf(m - m_new);   // first tile: no history to rescale
            float ts = 0.f;
            for (int p = threadIdx.x; p < nt; p += blockDim.x) { sc[p] = expf(sc[p] - m_new); ts += sc[p]; }
            ts = block_sum(ts, sh);
            const __half* vbase = vc + (size_t) kvh * rows * D;
            auto v_at = [&](int64_t p) -> float {
                const int64_t gp = first + t0 + p, slot = sliding ? gp % swa : gp;
                return __half2float(vbase[slot * D + di]);
            };
            float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
            if (bench_mode != 1) {   // score-only: neutralise the V pass
            int p = slice;
            for (; p + 3 * nslice < nt; p += 4 * nslice) {   // four independent chains, then the tail
                a0 += sc[p] * v_at(p);
                a1 += sc[p + nslice] * v_at(p + nslice);
                a2 += sc[p + 2 * nslice] * v_at(p + 2 * nslice);
                a3 += sc[p + 3 * nslice] * v_at(p + 3 * nslice);
            }
            for (; p < nt; p += nslice) a0 += sc[p] * v_at(p);
            }
            float a = (a0 + a1) + (a2 + a3);
            if (nslice > 1) {                       // reduce the slice partials for this dimension
                sh[threadIdx.x] = a;
                __syncthreads();
                if (slice == 0) {
                    float s = 0.f;
                    for (int q = 0; q < nslice; ++q) s += sh[q * D + di];
                    a = s;
                }
                __syncthreads();
            }
            if (slice == 0) acc = acc * resc + a;
            l = l * resc + ts;
            m = m_new;
            __syncthreads();
        }
        if (slice == 0) out[(size_t) h * D + di] = acc / l;
    }
}
__global__ void k_attn(const float* q, const __half* kc, const __half* vc, float* out, int H, int HKV, int D, int pos,
                       int64_t swa, int ctx, float scale, bool sliding, int bench_mode, int two_pass_max,
                       const __half* ktc = nullptr, const __half* vtc = nullptr) {
    const int n_vis = sliding ? (pos + 1 < (int) swa ? pos + 1 : (int) swa) : pos + 1;
    k_attn_body(q, kc, vc, out, H, HKV, D, pos, swa, ctx, scale, sliding, false, bench_mode, two_pass_max, ktc, vtc);
}
/// graph decode: the position from device memory; the dynamic shared memory is the two-pass size (0) so one
/// capture serves every position (the smem path bakes n_vis into the launch and cannot be replayed across pos).
__global__ void k_attn_p(const float* q, const __half* kc, const __half* vc, float* out, int H, int HKV, int D,
                         const int* pos_p, int64_t swa, int ctx, float scale, bool sliding) {
    k_attn_body(q, kc, vc, out, H, HKV, D, *pos_p, swa, ctx, scale, sliding, true);
}

/// STRATA_KOLIBRI_DEC_ATTN_DP: the SINGLE-PASS decode kernel with the position from device memory.
///
/// `k_attn_p` above cannot be used for a bitwise arm: it forces the two-pass body, and the two-pass rescaling
/// order changes the numbers (measured - the full logits differ at record 0 while the greedy ids agree).  This
/// variant keeps the eager body exactly: same branch, same block size, same shared-memory layout, same order of
/// every product.  The only differences are where `pos` comes from (one load at kernel entry) and who sizes the
/// dynamic shared memory: the caller passes the CAPACITY of its position bucket, and the body only reads up to
/// n_vis entries, so every position inside the bucket computes exactly what the eager launch computes.
__global__ void k_attn_dp(const float* q, const __half* kc, const __half* vc, float* out, int H, int HKV, int D,
                         const int* pos_p, int64_t swa, int ctx, float scale, bool sliding, int two_pass_max,
                         const __half* ktc, const __half* vtc) {
    k_attn_body(q, kc, vc, out, H, HKV, D, *pos_p, swa, ctx, scale, sliding, false, 0, two_pass_max, ktc, vtc);
}

/// The dynamic shared-memory capacity for a device-position attention launch: the next power of two at or above
/// `n_vis`, never above the two-pass threshold (past it the body would take the two-pass branch, whose launch
/// geometry a captured graph cannot change).  Rounding UP is safe because the body reads only the first `n_vis`
/// entries of that buffer.
static inline int attn_dp_bucket(int n_vis, int two_pass_max) {
    int b = 512;
    while (b < n_vis && b < two_pass_max) b <<= 1;
    return b > two_pass_max ? two_pass_max : b;
}

// Positive keys are single-pass score capacities (128 threads). Negative keys name online attention
// (512 threads and 4096 scratch floats). They must never share a captured launch, even at a bucket boundary.
static inline int attn_graph_key(int n_vis, int two_pass_max) {
    return n_vis <= two_pass_max ? attn_dp_bucket(n_vis, two_pass_max) : -4096;
}


__half* halloc(size_t n) {
    __half* p = nullptr;
    KC(cudaMalloc(&p, n * 2));
    KC(cudaMemset(p, 0, n * 2));
    return p;
}
float* dalloc(size_t n) {
    float* p = nullptr;
    KC(cudaMalloc(&p, n * 4));
    KC(cudaMemset(p, 0, n * 4));
    return p;
}

void* g_q8 = nullptr;            // the q8_1 activation buffer the native mmvq path quantizes x into
std::vector<int> g_ids_host;     // the router's picks, on the host (the native path indexes experts per selection)
constexpr size_t kActBuf = 1 << 16;

/// One matmul, bitwise kolibriref's gemv: quantized weights go through native_mmvq on a q8_1 copy of the
/// activation (one call per selection, expert ids[sel] from the host picks); F32/F16/BF16 through plain k_gemv.
/// `base`/`eb` point at an expert blob table (pinned host, UVA-read); null for the dense GPU tensors.
/// S6: the decode's dense gemv through the fp16 cache (the caller converts x to fp16 once).  n_sel must be 1
/// (the decode) and the weight must have its f16 copy.
void gemv_f16(const Dense& w, const __half* x, float* y, cudaStream_t st = 0) {
    const int n_in = (int) w.shape[0], n_out = (int) w.shape[1];
    if (gemv_gs_on() && n_out >= kGemvGsMinOut) {
        gemv_gs_note(n_out);
        k_gemv_f16_gs<<<kGemvGsBlocks, kThreads, 0, st>>>((const __half*) w.f16, x, n_in, y, n_out);
    } else {
        k_gemv_f16<<<dim3(n_out), kThreads, 0, st>>>((const __half*) w.f16, x, n_in, y, n_out);
    }
    KC(cudaGetLastError());
}

void gemv(const Dense& w, const float* x, float* y, const int* ids, int n_sel, size_t x_stride, const uint8_t* base,
          size_t eb, cudaStream_t st);

/// S6: the decode's dense call: the fp16 path when the weight has its pre-dequantized fp16 copy (and the
/// caller has the fp16 x ready), else the exact current path.  env STRATA_KOLIBRI_DEC_F16 gates the route.
void gemv_dec(const Dense& w, const float* x, const __half* x16, float* y, cudaStream_t st = 0) {
    static const bool on = env_on("STRATA_KOLIBRI_DEC_F16");
    if (on && w.f16 && x16) {
        gemv_f16(w, x16, y, st);
        return;
    }
    gemv(w, x, y, nullptr, 1, 0, nullptr, 0, st);
}

void gemv(const Dense& w, const float* x, float* y, const int* ids, int n_sel, size_t x_stride, const uint8_t* base,
          size_t eb, cudaStream_t st = 0) {
    const int n_in = (int) w.shape[0], n_out = (int) w.shape[1];
    const uint8_t* W = base ? base : w.d;
    const bool quant = w.type == T_Q8_0 || w.type == T_Q4_K || w.type == T_Q5_K || w.type == T_Q6_K;
    // `st == 0` keeps the historical g_st submission, which is what the legacy-stream decode relies on: g_st is a
    // blocking stream, so the legacy stream is implicitly ordered against it in both directions.  A caller that
    // passes its OWN non-blocking stream (STRATA_KOLIBRI_DEC_STREAM) gets the quantise and the mmvq on that stream
    // instead - otherwise every dense gemv would run on g_st and race the caller's other kernels.
    const cudaStream_t qst = st != 0 ? st : g_st;
    if (quant) {
        const int cols = x_stride ? n_sel : 1;
        strata::kernels::native_quantize_q8_1(x, g_q8, n_in, cols, qst);
        const size_t qb = strata::kernels::native_q8_1_bytes(n_in, 1);
        for (int sel = 0; sel < n_sel; ++sel) {
            const uint8_t* we = W + (ids ? (size_t) g_ids_host[(size_t) sel] * eb : 0);
            strata::kernels::native_mmvq(w.type, we, (const char*) g_q8 + (x_stride ? (size_t) sel * qb : 0),
                                         y + (size_t) sel * n_out, n_in, n_out, 1, qst);
        }
        KC(cudaGetLastError());
        return;
    }
    const size_t row_bytes = w.row_bytes;
    // T3: the capped grid, only where it was measured faster (kGemvGsMinOut) and only for the single-selection form
    // this task covers - the expert-id and multi-selection forms keep the block-per-row path untouched.
    if (gemv_gs_on() && n_sel == 1 && ids == nullptr && x_stride == 0 && n_out >= kGemvGsMinOut &&
        (w.type == T_F32 || w.type == T_F16 || w.type == T_BF16)) {
        gemv_gs_note(n_out);
        switch (w.type) {
        case T_F32: k_gemv_gs<T_F32><<<kGemvGsBlocks, kThreads, 0, st>>>(W, row_bytes, n_in, x, y, n_out); break;
        case T_F16: k_gemv_gs<T_F16><<<kGemvGsBlocks, kThreads, 0, st>>>(W, row_bytes, n_in, x, y, n_out); break;
        case T_BF16: k_gemv_gs<T_BF16><<<kGemvGsBlocks, kThreads, 0, st>>>(W, row_bytes, n_in, x, y, n_out); break;
        default: break;
        }
        KC(cudaGetLastError());
        return;
    }
    if (dec_gemv_v4_on() && w.type == T_F32 && n_sel == 1 && ids == nullptr && x_stride == 0) {
        k_gemv_f32_v4<<<n_out, kThreads, 0, st>>>((const float*) W, x, n_in, y, n_out);
        KC(cudaGetLastError());
        return;
    }
    switch (w.type) {
    case T_F32: k_gemv<T_F32><<<dim3(n_out, n_sel), kThreads, 0, st>>>(W, row_bytes, eb, ids, n_in, x, x_stride, y, n_out); break;
    case T_F16: k_gemv<T_F16><<<dim3(n_out, n_sel), kThreads, 0, st>>>(W, row_bytes, eb, ids, n_in, x, x_stride, y, n_out); break;
    case T_BF16: k_gemv<T_BF16><<<dim3(n_out, n_sel), kThreads, 0, st>>>(W, row_bytes, eb, ids, n_in, x, x_stride, y, n_out); break;
    default:
        std::fprintf(stderr, "strata-kolibri: no kernel for ggml type %d\n", w.type);
        std::exit(1);
    }
    KC(cudaGetLastError());
}

/// A2b: everything a prefill chunk owns that a second chunk would collide with.  Two of these are allocated (act1 is the
/// existing fields, act2 the STRATA_KOLIBRI_PF_PIPE set); the wavefront hands one per in-flight chunk.
struct ActSet {
    float *x_m = nullptr, *h_m = nullptr, *y_m = nullptr, *q_m = nullptr, *k_m = nullptr, *v_m = nullptr;
    float *att_m = nullptr, *router_m = nullptr, *moe_m = nullptr, *sy_m = nullptr;
    float *sg_m = nullptr, *su_m = nullptr, *sh_m = nullptr, *gout_m = nullptr;
    // A2b class 1: per-chunk device buffers the layer also owns.  Not in the first set's ActSet because they predate it -
    // they are here so a second in-flight chunk cannot collide on them (x16 in particular is written by the shared
    // expert's k_to_f16 and read by the routed gather, the aliasing that broke an earlier experiment).
    __half *x16 = nullptr, *q16 = nullptr;
    char *xq8_m = nullptr;
    int *ids_m = nullptr, *toks_d = nullptr;
    int *grp_start_m = nullptr, *n_groups_m = nullptr, *ent_dst_m = nullptr, *ent_tok_m = nullptr;
    unsigned long long *grp_ptr_m = nullptr;
    float *ey_h = nullptr;   // pinned host: the pool writes it, the H2D reads it
    float *wts_m = nullptr;  // the router weights: PRE writes them, POST's MoE sum reads them

    // A2b: the PUBLISHED ids, per chunk, in PINNED memory.  Pinned is load-bearing: cudaMemcpyAsync into pageable
    // memory silently synchronises, which would keep draining the device queue and make the doorbell look inert.
    int* ids_h = nullptr;
    // A2b/A2a: the doorbell - recorded at the END of this chunk's PRE, waited on at the start of its POST.  Waiting on
    // this instead of the stream is what lets the pool drain overlap the OTHER chunk's PRE.
    cudaEvent_t ev = nullptr;
    // A2b: the compacted SLIDING window, which k_kv_ring refills per chunk.  Shared, chunk 2 would
    // re-describe the window chunk 1's attention was built on.
    __half *k16 = nullptr;
    __half *v16 = nullptr;

    // G1 PF_PARTIAL (plan 2026-10-24-kolibri-pf-partial-sums.md Task 2).  The pool's miss contributions arrive as
    // ONE weighted partial per token instead of the whole NKc x N block: p_h is a 2-slot pinned ring the host
    // fills, p_m its device copy, and the resident rows stay in s.res_ey (block order) mapped by rr_sel_m.
    //
    // The ring is not the safety argument.  p_done[sl] is recorded on the default stream right after that slot's
    // H2D and waited on before the host writes the slot again, so correctness does not depend on the ring being
    // big enough - it only buys the host a slot of slack so the wait is normally a no-op.  (racecheck cannot see
    // a host-vs-DMA race: it observes device memory only.)
    float* p_h[2] = {nullptr, nullptr};
    cudaEvent_t p_done[2] = {nullptr, nullptr};
    int p_flip = 0;
    float* p_m = nullptr;     // device partial, Tc x N
    int* rr_sel_m = nullptr;  // device: block row -> selection index (t*K+k), -1 for a padded row
    float* wts_h = nullptr;   // pinned: the router weights, D2H'd alongside the ids on g_st
};

struct Scratch {
    int ctx = 8192, pos = 0;
    // P4: the MMQ expert leg.  The library's contract is PER GROUP (see src/prefill/prefill.cpp:4936): each group of
    // <= 16 experts gets its own quantize of its own rows, group-relative bounds, an identity id table, a zeroed
    // MMQ tail, and a group-sized destination that the caller scatters.  Anything else reads the wrong rows.
    void *mmq_xq = nullptr, *mmq_xqh = nullptr, *mmq_gu_stage = nullptr, *mmq_d_stage = nullptr;
    float *mmq_gu = nullptr, *mmq_h = nullptr, *mmq_dm = nullptr;
    int32_t *mmq_row_tok = nullptr, *mmq_row_sel = nullptr, *mmq_bounds = nullptr, *mmq_ident = nullptr;
    std::vector<int32_t> mmq_row_tok_h, mmq_row_sel_h, mmq_bounds_h;
    int64_t mmq_rows_cap = 0;
    bool mmq_gout = false;   // P4: gout_m already holds the chunk's expert rows (no host round trip)
    float *x, *h, *y, *q, *k, *v, *att, *router, *eg, *eu, *eh, *ey, *sg, *su, *sh, *sy, *moe, *wts, *logits;
    __half* dec_h16 = nullptr;    // S6: the decode's fp16 activation copies (for the fp16 dense gemv)
    // K29: the residency path's own fp16 buffers - h's copy and the SwiGLU output.  They must NOT be dec_h16/dec_h16's
    // neighbours: those are allocated only when STRATA_KOLIBRI_DENSE_F16 is set (off by default), and reusing
    // dec_h16 would clobber the copy the shared expert's gemvs still need later in the same layer.  Allocated when
    // DEC_RES is on, independent of the dense-f16 gate - passing a null pointer to cublasGemmEx is exactly the
    // CUBLAS_STATUS_INVALID_VALUE this cost one build cycle to find.
    __half* dec_x16 = nullptr;
    __half* dec_eh16 = nullptr;
    __half* dec_att16 = nullptr;
    int* ids;
    std::vector<__half*> kc, vc;   // S3: FP16 KV -- half the bytes/token and half the prefill traffic
    std::vector<__half*> kt;       // K27: the transposed K mirror for the sliding layers (STRATA_KOLIBRI_ATTN_KT)
    std::vector<__half*> vt;       // K27: and the transposed V mirror (the PV pass reads V the same way)
    std::vector<int64_t> kt_rows;  // K27: each mirror's row stride (min(rows, two_pass_max))
    // K27b: is this layer's mirror already built for the current request?  Filling it once and then appending the new
    // row per token costs 1/513th of rebuilding the window every token - which the profile showed as 6.5% of the
    // decode's GPU time (578 calls x 36.9 us; it was my own overhead, the largest single line I had added).
    std::vector<char> kt_ready;
    float* layer_trace_d = nullptr;   // --dump-layers snapshots, downloaded only after token completion
    // the CPU expert path's pinned host side: the layer's h, its quantized activations, the FF rows, the down rows
    float* h_host = nullptr; char* act = nullptr; float* ff = nullptr; char* hq = nullptr; float* ey_h = nullptr;
    // A2b: the SECOND host working set.  Only ey_h was double-buffered; h_host/act/ff/hq were shared between the two
    // in-flight chunks, which was latent while the input copy was BLOCKING (the blocking copy implicitly serialised
    // everything) and became a real race the moment it was made asynchronous: chunk B's copy landed in chunk A's buffer
    // before A's pool read it.  PF_HASH caught it (its lines differed run to run).  These pointers let the pool path take
    // the buffer of the chunk it is actually computing.
    float* h_host_b = nullptr; char* act_b = nullptr; float* ff_b = nullptr; char* hq_b = nullptr;
    float* h_host_cur = nullptr; char* act_cur = nullptr; float* ff_cur = nullptr; char* hq_cur = nullptr;
    std::vector<int32_t> exp_cnt, exp_ord;   // prefill pool: the per-expert grouping
    std::vector<int32_t> pf_touch;           // touched experts, row count DESCENDING: the LPT deal (phase timers showed 84% worker idle)
    // S7 increment B: the resident-expert fp16 GEMM working set.  Picks whose expert is in the VRAM fp16 cache
    // (id < b.n_cached) run their gate/up/down products as cublas fp16 GEMMs on the gathered activation rows -
    // no PCIe blob read, no per-call dequant, no pool worker.  Sized to a full chunk (kTmax * K selections).
    uint16_t *res_x16 = nullptr, *res_h16 = nullptr;
    float *res_g = nullptr, *res_u = nullptr, *res_h = nullptr, *res_ey = nullptr;
    int32_t* res_tok = nullptr;              // device: token row per selection (gather index; -1 = padded)
    float* res_ey_h = nullptr;               // pinned: D2H staging in expert-grouped row order, scattered to ey_h
    std::vector<int32_t> res_tok_h, pf_res, pf_miss, pf_res_slot;
    // G1 PF_PARTIAL: rr_sel_h is the host side of the block-row -> selection map (built in the same loop that
    // fills res_tok_h, so the two cannot disagree); res_mark stamps which experts are resident THIS chunk-layer
    // (generation-stamped, so nothing has to be cleared between layers - the P1 class of bug was a stale list).
    std::vector<int32_t> rr_sel_h;
    std::vector<int32_t> res_mark;
    int32_t res_gen = 0;
    bool pf_partial_used = false;   // set when this layer's expert leg took the partial path (read at the MoE sum)
    uint64_t pf_clock = 0;                   // LRU clock for the fp16 expert cache (one tick per layer-chunk)
    // S7 increment C: batched resident GEMMs.  Each expert's tokens are cut into blocks of kResBlk rows (last
    // block padded); one cublasGemmBatchedEx per slice per layer replaces ~155 launches.  Pointer arrays live on
    // the device (cublas dereferences them on the GPU); host staging builds them, H2D ships them.
    static constexpr int kResBlk = 32;
    static constexpr int kResBlkMax = 512;   // sum ceil(m_i/32) <= n_expert + NK/32 = 384 + 96
    const uint16_t **pa_g = nullptr, **pa_u = nullptr, **pa_d = nullptr;
    const uint16_t **pb_x = nullptr, **pb_h = nullptr;
    float **pc_g = nullptr, **pc_u = nullptr, **pc_ey = nullptr;
    std::vector<const uint16_t*> pa_g_h, pa_u_h, pa_d_h, pb_x_h, pb_h_h;
    std::vector<float*> pc_g_h, pc_u_h, pc_ey_h;
    std::vector<int32_t> blk_expert, blk_lo, blk_m;   // per block: expert id, selection offset, valid rows
    int pf_nblk = 0;
    // S7: per-chunk expert staging.  The grouped prefill kernel reads the full 2.21 MB blob PER ENTRY (3072
    // entries per layer-chunk = 6.79 GB over PCIe at ~12 GB/s = 566 ms/layer), while the chunk touches only ~150
    // distinct experts.  Stage the touched set once per layer-chunk (150 x 2.21 MB = 332 MB = ~28 ms) and run the
    // SAME kernel against the staged copy, which is byte-identical, so the result must be bitwise-equal.
    uint8_t* stage = nullptr; int stage_n = 0;
    int* stage_ids_m = nullptr;
    std::vector<int32_t> stage_rank, stage_ids_h;
    // --expert-slots: the GPU-slot experts' rows and the overflow rows (CPU pool or pinned table)
    float *eg2 = nullptr, *eu2 = nullptr, *eh2 = nullptr, *ey2 = nullptr, *ey_cpu = nullptr;
    float* yc = nullptr;            // the slots path's overflow rows: ey_cpu (pool) or ey (pinned table)
    std::vector<int> slot_ids, ovf_ids;
    // --grouped: the per-layer grouped-expert call's device side
    unsigned long long* grp_ptr = nullptr; int32_t *grp_start = nullptr, *n_groups = nullptr;
    int32_t *ent_dst = nullptr, *ent_tok = nullptr; char* xq8 = nullptr; char* scratch = nullptr; float* gout = nullptr;
    int* pos_d = nullptr; int* tok_d = nullptr;   // graph decode: position and token id on the device
    // --prefill: the chunk's rows (tmax tokens at once)
    static constexpr int kTmax = 512;
    int* toks_d = nullptr;
    float *x_m = nullptr, *h_m = nullptr, *y_m = nullptr, *q_m = nullptr, *k_m = nullptr, *v_m = nullptr;
    float *att_m = nullptr, *router_m = nullptr, *moe_m = nullptr, *sy_m = nullptr;
    // A2b step 1: the SECOND activation set, so two chunks can be in flight in the wavefront.  Allocated and unused at
    // this step - the point is to prove the model still loads and that PF_HASH/oracle do not move with the memory merely
    // committed (~85 MiB for the set + the second staging buffer, 37% of the 708 MiB the 32K config leaves free).
    float *x_m2 = nullptr, *h_m2 = nullptr, *y_m2 = nullptr, *q_m2 = nullptr, *k_m2 = nullptr, *v_m2 = nullptr;
    float *att_m2 = nullptr, *router_m2 = nullptr, *moe_m2 = nullptr, *sy_m2 = nullptr;
    float *sg_m2 = nullptr, *su_m2 = nullptr, *sh_m2 = nullptr, *gout_m2 = nullptr;
    __half *x16_2 = nullptr, *q16_2 = nullptr;
    char *xq8_m2 = nullptr;
    int *ids_m2 = nullptr, *toks_d2 = nullptr;
    int *grp_start_m2 = nullptr, *n_groups_m2 = nullptr, *ent_dst_m2 = nullptr, *ent_tok_m2 = nullptr;
    unsigned long long *grp_ptr_m2 = nullptr;
    float *ey_h2 = nullptr;   // A2b: the second chunk's pinned expert-output block
    float *wts_m2 = nullptr;
    __half *k16_2 = nullptr;
    __half *v16_2 = nullptr;
    int *ids_h = nullptr;      // A2a: chunk 1's pinned published ids
    cudaEvent_t ev = nullptr;   // A2a: chunk 1's doorbell
    int *ids_h2 = nullptr;
    cudaEvent_t ev2 = nullptr;
    float *sg_m = nullptr, *su_m = nullptr, *sh_m = nullptr;
    float *gout_m = nullptr, *eg_m = nullptr, *eu_m = nullptr, *eh_m = nullptr;
    ActSet act1, act2;   // A2b: the two per-chunk activation sets (act2 only when STRATA_KOLIBRI_PF_PIPE is on)
    int *ids_m = nullptr, *grp_start_m = nullptr, *n_groups_m = nullptr, *ent_dst_m = nullptr, *ent_tok_m = nullptr;
    float* wts_m = nullptr;
    unsigned long long* grp_ptr_m = nullptr;
    char *xq8_m = nullptr, *scratch_m = nullptr;
    __half *x16 = nullptr, *q16 = nullptr, *att16 = nullptr, *k16 = nullptr, *v16 = nullptr, *p16 = nullptr;
    float* att_s = nullptr;
};

// S7: the router's token tile per thread (STRATA_KOLIBRI_ROUTER_TJ, default 32): 32 is the original shape;
// smaller tiles give more CTAs (occupancy) and a shorter FMA dependency chain.  Bitwise-identical either way
// (the per-token sum's order and the reduction tree are unchanged).
static inline int router_tk() {
    // K31: DEFAULT 3 (three experts per block, TJ=8).  Measured at 32K, both sockets: TK=0 120.5, TK=2/TJ=16 141.8,
    // TK=2/TJ=32 143.3, **TK=3/TJ=8 147.6** (148.7/144.1/150.0); at 4K TK=3 161.0 against TK=0 120.7.  Bitwise: identical
    // generated ids for TK = 0, 2, 3 and 4 on the same prompt with ATTN_FLASH on.  STRATA_KOLIBRI_ROUTER_TK=0 restores the
    // per-expert path for an A/B, 2 or 4 select the other block shapes.
    const char* v = std::getenv("STRATA_KOLIBRI_ROUTER_TK");
    return v ? std::atoi(v) : 3;
}
static inline int router_tj() {
    static const int t = std::getenv("STRATA_KOLIBRI_ROUTER_TJ") ? std::atoi(std::getenv("STRATA_KOLIBRI_ROUTER_TJ")) : 32;
    return t;
}

// S7: the prefill's chunk size at runtime (STRATA_KOLIBRI_PF_CHUNK, default 512 = Scratch::kTmax).  The
// router's GEMV re-reads the chunk's activation matrix once per expert row, so the traffic scales with the
// chunk: at T=512 the x matrix (T x N x 4 = 5.2 MB) does NOT fit the T4's 4 MB L2 and the 384 reads go to
// DRAM; smaller chunks keep x L2-resident.  The buffers are sized for kTmax, so a smaller chunk is safe.
static int64_t g_pf_total = 0;   // K28: the request's prefix length, so pf_chunk_at can key its rule on it
// K28: key the small-chunk rule on the prefix LENGTH instead of the position.  ON by default: at a 12K prefix it cuts
// the chunk-layers 2000 -> 1200 and the pool leg 42.2 -> 38.8 s (bytes 661 -> 488 GB, better dedup), and it is
// BITWISE - the 12K ids match the position-keyed rule exactly ("4991 66 96 568 262 3882 12400 379" both ways).  The
// divergence first seen here was the P1 residency bug, not the chunking: a chunked causal prefill computes each
// token's row and per-token softmax identically regardless of how the prompt is cut.  STRATA_KOLIBRI_PF_CHUNK_KEY=0
// restores the position-keyed rule for an A/B.
/// P4: prompt experts through MMQ. Default on Linux gfx1201; requires direct shared-expert tuning.
/// An explicit 0 keeps the pool. Inert in a build without STRATA_PREFILL_MMQ.
static inline bool pf_mmq_on() {
#if defined(STRATA_KOLIBRI_MMQ)
    static const bool v = [] {
        if (std::getenv("STRATA_KOLIBRI_PF_MMQ")) return env_on("STRATA_KOLIBRI_PF_MMQ");
#if defined(__HIPCC__) && !defined(_WIN32)
        int dev = 0;
        hipDeviceProp_t prop{};
        if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&prop, dev) != hipSuccess) return false;
        const std::string arch(prop.gcnArchName);
        return arch == "gfx1201" || arch.starts_with("gfx1201:");
#else
        return false;
#endif
    }();
    return v;
#else
    return false;
#endif
}
/// dst[sel[r] * ld + c] = src[r * ld + c] - the group's rows into their selection slots (the library writes group
/// rows with an identity table; the scatter is the caller's, as in prefill.cpp's peer_scatter_add).
__global__ void k_scatter_rows(const float* __restrict__ src, const int32_t* __restrict__ sel, int64_t rows, float* __restrict__ dst, int64_t ld) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * ld) return;
    const int64_t r = i / ld, c = i - r * ld;
    const int32_t d = sel[r];
    if (d < 0) return;   // padding row: the group was padded to a minimum rows-per-expert
    dst[(size_t) d * (size_t) ld + (size_t) c] = src[i];
}
static inline bool pf_chunk_key_total() {
    const char* v = std::getenv("STRATA_KOLIBRI_PF_CHUNK_KEY");
    return v == nullptr || (v[0] != 0 && std::atoi(v) != 0);
}
static inline int pf_chunk_at(int64_t p0) {
    // The env value CAPS the chunk; the default is prefix-dependent: a short prefix runs a SMALL chunk (the
    // router's x matrix then fits the T4's L2 - the 384 expert GEMVs re-read it, and at T=512 it is 5.2 MB vs
    // the 4 MB L2: pp512 4.73 s at T=256 vs 5.13 s at T=512), while a long prefix runs the full 512 (more chunks
    // mean more per-chunk overheads: at 32K, T=256 measured 320.9 s vs T=512's 301.5 s).
    static const int cap = [] {
        const char* v = std::getenv("STRATA_KOLIBRI_PF_CHUNK");
        int64_t n = v ? (int64_t) std::atoi(v) : (int64_t) Scratch::kTmax;
        if (n < 1) n = (int64_t) Scratch::kTmax;
        if (n > (int64_t) Scratch::kTmax) n = (int64_t) Scratch::kTmax;
        return (int) n;
    }();
    const int small = 256;
    // The rule keys on the PREFIX LENGTH, not on the position - K28.  Keyed on p0 it made a 32K prompt run the
    // small chunk for its first 8,192 tokens (32 of its 80 chunk-layers), even though the measurement behind the
    // rule was taken on long prefixes, where the full chunk wins (301.5 s at T=512 against 320.9 s at T=256).  A
    // genuinely short prefix (< 8192 tokens in total) keeps the 256 it was tuned for.
    if ((g_pf_total > 0 && pf_chunk_key_total() ? g_pf_total : p0) < 8192 && cap >= 256) return small < cap ? small : cap;
    return cap;
}



std::string L(int64_t l, const char* s) { return "blk." + std::to_string(l) + "." + s; }

/// One token through the engine at position s.pos.  Dense matmuls on the GPU tensors; the routed experts read
/// their [gate|up|down] blob from the pinned host table over UVA (the plain path: exactness first, the native
/// grouped path with VRAM slots arrives with the placement work).  Bitwise-equal to strata-kolibriref by
/// construction: the same kernels, the same order, the same weights.  `layers_host` (optional, 50 x n_embd)
/// receives each layer's output for this position, kolibriref's --dump-layers record.

/// One decoder step for layer l.  graph=true: everything device-side (position and token id read from
/// device pointers, grouped experts, two-pass attention - capturable as a CUDA graph); graph=false: the
/// plain path (value position, host ids for the mmvq loop, layer dumps).  st = 0 (legacy stream) keeps the
/// plain launches byte-identical; the graph path runs on g_st.
/// P1b: the CPU expert pool for one layer, on the library's ExpertPool.  `sels` are the selection indices whose
/// expert the pool owns - all K on --cpu-threads without slots, the slot misses with --expert-slots.  Each
/// expert's gate/up rows, the intermediate quantization and the down rows are split across every worker
/// (run_split_multi_native), so a blob streams at the pool's aggregate bandwidth instead of one core's.
/// Measured on this box by pool_bench (cold blobs, past the 60 MB L3): 6 experts/layer at 24 workers pinned to
/// the GPU's socket = 47.7 GB/s = 71.6 tok/s; one core = 7.3 GB/s; 48 workers on one socket collapse to
/// 4.5 GB/s, so --cpu-threads belongs near 24 on node 0, not at 56.
void pool_experts(const Engine& e, const LayerBlobs& b, Scratch& s, const std::vector<int>& ids_host,
                  const std::vector<int>& sels) {
    if (sels.empty() || !e.epool) return;
    const int64_t N = e.g.n_embd;
    std::vector<strata::kernels::cpu::ExpertJobMulti> jobs(sels.size());
    for (size_t j = 0; j < sels.size(); ++j) {
        const int sel = sels[j];
        jobs[j].blob = b.table + (size_t) ids_host[sel] * b.stride;
        jobs[j].nt = 1;
        jobs[j].nact[0] = s.act;                       // the layer's quantized activation, shared read-only
        jobs[j].out[0] = s.ey_h + (size_t) sel * N;    // unweighted; k_moe_sum2 applies the router weight
    }
    e.epool->run_split_multi_native(b.fmt, jobs.data(), (int) sels.size());
}

// S6: the decode's per-layer phase timers (measurement-only: the syncs serialize the stream, so this is
// only for pricing runs).
static inline bool dec_phases_on() {
    static const bool on = env_on("STRATA_KOLIBRI_DEC_PHASES");
    return on;
}

// K29: route the decode's RESIDENT picks through the fp16 cache on the GPU, leaving only the misses to the pool.  The
// coverage probe says a 24-slot per-layer cache covers 45.0% of routing, so the pool's 15 ms/token of blob reads should
// fall to ~8 ms.  Off by default: it changes the decode's arithmetic (fp16 experts instead of the q8 pool), so it
// needs the KL gate before it can be a default.
static inline bool dec_res_on() { return env_on("STRATA_KOLIBRI_DEC_RES"); }


// K30: run the decode's SHARED EXPERT during the CPU pool drain.  It depends only on `h` (already computed), so its
// five launches can be enqueued before pool_experts and the GPU executes them while the host computes the routed
// experts - the one piece of per-layer GPU work the dependency chain does NOT forbid.  Bitwise: same kernels, same
// order, same stream; only the enqueue point moves.  (The prefill version of this idea failed because there the shared
// expert WRITES the f16 activation buffer the routed path reads; at decode it only reads dec_h16.)
static inline bool dec_xexp_on() { return env_on("STRATA_KOLIBRI_DEC_XEXP"); }

// ------------------------------------------------------------------------------------------------------------------
// STRATA_KOLIBRI_DEC_HANDOFF: the decode's publication handoff (docs/KOLIBRI_DECODE.md, plan Task 3).
//
// The eager decode blocks twice per layer on device-to-host copies: the router's picks and the normalized
// FFN input.  Each blocking copy is a driver round trip in the middle of the layer, and the host cannot start
// the pool until the second one returns.  This arm publishes the same bytes with ONE kernel on the layer
// stream (`doorbell_publish`: payload + ring) and waits for the ring on the host with a bounded spin.
//
// Why it is bitwise by construction: the same kernels read and write the same buffers in the same order on the
// same stream.  Only the TRANSPORT of two buffers to the host changes (a kernel's stores instead of a copy
// engine's), and the host then reads those bytes.  The shared expert is enqueued between the publication and
// the spin, which is the STRATA_KOLIBRI_DEC_XEXP change (already measured bitwise) moved one step earlier, so
// the GPU's independent half runs while the host is still waiting for the payload and then computing.
//
// What it is NOT: a captured graph.  See docs/KOLIBRI_DECODE.md - the capturable attention variant forces the
// two-pass arithmetic and therefore changes the numbers (measured), so capture is a separate design, not a
// scheduling change this arm may smuggle in.
// ------------------------------------------------------------------------------------------------------------------
static inline bool dec_handoff_on();   // resolved by dec_arms_resolve() during startup

/// STRATA_KOLIBRI_DEC_ATTN_DP: run the decode's attention from `k_attn_dp` (single-pass body, position from
/// device memory, shared memory sized per position bucket) instead of `k_attn` (position by value).  Same
/// arithmetic - see the kernel's comment - and it is the prerequisite for capturing a bitwise attention: the
/// eager launch bakes `n_vis` into the dynamic shared-memory size, which a replayed graph cannot change.
static inline bool dec_attn_dp_on() { return env_on("STRATA_KOLIBRI_DEC_ATTN_DP"); }

/// STRATA_KOLIBRI_DEC_STREAM: run the decode on ONE explicitly created, non-blocking stream instead of the
/// legacy default stream (the router's launch and the pool's result copy move with it).
///
/// Two reasons this is its own switch rather than a detail of an arm.  (1) A legacy stream cannot be captured at
/// all, so capture needs this first and it must be provable on its own, with the eager kernels unchanged.  (2)
/// Every launch on the legacy stream is implicitly ordered against every blocking stream in the process, which
/// is a per-launch cost the token pays 1 344 times; an explicit stream pays it once, and that is measurable.
/// The arithmetic is untouched: same kernels, same buffers, same order - only which queue they are submitted to.
static inline bool dec_stream_on();    // resolved by dec_arms_resolve() during startup

/// STRATA_KOLIBRI_DEC_HANDOFF_GRAPH: per-layer graph segments.  The GPU's half of a layer before the CPU pool
/// (attention through the router and the publication) and the half after it (the MoE sum and the residual) are
/// captured once per (layer, position bucket) and replayed.  The CPU still computes every routed expert with the
/// same kernels, and it still services the layer in the middle of the token - no host callback runs inside a
/// capture, and no device acknowledgement is needed because the HOST enqueues the post segment after its own work.
/// Requires STRATA_KOLIBRI_DEC_STREAM (a legacy stream cannot be captured) and STRATA_KOLIBRI_DEC_HANDOFF (the
/// publication is what the captured pre segment ends with).
// ------------------------------------------------------------------------------------------------------------------
// The four decode scheduling arms are ON BY DEFAULT where the engine supports them, and each keeps an explicit
// off switch.  A request is three-valued: unset = automatic, "0" = off, anything else = on and STRICT.
//
// Automatic mode must never turn an unsupported configuration into a startup failure: the Terminal's tuner passes
// `--expert-slots auto`, and a user who never asked for these arms should get the plain decode plus one line
// naming why, not exit 2.  An explicit request is different: there a silent fallback would let someone believe a
// measured path ran when it did not, so it is refused by name before any model I/O.
struct DecArms {
    bool handoff = false, stream = false, graph = false, token = false;
};
static DecArms g_dec_arms;
static inline bool dec_handoff_on() { return g_dec_arms.handoff; }
static inline bool dec_stream_on() { return g_dec_arms.stream; }
static inline bool dec_graph_on() { return g_dec_arms.graph; }
static inline bool dec_token_graph_on() { return g_dec_arms.token; }

// 0 = off, 1 = unset (automatic), 2 = on and strict.
static inline int dec_arm_request(const char* name) {
    const char* v = std::getenv(name);
    if (v == nullptr) return 1;
    return (v[0] != 0 && std::atoi(v) != 0) ? 2 : 0;
}

/// Resolve the arms from their requests and an eligibility verdict.  `why` == nullptr means eligible; otherwise it
/// names the unsupported configuration.  Returns false only when an EXPLICIT request must be refused, having
/// printed each such arm and the reason; true means the arm state is settled for this run.
static bool dec_arms_resolve(const char* why) {
    static const char* const names[4] = {"STRATA_KOLIBRI_DEC_HANDOFF", "STRATA_KOLIBRI_DEC_STREAM",
                                         "STRATA_KOLIBRI_DEC_HANDOFF_GRAPH", "STRATA_KOLIBRI_DEC_TOKEN_GRAPH"};
    bool* const on[4] = {&g_dec_arms.handoff, &g_dec_arms.stream, &g_dec_arms.graph, &g_dec_arms.token};
    bool rejected = false;
    for (int i = 0; i < 4; ++i) {
        const int request = dec_arm_request(names[i]);
        if (why == nullptr) {
            *on[i] = request != 0;
        } else {
            *on[i] = false;
            if (request == 2) {
                std::fprintf(stderr, "kolibri: %s is set but not supported: %s\n", names[i], why);
                rejected = true;
            }
        }
    }
    if (rejected) return false;
    if (why != nullptr) {
        std::fprintf(stderr, "kolibri: decode acceleration off (%s); running the plain decode path\n", why);
        return true;
    }
    // The whole-token graph joins the other three arms and all three segments; an automatic request that cannot
    // be joined stays off, an explicit one is named.
    const bool joined = g_dec_arms.handoff && g_dec_arms.stream && g_dec_arms.graph &&
                        env_on_default_on("STRATA_KOLIBRI_DEC_GRAPH_PRE") &&
                        env_on_default_on("STRATA_KOLIBRI_DEC_GRAPH_POST") &&
                        env_on_default_on("STRATA_KOLIBRI_DEC_GRAPH_SHARED");
    if (g_dec_arms.token && !joined) {
        if (dec_arm_request("STRATA_KOLIBRI_DEC_TOKEN_GRAPH") == 2) {
            std::fprintf(stderr, "kolibri: STRATA_KOLIBRI_DEC_TOKEN_GRAPH requires HANDOFF, STREAM, HANDOFF_GRAPH, "
                                 "all PRE/POST/SHARED segments and no --dump-layers\n");
            return false;
        }
        g_dec_arms.token = false;
    }
    return true;
}
static uint64_t dec_wait_ticks() {
    static const uint64_t ticks = []() -> uint64_t {
        const char* text = std::getenv("STRATA_KOLIBRI_DEC_WAIT_TICKS");
        if (!text) return 1000000000ULL;
        uint64_t value = 0;
        const char* end = text + std::strlen(text);
        const auto result = std::from_chars(text, end, value);
        if (result.ec != std::errc{} || result.ptr != end || value > 1000000000ULL) {
            std::fprintf(stderr, "kolibri: STRATA_KOLIBRI_DEC_WAIT_TICKS must be an integer in [0,1000000000]\n");
            std::exit(2);
        }
        return value;
    }();
    return ticks;
}
// A malformed budget is a typo, not an arm setting: refuse it whether or not the arm resolves on, so a bounded-wait
// test cannot silently run with a different bound.
static void dec_wait_budget_check() { (void) dec_wait_ticks(); }
static thread_local std::atomic<bool>* decode_cancel = nullptr;
static thread_local bool decode_aborted = false;

struct DecodeCancelScope {
    std::atomic<bool>* previous;
    explicit DecodeCancelScope(std::atomic<bool>* flag) : previous(decode_cancel) { decode_cancel = flag; }
    ~DecodeCancelScope() { decode_cancel = previous; }
};

struct DecGraphArm {
    std::vector<std::map<int, cudaGraphExec_t>> pre;   // [layer][geometry key] -> captured pre segment
    std::vector<cudaGraphExec_t> post;                // [layer] -> captured post segment
    std::vector<std::map<int, cudaGraph_t>> pre_graph;
    std::vector<cudaGraph_t> post_graph;
    std::map<std::vector<int>, cudaGraphExec_t> tokens;
    long long captures = 0;
};

static DecGraphArm& dec_graph_arm(int64_t n_layers) {
    static DecGraphArm a;
    if (a.pre.empty()) {
        a.pre.resize((size_t) n_layers);
        a.post.assign((size_t) n_layers, nullptr);
        a.pre_graph.resize((size_t) n_layers);
        a.post_graph.assign((size_t) n_layers, nullptr);
    }
    return a;
}

/// Capture `body` on `stream` and instantiate it.  Any failure is fatal and named: a half-captured graph that was
/// silently ignored would make the arm look active while it ran the eager path.
template <class F>
static cudaGraphExec_t dec_capture_graph(F body, cudaStream_t stream, const char* what, int64_t layer,
                                         cudaGraph_t* retained = nullptr) {
    cudaGraph_t gr = nullptr;
    cudaGraphExec_t ex = nullptr;
    if (cudaStreamSynchronize(stream) != cudaSuccess) {   // capture starts from a quiescent point
        std::fprintf(stderr, "kolibri: DEC_GRAPH: %s: cannot quiesce the decode stream at layer %lld\n", what,
                     (long long) layer);
        std::exit(1);
    }
    if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        std::fprintf(stderr, "kolibri: DEC_GRAPH: %s: capture begin failed at layer %lld\n", what, (long long) layer);
        std::exit(1);
    }
    body();
    if (cudaStreamEndCapture(stream, &gr) != cudaSuccess) {
        std::fprintf(stderr, "kolibri: DEC_GRAPH: %s: capture end failed at layer %lld\n", what, (long long) layer);
        std::exit(1);
    }
    if (cudaGraphInstantiate(&ex, gr, 0) != cudaSuccess) {
        cudaGraphDestroy(gr);
        std::fprintf(stderr, "kolibri: DEC_GRAPH: %s: instantiate failed at layer %lld\n", what, (long long) layer);
        std::exit(1);
    }
    if (retained) *retained = gr;
    else cudaGraphDestroy(gr);
    return ex;
}

/// The decode's own stream, created on first use.  Non-blocking, so it does not implicitly synchronise with the
/// legacy stream; ordering against the prefill and the logits read is established by the end-of-token device
/// synchronise and by the per-token position copy being on this same stream.
cudaStream_t dec_stream() {
    static cudaStream_t s = nullptr;
    if (s == nullptr) {
        if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) {
            std::fprintf(stderr, "kolibri: DEC_STREAM: cannot create the decode stream\n");
            std::exit(1);
        }
    }
    return s;
}

/// Process-lifetime mapped publication buffers, like g_q8.  The ring word is never reset: it only counts up, so
/// a request reset cannot confuse the expectation (a reset that clears the ring without clearing the
/// expectation is how the synthetic test's first version hung on a stale target).
struct HandoffBuffers {
    float* hx = nullptr;        // device aliases of mapped host memory
    int32_t* ids = nullptr;
    float* w = nullptr;
    float* y = nullptr;        // mapped CPU result, device alias for the compute-queue copy
    float* h_y = nullptr;
    float* host_x = nullptr;
    int32_t* host_ids = nullptr;
    strata::kolibri::DecodeWaitControl* control = nullptr;
    strata::kolibri::DecodeWaitControl* host_control = nullptr;
    uint32_t* seq = nullptr;
    uint32_t* h_seq = nullptr;  // the host's own view of the same word
    uint32_t expect = 0;
};

static HandoffBuffers* handoff_buffers(int64_t n, int64_t k) {
    static HandoffBuffers b;
    if (b.hx != nullptr) return &b;
    float* hx = nullptr;
    int32_t* ids = nullptr;
    float* w = nullptr;
    uint32_t* seq = nullptr;
    auto mapped = [](void** h, size_t bytes) {
        if (cudaHostAlloc(h, bytes, cudaHostAllocMapped) != cudaSuccess) {
            std::fprintf(stderr, "kolibri: DEC_HANDOFF: cannot allocate %zu bytes of mapped host memory\n", bytes);
            std::exit(1);
        }
        std::memset(*h, 0, bytes);
    };
    mapped((void**) &hx, (size_t) n * 4);
    mapped((void**) &ids, (size_t) k * 4);
    mapped((void**) &w, (size_t) k * 4);
    mapped((void**) &b.h_y, (size_t) k * n * 4);
    mapped((void**) &seq, 64);   // the sequence word plus room for the documented tag/checksum slots
    if (cudaHostGetDevicePointer((void**) &b.hx, hx, 0) != cudaSuccess ||
        cudaHostGetDevicePointer((void**) &b.ids, ids, 0) != cudaSuccess ||
        cudaHostGetDevicePointer((void**) &b.w, w, 0) != cudaSuccess ||
        cudaHostGetDevicePointer((void**) &b.y, b.h_y, 0) != cudaSuccess ||
        cudaHostGetDevicePointer((void**) &b.seq, seq, 0) != cudaSuccess) {
        std::fprintf(stderr, "kolibri: DEC_HANDOFF: mapped host memory has no device alias on this stack\n");
        std::exit(1);
    }
    ::new ((void*) seq) strata::kolibri::DecodeWaitControl{};
    b.h_seq = seq;
    b.host_x = hx;
    b.host_ids = ids;
    b.control = reinterpret_cast<strata::kolibri::DecodeWaitControl*>(b.seq);
    b.host_control = reinterpret_cast<strata::kolibri::DecodeWaitControl*>(seq);
    return &b;
}

/// The handoff's deadline, in milliseconds (STRATA_KOLIBRI_DEC_HANDOFF_TIMEOUT_MS, default 5000).  A deadline that
/// a stalled publication cannot exceed is the difference between a diagnosable failure and a hung request.
static inline int dec_handoff_timeout_ms() {
    static const int v = [] {
        const char* s = std::getenv("STRATA_KOLIBRI_DEC_HANDOFF_TIMEOUT_MS");
        const int n = s != nullptr ? std::atoi(s) : 5000;
        return n > 0 ? n : 5000;
    }();
    return v;
}

/// The handoff's bounded host wait: spin on the mapped sequence word, without calling the driver, under a hard
/// deadline.  `false` means the deadline passed, and the caller must not use the payload.
static bool handoff_wait(const uint32_t* h_seq, uint32_t want, int timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    const volatile uint32_t* seq = h_seq;
    while (*seq < want) {
        if (std::chrono::steady_clock::now() > deadline) return false;
        std::this_thread::yield();
    }
    std::atomic_thread_fence(std::memory_order_acquire);   // the payload is read after the ring (elementwise.cu)
    return true;
}

/// K29: seed the decode's fp16 cache with each layer's HOT experts.  Without this the slots hold only the last prefill
/// chunk's RES_TOP promotions, which measured 27% of decode picks; the prompt's own routing says which 24 per layer
/// matter (the coverage probe measures 45.0% for a top-24 set).  ~1200 promotions, ~2.2 GB over PCIe, once per request.
// The prompt's routing, into the cache's two counters: the LIFETIME count (the profile's data) and the DECAYED score
// (the swap policy's).  Called once per prompt, before the post-prompt round; the caller clears touch_hist afterwards,
// because the prefill path only increments it and a second request would otherwise count the first one again.
static void hot_absorb_prompt(Engine& e) {
    if (!e.hot_on || e.touch_hist.empty()) return;
    const int64_t n_exp = (int64_t) e.g.n_expert;
    for (size_t l = 0; l < e.touch_hist.size(); ++l) {
        const std::vector<uint64_t>& h = e.touch_hist[l];
        for (int64_t id = 0; id < n_exp && id < (int64_t) h.size(); ++id) {
            if (h[(size_t) id] == 0) continue;
            const size_t ix = e.hot_idx((int64_t) l, id);
            e.hot_cnt[ix] += (long long) h[(size_t) id];
            if (!e.hot_score.empty()) e.hot_score[ix] += (double) h[(size_t) id];
        }
    }
}

static void hot_forget_prompt(Engine& e) {
    for (auto& h : e.touch_hist) std::fill(h.begin(), h.end(), (uint64_t) 0);
}

static void seed_hot(Engine& e, Scratch& s) {
    if (!dec_res_on() || e.touch_hist.empty()) return;
    const int64_t N = e.g.n_embd, FF = e.g.n_ff, f16_stride = 3 * FF * N;
    for (size_t l = 0; l < e.blobs.size(); ++l) {
        LayerBlobs& b = e.blobs[l];
        if (b.f16 == nullptr || b.n_cached <= 0) continue;
        const std::vector<uint64_t>& h = e.touch_hist[l];
        std::vector<int> idx(h.size());
        for (size_t i = 0; i < h.size(); ++i) idx[i] = (int) i;
        const size_t take = std::min<size_t>((size_t) b.n_cached, idx.size());
        std::partial_sort(idx.begin(), idx.begin() + take, idx.end(),
                          [&](int a, int c) { return h[(size_t) a] > h[(size_t) c]; });
        for (int si = 0; si < b.n_cached; ++si) {
            const int x = idx[(size_t) si];
            if (h[(size_t) x] == 0) { b.f16_slot_id[(size_t) si] = -1; continue; }   // prompt never touched it
            const uint8_t* blob = b.table + (size_t) x * b.stride;
            uint16_t* out = b.f16 + (size_t) si * f16_stride;
            strata::kernels::dequant_f16(b.L.gu_type, blob, 0, (int) FF, (int) N, out, g_st);
            strata::kernels::dequant_f16(b.L.gu_type, blob + b.L.up_off, 0, (int) FF, (int) N,
                                         out + (int64_t) FF * N, g_st);
            strata::kernels::dequant_f16(b.L.d_type, blob + b.L.down_off, 0, (int) N, (int) FF,
                                         out + 2 * (int64_t) FF * N, g_st);
            b.f16_slot_id[(size_t) si] = x;
            b.f16_slot_last[(size_t) si] = s.pf_clock;
        }
    }
    KC(cudaStreamSynchronize(g_st));
}

void layer_step(const Engine& e, Scratch& s, int64_t l, bool graph, cudaStream_t st, float* layers_host) {
    const auto& g = e.g;
    const int N = (int) g.n_embd, K = (int) e.top_k, FF = (int) g.n_ff, FS = (int) g.n_ff_shexp;
    const int H = (int) g.n_head, HKV = (int) g.n_head_kv, D = (int) g.head_dim;
    const int B = 256;
    (void) layers_host;
    std::chrono::steady_clock::time_point dp0{};
    if (dec_phases_on()) { KC(cudaStreamSynchronize(st)); dp0 = std::chrono::steady_clock::now(); }

        const bool sliding = strata::core::is_swa_layer(g, l);
        const LayerBlobs& b = e.blobs[(size_t) l];
        const int vis_cap = (int) (sliding ? g.n_swa : s.ctx);
        const int n_vis = (int) (s.pos + 1 < (int64_t) vis_cap ? s.pos + 1 : (int64_t) vis_cap);
        // K27's transposed mirror, resolved on the host: the first decode token of a request copies the whole
        // window, every later token appends one row.  A captured graph may only hold the steady form, so the
        // first token runs eager.
        const __half* ktc = nullptr;
        const __half* vtc = nullptr;
        bool kt_full = false;
        if (!graph && (size_t) l < s.kt.size() && s.kt[(size_t) l] != nullptr && n_vis <= g_two_pass_max) {
            kt_full = s.kt_ready[(size_t) l] == 0;
            ktc = s.kt[(size_t) l];
            vtc = s.vt[(size_t) l];
        }
        const int geometry_key = attn_graph_key(n_vis, g_two_pass_max);
        const int smem_rows = geometry_key > 0 ? geometry_key : -geometry_key;
        HandoffBuffers* hbp = dec_handoff_on() ? handoff_buffers(N, K) : nullptr;
        bool pre_published = false;
        bool shexp_done = false;
        auto run_shexp = [&] {
            gemv_dec(e.dense.at(L(l, "ffn_gate_shexp.weight")), s.h, s.dec_h16, s.sg, st);
            gemv_dec(e.dense.at(L(l, "ffn_up_shexp.weight")), s.h, s.dec_h16, s.su, st);
            k_swiglu<<<(FS + B - 1) / B, B, 0, st>>>(s.sg, s.su, s.sh, FS);
            if (s.dec_h16) k_to_f16<<<dim3((FS + 127) / 128), 128, 0, st>>>(s.sh, s.dec_h16, FS);
            gemv_dec(e.dense.at(L(l, "ffn_down_shexp.weight")), s.sh, s.dec_h16, s.sy, st);
            shexp_done = true;
        };
        // THE GPU'S HALF OF THE LAYER, up to and including the publication.  One lambda, because the arm captures
        // exactly this sequence and the eager path runs it; `steady` selects the device-position kernels and the
        // incremental mirror row, which is what a captured graph may contain.  With `steady == false` this is the
        // eager sequence, unchanged, branch for branch.
        auto pre_body = [&](bool steady) {
        k_rms<<<1, kThreads, 0, st>>>(s.x, e.f32.at(L(l, "attn_norm.weight")), s.h, N, e.eps);
        if (s.dec_h16) k_to_f16<<<dim3((N + 127) / 128), 128, 0, st>>>(s.h, s.dec_h16, N);   // S6: the fp16 copy for the dense gemvs
        gemv_dec(e.dense.at(L(l, "attn_q.weight")), s.h, s.dec_h16, s.q, st);
        gemv_dec(e.dense.at(L(l, "attn_k.weight")), s.h, s.dec_h16, s.k, st);
        gemv_dec(e.dense.at(L(l, "attn_v.weight")), s.h, s.dec_h16, s.v, st);
        if (graph || steady || dec_attn_dp_on())
            k_attn_prep_p<<<H + HKV, kThreads, 0, st>>>(s.q, s.k, s.v, e.f32.at(L(l, "attn_q_norm.weight")),
                                           e.f32.at(L(l, "attn_k_norm.weight")), s.q, s.kc[l], s.vc[l], H, D, s.pos_d,
                                           g.n_swa, s.ctx, e.rope_base, e.eps, sliding);
        else
            k_attn_prep<<<H + HKV, kThreads, 0, st>>>(s.q, s.k, s.v, e.f32.at(L(l, "attn_q_norm.weight")),
                                           e.f32.at(L(l, "attn_k_norm.weight")), s.q, s.kc[l], s.vc[l], H, D, s.pos,
                                           g.n_swa, s.ctx, e.rope_base, e.eps, sliding);
        if (graph)
            k_attn_p<<<H, 512, (size_t) 4096 * sizeof(float), st>>>(
                s.q, s.kc[l], s.vc[l], s.att, H, HKV, D, s.pos_d, g.n_swa, s.ctx, 1.0f / sqrtf((float) D), sliding);
        else {
            if (ktc != nullptr) {
                const int64_t rows_kt = s.kt_rows[(size_t) l];
                if (steady) {   // one row: the position is on the device, so a replay appends the current one
                    k_kt_window_dp<<<(unsigned) ((HKV * D + 255) / 256), 256, 0, st>>>(
                        s.kc[l], s.vc[l], s.kt[(size_t) l], s.vt[(size_t) l], rows_kt, s.pos_d, g.n_swa, D, HKV, sliding);
                } else {
                    const int64_t n_kt = kt_full ? n_vis : 1;
                    const int64_t first_kt = sliding ? (kt_full ? (int64_t) s.pos + 1 - n_vis : (int64_t) s.pos) : 0;
                    const int64_t total = (int64_t) HKV * n_kt * (int64_t) D;
                    k_kt_window<<<(unsigned) ((total + 255) / 256), 256, 0, st>>>(
                        s.kc[l], s.vc[l], s.kt[(size_t) l], s.vt[(size_t) l], rows_kt, (int) n_kt, first_kt, g.n_swa, D,
                        HKV, sliding);
                }
            }
            if (steady || (dec_attn_dp_on() && n_vis <= g_two_pass_max))
                k_attn_dp<<<H, geometry_key > 0 ? kThreads : 512, (size_t) smem_rows * sizeof(float), st>>>(
                    s.q, s.kc[l], s.vc[l], s.att, H, HKV, D, s.pos_d, g.n_swa, s.ctx, 1.0f / sqrtf((float) D), sliding,
                    g_two_pass_max, ktc, vtc);
            else
            k_attn<<<H, n_vis <= g_two_pass_max ? kThreads : 512, n_vis <= g_two_pass_max ? (size_t) n_vis * sizeof(float) : (size_t) 4096 * sizeof(float), st>>>(
                s.q, s.kc[l], s.vc[l], s.att, H, HKV, D, s.pos, g.n_swa, s.ctx, 1.0f / sqrtf((float) D), sliding, 0, g_two_pass_max, ktc, vtc);
        }
        if (ktc != nullptr && !steady) s.kt_ready[(size_t) l] = 1;   // host bookkeeping: the eager form ran
        if (!steady && reuse_gate_kv_path() && (int) l == attn_dump_layer()) {
            const int64_t stride = sliding ? (int64_t) g.n_swa : (int64_t) s.ctx;
            const int64_t rows = std::min<int64_t>((int64_t) s.pos + 1, stride);
            std::vector<__half> kh, vh;
            if (reuse_gate_copy_kv(s.kc[l], stride, rows, HKV, D, kh) &&
                reuse_gate_copy_kv(s.vc[l], stride, rows, HKV, D, vh)) {
                if (std::FILE* f = std::fopen(reuse_gate_kv_path(), "ab")) {
                    const int32_t hdr[4] = {(int32_t) s.pos, (int32_t) rows, (int32_t) stride, 0x4b564344};
                    std::fwrite(hdr, 4, 4, f);
                    std::fwrite(kh.data(), sizeof(__half), kh.size(), f);
                    std::fwrite(vh.data(), sizeof(__half), vh.size(), f);
                    std::fclose(f);
                }
            }
        }
        if (s.dec_att16) k_to_f16<<<dim3((H * D + 127) / 128), 128, 0, st>>>(s.att, s.dec_att16, H * D);   // S6
        gemv_dec(e.dense.at(L(l, "attn_output.weight")), s.att, s.dec_att16, s.y, st);
        k_rms_add<<<1, kThreads, 0, st>>>(s.y, e.f32.at(L(l, "post_attention_norm.weight")), s.x, N, e.eps);
        k_rms<<<1, kThreads, 0, st>>>(s.x, e.f32.at(L(l, "ffn_norm.weight")), s.h, N, e.eps);
        gemv_dec(e.dense.at(L(l, "ffn_gate_inp.weight")), s.h, s.dec_h16, s.router, st);
        strata::kernels::native_router_sigmoid_logit_add(s.router, e.f32.at(L(l, "exp_probs_b.bias")), s.ids, s.wts, 1,
                                                         dec_stream_on() ? st : g_st);
        if (steady) {   // the captured segment ends with the publication the CPU is waiting for
            strata::kernels::doorbell_publish(s.h, s.ids, s.wts, N, K, hbp->hx, hbp->ids, hbp->w, hbp->seq, st);
            pre_published = true;
            if (env_on_default_on("STRATA_KOLIBRI_DEC_GRAPH_SHARED"))
                run_shexp();   // diagnostic control: capture shared work, or enqueue it after pre replay
        }
        };
        // The eager sequence runs for every token when the arm is off, and for the request's FIRST token (the
        // whole-window mirror copy) when it is on.  Later tokens go through the captured pre segment below.
        if (!dec_graph_on() || kt_full || !env_on_default_on("STRATA_KOLIBRI_DEC_GRAPH_PRE"))
            pre_body(false);
        // The layer's half AFTER the CPU: the MoE sum and the residual.  Also a lambda, for the same reason.
        bool post_done = false;
        auto post_body = [&] {
            if (dec_graph_on())
                strata::kernels::copy_from_mapped(s.ey, hbp->y, (int64_t) K * N, st);
            if (e.grouped > 0 && b.native_ok) k_moe_sum<<<(N + B - 1) / B, B, 0, st>>>(s.moe, s.gout, s.wts, K, s.sy, N);
            else if (e.slots > 0) k_moe_sum2<<<(N + B - 1) / B, B, 0, st>>>(s.moe, s.ey2, s.yc, s.wts, K, s.sy, N);
            else k_moe_sum<<<(N + B - 1) / B, B, 0, st>>>(s.moe, s.ey, s.wts, K, s.sy, N);
            k_rms_add<<<1, kThreads, 0, st>>>(s.moe, e.f32.at(L(l, "post_ffw_norm.weight")), s.x, N, e.eps);
        };
        const bool dph = dec_phases_on();
        std::chrono::steady_clock::time_point dpt{};
        if (dph) { KC(cudaStreamSynchronize(st)); dpt = std::chrono::steady_clock::now(); }
        if (!graph && !dec_handoff_on()) {   // the handoff arm reads the picks out of the published payload
            g_ids_host.resize((size_t) K);
            if (dec_stream_on()) {   // the picks are on the decode stream; a plain cudaMemcpy would read the legacy one
                KC(cudaMemcpyAsync(g_ids_host.data(), s.ids, K * sizeof(int), cudaMemcpyDeviceToHost, st));
                KC(cudaStreamSynchronize(st));
            } else
            KC(cudaMemcpy(g_ids_host.data(), s.ids, K * sizeof(int), cudaMemcpyDeviceToHost));
        }
        Dense gu; gu.type = b.L.gu_type; gu.shape = {g.n_embd, g.n_ff, g.n_expert}; gu.row_bytes = b.L.gu_row;
        Dense dn; dn.type = b.L.d_type; dn.shape = {g.n_ff, g.n_embd, g.n_expert}; dn.row_bytes = b.L.d_row;
        // --grouped: one native_expert_grouped launch per layer reads the blobs from the pinned table over UVA
        // (the per-selection mmvq loop's 18 launches per layer become one; the grouped kernel's rows are ggml's
        // own arithmetic, native_grouped_parity).  Unsupported layers (Q6_K down) keep the mmvq loop.
        if (e.grouped > 0 && b.native_ok) {
            if (e.counters) ++e.cnt.br_grouped;
            k_group<<<1, 1024, 0, st>>>(s.ids, K, K, (int) g.n_expert, b.table, (unsigned long long) b.stride,
                                 s.grp_ptr, s.grp_start, s.n_groups, s.ent_dst, s.ent_tok);
            strata::kernels::native_quantize_q8_1(s.h, s.xq8, (int) g.n_embd, 1, g_st);
            strata::kernels::native_expert_grouped(b.L, s.grp_ptr, s.grp_start, s.n_groups, s.ent_dst, s.ent_tok,
                                                   std::min<int64_t>(K, g.n_expert) + 1, K, s.xq8, s.scratch,
                                                   s.gout, g_st);
        } else
        // --expert-slots: the hot experts live in VRAM (the GPU expert cache); the rest overflow to the CPU
        // pool (--cpu-threads) or to the pinned table over UVA.  Bitwise: the same native_mmvq kernel reads the
        // same bytes wherever the blob sits; the side that skips a selection leaves its row zero (k_moe_sum2).
        if (!(e.grouped > 0 && b.native_ok)) {
        if (e.slots > 0) {
            if (e.counters) ++e.cnt.br_slots;
            const int S = e.slots;
            s.slot_ids.resize((size_t) K); s.ovf_ids.resize((size_t) K);
            for (int k = 0; k < K; ++k) {
                const int id = g_ids_host[(size_t) k];
                // Where this pick is served from.  The id path (default) is slot == id for the first `slots` ids,
                // which is the historical cache byte for byte.  The hot path asks the residency map instead, so a
                // slot can hold any expert; -1 means "not resident" and the pick falls through to the pool/UVA leg.
                int sidx;
                if (e.hot_on && (int) e.slot_of.size() ==
                                    (e.per_layer ? (int) (g.n_layers * g.n_expert) : (int) g.n_expert))
                    sidx = e.slot_of[e.hot_idx(l, id)];
                else sidx = id < S ? id : -1;
                s.slot_ids[k] = sidx >= 0 ? sidx : S;                                          // dev zero slot
                s.ovf_ids[k] = sidx >= 0 ? (e.cpu_threads > 0 ? sidx : (int) g.n_expert) : id;  // pinned zero slot
                // Counted ALWAYS, not only under --counters: the DONE line carries these as the request's hits and
                // lookups, which is what the monitor's hit rate is computed from.  Two integer adds per routed pick.
                // slot_bytes stays behind --counters because only the CNT report reads it.
                if (sidx >= 0) ++e.cnt.slot_hits;
                else { ++e.cnt.slot_miss; if (e.counters) e.cnt.slot_bytes += (long long) b.stride; }
                if (e.hot_on) {
                    ++e.hot_cnt[e.hot_idx(l, id)];   // the LIFETIME count the profile writes
                    if (!e.hot_score.empty()) e.hot_score[e.hot_idx(l, id)] += 1.0;   // the decayed score the swap
                    // policy sorts on: same index, halved about every 1.9 rounds (see kolibri_hot_policy.hpp)
                }
            }
            g_ids_host = s.slot_ids;
            gemv(gu, s.h, s.eg2, s.ids, K, 0, b.dev, b.stride);   // K21: this gate gemv ran twice per layer
            gemv(gu, s.h, s.eu2, s.ids, K, 0, b.dev + b.L.up_off, b.stride);
            k_swiglu<<<(K * FF + B - 1) / B, B, 0, st>>>(s.eg2, s.eu2, s.eh2, K * FF);
            gemv(dn, s.eh2, s.ey2, s.ids, K, FF, b.dev + b.L.down_off, b.stride);
            KC(cudaStreamSynchronize(g_st));   // the slot gemv run on g_st; k_moe_sum2 below reads ey2 on the default stream
            float* yc;
            if (e.cpu_threads > 0 && b.pool_ok) {   // the pool's own capability, not the GPU's (see LayerBlobs)
                if (e.counters) ++e.cnt.br_pool;
                g_ids_host.assign(g_ids_host.size(), 0);   // restored below from the device ids
                KC(cudaMemcpy(g_ids_host.data(), s.ids, K * sizeof(int), cudaMemcpyDeviceToHost));
                if (dec_stream_on()) {   // same reason as the picks: an explicit stream needs an explicit wait
                KC(cudaMemcpyAsync(s.h_host, s.h, N * 4, cudaMemcpyDeviceToHost, st));
                KC(cudaStreamSynchronize(st));
            } else
            KC(cudaMemcpy(s.h_host, s.h, N * 4, cudaMemcpyDeviceToHost));
                if (e.counters) { e.cnt.d2h_n += 2; e.cnt.d2h_bytes += K * sizeof(int) + (long long) N * 4; }
                strata::kernels::cpu::native_quant_act(b.fmt, s.h_host, s.act);
                std::memset(s.ey_h, 0, (size_t) K * N * 4);
                std::vector<int> sels;
                for (int k = 0; k < K; ++k) if (g_ids_host[k] >= S) sels.push_back(k);   // in-slot: the GPU did it
                pool_experts(e, b, s, g_ids_host, sels);
                // K15/K18: the result copy is enqueued on the compute stream, not run as a blocking host copy.
                // s.ey_h is cudaHostAlloc-pinned (:1127), so the DMA is genuinely asynchronous: the host returns
                // here and goes on to enqueue the next layer's router/attention launches while the 61 KB moves,
                // and k_moe_sum2 below is stream-ordered after the copy.  The blocking form serialized the host
                // against the GPU once per layer, 50 times per token.
                KC(cudaMemcpyAsync(s.ey_cpu, s.ey_h, (size_t) K * N * 4, cudaMemcpyHostToDevice, g_st));
                if (e.counters) { ++e.cnt.h2d_n; e.cnt.h2d_bytes += (long long) K * N * 4; }
                yc = s.ey_cpu;
            } else {
                g_ids_host = s.ovf_ids;
                if (e.counters) ++e.cnt.br_gpu;   // slots miss served from the pinned table over PCIe
                gemv(gu, s.h, s.eg, s.ids, K, 0, b.table, b.stride);
                gemv(gu, s.h, s.eu, s.ids, K, 0, b.table + b.L.up_off, b.stride);
                k_swiglu<<<(K * FF + B - 1) / B, B, 0, st>>>(s.eg, s.eu, s.eh, K * FF);
                gemv(dn, s.eh, s.ey, s.ids, K, FF, b.table + b.L.down_off, b.stride);
                yc = s.ey;
            }
            KC(cudaStreamSynchronize(g_st));   // the shared-expert gemv ran on g_st; the sum below reads sy on the default stream
            s.yc = yc;   // the MoE sum runs in the shared code, after the shared-expert gemv
        } else if (e.cpu_threads > 0 && b.pool_ok) {   // the pool's own capability, not the GPU's (see LayerBlobs)
            if (e.counters) ++e.cnt.br_pool;
            if (dph) { KC(cudaStreamSynchronize(st)); const auto n = std::chrono::steady_clock::now();
                       e.cnt.dec_d2h_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(n - dpt).count();
                       dpt = n; }
            if (dec_handoff_on()) {
                // ONE publication node for the activation, the picks and the weights, then a bounded host spin.
                // Nothing about the arithmetic changes: the CPU is handed the same bytes the blocking copies
                // would have handed it, and the same kernels consume them afterwards.
                HandoffBuffers& hb = *hbp;
                cudaGraphExec_t pre_ex = nullptr;
                if (dec_graph_on() && !kt_full && env_on_default_on("STRATA_KOLIBRI_DEC_GRAPH_PRE")) {
                    // The steady GPU half of this layer, captured once per (layer, shared-memory bucket) and then
                    // replayed.  The capture IS this same sequence: no launch in it is a capture-time constant, and
                    // the position reaches the device-position kernels through s.pos_d.
                    DecGraphArm& dga = dec_graph_arm((int64_t) e.blobs.size());
                    auto& slot = dga.pre[(size_t) l][geometry_key];
                    if (slot == nullptr) {
                        // Capture must not count as a publication: pre_body(true) ends with the publish, and during
                        // capture those launches are recorded, not executed.  The ring is incremented by the REPLAY.
                        const bool saved = pre_published;
                        slot = dec_capture_graph([&] { pre_body(true); }, st, "pre", l,
                                                 &dga.pre_graph[(size_t) l][geometry_key]);
                        pre_published = saved;
                        ++dga.captures;
                    }
                    pre_ex = slot;
                    KC(cudaGraphLaunch(pre_ex, st));
                    shexp_done = env_on_default_on("STRATA_KOLIBRI_DEC_GRAPH_SHARED");
                    static bool announced_replay = false;
                    if (!announced_replay) {
                        std::fprintf(stderr, "kolibri: DEC_HANDOFF_GRAPH replay active\n");
                        announced_replay = true;
                    }
                }
                // A replayed pre segment published from inside the graph; the eager forms (arm off, or the request's
                // first token) publish here.  Publishing twice would advance the ring past the host's expectation.
                if (pre_ex == nullptr)
                    strata::kernels::doorbell_publish(s.h, s.ids, s.wts, N, K, hb.hx, hb.ids, hb.w, hb.seq, st);
                if (!shexp_done) run_shexp();   // eager first use; replays already contain the shared expert
                const uint32_t want = ++hb.expect;
                if (!handoff_wait(hb.h_seq, want, dec_handoff_timeout_ms())) {
                    std::fprintf(stderr, "kolibri: DEC_HANDOFF: layer %lld: no publication within %d ms (ring %u); "
                                         "refusing to compute on a stale payload\n",
                                 (long long) l, dec_handoff_timeout_ms(), want);
                    std::fflush(stderr);
                    std::exit(1);
                }
                std::memcpy(s.h_host, hb.host_x, (size_t) N * 4);
                g_ids_host.resize((size_t) K);
                for (int k = 0; k < K; ++k) g_ids_host[(size_t) k] = hb.host_ids[(size_t) k];
                if (e.counters) { ++e.cnt.d2h_n; e.cnt.d2h_bytes += (long long) N * 4 + (long long) K * 4; }
                strata::kernels::cpu::native_quant_act(b.fmt, s.h_host, s.act);
            } else {
            if (dec_stream_on()) {   // same reason as the picks: an explicit stream needs an explicit wait
                KC(cudaMemcpyAsync(s.h_host, s.h, N * 4, cudaMemcpyDeviceToHost, st));
                KC(cudaStreamSynchronize(st));
            } else
            KC(cudaMemcpy(s.h_host, s.h, N * 4, cudaMemcpyDeviceToHost));
            if (e.counters) { ++e.cnt.d2h_n; e.cnt.d2h_bytes += (long long) N * 4; }   // was uncounted: the pool-only leg read as "no pool" from CNT
            strata::kernels::cpu::native_quant_act(b.fmt, s.h_host, s.act);
            if (dec_xexp_on()) run_shexp();   // K30: the GPU works during the drain below
            }
            std::vector<int> sels((size_t) K);
            for (int k = 0; k < K; ++k) sels[(size_t) k] = k;
            // K29: resident picks run on the GPU from the fp16 cache; only the misses go to the pool.
            std::vector<int> res_k;
            std::vector<int> res_slot;
            if (dec_res_on() && b.f16 != nullptr && !b.f16_slot_id.empty()) {
                res_k.reserve((size_t) K);
                sels.clear();
                for (int k = 0; k < K; ++k) {
                    const int id = g_ids_host[(size_t) k];
                    int slot = -1;
                    for (int si = 0; si < b.n_cached; ++si)
                        if (b.f16_slot_id[(size_t) si] == id) { slot = si; break; }
                    if (slot >= 0) { res_k.push_back(k); res_slot.push_back(slot); }
                    else sels.push_back(k);
                }
                if (e.counters) { e.cnt.dec_res_hits += (long long) res_k.size(); e.cnt.dec_res_picks += K; }
            }
            pool_experts(e, b, s, g_ids_host, sels);   // no-op when every pick was resident
            if (dph) { const auto n = std::chrono::steady_clock::now();
                       e.cnt.dec_pool_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(n - dpt).count();
                       dpt = n; }
            // K15/K18: async on the compute stream (pinned source); see the slots path above for why.
            if (dec_graph_on()) {
                std::memcpy(hbp->h_y, s.ey_h, (size_t) K * N * 4);
                std::atomic_thread_fence(std::memory_order_release);
            } else {
                KC(cudaMemcpyAsync(s.ey, s.ey_h, (size_t) K * N * 4, cudaMemcpyHostToDevice,
                                   dec_stream_on() ? st : g_st));
            }
            if (dec_graph_on() && env_on_default_on("STRATA_KOLIBRI_DEC_GRAPH_POST")) {
                // The layer's half after the CPU - the MoE sum and the residual - replayed from a captured graph.
                // No device acknowledgement is needed for this shape: the HOST enqueues it after its own work, so
                // stream order is the whole contract, and a CPU failure simply never launches it.
                DecGraphArm& dga = dec_graph_arm((int64_t) e.blobs.size());
                if (dga.post[(size_t) l] == nullptr)
                    dga.post[(size_t) l] = dec_capture_graph([&] { post_body(); }, st, "post", l,
                                                           &dga.post_graph[(size_t) l]);
                KC(cudaGraphLaunch(dga.post[(size_t) l], st));
                post_done = true;
            }
            if (e.counters) { ++e.cnt.h2d_n; e.cnt.h2d_bytes += (long long) K * N * 4; }   // was uncounted
            // K29: the resident rows are written AFTER the copy above (same stream), so they win over whatever the
            // pool left in s.ey_h for those rows.  No new kernel: gemm.f16 takes f16 weights and T = 1.
            if (!res_k.empty()) {
                // layer_step takes `const Engine&` (the decode never mutates the engine), but Gemm::f16 is non-const;
                // the GEMM object's only mutable state is its cuBLAS handle, which is what we want to use.
                auto& gemm = const_cast<strata::prefill::Gemm&>(e.gemm);
                // ONE BATCHED CALL PER SLICE, not a three-call chain per pick.  Measured: the per-pick chain cost ~40 us
                // of serialized cuBLAS latency per resident pick (174 picks/token = ~7 ms), which ate the ~8.7 ms the
                // pool saved.  The batched form is the prefill's own mechanism (f16_batched over pointer arrays), so the
                // block bookkeeping is already proven.
                const int nblk = (int) res_k.size();
                const int64_t f16_stride = 3 * (int64_t) FF * N;
                k_to_f16<<<dim3((N + 127) / 128), 128, 0, st>>>(s.h, s.dec_x16, N);   // h in f16 for this layer
                for (int r = 0; r < nblk; ++r) {
                    const uint16_t* xe = b.f16 + (int64_t) res_slot[(size_t) r] * f16_stride;
                    s.pa_g_h[(size_t) r] = xe;
                    s.pa_u_h[(size_t) r] = xe + (int64_t) FF * N;
                    s.pa_d_h[(size_t) r] = xe + 2 * (int64_t) FF * N;
                    s.pb_x_h[(size_t) r] = (const uint16_t*) s.dec_x16;
                    s.pb_h_h[(size_t) r] = (const uint16_t*) s.dec_eh16 + (int64_t) r * FF;
                    s.pc_g_h[(size_t) r] = s.eg + (int64_t) r * FF;
                    s.pc_u_h[(size_t) r] = s.eu + (int64_t) r * FF;
                    s.pc_ey_h[(size_t) r] = s.ey + (size_t) res_k[(size_t) r] * N;
                }
                const size_t pb = (size_t) nblk * sizeof(void*);
                KC(cudaMemcpyAsync(s.pa_g, s.pa_g_h.data(), pb, cudaMemcpyHostToDevice, g_st));
                KC(cudaMemcpyAsync(s.pa_u, s.pa_u_h.data(), pb, cudaMemcpyHostToDevice, g_st));
                KC(cudaMemcpyAsync(s.pa_d, s.pa_d_h.data(), pb, cudaMemcpyHostToDevice, g_st));
                KC(cudaMemcpyAsync(s.pb_x, s.pb_x_h.data(), pb, cudaMemcpyHostToDevice, g_st));
                KC(cudaMemcpyAsync(s.pb_h, s.pb_h_h.data(), pb, cudaMemcpyHostToDevice, g_st));
                KC(cudaMemcpyAsync(s.pc_g, s.pc_g_h.data(), pb, cudaMemcpyHostToDevice, g_st));
                KC(cudaMemcpyAsync(s.pc_u, s.pc_u_h.data(), pb, cudaMemcpyHostToDevice, g_st));
                KC(cudaMemcpyAsync(s.pc_ey, s.pc_ey_h.data(), pb, cudaMemcpyHostToDevice, g_st));
                gemm.f16_batched(s.pa_g, s.pb_x, s.pc_g, 1, FF, N, nblk);
                gemm.f16_batched(s.pa_u, s.pb_x, s.pc_u, 1, FF, N, nblk);
                k_swiglu<<<grid1((int64_t) nblk * FF), kThreads>>>(s.eg, s.eu, s.eh, nblk * FF);
                k_to_f16<<<grid1((int64_t) nblk * FF), kThreads>>>(s.eh, s.dec_eh16, nblk * FF);
                gemm.f16_batched(s.pa_d, s.pb_h, s.pc_ey, 1, N, FF, nblk);
            }
        } else {
        if (e.counters) ++e.cnt.br_gpu;   // the plain pinned-table mmvq path
        gemv(gu, s.h, s.eg, s.ids, K, 0, b.table, b.stride);
        gemv(gu, s.h, s.eu, s.ids, K, 0, b.table + b.L.up_off, b.stride);
        k_swiglu<<<(K * FF + B - 1) / B, B, 0, st>>>(s.eg, s.eu, s.eh, K * FF);
        gemv(dn, s.eh, s.ey, s.ids, K, FF, b.table + b.L.down_off, b.stride);
        }
        }
        if (!shexp_done) run_shexp();
        if (!post_done) post_body();
        if (layers_host && !graph)
            k_layer_snapshot<<<(N + 255) / 256, 256, 0, st>>>(s.x, s.layer_trace_d + (size_t) l * N, N);
        if (dph) { KC(cudaStreamSynchronize(st)); const auto n = std::chrono::steady_clock::now();
                   e.cnt.dec_post_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(n - dpt).count();
                   e.cnt.dec_pre_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(dpt - dp0).count();
                   e.cnt.dec_layers++; }
        KC(cudaGetLastError());
    
}


/// Chunked prompt processing (--prefill): T rows at once.  Dense projections run as GEMMs (FP16 activations, the
/// weights dequantized to FP16 in the Gemm scratch - what llama.cpp's batched CUDA path does), the attention
/// scores and P.V as batched FP16 GEMMs with the causal (and window) softmax in between, the routed experts as
/// one grouped launch per layer (T*K entries), the Q6_K-down layers through the mmvq loop with host ids.
/// Bitwise vs the per-token plain path is not expected (FP16 GEMM scores); the gates are the KL parity and the
/// serve==tokens agreement.
void forward_multi(Engine& e, Scratch& s, int T, int64_t p0, int T2 = 0, int64_t p02 = 0) {
    const auto& g = e.g;
    const int N = (int) g.n_embd, K = (int) e.top_k, FF = (int) g.n_ff, FS = (int) g.n_ff_shexp;
    const int H = (int) g.n_head, HKV = (int) g.n_head_kv, D = (int) g.head_dim;
    const int B = 256;
    const int NK = T * K;
    // The uncalibrated rocBLAS -> hipBLASLt route corrupted shared gate/up outputs in this launch mix.
    // A compatible direct tuning entry is required; otherwise keep the CPU pool. No timing drains are needed.
    [[maybe_unused]] const bool mmq_safe = pf_mmq_on() && !pf_pipe_on() &&
        !env_on("STRATA_DENSE_MMQ") && !env_on("STRATA_PF_GEMM") && !env_on("STRATA_WMMA_GEMM") &&
        e.gemm.f16_tuned(T, FS, N);
    if (pf_mmq_on() && !mmq_safe) {
        static bool warned = false;
        if (!warned) {
            std::fprintf(stderr, "strata-kolibri: MMQ falling back to the pool: requires compatible direct "
                                 "hipBLASLt shared-expert tuning, PF_PIPE=0 and no experimental dense overrides. "
                                 "See docs/AMD_R9700_PORT.md.\n");
            warned = true;
        }
    }
    cublasHandle_t blas = e.blas;
    const float one = 1.0f, zero = 0.0f;
    const Dense& emb = e.dense.at("token_embd.weight");
    const bool pph = attn_phases_on();   // S7: the prefill's per-layer phase marks (the same gate as the attn timers)
    std::chrono::steady_clock::time_point pp_t0{};
    auto pp_mark = [&](long long& acc) {
        if (!pph) return;
        cudaStreamSynchronize(g_st);
        const auto n = std::chrono::steady_clock::now();
        acc += std::chrono::duration_cast<std::chrono::nanoseconds>(n - pp_t0).count();
        pp_t0 = n;
    };
    // A2b: the per-chunk parameters, rebound per (chunk, layer) pair inside the diagonal.
    int Tc = T;
    int64_t p0c = p0;
    // **A2b: the diagonal.**  With one chunk (T2 == 0) this is exactly the old `for l` loop - same order, same values.  With two
    // chunks in flight it walks (c, L) along c + L == step, rebinding the three per-chunk values (Tc, p0c, A) for each pair,
    // so the GPU can run chunk c+1's PRE half while the host drains chunk c's pool.  The body below is untouched.
    if (pf_partial_on() && pf_pipe_on()) {
        // Review Focus 4: the two are declared incompatible rather than silently degraded.  PF_PIPE is the
        // two-chunks-in-flight mode; PF_PARTIAL keeps this layer's resident rows in one Scratch-wide buffer
        // (s.res_ey) and its partial ring per-ActSet, so a second in-flight chunk would read the first one's.
        std::fprintf(stderr, "kolibri: STRATA_KOLIBRI_PF_PARTIAL and STRATA_KOLIBRI_PF_PIPE cannot be combined "
                             "(the resident-row buffer is shared across chunks).  Unset one.\n");
        std::fflush(stderr);
        std::exit(1);
    }
    const int n_ch = (T2 > 0 && pf_pipe_on()) ? 2 : 1;
    for (int64_t step = 0; step < g.n_layers + n_ch - 1; ++step) {
    // **A2b: THE OVERLAP.**  With two chunks in flight each step runs TWO passes: every PRE (pure GPU enqueue) first,
    // then every POST (the host-blocking pool drains).  While the host blocks in chunk c's pool the GPU is executing
    // chunk c+1's PRE - the only overlap the dependency DAG permits.  No pair's own arithmetic changes: for a given
    // (c, L) the order is still PRE then POST, on the same buffers (act1/act2 by parity).
    const int n_pass = (n_ch == 2) ? 2 : 1;
    for (int pass = 0; pass < n_pass; ++pass)
    for (int ci = 0; ci < n_ch; ++ci) {
        const int64_t l = step - ci;
        if (l < 0 || l >= g.n_layers) continue;
        ActSet& A = (ci == 0) ? s.act1 : s.act2;
        // A2b: the pool path reads its INPUT buffers through these, so each in-flight chunk owns its own (see the note
        // at Scratch's h_host_b).  Chunk 0 uses the original set, so the single-chunk path is unchanged.
        s.h_host_cur = (ci == 0) ? s.h_host : s.h_host_b;
        s.act_cur    = (ci == 0) ? s.act    : s.act_b;
        s.ff_cur     = (ci == 0) ? s.ff     : s.ff_b;
        s.hq_cur     = (ci == 0) ? s.hq     : s.hq_b;
        Tc = (ci == 0) ? T : T2;
        p0c = (ci == 0) ? p0 : p02;
        const int NKc = Tc * K;   // A2b: NKc is per-chunk (it was T*K at function scope, stale for ci == 1)
        // A2b: the embedding is PRE work, and it must run on the PRE pass ONLY.  Sitting above the half-calls it ran on
        // BOTH passes, so on the POST pass it re-embedded the chunk over A.x_m and destroyed layer 0's attention residual
        // (x_m = embed + att_m) before the MoE sum added into it - silently, and only when there were two chunks, which is
        // why the one-chunk gate stayed green while the two-chunk KL ran to 0.72 (p06 3.16, p07 2.63) against a 0.0026 floor.
        if (l == 0 && pass == 0) {   // each chunk embeds its own tokens at its layer 0, with its own set
            switch (emb.type) {
            case T_Q8_0: k_row_multi<T_Q8_0><<<dim3(Tc, 4), 128>>>(emb.d, emb.row_bytes, A.toks_d, N, A.x_m); break;
            case T_Q4_K: k_row_multi<T_Q4_K><<<dim3(Tc, 4), 128>>>(emb.d, emb.row_bytes, A.toks_d, N, A.x_m); break;
            case T_Q6_K: k_row_multi<T_Q6_K><<<dim3(Tc, 4), 128>>>(emb.d, emb.row_bytes, A.toks_d, N, A.x_m); break;
            case T_F16: k_row_multi<T_F16><<<dim3(Tc, 4), 128>>>(emb.d, emb.row_bytes, A.toks_d, N, A.x_m); break;
            default: std::fprintf(stderr, "strata-kolibri: token_embd type %d\n", emb.type); std::exit(1);
            }
        }
        // A2b: Tc/p0c/NKc are rebound per (chunk, layer) pair ABOVE (just after `if (l < 0 ...) continue`).  They must NOT be
        // redeclared here: `int Tc = T; int64_t p0c = p0;` used to shadow the rebind, so every chunk-2 pair silently ran with
        // chunk 1's T and position p0 - which is what made the wavefront's ids differ while being perfectly deterministic.
        // **A2b: the body in two halves.**  layer_pre is pure GPU enqueue (norm -> attention -> router -> publish);
        // layer_post is the host-blocking half (expert leg, scatter/H2D, shared expert, MoE sum, residual).  Called
        // back to back this is the old order, and the gate (ids first) is what proves it.
        // A2b: hoisted OUT of layer_pre - layer_post needs both, and they are per-request/per-layer facts, not
        // per-half state.
        LayerBlobs& b = e.blobs[(size_t) l];   // non-const: the fp16 cache's LRU slot table is updated in the pool leg
        // POOL_ALL: the pool for EVERY layer (the 25 "native" ones too).  Measured on this box: neutral at
        // pp512 (the pool leg is small), -9% at 4K (the Q4_K grouped kernel's 8.6 s and the 98 GB of staging
        // both vanish) and -17% at 32K (255.7 s vs 308.4 s).  The env value CAPS the rule: 0 = off, 1 = on
        // always, 2 = on for a prefix >= 4K (the default: the win needs a long prefix).
        static const int pool_all_mode = [] {
            const char* v = std::getenv("STRATA_KOLIBRI_POOL_ALL");
            // DEFAULT 1 = the pool for every layer, at every length.  The old default (2: the pool only when
            // pf_total >= 1024) rested on the claim that "POOL_ALL is neutral at pp512".  Measured on this box, that
            // claim is false and expensive: a 512-token prefill runs 13,077/13,410 ms through the GPU grouped path and
            // 3,412/3,356 ms through the pool - a 3.9x regression on the short prompt, and exactly the leg the pp-short
            // floor is measured on.  The rule's own numbers (neutral at 512, -9% at 4K, -17% at 32K) all point the same
            // way, so there is no length at which the GPU grouped path is the better default here.
            if (!v || !v[0]) return 1;
            const int m = std::atoi(v);
            return m < 0 ? 0 : m;
        }();
        // The rule's threshold: the prompt's total length >= 1024 (POOL_ALL is neutral at pp512 - the pool leg
        // is small - and wins from ~4K up: -9% at 4K, -17% at 32K).
        const bool pool_all = pool_all_mode == 1 || (pool_all_mode >= 2 && e.pf_total >= 1024);
        auto layer_pre = [&] {
        if (pph) { cudaStreamSynchronize(g_st); pp_t0 = std::chrono::steady_clock::now(); }
        const bool sliding = strata::core::is_swa_layer(g, l);
        k_rms_rows<<<T, kThreads>>>(A.x_m, e.f32.at(L(l, "attn_norm.weight")), A.h_m, N, e.eps);
        strata::kernels::native_quantize_q8_1(A.h_m, A.xq8_m, N * Tc, 1, g_st);   // Tc contiguous rows = one column (q36)
        k_to_f16<<<grid1((int64_t) Tc * N), kThreads>>>((const float*) A.h_m, A.x16, (int64_t) Tc * N);
        {
            const Dense& wq = e.dense.at(L(l, "attn_q.weight"));
            if (dense_f16_on() && wq.f16) e.gemm.f16((const uint16_t*) A.x16, wq.f16, A.q_m, Tc, H * D, N);
            else e.gemm.native((const uint16_t*) A.x16, wq.type, wq.d, A.q_m, Tc, H * D, N);
            const Dense& wk = e.dense.at(L(l, "attn_k.weight"));
            if (dense_f16_on() && wk.f16) e.gemm.f16((const uint16_t*) A.x16, wk.f16, A.k_m, Tc, HKV * D, N);
            else e.gemm.native((const uint16_t*) A.x16, wk.type, wk.d, A.k_m, Tc, HKV * D, N);
            const Dense& wv = e.dense.at(L(l, "attn_v.weight"));
            if (dense_f16_on() && wv.f16) e.gemm.f16((const uint16_t*) A.x16, wv.f16, A.v_m, Tc, HKV * D, N);
            else e.gemm.native((const uint16_t*) A.x16, wv.type, wv.d, A.v_m, Tc, HKV * D, N);
        }
        pp_mark(e.cnt.pf_dense_ns);
        k_attn_prep_multi<<<dim3(H + HKV, Tc), kThreads>>>(A.q_m, A.k_m, A.v_m, e.f32.at(L(l, "attn_q_norm.weight")),
                                                          e.f32.at(L(l, "attn_k_norm.weight")), A.q_m, s.kc[l],
                                                          s.vc[l], H, HKV, D, p0c, g.n_swa, s.ctx, e.rope_base,
                                                          e.eps, sliding);
        // the chunk reads positions 0..p0c+Tc-1 (full) or the ring's last n_vis (sliding)
        const int P = sliding ? (int) std::min<int64_t>(p0c + Tc, g.n_swa) : (int) (p0c + Tc);
        const int64_t first = sliding ? p0c + Tc - P : 0;
        if (sliding)
            k_kv_ring<<<grid1((int64_t) HKV * P * D), kThreads>>>(s.kc[l], s.vc[l], A.k16, A.v16, g.n_swa, P,
                                                                  first, D, HKV);
        if (reuse_gate_kv_path() && (int) l == attn_dump_layer()) {
            // Per-chunk snapshot, so the chunk that perturbs a row is named rather than inferred.  Header magic
            // 0x4b564348 ("KVCH") distinguishes these from the decode-time records (0x4b564344, "KVCD").
            KC(cudaStreamSynchronize(g_st));
            const int64_t stride = sliding ? (int64_t) g.n_swa : (int64_t) s.ctx;
            const int64_t rows = std::min<int64_t>(p0c + Tc, stride);
            std::vector<__half> kh, vh;
            if (reuse_gate_copy_kv(s.kc[l], stride, rows, HKV, D, kh) &&
                reuse_gate_copy_kv(s.vc[l], stride, rows, HKV, D, vh)) {
                if (std::FILE* f = std::fopen(reuse_gate_kv_path(), "ab")) {
                    const int32_t hdr[4] = {(int32_t) p0c, (int32_t) Tc, (int32_t) stride, 0x4b564348};
                    std::fwrite(hdr, 4, 4, f);
                    std::fwrite(kh.data(), sizeof(__half), kh.size(), f);
                    std::fwrite(vh.data(), sizeof(__half), vh.size(), f);
                    std::fclose(f);
                }
            }
        }
        k_to_f16<<<grid1((int64_t) Tc * H * D), kThreads>>>(A.q_m, A.q16, (int64_t) Tc * H * D);
#if STRATA_KOLIBRI_MMA_AVAILABLE
        if (attn_mma_on() && H == 48 && HKV == 4 && D == 128) {
            k_attn_flash_mma<<<dim3((unsigned) ((Tc + 15) / 16), (unsigned) (HKV * 4)), 128, 0, g_st>>>(
                A.q16, sliding ? A.k16 : s.kc[l], sliding ? A.v16 : s.vc[l], A.att_m, (int) H,
                sliding ? (long long) P : (long long) s.ctx, (int) Tc, P, (int) p0c,
                sliding ? (int64_t) g.n_swa : 0, (int) first, 1.0f / sqrtf((float) D));
        } else
#endif
        if (attn_flash_on() && H == 48 && HKV == 4 && D == 128) {
            k_attn_flash<<<dim3((unsigned) ((Tc + 7) / 8), (unsigned) HKV), 256, 0, g_st>>>(
                A.q16, sliding ? A.k16 : s.kc[l], sliding ? A.v16 : s.vc[l], A.att_m, (int) H,
                sliding ? (long long) P : (long long) s.ctx, (int) Tc, P, (int) p0c,
                sliding ? (int64_t) g.n_swa : 0, (int) first, 1.0f / sqrtf((float) D));
        } else {
        const int ldp = (P + 7) / 8 * 8;
        const int G = H / HKV;
        const int64_t tq_lim = (int64_t) (attn_cap_floats() / ((size_t) H * 512)) * 512;   // nt * ldp <= tq_lim
        const int tq = (int) std::max<int64_t>(1, std::min<int64_t>(Tc, tq_lim / ldp));
        const bool attn_ph = attn_phases_on();
        std::chrono::steady_clock::time_point ph_last{};
        if (attn_ph) ph_last = std::chrono::steady_clock::now();
        auto ph_close = [&](long long& acc) {
            if (!attn_ph) return;
            cudaStreamSynchronize(g_st);
            const auto n = std::chrono::steady_clock::now();
            acc += std::chrono::duration_cast<std::chrono::nanoseconds>(n - ph_last).count();
            ph_last = n;
        };
        if (attn_kg_on() && (int64_t) Tc * ldp <= attn_cap_floats()) {
            const int Bh = (int) std::max<int64_t>(1, std::min<int64_t>(G, attn_cap_floats() / ((size_t) Tc * ldp)));
            for (int kvh = 0; kvh < HKV; ++kvh) {
                const __half* Kh = sliding ? A.k16 + (size_t) kvh * P * D
                                           : s.kc[l] + (size_t) kvh * s.ctx * D;
                const __half* Vh = sliding ? A.v16 + (size_t) kvh * P * D
                                           : s.vc[l] + (size_t) kvh * s.ctx * D;
                for (int hb = 0; hb < G; hb += Bh) {
                    const int nb = std::min(Bh, G - hb);
                    const __half* Qh = A.q16 + (size_t) (kvh * G + hb) * D;
                    // the cap-sized buffers hold Bh reusable slots (stream order: scores -> softmax -> PV
                    // per slot before the next (kvh, hb) reuses it), so base = the slot, NOT the global head
                    float* Sh = s.att_s + (size_t) (hb / Bh) * Tc * ldp;
                    if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, P, Tc, D, &one, Kh, CUDA_R_16F,
                                                   D, 0, Qh, CUDA_R_16F, H * D, D, &zero, Sh, CUDA_R_32F, ldp,
                                                   (long long) Tc * ldp, nb, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) !=
                        CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "strata-kolibri: scores GEMM failed\n"); std::exit(1); }
                    ph_close(e.cnt.attn_sc_ns);
                    k_attn_softmax16<<<dim3(Tc, nb), kThreads>>>(Sh, s.p16 + (size_t) (hb / Bh) * Tc * ldp,
                                                                ldp, Tc, P, p0c, 1.0f / sqrtf((float) D),
                                                                sliding ? g.n_swa : 0, (int) first, pv_scale());
                    ph_close(e.cnt.attn_sm_ns);
                    const __half* Ph = s.p16 + (size_t) (hb / Bh) * Tc * ldp;
                    float* Oh = A.att_m + (size_t) (kvh * G + hb) * D;
                    const float pv_alpha = 1.0f / pv_scale();   // undoes the fp16-mantissa-preserving scale above
                    if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, D, Tc, P, &pv_alpha, Vh, CUDA_R_16F, D, 0,
                                                   Ph, CUDA_R_16F, ldp, (long long) Tc * ldp, &zero, Oh, CUDA_R_32F,
                                                   H * D, D, nb, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) !=
                        CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "strata-kolibri: PV GEMM failed\n"); std::exit(1); }
                    ph_close(e.cnt.attn_pv_ns);
                }
            }
        } else for (int t0 = 0; t0 < Tc; t0 += tq) {
            const int nt = std::min(tq, Tc - t0);
            for (int kvh = 0; kvh < HKV; ++kvh) {
                const __half* Kh = sliding ? A.k16 + (size_t) kvh * P * D
                                               : s.kc[l] + (size_t) kvh * s.ctx * D;
                const __half* Qh = A.q16 + ((size_t) t0 * H + kvh * G) * D;
                float* Sh = s.att_s + (size_t) kvh * G * nt * ldp;
                if (attn_simt_on()) {
                    const dim3 sg((unsigned) ((P + 63) / 64), (unsigned) ((nt + 31) / 32), (unsigned) (G / 4));
                    k_attn_scores_simt<32, 64, 32, 4><<<sg, 256, 0, g_st>>>(A.q16, Kh, (int) H, (int) p0c, (int) t0,
                                                                           kvh * G, nt, P, (int) D, ldp,
                                                                           sliding ? (int64_t) g.n_swa : 0,
                                                                           (int) first, s.att_s);
                } else if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, P, nt, D, &one, Kh, CUDA_R_16F, D, 0,
                                               Qh, CUDA_R_16F, H * D, D, &zero, Sh, CUDA_R_32F, ldp,
                                               (long long) nt * ldp, G, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) !=
                    CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "strata-kolibri: scores GEMM failed\n"); std::exit(1); }
            }
            ph_close(e.cnt.attn_sc_ns);
            k_attn_softmax16<<<dim3(nt, H), kThreads>>>(s.att_s, s.p16, ldp, nt, P, (int) p0c + t0,
                                                        1.0f / sqrtf((float) D), sliding ? g.n_swa : 0, (int) first,
                                                        1.0f);   // the big-T branch is unscaled (its PV alpha stays 1)
            ph_close(e.cnt.attn_sm_ns);
            for (int kvh = 0; kvh < HKV; ++kvh) {
                const __half* Vh = sliding ? A.v16 + (size_t) kvh * P * D
                                               : s.vc[l] + (size_t) kvh * s.ctx * D;
                const __half* Ph = s.p16 + (size_t) kvh * G * nt * ldp;
                float* Oh = A.att_m + ((size_t) t0 * H + kvh * G) * D;
                if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, D, nt, P, &one, Vh, CUDA_R_16F, D, 0,
                                               Ph, CUDA_R_16F, ldp, (long long) nt * ldp, &zero, Oh, CUDA_R_32F,
                                               H * D, D, G, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) !=
                    CUBLAS_STATUS_SUCCESS) {
                    std::fprintf(stderr, "strata-kolibri: PV GEMM failed (t0=%d nt=%d P=%d ldp=%d ctx=%d)\n", t0, nt,
                                 P, ldp, s.ctx);
                    std::exit(1);
                }
            }
            ph_close(e.cnt.attn_pv_ns);
        }
        }   // attn_flash's else
        pp_mark(e.cnt.pf_attn_ns);
        k_to_f16<<<grid1((int64_t) Tc * H * D), kThreads>>>(A.att_m, s.att16, (int64_t) Tc * H * D);
        if (attn_dump_path() && (attn_dump_layer() < 0 || (int) l == attn_dump_layer())) {
            const size_t nf = (size_t) Tc * H * D;
            float* h = (float*) std::malloc(nf * sizeof(float));
            if (h) {
                KC(cudaMemcpy(h, A.att_m, nf * sizeof(float), cudaMemcpyDeviceToHost));
                std::FILE* f = std::fopen(attn_dump_path(), "ab");
                if (f) {
                    const int hdr[3] = {(int) l, (int) Tc, (int) 0x4b4d414d};   // layer, rows, magic
                    std::fwrite(hdr, sizeof(int), 3, f);
                    std::fwrite(h, sizeof(float), nf, f);
                    std::fclose(f);
                }
                std::free(h);
            }
        }
        {
            const Dense& wo = e.dense.at(L(l, "attn_output.weight"));
            if (dense_f16_on() && wo.f16) e.gemm.f16((const uint16_t*) s.att16, wo.f16, A.y_m, Tc, N, H * D);
            else e.gemm.native((const uint16_t*) s.att16, wo.type, wo.d, A.y_m, Tc, N, H * D);
        }
        k_rms_rows<<<Tc, kThreads>>>(A.y_m, e.f32.at(L(l, "post_attention_norm.weight")), A.y_m, N, e.eps);
        k_add_rows<<<grid1((int64_t) Tc * N), kThreads>>>(A.x_m, A.y_m, (int64_t) Tc * N);
        k_rms_rows<<<Tc, kThreads>>>(A.x_m, e.f32.at(L(l, "ffn_norm.weight")), A.h_m, N, e.eps);
        {
            const Dense& wr = e.dense.at(L(l, "ffn_gate_inp.weight"));
            if (env_on("STRATA_KOLIBRI_ROUTER_GEMM")) {
                // nsys priced the fp32 SIMT router gemv at 19 ms x 100 = 1.94 s of the pp512 wall.  cublas SGEMM
                // on the SAME fp32 inputs is ~100x faster, fp32 accumulate, fp32 logits.  DEFAULT OFF: measured
                // KL 0.02482 vs the 0.02 clause (cache-only baseline 0.01087) - cublas's accumulation order
                // differs from the SIMT gemv's at the last bits, and the router's top-k flips at near-ties.  An
                // f16 router is worse (KL 0.135).  The lever needs a SIMT kernel that keeps k_gemv's sum order.
                const float one = 1.0f, zero = 0.0f;
                cublasStatus_t cst = cublasSetStream(e.blas, g_st);
                if (cst == CUBLAS_STATUS_SUCCESS)
                    cst = cublasSgemm(e.blas, CUBLAS_OP_T, CUBLAS_OP_N, (int) g.n_expert, (int) Tc, (int) N, &one,
                                      (const float*) wr.d, (int) N, (const float*) A.h_m, (int) N, &zero,
                                      A.router_m, (int) g.n_expert);
                if (cst != CUBLAS_STATUS_SUCCESS) {
                    std::fprintf(stderr, "strata-kolibri: cublas router sgemm failed (%d)\n", (int) cst);
                    std::exit(1);
                }
            } else if (router_tk() > 1) {
                const int tk = router_tk(), tj = router_tj();
                if (tk >= 4)
                    k_router_tile_k<T_F32, 8, 4><<<dim3(((int) g.n_expert + 3) / 4, ((int) Tc + 7) / 8), kThreads, 0, g_st>>>(
                        wr.d, A.h_m, (size_t) N, A.router_m, (int) N, (int) Tc, (int) g.n_expert);
                else if (tk == 3) {
                    if (tj >= 32)
                        k_router_tile_k<T_F32, 32, 3><<<dim3(((int) g.n_expert + 2) / 3, ((int) Tc + 31) / 32), kThreads, 0, g_st>>>(
                            wr.d, A.h_m, (size_t) N, A.router_m, (int) N, (int) Tc, (int) g.n_expert);
                    else if (tj >= 16)
                        k_router_tile_k<T_F32, 16, 3><<<dim3(((int) g.n_expert + 2) / 3, ((int) Tc + 15) / 16), kThreads, 0, g_st>>>(
                            wr.d, A.h_m, (size_t) N, A.router_m, (int) N, (int) Tc, (int) g.n_expert);
                    else
                        k_router_tile_k<T_F32, 8, 3><<<dim3(((int) g.n_expert + 2) / 3, ((int) Tc + 7) / 8), kThreads, 0, g_st>>>(
                            wr.d, A.h_m, (size_t) N, A.router_m, (int) N, (int) Tc, (int) g.n_expert);
                }
                else if (tj >= 32)
                    k_router_tile_k<T_F32, 32, 2><<<dim3(((int) g.n_expert + 1) / 2, ((int) Tc + 31) / 32), kThreads, 0, g_st>>>(
                        wr.d, A.h_m, (size_t) N, A.router_m, (int) N, (int) Tc, (int) g.n_expert);
                else
                    k_router_tile_k<T_F32, 16, 2><<<dim3(((int) g.n_expert + 1) / 2, ((int) Tc + 15) / 16), kThreads, 0, g_st>>>(
                        wr.d, A.h_m, (size_t) N, A.router_m, (int) N, (int) Tc, (int) g.n_expert);
            } else if (env_on("STRATA_KOLIBRI_ROUTER_TILE")) {
                const int tj = router_tj();
                if (tj <= 8)
                    k_router_tile<T_F32, 8><<<dim3((int) g.n_expert, ((int) Tc + 7) / 8), kThreads, 0, g_st>>>(
                        wr.d, A.h_m, (size_t) N, A.router_m, (int) N, (int) Tc, (int) g.n_expert);
                else if (tj <= 16)
                    k_router_tile<T_F32, 16><<<dim3((int) g.n_expert, ((int) Tc + 15) / 16), kThreads, 0, g_st>>>(
                        wr.d, A.h_m, (size_t) N, A.router_m, (int) N, (int) Tc, (int) g.n_expert);
                else
                    k_router_tile<T_F32, 32><<<dim3((int) g.n_expert, ((int) Tc + 31) / 32), kThreads, 0, g_st>>>(
                    wr.d, A.h_m, (size_t) N, A.router_m, (int) N, (int) Tc, (int) g.n_expert);
            } else {
                k_gemv<T_F32><<<dim3(g.n_expert, Tc), kThreads>>>(wr.d, wr.row_bytes, 0, nullptr, N, A.h_m, (size_t) N,
                                                             A.router_m, (int) g.n_expert);
            }
            strata::kernels::native_router_sigmoid_logit_add(A.router_m, e.f32.at(L(l, "exp_probs_b.bias")),
                                                             A.ids_m, A.wts_m, Tc, g_st);
        }
        if (!e.touch_hist.empty() && (e.hot_on || e.hist_on || dec_res_on())) {   // coverage probe, the K29 seed's
            // counts, and the expert cache's own input: the prompt's routing is what the post-prompt round decides on
            std::vector<int> hid((size_t) NKc);
            KC(cudaMemcpy(hid.data(), A.ids_m, (size_t) NKc * sizeof(int), cudaMemcpyDeviceToHost));
            std::vector<uint64_t>& h = e.touch_hist[(size_t) l];
            for (int i = 0; i < NKc; ++i) ++h[(size_t) hid[i]];
        }
        strata::kernels::native_quantize_q8_1(A.h_m, A.xq8_m, N * Tc, 1, g_st);   // Tc contiguous rows = one column (q36)
        k_to_f16<<<grid1((int64_t) Tc * N), kThreads>>>((const float*) A.h_m, A.x16, (int64_t) Tc * N);   // ffn-norm h for the shared-FFN GEMMs (the router sgemm takes fp32 h_m directly)
        // The cache branch stays pool-side: routing the native_ok layers' grouped kernel through the fp16 cache
        // measured KL 0.13050 (gate FAIL) - that kernel is bitwise-equal to llama.cpp's arithmetic, and fp16
        // weights+activations deviate ~1e-3 per layer across 50 layers, which the KL-vs-llama clause forbids.
        // **A2b: PRE ENDS HERE.**  The routed ids are published once, for every branch - the pool consumes them, the
        // grouped kernel ignores them, the mmvq fallback reads them - so a wavefront has exactly one point at which a
        // chunk's PRE half is complete and its POST half may be scheduled against another chunk.  Two inner copies used
        // to do this per branch (:2321 and :2365); one copy of the same bytes earlier is byte-identical.
        // A2a: asynchronous, with the doorbell.  The synchronous version ran on the legacy default stream, which
        // implicitly drains g_st - including the other chunk's PRE - so POST's pool drain had no GPU work left to
        // overlap with (measured: 32K 220.9 s against a 117 s ceiling).
        KC(cudaMemcpyAsync(A.ids_h, A.ids_m, (size_t) NKc * sizeof(int), cudaMemcpyDeviceToHost, g_st));
        if (pf_partial_on())
            // G1: the host partial pass needs the router weights too.  Same stream as the ids, so the single
            // g_st drain before that pass orders both - no second synchronisation point.
            KC(cudaMemcpyAsync(A.wts_h, A.wts_m, (size_t) NKc * sizeof(float), cudaMemcpyDeviceToHost, g_st));
        KC(cudaEventRecord(A.ev, g_st));
        };
        auto layer_post = [&] {
        // A2a: wait for THIS chunk's publish (and everything PRE enqueued before it), not for the device.
        KC(cudaEventSynchronize(A.ev));
        // **0.1 PREFILL INSTRUMENT** (dynamic-expert-cache plan, step 0.1).  One stderr line per (chunk, layer) with
        // this chunk's routed picks and how many of them land on a resident expert according to `slot_of` - the SAME
        // question the decode counter answers, asked on the prefill side.  The step's premise was that this comes for
        // free from PFHASH; it does not, and the reason is measured, not guessed:
        //   * PFHASH's `res`/`miss` are the fp16 LRU cache's (`b.f16_slot_id`, STRATA_KOLIBRI_EXPERT_F16), a DIFFERENT
        //     cache from the expert-slot map - and it is off by default, so on the shipped config those lines do not
        //     exist at all;
        //   * PFHASH's `chunk=` is `e.pf_chunks`, which increments PER LAYER (inside the pool block, under
        //     --counters), so it is a (chunk,layer) counter and cannot be summed "across layers by chunk" as v2 said;
        //   * the prefill expert leg never consults `slot_of` (it is referenced only on the decode path), so a
        //     prefill hit rate has to be computed from the published ids, which is what this does.
        // Gated, so the default path prints nothing new.
        if (std::getenv("STRATA_KOLIBRI_SLOTTRACE") != nullptr) {
            static long long slot_tr_chunk = -1;
            if (l == 0) ++slot_tr_chunk;   // a new chunk starts at layer 0: the layer loop is the inner one
            const bool map_ok = (int) e.slot_of.size() == (int) g.n_expert;
            long long picks = 0, hits = 0;
            for (int i = 0; i < NKc; ++i) {
                ++picks;
                if (map_ok && e.slot_of[(size_t) A.ids_h[(size_t) i]] >= 0) ++hits;
            }
            std::fprintf(stderr, "SLOTTRACE chunk=%lld l=%d picks=%lld hits=%lld\n", slot_tr_chunk, (int) l, picks,
                         hits);
            std::fflush(stderr);
        }
        if (b.native_ok && !(pool_all && e.cpu_threads > 0)) {
            if (e.counters && e.slots > 0) {   // pricing only; --counters runs
                const int* hid = A.ids_h;   // already published - this used to be a SECOND synchronous D2H of the same ids
                std::vector<char> seen((size_t) g.n_expert, 0);
                int uniq = 0, hits = 0;
                for (int i = 0; i < NKc; ++i) { const int id = hid[(size_t) i]; if (!seen[(size_t) id]) { seen[(size_t) id] = 1; ++uniq; } if (id < e.slots) ++hits; }
                e.pf_hit_entries += hits; e.pf_entry_n += NKc; e.pf_uniq_sum += uniq; ++e.pf_cov_chunks;
            }
            static const bool pf_stage = env_on("STRATA_KOLIBRI_PF_STAGE");   // S7 A/B
            // NOTE (refuted by measurement): skipping the staging for the f16 layers (the staged table looked
            // dead - the pool serves the misses from the pinned table) made pp512 3.8x SLOWER (17.8 s vs 4.7 s):
            // the staged path is load-bearing (it carries the group table the native grouped kernel needs).
            if (pf_stage && s.stage != nullptr && NKc <= (int) s.stage_ids_h.size()) {
                int32_t* rank = s.stage_rank.data();
                std::fill(rank, rank + g.n_expert, -1);
                int nt = 0;
                for (int i = 0; i < NKc; ++i) {
                    const int id = A.ids_h[i];
                    if (rank[id] < 0) {
                        if (nt >= s.stage_n) { nt = -1; break; }   // touched set exceeds the stage: table path
                        rank[id] = nt;
                        KC(cudaMemcpyAsync(s.stage + (size_t) nt * b.stride, b.table + (size_t) id * b.stride,
                                           b.stride, cudaMemcpyHostToDevice, g_st));
                        ++nt;
                    }
                }
                if (nt > 0) {
                    for (int i = 0; i < NKc; ++i) s.stage_ids_h[(size_t) i] = rank[A.ids_h[i]];
                    KC(cudaMemcpyAsync(s.stage_ids_m, s.stage_ids_h.data(), (size_t) NKc * sizeof(int),
                                       cudaMemcpyHostToDevice, g_st));
                    k_group<<<1, 1024>>>(s.stage_ids_m, NKc, K, s.stage_n, s.stage, (unsigned long long) b.stride,
                                         A.grp_ptr_m, A.grp_start_m, A.n_groups_m, A.ent_dst_m, A.ent_tok_m);
                    // cap_groups MUST match the table path's (n_expert + 1), NOT (stage_n + 1).  The kernel strides
                    // its block rows over cap_groups, so a smaller cap changes the per-output reduction order and
                    // the staged path stopped being bitwise-equal (measured: max per-step KL 0.157 vs unstaged).
                    // The stage's group count (<= stage_n) is under the cap, so the larger cap is legal.
                    strata::kernels::native_expert_grouped(b.L, A.grp_ptr_m, A.grp_start_m, A.n_groups_m, A.ent_dst_m,
                                                           A.ent_tok_m, std::min<int64_t>(NKc, g.n_expert) + 1, NKc,
                                                           A.xq8_m, s.scratch_m, A.gout_m, g_st);
                    if (e.counters) { ++e.pf_stage_chunks; e.pf_stage_bytes += (long long) nt * b.stride; }
                } else {
                    k_group<<<1, 1024>>>(A.ids_m, NKc, K, (int) g.n_expert, b.table, (unsigned long long) b.stride,
                                         A.grp_ptr_m, A.grp_start_m, A.n_groups_m, A.ent_dst_m, A.ent_tok_m);
                    strata::kernels::native_expert_grouped(b.L, A.grp_ptr_m, A.grp_start_m, A.n_groups_m, A.ent_dst_m,
                                                           A.ent_tok_m, std::min<int64_t>(NKc, g.n_expert) + 1, NKc,
                                                           A.xq8_m, s.scratch_m, A.gout_m, g_st);
                }
            } else {
            k_group<<<1, 1024>>>(A.ids_m, NKc, K, (int) g.n_expert, b.table, (unsigned long long) b.stride,
                                 A.grp_ptr_m, A.grp_start_m, A.n_groups_m, A.ent_dst_m, A.ent_tok_m);
            strata::kernels::native_expert_grouped(b.L, A.grp_ptr_m, A.grp_start_m, A.n_groups_m, A.ent_dst_m,
                                                   A.ent_tok_m, std::min<int64_t>(NKc, g.n_expert) + 1, NKc, A.xq8_m,
                                                   s.scratch_m, A.gout_m, g_st);
            }
        } else {
            if (e.cpu_threads > 0) {   // pool_all reaches here for the native_ok layers too (the ids are copied above)   // the CPU pool: ggml's own expert arithmetic over the pinned blobs, the
                                       // chunk's NKc entries split over the pool's threads (bandwidth-bound,
                                       // ~10x the launch-bound mmvq loop on this box)
                // A2a (second half): the pool's INPUT copy, made asynchronous for the same reason the ids publish was.
                // The blocking form ran on the legacy default stream, which implicitly drained g_st - including the OTHER
                // chunk's already-enqueued PRE work - so POST's drain always began with an empty device queue and the
                // two-chunk wavefront bought nothing (measured: GPU 23% busy, no two kernels overlapping, 8K wall 53.7 s
                // against a max(pool, GPU) ceiling of 32.5 s).  h_host is pinned, so this copy is genuinely asynchronous;
                // A.ev is re-recorded here, AFTER the ids copy, so layer_post's single existing wait covers both copies
                // and nothing else - the same doorbell shape, no new event and no changed wait semantics.
                KC(cudaMemcpyAsync(s.h_host_cur, A.h_m, (size_t) Tc * N * 4, cudaMemcpyDeviceToHost, g_st));
                KC(cudaEventRecord(A.ev, g_st));
                // THE WAIT BELONGS HERE, not in layer_post: the quantization loop immediately below consumes h_host_cur, and
                // an asynchronous copy hands it nothing.  Scoping the wait to THIS chunk's event is the whole point - it does
                // not drain the device, so the other chunk's already-enqueued PRE work keeps running through it, which is
                // exactly what the blocking (legacy-stream) form destroyed.
                KC(cudaEventSynchronize(A.ev));
                for (int t = 0; t < Tc; ++t)
                    strata::kernels::cpu::native_quant_act(b.fmt, s.h_host_cur + (size_t) t * N,
                                                           s.act_cur + (size_t) t * b.fmt.act_bytes);
                // group the chunk's entries by expert: one blob read per expert (the box's RAM bandwidth is the
                // floor; per-entry reads re-stream 2.2 MB per row), all of an expert's rows in one call
                std::vector<int32_t>& cnt = s.exp_cnt;
                std::fill(cnt.begin(), cnt.begin() + g.n_expert + 1, 0);
                for (int sel = 0; sel < NKc; ++sel) cnt[A.ids_h[sel] + 1]++;
                for (int x = 0; x < (int) g.n_expert; ++x) cnt[x + 1] += cnt[x];
                std::vector<int32_t>& ord = s.exp_ord;
                {
                    std::vector<int32_t> cur(cnt.begin(), cnt.begin() + g.n_expert + 1);
                    for (int sel = 0; sel < NKc; ++sel) ord[cur[A.ids_h[sel]]++] = sel;
                }
                // The shard count must equal the pool's LANE count, not --cpu-threads: the fan-out maps shard i
                // to lane i (shard 0 = the host), so ns > lanes leaves some lanes with two shards and others
                // with one - equal bytes per SHARD, 2x unequal time per lane, and the barrier waits for the
                // double-shard lane.  That is the measured 2.0x spread on the 32K prefill.
                const int NS = std::min((int) g.n_expert, e.pf_lanes > 0 ? e.pf_lanes : e.cpu_threads);
                // Work-balanced shard deal (LPT).  The old split strode over ALL 384 expert ids, so a worker that
                // drew the hot experts set the barrier: the phase timers measured 88.2 s of kernel work in a 22.9 s
                // window on 24 workers = 16% parallel efficiency.  Deal the TOUCHED experts, biggest first, round
                // robin.  Each expert's rows stay in ONE native_gu_rows/native_down_rows call, so the arithmetic is
                // untouched and the result is bitwise-identical.
                s.pf_touch.clear();
                for (int x = 0; x < (int) g.n_expert; ++x) if (cnt[x + 1] > cnt[x]) s.pf_touch.push_back(x);
                // A/B gate for the bitwise verification: STRATA_KOLIBRI_LPT=0 keeps the touched list in ascending
                // expert order (the pre-LPT deal shape) so the two deal orders can be compared on identical code.
                static const bool pf_lpt = std::getenv("STRATA_KOLIBRI_LPT") == nullptr ||
                                           std::string(std::getenv("STRATA_KOLIBRI_LPT")) != "0";
                if (pf_lpt)
                    std::sort(s.pf_touch.begin(), s.pf_touch.end(),
                              [&](int a, int b) { return (cnt[a + 1] - cnt[a]) > (cnt[b + 1] - cnt[b]); });
                // S7 increment B: cache-resident experts run their GEMMs on the GPU from the fp16 cache - no
                // PCIe blob read, no per-call dequant, no pool worker.  The coverage probe showed the hot experts
                // are scattered across the id space (top-40 by frequency = 82.8% of touches), so residency is
                // LRU-promoted, not id-ordered: each touched expert is looked up in the layer's slots, a miss is
                // promoted into a free slot (evicting the least-recently-touched when full), and pf_touch's LPT
                // order (biggest-first) promotes the chunk's biggest experts first.  Launches go on g_st and
                // overlap the pool's CPU work below; the existing g_st drain before the ey_h H2D orders them.
                // Numerics: fp16 weights + fp16 activations, fp32 accumulate (the dense-fp16 template, KL 0.00210);
                // the pool path keeps its q8-activation arithmetic.
                // P1 BUG (fixed): `pf_res_slot` was never cleared.  pf_res was, so from the SECOND chunk on every
                // `pf_res[ri]` was paired with a slot index left over from an earlier chunk or layer, and the block loop
                // below (which reads pf_res_slot[ri]) built the resident GEMMs from whatever expert happened to sit in
                // that stale slot - wrong weights, and a result that depended on cache state and timing.  The probe
                // showed it directly: duplicate slot indices inside one chunk grew monotonically 120 -> 168 -> 224.
                s.pf_res.clear(); s.pf_miss.clear(); s.pf_res_slot.clear();
                int64_t res_rows = 0;
                ++s.pf_clock;
                const int64_t f16_stride = 3 * (int64_t) FF * N;
                // Promote only the chunk's top-P experts by row count (pf_touch is LPT-sorted, so the first P
                // cover ~85% of the rows): the promotion stream is PCIe-bound (2.21 MB/expert), and the small
                // experts are cheap pool-side.  P = STRATA_KOLIBRI_RES_TOP (default 48).
                static const int res_top = std::getenv("STRATA_KOLIBRI_RES_TOP") ?
                                               std::atoi(std::getenv("STRATA_KOLIBRI_RES_TOP")) : 48;
                // **P1 BUG (fixed): RESIDENCY MUST NOT EXCEED THE SLOT COUNT.**  res_top was applied as-is while the
                // layer's cache has only `b.n_cached` slots (24), so a chunk with res_top > n_cached assigned several
                // experts to the SAME slot: the later promotion's dequant overwrote the earlier expert's weights, the
                // earlier expert's blocks still pointed at that slot (the block loop stores `xe = b.f16 + slot*stride`),
                // and the batched GEMM multiplied activations by the WRONG expert's weights.  The probe measured it
                // directly: duplicate slot indices inside one chunk = 300 at res_top=48 (48 residents, 24 slots).
                // Capping at the slot count makes each resident own its slot, which is also the only reading under
                // which "resident" means anything.
                const int res_eff = b.f16 ? std::min(res_top, b.n_cached) : 0;
                if (b.f16) {
                    int pi = 0;
                    for (int x : s.pf_touch) {
                        if (++pi > res_eff) { s.pf_miss.push_back(x); continue; }
                        int slot = -1;
                        for (int s_i = 0; s_i < b.n_cached; ++s_i) if (b.f16_slot_id[s_i] == x) { slot = s_i; break; }
                        if (slot < 0) {   // promote: free slot, else evict least-recently-touched
                            for (int s_i = 0; s_i < b.n_cached; ++s_i) if (b.f16_slot_id[s_i] < 0) { slot = s_i; break; }
                            if (slot < 0) {
                                slot = 0;
                                for (int s_i = 1; s_i < b.n_cached; ++s_i) if (b.f16_slot_last[s_i] < b.f16_slot_last[slot]) slot = s_i;
                            }
                            const uint8_t* blob = b.table + (size_t) x * b.stride;
                            uint16_t* out = b.f16 + (size_t) slot * f16_stride;
                            strata::kernels::dequant_f16(b.L.gu_type, blob, 0, FF, N, out, g_st);
                            strata::kernels::dequant_f16(b.L.gu_type, blob + b.L.up_off, 0, FF, N, out + (int64_t) FF * N, g_st);
                            strata::kernels::dequant_f16(b.L.d_type, blob + b.L.down_off, 0, N, FF, out + 2 * (int64_t) FF * N, g_st);
                            b.f16_slot_id[slot] = x;
                            if (e.counters) {
                                ++e.pf_f16_promotes;
                                // three FP16 planes: gate FF x N, up FF x N, down N x FF
                                e.pf_fill_bytes += 3LL * FF * N * 2;
                            }
                        }
                        b.f16_slot_last[slot] = s.pf_clock;
                        s.pf_res.push_back(x);
                        s.pf_res_slot.push_back(slot);
                    }
                } else {
                    for (int x : s.pf_touch) s.pf_miss.push_back(x);
                }
                // S7 increment C: resident GEMMs as ONE batched cublas call per slice per layer.  Each touched
                // expert's tokens are cut into blocks of kResBlk rows (last block zero-padded via tok = -1); the
                // gather ships the activation rows block-contiguous, gate/up -> SwiGLU -> to-f16 -> down run as
                // three f16_batched calls, and the D2H lands the whole block array in res_ey_h for the scatter
                // below.  Launches per layer: ~15, replacing ~1085 (11.6k/chunk = ~0.9 s of launch overhead).
                if (b.f16) {
                    constexpr int M = Scratch::kResBlk;
                    int nblk = 0;
                    s.blk_expert.clear(); s.blk_lo.clear(); s.blk_m.clear();
                    for (size_t ri = 0; ri < s.pf_res.size(); ++ri) {
                        const int x = s.pf_res[ri], slot = s.pf_res_slot[ri];
                        const int lo = cnt[x], hi = cnt[x + 1], m = hi - lo;
                        const uint16_t* xe = b.f16 + (size_t) slot * f16_stride;
                        for (int off = 0; off < m; off += M) {
                            const int mb = std::min(M, m - off);
                            for (int a = 0; a < mb; ++a) {
                                s.res_tok_h[(size_t) nblk * M + a] = ord[(size_t) lo + off + a] / K;
                                s.rr_sel_h[(size_t) nblk * M + a] = ord[(size_t) lo + off + a];   // G1: full selection index
                            }
                            for (int a = mb; a < M; ++a) {
                                s.res_tok_h[(size_t) nblk * M + a] = -1;
                                s.rr_sel_h[(size_t) nblk * M + a] = -1;                            // G1: padding
                            }
                            s.pa_g_h[nblk] = xe;
                            s.pa_u_h[nblk] = xe + (int64_t) FF * N;
                            s.pa_d_h[nblk] = xe + 2 * (int64_t) FF * N;
                            s.pb_x_h[nblk] = s.res_x16 + (int64_t) nblk * M * N;
                            s.pb_h_h[nblk] = s.res_h16 + (int64_t) nblk * M * FF;
                            s.pc_g_h[nblk] = s.res_g + (int64_t) nblk * M * FF;
                            s.pc_u_h[nblk] = s.res_u + (int64_t) nblk * M * FF;
                            s.pc_ey_h[nblk] = s.res_ey + (int64_t) nblk * M * N;
                            s.blk_expert.push_back(x); s.blk_lo.push_back(lo + off); s.blk_m.push_back(mb);
                            ++nblk;
                        }
                    }
                    s.pf_nblk = nblk;
                    res_rows = (int64_t) nblk * M;
                    // RES_TOP=0 is a legal configuration: the promotion ceiling is zero, pf_res stays empty and
                    // every touched expert goes to the pool.  Nothing below may launch in that case.  A zero-block
                    // <<<>>> is "invalid configuration argument" on this stack (measured on the R9700, ROCm
                    // 7.2.2 / gfx1201: grid 0 -> error 9, one-block launch -> 0), and the launch error is STICKY:
                    // it does not report at the launch, it reports at whatever check runs next.  On the pre-guard
                    // binary that check was the next layer's dequant, which is why the ledger filed this as a
                    // dequant defect (docs/AMD_R9700_PORT.md §10).  Reached with nblk = 0 once per layer per
                    // chunk (100 times in a 512-token prefill: 2 chunks x 50 layers, measured), so the guard is
                    // load-bearing, not defensive dressing.
                    if (nblk > 0) {
                        const int64_t rows = res_rows;
                        KC(cudaMemcpyAsync(s.res_tok, s.res_tok_h.data(), (size_t) rows * 4, cudaMemcpyHostToDevice, g_st));
                        KC(cudaMemcpyAsync(s.pa_g, s.pa_g_h.data(), (size_t) nblk * 8, cudaMemcpyHostToDevice, g_st));
                        KC(cudaMemcpyAsync(s.pa_u, s.pa_u_h.data(), (size_t) nblk * 8, cudaMemcpyHostToDevice, g_st));
                        KC(cudaMemcpyAsync(s.pa_d, s.pa_d_h.data(), (size_t) nblk * 8, cudaMemcpyHostToDevice, g_st));
                        KC(cudaMemcpyAsync(s.pb_x, s.pb_x_h.data(), (size_t) nblk * 8, cudaMemcpyHostToDevice, g_st));
                        KC(cudaMemcpyAsync(s.pb_h, s.pb_h_h.data(), (size_t) nblk * 8, cudaMemcpyHostToDevice, g_st));
                        KC(cudaMemcpyAsync(s.pc_g, s.pc_g_h.data(), (size_t) nblk * 8, cudaMemcpyHostToDevice, g_st));
                        KC(cudaMemcpyAsync(s.pc_u, s.pc_u_h.data(), (size_t) nblk * 8, cudaMemcpyHostToDevice, g_st));
                        KC(cudaMemcpyAsync(s.pc_ey, s.pc_ey_h.data(), (size_t) nblk * 8, cudaMemcpyHostToDevice, g_st));
                        k_gather_f16<<<grid1(rows * N), kThreads>>>((const __half*) A.x16, s.res_tok, (__half*) s.res_x16, N);
                        e.gemm.f16_batched(s.pa_g, s.pb_x, s.pc_g, M, FF, N, nblk);
                        e.gemm.f16_batched(s.pa_u, s.pb_x, s.pc_u, M, FF, N, nblk);
                        k_swiglu_rows<<<grid1(rows * FF), kThreads>>>(s.res_g, s.res_u, s.res_h, rows * FF);
                        k_to_f16<<<grid1(rows * FF), kThreads>>>(s.res_h, (__half*) s.res_h16, rows * FF);
                        e.gemm.f16_batched(s.pa_d, s.pb_h, s.pc_ey, M, N, FF, nblk);
                        if (pf_partial_on()) {
                            // G1: the resident rows STAY in s.res_ey.  Today this D2H lands rows*N floats (~27 MB at
                            // ~85% coverage) that the host scatters and then ships straight back up as part of the
                            // 31.5 MB block; here only the 64 KB selection map goes down, and the rows never leave.
                            KC(cudaMemcpyAsync(A.rr_sel_m, s.rr_sel_h.data(), (size_t) rows * 4, cudaMemcpyHostToDevice, g_st));
                            // G1 CNT coverage: the prefill staging legs were invisible to the byte counters, so the
                            // "CNT bytes reconcile with the wall delta" gate could not be run on them.  Counted here,
                            // under the same e.counters guard as the decode legs, so a rate run stays untouched.
                            if (e.counters) { ++e.cnt.h2d_n; e.cnt.h2d_bytes += (long long) rows * 4; }
                        } else {
                            KC(cudaMemcpyAsync(s.res_ey_h, s.res_ey, (size_t) rows * N * 4, cudaMemcpyDeviceToHost, g_st));
                            if (e.counters) { ++e.cnt.d2h_n; e.cnt.d2h_bytes += (long long) rows * N * 4; }
                        }
                    }
                }
                const std::vector<int32_t>& pool_list = b.f16 ? s.pf_miss : s.pf_touch;
                const auto pf_t0 = std::chrono::steady_clock::now();
                s.mmq_gout = false;
                pp_mark(e.cnt.pf_exp_ns);   // S7: the expert leg (the GPU grouped GEMMs + the CPU pool)
                // P4: the chunk's experts through MMQ, PER GROUP, exactly as src/prefill/prefill.cpp:4936 does it -
                // its own quantize of its own rows, group-relative bounds, the identity id table, a zeroed MMQ tail,
                // and a group-sized destination that this caller scatters.  Every guard falls back to the pool.
                bool mmq_did = false;
#if defined(STRATA_KOLIBRI_MMQ)
                if (mmq_safe && !b.f16 && e.mmq_ctx != nullptr && !pool_list.empty() &&
                    strata::prefill::mmq::supported(b.L.gu_type) && strata::prefill::mmq::supported(b.L.d_type)) {
                    std::vector<int> ml;
                    int64_t rows = 0;
                    for (int x : pool_list) {
                        const int m = cnt[x + 1] - cnt[x];
                        if (m <= 0) continue;
                        ml.push_back(x);
                        rows += m;
                    }
                    const size_t gu_half = (size_t) b.L.up_off;      // one matrix: gate occupies [0, up_off)
                    const size_t gu_stride = (size_t) b.L.down_off;   // gate|up contiguous
                    const size_t d_region = b.stride > (size_t) b.L.down_off ? b.stride - (size_t) b.L.down_off : 0;
                    // Product derives its per-expert stride itself as expert_bytes / type_bytes(type) (integer
                    // division, src/prefill/moe_mmq.cu:478), so the stride MUST be a whole number of weight blocks.
                    // The blob's down region is not: 1,095,840 B / 210 B (Q6_K) = 5,218.28, truncated to 5,218 blocks,
                    // which shifts expert z's down matrix by 60*z bytes - expert 0 exact (so a one-expert group is
                    // perfect), expert 15 off by 900 B.  Stage and address the matrix itself instead.
                    const size_t d_sz = strata::prefill::mmq::matrix_bytes(b.L.d_type, N, FF);
                    if (!ml.empty() && rows > 0 && rows <= s.mmq_rows_cap && gu_half > 0 && d_sz > 0 &&
                        d_region >= d_sz) {
                        // MMQ's scheduler degenerates when a group's max rows per expert is tiny (measured: groups with
                        // maxr=1 wrote NaN into their rows - docs/AMD_R9700_PORT.md P4).  Pad every expert to kMinRows:
                        // the filler rows repeat that expert's first activation row, and the scatter skips them (sel<0).
                        // STRATA_KOLIBRI_MMQ_GROUP=1 isolates the multi-expert staging layout: with one expert a group the
                        // product needs no expert stride at all.
static const int grp_max = [] {
                            const char* v = std::getenv("STRATA_KOLIBRI_MMQ_GROUP");
                            int n = v ? std::atoi(v) : 16;
                            if (n < 1) n = 1;
                            if (n > strata::prefill::mmq::kGatherGroupMax) n = strata::prefill::mmq::kGatherGroupMax;
                            return n;
                        }();
                        bool ok = true;
                        const int bstride = strata::prefill::mmq::kGatherGroupMax + 1;
                        size_t ngroups = 0;
                        for (size_t j0p = 0; j0p < ml.size(); j0p += (size_t) grp_max) ++ngroups;
                        {
                            size_t gi = 0;
                            for (size_t j0p = 0; j0p < ml.size(); j0p += (size_t) grp_max, ++gi) {
                                const int gnp = (int) std::min<size_t>((size_t) grp_max, ml.size() - j0p);
                                int64_t acc = 0;
                                s.mmq_bounds_h[(size_t) gi * (size_t) bstride] = 0;
                                for (int q = 0; q < gnp; ++q) {
                                    const int x = ml[j0p + (size_t) q];
                                    const int m = cnt[x + 1] - cnt[x];
                                    acc += m < 8 ? 8 : m;
                                    s.mmq_bounds_h[(size_t) gi * (size_t) bstride + (size_t) q + 1] = (int32_t) acc;
                                }
                            }
                        }
                        constexpr int kMinRows = 8;
                        int64_t padded = 0;
                        for (int x : ml) {
                            const int m2 = cnt[x + 1] - cnt[x];
                            padded += m2 < kMinRows ? kMinRows : m2;
                        }
                        // Test the size BEFORE writing: a chunk whose experts mostly hold one or two rows needs up to
                        // 8 extra rows each, and writing past the table corrupted the heap at 8 191 tokens.  The table is
                        // now reserved for the padding (rcap) and anything larger still falls back to the pool.
                        if (padded > s.mmq_rows_cap) {
                            ok = false;
                        } else {
                            int64_t r = 0;
                            for (int x : ml) {
                                const int i0 = cnt[x];
                                int m = cnt[x + 1] - i0;
                                const int pr = m < kMinRows ? kMinRows : m;
                                for (int q = 0; q < pr; ++q, ++r) {
                                    s.mmq_row_tok_h[(size_t) r] = ord[(size_t) (i0 + (q < m ? q : 0))] / K;
                                    s.mmq_row_sel_h[(size_t) r] = q < m ? (int32_t) ord[(size_t) (i0 + q)] : -1;
                                }
                            }
                            rows = r;   // the padded row count is what the groups below walk
                        }
                        KC(cudaMemcpyAsync(s.mmq_row_tok, s.mmq_row_tok_h.data(), (size_t) rows * 4, cudaMemcpyHostToDevice, g_st));
                        KC(cudaMemcpyAsync(s.mmq_row_sel, s.mmq_row_sel_h.data(), (size_t) rows * 4, cudaMemcpyHostToDevice, g_st));
                        KC(cudaMemcpyAsync(s.mmq_bounds, s.mmq_bounds_h.data(),
                                           (size_t) ngroups * (size_t) bstride * 4, cudaMemcpyHostToDevice, g_st));
                        if (std::getenv("STRATA_KOLIBRI_MMQ_DEBUG") != nullptr) {
                            // coverage probe: stamp gout with NaN; anything still NaN after the scatter was never written
                            std::vector<float> ns((size_t) NKc * N, std::nanf(""));
                            KC(cudaMemcpy(A.gout_m, ns.data(), ns.size() * 4, cudaMemcpyHostToDevice));
                        }
                        if (std::getenv("STRATA_KOLIBRI_MMQ_HOSTDBG") != nullptr) {
                            // Host-side only (no device reads, so nothing is serialized): do two runs hand the product
                            // the same numbers?  If yes, the run-to-run difference is inside the product.
                            int64_t nrsum = 0, maxsum = 0;
                            for (size_t j0p = 0; j0p < ml.size(); j0p += (size_t) grp_max) {
                                const int gnp = (int) std::min<size_t>((size_t) grp_max, ml.size() - j0p);
                                int64_t nrg = 0;
                                int mxg = 0;
                                for (int q = 0; q < gnp; ++q) {
                                    const int x = ml[j0p + (size_t) q];
                                    const int m = cnt[x + 1] - cnt[x];
                                    const int pr = m < 8 ? 8 : m;
                                    nrg += pr;
                                    mxg = std::max(mxg, pr);
                                }
                                nrsum += nrg;
                                maxsum += mxg;
                            }
                            std::fprintf(stderr, "MMQH l=%d experts=%zu rows=%lld ngroups=%zu nrsum=%lld maxsum=%lld sel0=%d sel1=%d\n",
                                         (int) l, ml.size(), (long long) rows, ngroups, (long long) nrsum,
                                         (long long) maxsum, (int) ord[0], ml.empty() ? -1 : ml[0]);
                        }
                        int64_t r0 = 0;
                        // STRATA_KOLIBRI_MMQ_GROUP=1 isolates the multi-expert staging layout: with one expert a group the
                        // product needs no expert stride at all (measured defect: experts 1..n of a 16-expert group came
                        // back NaN, ledger §P4).
                        
                        for (size_t j0 = 0; ok && j0 < ml.size(); j0 += (size_t) grp_max) {
                            const int gn = (int) std::min<size_t>((size_t) grp_max, ml.size() - j0);
                            const size_t gi = j0 / (size_t) grp_max;
                            const int32_t* brel = s.mmq_bounds + gi * (size_t) bstride;   // this group's slice, uploaded once
                            int64_t nr = 0;
                            int maxr = 0;
                            for (int q = 0; q < gn; ++q) {
                                const int x = ml[j0 + (size_t) q];
                                const int m = cnt[x + 1] - cnt[x];
                                const int pr = m < 8 ? 8 : m;   // the same padding the row walk above used
                                nr += pr;
                                maxr = std::max(maxr, pr);
                            }
                            KC(cudaMemsetAsync(s.mmq_gu_stage, 0, (size_t) strata::prefill::mmq::kGatherGroupMax * gu_stride + 4096, g_st));
                            KC(cudaMemsetAsync(s.mmq_d_stage, 0, (size_t) strata::prefill::mmq::kGatherGroupMax * d_sz + 4096, g_st));
                            strata::prefill::mmq::GatherGroup grp;
                            grp.first = 0;   // a fresh group: its experts land at gu_dst + q * gu_stride
                            grp.n = gn;
                            // POSITIONAL TEST: every slot gets the SAME expert, so every slot's rows must produce
                            // identical values.  Anything else is the staging or the product, not the data.
                            static const bool same_exp = std::getenv("STRATA_KOLIBRI_MMQ_SAME") != nullptr;
                            for (int q = 0; q < gn; ++q)
                                grp.blob[q] = b.table + (size_t) ml[j0 + (same_exp ? 0 : (size_t) q)] * b.stride;
                            ok = strata::prefill::mmq::gather_native_group(grp, (size_t) b.L.up_off, gu_half,
                                                                          (size_t) b.L.down_off, d_sz, s.mmq_gu_stage,
                                                                          gu_stride, s.mmq_d_stage, d_sz, g_st);
                            if (!ok) break;
                            // the MMQ tile reads a tail past the group's last expert (prefill.cpp: MMQ_TAIL = 4096)
                            KC(cudaMemsetAsync((char*) s.mmq_gu_stage + (size_t) gn * gu_stride, 0, 4096, g_st));
                            KC(cudaMemsetAsync((char*) s.mmq_d_stage + (size_t) gn * d_sz, 0, 4096, g_st));
                            // A.h_m, not A.x_m: the experts' input is the post-attention NORMED hidden - the pool's leg
                            // copies exactly this buffer to s.h_host_cur and quantizes it per token (:2877-2886).  Feeding
                            // the residual stream here gave values ~100x small, a router that collapsed to one expert, NaN.
                            strata::prefill::mmq::quantize(A.h_m, s.mmq_row_tok + r0, s.mmq_xq, b.L.gu_type, N, N, nr, g_st);
                            strata::prefill::mmq::Product gp{};
                            gp.type = b.L.gu_type;
                            gp.n = gn;
                            gp.xq = s.mmq_xq;
                            gp.bounds = brel;
                            gp.ids = s.mmq_ident;
                            gp.total_rows = nr;
                            gp.max_rows = maxr;
                            gp.w = s.mmq_gu_stage;
                            gp.w_rows = 2 * FF;
                            gp.w_cols = N;
                            gp.expert_bytes = gu_stride;
                            gp.dst = s.mmq_gu;
                            gp.ld_dst = 2 * FF;
                            // The stream-k path on this card writes its partial tiles through a fixup that ACCUMULATES
                            // (dst[...] += sum, mmq.cuh:1367), so the destination must start at zero: without this the
                            // group's result depended on what the buffer held from earlier work and the generated ids
                            // changed from run to run (measured: same binary, same prompt, three runs, two answers).
                            KC(cudaMemsetAsync(s.mmq_gu, 0, (size_t) nr * 2 * FF * 4, g_st));
                            e.mmq_ctx->run(gp, g_st);
                            strata::prefill::mmq::swiglu(s.mmq_gu, s.mmq_h, nr, FF, false, g_st);
                                                                                                                strata::prefill::mmq::quantize(s.mmq_h, nullptr, s.mmq_xqh, b.L.d_type, FF, FF, nr, g_st);
                            if (std::getenv("STRATA_KOLIBRI_MMQ_DEBUG") != nullptr) {
                                // stamp dm so a product row that is never written is distinguishable from a NaN result
                                std::vector<float> ns((size_t) nr * N, std::nanf(""));
                                KC(cudaMemcpy(s.mmq_dm, ns.data(), ns.size() * 4, cudaMemcpyHostToDevice));
                            }
                            strata::prefill::mmq::Product dp = gp;
                            dp.type = b.L.d_type;
                            dp.xq = s.mmq_xqh;
                            dp.w = s.mmq_d_stage;
                            dp.w_rows = N;
                            dp.w_cols = FF;
                            dp.expert_bytes = d_sz;
                            dp.dst = s.mmq_dm;
                            dp.ld_dst = N;
                            KC(cudaMemsetAsync(s.mmq_dm, 0, (size_t) nr * N * 4, g_st));
                            e.mmq_ctx->run(dp, g_st);
                            // Into the DEVICE buffer the MoE combine reads (k_moe_sum_rows), not the pool's pinned host
                            // staging: the main engine never writes a product's output to host memory (prefill.cpp writes
                            // P.sum on device and copies back), and a kernel writing A.ey_h landed nowhere here - the H2D
                            // then shipped zeros, which is the `gen: 0 0 0 0`.  This also removes 31.5 MB of PCIe per
                            // layer-chunk that only existed because the pool is on the CPU.
                            if (std::getenv("STRATA_KOLIBRI_MMQ_DEBUG") != nullptr && l == 0 && j0 == 0) {
                                KC(cudaStreamSynchronize(g_st));
                                std::vector<float> d((size_t) nr * N);
                                KC(cudaMemcpy(d.data(), s.mmq_dm, d.size() * 4, cudaMemcpyDeviceToHost));
                                int64_t first = -1, nans = 0;
                                for (size_t q = 0; q < d.size(); ++q)
                                    if (std::isnan(d[q]) || std::isinf(d[q])) { ++nans; if (first < 0) first = (int64_t) q; }
                                std::fprintf(stderr, "MMQDM l=%d gn=%d nr=%lld dm_nan=%lld first_nan_float=%lld row=%lld col=%lld\n",
                                             (int) l, gn, (long long) nr, (long long) nans, (long long) first,
                                             (long long) (first / N), (long long) (first % N));
                            }
                            if (std::getenv("STRATA_KOLIBRI_MMQ_DEBUG") != nullptr) {
                                // One line per layer-chunk-group: is the DEVICE bounds table still what the host wrote, and
                                // did the two products produce anything?  A failing run is localized by diffing two logs.
                                int32_t bd4[4] = {0, 0, 0, 0};
                                float g0 = 0, g1 = 0, d0 = 0, h0 = 0;
                                KC(cudaMemcpy(bd4, s.mmq_bounds, (size_t) std::min<int>((gn + 1), 4) * 4,
                                              cudaMemcpyDeviceToHost));
                                KC(cudaMemcpy(&g0, s.mmq_gu, 4, cudaMemcpyDeviceToHost));
                                KC(cudaMemcpy(&g1, s.mmq_gu + (size_t) (nr / 2) * 2 * FF, 4, cudaMemcpyDeviceToHost));
                                KC(cudaMemcpy(&d0, s.mmq_dm, 4, cudaMemcpyDeviceToHost));
                                KC(cudaMemcpy(&h0, s.mmq_h, 4, cudaMemcpyDeviceToHost));
                                std::fprintf(stderr, "MMQL l=%d g0=%zu gn=%d nr=%lld bd=%d,%d,%d,%d gu0=%.6g guM=%.6g dm0=%.6g h0=%.6g\n",
                                             (int) l, j0, gn, (long long) nr, bd4[0], bd4[1], bd4[2], bd4[3],
                                             (double) g0, (double) g1, (double) d0, (double) h0);
                            }
                            k_scatter_rows<<<grid1(nr * N), kThreads, 0, g_st>>>(s.mmq_dm, s.mmq_row_sel + r0, nr, A.gout_m, N);
                            if (std::getenv("STRATA_KOLIBRI_MMQ_DEBUG") != nullptr) {
                                KC(cudaStreamSynchronize(g_st));
                                float dg = 0, dh = 0, dd = 0, dgo = 0;
                                const int32_t s0 = s.mmq_row_sel_h[(size_t) r0];
                                KC(cudaMemcpy(&dg, s.mmq_gu, 4, cudaMemcpyDeviceToHost));
                                KC(cudaMemcpy(&dh, s.mmq_h, 4, cudaMemcpyDeviceToHost));
                                KC(cudaMemcpy(&dd, s.mmq_dm, 4, cudaMemcpyDeviceToHost));
                                KC(cudaMemcpy(&dgo, A.gout_m + (size_t) s0 * N, 4, cudaMemcpyDeviceToHost));
                                float xh = 0, xx = 0;
                                KC(cudaMemcpy(&xh, A.h_m, 4, cudaMemcpyDeviceToHost));
                                KC(cudaMemcpy(&xx, A.x_m, 4, cudaMemcpyDeviceToHost));
                                float g1 = 0, g2 = 0, d1 = 0, d2 = 0;
                                KC(cudaMemcpy(&g1, A.gout_m + (size_t) s0 * N + 1, 4, cudaMemcpyDeviceToHost));
                                KC(cudaMemcpy(&g2, A.gout_m + (size_t) s0 * N + 2, 4, cudaMemcpyDeviceToHost));
                                KC(cudaMemcpy(&d1, s.mmq_dm + 1, 4, cudaMemcpyDeviceToHost));
                                KC(cudaMemcpy(&d2, s.mmq_dm + 2, 4, cudaMemcpyDeviceToHost));
                                std::fprintf(stderr, "MMQPICK l=%d sel=%d gout=%.6g %.6g %.6g | dm=%.6g %.6g %.6g | nr=%lld maxr=%d gn=%d\n",
                                             (int) l, (int) s0, (double) dgo, (double) g1, (double) g2, (double) dd,
                                             (double) d1, (double) d2, (long long) nr, maxr, gn);
                                std::fprintf(stderr, "MMQDBG l=%d g0=%zu gn=%d nr=%lld maxr=%d | gu=%.6g h=%.6g dm=%.6g gout=%.6g"
                                                     " | h_m[0]=%.6g x_m[0]=%.6g | gt=%d dt=%d up_off=%lld down_off=%lld stride=%zu\n",
                                             (int) l, j0, gn, (long long) nr, maxr, dg, dh, dd, dgo,
                                             (double) xh, (double) xx, (int) b.L.gu_type, (int) b.L.d_type,
                                             (long long) b.L.up_off, (long long) b.L.down_off, b.stride);
                            }
                            r0 += nr;
                        }
                        if (ok) {
                            KC(cudaGetLastError());
                            KC(cudaStreamSynchronize(g_st));
                            mmq_did = true;
                            s.mmq_gout = true;   // gout_m is final for this layer-chunk: skip the pool's host H2D
                            static bool announced = false;
                            if (!announced) {
                                std::fprintf(stderr, "strata-kolibri: MMQ active (direct shared-expert tuning verified)\n");
                                announced = true;
                            }
                            if (std::getenv("STRATA_KOLIBRI_MMQ_DEBUG") != nullptr) {
                                std::vector<float> g((size_t) NKc * N);
                                KC(cudaMemcpy(g.data(), A.gout_m, g.size() * 4, cudaMemcpyDeviceToHost));
                                int64_t nans = 0, zeros = 0;
                                double mx = 0.0;
                                for (float v : g) {
                                    if (std::isnan(v) || std::isinf(v)) ++nans;
                                    else { if (v == 0.f) ++zeros; const double a = std::fabs((double) v); if (a > mx) mx = a; }
                                }
                                std::fprintf(stderr, "MMQCOV l=%d rows=%lld unwritten(NaN)=%lld (of %lld)\n",
                                             (int) l, (long long) rows, (long long) nans, (long long) g.size());
                                std::fprintf(stderr, "MMQGOUT l=%d rows=%lld max|v|=%.6g nan=%lld zeros=%lld (of %lld)\n",
                                             (int) l, (long long) rows, mx, (long long) nans, (long long) zeros,
                                             (long long) g.size());
                            }
                            if (e.counters) { e.pf_rows += rows; ++e.pf_chunks; e.pf_experts += (int) ml.size(); }
                        }
                    }
                }
#endif
                if (!mmq_did) e.pool.run([&](int shard) {
                    const auto sh_t0 = std::chrono::steady_clock::now();
                    const void* acts[64]; float* ffs[64]; const void* hqs[64]; float* eys[64];
                    for (size_t xi = (size_t) shard; xi < pool_list.size(); xi += (size_t) NS) {
                        const int x = pool_list[xi];
                        const int lo = cnt[x], hi = cnt[x + 1], m = hi - lo;
                        if (m <= 0) continue;
                        if (e.counters && (size_t) shard < e.pf_shard_bytes.size()) e.pf_shard_bytes[(size_t) shard] += (long long) b.stride;
                        const uint8_t* blob = b.table + (size_t) x * b.stride;
                        for (int a = 0; a < m; a += 64) {
                            const int mm = std::min(64, m - a);
                            for (int k = 0; k < mm; ++k) {
                                const int sel = ord[(size_t) lo + a + k];
                                acts[k] = s.act_cur + (size_t) (sel / K) * b.fmt.act_bytes;
                                ffs[k] = s.ff_cur + (size_t) sel * FF;
                            }
                            const auto tA = std::chrono::steady_clock::now();
                            strata::kernels::cpu::native_gu_rows(b.fmt, blob, acts, mm, ffs, 0, FF);
                            const auto tB = std::chrono::steady_clock::now();
                            for (int k = 0; k < mm; ++k) {
                                const int sel = ord[(size_t) lo + a + k];
                                char* hqp = s.hq_cur + (size_t) sel * b.fmt.h_bytes;
                                strata::kernels::cpu::native_quant_h(b.fmt, ffs[k], hqp);
                                hqs[k] = hqp;
                                eys[k] = A.ey_h + (size_t) sel * N;
                            }
                            const auto tC = std::chrono::steady_clock::now();
                            strata::kernels::cpu::native_down_rows(b.fmt, blob, hqs, mm, eys, 0, N);
                            const auto tD = std::chrono::steady_clock::now();
                            if (e.counters) e.pf_rows += mm;   // the pool's real work: one row = gate/up + down over N
                            e.pf_gu_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(tB - tA).count();
                            e.pf_qh_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(tC - tB).count();
                            e.pf_dn_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(tD - tC).count();
                        }
                    }
                    if (e.counters && (size_t) shard < e.pf_shard_ns.size())
                        e.pf_shard_ns[(size_t) shard] +=
                            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - sh_t0).count();
                }, NS);
                if (std::getenv("STRATA_KOLIBRI_PF_HASH") != nullptr) {
                    double mx = 0.0;
                    for (int64_t q = 0; q < NKc * N; ++q) { const double v = std::fabs((double) A.ey_h[(size_t) q]); if (v > mx) mx = v; }
                    std::fprintf(stderr, "PFPOOL l=%d ey_h max|v|=%.6g first=%.6g %.6g pool_ran=%d NKc=%lld\n", (int) l, mx,
                                 (double) A.ey_h[0], (double) A.ey_h[1], (int) (!mmq_did), (long long) NKc);
                }
                if (e.counters && !mmq_did) {
                    const auto pf_t1 = std::chrono::steady_clock::now();
                    e.pf_pool_ms += std::chrono::duration_cast<std::chrono::microseconds>(pf_t1 - pf_t0).count() / 1000;
                    int touched = 0;
                    for (int x = 0; x < (int) g.n_expert; ++x) if (cnt[x + 1] > cnt[x]) ++touched;
                    e.pf_pool_bytes += (long long) pool_list.size() * b.stride;
                    e.pf_experts += touched;
                    ++e.pf_chunks;
                }
                pp_mark(e.cnt.pf_other_ns);   // S7: the layer's other kernels (the shared expert, the MoE sum, the norms)
        e.cnt.pf_layers++;
        // S7 increment C: the resident rows land in res_ey_h (block order, padded rows ignored); scatter
                // them into A.ey_h at their selection rows.  Every selection row is covered exactly once, and the
                // existing H2D below ships the whole NKc x N block unchanged.
                if (b.f16 && s.pf_nblk > 0 && !pf_partial_on()) {
                    KC(cudaStreamSynchronize(g_st));
                    constexpr int M = Scratch::kResBlk;
                    for (int bi = 0; bi < s.pf_nblk; ++bi) {
                        const int lo = s.blk_lo[bi], mb = s.blk_m[bi];
                        for (int a = 0; a < mb; ++a)
                            std::memcpy(A.ey_h + (size_t) ord[(size_t) lo + a] * N, s.res_ey_h + ((int64_t) bi * M + a) * N, (size_t) N * 4);
                    }
                }
                // STREAM RACE (self-determinism probe: two runs, same binary/config, self-KL mean 9.67,
                // argmax-agree 5/32, maxabs 20.1 logits).  The grouped kernel above writes A.gout_m on g_st; this
                // H2D overwrites the SAME buffer on the default stream.  The two streams are unsynchronised here,
                // so the grouped kernel's writes race the pool's - a per-run-varying blend, which is exactly the
                // observed magnitude.  Drain g_st first.
                KC(cudaStreamSynchronize(g_st));
                if (std::getenv("STRATA_KOLIBRI_PF_HASH") != nullptr) {
                    // P1 probe: a per-chunk checksum of the expert leg's OUTPUT (the pool's miss rows plus the
                    // resident scatter into the same host buffer).  Two identical runs print two sequences; the first
                    // line that differs names the exact chunk where the prefill stops being reproducible.  The dump
                    // instruments cannot do this - --dump-layers disables chunking entirely, so it never runs this code.
                    // The MMQ arm writes the device selection buffer, not the pool's host output. Hashing the
                    // untouched host buffer used to record all zeros and could not detect a broken MMQ result.
                    if (s.mmq_gout) KC(cudaMemcpy(A.ey_h, A.gout_m, (size_t) NKc * N * 4, cudaMemcpyDeviceToHost));
                    double hs = 0.0, hmax = 0.0;
                    for (int64_t q = 0; q < NKc * N; ++q) {
                        const double v = (double) A.ey_h[(size_t) q];
                        hs += v;
                        if (std::fabs(v) > hmax) hmax = std::fabs(v);
                    }
                    std::fprintf(stderr, "PFMAG ey_h: max|v|=%.6g first3=%.6g %.6g %.6g NKc=%lld\n", hmax,
                                 (double) A.ey_h[0], (double) A.ey_h[1], (double) A.ey_h[2], (long long) NKc);
                    // The structural discriminators: if these differ between two identical runs the defect is in the
                    // cache BOOKKEEPING (which experts are resident, which slot each got); if they agree while the sum
                    // differs, the bookkeeping is fine and the arithmetic/GPU side is at fault.
                    int dup = 0;
                    for (size_t a = 0; a < s.pf_res_slot.size(); ++a)
                        for (size_t b2 = a + 1; b2 < s.pf_res_slot.size(); ++b2)
                            if (s.pf_res_slot[a] == s.pf_res_slot[b2]) ++dup;
                    std::fprintf(stderr, "PFHASH l=%d chunk=%lld sum=%.9e res=%zu miss=%zu nblk=%d dupslot=%d\n",
                                 (int) l, (long long) e.pf_chunks, hs, s.pf_res.size(), s.pf_miss.size(),
                                 (int) s.pf_nblk, dup);
                }
                if (pf_partial_on() && b.f16) {
                    // The pool has returned (e.pool.run waits for every shard), so A.ey_h's MISS rows are final and
                    // its RESIDENT rows are stale from an earlier chunk-layer by design - never read.  A selection
                    // is a miss iff its expert is not resident, decided from the published ids and NOT from the
                    // block bookkeeping, which is the P1 bug surface.
                    const int flip = A.p_flip;
                    if (A.p_done[flip] != nullptr) KC(cudaEventSynchronize(A.p_done[flip]));
                    float* ph = A.p_h[flip];
                    std::memset(ph, 0, (size_t) Tc * N * 4);
                    if ((int) s.res_mark.size() < (int) g.n_expert) s.res_mark.assign(g.n_expert, -1);
                    const int32_t gen = ++s.res_gen;
                    for (size_t ri = 0; ri < s.pf_res.size(); ++ri) s.res_mark[s.pf_res[ri]] = gen;
                    for (int t = 0; t < Tc; ++t) {
                        float* dst = ph + (size_t) t * N;
                        for (int k = 0; k < K; ++k) {
                            const int sel = t * K + k;
                            if (s.res_mark[A.ids_h[sel]] == gen) continue;   // resident: already on device
                            const float w = A.wts_h[sel];
                            const float* src = A.ey_h + (size_t) sel * N;
                            for (int i = 0; i < N; ++i) dst[i] += w * src[i];
                        }
                    }
                    // On the DEFAULT stream, because the combine consumes it there.  Enqueuing this on g_st and
                    // consuming it on the default stream is exactly the cross-stream pattern the note above
                    // names as the self-determinism race.
                    KC(cudaMemcpyAsync(A.p_m, ph, (size_t) Tc * N * 4, cudaMemcpyHostToDevice));
                    if (e.counters) { ++e.cnt.h2d_n; e.cnt.h2d_bytes += (long long) Tc * N * 4; }
                    if (A.p_done[flip] != nullptr) KC(cudaEventRecord(A.p_done[flip]));
                    A.p_flip ^= 1;
                    s.pf_partial_used = true;
                } else if (!s.mmq_gout) {
                    KC(cudaMemcpy(A.gout_m, A.ey_h, (size_t) NKc * N * 4, cudaMemcpyHostToDevice));
                    if (e.counters) { ++e.cnt.h2d_n; e.cnt.h2d_bytes += (long long) NKc * N * 4; }
                }
                pp_mark(e.cnt.pf_post_ns);   // the scatter + this H2D (NKc x N x 4 = 31.5 MB at Tc=512)
            } else {
            // The mmvq fallback leg: per-token activation quantize, gate/up per selection, SwiGLU, h quantize,
            // down per selection.  This leg MUST stay inside the else.  The pool leg above already produced the
            // expert output (native_down_rows -> A.ey_h) and copied it into A.gout_m, and this leg writes
            // A.gout_m from s.eg_m/s.eu_m, which the pool NEVER fills.  Measured (nsys, staged pp512, 24 workers):
            // 76,750 native_q4_k_mmvq launches = 6.10 s = 67% of all GPU kernel time and 42% of the pp512 wall,
            // all of it dead work, and it OVERWROTE the pool's result with stale-buffer garbage - the cause of
            // the prefill path's KL 0.075 and of its run-to-run non-determinism (unstaged vs itself meanKL 0.0196).
            for (int t = 0; t < Tc; ++t) {
                strata::kernels::native_quantize_q8_1(A.h_m + (size_t) t * N, g_q8, N, 1, g_st);
                for (int k = 0; k < K; ++k) {
                    const int sel = t * K + k;
                    const uint8_t* blob = b.table + (size_t) A.ids_h[sel] * b.stride;
                    strata::kernels::native_mmvq(b.L.gu_type, blob, (const char*) g_q8, s.eg_m + (size_t) sel * FF,
                                                 N, FF, 1, g_st);
                    strata::kernels::native_mmvq(b.L.gu_type, blob + b.L.up_off, (const char*) g_q8,
                                                 s.eu_m + (size_t) sel * FF, N, FF, 1, g_st);
                }
            }
            k_swiglu_rows<<<grid1((int64_t) NKc * FF), kThreads>>>(s.eg_m, s.eu_m, s.eh_m, (int64_t) NKc * FF);
            for (int sel = 0; sel < NKc; ++sel) {
                strata::kernels::native_quantize_q8_1(s.eh_m + (size_t) sel * FF, g_q8, FF, 1, g_st);
                const uint8_t* blob = b.table + (size_t) A.ids_h[sel] * b.stride + b.L.down_off;
                strata::kernels::native_mmvq(b.L.d_type, blob, (const char*) g_q8, A.gout_m + (size_t) sel * N,
                                             FF, N, 1, g_st);
            }
            }   // end the mmvq fallback: the pool leg above must not fall through into the swiglu+down leg
        }
        {
            const Dense& wg = e.dense.at(L(l, "ffn_gate_shexp.weight"));
            if (dense_f16_on() && wg.f16) e.gemm.f16((const uint16_t*) A.x16, wg.f16, A.sg_m, Tc, FS, N);
            else e.gemm.native((const uint16_t*) A.x16, wg.type, wg.d, A.sg_m, Tc, FS, N);
            const Dense& wu = e.dense.at(L(l, "ffn_up_shexp.weight"));
            if (dense_f16_on() && wu.f16) e.gemm.f16((const uint16_t*) A.x16, wu.f16, A.su_m, Tc, FS, N);
            else e.gemm.native((const uint16_t*) A.x16, wu.type, wu.d, A.su_m, Tc, FS, N);
            k_swiglu_rows<<<grid1((int64_t) Tc * FS), kThreads>>>(A.sg_m, A.su_m, A.sh_m, (int64_t) Tc * FS);
            k_to_f16<<<grid1((int64_t) Tc * FS), kThreads>>>(A.sh_m, A.x16, (int64_t) Tc * FS);
            const Dense& wd = e.dense.at(L(l, "ffn_down_shexp.weight"));
            if (dense_f16_on() && wd.f16) e.gemm.f16((const uint16_t*) A.x16, wd.f16, A.sy_m, Tc, N, FS);
            else e.gemm.native((const uint16_t*) A.x16, wd.type, wd.d, A.sy_m, Tc, N, FS);
        }
        pp_mark(e.cnt.pf_shx_ns);   // the shared expert's three gemvs + SwiGLU
        if (s.pf_partial_used) {
            // G1: one combine over shared-expert row + the pool's partial + the resident rows that never left
            // the device, replacing both the 31.5 MB H2D and k_moe_sum_rows for this layer.
            strata::kolibri::pf_combine_rows(A.moe_m, A.sy_m, A.p_m, s.res_ey, A.rr_sel_m, A.wts_m,
                                             Tc, s.pf_nblk * Scratch::kResBlk, K, N, 0);
            s.pf_partial_used = false;
        } else {
            k_moe_sum_rows<<<dim3(Tc, (N + B - 1) / B), B>>>(A.moe_m, A.gout_m, A.wts_m, A.sy_m, K, N);
        }
        k_rms_rows<<<Tc, kThreads>>>(A.moe_m, e.f32.at(L(l, "post_ffw_norm.weight")), A.moe_m, N, e.eps);
        k_add_rows<<<grid1((int64_t) Tc * N), kThreads>>>(A.x_m, A.moe_m, (int64_t) Tc * N);
        pp_mark(e.cnt.pf_tail_ns);   // the MoE sum + post-ffw norm + residual
        KC(cudaGetLastError());
        };
        if (pass == 0) layer_pre();
        if (pass == 1 || n_ch == 1) layer_post();
    }
    }
    // the tail runs once, on the LAST chunk processed (chunk 2 when the wavefront is on)
    ActSet& Alast = (n_ch == 2) ? s.act2 : s.act1;
    const int Tlast = (n_ch == 2) ? T2 : T;
    k_rms<<<1, kThreads>>>(Alast.x_m + (size_t) (Tlast - 1) * N, e.f32.at("output_norm.weight"), s.h, N, e.eps);
    gemv_dec(e.dense.at("output.weight"), s.h, s.dec_h16, s.logits, 0);
    KC(cudaDeviceSynchronize());
}

#include "token_graph.inc"
#include "prefix_fingerprint.inc"

void forward(const Engine& e, Scratch& s, int tok, float* layers_host) {
    decode_aborted = false;
    if (e.counters) ++e.cnt.tok;   // every processed token (prompt + decode): the per-token normaliser
    // **0.1 DECODE INSTRUMENT** (dynamic-expert-cache plan, step 0.1).  ONE stderr line per decoded token with that
    // token's own routed-pick split, so the hit-rate-vs-token curve costs one printf and no new counter: the two
    // counters it reads are the ones the DONE line already carries.  Gated, so with the variable unset the default
    // path prints nothing new and the 16-field DONE/CNT contract is untouched.
    const bool hit_trace = std::getenv("STRATA_KOLIBRI_HITTRACE") != nullptr;
    const long long hit0 = e.cnt.slot_hits, miss0 = e.cnt.slot_miss;
    const auto& g = e.g;
    const int N = (int) g.n_embd, K = (int) e.top_k, FF = (int) g.n_ff, FS = (int) g.n_ff_shexp;
    const int H = (int) g.n_head, HKV = (int) g.n_head_kv, D = (int) g.head_dim;
    const int B = 256;
    // Reserve a whole token's nonzero publication identities BEFORE cold segmented execution too.
    // Checking only inside run_token_graph misses rollover during mirror/bucket warmup.
    if (dec_handoff_on()) {
        HandoffBuffers& hb = *handoff_buffers(N, K);
        if (hb.expect > UINT32_MAX - (uint32_t) g.n_layers) {
            KC(cudaDeviceSynchronize());  // rare token-boundary quiescence before descriptor reuse
            hb.expect = 0;
            strata::kolibri::decode_control_store(hb.host_control->ready, 0);
            strata::kolibri::decode_control_store(hb.host_control->acknowledged, 0);
            strata::kolibri::decode_control_store(hb.host_control->acknowledged_generation, 0);
        }
    }
    const Dense& emb = e.dense.at("token_embd.weight");
    if (layers_host && s.layer_trace_d == nullptr) s.layer_trace_d = dalloc((size_t) g.n_layers * N);
        if (e.graphs > 0 && layers_host == nullptr) {
        cudaMemcpyAsync(s.tok_d, &tok, 4, cudaMemcpyHostToDevice, g_st);
        switch (emb.type) {
        case T_Q8_0: k_row_p<T_Q8_0><<<4, 128, 0, g_st>>>(emb.d, emb.row_bytes, s.tok_d, N, s.x); break;
        case T_Q4_K: k_row_p<T_Q4_K><<<4, 128, 0, g_st>>>(emb.d, emb.row_bytes, s.tok_d, N, s.x); break;
        case T_Q6_K: k_row_p<T_Q6_K><<<4, 128, 0, g_st>>>(emb.d, emb.row_bytes, s.tok_d, N, s.x); break;
        case T_F16: k_row_p<T_F16><<<4, 128, 0, g_st>>>(emb.d, emb.row_bytes, s.tok_d, N, s.x); break;
        default: std::fprintf(stderr, "strata-kolibri: token_embd type %d\n", emb.type); std::exit(1);
        }
        for (int64_t l = 0; l < g.n_layers; ++l)
            if (e.gexec[(size_t) l] != 0) cudaGraphLaunch((cudaGraphExec_t) e.gexec[(size_t) l], g_st);
            else layer_step(e, s, l, false, 0, nullptr);
        k_pos_add<<<1, 1, 0, g_st>>>(s.pos_d);
    } else {
        // The embedding gather writes s.x, which the first layer reads: with the arm on it must be on the same
        // stream as the layers, or a non-blocking decode stream would race the legacy-stream gather.
        const cudaStream_t emb_st = dec_stream_on() ? dec_stream() : (cudaStream_t) 0;
        switch (emb.type) {
        case T_Q8_0: k_row<T_Q8_0><<<4, 128, 0, emb_st>>>(emb.d, emb.row_bytes, tok, N, s.x); break;
        case T_Q4_K: k_row<T_Q4_K><<<4, 128, 0, emb_st>>>(emb.d, emb.row_bytes, tok, N, s.x); break;
        case T_Q6_K: k_row<T_Q6_K><<<4, 128, 0, emb_st>>>(emb.d, emb.row_bytes, tok, N, s.x); break;
        case T_F16: k_row<T_F16><<<4, 128, 0, emb_st>>>(emb.d, emb.row_bytes, tok, N, s.x); break;
        default: std::fprintf(stderr, "strata-kolibri: token_embd type %d\n", emb.type); std::exit(1);
        }
        // The device-position attention reads s.pos from s.pos_d, so the host's position must be there before the
        // first layer runs.  One 4-byte copy per token, on the same stream the layers use.
        if (dec_attn_dp_on() || dec_graph_on()) {
            if (dec_stream_on()) KC(cudaMemcpyAsync(s.pos_d, &s.pos, 4, cudaMemcpyHostToDevice, dec_stream()));
            else KC(cudaMemcpy(s.pos_d, &s.pos, 4, cudaMemcpyHostToDevice));
        }
        const bool token_ran = dec_token_graph_on() && layers_host == nullptr &&
                               run_token_graph(e, s, dec_stream());
        if (decode_aborted) return;
        if (!token_ran)
            for (int64_t l = 0; l < g.n_layers; ++l)
                layer_step(e, s, l, false, dec_stream_on() ? dec_stream() : 0, layers_host);
    }
    k_rms<<<1, kThreads, 0, dec_stream_on() ? dec_stream() : 0>>>(s.x, e.f32.at("output_norm.weight"), s.h, N, e.eps);
    gemv_dec(e.dense.at("output.weight"), s.h, s.dec_h16, s.logits, dec_stream_on() ? dec_stream() : 0);
    KC(cudaDeviceSynchronize());
    if (layers_host)
        KC(cudaMemcpy(layers_host, s.layer_trace_d, (size_t) g.n_layers * N * 4, cudaMemcpyDeviceToHost));
    ++s.pos;
    if (hit_trace) {   // pos is 0-based after the increment: the first decoded token prints pos=1
        const long long h = e.cnt.slot_hits - hit0, m = e.cnt.slot_miss - miss0;
        std::fprintf(stderr, "HITTRACE pos=%lld hits=%lld miss=%lld\n", (long long) s.pos, h, m);
        std::fflush(stderr);
    }
}

void init_scratch(const Engine& e, Scratch& s) {
    const auto& g = e.g;
    const int N = (int) g.n_embd, FF = (int) g.n_ff, K = (int) e.top_k;
    s.x = dalloc(N); s.h = dalloc(N); s.y = dalloc(N); s.moe = dalloc(N); s.sy = dalloc(N);
    s.q = dalloc(g.n_head * g.head_dim); s.k = dalloc(g.n_head_kv * g.head_dim); s.v = dalloc(g.n_head_kv * g.head_dim);
    s.att = dalloc(g.n_head * g.head_dim);
    s.router = dalloc(g.n_expert);
    s.eg = dalloc((size_t) K * FF); s.eu = dalloc((size_t) K * FF); s.eh = dalloc((size_t) K * FF); s.ey = dalloc((size_t) K * N);
    s.sg = dalloc(g.n_ff_shexp); s.su = dalloc(g.n_ff_shexp); s.sh = dalloc(g.n_ff_shexp);
    s.wts = dalloc(K);
    KC(cudaMalloc((void**) &s.pos_d, 4));
    KC(cudaMalloc((void**) &s.tok_d, 4));
    if (e.prefill > 0) {   // the chunk's rows: tmax tokens, tmax*K expert entries, the FP16 attention images
        const int T = Scratch::kTmax, NK = T * K;
        s.exp_cnt.resize((size_t) g.n_expert + 1);
        // S7 staging buffers: 192 slots x max stride (~460 MB, inside the 13.53 GiB free; 122 slots/layer of the
        // full cache fails to allocate, so the stage is sized per-chunk, not per-model) + the id->rank map.
        // top_k lives on the Engine (e.top_k), NOT on ModelGeometry - g.top_k does not compile.
        {
            size_t stride_max = 0;
            for (const auto& bb : e.blobs) stride_max = std::max(stride_max, (size_t) bb.stride);
            s.stage_n = 192;
            KC(cudaMalloc((void**) &s.stage, (size_t) s.stage_n * stride_max));
            KC(cudaMalloc((void**) &s.stage_ids_m, (size_t) Scratch::kTmax * (size_t) e.top_k * sizeof(int)));
            s.stage_rank.assign((size_t) g.n_expert, -1);
            s.stage_ids_h.assign((size_t) Scratch::kTmax * (size_t) e.top_k, 0);
        }
        s.exp_ord.resize((size_t) NK);
        const size_t hd = (size_t) g.n_head * g.head_dim, hkd = (size_t) g.n_head_kv * g.head_dim;
        KC(cudaMalloc((void**) &s.toks_d, T * 4));
        if (pf_pipe_on()) KC(cudaMalloc((void**) &s.toks_d2, T * 4));
        KC(cudaMalloc((void**) &s.x_m, (size_t) T * N * 4));
        KC(cudaMalloc((void**) &s.h_m, (size_t) T * N * 4));
        KC(cudaMalloc((void**) &s.y_m, (size_t) T * N * 4));
        KC(cudaMalloc((void**) &s.q_m, (size_t) T * hd * 4));
        KC(cudaMalloc((void**) &s.k_m, (size_t) T * hkd * 4));
        KC(cudaMalloc((void**) &s.v_m, (size_t) T * hkd * 4));
        KC(cudaMalloc((void**) &s.att_m, (size_t) T * hd * 4));
        KC(cudaMalloc((void**) &s.router_m, (size_t) T * g.n_expert * 4));
        KC(cudaMalloc((void**) &s.moe_m, (size_t) T * N * 4));
        KC(cudaMalloc((void**) &s.sy_m, (size_t) T * N * 4));
        KC(cudaMalloc((void**) &s.sg_m, (size_t) T * g.n_ff_shexp * 4));
        KC(cudaMalloc((void**) &s.su_m, (size_t) T * g.n_ff_shexp * 4));
        KC(cudaMalloc((void**) &s.sh_m, (size_t) T * g.n_ff_shexp * 4));
        KC(cudaMalloc((void**) &s.gout_m, (size_t) NK * N * 4));
        s.act1 = ActSet{s.x_m, s.h_m, s.y_m, s.q_m, s.k_m, s.v_m, s.att_m, s.router_m, s.moe_m, s.sy_m, s.sg_m, s.su_m,
                        s.sh_m, s.gout_m};
        if (pf_pipe_on()) {   // A2b step 1
            KC(cudaMalloc((void**) &s.x_m2, (size_t) T * N * 4));
            KC(cudaMalloc((void**) &s.h_m2, (size_t) T * N * 4));
            KC(cudaMalloc((void**) &s.y_m2, (size_t) T * N * 4));
            KC(cudaMalloc((void**) &s.q_m2, (size_t) T * hd * 4));
            KC(cudaMalloc((void**) &s.k_m2, (size_t) T * hkd * 4));
            KC(cudaMalloc((void**) &s.v_m2, (size_t) T * hkd * 4));
            KC(cudaMalloc((void**) &s.att_m2, (size_t) T * hd * 4));
            KC(cudaMalloc((void**) &s.router_m2, (size_t) T * g.n_expert * 4));
            KC(cudaMalloc((void**) &s.moe_m2, (size_t) T * N * 4));
            KC(cudaMalloc((void**) &s.sy_m2, (size_t) T * N * 4));
            KC(cudaMalloc((void**) &s.sg_m2, (size_t) T * g.n_ff_shexp * 4));
            KC(cudaMalloc((void**) &s.su_m2, (size_t) T * g.n_ff_shexp * 4));
            KC(cudaMalloc((void**) &s.sh_m2, (size_t) T * g.n_ff_shexp * 4));
            KC(cudaMalloc((void**) &s.gout_m2, (size_t) NK * N * 4));
            std::fprintf(stderr, "strata-kolibri: prefill pipeline: second activation set allocated (A2b step 1, unused)\n");
            s.act2 = ActSet{s.x_m2, s.h_m2, s.y_m2, s.q_m2, s.k_m2, s.v_m2, s.att_m2, s.router_m2, s.moe_m2, s.sy_m2,
                            s.sg_m2, s.su_m2, s.sh_m2, s.gout_m2};
        }
        KC(cudaMalloc((void**) &s.eg_m, (size_t) NK * g.n_ff * 4));
        KC(cudaMalloc((void**) &s.eu_m, (size_t) NK * g.n_ff * 4));
        KC(cudaMalloc((void**) &s.eh_m, (size_t) NK * g.n_ff * 4));
        KC(cudaMalloc((void**) &s.ids_m, NK * 4));
        if (pf_pipe_on()) KC(cudaMalloc((void**) &s.ids_m2, NK * 4));
        KC(cudaMalloc((void**) &s.wts_m, NK * 4));
        if (pf_pipe_on()) KC(cudaMalloc((void**) &s.wts_m2, NK * 4));
        KC(cudaMalloc((void**) &s.grp_ptr_m, ((size_t) std::min<int64_t>(NK, g.n_expert) + 1) * sizeof(unsigned long long)));
        if (pf_pipe_on()) KC(cudaMalloc((void**) &s.grp_ptr_m2, ((size_t) std::min<int64_t>(NK, g.n_expert) + 1) * sizeof(unsigned long long)));
        KC(cudaMalloc((void**) &s.grp_start_m, ((size_t) std::min<int64_t>(NK, g.n_expert) + 1) * 4));
        if (pf_pipe_on()) KC(cudaMalloc((void**) &s.grp_start_m2, ((size_t) std::min<int64_t>(NK, g.n_expert) + 1) * 4));
        KC(cudaMalloc((void**) &s.n_groups_m, 4));
        if (pf_pipe_on()) KC(cudaMalloc((void**) &s.n_groups_m2, 4));
    // Self-determinism probe (two runs, same binary/config: self-KL mean 9.67, argmax-agree 5/32).  The grouped
    // launch runs with grid_groups = 0, which the header documents as cap_groups (min(NK,n_expert)+1 = 385) block
    // rows, while k_group fills only the first *n_groups (~155) entries.  Zero the group tables so the surplus
    // rows read a benign 0 pointer/offset instead of uninitialized device memory.
    KC(cudaMemset(s.grp_ptr_m, 0, ((size_t) std::min<int64_t>(NK, g.n_expert) + 1) * sizeof(unsigned long long)));
    KC(cudaMemset(s.grp_start_m, 0, ((size_t) std::min<int64_t>(NK, g.n_expert) + 1) * 4));
    KC(cudaMemset(s.n_groups_m, 0, 4));
        KC(cudaMalloc((void**) &s.ent_dst_m, NK * 4));
        if (pf_pipe_on()) KC(cudaMalloc((void**) &s.ent_dst_m2, NK * 4));
        KC(cudaMalloc((void**) &s.ent_tok_m, NK * 4));
        if (pf_pipe_on()) KC(cudaMalloc((void**) &s.ent_tok_m2, NK * 4));
        KC(cudaMalloc((void**) &s.xq8_m, strata::kernels::native_q8_1_bytes((int) g.n_embd, 1) * (size_t) T));
        if (pf_pipe_on()) KC(cudaMalloc((void**) &s.xq8_m2, strata::kernels::native_q8_1_bytes((int) g.n_embd, 1) * (size_t) T));
        KC(cudaMalloc((void**) &s.scratch_m, strata::kernels::native_expert_scratch_bytes(NK, g.n_ff)));
        const size_t f16 = 2;
        KC(cudaMalloc((void**) &s.x16, (size_t) T * std::max<int64_t>(N, std::max<int64_t>(g.n_ff, (int64_t) hd)) * f16));
        if (pf_pipe_on()) KC(cudaMalloc((void**) &s.x16_2, (size_t) T * std::max<int64_t>(N, std::max<int64_t>(g.n_ff, (int64_t) hd)) * f16));
        KC(cudaMalloc((void**) &s.q16, (size_t) T * hd * f16));
        if (pf_pipe_on()) KC(cudaMalloc((void**) &s.q16_2, (size_t) T * hd * f16));
            s.act2.x16 = s.x16_2; s.act2.q16 = s.q16_2; s.act2.xq8_m = s.xq8_m2;
            s.act2.ids_m = s.ids_m2; s.act2.toks_d = s.toks_d2;
            s.act2.grp_start_m = s.grp_start_m2; s.act2.n_groups_m = s.n_groups_m2;
            s.act2.ent_dst_m = s.ent_dst_m2; s.act2.ent_tok_m = s.ent_tok_m2; s.act2.grp_ptr_m = s.grp_ptr_m2;
        s.act1.x16 = s.x16; s.act1.q16 = s.q16; s.act1.xq8_m = s.xq8_m;
        s.act1.ids_m = s.ids_m; s.act1.toks_d = s.toks_d;
        s.act1.grp_start_m = s.grp_start_m; s.act1.n_groups_m = s.n_groups_m;
        s.act1.ent_dst_m = s.ent_dst_m; s.act1.ent_tok_m = s.ent_tok_m; s.act1.grp_ptr_m = s.grp_ptr_m;

        KC(cudaMalloc((void**) &s.att16, (size_t) T * hd * f16));
        KC(cudaMalloc((void**) &s.k16, (size_t) g.n_head_kv * s.ctx * g.head_dim * f16));
        KC(cudaMalloc((void**) &s.v16, (size_t) g.n_head_kv * s.ctx * g.head_dim * f16));
        if (pf_pipe_on()) {
            KC(cudaMalloc((void**) &s.k16_2, (size_t) g.n_head_kv * s.ctx * g.head_dim * f16));
            KC(cudaMalloc((void**) &s.v16_2, (size_t) g.n_head_kv * s.ctx * g.head_dim * f16));
        }
        // the scores slice is H * tq * ldp; the runtime tq uses the chunk's ldp (>= 512), so the worst case is
        // tq_w * 512 is the nt * ldp bound the buffer must cover (== the attn cap / H); nt * ldp <= tq_w * 512
        const size_t tq_w = std::max<int64_t>(1, (int64_t) (attn_cap_floats() / ((size_t) g.n_head * 512)));
        KC(cudaMalloc((void**) &s.att_s, (size_t) g.n_head * tq_w * 512 * 4 + (size_t) g.n_head * 8 * 4));
        KC(cudaMalloc((void**) &s.p16, ((size_t) g.n_head * tq_w * 512 + (size_t) g.n_head * 8) * f16));
    }
    if (e.grouped > 0) {   // one grouped launch per layer: K entries, K groups at most, q8_1 acts for one token
        const int64_t cap_g = std::min<int64_t>(K, g.n_expert) + 1, cap_e = K;
        KC(cudaMalloc((void**) &s.grp_ptr, cap_g * sizeof(unsigned long long)));
        KC(cudaMalloc((void**) &s.grp_start, cap_g * sizeof(int32_t)));
        KC(cudaMalloc((void**) &s.n_groups, sizeof(int32_t)));
        KC(cudaMalloc((void**) &s.ent_dst, cap_e * sizeof(int32_t)));
        KC(cudaMalloc((void**) &s.ent_tok, cap_e * sizeof(int32_t)));
        KC(cudaMalloc((void**) &s.xq8, strata::kernels::native_q8_1_bytes((int) g.n_embd, 1)));
        KC(cudaMalloc((void**) &s.scratch, strata::kernels::native_expert_scratch_bytes(cap_e, g.n_ff)));
        KC(cudaMalloc((void**) &s.gout, (size_t) K * N * 4));
    }
    if (e.slots != 0) {   // the GPU expert cache's rows: the in-slot side and the overflow side.  kSlotsAuto (-1)
                          // counts as "has a cache": the fit runs after this, but the buffers it will need are these.
        KC(cudaMalloc((void**) &s.eg2, (size_t) K * FF * 4));
        KC(cudaMalloc((void**) &s.eu2, (size_t) K * FF * 4));
        KC(cudaMalloc((void**) &s.eh2, (size_t) K * FF * 4));
        KC(cudaMalloc((void**) &s.ey2, (size_t) K * N * 4));
        KC(cudaMalloc((void**) &s.ey_cpu, (size_t) K * N * 4));
    }
    if (e.cpu_threads > 0) {   // the CPU expert path's pinned host side (one layer's working set); the quantized
                               // activations are sized from the layers' fmt (kolibri's 2560-as-Q8_K is 23,360 B)
        size_t act_b = 0, h_b = 0;
        for (const auto& b : e.blobs) if (b.native_ok) { act_b = std::max(act_b, b.fmt.act_bytes); h_b = std::max(h_b, b.fmt.h_bytes); }
        const int64_t rows = e.prefill > 0 ? Scratch::kTmax : 1;          // prefill: a whole chunk on the host
        const int64_t ents = e.prefill > 0 ? (int64_t) Scratch::kTmax * K : K;
        KC(cudaHostAlloc((void**) &s.h_host, (size_t) rows * N * 4, cudaHostAllocDefault));
        KC(cudaHostAlloc((void**) &s.act, (size_t) rows * act_b, cudaHostAllocDefault));
        KC(cudaHostAlloc((void**) &s.ff, (size_t) ents * FF * 4, cudaHostAllocDefault));
        KC(cudaHostAlloc((void**) &s.hq, (size_t) ents * h_b, cudaHostAllocDefault));
        KC(cudaHostAlloc((void**) &s.ey_h, (size_t) ents * N * 4, cudaHostAllocDefault));
#if defined(STRATA_KOLIBRI_MMQ)
        if (e.prefill > 0 && pf_mmq_on()) {
            // staging sizes from the blob's OWN offsets: [0, up_off) gate, [up_off, down_off) up, [down_off, stride) down
            size_t gu_max = 0, d_max = 0;
            for (const auto& b : e.blobs) {
                if (!strata::prefill::mmq::supported(b.L.gu_type) || !strata::prefill::mmq::supported(b.L.d_type)) continue;
                gu_max = std::max(gu_max, (size_t) b.L.down_off);
                d_max = std::max(d_max, b.stride > (size_t) b.L.down_off ? b.stride - (size_t) b.L.down_off : 0);
            }
            // The rows of a chunk are the chunk's selections PADDED to kMinRows per expert, so a chunk whose experts
            // mostly hold one or two rows needs up to 8 extra rows for every expert it touches.  Sizing these by `ents`
            // alone overflowed a std::vector at 8 191 tokens (glibc: free(): invalid next size).  Reserve the padding.
            const int64_t rcap = ents + (int64_t) 8 * ((int64_t) e.g.n_expert + 1);
            s.mmq_rows_cap = rcap;
            s.mmq_row_tok_h.resize((size_t) rcap);
            s.mmq_row_sel_h.resize((size_t) rcap);
            // One bounds slice PER GROUP, filled and uploaded once per chunk: a per-group upload reused the same
            // small host array while the GPU could still be reading it (pageable H2D is not guaranteed to be copied
            // at call time), which intermittently fed the product col_diff = 0 - every tile returns early
            // (mmq.cuh:1017) and the layer came out all zeros: gen 0 0 0 ... once in five runs.
            s.mmq_bounds_h.resize((size_t) (e.g.n_expert + 1) * (strata::prefill::mmq::kGatherGroupMax + 1));
            KC(cudaMalloc(&s.mmq_xq, strata::prefill::mmq::q8_bytes(rcap, N) + 4096));
            KC(cudaMalloc(&s.mmq_xqh, strata::prefill::mmq::q8_bytes(rcap, FF) + 4096));
            KC(cudaMalloc((void**) &s.mmq_gu, (size_t) rcap * 2 * FF * 4));
            KC(cudaMalloc((void**) &s.mmq_h, (size_t) rcap * FF * 4));
            KC(cudaMalloc((void**) &s.mmq_dm, (size_t) rcap * N * 4));
            // MMQ reads up to 4096 bytes past an expert's matrix (prefill.cpp: MMQ_TAIL), so EVERY expert's slot in the
            // staging carries the tail - not just the last one.  Without it, expert 0's tile read expert 1's weights and
            // experts 1..n came back NaN (measured: first NaN at row = maxr, ledger §P4).
            KC(cudaMalloc(&s.mmq_gu_stage, 16 * gu_max + 8192));
            KC(cudaMalloc(&s.mmq_d_stage, 16 * d_max + 8192));
            KC(cudaMalloc((void**) &s.mmq_row_tok, (size_t) rcap * 4));
            KC(cudaMalloc((void**) &s.mmq_row_sel, (size_t) rcap * 4));
            KC(cudaMalloc((void**) &s.mmq_ident, (size_t) rcap * 4));
            KC(cudaMalloc((void**) &s.mmq_bounds,
                          (size_t) (e.g.n_expert + 1) * (strata::prefill::mmq::kGatherGroupMax + 1) * 4));
            strata::prefill::mmq::iota(s.mmq_ident, rcap, g_st);
            if (gu_max == 0) {
                // An arm that is switched on but cannot run must say so: the guard above falls through to the CPU pool
                // for every layer, so without this line the operator sees "on" and measures pool speed.
                std::fprintf(stderr, "strata-kolibri: STRATA_KOLIBRI_PF_MMQ is set, but no expert type in this model has an "
                                     "MMQ path here - K-quants need STRATA_MMQ_KQ=1 (static on HIP; CUDA also needs "
                                     "strata_kq beside the engine). Falling back to the CPU pool.\n");
            }
            if (e.mmq_ctx == nullptr) e.mmq_ctx = std::make_unique<strata::prefill::mmq::Context>();
            std::fprintf(stderr, "strata-kolibri: STRATA_KOLIBRI_PF_MMQ: prompt experts on the int8 tensor units "
                                 "(gate/up %zu B, down %zu B per expert, %lld selections a chunk)\n",
                         gu_max, d_max, (long long) ents);
            if (gu_max != 0)
                std::fprintf(stderr, "strata-kolibri: MMQ requires direct shared-expert tuning; "
                                     "FP16-resident expert layers and unsupported configurations retain the pool.\n");
        }
#endif
        if (pf_pipe_on()) {
            KC(cudaHostAlloc((void**) &s.ey_h2, (size_t) ents * N * 4, cudaHostAllocDefault));
            KC(cudaHostAlloc((void**) &s.h_host_b, (size_t) rows * N * 4, cudaHostAllocDefault));
            KC(cudaHostAlloc((void**) &s.act_b, (size_t) rows * act_b, cudaHostAllocDefault));
            KC(cudaHostAlloc((void**) &s.ff_b, (size_t) ents * FF * 4, cudaHostAllocDefault));
            KC(cudaHostAlloc((void**) &s.hq_b, (size_t) ents * h_b, cudaHostAllocDefault));
        }
        s.h_host_cur = s.h_host; s.act_cur = s.act; s.ff_cur = s.ff; s.hq_cur = s.hq;   // chunk 0 is the original set
        s.act1.ey_h = s.ey_h;
        if (pf_pipe_on()) s.act2.ey_h = s.ey_h2;
        s.act1.wts_m = s.wts_m;
        if (pf_pipe_on()) s.act2.wts_m = s.wts_m2;
        // A2b/needle: the RoPE angle precision gate (see rope_cs_sn).  Mirrored through the ACCESSOR, not the raw
        // env parse, so the device symbol and the host agree about the default.
        { const int rf64 = env_on("STRATA_KOLIBRI_ROPE_F64") ? 1 : 0;
          KC(cudaMemcpyToSymbol(g_rope_f64, &rf64, sizeof(int))); }
        // A2a: pinned ids + one event per chunk (both chunk 1's, always: the single-chunk path uses them too and
        // cudaMemcpyAsync + record + wait is bitwise the old blocking copy).
        KC(cudaHostAlloc((void**) &s.ids_h, (size_t) 8192 * K * sizeof(int), cudaHostAllocDefault));
        KC(cudaEventCreateWithFlags(&s.ev, cudaEventDisableTiming));
        if (pf_pipe_on()) {
            KC(cudaHostAlloc((void**) &s.ids_h2, (size_t) 8192 * K * sizeof(int), cudaHostAllocDefault));
            KC(cudaEventCreateWithFlags(&s.ev2, cudaEventDisableTiming));
        }
        s.act1.ids_h = s.ids_h; s.act1.ev = s.ev;
        if (pf_pipe_on()) { s.act2.ids_h = s.ids_h2; s.act2.ev = s.ev2; }
        s.act1.k16 = s.k16; s.act1.v16 = s.v16;
        if (pf_pipe_on()) { s.act2.k16 = s.k16_2; s.act2.v16 = s.v16_2; }
        if (expert_f16_n() > 0) {   // S7 increment B/C: resident-expert GEMM working set (device + pinned)
            const int64_t rrows = (int64_t) Scratch::kResBlkMax * Scratch::kResBlk;
            KC(cudaMalloc((void**) &s.res_x16, (size_t) rrows * N * 2));
            KC(cudaMalloc((void**) &s.res_g, (size_t) rrows * FF * 4));
            KC(cudaMalloc((void**) &s.res_u, (size_t) rrows * FF * 4));
            KC(cudaMalloc((void**) &s.res_h, (size_t) rrows * FF * 4));
            KC(cudaMalloc((void**) &s.res_h16, (size_t) rrows * FF * 2));
            KC(cudaMalloc((void**) &s.res_ey, (size_t) rrows * N * 4));
            KC(cudaMalloc((void**) &s.res_tok, (size_t) rrows * 4));
            KC(cudaHostAlloc((void**) &s.res_ey_h, (size_t) rrows * N * 4, cudaHostAllocDefault));
            s.rr_sel_h.resize((size_t) rrows);
            // G1 PF_PARTIAL: the partial ring + its device copy + the selection map.  Allocated only under the
            // knob, so the default path's memory footprint is untouched as well as its bytes.
            if (pf_partial_on()) {
                const int64_t pc = Scratch::kTmax;
                KC(cudaHostAlloc((void**) &s.act1.p_h[0], (size_t) pc * N * 4, cudaHostAllocDefault));
                KC(cudaHostAlloc((void**) &s.act1.p_h[1], (size_t) pc * N * 4, cudaHostAllocDefault));
                KC(cudaHostAlloc((void**) &s.act1.wts_h, (size_t) pc * K * 4, cudaHostAllocDefault));
                KC(cudaMalloc((void**) &s.act1.p_m, (size_t) pc * N * 4));
                KC(cudaMalloc((void**) &s.act1.rr_sel_m, (size_t) rrows * 4));
                KC(cudaEventCreateWithFlags(&s.act1.p_done[0], cudaEventDisableTiming));
                KC(cudaEventCreateWithFlags(&s.act1.p_done[1], cudaEventDisableTiming));
            }
            s.res_tok_h.resize((size_t) rrows);
            const int nb = Scratch::kResBlkMax;
            KC(cudaMalloc((void**) &s.pa_g, (size_t) nb * 8)); KC(cudaMalloc((void**) &s.pa_u, (size_t) nb * 8));
            KC(cudaMalloc((void**) &s.pa_d, (size_t) nb * 8)); KC(cudaMalloc((void**) &s.pb_x, (size_t) nb * 8));
            KC(cudaMalloc((void**) &s.pb_h, (size_t) nb * 8));
            KC(cudaMalloc((void**) &s.pc_g, (size_t) nb * 8)); KC(cudaMalloc((void**) &s.pc_u, (size_t) nb * 8));
            KC(cudaMalloc((void**) &s.pc_ey, (size_t) nb * 8));
            s.pa_g_h.resize(nb); s.pa_u_h.resize(nb); s.pa_d_h.resize(nb);
            s.pb_x_h.resize(nb); s.pb_h_h.resize(nb);
            s.pc_g_h.resize(nb); s.pc_u_h.resize(nb); s.pc_ey_h.resize(nb);
        }
    }
    s.logits = dalloc(g.n_vocab);
    if (dec_res_on()) {   // K29: the residency path's own fp16 scratch (see the Scratch declarations)
        s.dec_x16 = (__half*) dalloc((size_t) g.n_embd * 2);
        s.dec_eh16 = (__half*) dalloc((size_t) 16 * g.n_ff * 2);   // one FF block per resident pick
    }
    if (dense_f16_on()) {   // S6: the decode's fp16 activation buffers (one dense vector + the attention output)
        s.dec_h16 = (__half*) dalloc((size_t) g.n_embd * 2);
        s.dec_att16 = (__half*) dalloc((size_t) g.n_head * g.head_dim * 2);
    }
    KC(cudaMalloc(&s.ids, K * sizeof(int)));
    // **THE FULL-ATTENTION KV IS THE 32K DECODE'S LARGEST PCIe TERM.**  kc/vc are pinned HOST buffers, so every decode
    // token pulls the whole visible window across PCIe: the 11 full layers read 2 x 33024 x 4 kv heads x 128 x 2 B =
    // 744 MB/token, ~62 ms at 12 GB/s - against a measured 67 ms/token (14.9 tok/s) for the whole token.  Under UVA a
    // kernel dereferences either kind of pointer and cudaMemset clears either, and nothing on the host reads these
    // buffers (every use is a device kernel or the clear), so moving ONLY the full layers to VRAM changes the address
    // and nothing else.  Cost: 709.5 MiB of the ~4.5 GiB free.  The sliding layers keep their host ring (513 rows,
    // 26 MB in total).  STRATA_KOLIBRI_KV_DEV=1, default off until measured.
    const bool kv_dev = env_on("STRATA_KOLIBRI_KV_DEV");
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const bool sliding = strata::core::is_swa_layer(g, l);
        const int64_t rows = sliding ? g.n_swa : s.ctx;
        const size_t elems = (size_t) g.n_head_kv * (size_t) rows * (size_t) g.head_dim;   // halves
        if (kv_dev && !sliding) {
            void* pk = nullptr;
            void* pv = nullptr;
            KC(cudaMalloc(&pk, elems * 2));
            KC(cudaMalloc(&pv, elems * 2));
            s.kc.push_back((__half*) pk);
            s.vc.push_back((__half*) pv);
        } else {
            s.kc.push_back(halloc(elems));
            s.vc.push_back(halloc(elems));
        }
        if (attn_kt_on() && sliding) {
            // SLIDING LAYERS ONLY, and measured: mirroring the full layers too is 25.3/25.4 tok/s against 25.7/25.7
            // here, because their stride is min(ctx, two_pass_max) = 11000 halves (22 KB between successive d's, so
            // every step of the dot lands in a new line), where the sliding window's 513 rows are 1 KB apart.  The
            // mirror is also 6x smaller this way (26 MB against 150 MB).
            const int64_t kt_rows = rows;
            void* pkt = nullptr;
            void* pvt = nullptr;
            KC(cudaMalloc(&pkt, (size_t) g.n_head_kv * (size_t) kt_rows * (size_t) g.head_dim * 2));
            KC(cudaMalloc(&pvt, (size_t) g.n_head_kv * (size_t) kt_rows * (size_t) g.head_dim * 2));
            s.kt_rows.push_back(kt_rows);
            s.kt_ready.push_back(0);
            s.kt.push_back((__half*) pkt);   // the transposed mirrors: [kvh][d][slot]
            s.vt.push_back((__half*) pvt);
        } else {
            s.kt_rows.push_back(0);
            s.kt_ready.push_back(0);
            s.kt.push_back(nullptr);
            s.vt.push_back(nullptr);
        }
    }
}

/// Per-layer CUDA graphs (--grouped + graphs): a grouped layer's decode step is fully device-side (position and
/// token id from device pointers, expert blobs from the pinned table over UVA, two-pass attention so the launch
/// carries no position-dependent shared memory) - one capture per layer, replayed per token.  The Q6_K-down
/// layers keep the mmvq loop (the blob pointer is host-computed) and are launched normally.  Warmup runs each
/// layer once (module load, allocations) before capture; the KV rows the warmup touches are pos 0, which the
/// first real token overwrites.
void capture_graphs(Engine& e, Scratch& s) {
    const int zero = 0;
    cudaMemcpy(s.pos_d, &zero, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(s.tok_d, &zero, 4, cudaMemcpyHostToDevice);
    e.gexec.assign((size_t) e.g.n_layers, 0);
    for (int64_t l = 0; l < e.g.n_layers; ++l)
        if (e.grouped > 0 && e.blobs[(size_t) l].native_ok)
            layer_step(e, s, l, true, g_st, nullptr);
    cudaDeviceSynchronize();
    for (int64_t l = 0; l < e.g.n_layers; ++l) {
        if (!(e.grouped > 0 && e.blobs[(size_t) l].native_ok)) continue;
        cudaGraph_t gr = nullptr; cudaGraphExec_t ex = nullptr;
        KC(cudaStreamBeginCapture(g_st, cudaStreamCaptureModeRelaxed));
        layer_step(e, s, l, true, g_st, nullptr);
        KC(cudaStreamEndCapture(g_st, &gr));
        KC(cudaGraphInstantiate(&ex, gr, 0));
        cudaGraphDestroy(gr);
        e.gexec[(size_t) l] = (unsigned long long) ex;
    }
    cudaDeviceSynchronize();
    s.pos = 0;
    cudaMemcpy(s.pos_d, &zero, 4, cudaMemcpyHostToDevice);
    std::printf("kolibri: %d / %d layers captured as CUDA graphs (the rest use the mmvq loop)\n",
                (int) (std::count(e.gexec.begin(), e.gexec.end(), 0ull) == 0 ? e.g.n_layers
                    : e.g.n_layers - std::count(e.gexec.begin(), e.gexec.end(), 0ull)), (int) e.g.n_layers);
    std::fflush(stdout);
}

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

/// Drop the held state: position 0 and empty KV caches (the serve protocol's fresh-request path).
void reset_state(Engine& e, Scratch& s) {
    const Clock::time_point t0 = Clock::now();
    std::fill(s.kt_ready.begin(), s.kt_ready.end(), (char) 0);   // K27b: the mirrors describe the old KV
    // K11: the old clear memset the WHOLE cache (1.35 GB at ctx 33024 = 4 ms per request).  Attention only
    // reads rows below the position counter, so a fresh request only needs the rows it can reach: full-attention
    // layers clear 0..pos, and the sliding layers keep their full (cheap, 513-row) ring clear.
    const int64_t used = (int64_t) s.pos + 1;   // rows the previous request touched (pos is 0-based)
    s.pos = 0;
    for (size_t l = 0; l < s.kc.size(); ++l) {
        const bool sliding = strata::core::is_swa_layer(e.g, (int64_t) l);
        const int64_t rows = sliding ? e.g.n_swa : s.ctx;
        const int64_t clear_rows = sliding ? rows : std::min<int64_t>(rows, used);
        const size_t bytes = (size_t) e.g.n_head_kv * clear_rows * e.g.head_dim * 2;   // FP16 KV (S3)
        KC(cudaMemset(s.kc[l], 0, bytes));
        KC(cudaMemset(s.vc[l], 0, bytes));
    }
    // cudaMemset is asynchronous: without this the timer would measure the enqueue, not the clear (K11).
    if (e.counters) { cudaDeviceSynchronize(); e.cnt.reset_ms += (long long) ms_since(t0); ++e.cnt.reset_n; }
}

void print_counters(Engine& e, const char* tag) {
    if (!e.counters) return;
    const auto& c = e.cnt;
    const double tok = c.tok > 0 ? (double) c.tok : 1.0;
    std::fprintf(stderr,
                 "CNT %s tok=%lld | slot hits=%lld miss=%lld (%.1f%% hit) expert PCIe=%.1f MB/tok | "
                 "d2h=%lld/%lld B (%.1f MB/tok) h2d=%lld/%lld B (%.1f MB/tok) | reset=%lld ms / %lld (%.2f ms)\n",
                 tag, c.tok, c.slot_hits, c.slot_miss, 100.0 * c.slot_hits / (double) (c.slot_hits + c.slot_miss),
                 c.slot_bytes / tok / 1e6, c.d2h_n, c.d2h_bytes, c.d2h_bytes / tok / 1e6,
                 c.h2d_n, c.h2d_bytes, c.h2d_bytes / tok / 1e6, c.reset_ms, c.reset_n,
                 c.reset_n ? (double) c.reset_ms / c.reset_n : 0.0);
    // The pool's OWN per-phase counters (pool.hpp:188-189).  The engine's d2h/h2d counters did not cover the
    // pool-only branch at all, which is why the P1b "pool" legs were unreadable from CNT alone - the only
    // in-situ record of what the pool actually streamed, and at what rate.
    // S7: the prefill pool leg's own line - bytes, achieved rate, and ms per chunk, which is what says whether
    // a slow pp is the pool, the GPU, or the seam between them.
    if (e.pf_stage_chunks > 0) {
        std::fprintf(stderr, "CNT prefill staged: chunks=%lld staged=%.2f GB (%.0f MB/chunk)\n", e.pf_stage_chunks,
                     e.pf_stage_bytes / 1e9, e.pf_stage_bytes / 1e6 / e.pf_stage_chunks);
    }
    if (e.cnt.pf_layers > 0) {
        const double n = (double) e.cnt.pf_layers;
        std::fprintf(stderr, "CNT prefill phases: dense=%.1f attn=%.1f experts=%.1f pool=%.1f post=%.1f shx=%.1f"
                             " tail=%.1f ms (per layer)\n",
                     e.cnt.pf_dense_ns / 1e6 / n, e.cnt.pf_attn_ns / 1e6 / n, e.cnt.pf_exp_ns / 1e6 / n,
                     e.cnt.pf_other_ns / 1e6 / n, e.cnt.pf_post_ns / 1e6 / n, e.cnt.pf_shx_ns / 1e6 / n,
                     e.cnt.pf_tail_ns / 1e6 / n);
    }
    if (e.cnt.attn_sc_ns > 0)
        std::fprintf(stderr, "CNT prefill attn: scores=%.0f ms softmax=%.0f ms pv=%.0f ms\n",
                     e.cnt.attn_sc_ns / 1e6, e.cnt.attn_sm_ns / 1e6, e.cnt.attn_pv_ns / 1e6);
    if (e.pf_cov_chunks > 0) {
        std::fprintf(stderr, "CNT prefill slot coverage: S=%d chunks=%lld entries=%lld hits=%.1f%% uniq/chunk=%.0f\n",
                     e.slots, e.pf_cov_chunks, e.pf_entry_n, 100.0 * e.pf_hit_entries / (double) e.pf_entry_n,
                     (double) e.pf_uniq_sum / e.pf_cov_chunks);
    }
    if (e.hist_on && !e.touch_hist.empty()) {
        // Top-K coverage per layer: the fp16 VRAM cache's hit-rate curve.  16 GB / 7.86 MB = ~2036 experts =
        // ~40/layer.  Uniform router touches give coverage ~K/n_expert; skew above that line is what makes the
        // cache pay.
        const int NE = (int) e.touch_hist[0].size();
        const int ks[] = {16, 24, 32, 40, 48, 64, 75, 80, 96, 128, 192, 256, 384};   // C0: 80 and 384 added; 75 = the shipped slot count
        std::fprintf(stderr, "CNT expert touch coverage (per-layer top-K, %d layers, %d experts):\n",
                     (int) e.touch_hist.size(), NE);
        for (int K : ks) {
            if (K > NE) break;
            double cov = 0;
            for (auto& h : e.touch_hist) {
                std::vector<uint64_t> s(h);
                std::partial_sort(s.begin(), s.begin() + K, s.end(), std::greater<uint64_t>());
                uint64_t tot = 0, top = 0;
                for (int i = 0; i < NE; ++i) tot += h[(size_t) i];
                for (int i = 0; i < K; ++i) top += s[(size_t) i];
                cov += (double) top / (double) tot;
            }
            cov /= (double) e.touch_hist.size();
            std::fprintf(stderr, "  top-%d: %.1f%% covered (uniform would be %.1f%%)\n", K, 100.0 * cov,
                         100.0 * K / NE);
        }
        // C0 also wants the PER-LAYER spread: if the best s differs a lot between layers, per-layer slot allocation is a
        // second-order win at zero VRAM, and a single averaged number would hide it.
        for (int K : {24, 80}) {
            if (K > NE) continue;
            std::vector<double> per;
            per.reserve(e.touch_hist.size());
            for (auto& h : e.touch_hist) {
                std::vector<uint64_t> t(h);
                std::partial_sort(t.begin(), t.begin() + K, t.end(), std::greater<uint64_t>());
                uint64_t tot = 0, top = 0;
                for (int i = 0; i < NE; ++i) tot += h[(size_t) i];
                for (int i = 0; i < K; ++i) top += t[(size_t) i];
                per.push_back(tot ? (double) top / (double) tot : 0.0);
            }
            std::sort(per.begin(), per.end());
            const double med = per[per.size() / 2];
            std::fprintf(stderr, "  top-%d per-layer spread: min %.1f%% median %.1f%% max %.1f%% (n=%d)\n",
                         K, 100.0 * per.front(), 100.0 * med, 100.0 * per.back(), (int) per.size());
        }
        // **0.5 HEAD-TO-HEAD (dynamic-expert-cache plan).**  The per-layer question in one pair of numbers: the SAME
        // VRAM budget spent two ways on the same touches - one global set of S experts (what --expert-slots-hot
        // learns today: `hot_cnt` has no layer dimension, so the same 75 experts sit in every layer) against S chosen
        // per layer.  S = the shipped slot count, so the two arms buy the same number of slots; only the shape differs.
        // The global set's coverage cannot be read off the per-layer top-K lines above, which is why it is computed
        // here: a global set's picks in a layer are not that layer's top-K.
        if (e.slots > 0) {
            const int S = std::min((int) e.slots, NE);
            std::vector<uint64_t> tot((size_t) NE, 0);
            uint64_t all = 0;
            for (auto& h : e.touch_hist)
                for (int i = 0; i < NE; ++i) { tot[(size_t) i] += h[(size_t) i]; all += h[(size_t) i]; }
            std::vector<uint64_t> gs(tot);
            std::partial_sort(gs.begin(), gs.begin() + S, gs.end(), std::greater<uint64_t>());
            uint64_t gtop = 0;
            for (int i = 0; i < S; ++i) gtop += gs[(size_t) i];
            double layered = 0;
            for (auto& h : e.touch_hist) {
                std::vector<uint64_t> t(h);
                std::partial_sort(t.begin(), t.begin() + S, t.end(), std::greater<uint64_t>());
                uint64_t tt = 0, tp = 0;
                for (int i = 0; i < NE; ++i) tt += h[(size_t) i];
                for (int i = 0; i < S; ++i) tp += t[(size_t) i];
                if (tt > 0) layered += (double) tp / (double) tt;
            }
            layered /= (double) e.touch_hist.size();
            std::fprintf(stderr,
                         "CNT expert touch %d-slot head-to-head: ONE GLOBAL SET of %d experts covers %.1f%%; %d per"
                         " layer covers %.1f%% (same VRAM, %d touches over %d layers)\n",
                         e.slots, S, all > 0 ? 100.0 * (double) gtop / (double) all : 0.0, S, 100.0 * layered,
                         (long long) all, (int) e.touch_hist.size());
        }
    }
    if (e.pf_chunks > 0) {
        const double ms = (double) e.pf_pool_ms, gb = e.pf_pool_bytes / 1e9;
        std::fprintf(stderr,
                     "CNT prefill pool: chunks=%lld experts/chunk=%.0f bytes=%.2f GB (%.0f MB/chunk) pool=%.0f ms"
                     " (%.1f ms/chunk, %.1f GB/s)%s\n",
                     e.pf_chunks, (double) e.pf_experts / e.pf_chunks, gb, gb * 1e3 / e.pf_chunks, ms, ms / e.pf_chunks,
                     ms > 0 ? gb / (ms / 1000.0) : 0.0,
                     [&] {
                         if (e.pf_f16_promotes <= 0) return std::string();
                         char buf[96];
                         std::snprintf(buf, sizeof buf, " promotes=%lld fill=%.2f GB",
                                       e.pf_f16_promotes, double(e.pf_fill_bytes) / 1e9);
                         return std::string(buf);
                     }().c_str());
        // The leg's honest rate: rows and GFLOP/s against the CPU pool's measured roof (226 GFLOP/s).  Every
        // earlier reading of this leg quoted GB/s, which measures the bytes it streamed rather than the arithmetic
        // it did - and the arithmetic is what binds once the fp16 cache is doing its job.
        {
            const double rows = (double) e.pf_rows.load();
            const double flop_per_row =
                2.0 * ((double) e.g.n_embd * (2.0 * (double) e.g.n_ff) + (double) e.g.n_ff * (double) e.g.n_embd);
            const double gflops = ms > 0 ? rows * flop_per_row / (ms * 1e6) : 0.0;
            std::fprintf(stderr,
                         "CNT prefill pool work: rows=%.0f (%.1f/chunk) = %.1f GFLOP at %.0f GFLOP/s (%.0f%% of the"
                         " 226 GFLOP/s pool roof); %.1f MB/row streamed\n",
                         rows, rows / (double) e.pf_chunks, rows * flop_per_row / 1e9, gflops,
                         gflops / 226.0 * 100.0, rows > 0 ? gb * 1e3 / rows : 0.0);
        }
        int active = 0; long long mx = 0, mn = -1;
        for (long long v : e.pf_shard_bytes) if (v > 0) { ++active; mx = std::max(mx, v); if (mn < 0) mn = v; }
        std::fprintf(stderr, "CNT prefill shards: workers=%d active=%d bytes/min=%.2f GB max=%.2f GB\n",
                     e.cpu_threads, active, mn / 1e9, mx / 1e9);
        long long bmax = 0, bmin = -1, bsum = 0; int bact = 0;
        for (long long v : e.pf_shard_ns) if (v > 0) { ++bact; bmax = std::max(bmax, v); if (bmin < 0) bmin = v; bsum += v; }
        if (bact > 0) {
            const double win = (double) e.pf_pool_ms * 1e6;   // summed pool windows across chunks
            std::fprintf(stderr, "CNT prefill shard-busy: max=%.0f ms min=%.0f ms mean=%.0f ms  window=%.0f ms -> "
                                 "barrier idle = window - max = %.0f ms (%.0f%% of window); spread max/min=%.1fx\n",
                         bmax / 1e6, bmin / 1e6, bsum / (double) bact / 1e6, win / 1e6, (win - bmax) / 1e6,
                         win > 0 ? 100.0 * (win - bmax) / win : 0.0, bmin > 0 ? (double) bmax / bmin : 0.0);
        }
        const double gu = e.pf_gu_ns / 1e6, qh = e.pf_qh_ns / 1e6, dn = e.pf_dn_ns / 1e6;
        const double ptot = gu + qh + dn;
        std::fprintf(stderr, "CNT prefill phases: gu=%.0f ms (%.0f%%) quant_h=%.0f ms (%.0f%%) down=%.0f ms (%.0f%%)"
                             " sum=%.0f ms of pool=%.0f ms (%.0f%% outside the batch loop)\n",
                     gu, ptot > 0 ? 100.0 * gu / ptot : 0.0, qh, ptot > 0 ? 100.0 * qh / ptot : 0.0,
                     dn, ptot > 0 ? 100.0 * dn / ptot : 0.0, ptot, ms, ms > 0 ? 100.0 * ptot / ms : 0.0);
    }
    if (e.epool) {
        const double ms_gu = e.epool->ms_multi_gu, ms_q = e.epool->ms_multi_q, ms_dn = e.epool->ms_multi_down;
        const double ms = ms_gu + ms_q + ms_dn;
        std::fprintf(stderr,
                     "CNT pool workers=%d | streamed %.1f MB/tok (%.2f GB total) | gu %.0f ms q %.0f ms"
                     " down %.0f ms = %.0f ms (%.2f ms/tok, %.1f GB/s)\n",
                     e.epool->workers(), c.tok ? e.epool->multi_bytes / tok / 1e6 : 0.0,
                     e.epool->multi_bytes / 1e9, ms_gu, ms_q, ms_dn, ms, ms / tok,
                     ms > 0 ? e.epool->multi_bytes / 1e9 / (ms / 1000.0) : 0.0);
        // Where run() spends its time (pool.hpp:202).  Three waits per layer - every worker parked, the drain,
        // the re-park.  Without the split a slow pool is ambiguous: slow at the WORK is a kernel problem, slow at
        // the SYNCHRONISATION is a barrier problem, and the fixes are opposites.  The decode token at 22.7 tok/s
        // has ~17 ms/token unaccounted for by pool work (13.4) + GPU non-expert (~10) + result H2D (3.5); this
        // says how much of that is the per-layer barrier.
        double wp = 0, dr = 0, rp = 0;
        e.epool->phase_ms(wp, dr, rp);
        std::fprintf(stderr, "CNT pool phases: wait_park %.0f ms drain %.0f ms repark %.0f ms = %.0f ms"
                             " (%.2f ms/tok) vs work %.2f ms/tok\n",
                     wp, dr, rp, wp + dr + rp, (wp + dr + rp) / tok, ms / tok);
    }
    std::fflush(stderr);
    if (e.cnt.dec_layers > 0) {
        const double n = (double) e.cnt.dec_layers;
        std::fprintf(stderr, "CNT decode phases: pre=%.1f ms d2h=%.1f ms pool=%.1f ms post=%.1f ms (per layer)\n",
                     e.cnt.dec_pre_ns / 1e6 / n, e.cnt.dec_d2h_ns / 1e6 / n, e.cnt.dec_pool_ns / 1e6 / n,
                     e.cnt.dec_post_ns / 1e6 / n);
    }
    if (e.cnt.br_pool + e.cnt.br_gpu + e.cnt.br_slots + e.cnt.br_grouped > 0)
        std::fprintf(stderr, "CNT decode residency: %lld of %lld picks served from the fp16 cache (%.1f%%)\n",
                     e.cnt.dec_res_hits, e.cnt.dec_res_picks,
                     e.cnt.dec_res_picks ? 100.0 * (double) e.cnt.dec_res_hits / (double) e.cnt.dec_res_picks : 0.0);
        std::fprintf(stderr, "CNT branches per tok: pool %.1f gpu-expert %.1f slots-path %.1f grouped %.1f"
                             " (layers/token total %.1f of %lld)\n",
                     e.cnt.br_pool / tok, e.cnt.br_gpu / tok, e.cnt.br_slots / tok, e.cnt.br_grouped / tok,
                     (e.cnt.br_pool + e.cnt.br_gpu) / tok, e.g.n_layers);
}

// --------------------------------------------------------------------------- sampling (q36's, verbatim semantics)
struct Sampling {
    float temperature = 0.f, top_p = 1.f, min_p = 0.f, repeat = 1.f, freq = 0.f, present = 0.f;
    int top_k = 20, last_n = 0;
    unsigned long long seed = 0;
    bool needs_logits() const { return temperature > 0.f || repeat != 1.f || freq != 0.f || present != 0.f; }
};

int sample(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist, std::mt19937_64& rng) {
    if (s.last_n != 0 && (s.repeat != 1.f || s.freq != 0.f || s.present != 0.f)) {
        std::map<int, int> cnt;
        const size_t from = s.last_n < 0 || (size_t) s.last_n >= hist.size() ? 0 : hist.size() - (size_t) s.last_n;
        for (size_t i = from; i < hist.size(); ++i) ++cnt[hist[i]];
        for (const auto& [id, c] : cnt) {
            float& v = lg[(size_t) id];
            if (s.repeat != 1.f) v = v > 0 ? v / s.repeat : v * s.repeat;
            v -= (float) c * s.freq + s.present;
        }
    }
    if (s.temperature <= 0.f) return (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
    const int kk = std::max(1, std::min(s.top_k > 0 ? s.top_k : 64, (int) lg.size()));
    std::vector<int> idx(lg.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    idx.resize((size_t) kk);
    std::vector<double> p(idx.size());
    const double mx = lg[(size_t) idx[0]];
    double sum = 0;
    for (size_t i = 0; i < idx.size(); ++i) sum += (p[i] = std::exp((lg[(size_t) idx[i]] - mx) / s.temperature));
    size_t keep = idx.size();
    double cum = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (p[i] / sum < s.min_p * (p[0] / sum)) { keep = i; break; }
        cum += p[i] / sum;
        if (cum >= s.top_p) { keep = i + 1; break; }
    }
    keep = std::max<size_t>(1, keep);
    double tot = 0;
    for (size_t i = 0; i < keep; ++i) tot += p[i];
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(rng);
    for (size_t i = 0; i < keep; ++i) {
        if ((r -= p[i]) <= 0) return idx[i];
    }
    return idx[keep - 1];
}

std::vector<int32_t> parse_ids(const char* s) {
    std::vector<int32_t> v;
    while (*s) {
        while (*s == ',' || *s == ' ' || *s == '\n') ++s;
        if (!*s) break;
        char* e = nullptr;
        const long x = std::strtol(s, &e, 10);
        if (e == s) break;
        v.push_back((int32_t) x);
        s = e;
    }
    return v;
}

bool is_eos(int t) { return t == 127906; }   // kolibri1: im_end

// ===================== the expert cache: fitting it, and what lives in it (opt-in) ==============================
// Three things this engine used to leave to the operator, who had to know two constants by heart -- the per-slot
// stride (114.4 MiB here) and the KV cost per token (20 KB) -- and who got an OOM at load if the two over-committed
// the card.  The Qwen engine (src/program/generate.cpp) already fits its cache from the free VRAM; this is the same
// idea, on this engine's own allocation order:
//
//   --expert-slots auto       fit the slot count to the VRAM that is free once everything else is resident
//   --expert-slots-hot        choose WHICH experts fill the slots by routed-pick frequency instead of by id
//   --expert-slots-profile P  keep that learned set across restarts
//
// The reserve (--vram-reserve-mib, default 700 as in the Qwen engine) is deliberately NOT the last byte: the CUDA
// runtime, the driver's allocations and the prefill's transients need headroom, and a cache that fills the card
// exactly dies on the first long prompt.  With every knob off, none of this runs and the cache is allocated by id
// exactly as before.
static constexpr int kSlotsAuto = -1;
static inline bool poison_zero_slot();   // defined with the loader below; alloc_slots needs it first

// The KV of ONE context token in bytes: K and V for every full-attention layer.  The sliding layers keep a fixed
// ring (g.n_swa rows), so they cost nothing per token of context.  This is why --ctx and --expert-slots are one
// decision rather than two: at 20 KB/token, one 114 MiB slot is ~5,700 tokens of context.
static long long kv_bytes_per_token(const Engine& e) {
    long long n_full = 0;
    for (int64_t l = 0; l < e.g.n_layers; ++l)
        if (!strata::core::is_swa_layer(e.g, l)) ++n_full;
    return n_full * 2LL * (long long) e.g.n_head_kv * (long long) e.g.head_dim * 2LL;
}

static long long slots_sum_stride(const Engine& e) {
    long long t = 0;
    for (const auto& b : e.blobs) t += (long long) b.stride;
    return t;
}

// One expert's blob into one device slot, for every layer: n_layers memcpys of stride bytes, queued on the engine's
// stream.  Asynchronous on purpose: the copies are ordered behind every kernel already launched on g_st (which read
// the OLD occupants) and ahead of the next token's kernels, so a round neither drains the device nor waits per move.
static void slot_promote(Engine& e, int id, int slot) {
    for (auto& b : e.blobs)
        KC(cudaMemcpyAsync(b.dev + b.stride * (size_t) slot, b.table + b.stride * (size_t) id, b.stride,
                           cudaMemcpyHostToDevice, g_st));
}

// The same move for ONE layer: 1 queued copy of stride bytes (2.21 MB at this model).  The source is the PINNED host
// table (LayerBlobs::table), so the copy is a real async DMA rather than a staged one.
static void slot_promote_layer(Engine& e, int64_t l, int id, int slot) {
    LayerBlobs& b = e.blobs[(size_t) l];
    KC(cudaMemcpyAsync(b.dev + b.stride * (size_t) slot, b.table + b.stride * (size_t) id, b.stride,
                       cudaMemcpyHostToDevice, g_st));
}

// Write the learned hot set, via a temp file and a rename so a kill or a full disk leaves the previous profile intact.
// With --expert-slots-per-layer the file grows a layer column (`layer id count`, header v2); without it the file is
// exactly what it was (`id count`, header v1), so the default path's profile stays byte-identical too.
static void hot_profile_save(Engine& e) {
    if (!e.hot_on || e.hot_profile.empty() || e.slots <= 0) return;
    const int n_exp = (int) e.g.n_expert;
    const int n_lay = e.per_layer ? (int) e.g.n_layers : 1;
    const std::string tmp = e.hot_profile + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "w");
    if (f == nullptr) { std::fprintf(stderr, "strata-kolibri: cannot write profile %s\n", tmp.c_str()); return; }
    std::fprintf(f, "# strata-kolibri expert-cache profile v%d slots=%d layers=%d stride_sum=%lld\n",
                 e.per_layer ? 2 : 1, e.slots, (int) e.g.n_layers, slots_sum_stride(e));
    const size_t keep = (size_t) std::min<int>(e.slots, n_exp);
    size_t kept = 0;
    for (int li = 0; li < n_lay; ++li) {
        std::vector<int> order((size_t) n_exp);
        for (int i = 0; i < n_exp; ++i) order[(size_t) i] = i;
        const size_t row = e.per_layer ? (size_t) li * (size_t) n_exp : 0;   // the profile's row for this layer
        std::partial_sort(order.begin(), order.begin() + keep, order.end(), [&](int a, int b) {
            const long long ca = e.hot_cnt[row + (size_t) a], cb = e.hot_cnt[row + (size_t) b];
            if (ca != cb) return ca > cb;
            return a < b;
        });
        for (size_t i = 0; i < keep; ++i) {
            const int id = order[i];
            const long long c = e.hot_cnt[row + (size_t) id];
            if (c == 0) break;   // never picked: writing zeros teaches the next run nothing
            if (e.per_layer) std::fprintf(f, "%d %d %lld\n", li, id, c);
            else std::fprintf(f, "%d %lld\n", id, c);
            ++kept;
        }
    }
    std::fclose(f);
    if (std::rename(tmp.c_str(), e.hot_profile.c_str()) != 0)
        std::fprintf(stderr, "strata-kolibri: cannot replace profile %s\n", e.hot_profile.c_str());
    std::fprintf(stderr, "strata-kolibri: hot profile: %zu of %d slot(s) written to %s\n", kept, e.slots * n_lay,
                 e.hot_profile.c_str());
}

// Seed order from the profile, one list per layer: ids by descending saved count, trimmed to this run's slot count and
// topped up in id order by alloc_slots.  BOTH formats are read: a v1 file (`id count`) has no layer information, so it
// seeds every layer with the same global order - which is exactly what the single-set cache did with it, so an old
// profile keeps working and keeps meaning the same thing.
static std::vector<std::vector<int>> hot_profile_load(Engine& e) {
    const int n_exp = (int) e.g.n_expert;
    const int n_lay = e.per_layer ? (int) e.g.n_layers : 1;
    std::vector<std::vector<int>> seeds((size_t) n_lay);
    if (!e.hot_on || e.hot_profile.empty()) return seeds;
    std::FILE* f = std::fopen(e.hot_profile.c_str(), "r");
    if (f == nullptr) return seeds;
    std::vector<std::vector<int>> ids((size_t) n_lay);
    std::vector<char> seen((size_t) n_exp, 0);   // the global (knob-off) seed wants each id once
    char buf[512];
    int layered = 0, flat = 0;
    while (std::fgets(buf, sizeof buf, f) != nullptr) {
        if (buf[0] == '#' || buf[0] == '\n') continue;
        int a = -1, b = -1;
        long long c = 0;
        if (std::sscanf(buf, "%d %d %lld", &a, &b, &c) == 3) {   // v2: layer id count
            if (b < 0 || b >= n_exp) continue;
            if (e.per_layer) {
                if (a < 0 || a >= n_lay) continue;
                e.hot_cnt[(size_t) a * (size_t) n_exp + (size_t) b] = c;
                ids[(size_t) a].push_back(b);
            } else {   // knob off reading a per-layer file: the global histogram is the SUM over layers
                e.hot_cnt[(size_t) b] += c;
                if (!seen[(size_t) b]) { seen[(size_t) b] = 1; ids[0].push_back(b); }
            }
            ++layered;
        } else if (std::sscanf(buf, "%d %lld", &a, &c) == 2) {   // v1: id count, one global order for every layer
            if (a < 0 || a >= n_exp) continue;
            for (int l = 0; l < n_lay; ++l) {
                e.hot_cnt[(size_t) l * (size_t) n_exp + (size_t) a] = c;
                ids[(size_t) l].push_back(a);
            }
            ++flat;
        }
    }
    std::fclose(f);
    for (int l = 0; l < n_lay; ++l) {
        std::vector<int>& s = ids[(size_t) l];
        const size_t row = (size_t) l * (size_t) n_exp;
        std::stable_sort(s.begin(), s.end(), [&](int x, int y) {
            const long long cx = e.hot_cnt[row + (size_t) x], cy = e.hot_cnt[row + (size_t) y];
            if (cx != cy) return cx > cy;
            return x < y;
        });
        if ((int) s.size() > e.slots) s.resize((size_t) e.slots);
        seeds[(size_t) l] = s;
    }
    // Say which map was actually seeded: with the knob off there is ONE row, and claiming "per layer" there is the
    // exact distinction this work is about (found in review: the knob-off log said "per layer" and meant nothing of
    // the kind).
    // The seed size is per ROW: one number with the knob off, a range with it on (each layer's profile row can be
    // shorter than the slot count, which alloc_slots then tops up in id order).
    size_t smin = seeds.empty() ? 0 : seeds[0].size(), smax = smin;
    for (const auto& v : seeds) { smin = std::min(smin, v.size()); smax = std::max(smax, v.size()); }
    if (e.per_layer)
        std::fprintf(stderr, "strata-kolibri: hot profile: %zu-%zu resident expert(s) per layer seeded from %s -- one set"
                     " per layer (%d layered line(s), %d global line(s))\n", smin, smax, e.hot_profile.c_str(),
                     layered, flat);
    else
        std::fprintf(stderr, "strata-kolibri: hot profile: %zu resident expert(s) seeded from %s -- ONE GLOBAL set,"
                     " shared by every layer (%d layered line(s), %d global line(s))\n", smax, e.hot_profile.c_str(),
                     layered, flat);
    return seeds;
}

// Allocate the slot table and decide what starts in it.  Called AFTER init_scratch on purpose: the fit must see the
// KV and the activations already allocated, or it counts their room as free and OOMs on the first long prompt.
static void alloc_slots(Engine& e, int ctx) {
    if (e.slots == 0) return;                       // no cache: the pool / UVA path, untouched
    const int n_exp = (int) e.g.n_expert;
    const long long stride_sum = slots_sum_stride(e);
    if (e.slots == kSlotsAuto) {
        size_t fb = 0, tb = 0;
        KC(cudaMemGetInfo(&fb, &tb));
        const long long reserve = (long long) e.vram_reserve_mib << 20;
        const long long room = (long long) fb - reserve;
        int n = (stride_sum > 0 && room > stride_sum) ? (int) (room / stride_sum) - 1 : 0;  // + zero slot per layer
        if (n > n_exp) n = n_exp;
        if (n < 1) {
            const long long per_tok = std::max<long long>(1, kv_bytes_per_token(e));
            std::fprintf(stderr,
                         "strata-kolibri: --expert-slots auto: %.0f MiB free, %lld MiB per slot, %d MiB reserve:"
                         " not one slot fits.  At %.1f KB/token, cutting --ctx by %lld tokens buys one slot.\n",
                         (double) fb / (1 << 20), stride_sum >> 20, e.vram_reserve_mib, (double) per_tok / 1024.0,
                         (long long) ((stride_sum + per_tok - 1) / per_tok));
            std::exit(2);
        }
        e.slots = n;
        std::fprintf(stderr,
                     "strata-kolibri: expert cache fit: %.0f MiB free after dense+KV+scratch -> %d slots/layer"
                     " (%.2f GiB), %d MiB kept for the runtime, %.0f MiB left\n",
                     (double) fb / (1 << 20), e.slots, (double) (stride_sum * (e.slots + 1)) / (1 << 30),
                     e.vram_reserve_mib,
                     (double) ((long long) fb - reserve - stride_sum * (e.slots + 1)) / (1 << 20));
    }
    e.slots = std::min(e.slots, n_exp);
    // What starts resident: the profile's hot set if there is one, else experts 0..S-1 (the id-ordered cache).  Per
    // layer when --expert-slots-per-layer, one shared list otherwise (seeds[l] is then the same list for every l).
    const int n_lay = e.per_layer ? (int) e.g.n_layers : 1;
    std::vector<std::vector<int>> seeds((size_t) n_lay);
    if (e.hot_on) {
        e.hot_cnt.assign((size_t) (e.per_layer ? (int64_t) n_lay * n_exp : n_exp), 0);
        e.hot_score.assign(e.hot_cnt.size(), 0.0);
        seeds = hot_profile_load(e);
    }
    for (int l = 0; l < n_lay; ++l) {
        std::vector<int>& seed = seeds[(size_t) l];
        if (!e.hot_on || seed.empty()) {   // the historical seed, and also the top-up when a profile is short
            std::vector<char> taken((size_t) n_exp, 0);
            for (int id : seed) taken[(size_t) id] = 1;
            for (int id = 0; id < n_exp && (int) seed.size() < e.slots; ++id)
                if (!taken[(size_t) id]) seed.push_back(id);
        }
        seed.resize((size_t) e.slots);
    }
    e.slot_of.assign((size_t) (e.per_layer ? (int64_t) n_lay * n_exp : n_exp), -1);
    for (int l = 0; l < n_lay; ++l)
        for (int s = 0; s < e.slots; ++s)
            e.slot_of[(size_t) l * (size_t) n_exp + (size_t) seeds[(size_t) l][(size_t) s]] = s;
    const bool id_seed = !e.hot_on;
    for (int64_t l = 0; l < (int64_t) e.blobs.size(); ++l) {
        LayerBlobs& b = e.blobs[(size_t) l];
        KC(cudaMalloc((void**) &b.dev, b.stride * (size_t) (e.slots + 1)));
        if (id_seed) {
            KC(cudaMemcpy(b.dev, b.table, b.stride * (size_t) e.slots, cudaMemcpyHostToDevice));   // as before
        } else {
            const std::vector<int>& seed = seeds[(size_t) std::min<int64_t>(l, (int64_t) n_lay - 1)];
            for (int s = 0; s < e.slots; ++s)
                KC(cudaMemcpy(b.dev + b.stride * (size_t) s, b.table + b.stride * (size_t) seed[(size_t) s],
                              b.stride, cudaMemcpyHostToDevice));
        }
        if (poison_zero_slot()) KC(cudaMemset(b.dev + b.stride * (size_t) e.slots, 0xA5, b.stride));
        else KC(cudaMemset(b.dev + b.stride * (size_t) e.slots, 0, b.stride));
    }
    // The (ctx, slots) exchange, printed together: they draw on the same VRAM and the operator should see the rate
    // rather than discover it as an OOM.
    size_t fb = 0, tb = 0;
    const bool have = cudaMemGetInfo(&fb, &tb) == cudaSuccess;
    const long long per_tok = kv_bytes_per_token(e);
    std::fprintf(stderr, "strata-kolibri: expert cache: %d slots/layer = %.2f GiB VRAM\n", e.slots,
                 (double) e.slots * stride_sum / (1 << 30));
    if (per_tok > 0)
        std::fprintf(stderr,
                     "strata-kolibri: (ctx, slots) draw on the same VRAM: 1 slot = %lld MiB = %lld context tokens"
                     " at %.1f KB/token; ctx %d x %lld = %.2f GiB, %s%.0f MiB free (reserve %d MiB)\n",
                     stride_sum >> 20, (long long) (stride_sum / per_tok), (double) per_tok / 1024.0, ctx, per_tok,
                     (double) per_tok * ctx / (1 << 30), have ? "" : "", have ? (double) fb / (1 << 20) : -1.0,
                     e.vram_reserve_mib);
    if (have && (long long) fb < ((long long) e.vram_reserve_mib << 20))
        std::fprintf(stderr,
                     "strata-kolibri: WARNING: %.0f MiB free is below the %d MiB reserve: a long prompt can run"
                     " the card out.  %s\n", (double) fb / (1 << 20), e.vram_reserve_mib,
                     e.slots == kSlotsAuto ? "" : "Use --expert-slots auto to fit it, or cut --ctx.");
}

// Re-pick which experts are resident and move the blobs that changed.  Called BETWEEN requests, never during one:
// the copies are synchronous and the slot contents must not change under a running kernel.  Slots that stay
// resident keep their number, so a steady workload settles and the PCIe traffic goes to zero.
// **What residency does and does not preserve (measured, findings-log "0.3").**  Both device legs are bitwise: a pick
// served from a VRAM slot and the same pick served from the pinned table run the same `gemv` on the same bytes (proven
// with the CPU pool switched off: byte-identical logits).  The CPU POOL is not that leg - it quantizes the layer's
// hidden state to q8_1 (`native_quant_act`) where the device reads fp32 - so a map that moves a pick between the pool
// and a slot changes the values, and greedy decoding turns that into different generated ids within a few tokens.  That
// is true of the global map today and of the per-layer map below; it is why no gate here may claim "ids unchanged"
// across two different resident sets, only across two runs of the same one.
static void slots_reselect(Engine& e, const char* why = nullptr, int budget = -1) {
    if (!e.hot_on || e.slots <= 0) return;
    const int cap = budget >= 0 ? budget : e.hot_swaps;
    if (cap == 0) return;   // this pass is off (--expert-slots-swaps 0 / --expert-slots-swaps-final 0)
    long long picks = 0;
    for (long long c : e.hot_cnt) picks += c;
    if (picks - e.hot_picks_at_last < e.hot_min_picks) return;   // no churn on every token
    e.hot_picks_at_last = picks;
    const int n_exp = (int) e.g.n_expert;
    const int n_lay = e.per_layer ? (int) e.g.n_layers : 1;
    // The score the policy sorts on: the decayed array, or the lifetime counts when the cache was opened without one
    // (a build or a test path that never sized hot_score keeps the counters' behaviour rather than moving nothing).
    std::vector<double> from_cnt;
    const std::vector<double>* sc = &e.hot_score;
    if (e.hot_score.empty()) {
        from_cnt.assign(e.hot_cnt.begin(), e.hot_cnt.end());
        sc = &from_cnt;
    }
    // hot_round() plans with the scores as they stand, then decays them: the order is pinned by
    // tests/kolibri/hot_policy_test.cpp case I, and it is why this is one call and not two.
    std::vector<double> score = *sc;
    const std::vector<strata::kolibri::HotSwap> plan =
        strata::kolibri::hot_round(n_lay, n_exp, e.slots, score, e.slot_of.data(), cap, e.hot_min_gain,
                                   e.hot_decay_factor);
    if (!e.hot_score.empty()) e.hot_score.swap(score);   // the decay lands in the engine's array
    if (plan.empty()) return;
    long long moved_bytes = 0;
    double gain_sum = 0.0;
    for (const strata::kolibri::HotSwap& s : plan) {
        const size_t row = e.per_layer ? (size_t) s.layer * (size_t) n_exp : 0;
        if (s.victim >= 0) e.slot_of[row + (size_t) s.victim] = -1;   // the occupant stops being resident
        e.slot_of[row + (size_t) s.expert] = s.slot;
        moved_bytes += (long long) (e.per_layer ? e.blobs[(size_t) s.layer].stride : slots_sum_stride(e));
        gain_sum += s.gain;
    }
    // The copies are STREAM-ORDERED, not device-serialised.  Every kernel that could read the slots being overwritten
    // was launched on g_st before this point, so the queued copies run after it; the next token's kernels are queued
    // after the copies.  slot_of is a HOST table - changing it needs no device drain at all - and the previous version's
    // cudaDeviceSynchronize() plus one blocking cudaMemcpy per move was what turned a 96-move round into a 629 ms stall
    // (the 2,699-move round measured on 2026-10-27).  This is the Qwen3.8-Flash-Next engine's ordering: copies on the
    // engine's own stream, joined once.
    const auto t0 = Clock::now();
    for (const strata::kolibri::HotSwap& s : plan) {
        if (e.per_layer) slot_promote_layer(e, s.layer, s.expert, s.slot);
        else slot_promote(e, s.expert, s.slot);
    }
    KC(cudaStreamSynchronize(g_st));
    e.hot_promotions += (long long) plan.size();
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    std::fprintf(stderr, "strata-kolibri: hot slots%s: %zu expert(s) newly resident after %lld picks (mean gain"
                 " %.2f+), %.0f MiB over PCIe in %.1f ms, budget %d%s\n",
                 why != nullptr ? why : "", plan.size(), picks, gain_sum / (double) plan.size(),
                 (double) moved_bytes / (1 << 20), ms, cap, e.per_layer ? " (per-layer map)" : "");
}

// Mid-request re-selection, on a POSITION count instead of on a wall clock.  Called from the two boundaries the 0.2
// probe priced: after a prefill chunk, and after a decode token.  The gate is hot_due()'s: N positions since the last
// round AND the pick floor, so two runs of one prompt take their rounds at the same positions and produce the same ids
// (the clock could not: 3 s was a different number of positions on every box and every load).  No drain here either -
// slots_reselect's copies are queued on g_st, behind whatever was already launched there.
static void slots_tick(Engine& e, const char* site, long long pos) {
    if (!e.hot_on || e.slots <= 0 || e.hot_every <= 0) return;
    if (pos < e.hot_pos_at_last) e.hot_pos_at_last = pos;   // a request rewound the position: never a negative wait
    long long picks = 0;
    for (long long c : e.hot_cnt) picks += c;
    if (!strata::kolibri::hot_due(e.hot_every, pos, e.hot_pos_at_last, picks, e.hot_picks_at_last, e.hot_min_picks))
        return;
    e.hot_pos_at_last = pos;
    char why[96];
    std::snprintf(why, sizeof why, " [round: %s at position %lld]", site, pos);
    slots_reselect(e, why);
}

// **0.2 PROBE — what one expert-swap round costs, at the boundaries a round could use.**  Measurement only, and gated
// by `STRATA_KOLIBRI_DYN_PROBE=1`: the bytes land in a PRIVATE buffer, never in a slot, so a probe run cannot change a
// result and an un-gated run takes no branch here at all.  It answers the two questions the plan's decision gate asks,
// in the unit that decides them: does a round stay under 10% of a token once it can overlap, and does ONE COPY PER
// BATCH close the 10.2 -> 12.2 GB/s gap that one copy per expert measures?
//
// Sizes are the plan's 1/4/8/16 experts per round, one per boundary, cycled, so one run covers all four at both sites.
// A round moves N experts across all n_layers layers - the plan's unit: 1 expert = 50 x 2.21 MB = 114.4 MiB.
//   "single" issues one copy per (layer, expert);
//   "batch"  relies on consecutive expert blobs being contiguous in the pinned table and issues one copy per layer
//            of N x stride (`STRATA_KOLIBRI_DYN_PROBE_BATCH=1`).
// Each boundary prices both a SYNCHRONOUS round (inserted latency: drain, copy, drain) and a round launched on a side
// stream and joined at the NEXT boundary (the `adapt_stream` shape) - where the number that matters is not the copy's
// duration but the wait the join had to add, i.e. how much of it the previous boundary's work hid.
struct DynProbe {
    uint8_t* dst_sync = nullptr;   // separate buffers: the two rounds run on two streams and must not share a target
    uint8_t* dst_side = nullptr;
    size_t cap = 0;
    cudaStream_t st = nullptr;
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    int pending_n = 0;
    const char* pending_site = nullptr;
    double pending_launch = 0.0;
    int call = 0;
};
static DynProbe& dyn_probe() {
    static DynProbe p;
    return p;
}
static bool dyn_probe_on() {
    static const bool v = std::getenv("STRATA_KOLIBRI_DYN_PROBE") != nullptr;
    return v;
}
static double dyn_now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
// Enqueue one round of `n` experts, every layer, into `dst`.  Both shapes move the same bytes; only the number of
// cudaMemcpyAsync calls differs, which is the whole batching question.
static void dyn_probe_issue(const Engine& e, cudaStream_t st, uint8_t* dst, int n, bool batch, size_t stride_max) {
    for (const LayerBlobs& b : e.blobs) {
        if (batch)
            KC(cudaMemcpyAsync(dst, b.table, b.stride * (size_t) n, cudaMemcpyHostToDevice, st));
        else
            for (int k = 0; k < n; ++k)
                KC(cudaMemcpyAsync(dst + (size_t) k * stride_max, b.table + (size_t) k * b.stride, b.stride,
                                   cudaMemcpyHostToDevice, st));
    }
}
static void dyn_probe_init(DynProbe& p, size_t stride_max) {
    const size_t need = stride_max * 16;
    if (p.cap < need) {
        if (p.dst_sync != nullptr) KC(cudaFree(p.dst_sync));
        if (p.dst_side != nullptr) KC(cudaFree(p.dst_side));
        KC(cudaMalloc((void**) &p.dst_sync, need));
        KC(cudaMalloc((void**) &p.dst_side, need));
        p.cap = need;
    }
    if (p.st == nullptr) {
        KC(cudaStreamCreate(&p.st));
        KC(cudaEventCreate(&p.e0));
        KC(cudaEventCreate(&p.e1));
    }
}
// One boundary: join the round launched at the previous one, time a synchronous round, launch the next side-stream
// round.  `boundary_ms` is the wall the boundary itself took (a chunk, or a token), i.e. what a side-stream round had
// to hide behind.
static void dyn_probe_step(const Engine& e, const char* site, double boundary_ms) {
    if (!dyn_probe_on()) return;
    static const int sizes[4] = {1, 4, 8, 16};
    static const bool batch = [] {
        const char* v = std::getenv("STRATA_KOLIBRI_DYN_PROBE_BATCH");
        return v != nullptr && std::atoi(v) > 0;
    }();
    DynProbe& p = dyn_probe();
    size_t stride_max = 0;
    for (const LayerBlobs& b : e.blobs) stride_max = std::max(stride_max, b.stride);
    dyn_probe_init(p, stride_max);
    const double mb = (double) slots_sum_stride(e) / 1e6;   // one expert, all layers
    if (p.pending_n > 0) {   // the side-stream round from the previous boundary, joined here
        const double t_wait0 = dyn_now_ms();
        KC(cudaEventSynchronize(p.e1));
        const double wait = dyn_now_ms() - t_wait0;
        float gpu = 0.f;
        cudaEventElapsedTime(&gpu, p.e0, p.e1);
        const double window = dyn_now_ms() - p.pending_launch;   // boundary work available to hide behind
        std::fprintf(stderr,
                     "DYNPROBE site=%s n=%d mode=stream batch=%d bytes_mb=%.1f gpu_ms=%.1f window_ms=%.1f wait_ms=%.1f\n",
                     p.pending_site, p.pending_n, (int) batch, mb * p.pending_n, gpu, window, wait);
        p.pending_n = 0;
    }
    const int n = sizes[p.call++ % 4];
    KC(cudaStreamSynchronize(g_st));
    const double t0 = dyn_now_ms();
    dyn_probe_issue(e, g_st, p.dst_sync, n, batch, stride_max);
    KC(cudaStreamSynchronize(g_st));
    const double ms = dyn_now_ms() - t0;
    std::fprintf(stderr, "DYNPROBE site=%s n=%d mode=sync batch=%d bytes_mb=%.1f ms=%.1f gbs=%.1f boundary_ms=%.1f\n", site, n,
                 (int) batch, mb * n, ms, mb * n / ms, boundary_ms);
    KC(cudaEventRecord(p.e0, p.st));
    dyn_probe_issue(e, p.st, p.dst_side, n, batch, stride_max);
    KC(cudaEventRecord(p.e1, p.st));
    p.pending_n = n;
    p.pending_site = site;
    p.pending_launch = dyn_now_ms();
}
// Join the last round: the decode loop's final boundary has no successor to join it, and a measurement that is
// launched and never joined is a number nobody reads.
static void dyn_probe_finish(const Engine& e) {
    if (!dyn_probe_on()) return;
    DynProbe& p = dyn_probe();
    if (p.pending_n == 0) return;
    const double t_wait0 = dyn_now_ms();
    KC(cudaEventSynchronize(p.e1));
    const double wait = dyn_now_ms() - t_wait0;
    float gpu = 0.f;
    cudaEventElapsedTime(&gpu, p.e0, p.e1);
    std::fprintf(stderr, "DYNPROBE site=%s n=%d mode=stream tail=1 bytes_mb=%.1f gpu_ms=%.1f wait_ms=%.1f\n",
                 p.pending_site, p.pending_n, (double) slots_sum_stride(e) / 1e6 * p.pending_n, gpu, wait);
    p.pending_n = 0;
}

/// The serve protocol (serve/server.py's engine lines, q36's shape): READY, then per request GEN <max_new>
/// [key=value ...] <ids>; PP lines as the prompt is consumed; T <id> per token; DONE at the end.  STOP cancels
/// the running request.  A request whose ids extend the held prefix continues it (the serve `live` logic).
int serve(Engine& e, Scratch& s) {
    // INFO key=value before READY is what fills serve/server.py's engine.info (parser at :839), and /metrics
    // publishes that dict as its "engine" block - which is where the Monitor tab reads the expert figures.  The
    // Kolibri engine never printed one, so the tab had no slots and no hit-rate context to show.  Keys match the
    // names the server already understands (expert_slots / expert_cache_mib / vram_free_mib).
    {
        long long cache_bytes = 0;
        for (size_t li = 0; li < e.blobs.size(); ++li)
            cache_bytes += (long long) e.blobs[li].stride * (long long) (e.slots + 1);
        size_t fb = 0, tb = 0;
        const bool have = cudaMemGetInfo(&fb, &tb) == cudaSuccess;
        std::printf("INFO engine=kolibri-1 expert_slots=%lld expert_slots_per_layer=%d expert_cache_mib=%lld"
                    " expert_cache_full_mib=%lld vram_free_mib=%lld kv=device cpu_threads=%d prefill=%d slots=%d\n",
                    (long long) e.slots * (long long) e.g.n_layers, e.slots, cache_bytes >> 20, cache_bytes >> 20,
                    have ? (long long) (fb >> 20) : -1LL, e.cpu_threads, e.prefill, e.slots);
    }
    // Measurement only: an arm that is switched on must say so, so a benchmark can prove that the run it
    // scored was the arm it asked for (tools/kolibri/decode_bench.py --require-marker).  Default off, so the
    // output of the plain path is unchanged.
    if (dec_xexp_on())
        std::fprintf(stderr, "strata-kolibri: STRATA_KOLIBRI_DEC_XEXP active: the shared expert is enqueued "
                             "before the pool drain\n");
    std::printf("READY %d stop\n", s.ctx);
    std::fflush(stdout);
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
    std::atomic<bool> stop{false};
    DecodeCancelScope cancel_scope(&stop);
    std::thread reader([&] {
        std::string l;
        while (std::getline(std::cin, l)) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            if (l == "STOP") { stop.store(true); continue; }
            std::lock_guard<std::mutex> lk(mu);
            lines.push_back(l);
            cv.notify_one();
        }
        std::lock_guard<std::mutex> lk(mu);
        eof = true;
        cv.notify_one();
    });
    reader.detach();
    std::vector<int32_t> live;   // the tokens whose state the engine holds
    std::vector<float> lg((size_t) e.g.n_vocab);
    for (;;) {
        std::string line;
        {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&] { return eof || !lines.empty(); });
            if (lines.empty()) return 0;
            line = lines.front();
            lines.pop_front();
        }
        if (line.empty()) continue;
        if (line == "VRAM" || line.rfind("VRAM ", 0) == 0) {
            // The monitor's expert figures: serve/server.py's StrataEngine.vram() writes "VRAM[ <mib>]" and reads
            // expert_slots / vram_free_mib out of a "VRAM key=value ..." reply, which is what the web page shows as
            // the expert count.  This engine sizes its cache at load and has no elastic tier, so a requested
            // reserve is reported back rather than applied - saying so beats pretending to resize.
            long long cache_bytes = 0;
            for (size_t li = 0; li < e.blobs.size(); ++li)
                cache_bytes += (long long) e.blobs[li].stride * (long long) (e.slots + 1);
            size_t fb = 0, tb = 0;
            const bool have = cudaMemGetInfo(&fb, &tb) == cudaSuccess;
            std::printf("VRAM expert_slots=%lld expert_cache_mib=%lld expert_cache_full_mib=%lld vram_free_mib=%lld"
                        " elastic=0\n",
                        (long long) e.slots * (long long) e.g.n_layers, cache_bytes >> 20, cache_bytes >> 20,
                        have ? (long long) (fb >> 20) : -1LL);
            std::fflush(stdout);
            continue;
        }
        if (line.rfind("GEN ", 0) != 0) {
            std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...>\n");
            std::fflush(stdout);
            continue;
        }
        char* p = nullptr;
        const long long max_new = std::strtoll(line.c_str() + 4, &p, 10);
        Sampling smp;
        for (;;) {
            while (*p == ' ') ++p;
            const char* st0 = p;
            while (*p && *p != ' ') ++p;
            const std::string tk(st0, (size_t) (p - st0));
            const size_t eq = tk.find('=');
            if (tk.empty() || eq == std::string::npos) { p = const_cast<char*>(st0); break; }
            const std::string key = tk.substr(0, eq);
            const char* v = tk.c_str() + eq + 1;
            if (key == "temperature") smp.temperature = std::strtof(v, nullptr);
            else if (key == "top_p") smp.top_p = std::strtof(v, nullptr);
            else if (key == "top_k") smp.top_k = std::atoi(v);
            else if (key == "min_p") smp.min_p = std::strtof(v, nullptr);
            else if (key == "penalty_last_n") smp.last_n = std::atoi(v);
            else if (key == "penalty_repeat") smp.repeat = std::strtof(v, nullptr);
            else if (key == "penalty_freq") smp.freq = std::strtof(v, nullptr);
            else if (key == "penalty_present") smp.present = std::strtof(v, nullptr);
            else if (key == "seed") smp.seed = std::strtoull(v, nullptr, 10);
        }
        const std::vector<int32_t> ids = parse_ids(p);
        if (max_new < 1 || ids.empty()) {
            std::printf("ERR bad request: max_new or the ids\n");
            std::fflush(stdout);
            continue;
        }
        if ((int64_t) ids.size() + max_new > s.ctx) {
            std::printf("ERR prompt (%zu tokens) + max_new (%lld) exceeds the context (%d)\n", ids.size(), max_new, s.ctx);
            std::fflush(stdout);
            continue;
        }
        bool bad = false;
        for (int32_t t : ids) bad |= t < 0 || t >= (int32_t) e.g.n_vocab;
        if (bad) { std::printf("ERR a token id is outside the vocabulary\n"); std::fflush(stdout); continue; }
        stop.store(false);
        // the held state continues when the request extends it (the KV rings cannot step back)
        // The reusable prefix is what the KV actually HOLDS, not what the transcript claims.  The last
        // token the previous request produced was sampled and never fed, so it has no KV row: `live`
        // counts it (the decode loop pushes every produced token) and `s.pos` does not.  Reusing
        // live.size() wrote every reused token one row early, which read as the model answering as if
        // the new turn had never arrived (measured 2026-10-25, same 3,065-id prompt both ways:
        // cold [4037 622 3515 622] against reused [310 310 310 310] - the reused arm just re-emitted the
        // token it had written after the bare prefix).
        const size_t held = std::min<size_t>(live.size(), (size_t) s.pos);
        size_t reuse = 0;
        // STRATA_KOLIBRI_REUSE=0 forces the cold path (every request re-reads its whole prompt).  It exists so the
        // feature can be priced against itself on one binary instead of against an older one that crashed here.
        const bool reuse_on = [] {
            const char* v = std::getenv("STRATA_KOLIBRI_REUSE");
            return v == nullptr || std::atoi(v) != 0;
        }();
        // STRATA_KOLIBRI_REUSE_PREFIX=1 reuses the LONGEST COMMON PREFIX instead of demanding that the whole held
        // prefix match.  Measured reason: real clients re-render the conversation as TEXT, and a BPE merge across
        // the turn boundary rewrites the prompt's last token or two - through /v1/chat/completions, turn 2 agreed
        // with turn 1 on 57 of 59 ids, so the strict check reset and the turn read all 72 tokens (cache_n 0).
        // Rows 0..reuse-1 belong to the tokens that DO match, so reusing them is exactly as safe as the strict
        // case, and the rewind below re-derives the chunk the same way.  OPT-IN and off by default, because the
        // strict rule is the documented contract and the default path must stay byte-identical.
        const bool prefix_reuse = [] {
            const char* v = std::getenv("STRATA_KOLIBRI_REUSE_PREFIX");
            return v != nullptr && v[0] != 0 && std::atoi(v) != 0;
        }();
        // The prefill never feeds the last prompt token - the decode step does - so that is the most a request can
        // reuse however long the held prefix is.
        const size_t reuse_cap = ids.size() > 1 ? ids.size() - 1 : 0;
        size_t match = 0;
        if (reuse_on && held > 0 && ids.size() > 1) {
            if (prefix_reuse) {
                // Cap at reuse_cap rather than requiring `held < ids.size()`.  A real turn often re-renders to
                // FEWER tokens than the engine holds (the reply is truncated by max_tokens, and re-tokenizing it
                // gives a different count), and the strict guard then rejected a prefix that genuinely matched:
                // measured through /v1/chat/completions, the engine held 104 tokens while turn 2 was 78, so every
                // turn read its whole prompt.  The rows for the matched ids are valid either way.
                const size_t lim = std::min(held, reuse_cap);
                while (match < lim && live[match] == ids[match]) ++match;
            } else if (held < ids.size() && std::equal(live.begin(), live.begin() + held, ids.begin())) {
                match = held;
            }
        }
        if (match > 0) {
            reuse = match;
            // K27b rebuilds each layer's K/V transpose mirror ONCE per REQUEST, on the first decode token,
            // on the documented assumption that this "also covers everything a prefill wrote".  reset_state()
            // zeroes kt_ready for exactly that reason (see its K27b comment).  This path skips reset_state -
            // that is the whole point of reusing - so the flags survived from the previous request and the
            // first decode token appended ONE row (n_kt = 1, first_kt = s.pos) instead of rebuilding the
            // window, leaving the rows THIS request's prefill had just written absent from the mirror, so
            // attention read stale ones.  Measured 2026-10-25 on a 521-id prompt: cold [259 735 12416 46]
            // against reused [259 923 262 1077] on every one of three processes, with the reused arm's
            // logits 6.8e-01 away in relative terms - structural, not round-off.  Zeroing the flags costs a
            // full window copy once per layer per request, which is what a cold request already pays.
            std::fill(s.kt_ready.begin(), s.kt_ready.end(), (char) 0);
            // (1) Rewind to the start of the PREVIOUS request's last prefill chunk, so this request's first chunk
            // starts exactly where a cold read's chunk starts and every row is recomputed by a GEMM of the same
            // length.  Two caveats, stated because the gate does not cover them: the rule below keys the chunk
            // size on the prefix LENGTH, so above 8,192 tokens the resumed arm and a cold read can still pick
            // different chunk sizes (256 against 512) and this alignment does not make them match - unmeasured;
            // and the "same length" argument is about VALUES (the K/V bytes and the logits), not about ids, which
            // is what the K28 note above measured when it found the generated ids identical however the prompt is
            // cut.  Both statements are true at once; the id-level measurement simply cannot see the last bits.  Two boundaries matter: the row at the reuse boundary (a cold read computes it inside a
            // large prefill GEMM, a resumed read inherits the previous request's single-token DECODE result) and
            // the previous request's final chunk, which is SHORT because a prefill stops one token early - a
            // 3,071-row prefill cuts as 11x256 + 255, so a cold read would then compute positions 2816..3071 in
            // a 256-row chunk against the resumed arm's 255-row one.  Both differences are a few FP16 ulps and
            // both disappear when the rows are recomputed in a chunk of the same length.  STRATA_KOLIBRI_REUSE_ALIGN=0
            // turns it off for an A/B.  Cost: up to chunk-1 tokens read again.
            if (std::getenv("STRATA_KOLIBRI_REUSE_ALIGN") == nullptr ||
                std::atoi(std::getenv("STRATA_KOLIBRI_REUSE_ALIGN")) != 0) {
                const int64_t prev_end = s.pos;   // tokens the previous request consumed (read before we move it)
                g_pf_total = (int64_t) ids.size() - 1 - (int64_t) reuse;   // the rule keys on the prefix length
                const int64_t chunk = (int64_t) pf_chunk_at((int64_t) reuse);
                if (chunk > 1 && prev_end > 1) {
                    const int64_t aligned = ((prev_end - 1) / chunk) * chunk;
                    if (aligned > 0 && (size_t) aligned < reuse) reuse = (size_t) aligned;
                }
            }
            // (2) THE PREFILL WRITES AT s.pos, so s.pos must be the reused length before the prefill runs.  After
            // the previous request s.pos is one PAST the tokens it consumed (its last token went through the
            // decode path), so leaving it alone made this request write its chunk that much too far: with a
            // 513-id prefix the resumed prefill wrote 513..521 instead of 512..520, which left position 512
            // holding the previous request's decode-written K/V and wrote 521 twice.  That was the actual bug.
            s.pos = (int64_t) reuse;
        } else {
            reset_state(e, s);
            live.clear();
        }
        const auto t0 = Clock::now();
        const int64_t fresh = (int64_t) (ids.size() - reuse);   // a COUNT of new tokens - display only
        // `fresh` is a count and `reuse` is a position; the two must not meet in a loop bound.  They used
        // to: the plain loop ran `t < fresh`, so with a reuse of more than half the prompt it never ran at
        // all, and below that it ran a PARTIAL prefill that left holes in the KV - the worse failure of the
        // two, because it answers confidently instead of crashing (the 580-token case in the log read 0 of
        // its 64 new tokens and answered from KV that had never seen the new turn), and the chunked loop's
        // `n = fresh - 1 - reuse` went negative, which made
        // T = min(chunk, n) negative and (size_t)T * 4 wrap into a cudaMemcpy size the driver rejects
        // (:3604) - so an agent's second turn killed the engine outright.  With reuse == 0 every one of
        // these expressions already reduced to the cold-path value, which is why every benchmark ever run
        // through this path looked normal.  `stop` is the position to stop at: the prompt minus its last
        // token, which the decode step feeds.  (Named pf_end, not stop: `stop` is this request's
        // cancellation flag, and shadowing it here is a compile error the first time it is tried.)
        const int64_t pf_end = (int64_t) ids.size() - 1;
        // The contract serve/server.py already parses (:1934) and src/program/generate.cpp already keeps
        // (:11225): <n> prompt tokens are not read again.  Printed before the first PP line and only when
        // there are some, so a client that ignores it sees exactly today's stream.
        if (reuse > 0) {
            std::printf("RESUME %lld\n", (long long) reuse);
            std::fflush(stdout);
        }
        if (e.prefill > 0) {   // chunked prefill: 512 rows per pass, progress per chunk
            int64_t i = reuse;
            const int64_t n = pf_end - reuse;   // the tokens still to read (== ids.size()-1 when reuse is 0)
            e.hot_pos_at_last = s.pos;   // the request's own position count, so a reuse does not inherit the last one
            e.pf_total = n;   // the POOL_ALL rule: one decision for the whole prompt
        g_pf_total = n;   // K28: and the chunk rule keys on it (see pf_chunk_at)
            while (i < pf_end) {
                const int T = (int) std::min<int64_t>(pf_chunk_at(s.pos), pf_end - i);
                KC(cudaMemcpy(s.toks_d, ids.data() + i, (size_t) T * 4, cudaMemcpyHostToDevice));
                forward_multi(e, s, T, s.pos);
                s.pos += T;
                i += T;
                std::printf("PP %lld %lld %.0f %.1f\n", (long long) i, (long long) ids.size(), ms_since(t0),
                            (double) (i - reuse) / std::max(1e-9, ms_since(t0) / 1000.0));
                std::fflush(stdout);
                slots_tick(e, "chunk", (long long) i);   // a round every --expert-slots-reselect-every positions
            }
        } else
        for (int64_t t = (int64_t) reuse + 1; t < (int64_t) ids.size(); ++t) {   // the prompt minus its last token
            forward(e, s, ids[t - 1], nullptr);   // ONE token through the plain path: this line is the read
            std::printf("PP %lld %lld %.0f %.1f\n", (long long) t, (long long) ids.size(), ms_since(t0),
                        (double) (t - reuse) / std::max(1e-9, ms_since(t0) / 1000.0));
            std::fflush(stdout);
        }
        e.hot_pos_at_last = s.pos;   // the request's own position count: a reuse starts at `reuse`, not at 0
        // The prompt has now been routed through every layer, so one large round is worth more than any number of
        // 96-move rounds during the decode: this is the Qwen3.8-Flash-Next engine's `hybrid_swaps_after_prompt = 2048`
        // pass (generate.cpp:1821), run at the same place - after process(), before the decode loop.  Measured on
        // 2026-10-27: 1,719 admissible moves, 3,925 MiB, 394 ms, and the round's own budget bounds it.
        hot_absorb_prompt(e);   // the prompt routed every layer: its counts are what the round below decides on
        slots_reselect(e, " [after prompt]", e.hot_swaps_final);
        seed_hot(e, s);   // K29: the prompt's hot experts into the fp16 slots before decoding
        hot_forget_prompt(e);   // prompt-scoped: the next request counts its own
    if (e.graphs > 0) KC(cudaMemcpy(s.pos_d, &s.pos, 4, cudaMemcpyHostToDevice));
        // The held list must describe exactly the tokens this request consumed.  A reusing request already has
        // the prefix in `live`, so appending from `reuse` duplicated positions reuse..held-1 and left the list
        // describing a prompt that never existed: the NEXT request then failed its prefix check and reset, which
        // made reuse work every other turn (measured in an agent-shaped run: turns 2, 4, 6 reused while 3 and 5
        // read everything).  Truncating to `reuse` first makes the insert correct for every value of reuse,
        // including the rewind that the bitwise fix introduced, and 0 (where the else branch already cleared it).
        live.resize(reuse);
        live.insert(live.end(), ids.begin() + (long) reuse, ids.end());
        const double prompt_ms = ms_since(t0);
        // Per-request expert-cache figures for the DONE line: the counters are cumulative, so the request's own
        // numbers are the deltas.  slot hits are the routed picks served from VRAM slots, misses are the ones the
        // CPU pool computed from the pinned table - which is why offloaded stays 0: nothing went over PCIe.
        const long long hits0 = e.cnt.slot_hits, miss0 = e.cnt.slot_miss;
        std::mt19937_64 rng(smp.seed);
        std::vector<int32_t> hist = ids;
        const auto d0 = Clock::now();
        long long produced = 0;
        const char* finish = "length";
        int32_t tok = ids.back();   // the last prompt token: the first decode step runs it
        for (;;) {
            forward(e, s, tok, nullptr);
            if (decode_aborted) {
                reset_state(e, s);
                live.clear();
                finish = "cancel";
                break;
            }
            KC(cudaMemcpy(lg.data(), s.logits, lg.size() * 4, cudaMemcpyDeviceToHost));
            if (produced == 0 && reuse_gate_dump_path()) {
                // Measurement-only. The first decode step's logits ARE the logits for the last prompt position,
                // so a cold read of a prompt and a reused read that extends a prefix write the same position
                // here and the two records are directly comparable. Written as [reuse][fresh][nv] + floats,
                // appended, so a comparator can tell the arms apart by their own fields.
                if (std::FILE* lf = std::fopen(reuse_gate_dump_path(), "ab")) {
                    const int32_t hdr[3] = {(int32_t) reuse, (int32_t) fresh, (int32_t) lg.size()};
                    std::fwrite(hdr, 4, 3, lf);
                    std::fwrite(lg.data(), 4, lg.size(), lf);
                    std::fclose(lf);
                }
            }
            if (produced == 0 && reuse_gate_hidden_path()) {
                if (std::FILE* hf = std::fopen(reuse_gate_hidden_path(), "ab")) {
                    std::vector<float> hh((size_t) e.g.n_embd);
                    KC(cudaMemcpy(hh.data(), s.h, (size_t) e.g.n_embd * 4, cudaMemcpyDeviceToHost));
                    const int32_t hdr[3] = {(int32_t) reuse, (int32_t) fresh, (int32_t) hh.size()};
                    std::fwrite(hdr, 4, 3, hf);
                    std::fwrite(hh.data(), 4, hh.size(), hf);
                    std::fclose(hf);
                }
            }
            if (e.counters) { e.cnt.d2h_n += 1; e.cnt.d2h_bytes += (long long) lg.size() * 4; }
            const int32_t nx = smp.needs_logits() ? sample(lg, smp, hist, rng)
                                                  : (int32_t) (std::max_element(lg.begin(), lg.end()) - lg.begin());
            std::printf("T %d\n", nx);
            std::fflush(stdout);
            slots_tick(e, "token", (long long) s.pos);   // a round every --expert-slots-reselect-every positions
            ++produced;
            hist.push_back(nx);
            live.push_back(nx);
            if (is_eos(nx)) { finish = "stop"; break; }
            if (produced >= max_new) break;
            if (stop.load()) { finish = "cancel"; break; }
            tok = nx;
        }
        const double decode_ms = ms_since(d0);
        const long long dh = e.cnt.slot_hits - hits0, dm = e.cnt.slot_miss - miss0;
        // Field contract unchanged (16 fields, f[8]=reuse, f[14]=prompt_read): what changed is that f[9]=hits,
        // f[10]=lookups and f[11]=RAM blobs now carry the request's real numbers instead of literal zeros, so the
        // monitor can show a hit rate and the expert tiers instead of nulls.
        std::printf("DONE %lld %zu %.1f %.1f %s 0 0 %zu %lld %lld %lld 0 0.0 %lld 0\n", produced, ids.size(),
                    prompt_ms, decode_ms, finish, reuse, dh, dh + dm, dm, (long long) fresh);
        std::fflush(stdout);
        // Between requests is the default place to change what is resident: the promotions are synchronous and the
        // slot contents must not move under a running kernel.  slots_tick() is the same pass run mid-request, on a
        // timer and after a drain, between two decode tokens or two prefill chunks.
        slots_reselect(e, nullptr, e.hot_swaps_final);   // the after-request pass: the larger budget
        hot_profile_save(e);
        print_counters(e, "serve");
    }
}

// S5 (task-8, the poison test's control): fill the zero slot with a POISON pattern instead of zeros, so the
// zero-slot poison test can prove the poison reaches the arithmetic (a test with no teeth).  Debug-only: the
// env is unset in every production run, and the zero slot is read only by the overflow path.
static inline bool poison_zero_slot() {
    static const bool on = env_on("STRATA_KOLIBRI_POISON_ZERO_SLOT");
    return on;
}

/// Load: geometry + layout from the header; dense tensors to the GPU; routed experts repacked to pinned blob
/// tables.  Split files: every tensor lives in one shard; the shard's mmap pointer is the source.
bool load(const std::string& path, Engine& e, std::string& err) {
    const auto t0 = std::chrono::steady_clock::now();
    const strata::GgufModel gm = strata::GgufModel::open(path);
    if (!strata::core::geometry_from_gguf(gm.meta(), e.g, e.top_k, err)) return false;
    if (e.g.family != strata::ModelFamily::Kolibri1) { err = "strata-kolibri runs kolibri1 files only"; return false; }
    // S5 (task-8): the mixed-type expert check runs BEFORE the layout check, so a file whose
    // ffn_gate_exps.type != ffn_up_exps.type is always rejected with the type message (the layout check
    // would otherwise report a size error first for a header-only or truncated file, hiding the real cause).
    // The repack writes gate and up into ONE [gate|up] blob under a single layout, so a mixed-type file would
    // otherwise have its up rows parsed with gate's type - silent numeric corruption, not a failure.
    for (int64_t l = 0; l < e.g.n_layers; ++l) {
        const strata::TensorInfo *gate = nullptr, *up = nullptr;
        for (size_t s = 0; s < gm.size(); ++s)
            for (const strata::TensorInfo& ti : gm.shard(s).tensors()) {
                if (ti.name == L(l, "ffn_gate_exps.weight")) gate = &ti;
                if (ti.name == L(l, "ffn_up_exps.weight")) up = &ti;
            }
        if (gate && up && gate->type != up->type) {
            err = "layer " + std::to_string(l) + ": ffn_gate_exps type " + std::to_string((int) gate->type) +
                  " != ffn_up_exps type " + std::to_string((int) up->type) +
                  "; the [gate|up] blob has one layout, so a mixed-type expert file is rejected";
            return false;
        }
    }
    size_t checked = 0;
    if (!strata::core::check_gguf_layout(gm, e.g, err, &checked)) return false;
    const std::string p = "kolibri1.";
    if (const strata::MetaValue* v = gm.meta().get(p + "attention.layer_norm_rms_epsilon")) e.eps = (float) v->num();
    if (const strata::MetaValue* v = gm.meta().get(p + "rope.freq_base")) e.rope_base = (float) v->num();

    // pass 1: the routed experts, layer by layer, repacked into pinned [gate|up|down] blob tables
    e.blobs.resize((size_t) e.g.n_layers);
    // The prompt's own routing counts (per layer, per expert).  `--hist` prints them, the K29 fp16 seed reads them,
    // and the expert cache's post-prompt round needs them: without them a prompt feeds the hot set nothing, and the
    // first request of a cold start can only learn from its own decode.  Always allocated (50 x 384 x 8 = 150 KB).
    e.touch_hist.assign((size_t) e.g.n_layers, std::vector<uint64_t>((size_t) e.g.n_expert, 0));
    size_t expert_bytes = 0;
    for (int64_t l = 0; l < e.g.n_layers; ++l) {
        const strata::TensorInfo *gate = nullptr, *up = nullptr, *down = nullptr;
        const strata::GgufFile* gf = nullptr;
        for (size_t s = 0; s < gm.size(); ++s)
            for (const strata::TensorInfo& ti : gm.shard(s).tensors()) {
                if (ti.name == L(l, "ffn_gate_exps.weight")) { gate = &ti; gf = &gm.shard(s); }
                if (ti.name == L(l, "ffn_up_exps.weight")) up = &ti;
                if (ti.name == L(l, "ffn_down_exps.weight")) down = &ti;
            }
        if (!gate || !up || !down) { err = "layer " + std::to_string(l) + ": expert tensors missing"; return false; }
        // The repack writes gate and up into ONE gu blob under a single layout (b.L.gu_type, taken from gate), and
        // native_expert_layout is only told gate/down.  A file with gate->type != up->type would therefore have its
        // up rows parsed with gate's type - silent numeric corruption, not a failure.  Reject it at load.
        if (gate->type != up->type) {
            err = "layer " + std::to_string(l) + ": ffn_gate_exps type " + std::to_string((int) gate->type) +
                  " != ffn_up_exps type " + std::to_string((int) up->type) +
                  "; the [gate|up] blob has one layout, so a mixed-type expert file is rejected";
            return false;
        }
        LayerBlobs& b = e.blobs[(size_t) l];
        b.L = strata::kernels::native_expert_layout((int) gate->type, (int) down->type, e.g.n_embd, e.g.n_ff);
        b.native_ok = strata::kernels::native_expert_supported((int) gate->type, (int) down->type, e.g.n_embd, e.g.n_ff);
        {
            std::string ferr;
            // the CPU pool's fmt is needed for every layer (the pool covers the Q6_K-down layers the GPU
            // grouped kernel cannot); && would short-circuit on the GPU check and leave fmt unfilled
            const bool fmt_ok = strata::kernels::cpu::native_fmt((int) gate->type, (int) down->type,
                                                                 e.g.n_embd, e.g.n_ff, b.fmt, ferr);
            b.native_ok = b.native_ok && fmt_ok;
            b.pool_ok = fmt_ok;   // the pool's capability alone; see the LayerBlobs note
        }
        const size_t sb = lcm(lcm((size_t) 144, (size_t) 210), (size_t) 34);   // Q4_K / Q6_K / Q8_0 super-blocks
        b.stride = ((b.L.bytes + sb - 1) / sb) * sb;
        KC(cudaHostAlloc((void**) &b.table, b.stride * (size_t) (e.g.n_expert + 1), cudaHostAllocDefault));
        if (poison_zero_slot())
            std::memset(b.table + b.stride * (size_t) e.g.n_expert, 0xA5, b.stride);   // the poison control
        else
            std::memset(b.table + b.stride * (size_t) e.g.n_expert, 0, b.stride);  // the slots path's zero slot
        const size_t gu_rows = (size_t) e.g.n_ff, d_rows = (size_t) e.g.n_embd;
        const size_t gu_eb = gu_rows * b.L.gu_row, d_eb = d_rows * b.L.d_row;
        const uint8_t* gs = gf->tensor_data(*gate);
        // the up / down tensors may live in other shards: find their file pointers
        const uint8_t* us = nullptr;
        const uint8_t* ds = nullptr;
        for (size_t s = 0; s < gm.size(); ++s) {
            for (const strata::TensorInfo& ti : gm.shard(s).tensors()) {
                if (ti.name == L(l, "ffn_up_exps.weight")) us = gm.shard(s).tensor_data(ti);
                if (ti.name == L(l, "ffn_down_exps.weight")) ds = gm.shard(s).tensor_data(ti);
            }
        }
        for (int64_t ex = 0; ex < e.g.n_expert; ++ex) {
            uint8_t* blob = b.table + (size_t) ex * b.stride;
            std::memcpy(blob, gs + (size_t) ex * gu_eb, gu_rows * b.L.gu_row);
            std::memcpy(blob + b.L.up_off, us + (size_t) ex * gu_eb, gu_rows * b.L.gu_row);
            std::memcpy(blob + b.L.down_off, ds + (size_t) ex * d_eb, d_rows * b.L.d_row);
        }
        // S7 increment A: warm the fp16 expert cache.  Env-gated; gate-off leaves the plain path untouched.
        const int expert_f16 = expert_f16_n();
        if (expert_f16 > 0 && strata::kernels::dequant_bf16_supported(b.L.gu_type) &&
            strata::kernels::dequant_bf16_supported(b.L.d_type)) {
            const int nc = std::min<int>(expert_f16, (int) e.g.n_expert);
            const size_t gu_n = (size_t) e.g.n_ff * (size_t) e.g.n_embd, d_n = (size_t) e.g.n_embd * (size_t) e.g.n_ff;
            uint16_t* p = nullptr;
            KC(cudaMalloc((void**) &p, (size_t) nc * (2 * gu_n + d_n) * 2));
            for (int ex = 0; ex < nc; ++ex) {
                const uint8_t* blob = b.table + (size_t) ex * b.stride;
                uint16_t* out = p + (size_t) ex * (2 * gu_n + d_n);
                strata::kernels::dequant_f16(b.L.gu_type, blob, 0, e.g.n_ff, e.g.n_embd, out, g_st);
                strata::kernels::dequant_f16(b.L.gu_type, blob + b.L.up_off, 0, e.g.n_ff, e.g.n_embd, out + gu_n, g_st);
                strata::kernels::dequant_f16(b.L.d_type, blob + b.L.down_off, 0, e.g.n_embd, e.g.n_ff, out + 2 * gu_n, g_st);
            }
            b.f16 = p;
            b.n_cached = nc;
            b.f16_slot_id.assign((size_t) nc, -1);
            b.f16_slot_last.assign((size_t) nc, 0);
        }
        expert_bytes += b.stride * (size_t) e.g.n_expert;
    }

    // pass 2: the dense tensors to the GPU (MMVQ types as is; 1-D vectors as F32)
    size_t dense_bytes = 0;
    for (size_t s = 0; s < gm.size(); ++s) {
        const strata::GgufFile& f = gm.shard(s);
        for (const strata::TensorInfo& ti : f.tensors()) {
            if (ti.shape.size() == 3) continue;   // the routed experts: repacked above
            const uint64_t nb = strata::tensor_payload_bytes(ti);
            const uint8_t* src = f.tensor_data(ti);
            if (ti.shape.size() == 1) {
                if (ti.type != 0) { err = ti.name + ": a 1-D tensor of type " + ti.type_name(); return false; }
                float* d = nullptr;
                KC(cudaMalloc(&d, nb));
                KC(cudaMemcpy(d, src, nb, cudaMemcpyHostToDevice));
                e.f32[ti.name] = d;
                dense_bytes += nb;
                continue;
            }
            Dense w;
            w.type = (int) ti.type;
            for (uint64_t d : ti.shape) w.shape.push_back((int64_t) d);
            w.row_bytes = row_bytes(w.type, w.shape[0]);
            uint8_t* d = nullptr;
            KC(cudaMalloc(&d, nb));
            KC(cudaMemcpy(d, src, nb, cudaMemcpyHostToDevice));
            w.d = d;
            e.dense[ti.name] = w;
            dense_bytes += nb;
        }
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    // S7 dense-fp16 cache: dequantize every 2-D dense weight to FP16 on the device, once.  Env-gated so the plain
    // path (and the oracle) is untouched when the flag is absent.
    if (dense_f16_on()) {
        int64_t f16_bytes = 0;
        for (auto& kv : e.dense) {
            Dense& w = kv.second;
            if (w.shape.size() != 2) continue;
            const int64_t K = w.shape[0], N = w.shape[1];
            const bool dq = strata::kernels::dequant_bf16_supported(w.type);
            const bool cp = (w.type == T_F32);   // F32 weights (the router) need a plain f32->f16 copy, not a dequant
            if (!dq && !cp) continue;
            uint16_t* p = nullptr;
            KC(cudaMalloc((void**) &p, (size_t) N * K * 2));
            if (dq) strata::kernels::dequant_f16(w.type, w.d, 0, N, K, p, g_st);
            else k_to_f16<<<grid1(N * K), kThreads>>>((const float*) w.d, (__half*) p, N * K);
            w.f16 = p;
            f16_bytes += (int64_t) N * K * 2;
        }
        KC(cudaStreamSynchronize(g_st));
        std::fprintf(stderr, "strata-kolibri: dense fp16 cache: %.2f GiB dequantized at load (STRATA_KOLIBRI_DENSE_F16)\n",
                     (double) f16_bytes / (1 << 30));
    }
    int native_layers = 0;
    for (const LayerBlobs& b : e.blobs) native_layers += b.native_ok ? 1 : 0;
    std::fprintf(stderr,
                 "strata-kolibri: %s: %zu tensors (layout checked); experts %zu x %zu blobs repacked, %.2f GiB "
                 "pinned; dense %.2f GiB on the GPU; native expert kernels for %zu / %zu layers; %.1f s\n",
                 path.c_str(), checked, e.blobs.size(), (size_t) e.g.n_expert, (double) expert_bytes / (1 << 30),
                 (double) dense_bytes / (1 << 30), (size_t) native_layers, e.blobs.size(), secs);
    if (e.counters) {
        size_t fb = 0, tb = 0;
        if (cudaMemGetInfo(&fb, &tb) == cudaSuccess)
            std::fprintf(stderr, "CNT vram free=%.2f GiB total=%.2f GiB (after dense upload)\n",
                         fb / 1073741824.0, tb / 1073741824.0);
    }
    return true;
}

}  // namespace

int strata::kolibri::run(int argc, char** argv) {
    std::string model, tokens, dump_layers, dump_logits, dump_full, force, tokens_file;
    bool load_only = false, serve_mode = false;
    int n_gen = 8, ctx = 8192, cpu_threads = 0, slots = 0, grouped = 0, no_graph = 0, prefill = 0, counters = 0;
    // the expert cache's opt-in knobs (see alloc_slots / slots_reselect): off means the historical by-id cache
    int hot = env_on("STRATA_KOLIBRI_SLOTS_HOT") ? 1 : 0;
    // --expert-slots-per-layer: the per-layer residency map (Phase 1A of the dynamic-expert-cache plan), ON BY
    // DEFAULT; --no-expert-slots-per-layer or STRATA_KOLIBRI_SLOTS_PER_LAYER=0 restores one set for all 50 layers.
    // Needs --expert-slots-hot; with the by-id cache nothing below is touched either way.
    int per_layer = env_on_default_on("STRATA_KOLIBRI_SLOTS_PER_LAYER") ? 1 : 0;
    bool per_layer_asked = std::getenv("STRATA_KOLIBRI_SLOTS_PER_LAYER") != nullptr;
    // The round's knobs (see include/strata/kernels/kolibri_hot_policy.hpp and the Engine block's comment): the
    // cadence in POSITIONS, two move budgets, the admission margin and the decay.  All of them are opt-out, and 0
    // means off for the pass it governs, the same meaning --adapt-swaps 0 has in the Qwen3.8-Flash-Next engine.
    int reselect_every = 32;
    int swaps_mid = 96;
    int swaps_final = 2048;
    double min_gain = 1.5;
    double decay = 0.7;
    int reserve_mib = 700;                     // --vram-reserve-mib: the Qwen engine's own default
    std::string profile;                       // --expert-slots-profile: the learned hot set, read and written here
    int attn_bench = 0;             // K25 instrument: --attn-bench N launches the real attention kernel N times
    int attn_mode = 0;              // 0 = full, 1 = score pass only (V neutralised), 2 = V pass only (scores const)
    double expert_vram = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "--model" || a == "--native") && i + 1 < argc) model = argv[++i];
        else if (a == "--tokens" && i + 1 < argc) tokens = argv[++i];
        // A 32K-token prompt is ~230 KB of digits, and Linux caps ONE argv entry at 128 KB (MAX_ARG_STRLEN), so
        // long prompts (the 32K gates, the 100K needle) must come from a file.  Same format as --tokens.
        else if (a == "--tokens-file" && i + 1 < argc) tokens_file = argv[++i];
        else if (a == "--n" && i + 1 < argc) n_gen = std::atoi(argv[++i]);
        else if (a == "--ctx" && i + 1 < argc) ctx = std::atoi(argv[++i]);
        else if (a == "--dump-layers" && i + 1 < argc) dump_layers = argv[++i];
        else if (a == "--dump-logits" && i + 1 < argc) dump_logits = argv[++i];
        else if (a == "--dump-full-logits" && i + 1 < argc) dump_full = argv[++i];
        else if (a == "--force" && i + 1 < argc) force = argv[++i];
        else if (a == "--load-only") load_only = true;
        else if (a == "--serve") serve_mode = true;
        else if (a == "--cpu-threads" && i + 1 < argc) {
            cpu_threads = std::atoi(argv[++i]);
            // T16: ONE lane is not a valid pool.  The prefill fan-out gives each lane a shard of the expert rows
            // and blocks on a barrier across them, so a single lane has nobody to wait for and the decode
            // degenerates into the 51 51 51 ... repeat loop (measured on this box 2026-10-28, recorded in
            // docs/AMD_R9700_PORT.md).  A lane count is a throughput knob, not a correctness switch, so refuse
            // the silently-wrong case here - before any model read or CUDA init - and run the smallest count that
            // works.  0 still means "no pool" (the opt-in absence, kolibri_engine.cu:4827).
            if (cpu_threads == 1) {
                std::fprintf(stderr, "strata-kolibri: WARNING: --cpu-threads 1 is not a valid pool: one lane emits "
                                     "the degenerate 51 51 51 ... repeat loop (measured 2026-10-28); using 2\n");
                cpu_threads = 2;
            }
        }
        else if (a == "--expert-slots" && i + 1 < argc) {
        // "auto" = fit the count to the VRAM that is still free once dense+KV+scratch are allocated.
        const std::string v = argv[i + 1];
        slots = (v == "auto") ? kSlotsAuto : std::atoi(v.c_str());
        ++i;
    }
    else if (a == "--expert-slots-hot") hot = 1;
        else if (a == "--expert-slots-per-layer") { hot = 1; per_layer = 1; per_layer_asked = true; }
        else if (a == "--no-expert-slots-per-layer") { per_layer = 0; per_layer_asked = true; }
    else if (a == "--expert-slots-profile" && i + 1 < argc) { hot = 1; profile = argv[++i]; }
    else if (a == "--expert-slots-reselect-every" && i + 1 < argc) { hot = 1; reselect_every = std::atoi(argv[++i]); }
    else if (a == "--expert-slots-swaps" && i + 1 < argc) { hot = 1; swaps_mid = std::atoi(argv[++i]); }
    else if (a == "--expert-slots-swaps-final" && i + 1 < argc) { hot = 1; swaps_final = std::atoi(argv[++i]); }
    else if (a == "--expert-slots-min-gain" && i + 1 < argc) { hot = 1; min_gain = std::atof(argv[++i]); }
    else if (a == "--expert-slots-decay" && i + 1 < argc) { hot = 1; decay = std::atof(argv[++i]); }
    else if (a == "--vram-reserve-mib" && i + 1 < argc) reserve_mib = std::atoi(argv[++i]);
        else if (a == "--expert-vram" && i + 1 < argc) expert_vram = std::atoi(argv[++i]);
        else if (a == "--grouped") grouped = 1;
        else if (a == "--no-graph") no_graph = 1;
        else if (a == "--prefill") prefill = 1;
        else if (a == "--attn-bench" && i + 1 < argc) attn_bench = std::atoi(argv[++i]);
        else if (a == "--attn-mode" && i + 1 < argc) attn_mode = std::atoi(argv[++i]);
        else if (a == "--attn-2pass-max" && i + 1 < argc) g_two_pass_max = std::atoi(argv[++i]);
        else if (a == "--counters") counters = 1;
    }
    if (pf_partial_on() && expert_f16_n() <= 0) {
        // G1 follow-up (plan 2026-10-26-kolibri-pf-partial-f16-guard.md): the pf_partial scratch (act1.wts_h,
        // p_h, p_m, rr_sel_m, p_done - kolibri_engine.cu:3252-3263) is allocated only under the fp16 expert
        // cache, and the partial legs read the cache's resident rows.  Setting the knob alone reaches the first
        // prefill publish and dies with CUDA "invalid argument" (copy into the null wts_h, :2584).  Refuse here,
        // before any model read or CUDA init, and name the missing knob.
        std::fprintf(stderr, "kolibri: STRATA_KOLIBRI_PF_PARTIAL rides the fp16 expert cache's scratch; "
                             "set STRATA_KOLIBRI_EXPERT_F16 (experts per layer, e.g. 28) first\n");
        return 2;
    }
    if (model.empty()) {
        std::fprintf(stderr, "strata-kolibri: the Kolibri 1 engine path (plan T7); --model GGUF [--load-only] for now\n");
        return 2;
    }
    dec_wait_budget_check();   // a malformed budget is a typo, whatever the arm state: refuse it before I/O
    const char* dec_why = nullptr;
    if (!dump_layers.empty()) dec_why = "--dump-layers is a diagnostic path that cannot be captured";
    else if (cpu_threads <= 0) dec_why = "--cpu-threads must select the CPU pool";
    else if (grouped != 0) dec_why = "--grouped serves picks on the GPU";
    else if (slots != 0) dec_why = "--expert-slots must be zero";
    else if (expert_f16_n() > 0 || dec_res_on()) dec_why = "the fp16 expert cache must be off";
    if (!dec_arms_resolve(dec_why)) return 2;   // an explicit request that cannot be honoured
    if (dec_token_graph_on() && dec_arm_request("STRATA_KOLIBRI_DEC_TOKEN_GRAPH") == 2)
        std::fprintf(stderr, "kolibri: DEC_TOKEN_GRAPH: cold mirrors and new buckets warm through segments\n");
    Engine e;
    std::string err;
    KC(cudaStreamCreate(&g_st));
#if defined(STRATA_KOLIBRI_MMQ) && defined(__HIPCC__)
    // Enable only this engine's measured prompt arm, before MMQ caches its K-quant availability.
    // Honor an explicit STRATA_MMQ_KQ=0. HIP links these instances statically (no strata_kq DSO).
    if ((prefill > 0 || serve_mode) && pf_mmq_on() && !std::getenv("STRATA_MMQ_KQ")) {
#if defined(_WIN32)
        if (_putenv_s("STRATA_MMQ_KQ", "1") != 0) return 2;
#else
        if (setenv("STRATA_MMQ_KQ", "1", 0) != 0) return 2;
#endif
    }
#endif
    {   // S6: mirror the decode's coalesced-dot flag to the device (the kernel body is device code)
        const bool v = env_on("STRATA_KOLIBRI_ATTN_COALESCED");
        KC(cudaMemcpyToSymbol(g_attn_coalesced, &v, sizeof(bool)));
    }
    {   // T4: the packed Q4_K/Q6_K nibble extraction (bitwise; the parity arm in tools/kolibri/grid_shape_bench.cu
        // asserts it element by element).  Its only consumers in this engine are the token-embedding gathers
        // (`k_row`, `k_row_multi`) - the quantized dense chain goes to native_mmvq before it ever reaches deq32.
        const int pd = env_on("STRATA_KOLIBRI_PACKED_DEQ") ? 1 : 0;
        KC(cudaMemcpyToSymbol(g_packed_deq, &pd, sizeof(int)));
    }
    {   // S8b: and the PV 4-chain flag (a numerics change, so it is KL-gated rather than bitwise-gated).
        // It must go through attn_pv4_on(), whose host branch carries the default-ON logic: mirroring env_on() directly
        // made the "default" inert on the device side (measured: 26.7 with no env, 30.1 only when the env was set).
        const bool v4 = attn_pv4_on();
        KC(cudaMemcpyToSymbol(g_attn_pv4, &v4, sizeof(bool)));
    }
    KC(cudaMalloc(&g_q8, strata::kernels::native_q8_1_bytes((int) kActBuf, 1)));
    e.counters = counters != 0;   // before load(): the VRAM report is printed there
    e.hist_on = env_on("STRATA_KOLIBRI_EXPERT_HIST");   // coverage probe; before load() so the table is sized
    if (!load(model, e, err)) {
        std::fprintf(stderr, "strata-kolibri: %s\n", err.c_str());
        return 1;
    }
    // K3: a 512-token chunk writes 512 of the 513 sliding ring slots, leaving exactly the oldest visible slot.
    // Safe for n_swa = 513 by ONE slot, so the invariant is asserted, not assumed.
    if (Scratch::kTmax >= e.g.n_swa) {
        std::fprintf(stderr, "strata-kolibri: n_swa %lld must exceed the %d-token prompt chunk, else one chunk "
                             "overwrites the whole sliding ring\n", (long long) e.g.n_swa, Scratch::kTmax);
        return 1;
    }
    // T15 (plan 2026-10-28-kolibri-r9700-speed.md): the load-time warm-up, measured and therefore DEFAULT OFF.
    // The LM head's first call costs more than its steady state (697.8 us against 519.8-520.3 us, docs/AMD_R9700_PORT.md
    // §T1), and the question was whether that is code load.  It is not: warming the same kernel at n_out = 1 changed
    // nothing (696.2 us), and warming it at the LM head's own 128000 rows moves the first real call to 618.4 us -
    // 79.4 us (11 %) of a 178 us excess, bought with 2.9 ms of extra load and a 512 KB throwaway buffer.  What is
    // left is the real output buffer's first touch and the weight's cold pages, which a warm-up cannot pre-pay
    // because the buffers it would have to touch are the ones the first request allocates.  So the hook is here,
    // opt-in, for a card or a kernel where the balance differs - STRATA_KOLIBRI_WARM=1 - and the shipped default is
    // the un-warmed path, unchanged to the bit (ledger §T15).
    if (env_on("STRATA_KOLIBRI_WARM")) {
        const Dense& wo = e.dense.at("output.weight");
        const bool quant = wo.d != nullptr &&
                           (wo.type == T_Q8_0 || wo.type == T_Q4_K || wo.type == T_Q5_K || wo.type == T_Q6_K);
        if (quant) {   // the LM head on this model is Q6_K, and its token path is exactly this kernel
            // The row count MATTERS: a warm-up at n_out = 1 was measured and did NOT remove the first-call cost
            // (696.2 us for the first real call against 697.8 us without any warm-up, ledger §T15), so the cost is
            // not code load - it is the real grid and the real buffers.  Warm at the LM head's own n_out, into a
            // throwaway buffer of the same size: that is the form worth 79.4 us.
            const int64_t n_out = (int64_t) wo.shape[1];
            if (n_out > 0 && n_out <= (1 << 20)) {
                float* warm_y = nullptr;
                KC(cudaMalloc((void**) &warm_y, (size_t) n_out * sizeof(float)));
                KC(cudaMemsetAsync(g_q8, 0, strata::kernels::native_q8_1_bytes((int) e.g.n_embd, 1), g_st));
                strata::kernels::native_mmvq(wo.type, wo.d, (const char*) g_q8, warm_y, (int) e.g.n_embd, (int) n_out,
                                             1, g_st);
                KC(cudaGetLastError());
                KC(cudaStreamSynchronize(g_st));
                KC(cudaFree(warm_y));
            }
        }
        // An F32/F16/BF16 LM head takes k_gemv and is not warmed here: this model's is quantised, and warming a
        // path the build does not run would be a line that never executes.
    }
    e.cpu_threads = cpu_threads;
    e.grouped = grouped;
    // CUDA-graph decode rides the grouped path (device-side expert pointers); the slots/pool paths read host
    // ids per layer and are not capturable.  --no-graph keeps the plain launches.
    e.graphs = grouped && !no_graph && slots == 0 && cpu_threads == 0 ? 1 : 0;
    e.prefill = prefill;
    // The CPU pool is OPT-IN (--cpu-threads) and its absence is SILENT: with no pool every routed expert falls
    // through to the per-expert GPU mmvq loop (measured on 2026-10-21: 364,350 mmvq launches = 26.8 s of 35 s
    // GPU busy, and the prefill 4.6x slower - 22,126 ms vs 4,744 ms - with nothing in the log).  A future reader
    // would read that as a regression, so the prefill says so out loud.
    if (e.prefill > 0 && e.cpu_threads == 0)
        std::fprintf(stderr, "strata-kolibri: WARNING: --prefill without --cpu-threads means NO CPU pool: every "
                             "routed expert takes the per-expert GPU mmvq fallback (measured 4.6x slower on the "
                             "512-token prefill: 22,126 ms vs 4,744 ms).  Pass --cpu-threads 24 for the speed "
                             "configuration.\n");
    if (e.prefill > 0) {
        if (cublasCreate(&e.blas) != CUBLAS_STATUS_SUCCESS || cublasSetStream(e.blas, g_st) != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "strata-kolibri: cuBLAS init failed\n");
            return 1;
        }
        // P1: THE ATTENTION HAS ITS OWN HANDLE, which is why STRATA_GEMM_DETERMINISTIC changed nothing when it was
        // applied only to the Gemm object.  Its strided-batched score/PV calls are exactly the multi-token shapes at
        // which cuBLAS may pick an atomics-based (split-K) kernel - and the default attention is the one component that
        // measured non-deterministic (T>=64) while ATTN_FLASH, which does not use it, is bit-identical.
        if (std::getenv("STRATA_GEMM_DETERMINISTIC") != nullptr)
            cublasSetAtomicsMode(e.blas, CUBLAS_ATOMICS_NOT_ALLOWED);
        const int64_t big = std::max(e.g.n_head * e.g.head_dim, e.g.n_embd) * (int64_t) e.g.n_embd;
        if (!e.gemm.init(g_st, big, err)) {
            std::fprintf(stderr, "strata-kolibri: prompt GEMM init failed: %s\n", err.c_str());
            return 1;
        }
    }
    if (expert_vram > 0 && slots == 0) {
        // S5 (task-8 defect): the slot count must be budgeted with the LARGEST layer stride, not the first
        // layer's: the stride is the block-aligned blob size (the lcm of the types' super-blocks), and the
        // Q6_K layers need a larger alignment than the Q4_K ones, so the first layer's stride UNDER-estimates
        // the per-layer allocation - the 2 GiB budget asked for 1018 slots/layer and the Q6_K layers' table
        // then did not fit (the engine died before READY).  Budget with the max stride: the total stays
        // within the requested VRAM for every layer.
        size_t stride_max = 0;
        for (auto& b : e.blobs) stride_max = std::max(stride_max, b.stride);
        slots = (int) (expert_vram * (1 << 30) / ((double) stride_max * (double) e.g.n_layers));
    }
    // The slot TABLE is allocated in alloc_slots(), after init_scratch: that is the only point where a fit can see
    // the KV and the activations already paid for.  -1 (auto) and 0 (no cache) both pass through here.
    e.slots = slots;
    e.hot_on = hot;
    e.per_layer = (hot && per_layer) ? 1 : 0;
    if (per_layer && per_layer_asked && !hot)
        std::fprintf(stderr, "strata-kolibri: --expert-slots-per-layer needs --expert-slots-hot: ignored\n");
    if (e.per_layer)
        std::fprintf(stderr, "strata-kolibri: expert cache: PER-LAYER residency map (%d layers x %d experts), not one"
                             " global set (default; --no-expert-slots-per-layer for a single set)\n",
                     (int) e.g.n_layers, (int) e.g.n_expert);
    e.hot_every = reselect_every > 0 ? reselect_every : 0;
    e.hot_swaps = swaps_mid > 0 ? swaps_mid : 0;
    e.hot_swaps_final = swaps_final > 0 ? swaps_final : 0;
    e.hot_min_gain = min_gain > 0.0 ? min_gain : 0.0;
    const bool decay_ok = decay > 0.0 && decay <= 1.0;
    e.hot_decay_factor = decay_ok ? decay : 1.0;
    if (!decay_ok)
        std::fprintf(stderr, "strata-kolibri: --expert-slots-decay %.2f is outside (0, 1]: using 1.0 (no decay)\n",
                     decay);
    if (e.hot_on)
        std::fprintf(stderr, "strata-kolibri: expert cache: a round every %d position(s)%s, at most %d move(s)"
                             " mid-request and %d after a request, min gain %.2f, decay %.2f\n",
                     e.hot_every, e.hot_every > 0 ? "" : " (between requests only)", e.hot_swaps, e.hot_swaps_final,
                     e.hot_min_gain, e.hot_decay_factor);
    e.hot_profile = profile;
    e.vram_reserve_mib = reserve_mib > 0 ? reserve_mib : 700;
    {
        const char* rv = std::getenv("STRATA_KOLIBRI_VRAM_RESERVE_MIB");
        if (rv != nullptr && std::atoi(rv) > 0) e.vram_reserve_mib = std::atoi(rv);
        const char* hmin = std::getenv("STRATA_KOLIBRI_SLOTS_HOT_MIN");
        if (hmin != nullptr && std::atoi(hmin) > 0) e.hot_min_picks = std::atoi(hmin);
        const char* rev = std::getenv("STRATA_KOLIBRI_SLOTS_RESELECT_EVERY");
        if (rev != nullptr) reselect_every = std::atoi(rev);
        const char* sw = std::getenv("STRATA_KOLIBRI_SLOTS_SWAPS");
        if (sw != nullptr) swaps_mid = std::atoi(sw);
        const char* swf = std::getenv("STRATA_KOLIBRI_SLOTS_SWAPS_FINAL");
        if (swf != nullptr) swaps_final = std::atoi(swf);
        const char* mg = std::getenv("STRATA_KOLIBRI_SLOTS_MIN_GAIN");
        if (mg != nullptr) min_gain = std::atof(mg);
        const char* dk = std::getenv("STRATA_KOLIBRI_SLOTS_DECAY");
        if (dk != nullptr) decay = std::atof(dk);
    }
    if (e.cpu_threads > 1) {
        // The prefill fan-out's pool: lanes = the host lane + one lane per pinnable PHYSICAL core of this
        // cpuset.  Asking for more lanes than there are cores does not add bandwidth (pool_bench on this node:
        // 12 workers 34.5 GB/s, 24 workers 34.0) - it only puts two lanes on one core, and the barrier then waits
        // for the HT-shared one.  The old code passed --cpu-threads through verbatim and said nothing.
        const std::vector<int> pf_cores = strata::kernels::cpu::physical_cores(true);
        e.pf_cores = pf_cores;   // stored: deriving it again later can return EMPTY once a thread is pinned
        const int lanes_cap = pf_cores.empty() ? e.cpu_threads : (int) pf_cores.size() + 1;
        int lanes = e.cpu_threads;
        const char* lanes_env = std::getenv("STRATA_KOLIBRI_PF_LANES");
        if (lanes_env != nullptr && std::atoi(lanes_env) > 0)
            lanes = std::atoi(lanes_env);   // the A/B override: reproduces the old lane count on identical code
        else if (lanes > lanes_cap)
            lanes = lanes_cap;
        if (lanes > e.cpu_threads) lanes = e.cpu_threads;
        if (lanes < 1) lanes = 1;
        e.pf_lanes = lanes;
        e.pool.pin = env_on_default_on("STRATA_KOLIBRI_PF_PIN");
        std::fprintf(stderr,
                     "strata-kolibri: prefill pool: %d lane(s) = the host + %d thread(s), %zu pinnable physical"
                     " core(s)%s%s\n",
                     lanes, lanes - 1, pf_cores.size(), e.pool.pin ? ", pinned" : ", PINNING OFF",
                     lanes < e.cpu_threads ? " (capped from --cpu-threads)" : "");
        e.pool.start(lanes, e.pf_cores);
        // P1b: the expert rows go to the library's pool (row-split over all workers, affinity-aware).  The
        // measured placement rule (pool_bench): ~24 workers on the GPU's socket; 48 on one socket collapse.
        e.epool = std::make_unique<strata::kernels::cpu::ExpertPool>(
            e.cpu_threads, true, true, strata::kernels::cpu::PoolAffinity::All);
    }
    // K25 instrument.  The 32K decode term is 12 full-attention layers x one k_attn launch at n_vis = pos+1,
    // and an end-to-end profiled run needs a ~55 minute prefill before the kernel of interest runs once.  This
    // launches the REAL kernel at the REAL geometry (H=48, HKV=4, D=128, ctx from --ctx) so a kernel change is
    // measured in seconds.  The buffers are the kernel's own types; the KV is filled with a nonzero pattern so
    // the loads are real traffic, not zeros.
    if (attn_bench > 0) {
        const int H = (int) e.g.n_head, HKV = (int) e.g.n_head_kv, D = (int) e.g.head_dim;
        const int64_t rows = ctx;
        float *q = nullptr, *out = nullptr; __half *kc = nullptr, *vc = nullptr;
        KC(cudaMalloc((void**) &q, (size_t) H * D * 4));
        KC(cudaMalloc((void**) &out, (size_t) H * D * 4));
        KC(cudaMalloc((void**) &kc, (size_t) HKV * rows * D * 2));
        KC(cudaMalloc((void**) &vc, (size_t) HKV * rows * D * 2));
        KC(cudaMemset(q, 0, (size_t) H * D * 4));
        KC(cudaMemset(kc, 0x3c, (size_t) HKV * rows * D * 2));
        KC(cudaMemset(vc, 0x3c, (size_t) HKV * rows * D * 2));
        const int pos = (int) (rows - 2);
        const int n_vis = pos + 1;
        const size_t smem = n_vis <= 11000 ? (size_t) n_vis * 4 : (size_t) 4096 * 4;
        const int thr = n_vis <= 11000 ? kThreads : 512;
        cudaEvent_t ev0, ev1; cudaEventCreate(&ev0); cudaEventCreate(&ev1);
        for (int r = 0; r < 3; ++r)   // warm the clocks and the TLB
            k_attn<<<H, thr, smem, g_st>>>(q, kc, vc, out, H, HKV, D, pos, e.g.n_swa, ctx, 1.0f / sqrtf((float) D), false, attn_mode, g_two_pass_max);
        cudaEventRecord(ev0, g_st);
        for (int r = 0; r < attn_bench; ++r)
            k_attn<<<H, thr, smem, g_st>>>(q, kc, vc, out, H, HKV, D, pos, e.g.n_swa, ctx, 1.0f / sqrtf((float) D), false, attn_mode, g_two_pass_max);
        cudaEventRecord(ev1, g_st);
        KC(cudaEventSynchronize(ev1));
        float ms = 0.f; cudaEventElapsedTime(&ms, ev0, ev1);
        const double per = (double) ms / attn_bench;
        std::printf("attn-bench: mode=%d pos=%d n_vis=%d H=%d thr=%d smem=%zu KB | %.3f ms/launch | x12 full layers = %.1f ms/token\n",
                    attn_mode, pos, n_vis, H, thr, smem / 1024, per, per * 12.0);
        std::fflush(stdout);
        return 0;
    }
    if (load_only) return 0;
    // The arms are a scheduling change, so they are only allowed where the arithmetic is the plain pool's: the pool
    // feeds the routed experts, nothing else serves a pick, and no slot or fp16 cache moves bytes.  The flags were
    // checked before loading; this is the one verdict that needs the loaded layer table.
    if (dec_handoff_on()) {
        const char* why = nullptr;
        for (size_t li = 0; li < e.blobs.size(); ++li)
            if (!e.blobs[li].pool_ok) { why = "a layer's experts cannot run in the CPU pool"; break; }
        if (why != nullptr && !dec_arms_resolve(why)) {
            std::fprintf(stderr, "kolibri: the arms are a scheduling change over the plain CPU-pool decode; see "
                                 "docs/KOLIBRI_DECODE.md\n");
            std::fflush(stderr);
            return 2;
        }
    }
    if (dec_handoff_on())
        std::fprintf(stderr, "kolibri: STRATA_KOLIBRI_DEC_HANDOFF active: the layer publishes its FFN input, picks "
                             "and weights in one kernel and the host waits on the ring (bounded at %d ms)\n",
                     dec_handoff_timeout_ms());
    if (dec_attn_dp_on())
        std::fprintf(stderr, "kolibri: STRATA_KOLIBRI_DEC_ATTN_DP active: decode attention is the single-pass "
                             "body with the position from device memory, shared memory sized per bucket\n");
    if (dec_stream_on())
        std::fprintf(stderr, "kolibri: STRATA_KOLIBRI_DEC_STREAM active: the decode runs on one explicit "
                             "non-blocking stream (router and pool result copy included)\n");
    if (dec_graph_on()) {
        // Diagnostics that do host I/O or synchronise inside a layer cannot be captured.  An explicit request for
        // capture is refused; an automatic one steps down to the eager layer path and says why.
        const char* why = nullptr;
        if (reuse_gate_kv_path() != nullptr) why = "STRATA_KOLIBRI_KV_DUMP does host I/O inside the layer";
        else if (dec_phases_on()) why = "STRATA_KOLIBRI_DEC_PHASES synchronises inside the layer";
        if (why != nullptr && dec_arm_request("STRATA_KOLIBRI_DEC_HANDOFF_GRAPH") == 2) {
            std::fprintf(stderr, "kolibri: STRATA_KOLIBRI_DEC_HANDOFF_GRAPH is set but not supported: %s.\n", why);
            std::fflush(stderr);
            return 2;
        }
        if (why != nullptr) {
            g_dec_arms.graph = false;
            g_dec_arms.token = false;
            std::fprintf(stderr, "kolibri: per-layer capture skipped (%s); the decode runs the eager layer path\n", why);
        } else {
            std::fprintf(stderr, "kolibri: STRATA_KOLIBRI_DEC_HANDOFF_GRAPH active: per-layer pre/post graph segments, "
                                 "CPU pool unchanged\n");
        }
    }
    if (serve_mode) {
        Scratch s;
        s.ctx = ctx;
        init_scratch(e, s);
        alloc_slots(e, ctx);   // after init_scratch: the fit measures the KV and the scratch as already spent
        if (e.graphs > 0 && e.gexec.empty()) capture_graphs(e, s);
        return serve(e, s);
    }
    if (!tokens_file.empty()) {
        // Read before the emptiness check below: --tokens-file is an alternative SOURCE for --tokens, not a post-step.
        std::FILE* ft = std::fopen(tokens_file.c_str(), "rb");
        if (ft == nullptr) { std::fprintf(stderr, "strata-kolibri: cannot open --tokens-file %s\n", tokens_file.c_str()); return 1; }
        std::string buf;
        char chunk[65536];
        size_t got;
        while ((got = std::fread(chunk, 1, sizeof(chunk), ft)) > 0) buf.append(chunk, got);
        std::fclose(ft);
        tokens = std::move(buf);
    }
    if (tokens.empty()) {
        std::fprintf(stderr, "strata-kolibri: --tokens ID,ID,... (the layer driver is plain; serve arrives in plan T8)\n");
        return 2;
    }
    Scratch s;
    s.ctx = ctx;
    init_scratch(e, s);
    alloc_slots(e, ctx);   // after init_scratch: the fit measures the KV and the scratch as already spent
    if (e.graphs > 0 && e.gexec.empty()) capture_graphs(e, s);
    std::vector<float> lg((size_t) e.g.n_vocab);
    std::vector<float> layers((size_t) e.g.n_layers * (size_t) e.g.n_embd);
    std::FILE* fl = dump_layers.empty() ? nullptr : std::fopen(dump_layers.c_str(), "wb");
    std::FILE* fg = dump_logits.empty() ? nullptr : std::fopen(dump_logits.c_str(), "wb");
    std::FILE* ff = dump_full.empty() ? nullptr : std::fopen(dump_full.c_str(), "wb");
    auto parse_ids = [](const std::string& s) {
        std::vector<int> v;
        for (size_t i = 0; i < s.size(); ++i) {
            while (i < s.size() && (s[i] == ',' || s[i] == ' ' || s[i] == '\n')) ++i;
            if (i >= s.size()) break;
            v.push_back(std::atoi(s.c_str() + i));
            while (i < s.size() && s[i] != ',' && s[i] != ' ' && s[i] != '\n') ++i;
        }
        return v;
    };
    std::vector<int> ids = parse_ids(tokens);
    std::vector<int> forced = parse_ids(force);
    // **BOUND THE REQUEST BY THE CONTEXT (the `:2244` crash).**  Nothing on this path compared the request with `s.ctx`:
    // a 116,025-token prompt run with `--ctx 104448` prefilled to exactly 104,448 and then wrote the next chunk's K/V
    // outside `kc/vc` (allocated from `s.ctx`, :2713) - the fault surfaced later, at the first blocking copy, as
    // "illegal memory access".  The serve path has its own guard (:3077); this is the CLI equivalent, which was missing.
    {
        const int64_t n_steps_est = forced.empty() ? (int64_t) n_gen : (int64_t) forced.size() + 1;
        if ((int64_t) ids.size() + n_steps_est > (int64_t) s.ctx) {
            std::fprintf(stderr, "strata-kolibri: prompt (%zu tokens) + %lld step(s) exceeds ctx (%d); raise --ctx\n",
                         ids.size(), (long long) n_steps_est, s.ctx);
            std::fflush(stderr);
            return 2;
        }
    }
    int tok = ids[0];
    // Chunked prefill: the whole prompt minus its last token.  A layer dump used to force the per-token path here
    // (`fl == nullptr`), which had two costs: the instrument became unusable at 100K (per-token prefill > 65 min) AND
    // it measured a path the engine no longer ships.  It is not needed - the chunks fill the KV for ids[0..n-2] and
    // the decode loop's first forward() below processes ids.back() WITH the dump, so the chunked path emits exactly
    // the rows a layer-by-layer comparison wants: 1 (final prompt position) + n_gen (generated steps).
    if (e.prefill > 0) {
        int64_t i = 0;
        const int64_t n = (int64_t) ids.size() - 1;
        e.pf_total = n;   // the POOL_ALL rule: one decision for the whole prompt
        // K28: the chunk rule keys on the PREFIX LENGTH.  `n` is ids.size() - 1 (the last id seeds generation), so an
        // 8192-token request prefills 8191 and fell on the SHORT side of the rule's 8192 threshold: 32 chunks of 256
        // where 16 of 512 were intended (docs/AMD_R9700_PORT.md §P2).  Key on the request's length, which is what the
        // rule's constant was measured against.
        g_pf_total = (int64_t) ids.size();   // K28: and the chunk rule keys on it (see pf_chunk_at)
        const auto tp0 = Clock::now();
        // **THE HOST IS A POOL LANE, SO IT MUST SIT ON ITS OWN CORE.**  The prefill fan-out runs shard 0 on the
        // calling thread; leaving that thread unpinned lets it land on a worker's core (or its SMT sibling) and
        // makes the barrier wait for it - the library's own measurement of this reservation being a fiction is
        // 36.32 GB/s on 5 pinned workers vs 26.9 inside an unpinned host loop (1.35x, include/strata/kernels/
        // cpu/pool.hpp).  Same rule, same reason, here.
        const strata::kernels::cpu::ThreadAffinity pf_host_prev =
            (e.pool.pin && e.cpu_threads > 1)
                ? strata::kernels::cpu::pin_current_thread(strata::kernels::cpu::planned_host_core())
                : strata::kernels::cpu::ThreadAffinity{};
        while (i < n) {
            int T = (int) std::min<int64_t>(pf_chunk_at(s.pos), n - i);
            // DEFENCE IN DEPTH for the `:2244` crash: whatever the caller computed, never let a chunk's KV write run past
            // the allocation (kc/vc are sized from s.ctx).  The entry guard above should make this unreachable; if the
            // clamp ever fires, the request is truncated rather than corrupting memory past the buffer.
            if (s.pos + (int64_t) T > (int64_t) s.ctx) {
                T = (int) std::max<int64_t>(0, (int64_t) s.ctx - s.pos);
                if (T <= 0) break;
            }
            KC(cudaMemcpy(s.act1.toks_d, ids.data() + i, (size_t) T * 4, cudaMemcpyHostToDevice));
            // **A2b: peek the next chunk** so the wavefront has two in flight.  T2 == 0 leaves the single-chunk path
            // byte-identical (which is what the gate checks); with T2 > 0 the diagonal runs the two-chunk schedule.
            int T_next = 0;
            const int64_t pos_next = s.pos + T;
            if (pf_pipe_on() && i + T < n) {
                int64_t t2 = std::min<int64_t>(pf_chunk_at(pos_next), n - (i + T));
                if (pos_next + t2 > (int64_t) s.ctx) t2 = std::max<int64_t>(0, (int64_t) s.ctx - pos_next);
                T_next = (int) t2;
                if (T_next > 0)
                    KC(cudaMemcpy(s.act2.toks_d, ids.data() + i + T, (size_t) T_next * 4, cudaMemcpyHostToDevice));
            }
            const double dyn_t0 = dyn_probe_on() ? dyn_now_ms() : 0.0;
            forward_multi(e, s, T, s.pos, T_next, pos_next);
            // P1 probe: a fence between chunks.  A single-chunk prefill is bit-exact and a two-chunk one is not, so
            // the race lives in the chunk-to-chunk handoff; this says whether the handoff is missing a fence (and if
            // it is, which producer needs it can then be found by removing the fence again and adding targeted waits).
            if (std::getenv("STRATA_KOLIBRI_PF_FENCE") != nullptr) KC(cudaStreamSynchronize(g_st));
            s.pos += T + T_next;
            i += T + T_next;
            std::printf("PFchunk %lld/%lld %.0fms\n", (long long) i, (long long) n, ms_since(tp0));
            std::fflush(stdout);
            dyn_probe_step(e, "chunk", dyn_probe_on() ? dyn_now_ms() - dyn_t0 : 0.0);
            slots_tick(e, "chunk", (long long) i);   // a round every --expert-slots-reselect-every positions
        }
        if (pf_host_prev.valid) strata::kernels::cpu::restore_thread_affinity(pf_host_prev);
        tok = ids.back();
    } else
    for (size_t t = 1; t < ids.size(); ++t) {
        forward(e, s, ids[t - 1], fl ? layers.data() : nullptr);
        if (fl) {
            for (int64_t l = 0; l < e.g.n_layers; ++l)
                std::fwrite(layers.data() + (size_t) l * e.g.n_embd, 4, e.g.n_embd, fl);
        }
        tok = ids[t];
    }
    hot_absorb_prompt(e);   // the prompt routed every layer: its counts are what the round below decides on
    slots_reselect(e, " [after prompt]", e.hot_swaps_final);   // the prompt is read: the one large round (above)
    hot_forget_prompt(e);
    if (e.graphs > 0) KC(cudaMemcpy(s.pos_d, &s.pos, 4, cudaMemcpyHostToDevice));
    prefix_fingerprint(e, s);
    // the last prompt token and every step after it: run, argmax, feed back (a --force sequence replaces the
    // greedy picks - kl.py's teacher-forced KL path, kolibriref's semantics)
    const int n_steps = (int) forced.empty() ? n_gen : (int) forced.size() + 1;
    std::vector<int> gen;
    std::printf("gen:");
    for (int i = 0; i < n_steps; ++i) {
        const double dyn_t0 = dyn_probe_on() ? dyn_now_ms() : 0.0;
        forward(e, s, tok, fl ? layers.data() : nullptr);
        if (fl)
            for (int64_t l = 0; l < e.g.n_layers; ++l)
                std::fwrite(layers.data() + (size_t) l * e.g.n_embd, 4, e.g.n_embd, fl);
        KC(cudaMemcpy(lg.data(), s.logits, lg.size() * 4, cudaMemcpyDeviceToHost));
        if (e.counters) { e.cnt.d2h_n += 1; e.cnt.d2h_bytes += (long long) lg.size() * 4; }
        int a = 0, c = -1;
        for (int j = 1; j < (int) lg.size(); ++j) if (lg[j] > lg[a]) a = j;
        for (int j = 0; j < (int) lg.size(); ++j) if (j != a && (c < 0 || lg[j] > lg[c])) c = j;
        std::printf(" %d", a);
        if (fg) { const float rec[4] = {(float) a, lg[a], (float) c, lg[c]}; std::fwrite(rec, 4, 4, fg); }
        if (ff) std::fwrite(lg.data(), 4, lg.size(), ff);
        gen.push_back(a);
        tok = i + 1 == n_steps ? a : (i < (int) forced.size() ? forced[i] : a);
        if (dyn_probe_on()) dyn_probe_step(e, "token", dyn_now_ms() - dyn_t0);
        slots_tick(e, "token", (long long) s.pos);   // a round every --expert-slots-reselect-every positions
    }
    dyn_probe_finish(e);
    std::printf("\n");
    print_counters(e, "run");
    if (fl) std::fclose(fl);
    if (fg) std::fclose(fg);
    if (ff) std::fclose(ff);
    return 0;
}
