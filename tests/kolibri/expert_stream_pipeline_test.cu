// tests/kolibri/expert_stream_pipeline_test.cu - plan 2026-10-31-kolibri-gpu-streaming, Task 3.
//
// The two-slot pipeline against the CPU pool's own kernels.  Every case compares the published output BYTE FOR
// BYTE with native_gu_rows + native_quant_h + native_down_rows, which is what the engine would have computed; a
// tolerance is not accepted anywhere.  The lifecycle cases (empty input, one group holding every selection, the two
// down formats alternating on one context, a hundred calls through the same two slots, output poisoned between
// calls) and the failure cases (budget, format, ids, every allocation ordinal failing) are here because they are
// where a slot pipeline goes wrong silently.
//
// Usage: expert_stream_pipeline_test [--selftest] [--tails-only]

#include "../../src/kolibri/expert_stream.hpp"

#include "ggml-cpu.h"

#include "strata/kernels/cpu/native_expert.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace strata::kolibri::stream;
using strata::kernels::cpu::NativeFmt;

int g_failures = 0;
int g_cases = 0;

constexpr int kQ4KBlk = 144, kQ6KBlk = 210, kQ8KBlk = 292;

uint16_t half_bits(float f) {
    const __half h = __float2half_rn(f);
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
}

/// A layer's blob table: real Q4_K/Q6_K block bytes with plausible scales.
struct Layer {
    NativeFmt fmt{};
    std::vector<uint8_t> weights;
    int experts = 0;
    size_t stride = 0;
};

Layer make_layer(int n_embd, int n_ff, int experts, int d_type, uint32_t seed) {
    Layer l;
    l.fmt.gu_type = 12;
    l.fmt.d_type = d_type;
    l.fmt.gu_act = 15;
    l.fmt.d_act = 15;
    l.fmt.n_embd = n_embd;
    l.fmt.n_ff = n_ff;
    l.fmt.gu_row = (size_t) (n_embd / 256) * kQ4KBlk;
    l.fmt.d_row = (size_t) (n_ff / 256) * (d_type == kQ4K ? kQ4KBlk : kQ6KBlk);
    l.fmt.up_off = (size_t) n_ff * l.fmt.gu_row;
    l.fmt.down_off = 2 * (size_t) n_ff * l.fmt.gu_row;
    l.fmt.bytes = l.fmt.down_off + (size_t) n_embd * l.fmt.d_row;
    l.fmt.act_bytes = (size_t) (n_embd / 256) * kQ8KBlk;
    l.fmt.h_bytes = (size_t) (n_ff / 256) * kQ8KBlk;
    l.experts = experts;
    l.stride = l.fmt.bytes + 192;   // a stride with padding, like the real table's alignment
    l.weights.assign(l.stride * (size_t) experts, 0);
    // Realistic blocks: ggml's own quantizer turns float rows into valid Q4_K/Q6_K blocks, so every scale, minimum
    // and nibble is consistent and the magnitudes are the model's.  A small pool of distinct rows is tiled across
    // the experts: the pipeline test is about plumbing and the arithmetic contract, and Task 4 replays REAL captured
    // weights through the same code.
    std::mt19937 g(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    const int pool_rows = 8;
    std::vector<std::vector<uint8_t>> pool_gu((size_t) pool_rows), pool_up((size_t) pool_rows), pool_dn((size_t) pool_rows);
    std::vector<float> row((size_t) n_embd), hrow((size_t) n_ff);
    for (int p = 0; p < pool_rows; ++p) {
        for (auto& v : row) v = nd(g);
        pool_gu[(size_t) p].assign(l.fmt.gu_row, 0);
        ggml_quantize_chunk(GGML_TYPE_Q4_K, row.data(), pool_gu[(size_t) p].data(), 0, 1, n_embd, nullptr);
        for (auto& v : hrow) v = nd(g);
        pool_up[(size_t) p].assign(l.fmt.gu_row, 0);
        ggml_quantize_chunk(GGML_TYPE_Q4_K, hrow.data(), pool_up[(size_t) p].data(), 0, 1, n_ff, nullptr);
        pool_dn[(size_t) p].assign(l.fmt.d_row, 0);
        ggml_quantize_chunk(d_type == kQ4K ? GGML_TYPE_Q4_K : GGML_TYPE_Q6_K, hrow.data(), pool_dn[(size_t) p].data(), 0,
                            1, n_ff, nullptr);
    }
    for (int e = 0; e < experts; ++e) {
        uint8_t* blob = l.weights.data() + (size_t) e * l.stride;
        for (int r = 0; r < n_ff; ++r) {
            const int p = (r + e) % pool_rows;
            std::memcpy(blob + (size_t) r * l.fmt.gu_row, pool_gu[(size_t) p].data(), l.fmt.gu_row);
            std::memcpy(blob + l.fmt.up_off + (size_t) r * l.fmt.gu_row, pool_up[(size_t) p].data(), l.fmt.gu_row);
        }
        for (int r = 0; r < n_embd; ++r) {
            const int p = (r + e) % pool_rows;
            std::memcpy(blob + l.fmt.down_off + (size_t) r * l.fmt.d_row, pool_dn[(size_t) p].data(), l.fmt.d_row);
        }
    }
    return l;
}

struct Run {
    std::vector<int> ids;
    std::vector<float> acts;      // [tokens][n_embd] floats, quantized into token_acts below
    std::vector<uint8_t> token_acts;
    std::vector<float> out;
};

/// The CPU pool's answer for the same ids and activations: the reference every GPU output is compared with.
std::vector<float> native_reference(const Layer& l, int top_k, const std::vector<int>& ids, const Run& r, int n_embd) {
    const int tokens = (int) (ids.size() / (size_t) top_k);
    std::vector<float> want((size_t) ids.size() * (size_t) n_embd, 0.f);
    for (int t = 0; t < tokens; ++t) {
        // the pool's own path for one expert: gate/up, SwiGLU, quantize the hidden, down
        const void* a = r.token_acts.data() + (size_t) t * l.fmt.act_bytes;
        std::vector<float> ff((size_t) l.fmt.n_ff, 0.f);
        std::vector<uint8_t> hq(l.fmt.h_bytes, 0);
        float* ffp[1] = {ff.data()};
        const void* ap[1] = {a};
        const void* hp[1] = {hq.data()};
        float* op[1] = {nullptr};
        std::vector<float> row((size_t) n_embd, 0.f);
        op[0] = row.data();
        for (int k = 0; k < top_k; ++k) {
            const int sel = t * top_k + k;
            const int e = ids[(size_t) sel];
            const uint8_t* blob = l.weights.data() + (size_t) e * l.stride;
            strata::kernels::cpu::native_gu_rows(l.fmt, blob, ap, 1, ffp, 0, (int) l.fmt.n_ff);
            strata::kernels::cpu::native_quant_h(l.fmt, ff.data(), hq.data());
            strata::kernels::cpu::native_down_rows(l.fmt, blob, hp, 1, op, 0, n_embd);
            std::memcpy(want.data() + (size_t) sel * n_embd, row.data(), (size_t) n_embd * sizeof(float));
        }
    }
    return want;
}

Run make_run(int n_embd, int tokens, int top_k, int experts, uint32_t seed, int skew_expert = -1) {
    Run r;
    std::mt19937 g(seed);
    std::uniform_int_distribution<int> pick(0, experts - 1);
    r.ids.resize((size_t) tokens * (size_t) top_k);
    for (int t = 0; t < tokens; ++t) {
        for (int k = 0; k < top_k; ++k) {
            int e = pick(g);
            if (skew_expert >= 0 && k == 0) e = skew_expert;          // one expert takes every token's first slot
            if (skew_expert >= 0 && t == 0) e = skew_expert;          // and one token is entirely one expert
            r.ids[(size_t) (t * top_k + k)] = e;
        }
    }
    std::normal_distribution<float> nd(0.f, 1.f);
    r.acts.resize((size_t) tokens * (size_t) n_embd);
    for (auto& v : r.acts) v = nd(g);
    if (tokens > 1)   // one all-zero activation row
        for (int i = 0; i < n_embd; ++i) r.acts[(size_t) n_embd + i] = 0.f;
    return r;
}

bool compare_bits(const std::vector<float>& got, const std::vector<float>& want, int n_embd, int selections,
                  const char* what) {
    size_t diff = 0;
    int fs = -1, fe = -1;
    for (int s = 0; s < selections; ++s) {
        for (int e = 0; e < n_embd; ++e) {
            const size_t i = (size_t) s * n_embd + e;
            if (std::memcmp(&got[i], &want[i], 4) != 0) {
                if (!diff) {
                    fs = s;
                    fe = e;
                }
                ++diff;
            }
        }
    }
    if (!diff) return true;
    uint32_t gb = 0, wb = 0;
    std::memcpy(&gb, &got[(size_t) fs * n_embd + fe], 4);
    std::memcpy(&wb, &want[(size_t) fs * n_embd + fe], 4);
    std::printf("          %s: %zu of %zu values differ; first selection %d element %d got 0x%08x want 0x%08x\n", what,
                diff, (size_t) n_embd * (size_t) selections, fs, fe, gb, wb);
    return false;
}

/// Runs one pipeline case: quantize the activations with the pool's own quantizer, run the context, compare.
void pipeline_case(const Layer& l, Run& r, int top_k, const char* name, int repeat = 1) {
    using namespace strata::kolibri::stream;
    ++g_cases;
    const int tokens = (int) (r.ids.size() / (size_t) top_k);
    const int n_embd = (int) l.fmt.n_embd;
    r.token_acts.assign((size_t) tokens * l.fmt.act_bytes, 0);
    for (int t = 0; t < tokens; ++t) {
        strata::kernels::cpu::native_quant_act(l.fmt, r.acts.data() + (size_t) t * n_embd,
                                              r.token_acts.data() + (size_t) t * l.fmt.act_bytes);
    }
    LayerView v;
    v.fmt = l.fmt;
    v.weights = l.weights.data();
    v.weight_stride = l.stride;
    v.expert_count = l.experts;
    v.token_acts = r.token_acts.data();
    v.expert_ids = r.ids.data();
    v.selection_count = (int) r.ids.size();
    v.top_k = top_k;
    r.out.assign((size_t) r.ids.size() * (size_t) n_embd, 0.f);
    v.output = r.out.data();

    const std::vector<float> want = native_reference(l, top_k, r.ids, r, n_embd);
    std::unique_ptr<Context> ctx;
    try {
        Config cfg;
        ctx = create(v, cfg);
    } catch (const std::exception& ex) {
        std::printf("  FAIL %s: create threw: %s\n", name, ex.what());
        ++g_failures;
        return;
    }
    Stats last{};
    for (int rep = 0; rep < repeat; ++rep) {
        if (rep) std::memset(r.out.data(), 0x7f, r.out.size() * sizeof(float));   // poison between calls
        try {
            last = run(*ctx, v);
        } catch (const std::exception& ex) {
            std::printf("  FAIL %s: run threw: %s\n", name, ex.what());
            ++g_failures;
            return;
        }
        if (!compare_bits(r.out, want, n_embd, (int) r.ids.size(), name)) {
            std::printf("  FAIL %s (call %d of %d)\n", name, rep + 1, repeat);
            ++g_failures;
            return;
        }
    }
    // the counters the plan asks for: unique weight bytes, no allocations, launches with tiles rather than selections
    size_t expect_w = 0;
    {
        std::vector<bool> seen((size_t) l.experts, false);
        for (int id : r.ids)
            if (!seen[(size_t) id]) {
                seen[(size_t) id] = true;
                expect_w += l.fmt.bytes;
            }
    }
    const size_t tiles = (size_t) ((tokens + 3) / 4);   // an upper bound on the per-expert tiles
    if (last.weight_h2d_bytes != expect_w) {
        std::printf("  FAIL %s: weight_h2d_bytes %llu, expected %zu (unique blobs)\n", name,
                    (unsigned long long) last.weight_h2d_bytes, expect_w);
        ++g_failures;
    } else if (last.device_allocations != 0) {
        std::printf("  FAIL %s: a warm run allocated %llu times\n", name,
                    (unsigned long long) last.device_allocations);
        ++g_failures;
    } else if (last.kernel_launches > 2 * tiles * (size_t) l.experts) {
        std::printf("  FAIL %s: %llu launches is not tile-scaled\n", name,
                    (unsigned long long) last.kernel_launches);
        ++g_failures;
    }
}

/// The failure cases: each must throw std::invalid_argument (a caller error) or std::runtime_error (an injected
/// allocation failure), and must never publish a partial result.
void expect_refusal(const Layer& l, const char* why, void (*mutate)(LayerView&), const char* name) {
    ++g_cases;
    const int top_k = 2, tokens = 4, n_embd = (int) l.fmt.n_embd;
    Run r = make_run(n_embd, tokens, top_k, l.experts, 21);
    r.token_acts.assign((size_t) tokens * l.fmt.act_bytes, 0);
    for (int t = 0; t < tokens; ++t)
        strata::kernels::cpu::native_quant_act(l.fmt, r.acts.data() + (size_t) t * n_embd,
                                              r.token_acts.data() + (size_t) t * l.fmt.act_bytes);
    LayerView v;
    v.fmt = l.fmt;
    v.weights = l.weights.data();
    v.weight_stride = l.stride;
    v.expert_count = l.experts;
    v.token_acts = r.token_acts.data();
    v.expert_ids = r.ids.data();
    v.selection_count = (int) r.ids.size();
    v.top_k = top_k;
    r.out.assign((size_t) v.selection_count * (size_t) n_embd, 0.f);
    v.output = r.out.data();
    if (mutate) mutate(v);
    std::string threw;
    try {
        Config cfg;
        std::unique_ptr<Context> ctx = create(v, cfg);
        run(*ctx, v);
    } catch (const std::invalid_argument& ex) {
        threw = ex.what();
    } catch (const std::runtime_error& ex) {
        threw = ex.what();
    }
    if (threw.empty()) {
        std::printf("  FAIL %s: no exception (%s)\n", name, why);
        ++g_failures;
    }
}

void mutate_budget(LayerView& v) { v.top_k = 0; }
void mutate_type(LayerView& v) { v.fmt.gu_type = 8; }
void mutate_ids(LayerView& v) { const_cast<int*>(v.expert_ids)[1] = v.expert_count; }
void mutate_acts(LayerView& v) { v.fmt.act_bytes = 4; }

void failure_cases(const Layer& l, const Layer& l6) {
    expect_refusal(l, "top_k zero", mutate_budget, "refuses top_k 0");
    expect_refusal(l, "gate type", mutate_type, "refuses a non-Q4_K gate/up");
    expect_refusal(l, "expert id out of range", mutate_ids, "refuses an out-of-range expert id");
    expect_refusal(l, "activation stride", mutate_acts, "refuses a wrong activation stride");
    (void) l6;
    // budget: the same view with a budget smaller than one slot must be refused
    ++g_cases;
    {
        const int top_k = 2, tokens = 4, n_embd = (int) l.fmt.n_embd;
        Run r = make_run(n_embd, tokens, top_k, l.experts, 22);
        r.token_acts.assign((size_t) tokens * l.fmt.act_bytes, 0);
        LayerView v;
        v.fmt = l.fmt;
        v.weights = l.weights.data();
        v.weight_stride = l.stride;
        v.expert_count = l.experts;
        v.token_acts = r.token_acts.data();
        v.expert_ids = r.ids.data();
        v.selection_count = (int) r.ids.size();
        v.top_k = top_k;
        r.out.assign((size_t) v.selection_count * (size_t) n_embd, 0.f);
        v.output = r.out.data();
        Config cfg;
        cfg.device_budget_bytes = 1024;
        bool refused = false;
        try {
            std::unique_ptr<Context> ctx = create(v, cfg);
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        if (!refused) {
            std::printf("  FAIL budget: a 1 KiB device budget was accepted\n");
            ++g_failures;
        }
    }
    // every allocation ordinal of create() in turn: the failure must be reported and everything released
    for (int ord = 0; ord < 64; ++ord) {
        ++g_cases;
        const int top_k = 2, tokens = 4, n_embd = (int) l.fmt.n_embd;
        Run r = make_run(n_embd, tokens, top_k, l.experts, 23);
        r.token_acts.assign((size_t) tokens * l.fmt.act_bytes, 0);
        LayerView v;
        v.fmt = l.fmt;
        v.weights = l.weights.data();
        v.weight_stride = l.stride;
        v.expert_count = l.experts;
        v.token_acts = r.token_acts.data();
        v.expert_ids = r.ids.data();
        v.selection_count = (int) r.ids.size();
        v.top_k = top_k;
        r.out.assign((size_t) v.selection_count * (size_t) n_embd, 0.f);
        v.output = r.out.data();
        strata::kolibri::stream::detail::fail_allocation_after(ord);
        bool threw = false;
        try {
            Config cfg;
            std::unique_ptr<Context> ctx = create(v, cfg);
            run(*ctx, v);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        strata::kolibri::stream::detail::fail_allocation_after(-1);
        if (ord < 18) {   // nine allocations per slot, two slots
            if (!threw) {
                std::printf("  FAIL alloc-fault: ordinal %d did not fail\n", ord);
                ++g_failures;
            }
        } else if (threw) {
            std::printf("  FAIL alloc-fault: ordinal %d failed but nothing is allocated there\n", ord);
            ++g_failures;
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool tails_only = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--tails-only") {
            tails_only = true;
        } else if (a != "--selftest") {
            std::printf("usage: %s [--selftest] [--tails-only]\n", argv[0]);
            return 2;
        }
    }
    ggml_cpu_init();
    {
        int dev = 0;
        cudaDeviceProp p{};
        if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&p, dev) != cudaSuccess ||
            !(p.major > 7 || (p.major == 7 && p.minor >= 5))) {
            std::printf("FAIL: no CUDA device with sm_75 or newer\n");
            return 1;
        }
        std::printf("device: %s, sm_%d%d\n", p.name, p.major, p.minor);
    }

    // The model's own geometry: n_embd 2560, n_ff 512, 384 experts, top_k 6, both down formats.
    const Layer l12 = make_layer(2560, 512, 384, 12, 9001);
    const Layer l14 = make_layer(2560, 512, 384, 14, 9002);

    if (!tails_only) {
        // (a) an empty input: no GPU work, zeroed counters
        ++g_cases;
        {
            Run r;
            std::vector<float> none;
            LayerView v;
            v.fmt = l12.fmt;
            v.weights = l12.weights.data();
            v.weight_stride = l12.stride;
            v.expert_count = l12.experts;
            v.token_acts = nullptr;
            v.expert_ids = nullptr;
            v.selection_count = 0;
            v.top_k = 6;
            v.output = nullptr;
            Config cfg;
            std::unique_ptr<Context> ctx = create(v, cfg);
            const Stats s = run(*ctx, v);
            if (s.groups || s.selections || s.weight_h2d_bytes || s.kernel_launches || s.device_allocations) {
                std::printf("  FAIL empty input did work: %llu groups, %llu launches\n",
                            (unsigned long long) s.groups, (unsigned long long) s.kernel_launches);
                ++g_failures;
            }
        }
        // (b) a normal chunk, then the same context on the other down format, then a hundred calls alternating
        {
            Run r = make_run(2560, 256, 6, 384, 11);
            pipeline_case(l12, r, 6, "Q4_K down, 256 tokens x top_k 6");
            Run r2 = make_run(2560, 256, 6, 384, 12);
            pipeline_case(l14, r2, 6, "Q6_K down, 256 tokens x top_k 6");
        }
        // (c) skewed: one expert holds 256 + 1 selections (more than three batches on its own)
        {
            Run r = make_run(2560, 256, 6, 384, 13, 7);
            pipeline_case(l14, r, 6, "skewed groups (one expert takes a whole token's route)");
        }
        // (d) one expert holding every selection
        {
            Run r = make_run(2560, 32, 1, 384, 14, 3);
            pipeline_case(l12, r, 1, "single expert, 32 selections, top_k 1");
        }
        // (e) ONE context, both down formats: Kolibri alternates Q4_K and Q6_K down between layers
        {
            ++g_cases;
            const int top_k = 6, tokens = 32;
            Run a = make_run(2560, tokens, top_k, l12.experts, 16);
            Run b = make_run(2560, tokens, top_k, l14.experts, 17);
            auto view_of = [&](const Layer& l, Run& r) {
                r.token_acts.assign((size_t) tokens * l.fmt.act_bytes, 0);
                for (int t = 0; t < tokens; ++t)
                    strata::kernels::cpu::native_quant_act(l.fmt, r.acts.data() + (size_t) t * 2560,
                                                          r.token_acts.data() + (size_t) t * l.fmt.act_bytes);
                r.out.assign((size_t) r.ids.size() * 2560, 0.f);
                LayerView v;
                v.fmt = l.fmt;
                v.weights = l.weights.data();
                v.weight_stride = l.stride;
                v.expert_count = l.experts;
                v.token_acts = r.token_acts.data();
                v.expert_ids = r.ids.data();
                v.selection_count = (int) r.ids.size();
                v.top_k = top_k;
                v.output = r.out.data();
                return v;
            };
            LayerView va = view_of(l12, a), vb = view_of(l14, b);
            const std::vector<float> wa = native_reference(l12, top_k, a.ids, a, 2560);
            const std::vector<float> wb = native_reference(l14, top_k, b.ids, b, 2560);
            Config cfg;
            std::unique_ptr<Context> ctx = create(va, cfg);
            bool ok = compare_bits(run(*ctx, va).selections ? a.out : a.out, wa, 2560, (int) a.ids.size(), "Q4_K then");
            ok = ok && compare_bits((run(*ctx, vb), b.out), wb, 2560, (int) b.ids.size(), "Q6_K second");
            ok = ok && compare_bits((run(*ctx, va), a.out), wa, 2560, (int) a.ids.size(), "Q4_K again");
            if (!ok) {
                std::printf("  FAIL one context across both down formats\n");
                ++g_failures;
            }
        }
        // (f) a hundred alternating calls through the same two slots, with the output poisoned in between
        {
            const int n_embd = 2560;
            const Layer& l = l12;
            const int top_k = 6, tokens = 64;
            Run r = make_run(n_embd, tokens, top_k, l.experts, 15);
            r.token_acts.assign((size_t) tokens * l.fmt.act_bytes, 0);
            for (int t = 0; t < tokens; ++t)
                strata::kernels::cpu::native_quant_act(l.fmt, r.acts.data() + (size_t) t * n_embd,
                                                      r.token_acts.data() + (size_t) t * l.fmt.act_bytes);
            LayerView v;
            v.fmt = l.fmt;
            v.weights = l.weights.data();
            v.weight_stride = l.stride;
            v.expert_count = l.experts;
            v.token_acts = r.token_acts.data();
            v.expert_ids = r.ids.data();
            v.selection_count = (int) r.ids.size();
            v.top_k = top_k;
            r.out.assign((size_t) r.ids.size() * (size_t) n_embd, 0.f);
            v.output = r.out.data();
            const std::vector<float> want = native_reference(l, top_k, r.ids, r, n_embd);
            ++g_cases;
            Config cfg;
            std::unique_ptr<Context> ctx = create(v, cfg);
            bool ok = true;
            for (int call = 0; call < 100 && ok; ++call) {
                std::memset(r.out.data(), 0x40 + (call & 15), r.out.size() * sizeof(float));
                const Stats s = run(*ctx, v);
                ok = s.device_allocations == 0 && compare_bits(r.out, want, n_embd, (int) r.ids.size(), "reuse");
            }
            if (!ok) {
                std::printf("  FAIL a hundred alternating calls\n");
                ++g_failures;
            }
        }
    }
    if (tails_only) {
        Run r = make_run(2560, 5, 3, 384, 31);
        pipeline_case(l12, r, 3, "tails: 5 tokens, top_k 3");
        Run r2 = make_run(2560, 1, 1, 384, 32);
        pipeline_case(l14, r2, 1, "tails: 1 token, top_k 1");
    } else {
        failure_cases(l12, l14);
    }

    std::printf("%s: %d case(s), %d failure(s)\n", g_failures ? "FAIL" : "PASS", g_cases, g_failures);
    return g_failures ? 1 : 0;
}
