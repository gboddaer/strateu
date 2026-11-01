// tests/kolibri/expert_stream_plan_test.cpp - Task 1 of plan 2026-10-31-kolibri-gpu-streaming.
//
// The GPU-streaming plan's host half, tested without CUDA, without ggml and without a model: the stable grouping of
// a chunk's selections by expert, and the exact capacity accounting the slots are allocated from.  Both are pinned
// to numbers here so a change in the accounting (or in the group order the kernels will rely on) fails the test
// instead of quietly resizing a device buffer.
//
//   kolibri_expert_stream_plan_test          run every case
#include "kolibri/expert_stream_plan.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace stream = strata::kolibri::stream;
namespace cpu = strata::kernels::cpu;

static int g_failed = 0;

static void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failed;
    }
}

// The initial path's only supported layer: Q4_K gate/up, Q4_K or Q6_K down, Q8_K activations, blob
// [gate rows | up rows | down rows].  Built by hand (not through native_fmt) so the test needs no ggml.
static cpu::NativeFmt make_fmt(int64_t n_embd, int64_t n_ff, int d_type = stream::kQ6K) {
    cpu::NativeFmt f;
    f.gu_type = stream::kQ4K;
    f.d_type = d_type;
    f.gu_act = stream::kQ8K;
    f.d_act = stream::kQ8K;
    f.n_embd = n_embd;
    f.n_ff = n_ff;
    f.gu_row = stream::q4k_row_bytes(n_embd);
    f.d_row = d_type == stream::kQ4K ? stream::q4k_row_bytes(n_ff) : stream::q6k_row_bytes(n_ff);
    f.up_off = (size_t) n_ff * f.gu_row;
    f.down_off = 2 * (size_t) n_ff * f.gu_row;
    f.bytes = f.down_off + (size_t) n_embd * f.d_row;
    f.act_bytes = stream::q8k_row_bytes(n_embd);
    f.h_bytes = stream::q8k_row_bytes(n_ff);
    return f;
}

// A capacity view over caller-owned buffers.  The accounting must not read them, so the values are irrelevant.
struct View {
    cpu::NativeFmt fmt;
    std::vector<uint8_t> weights;
    std::vector<uint8_t> acts;
    std::vector<int> ids;
    std::vector<float> out;
    stream::LayerView view;

    View(const cpu::NativeFmt& f, int selections, int top_k, int experts, size_t stride) : fmt(f) {
        weights.assign(stride * (size_t) experts, 0);
        acts.assign((size_t) (top_k ? selections / top_k : 0) * f.act_bytes, 0);
        ids.assign((size_t) (selections > 0 ? selections : 0), 0);
        out.assign((size_t) (selections > 0 ? selections : 0) * (size_t) f.n_embd, 0.f);
        view.fmt = fmt;
        view.weights = weights.data();
        view.weight_stride = stride;
        view.expert_count = experts;
        view.token_acts = acts.data();
        view.expert_ids = ids.data();
        view.selection_count = selections;
        view.top_k = top_k;
        view.output = out.data();
    }
};

// The same view with no buffers: for geometries whose arithmetic must be refused BEFORE anything is allocated.
struct Bare {
    cpu::NativeFmt fmt;
    stream::LayerView view;
    static int* ids() { static int z[8] = {0}; return z; }
    static uint8_t* bytes() { static uint8_t z[256] = {0}; return z; }
    static float* floats() { static float z[16] = {0}; return z; }
    Bare(const cpu::NativeFmt& f, int selections, int top_k, int experts, size_t stride) : fmt(f) {
        view.fmt = fmt;
        view.weights = bytes();
        view.weight_stride = stride;
        view.expert_count = experts;
        view.token_acts = bytes();
        view.expert_ids = ids();
        view.selection_count = selections;
        view.top_k = top_k;
        view.output = floats();
    }
};

static bool throws_invalid(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

int main() {
    // ---- grouping: ascending expert, ascending original selection index within a group -----------------------
    {
        const int ids[] = {2, 0, 2, 1, 0, 1};
        const std::vector<stream::Group> g = stream::make_groups(ids, 6, 2, 3);
        check(g.size() == 3, "group count for {2,0,2,1,0,1}");
        if (g.size() == 3) {
            check(g[0].expert == 0 && g[0].selections == std::vector<int>({1, 4}), "group 0 = {1,4}");
            check(g[1].expert == 1 && g[1].selections == std::vector<int>({3, 5}), "group 1 = {3,5}");
            check(g[2].expert == 2 && g[2].selections == std::vector<int>({0, 2}), "group 2 = {0,2}");
        }
    }
    // ---- zero selections is legal and does no work ------------------------------------------------------------
    {
        check(stream::make_groups(nullptr, 0, 2, 3).empty(), "zero selections groups to nothing");
        const cpu::NativeFmt zf = make_fmt(256, 512);
        View v(zf, 0, 2, 4, zf.bytes);
        const stream::Capacity c = stream::required_capacity(v.view, stream::Config{});
        check(c.max_selections == 0, "zero selections: max_selections 0");
        // two slots of {one blob-batch worth of weights} + one descriptor each, nothing per selection
        check(c.weight_bytes_per_slot == 8u * zf.bytes, "zero selections: per-slot weight bytes");
        check(c.device_bytes == 2u * (8u * zf.bytes + stream::kDescriptorBytes), "zero selections: device bytes");
    }
    // ---- invalid arguments are refused, never silently accepted ------------------------------------------------
    {
        const int bad_hi[] = {0, 3};
        check(throws_invalid([&] { stream::make_groups(bad_hi, 2, 2, 3); }), "expert id == experts is refused");
        const int bad_lo[] = {0, -1};
        check(throws_invalid([&] { stream::make_groups(bad_lo, 2, 2, 3); }), "negative expert id is refused");
        const int ok[] = {0, 1};
        check(throws_invalid([&] { stream::make_groups(ok, 2, 0, 3); }), "top_k 0 is refused");
        check(throws_invalid([&] { stream::make_groups(ok, 3, 2, 3); }), "selections not divisible by top_k is refused");
        check(throws_invalid([&] { stream::make_groups(ok, -2, 2, 3); }), "negative selection count is refused");
        check(throws_invalid([&] { stream::make_groups(nullptr, 2, 2, 3); }), "null ids with selections is refused");
        check(throws_invalid([&] { stream::make_groups(ok, 2, 2, 0); }), "zero experts is refused");
    }
    // ---- a group larger than any fixed pointer array stays one group -------------------------------------------
    {
        std::vector<int> ids(200, 3);
        const std::vector<stream::Group> g = stream::make_groups(ids.data(), 200, 1, 8);
        check(g.size() == 1, "200 selections of one expert = one group");
        if (g.size() == 1) {
            check(g[0].expert == 3 && g[0].selections.size() == 200, "the 200-selection group keeps its size");
            check(g[0].selections.front() == 0 && g[0].selections.back() == 199, "the group keeps original order");
        }
        const cpu::NativeFmt kf = make_fmt(2560, 512);   // n_embd 2560, n_ff 512: 2,549,760 B per blob
        View v(kf, 200, 1, 8, kf.bytes);
        const stream::Capacity c = stream::required_capacity(v.view, stream::Config{});
        check(c.max_selections == 200, "capacity is sized for the whole 200-selection group");
    }
    // ---- capacity: exact numbers, both slots, alignment, descriptors ------------------------------------------
    {
        const cpu::NativeFmt kf = make_fmt(2560, 512);   // the model's own expert shape: 2,549,760 B per blob
        // n_embd 256 (act 292 B), n_ff 512 (hidden 584 B), 6 selections, the blob's own stride
        const cpu::NativeFmt sf = make_fmt(256, 512);
        View v(sf, 6, 2, 4, sf.bytes);
        const stream::Capacity c = stream::required_capacity(v.view, stream::Config{});
        const size_t weights = 8u * sf.bytes;           // the batch's eight blobs, under the 32 MiB cap
        const size_t acts = 6u * 292u;                  // 1,752 -> 1,792 aligned
        const size_t gu = 6u * 2u * 512u * 4u;          // 24,576: gate and up floats per selection
        const size_t hidden = 6u * 584u;                // 3,504 -> 3,584 aligned
        const size_t out = 6u * 256u * 4u;              // 6,144: n_embd floats per selection
        const size_t slot = weights + 1792u + gu + 3584u + out + stream::kDescriptorBytes;
        check(c.weight_bytes_per_slot == weights, "per-slot weight bytes are capped by experts_per_batch");
        check(c.max_selections == 6, "max_selections is the chunk's selection count");
        check(c.device_bytes == 2u * slot, "device bytes = two slots of aligned buffers + descriptors");
        check(c.pinned_bytes == 2u * slot, "pinned bytes mirror the two slots");
        // the batch bound wins while eight real blobs fit under the cap: 8 x 2,549,760 B = 19.5 MiB
        Bare real(kf, 6, 2, 4, kf.bytes);
        const stream::Capacity cr = stream::required_capacity(real.view, stream::Config{});
        check(cr.weight_bytes_per_slot == 8u * kf.bytes, "per-slot weights are the eight-blob batch when it fits");
        // and the 32 MiB cap wins when it does not: 8 x 5 MiB = 40 MiB
        Bare wide(kf, 6, 2, 4, 5u << 20);
        const stream::Capacity cw = stream::required_capacity(wide.view, stream::Config{});
        check(cw.weight_bytes_per_slot == 32u << 20, "a batch above the slot cap is capped at 32 MiB");
        // the budget is a refusal, not a resize
        stream::Config tight;
        tight.device_budget_bytes = 1024;
        check(throws_invalid([&] { stream::required_capacity(v.view, tight); }), "a slot set larger than the budget is refused");
    }
    // ---- overflow is refused before any allocation ------------------------------------------------------------
    {
        // eight blobs of weight_stride: 8 * (SIZE_MAX/2) wraps a size_t
        const cpu::NativeFmt of = make_fmt(256, 512);
        Bare v(of, 6, 2, 1, std::numeric_limits<size_t>::max() / 2);
        check(throws_invalid([&] { stream::required_capacity(v.view, stream::Config{}); }), "weight-batch overflow is refused");
        // INT_MAX selections x a Q8_K row of a 2^40-value embedding: the per-selection byte count wraps a size_t
        Bare huge(make_fmt(int64_t(1) << 40, 512), std::numeric_limits<int>::max(), 1, 1,
                  make_fmt(int64_t(1) << 40, 512).bytes);
        check(throws_invalid([&] { stream::required_capacity(huge.view, stream::Config{}); }), "selection-byte overflow is refused");
    }
    // ---- unsupported formats are refused by name, not silently executed ---------------------------------------
    {
        const cpu::NativeFmt bf = make_fmt(256, 512);
        View base(bf, 6, 2, 4, bf.bytes);
        stream::LayerView v = base.view;
        v.fmt.tail_off = 16;
        check(throws_invalid([&] { stream::required_capacity(v, stream::Config{}); }), "an NVFP4 tail is refused");
        v = base.view;
        v.fmt.had_block = 256;
        check(throws_invalid([&] { stream::required_capacity(v, stream::Config{}); }), "a rotated-basis layer is refused");
        v = base.view;
        v.fmt.had_gu = true;
        check(throws_invalid([&] { stream::required_capacity(v, stream::Config{}); }), "a per-projection rotation is refused");
        v = base.view;
        v.fmt.gu_type = 8;   // a non-Q4_K gate/up
        check(throws_invalid([&] { stream::required_capacity(v, stream::Config{}); }), "a non-Q4_K gate/up is refused");
        v = base.view;
        v.fmt.d_type = 8;   // a non-Q4_K/Q6_K down
        check(throws_invalid([&] { stream::required_capacity(v, stream::Config{}); }), "a non-Q4_K/Q6_K down is refused");
        v = base.view;
        v.fmt.gu_act = 8;   // not Q8_K
        check(throws_invalid([&] { stream::required_capacity(v, stream::Config{}); }), "a non-Q8_K gate/up activation is refused");
        v = base.view;
        v.fmt.act_bytes += 1;
        check(throws_invalid([&] { stream::required_capacity(v, stream::Config{}); }), "an act_bytes that is not Q8_K of n_embd is refused");
        v = base.view;
        v.weights = nullptr;
        check(throws_invalid([&] { stream::required_capacity(v, stream::Config{}); }), "a missing weight table is refused");
        v = base.view;
        v.expert_ids = nullptr;
        check(throws_invalid([&] { stream::required_capacity(v, stream::Config{}); }), "missing selection ids are refused");
        v = base.view;
        v.output = nullptr;
        check(throws_invalid([&] { stream::required_capacity(v, stream::Config{}); }), "a missing output view is refused");
        v = base.view;
        v.weight_stride = 128;   // shorter than one blob
        check(throws_invalid([&] { stream::required_capacity(v, stream::Config{}); }), "a stride shorter than the blob is refused");
        stream::Config cfg;
        cfg.experts_per_batch = 0;
        v = base.view;
        check(throws_invalid([&] { stream::required_capacity(v, cfg); }), "experts_per_batch 0 is refused");
        cfg = stream::Config{};
        cfg.experts_per_batch = stream::kMaxExpertsPerBatch + 1;
        check(throws_invalid([&] { stream::required_capacity(v, cfg); }), "experts_per_batch above the cap is refused");
        cfg = stream::Config{};
        cfg.weight_bytes_per_slot = 0;
        check(throws_invalid([&] { stream::required_capacity(v, cfg); }), "a zero weight cap is refused");
    }

    if (g_failed != 0) {
        std::fprintf(stderr, "kolibri_expert_stream_plan_test: %d failure(s)\n", g_failed);
        return 1;
    }
    std::printf("kolibri_expert_stream_plan_test: grouping and capacity accounting pass\n");
    return 0;
}
