// src/kolibri/expert_stream.cu - plan 2026-10-31-kolibri-gpu-streaming, Task 3.
//
// Two-slot streaming pipeline.  Slot lifecycle per batch:
//     FREE -> H2D -> GU -> GU_D2H -> CPU_ACT_QUANT -> HIDDEN_H2D -> DOWN -> OUT_D2H -> FREE
//
// While one slot is in H2D/GU the host is in the other slot's CPU_ACT_QUANT/DOWN, so the serial CPU SwiGLU and
// quantization (which must stay exact, and must be the pool's own arithmetic) do not idle the copy engine or the
// SMs.  The batch is what is bounded: eight groups and the weight cap per slot, from expert_stream_plan.hpp.
//
// Nothing here allocates during a run, blocks on the device as a whole, or publishes a result before its event has
// completed.  Weights are copied straight from the caller's table (the engine's is pinned, so the copy is real
// asynchronous DMA); the gathered activations and every result go through the slot's own pinned staging, because
// those are produced or consumed on the host.
#include "expert_stream.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "expert_stream_kernels.cuh"
#include "strata/kernels/cpu/native_expert.hpp"

namespace strata::kolibri::stream {
namespace {

[[noreturn]] void refuse(const std::string& why) { throw std::runtime_error("expert streaming: " + why); }

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) refuse(std::string(what) + ": " + cudaGetErrorString(e));
}

/// A byte counter for the test-only allocation-failure injection (never read by production paths).
int g_fail_ordinal = -1;
int g_alloc_ordinal = 0;

void* dev_alloc(size_t bytes) {
    if (g_fail_ordinal >= 0 && g_alloc_ordinal++ == g_fail_ordinal) refuse("injected device allocation failure");
    void* p = nullptr;
    ck(cudaMalloc(&p, bytes ? bytes : 1), "cudaMalloc");
    return p;
}

void* pin_alloc(size_t bytes) {
    if (g_fail_ordinal >= 0 && g_alloc_ordinal++ == g_fail_ordinal) refuse("injected pinned allocation failure");
    void* p = nullptr;
    ck(cudaHostAlloc(&p, bytes ? bytes : 1, cudaHostAllocDefault), "cudaHostAlloc");
    return p;
}

struct Slot {
    uint8_t* weights = nullptr;    ///< packed blobs of the current batch
    uint8_t* acts = nullptr;       ///< gathered Q8_K rows, one per selection of the batch
    float* gu = nullptr;           ///< gate/up rows, 2 * n_ff floats each
    uint8_t* hidden = nullptr;     ///< quantized hidden rows, one per selection
    float* out = nullptr;          ///< down rows, n_embd floats each
    uint8_t* acts_pin = nullptr;
    float* gu_pin = nullptr;
    uint8_t* hidden_pin = nullptr;
    float* out_pin = nullptr;
    cudaStream_t st = nullptr;
    cudaEvent_t ev_gu = nullptr;    ///< gate/up results are on the host
    cudaEvent_t ev_out = nullptr;   ///< down results are on the host
};

/// One batch: the groups that share a slot, and the row (selection) offset of each of them.
struct Batch {
    std::vector<int> group;        ///< indices into the run's group vector
    std::vector<size_t> row;       ///< first gathered row of each group
    size_t rows = 0;
    size_t weight_bytes = 0;
};

}  // namespace

struct Context::Impl {
    Config cfg;
    Capacity cap;
    // The geometry the slots were sized for: everything except the down format is fixed across layers.
    int n_embd = 0, n_ff = 0, top_k_hint = 0;
    size_t act_bytes = 0, h_bytes = 0, gu_row = 0, up_off = 0;
    size_t weight_cap = 0;   ///< bytes for one batch's blobs (the plan's 32 MiB, or one blob if larger)
    Slot s[2];
    std::vector<float> ff;   ///< host scratch: one hidden row before quantization (n_ff floats)

    ~Impl() {
        for (Slot& x : s) {
            if (x.st) cudaStreamSynchronize(x.st);
            for (cudaEvent_t e : {x.ev_gu, x.ev_out})
                if (e) cudaEventDestroy(e);
            for (void* p : {(void*) x.weights, (void*) x.acts, (void*) x.gu, (void*) x.hidden, (void*) x.out})
                if (p) cudaFree(p);
            for (void* p : {(void*) x.acts_pin, (void*) x.gu_pin, (void*) x.hidden_pin, (void*) x.out_pin})
                if (p) cudaFreeHost(p);
            if (x.st) cudaStreamDestroy(x.st);
        }
    }
};

namespace {

/// The capacity of a view whose down type has been swapped: create() must size for the larger of the two formats.
LayerView with_down_type(const LayerView& v, int d_type) {
    LayerView w = v;
    w.fmt.d_type = d_type;
    w.fmt.d_row = d_type == kQ4K ? q4k_row_bytes(v.fmt.n_ff) : q6k_row_bytes(v.fmt.n_ff);
    w.fmt.bytes = (size_t) 2 * v.fmt.n_ff * v.fmt.gu_row + (size_t) v.fmt.n_embd * w.fmt.d_row;
    if (w.weight_stride < w.fmt.bytes) w.weight_stride = w.fmt.bytes;   // the other format's blob is larger
    return w;
}

Capacity max_capacity(const Capacity& a, const Capacity& b) {
    Capacity c = a;
    if (b.device_bytes > c.device_bytes) c.device_bytes = b.device_bytes;
    if (b.pinned_bytes > c.pinned_bytes) c.pinned_bytes = b.pinned_bytes;
    if (b.weight_bytes_per_slot > c.weight_bytes_per_slot) c.weight_bytes_per_slot = b.weight_bytes_per_slot;
    if (b.max_selections > c.max_selections) c.max_selections = b.max_selections;
    return c;
}

/// Per-slot buffer sizes, derived from the run geometry exactly as required_capacity() does.
struct Sizes {
    size_t weights, acts, gu, hidden, out;
};

Sizes slot_sizes(const LayerView& v, const Config& cfg, const Capacity& cap) {
    Sizes z{};
    z.weights = cap.weight_bytes_per_slot;
    z.acts = (size_t) v.selection_count * v.fmt.act_bytes;
    z.gu = (size_t) v.selection_count * 2 * (size_t) v.fmt.n_ff * sizeof(float);
    z.hidden = (size_t) v.selection_count * v.fmt.h_bytes;
    z.out = (size_t) v.selection_count * (size_t) v.fmt.n_embd * sizeof(float);
    (void) cfg;
    return z;
}

/// Splits the run's groups into batches: at most `experts_per_batch` groups and the weight cap per batch.  A single
/// group larger than the cap still runs, alone, because the cap is never smaller than one blob.
std::vector<Batch> make_batches(const std::vector<Group>& groups, const Config& cfg, size_t weight_cap, size_t blob) {
    std::vector<Batch> out;
    Batch b;
    size_t row = 0;
    for (size_t g = 0; g < groups.size(); ++g) {
        const bool full = (int) b.group.size() >= cfg.experts_per_batch ||
                          (!b.group.empty() && b.weight_bytes + blob > weight_cap);
        if (full) {
            out.push_back(b);
            b = Batch{};
            row = 0;
        }
        b.group.push_back((int) g);
        b.row.push_back(row);
        row += groups[g].selections.size();
        b.rows = row;
        b.weight_bytes += blob;
        if ((int) b.group.size() >= cfg.experts_per_batch) {
            out.push_back(b);
            b = Batch{};
            row = 0;
        }
    }
    if (!b.group.empty()) out.push_back(b);
    return out;
}

}  // namespace

void detail::fail_allocation_after(int ordinal) {
    g_fail_ordinal = ordinal;
    g_alloc_ordinal = 0;
}

std::unique_ptr<Context> create(const LayerView& capacity_view, const Config& config) {
    const Capacity c1 = required_capacity(capacity_view, config);
    const Capacity c2 = required_capacity(with_down_type(capacity_view, capacity_view.fmt.d_type == kQ4K ? kQ6K : kQ4K),
                                          config);
    auto m = std::unique_ptr<Context::Impl>(new Context::Impl());
    m->cfg = config;
    m->cap = max_capacity(c1, c2);
    m->n_embd = (int) capacity_view.fmt.n_embd;
    m->n_ff = (int) capacity_view.fmt.n_ff;
    m->top_k_hint = capacity_view.top_k;
    m->act_bytes = capacity_view.fmt.act_bytes;
    m->h_bytes = capacity_view.fmt.h_bytes;
    m->gu_row = capacity_view.fmt.gu_row;
    m->up_off = capacity_view.fmt.up_off;
    m->weight_cap = m->cap.weight_bytes_per_slot;
    m->ff.assign((size_t) m->n_ff, 0.f);
    const Sizes z = slot_sizes(capacity_view, config, m->cap);
    g_alloc_ordinal = 0;
    try {
        for (Slot& x : m->s) {
            x.weights = (uint8_t*) dev_alloc(z.weights);
            x.acts = (uint8_t*) dev_alloc(z.acts);
            x.gu = (float*) dev_alloc(z.gu);
            x.hidden = (uint8_t*) dev_alloc(z.hidden);
            x.out = (float*) dev_alloc(z.out);
            x.acts_pin = (uint8_t*) pin_alloc(z.acts);
            x.gu_pin = (float*) pin_alloc(z.gu);
            x.hidden_pin = (uint8_t*) pin_alloc(z.hidden);
            x.out_pin = (float*) pin_alloc(z.out);
            ck(cudaStreamCreateWithFlags(&x.st, cudaStreamNonBlocking), "cudaStreamCreate");
            ck(cudaEventCreateWithFlags(&x.ev_gu, cudaEventDisableTiming), "cudaEventCreate");
            ck(cudaEventCreateWithFlags(&x.ev_out, cudaEventDisableTiming), "cudaEventCreate");
        }
    } catch (...) {
        g_fail_ordinal = -1;   // the destructor releases whatever was created; never let the injection leak into it
        throw;
    }
    g_fail_ordinal = -1;
    return std::unique_ptr<Context>(new Context(std::move(m)));
}

Context::Context(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Context::~Context() = default;

Stats run(Context& ctx, const LayerView& view) {
    Context::Impl& m = *ctx.impl_;
    const auto t_start = std::chrono::steady_clock::now();
    if (g_fail_ordinal >= 0) {
        // never inject through a run: a run allocates nothing
        g_alloc_ordinal = 0;
    }
    // ---- validation: required_capacity() names the unsupported formats, sizes and ids ----------------------
    const Capacity need = required_capacity(view, m.cfg);
    // The slots were sized by max_selections and weight_bytes_per_slot (create() took the larger of the two down
    // formats), so those two are the bounds a later view must respect.  Comparing whole Capacity totals instead
    // would refuse the larger down format whenever the layer's stride padding differs from the created one.
    if (need.max_selections > m.cap.max_selections)
        throw std::invalid_argument("expert streaming: this view has more selections than the slots hold");
    if (view.fmt.bytes > m.cap.weight_bytes_per_slot)
        throw std::invalid_argument("expert streaming: this view's expert blob is larger than a slot's weight budget");
    const LayerView& f = view;
    if ((int) f.fmt.n_embd != m.n_embd || (int) f.fmt.n_ff != m.n_ff || f.fmt.act_bytes != m.act_bytes ||
        f.fmt.h_bytes != m.h_bytes || f.fmt.gu_row != m.gu_row || f.fmt.up_off != m.up_off)
        throw std::invalid_argument("expert streaming: the view's geometry differs from the created one");

    Stats st{};
    st.selections = (uint64_t) view.selection_count;
    const std::vector<Group> groups = make_groups(view.expert_ids, view.selection_count, view.top_k, view.expert_count);
    st.groups = (uint64_t) groups.size();
    if (groups.empty()) {
        st.wall_ms = 0.0;
        return st;   // empty input: no GPU work at all
    }
    const size_t blob = f.fmt.bytes;
    const size_t weight_cap = m.weight_cap < need.weight_bytes_per_slot ? m.weight_cap : need.weight_bytes_per_slot;
    const std::vector<Batch> batches = make_batches(groups, m.cfg, weight_cap, blob);

    auto stage_gu = [&](const Batch& b, Slot& s) {
        size_t woff = 0, row = 0;
        for (int gi : b.group) {
            const Group& g = groups[(size_t) gi];
            const uint8_t* src = view.weights + (size_t) g.expert * view.weight_stride;
            ck(cudaMemcpyAsync(s.weights + woff, src, blob, cudaMemcpyHostToDevice, s.st), "weights H2D");
            st.weight_h2d_bytes += blob;
            // gather this group's activations (one Q8_K row per selection, in its original order)
            for (int sel : g.selections) {
                const uint8_t* a = view.token_acts + (size_t) (sel / view.top_k) * f.fmt.act_bytes;
                std::memcpy(s.acts_pin + row * f.fmt.act_bytes, a, f.fmt.act_bytes);
                ++row;
            }
            woff += blob;
        }
        ck(cudaMemcpyAsync(s.acts, s.acts_pin, b.rows * f.fmt.act_bytes, cudaMemcpyHostToDevice, s.st), "acts H2D");
        st.activation_h2d_bytes += b.rows * f.fmt.act_bytes;
        // one gate/up launch per expert: rows [gate | up] are contiguous at the same stride
        woff = 0;
        for (size_t i = 0; i < b.group.size(); ++i) {
            const Group& g = groups[(size_t) b.group[i]];
            const int ntok = (int) g.selections.size();
            const int rc = es_q4k_rows(s.weights + woff, (int) (f.fmt.n_embd / 256), (int) (2 * f.fmt.n_ff),
                                       s.acts + b.row[i] * f.fmt.act_bytes, (int) f.fmt.act_bytes, ntok,
                                       s.gu + b.row[i] * 2 * (size_t) f.fmt.n_ff, (int) (2 * f.fmt.n_ff), s.st);
            if (rc != 0) refuse("the gate/up product was refused");
            st.kernel_launches += 1;   // one launch per expert per product; the launcher tiles tokens in its grid

            woff += blob;
        }
        ck(cudaMemcpyAsync(s.gu_pin, s.gu, b.rows * 2 * (size_t) f.fmt.n_ff * sizeof(float), cudaMemcpyDeviceToHost,
                           s.st),
           "gate/up D2H");
        st.gu_d2h_bytes += b.rows * 2 * (size_t) f.fmt.n_ff * sizeof(float);
        ck(cudaEventRecord(s.ev_gu, s.st), "record gate/up");
    };

    auto stage_down = [&](const Batch& b, Slot& s) {
        ck(cudaEventSynchronize(s.ev_gu), "wait gate/up");
        ++st.slot_waits;
        const auto t0 = std::chrono::steady_clock::now();
        size_t row = 0;
        for (int gi : b.group) {
            for (size_t j = 0; j < groups[(size_t) gi].selections.size(); ++j, ++row) {
                const float* gu = s.gu_pin + row * 2 * (size_t) f.fmt.n_ff;
                for (int r = 0; r < m.n_ff; ++r) {
                    const float g = gu[r];
                    const float u = gu[(size_t) m.n_ff + r];
                    m.ff[(size_t) r] = (g / (1.f + std::exp(-g))) * u;   // the pool's own expression
                }
                strata::kernels::cpu::native_quant_h(f.fmt, m.ff.data(), s.hidden_pin + row * f.fmt.h_bytes);
            }
        }
        st.cpu_activation_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        ck(cudaMemcpyAsync(s.hidden, s.hidden_pin, b.rows * f.fmt.h_bytes, cudaMemcpyHostToDevice, s.st), "hidden H2D");
        st.hidden_h2d_bytes += b.rows * f.fmt.h_bytes;
        // one down launch per expert, from the same packed slot layout as stage_gu()
        size_t woff = 0;
        for (size_t i = 0; i < b.group.size(); ++i) {
            const Group& g = groups[(size_t) b.group[i]];
            const int ntok = (int) g.selections.size();
            const uint8_t* down = s.weights + woff + f.fmt.down_off;
            const int rc = f.fmt.d_type == kQ4K
                               ? es_q4k_rows(down, (int) (f.fmt.n_ff / 256), m.n_embd,
                                             s.hidden + b.row[i] * f.fmt.h_bytes, (int) f.fmt.h_bytes, ntok,
                                             s.out + b.row[i] * (size_t) m.n_embd, m.n_embd, s.st)
                               : es_q6k_rows(down, (int) (f.fmt.n_ff / 256), m.n_embd,
                                             s.hidden + b.row[i] * f.fmt.h_bytes, (int) f.fmt.h_bytes, ntok,
                                             s.out + b.row[i] * (size_t) m.n_embd, m.n_embd, s.st);
            if (rc != 0) refuse("the down product was refused");
            st.kernel_launches += 1;   // one launch per expert per product; the launcher tiles tokens in its grid

            woff += blob;
            (void) g;
        }
        ck(cudaMemcpyAsync(s.out_pin, s.out, b.rows * (size_t) m.n_embd * sizeof(float), cudaMemcpyDeviceToHost,
                           s.st),
           "output D2H");
        st.output_d2h_bytes += b.rows * (size_t) m.n_embd * sizeof(float);
        ck(cudaEventRecord(s.ev_out, s.st), "record output");
        ck(cudaEventSynchronize(s.ev_out), "wait output");
        ++st.slot_waits;
        // scatter to the ORIGINAL selection order; nothing is weighted here
        size_t r = 0;
        for (int gi : b.group) {
            for (int sel : groups[(size_t) gi].selections) {
                std::memcpy(view.output + (size_t) sel * m.n_embd, s.out_pin + r * (size_t) m.n_embd,
                            (size_t) m.n_embd * sizeof(float));
                ++r;
            }
        }
    };

    // ---- the two-slot schedule --------------------------------------------------------------------------------
    // Batch i loads and computes gate/up on slot i%2 while the host finishes batch i-1 on the other slot.  The slot
    // reused by batch i was released by batch i-2's scatter, which happened in the previous iteration.
    for (size_t i = 0; i < batches.size(); ++i) {
        Slot& s = m.s[i % 2];
        stage_gu(batches[i], s);
        if (i >= 1) stage_down(batches[i - 1], m.s[(i - 1) % 2]);
    }
    stage_down(batches.back(), m.s[(batches.size() - 1) % 2]);

    st.wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
    st.device_allocations = 0;   // nothing is allocated after create()
    return st;
}

}  // namespace strata::kolibri::stream
