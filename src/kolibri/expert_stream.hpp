// src/kolibri/expert_stream.hpp - plan 2026-10-31-kolibri-gpu-streaming, Task 3.
//
// The streamed expert prefill's public, CUDA-free surface.  The engine (and the replay tool) hand it one
// layer-chunk at a time: the expert ids of every selection, the already-quantized token activations, the layer's
// blob table, and a [selection][n_embd] output view in ORIGINAL selection order.  run() returns only when every
// output element of that view has been published on the host, bit for bit as the CPU pool would have produced it.
//
// The context owns two reusable slots.  Each slot holds a bounded batch of expert groups and their weights, the
// gathered activations, the gate/up results, the CPU-quantized hidden rows and the down results.  One slot loads
// and computes gate/up while the host does the other slot's CPU SwiGLU and quantization, so the CPU step is not
// dead time on the transfer path.  Everything is allocated at create(); a run never allocates.
#pragma once

#include <cstdint>
#include <memory>

#include "expert_stream_plan.hpp"

namespace strata::kolibri::stream {

/// Per-run counters.  Every transfer direction and the actual unique weight bytes; launches, waits and allocations
/// so a test can assert the work is grouped (launches with tiles, not with selections) and that a warm run
/// allocates nothing.
struct Stats {
    uint64_t groups = 0;
    uint64_t selections = 0;
    uint64_t weight_h2d_bytes = 0;
    uint64_t activation_h2d_bytes = 0;
    uint64_t gu_d2h_bytes = 0;
    uint64_t hidden_h2d_bytes = 0;
    uint64_t output_d2h_bytes = 0;
    uint64_t kernel_launches = 0;
    uint64_t slot_waits = 0;
    uint64_t device_allocations = 0;
    double wall_ms = 0.0;
    double cpu_activation_ms = 0.0;
};

/// Two slots, created once for the largest geometry the layer family uses, then reused by every call.
/// Invalid arguments throw std::invalid_argument; allocation or runtime failures throw std::runtime_error.
class Context {
public:
    ~Context();   ///< drains in-flight work before releasing anything
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

private:
    struct Impl;
    explicit Context(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    friend std::unique_ptr<Context> create(const LayerView&, const Config&);
    friend Stats run(Context&, const LayerView&);
};

/// Sizes both slots from `capacity_view` for BOTH down formats this path supports (Kolibri alternates Q4_K and Q6_K
/// down between layers), so a later call may use either.  Refuses a configuration that cannot fit the budget.
std::unique_ptr<Context> create(const LayerView& capacity_view, const Config& config);

/// Runs one layer-chunk.  `view` must share its n_embd/n_ff/act_bytes/h_bytes/gu_row/up_off with the view create()
/// was given, and may differ only in the down type, the weights, the ids, the activations and the output.
Stats run(Context& ctx, const LayerView& view);

namespace detail {
/// Test-only: make the `ordinal`-th allocation of the next create() throw, to prove a partial allocation is
/// released.  -1 (the default) disables it.  It changes no production behaviour.
void fail_allocation_after(int ordinal);
}  // namespace detail

}  // namespace strata::kolibri::stream
