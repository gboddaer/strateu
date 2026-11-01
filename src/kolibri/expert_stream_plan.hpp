// src/kolibri/expert_stream_plan.hpp - Task 1 of plan 2026-10-31-kolibri-gpu-streaming.
//
// The CUDA-free half of the streamed expert prefill: how a chunk's expert selections are grouped, and the exact
// device/pinned capacity those groups are run in.  Nothing here touches a device or a model, so it is the one part
// of the campaign a plain host test can pin (tests/kolibri/expert_stream_plan_test.cpp).
//
// Supported geometry in this initial path (the design's "initial supported target"): Q4_K gate/up, Q4_K or Q6_K
// down, both against Q8_K activations, blob layout [gate rows | up rows | down rows].  Anything else - another
// weight type, another activation type, an NVFP4 scale tail, a Hadamard-rotated basis - is refused here by name
// rather than reached by a kernel that would compute something else.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "strata/kernels/cpu/native_expert.hpp"

namespace strata::kolibri::stream {

// ggml type ids used by the initial path (ggml.h's enum; repeated here so this header needs no ggml include).
inline constexpr int kQ4K = 12;
inline constexpr int kQ6K = 14;
inline constexpr int kQ8K = 15;

// Row sizes of the three supported formats: per 256-value block, Q4_K 144 B, Q6_K 210 B, Q8_K 292 B.
inline constexpr size_t q4k_row_bytes(int64_t n) { return (size_t) (n / 256) * 144u; }
inline constexpr size_t q6k_row_bytes(int64_t n) { return (size_t) (n / 256) * 210u; }
inline constexpr size_t q8k_row_bytes(int64_t n) { return (size_t) (n / 256) * 292u; }

/// Every device and pinned buffer starts on this boundary, so a slot's size is the sum of its aligned buffers.
inline constexpr size_t kBufferAlign = 256;
/// One batch descriptor per slot (group ids, counts, offsets): reserved, never re-sized by a request.
inline constexpr size_t kDescriptorBytes = 512;
/// The design's batch bound: at most eight expert groups per slot.
inline constexpr int kMaxExpertsPerBatch = 8;

/// One expert's selections, in ascending ORIGINAL selection index (the scatter target order).
struct Group {
    int expert = -1;
    std::vector<int> selections;
};

/// Groups `selections` expert ids (one per selection, `top_k` selections per token) by expert, ascending expert id,
/// selections ascending within a group.  Throws std::invalid_argument for a negative/non-divisible selection count,
/// a non-positive top_k or expert count, a null id array with work to do, or an id outside [0, experts).
std::vector<Group> make_groups(const int* ids, int selections, int top_k, int experts);

/// One layer-chunk's geometry for the streaming path.  Deliberately CUDA-free: the engine fills it from its
/// LayerBlobs/Scratch and the replay tool from a capture record.
struct LayerView {
    strata::kernels::cpu::NativeFmt fmt;
    const uint8_t* weights = nullptr;   ///< expert_count blobs at weight_stride
    size_t weight_stride = 0;
    int expert_count = 0;
    const uint8_t* token_acts = nullptr;  ///< selection_count / top_k rows, stride fmt.act_bytes
    const int* expert_ids = nullptr;      ///< one expert id per selection
    int selection_count = 0;
    int top_k = 0;
    float* output = nullptr;   ///< [selection_count][fmt.n_embd], ORIGINAL selection order
};

/// The two reusable slots' bounds.  Defaults are the design's: 8 groups and 32 MiB of compressed weights per slot,
/// 256 MiB of new device memory in total.
struct Config {
    int experts_per_batch = kMaxExpertsPerBatch;
    size_t weight_bytes_per_slot = 32u << 20;
    size_t device_budget_bytes = 256u << 20;
};

/// What create() must allocate.  Both slots are counted; every buffer is aligned to kBufferAlign.
struct Capacity {
    size_t device_bytes = 0;
    size_t pinned_bytes = 0;
    size_t weight_bytes_per_slot = 0;
    int max_selections = 0;
};

/// Validates the view against the supported geometry and returns its checked capacity.  Throws
/// std::invalid_argument for an unsupported format, an inconsistent stride/offset, a missing buffer, an expert id
/// out of range, or size arithmetic that would overflow - always before anything is allocated.
Capacity required_capacity(const LayerView& view, const Config& config);


namespace detail {

inline size_t checked_mul(size_t a, size_t b, const char* what) {
    if (a != 0 && b > (size_t) -1 / a) throw std::invalid_argument(what);
    return a * b;
}
inline size_t checked_add(size_t a, size_t b, const char* what) {
    if (b > (size_t) -1 - a) throw std::invalid_argument(what);
    return a + b;
}
/// The buffer's size once it starts on the next kBufferAlign boundary.
inline size_t aligned(size_t v, const char* what) {
    const size_t r = v % kBufferAlign;
    return r == 0 ? v : checked_add(v, kBufferAlign - r, what);
}
inline void require(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(what);
}

}  // namespace detail

inline std::vector<Group> make_groups(const int* ids, int selections, int top_k, int experts) {
    detail::require(selections >= 0, "expert streaming: the selection count is negative");
    detail::require(top_k > 0, "expert streaming: top_k must be positive");
    detail::require(experts > 0, "expert streaming: the expert count must be positive");
    detail::require(selections % top_k == 0, "expert streaming: the selection count is not a multiple of top_k");
    detail::require(ids != nullptr || selections == 0, "expert streaming: selection ids are missing");
    std::vector<Group> groups;
    if (selections == 0) return groups;
    std::vector<int> count((size_t) experts, 0);
    for (int sel = 0; sel < selections; ++sel) {
        const int id = ids[sel];
        detail::require(id >= 0 && id < experts, "expert streaming: an expert id is out of range");
        ++count[(size_t) id];
    }
    groups.reserve((size_t) experts);
    std::vector<int> slot((size_t) experts, -1);   // expert -> index in `groups`
    for (int e = 0; e < experts; ++e) {
        if (count[(size_t) e] == 0) continue;
        slot[(size_t) e] = (int) groups.size();
        groups.push_back(Group{e, {}});
        groups.back().selections.reserve((size_t) count[(size_t) e]);
    }
    // ONE ascending pass over the selections: each group's list is therefore ascending in original index, which is
    // the order the scatter and the per-token gather both read.
    for (int sel = 0; sel < selections; ++sel) groups[(size_t) slot[(size_t) ids[sel]]].selections.push_back(sel);
    return groups;
}

inline Capacity required_capacity(const LayerView& view, const Config& config) {
    const strata::kernels::cpu::NativeFmt& f = view.fmt;
    // ---- the formats this initial path may stream ------------------------------------------------------------
    detail::require(f.gu_type == kQ4K, "expert streaming: only Q4_K gate/up is supported");
    detail::require(f.d_type == kQ4K || f.d_type == kQ6K, "expert streaming: only Q4_K or Q6_K down is supported");
    detail::require(f.gu_act == kQ8K && f.d_act == kQ8K,
                    "expert streaming: only Q8_K activations are supported");
    detail::require(f.n_embd > 0 && f.n_ff > 0 && f.n_embd % 256 == 0 && f.n_ff % 256 == 0,
                    "expert streaming: the expert geometry is not whole 256-value blocks");
    // ---- and its blob layout, checked field by field (a mismatch would read another expert's rows) -------------
    detail::require(f.gu_row == q4k_row_bytes(f.n_embd), "expert streaming: gu_row is not Q4_K of n_embd");
    detail::require(f.d_row == (f.d_type == kQ4K ? q4k_row_bytes(f.n_ff) : q6k_row_bytes(f.n_ff)),
                    "expert streaming: d_row is not the down type's row size of n_ff");
    detail::require(f.up_off == (size_t) f.n_ff * f.gu_row, "expert streaming: up_off is not gate_rows x gu_row");
    detail::require(f.down_off == 2 * (size_t) f.n_ff * f.gu_row,
                    "expert streaming: down_off is not the end of the up rows");
    detail::require(f.bytes == f.down_off + (size_t) f.n_embd * f.d_row,
                    "expert streaming: the blob size is not [gate|up|down]");
    detail::require(f.act_bytes == q8k_row_bytes(f.n_embd), "expert streaming: act_bytes is not Q8_K of n_embd");
    detail::require(f.h_bytes == q8k_row_bytes(f.n_ff), "expert streaming: h_bytes is not Q8_K of n_ff");
    detail::require(f.tail_off == 0, "expert streaming: an NVFP4 scale tail is not supported");
    detail::require(f.had_block == 0 && !f.had_gu && !f.had_d,
                    "expert streaming: a Hadamard-rotated basis is not supported");
    // ---- the view's own geometry -----------------------------------------------------------------------------
    detail::require(view.selection_count >= 0, "expert streaming: the selection count is negative");
    detail::require(view.top_k > 0, "expert streaming: top_k must be positive");
    detail::require(view.selection_count % view.top_k == 0,
                    "expert streaming: the selection count is not a multiple of top_k");
    detail::require(view.expert_count > 0, "expert streaming: the layer has no experts");
    detail::require(view.weight_stride >= f.bytes, "expert streaming: weight_stride is shorter than the blob");
    detail::require(view.weights != nullptr, "expert streaming: the weight table is missing");
    detail::require(view.selection_count == 0 || view.token_acts != nullptr,
                    "expert streaming: the token activations are missing");
    detail::require(view.selection_count == 0 || view.expert_ids != nullptr,
                    "expert streaming: the selection ids are missing");
    detail::require(view.selection_count == 0 || view.output != nullptr,
                    "expert streaming: the output view is missing");
    // ---- the batch's bounds -----------------------------------------------------------------------------------
    detail::require(config.experts_per_batch > 0 && config.experts_per_batch <= kMaxExpertsPerBatch,
                    "expert streaming: experts_per_batch is outside [1, 8]");
    detail::require(config.weight_bytes_per_slot > 0, "expert streaming: the weight cap is zero");
    detail::require(config.device_budget_bytes > 0, "expert streaming: the device budget is zero");

    const char* overflow = "expert streaming: the capacity arithmetic overflows this platform's size_t";
    const size_t selections = (size_t) view.selection_count;
    Capacity c;
    // Weights: one batch of blobs, never below a single blob (a group bigger than the batch bound still runs alone).
    const size_t batch_bytes = detail::checked_mul((size_t) config.experts_per_batch, view.weight_stride, overflow);
    const size_t capped = batch_bytes < config.weight_bytes_per_slot ? batch_bytes : config.weight_bytes_per_slot;
    c.weight_bytes_per_slot = capped > view.weight_stride ? capped : view.weight_stride;
    c.max_selections = view.selection_count;
    // Per-selection buffers: gathered Q8_K activations, gate and up floats, the quantized hidden rows, down floats.
    const size_t acts = detail::checked_mul(selections, f.act_bytes, overflow);
    const size_t gu = detail::checked_mul(detail::checked_mul(selections, (size_t) f.n_ff, overflow), 2u * sizeof(float), overflow);
    const size_t hidden = detail::checked_mul(selections, f.h_bytes, overflow);
    const size_t out = detail::checked_mul(detail::checked_mul(selections, (size_t) f.n_embd, overflow), sizeof(float), overflow);
    const size_t slot = detail::aligned(c.weight_bytes_per_slot, overflow) + detail::aligned(acts, overflow) +
                        detail::aligned(gu, overflow) + detail::aligned(hidden, overflow) +
                        detail::aligned(out, overflow) + detail::aligned(kDescriptorBytes, overflow);
    c.device_bytes = detail::checked_mul(2u, slot, overflow);
    c.pinned_bytes = detail::checked_mul(2u, slot, overflow);
    detail::require(c.device_bytes <= config.device_budget_bytes,
                    "expert streaming: the two slots need more device memory than the configured budget");
    // ---- and only now the ids, whose scan is O(selections) -----------------------------------------------------
    for (int sel = 0; sel < view.selection_count; ++sel) {
        const int id = view.expert_ids[sel];
        detail::require(id >= 0 && id < view.expert_count, "expert streaming: an expert id is out of range");
    }
    return c;
}

}  // namespace strata::kolibri::stream
