#pragma once
#include <cstdint>

namespace strata::kolibri {

// One coherent mapped cache line. Reset only when the preceding graph is quiescent.
// Both sequence AND generation must match: an acknowledgement from a previous request is not a result.
struct alignas(64) DecodeWaitControl {
    uint32_t ready = 0;
    uint32_t acknowledged = 0;
    uint32_t generation = 0;
    uint32_t acknowledged_generation = 0;
    uint32_t abort = 0;
    uint32_t status = 0;
};

enum : uint32_t { DecodeWaitOk = 0, DecodeWaitAborted = 1, DecodeWaitTimedOut = 2 };

// Capturable, compute-queue wait. The finite budget is in device clock64 ticks, NOT wall milliseconds.
// On cancellation or timeout the graph may drain, but the caller must invalidate the token and reset
// affected state before reuse. Never emit logits from an aborted graph as a successful token.
void enqueue_decode_wait(DecodeWaitControl* device_control, uint64_t tick_budget, void* stream);

}  // namespace strata::kolibri
