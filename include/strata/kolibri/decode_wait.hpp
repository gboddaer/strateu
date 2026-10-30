#pragma once
#include <cstdint>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

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

// Host accesses to naturally aligned mapped control words. CUDA TUs use C++17, so do not require
// C++20 atomic_ref or reinterpret ordinary storage as a std::atomic object. The compiler intrinsics
// operate on the actual uint32_t storage; MSVC's interlocked operations provide stronger ordering.
inline uint32_t decode_control_load(const uint32_t& word) noexcept {
#if defined(_MSC_VER)
    static_assert(sizeof(long) == sizeof(uint32_t));
    return static_cast<uint32_t>(_InterlockedCompareExchange(
        reinterpret_cast<volatile long*>(const_cast<uint32_t*>(&word)), 0, 0));
#else
    static_assert(__atomic_always_lock_free(sizeof(uint32_t), nullptr), "mapped control needs lock-free words");
    return __atomic_load_n(&word, __ATOMIC_ACQUIRE);
#endif
}

inline void decode_control_store(uint32_t& word, uint32_t value) noexcept {
#if defined(_MSC_VER)
    _InterlockedExchange(reinterpret_cast<volatile long*>(&word), static_cast<long>(value));
#else
    __atomic_store_n(&word, value, __ATOMIC_RELEASE);
#endif
}

enum : uint32_t { DecodeWaitOk = 0, DecodeWaitAborted = 1, DecodeWaitTimedOut = 2 };

// Capturable, compute-queue wait. The finite budget is in device clock64 ticks, NOT wall milliseconds.
// On cancellation or timeout the graph may drain, but the caller must invalidate the token and reset
// affected state before reuse. Never emit logits from an aborted graph as a successful token.
void enqueue_decode_wait(DecodeWaitControl* device_control, uint64_t tick_budget, void* stream);

}  // namespace strata::kolibri
