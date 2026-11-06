#include "strata/kolibri/decode_wait.hpp"
#include <cuda_runtime.h>

namespace strata::kolibri {
namespace {

__global__ void wait_kernel(DecodeWaitControl* storage, uint64_t tick_budget) {
    volatile DecodeWaitControl* control = storage;
    // Read identity on every replay, never at host capture time.
    const uint32_t want = control->ready;
    const uint32_t generation = control->generation;
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
    // AMD gfx11 (measured on a W7800/gfx1100): clock64() is a 20-bit SHADER_CYCLES counter zero-extended to
    // 64 that wraps at 2^20 (Mesa ACO lowers the subgroup clock this way on GFX10.3/GFX11), so it steps
    // backwards across a wrap and a 64-bit tick budget can never be reached - the wait either expired on the
    // first poll or never did. The budget is counted in poll iterations instead: one s_sleep(1) measures
    // 0.023-0.034 us here. One clock64 tick is ~0.5 ns on the CUDA side, so 64 ticks ~ one poll; dividing
    // the budget keeps a given call's wall time on both backends (1e6 ~ 0.5 ms, 1e8 ~ 45 ms).
    uint64_t polls = 0;
    const uint64_t poll_budget = tick_budget / 64u + 1u;
#else
    const uint64_t start = clock64();
#endif
    for (;;) {
        if (control->abort != 0) {
            if (control->status == DecodeWaitOk) control->status = DecodeWaitAborted;
            __threadfence_system();
            return;
        }
        if (want != 0 && control->acknowledged == want && control->acknowledged_generation == generation) {
            __threadfence_system();   // the result payload is consumed only after its acknowledgement
            return;
        }
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
        if (++polls >= poll_budget) {
#else
        if (static_cast<uint64_t>(clock64()) - start >= tick_budget) {
#endif
            control->status = DecodeWaitTimedOut;
            control->abort = 1;
            __threadfence_system();
            return;
        }
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
        __builtin_amdgcn_s_sleep(1);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(64);
#endif
    }
}

}  // namespace

void enqueue_decode_wait(DecodeWaitControl* control, uint64_t tick_budget, void* stream) {
    wait_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(control, tick_budget);
}

}  // namespace strata::kolibri
