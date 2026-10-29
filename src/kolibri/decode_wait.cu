#include "strata/kolibri/decode_wait.hpp"
#include <cuda_runtime.h>

namespace strata::kolibri {
namespace {

__global__ void wait_kernel(DecodeWaitControl* storage, uint64_t tick_budget) {
    volatile DecodeWaitControl* control = storage;
    // Read identity on every replay, never at host capture time.
    const uint32_t want = control->ready;
    const uint32_t generation = control->generation;
    const uint64_t start = clock64();
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
        if (static_cast<uint64_t>(clock64()) - start >= tick_budget) {
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
