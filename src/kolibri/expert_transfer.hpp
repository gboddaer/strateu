#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <memory>

namespace strata::kolibri {
// Two bounded, event-owned host buffers. No expert model pages are registered.
// A single host producer packs into acquire(), then submits; buffers cannot be
// overwritten before their H2D event. Compute waits on events, not a host drain.
class ExpertTransfer {
public:
    // fail_after_buffers is a deterministic allocation-failure test hook; production leaves it -1.
    ExpertTransfer(size_t bytes, bool pinned, int fail_after_buffers = -1);
    ~ExpertTransfer();
    ExpertTransfer(const ExpertTransfer&) = delete;
    ExpertTransfer& operator=(const ExpertTransfer&) = delete;
    unsigned char* acquire();
    void submit(void* device, size_t bytes);
    void wait_on(cudaStream_t compute);
    void wait_for(cudaStream_t previous_reader);
    void drain();
    cudaStream_t stream() const;
    size_t waits() const;
private:
    struct Impl;
    std::unique_ptr<Impl> p;
};
}
