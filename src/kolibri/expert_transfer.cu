#include "expert_transfer.hpp"
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::kolibri {
static void check(cudaError_t s) {
    if (s != cudaSuccess) throw std::runtime_error(std::string("expert transfer: ") + cudaGetErrorString(s));
}
struct ExpertTransfer::Impl {
    struct Buffer { unsigned char* h = nullptr; cudaEvent_t ready = nullptr; bool live = false; } b[2];
    cudaStream_t copy = nullptr;
    cudaEvent_t readers_done = nullptr;
    size_t cap = 0;
    unsigned next = 0;
    size_t generation = 0, waited = 0, wait_count = 0;
    cudaStream_t waited_stream = nullptr;
    bool pinned = false, borrowed = false;
    ~Impl() {
        if (copy) cudaStreamSynchronize(copy);
        for (auto& s : b) {
            if (s.ready) cudaEventDestroy(s.ready);
            if (s.h) { if (pinned) cudaFreeHost(s.h); else std::free(s.h); }
        }
        if (readers_done) cudaEventDestroy(readers_done);
        if (copy) cudaStreamDestroy(copy);
    }
};
ExpertTransfer::ExpertTransfer(size_t bytes, bool pinned, int fail_after_buffers) : p(new Impl) {
    if (bytes == 0 || bytes > (32u << 20)) throw std::invalid_argument("expert transfer: capacity outside 1..32 MiB");
    p->cap = bytes; p->pinned = pinned;
    check(cudaStreamCreateWithFlags(&p->copy, cudaStreamNonBlocking));
    check(cudaEventCreateWithFlags(&p->readers_done, cudaEventDisableTiming));
    for (auto& s : p->b) {
        if (fail_after_buffers == 0) throw std::bad_alloc();
        if (fail_after_buffers > 0) --fail_after_buffers;
        if (pinned) check(cudaHostAlloc((void**)&s.h, bytes, cudaHostAllocDefault));
        else { s.h = (unsigned char*)std::malloc(bytes); if (!s.h) throw std::bad_alloc(); }
        check(cudaEventCreateWithFlags(&s.ready, cudaEventDisableTiming));
    }
}
ExpertTransfer::~ExpertTransfer() = default;
unsigned char* ExpertTransfer::acquire() {
    if (p->borrowed) throw std::logic_error("expert transfer: previous buffer not submitted");
    auto& s = p->b[p->next];
    if (s.live) check(cudaEventSynchronize(s.ready));
    p->borrowed = true;
    return s.h;
}
void ExpertTransfer::submit(void* device, size_t bytes) {
    if (!p->borrowed || !device || bytes > p->cap || bytes == 0)
        throw std::invalid_argument("expert transfer: invalid submission");
    auto& s = p->b[p->next];
    check(cudaMemcpyAsync(device, s.h, bytes, cudaMemcpyHostToDevice, p->copy));
    check(cudaEventRecord(s.ready, p->copy));
    s.live = true; p->borrowed = false; p->next ^= 1; ++p->generation;
}
void ExpertTransfer::wait_on(cudaStream_t compute) {
    if (p->generation == 0 || (p->waited == p->generation && p->waited_stream == compute)) return;
    // All uploads share one copy stream: its last event covers every earlier buffer.
    check(cudaStreamWaitEvent(compute, p->b[p->next ^ 1].ready, 0));
    p->waited = p->generation; p->waited_stream = compute; ++p->wait_count;
}
void ExpertTransfer::wait_for(cudaStream_t previous_reader) {
    check(cudaEventRecord(p->readers_done, previous_reader));
    check(cudaStreamWaitEvent(p->copy, p->readers_done, 0));
}
void ExpertTransfer::drain() { check(cudaStreamSynchronize(p->copy)); p->borrowed = false; }
cudaStream_t ExpertTransfer::stream() const { return p->copy; }
size_t ExpertTransfer::waits() const { return p->wait_count; }
}
