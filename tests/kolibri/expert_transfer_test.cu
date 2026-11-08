#include "kolibri/expert_transfer.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <cstdio>

static void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
int main() {
    constexpr size_t cap = 8192;
    unsigned char* dev = nullptr;
    ck(cudaMalloc((void**)&dev, 32 * cap));
    for (bool pinned : {false, true}) {
        bool failed = false;
        try { strata::kolibri::ExpertTransfer partial(cap, pinned, 1); }
        catch (const std::bad_alloc&) { failed = true; }
        if (!failed) throw std::runtime_error("allocation failure injection did not fire");
        strata::kolibri::ExpertTransfer ring(cap, pinned);
        // Delayed DMA must not allow the producer to overwrite a previous batch's source.
        ck(cudaLaunchHostFunc(ring.stream(), [](void*) { std::this_thread::sleep_for(std::chrono::milliseconds(10)); }, nullptr));
        for (int i = 0; i < 32; ++i) {
            const size_t n = i % 2 ? cap : cap / 2;  // two layouts, including partial batches
            auto* h = ring.acquire();
            std::memset(h, i + 1, n);
            ring.submit(dev + (size_t)i * cap, n);
        }
        ring.wait_on(nullptr);
        if (ring.waits() != 1) throw std::runtime_error("transfer fence must cover the copy queue once");
        ring.wait_on(nullptr);
        if (ring.waits() != 1) throw std::runtime_error("completed generation redundantly fenced");
        ck(cudaStreamSynchronize(nullptr));
        cudaStream_t other = nullptr;
        ck(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking));
        ring.wait_on(other);
        if (ring.waits() != 2) throw std::runtime_error("second compute stream missed its dependency");
        ck(cudaStreamSynchronize(other));
        ck(cudaStreamDestroy(other));
        cudaStream_t reader = nullptr;
        unsigned char* observed = nullptr;
        ck(cudaStreamCreateWithFlags(&reader, cudaStreamNonBlocking));
        ck(cudaHostAlloc((void**)&observed, cap, cudaHostAllocDefault));
        ck(cudaMemset(dev, 17, cap));
        ck(cudaDeviceSynchronize());
        ck(cudaLaunchHostFunc(reader, [](void*) { std::this_thread::sleep_for(std::chrono::milliseconds(10)); }, nullptr));
        ck(cudaMemcpyAsync(observed, dev, cap, cudaMemcpyDeviceToHost, reader));
        ring.wait_for(reader);
        std::memset(ring.acquire(), 31, cap);
        ring.submit(dev, cap);
        ring.wait_on(nullptr);
        ck(cudaStreamSynchronize(nullptr));
        for (size_t j=0; j<cap; ++j) if (observed[j] != 17) throw std::runtime_error("old device reader overwritten early");
        ck(cudaStreamDestroy(reader));
        ck(cudaFreeHost(observed));
        // Restore the first transfer's expected data for the checks below.
        ck(cudaMemset(dev, 1, cap / 2));
        unsigned char host[cap];
        for (int i = 0; i < 32; ++i) {
            const size_t n = i % 2 ? cap : cap / 2;
            ck(cudaMemcpy(host, dev + (size_t)i * cap, n, cudaMemcpyDeviceToHost));
            for (size_t j = 0; j < n; ++j) if (host[j] != i+1) throw std::runtime_error("staging source reused early");
        }
        bool rejected = false;
        ring.acquire();
        try { ring.submit(dev, cap + 1); } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("oversize submission accepted");
        ring.drain();  // cancellation/reset also drains a borrowed but unsubmitted buffer
        std::memset(ring.acquire(), 231, cap);
        ring.submit(dev, cap);
        ring.wait_on(nullptr);
        ck(cudaStreamSynchronize(nullptr));
        ck(cudaMemcpy(host, dev, cap, cudaMemcpyDeviceToHost));
        for (auto x : host) if (x != 231) throw std::runtime_error("reset/replay copied stale weights");
    }
    ck(cudaFree(dev));
    std::puts("PASS pageable/pinned transfer ring: delayed DMA, wraparound, partial layouts, bounds, drain");
}
