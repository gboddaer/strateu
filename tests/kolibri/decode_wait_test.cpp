#include "strata/kolibri/decode_wait.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>
#include <stdexcept>
#include "strata/kernels/elementwise.hpp"

using namespace strata::kolibri;
#define CHECK(call) do { auto e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); return 2; } } while (0)

static void store(uint32_t& dst, uint32_t value) {
    std::atomic_ref<uint32_t>(dst).store(value, std::memory_order_release);
}

#include "decode_chain_test.inc"

int main() {
    if (const int result = chain_test()) return result;
    DecodeWaitControl *host = nullptr, *device = nullptr;
    CHECK(cudaHostAlloc(reinterpret_cast<void**>(&host), sizeof(*host), cudaHostAllocMapped));
    *host = DecodeWaitControl{};
    CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device), host, 0));
    cudaStream_t stream{};
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaGraph_t graph{};
    cudaGraphExec_t exec{};
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    enqueue_decode_wait(device, 10000000000ULL, stream);
    CHECK(cudaStreamEndCapture(stream, &graph));
    CHECK(cudaGraphInstantiate(&exec, graph, 0));

    // Replay the SAME graph with changing payload identity. Future sequences and old generations must
    // not pass, even when their other half matches. This also catches frozen capture-time arguments.
    for (uint32_t replay = 1; replay <= 32; ++replay) {
        *host = DecodeWaitControl{};
        host->ready = replay;
        host->generation = replay + 40;
        host->acknowledged = replay + 1;  // future ACK is not the current result
        host->acknowledged_generation = host->generation;
        CHECK(cudaGraphLaunch(exec, stream));
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (cudaStreamQuery(stream) != cudaErrorNotReady) {
            std::fprintf(stderr, "FAIL accepted future acknowledgement, replay %u\n", replay);
            return 1;
        }
        store(host->acknowledged_generation, replay + 39);  // stale generation, correct sequence
        store(host->acknowledged, replay);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (cudaStreamQuery(stream) != cudaErrorNotReady) {
            std::fprintf(stderr, "FAIL accepted stale generation, replay %u\n", replay);
            return 1;
        }
        store(host->acknowledged_generation, replay + 40);
        CHECK(cudaStreamSynchronize(stream));
        if (host->status != DecodeWaitOk) return 1;
    }
    std::puts("PASS 32 changing replays: future ACK and stale generation refused, exact ACK accepted");

    // Same captured graph, different failures: cancellation and a CPU producer failure must both drain.
    for (int failure = 0; failure < 2; ++failure) {
        *host = DecodeWaitControl{};
        host->ready = 1;
        host->generation = 90 + failure;
        CHECK(cudaGraphLaunch(exec, stream));
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        store(host->abort, 1);
        CHECK(cudaStreamSynchronize(stream));
        if (host->status != DecodeWaitAborted) return 1;
    }
    std::puts("PASS cancellation and CPU producer failure drain without an acknowledgement");
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));

    // A missing producer cannot spin forever even when the CPU never publishes abort.
    *host = DecodeWaitControl{};
    host->ready = 1;
    host->generation = 100;
    enqueue_decode_wait(device, 1000000ULL, stream);
    CHECK(cudaStreamSynchronize(stream));
    if (host->status != DecodeWaitTimedOut || !host->abort) {
        std::fprintf(stderr, "FAIL missing ACK did not invalidate the operation\n");
        return 1;
    }
    std::puts("PASS withheld ACK hits finite device tick budget and invalidates the operation");
    *host = DecodeWaitControl{};   // no publisher: zero ACK must not masquerade as a valid result
    enqueue_decode_wait(device, 1000000ULL, stream);
    CHECK(cudaStreamSynchronize(stream));
    if (host->status != DecodeWaitTimedOut || !host->abort) {
        std::fprintf(stderr, "FAIL unpublished zero identity was accepted\n");
        return 1;
    }
    std::puts("PASS absent publication does not accept an all-zero acknowledgement");
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFreeHost(host));
    return 0;
}
