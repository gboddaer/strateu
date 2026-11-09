// tools/kolibri/transfer_probe.cu - isolated H2D copy ceiling: pageable vs pinned, sizes, HSA_ENABLE_SDMA is external.
// Verifies every byte, times copy and verification separately, caps pinned allocation at 64 MiB.
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "FAIL %s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

int main() {
    int failures = 0;
    const size_t sizes[] = {1u << 20, 4u << 20, 16u << 20, 64u << 20};
    void* d = nullptr;
    ck(cudaMalloc(&d, 64u << 20), "malloc device");
    std::printf("kind,bytes,copy_ms,GBps,verify_ms,cpu_memcpy_ms,cpu_GBps,ok\n");
    for (int pinned = 0; pinned <= 1; ++pinned) {
        unsigned char* h = nullptr;
        if (pinned) ck(cudaHostAlloc((void**)&h, 64u << 20, cudaHostAllocDefault), "host alloc pinned");
        else { h = (unsigned char*) std::malloc(64u << 20); if (!h) return 1; }
        for (size_t i = 0; i < (64u << 20); ++i) h[i] = (unsigned char) (i * 131u + 7u);
        std::vector<unsigned char> sink(64u << 20);
        cudaStream_t s = nullptr;
        ck(cudaStreamCreate(&s), "stream");
        cudaEvent_t a = nullptr, b = nullptr;
        ck(cudaEventCreate(&a), "event a"); ck(cudaEventCreate(&b), "event b");
        for (size_t n : sizes) {
            ck(cudaMemcpyAsync(d, h, n, cudaMemcpyHostToDevice, s), "warm copy");
            ck(cudaStreamSynchronize(s), "warm sync");
            double best = 1e30;
            for (int rep = 0; rep < 5; ++rep) {
                ck(cudaEventRecord(a, s), "record a");
                ck(cudaMemcpyAsync(d, h, n, cudaMemcpyHostToDevice, s), "timed copy");
                ck(cudaEventRecord(b, s), "record b");
                ck(cudaEventSynchronize(b), "event sync");
                float ms = 0;
                ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
                if (ms < best) best = ms;
            }
            ck(cudaMemcpy(sink.data(), d, n, cudaMemcpyDeviceToHost), "readback");
            const auto v0 = std::chrono::steady_clock::now();
            bool ok = std::memcmp(sink.data(), h, n) == 0;
            const auto v1 = std::chrono::steady_clock::now();
            const double verify_ms = std::chrono::duration<double, std::milli>(v1 - v0).count();
            const auto c0 = std::chrono::steady_clock::now();
            std::memcpy(sink.data(), h, n);
            const auto c1 = std::chrono::steady_clock::now();
            const double cpu_ms = std::chrono::duration<double, std::milli>(c1 - c0).count();
            if (!ok) ++failures;
            std::printf("%s,%zu,%.3f,%.2f,%.3f,%.3f,%.2f,%d\n", pinned ? "pinned" : "pageable", n, best,
                        (double) n / best / 1e6, verify_ms, cpu_ms, (double) n / cpu_ms / 1e6, int(ok));
        }
        ck(cudaEventDestroy(a), "destroy a"); ck(cudaEventDestroy(b), "destroy b");
        ck(cudaStreamDestroy(s), "destroy stream");
        if (pinned) ck(cudaFreeHost(h), "free pinned"); else std::free(h);
    }
    ck(cudaFree(d), "free device");
    return failures ? 1 : 0;
}
