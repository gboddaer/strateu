// Is the 2.93 s floor reachable? The prefill streams 28.09 GB of expert blobs per 512 tokens (measured); at the box's
// 9.6 GB/s host->device that is 2.93 s against today's 8.0 s. This measures the stream with and without work overlapping
// it, so the floor is a measurement rather than an extrapolation.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

__global__ void consume(const uint8_t* __restrict__ p, size_t n, unsigned long long* acc) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long s = 0;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) s += p[i];
    if (s == 0x9e3779b97f4a7c15ull) atomicAdd(acc, 1ull);   // never true: keeps the reads from being optimised away
}

int main(int argc, char** argv) {
    const size_t chunk = 20u << 20;              // 20 MB: about eight experts' blobs
    const double total_gb = 28.09;               // one 512-token prefill, measured (Phase 0)
    const size_t iters = (size_t)(total_gb * 1e9 / (double)chunk);
    uint8_t* h;
    cudaHostAlloc((void**)&h, chunk, cudaHostAllocDefault);
    for (size_t i = 0; i < chunk; i += 4096) h[i] = (uint8_t)(i * 2654435761u >> 24);   // touch one byte per page
    uint8_t *d0, *d1; unsigned long long* acc;
    cudaMalloc(&d0, chunk); cudaMalloc(&d1, chunk); cudaMalloc(&acc, 8); cudaMemset(acc, 0, 8);
    cudaStream_t s; cudaStreamCreate(&s);

    for (int mode = 0; mode < 2; ++mode) {       // 0: pure copy, 1: copy with a kernel reading the previous buffer
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        cudaDeviceSynchronize(); cudaEventRecord(a, s);
        for (size_t i = 0; i < iters; ++i) {
            uint8_t* dst = (i & 1) ? d1 : d0;
            cudaMemcpyAsync(dst, h, chunk, cudaMemcpyHostToDevice, s);
            if (mode == 1 && i > 0) consume<<<512, 256, 0, s>>>((i & 1) ? d0 : d1, chunk, acc);
        }
        cudaEventRecord(b, s); cudaEventSynchronize(b);
        float ms = 0; cudaEventElapsedTime(&ms, a, b);
        const double gb = (double)iters * (double)chunk / 1e9;
        std::printf("%s: %.2f GB in %.3f s -> %.2f GB/s  =>  28.09 GB would take %.2f s\n",
                    mode ? "copy + consume kernel" : "pure copy        ", gb, ms / 1000.0, gb / (ms / 1000.0),
                    total_gb / (gb / (ms / 1000.0)));
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
    std::printf("(today's tuned prefill: 8.0 s; the pool streams the same bytes through the CPU at 3.3 GB/s)\n");
    return 0;
}
