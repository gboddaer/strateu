// Is the pool's expf reproducible bit for bit on the device?  Exhaustive, over every float.
//
// Section 8 of docs/experiments/kolibri-stream-kernels-layout.md stopped the device-side SwiGLU on a measured rate:
// CUDA's expf is not glibc's, so one hidden row in twenty-three quantized a Q8_K code differently.  A rate is not a
// proof either way, and the input domain is finite: there are 2^32 float bit patterns.  This sweep compares every one
// of them, host against device, for the four candidate definitions of exp:
//
//   H1  glibc expf(x)                 - what the pool does today; the reference
//   H2  (float)exp((double)x) on host - a definition both sides can state
//   D1  expf(x) on the device         - CUDA's float expf
//   D2  (float)exp((double)x) on the device
//
// If any device variant equals H1 on all 2^32 patterns, the device can compute the pool's own SwiGLU exactly and the
// change is an unchanged-arithmetic optimization, not a numerical one.  Any mismatch is printed with its bit pattern
// so the size of the numerical change can be seen rather than guessed.
//
// Build: cmake --build build-sram --target kolibri_expf_bit_probe -j16
// Run:   ./build-sram/kolibri_expf_bit_probe [chunk_log2]
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace {

constexpr int kChunkLog2Default = 24;   // 16.7 M patterns per pass: 67 MB per buffer

__global__ void exp_variants(const uint32_t* __restrict__ bits, float* __restrict__ d1, float* __restrict__ d2,
                             int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x;
    __builtin_memcpy(&x, &bits[i], 4);
    d1[i] = expf(x);                                   // CUDA's float expf
    const double e = exp((double) x);                  // CUDA's double exp, rounded to float
    d2[i] = (float) e;
}

struct Diff {
    long long n = 0;
    uint32_t first_bits = 0;
    float got = 0.f, want = 0.f;
};

void note(Diff& d, uint32_t b, float got, float want) {
    if (d.n == 0) {
        d.first_bits = b;
        d.got = got;
        d.want = want;
    }
    ++d.n;
}

}  // namespace

int main(int argc, char** argv) {
    const int chunk_log2 = argc > 1 ? std::atoi(argv[1]) : kChunkLog2Default;
    const uint32_t chunk = 1u << chunk_log2;
    const uint64_t total = 1ull << 32;
    const unsigned hw = std::thread::hardware_concurrency();
    const unsigned nthreads = hw > 1 ? hw - 1 : 1;

    std::vector<uint32_t> bits(chunk);
    std::vector<float> h1(chunk), h2(chunk), d1(chunk), d2(chunk);
    uint32_t *dbits = nullptr;
    float *dd1 = nullptr, *dd2 = nullptr;
    if (cudaMalloc(&dbits, chunk * 4) != cudaSuccess || cudaMalloc(&dd1, chunk * 4) != cudaSuccess ||
        cudaMalloc(&dd2, chunk * 4) != cudaSuccess) {
        std::fprintf(stderr, "no device buffer for a %u-pattern chunk\n", chunk);
        return 1;
    }
    std::printf("expf, host against device, every one of %llu float bit patterns; chunk 2^%d, %u host thread(s)\n",
                (unsigned long long) total, chunk_log2, nthreads);

    Diff h2_vs_h1, d1_vs_h1, d2_vs_h1, d2_vs_h2, d1_vs_h2;
    for (uint64_t base = 0; base < total; base += chunk) {
        for (uint32_t i = 0; i < chunk; ++i) bits[i] = (uint32_t) (base + i);
        // the host's two definitions, over all cores
        const unsigned per = (chunk + nthreads - 1) / nthreads;
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < nthreads; ++t) {
            pool.emplace_back([&, t] {
                const uint32_t b0 = t * per, b1 = (b0 + per) < chunk ? (b0 + per) : chunk;
                for (uint32_t i = b0; i < b1; ++i) {
                    float x;
                    std::memcpy(&x, &bits[i], 4);
                    h1[i] = std::exp(x);                        // glibc's expf: the pool's own call
                    h2[i] = (float) std::exp((double) x);
                }
            });
        }
        for (std::thread& th : pool) th.join();

        if (cudaMemcpy(dbits, bits.data(), chunk * 4, cudaMemcpyHostToDevice) != cudaSuccess) return 1;
        exp_variants<<<(chunk + 255) / 256, 256>>>(dbits, dd1, dd2, (int) chunk);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::fprintf(stderr, "kernel: %s\n", cudaGetErrorString(cudaGetLastError()));
            return 1;
        }
        if (cudaMemcpy(d1.data(), dd1, chunk * 4, cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(d2.data(), dd2, chunk * 4, cudaMemcpyDeviceToHost) != cudaSuccess)
            return 1;

        for (uint32_t i = 0; i < chunk; ++i) {
            const uint32_t b = bits[i];
            const uint32_t bh1 = [&] { uint32_t v; std::memcpy(&v, &h1[i], 4); return v; }();
            const uint32_t bh2 = [&] { uint32_t v; std::memcpy(&v, &h2[i], 4); return v; }();
            const uint32_t bd1 = [&] { uint32_t v; std::memcpy(&v, &d1[i], 4); return v; }();
            const uint32_t bd2 = [&] { uint32_t v; std::memcpy(&v, &d2[i], 4); return v; }();
            if (bh2 != bh1) note(h2_vs_h1, b, h2[i], h1[i]);
            if (bd1 != bh1) note(d1_vs_h1, b, d1[i], h1[i]);
            if (bd2 != bh1) note(d2_vs_h1, b, d2[i], h1[i]);
            if (bd2 != bh2) note(d2_vs_h2, b, d2[i], h2[i]);
            if (bd1 != bh2) note(d1_vs_h2, b, d1[i], h2[i]);
        }
        if (((base / chunk) & 15) == 15) {
            std::printf("  ... %5.1f %%  h2!=h1 %lld  d1!=h1 %lld  d2!=h1 %lld\n", 100.0 * (double) (base + chunk) / total,
                        h2_vs_h1.n, d1_vs_h1.n, d2_vs_h1.n);
            std::fflush(stdout);
        }
    }

    const struct {
        const char* name;
        const Diff* d;
    } rows[] = {{"(float)exp(double), host   vs glibc expf", &h2_vs_h1},
                {"expf, device              vs glibc expf", &d1_vs_h1},
                {"(float)exp(double), device vs glibc expf", &d2_vs_h1},
                {"(float)exp(double), device vs the same on host", &d2_vs_h2},
                {"expf, device              vs (float)exp(double), host", &d1_vs_h2}};
    std::printf("\nover all %llu patterns:\n", (unsigned long long) total);
    for (const auto& r : rows) {
        std::printf("  %-52s differing %12lld", r.name, r.d->n);
        if (r.d->n) {
            float x;
            std::memcpy(&x, &r.d->first_bits, 4);
            std::printf("   first: x = %.9g (0x%08x) got %.9g want %.9g", (double) x, r.d->first_bits,
                        (double) r.d->got, (double) r.d->want);
        }
        std::printf("\n");
    }
    cudaFree(dbits);
    cudaFree(dd1);
    cudaFree(dd2);
    return 0;
}
