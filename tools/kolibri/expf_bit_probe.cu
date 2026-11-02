// Can a port of glibc's expf compute, on the device, exactly the bits this host's expf computes?  Exhaustive.
//
// Section 8 of docs/experiments/kolibri-stream-kernels-layout.md stopped the device-side SwiGLU on a measured rate;
// section 9 turned the rate into a proof over all 2^32 float patterns and showed no stock device exp (CUDA's expf,
// or (float)exp(double)) matches this host's glibc expf everywhere.  This is the route that moves nothing
// numerically: port glibc's own algorithm (sysdeps/ieee754/flt-32/e_expf.c and e_exp2f_data.c) and check it against
// the host's expf over every float.
//
// The port has one freedom the source does not show: glibc is compiled by GCC with its default -ffp-contract=fast,
// so each of the three `a * b + c` expressions may or may not be a single fma in the binary this host runs.  All
// eight combinations are therefore swept, and the answer is which one differs in nothing:
//
//   bit 0: z = C0*r + C1      bit 1: y = C2*r + 1      bit 2: y = z*r2 + y
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

constexpr int kVariants = 3;   // 0 = the port, 1/2 = smallest/largest |x| that differs, 2 handled below

// sysdeps/ieee754/flt-32/e_exp2f_data.c, verbatim.  tab[i] = bits(2^(i/32)) - (i << 47); the shift constants are
// the source's own expressions, evaluated in double in the source's order.
__device__ __constant__ unsigned long long kTab[32] = {
    0x3ff0000000000000ull, 0x3fefd9b0d3158574ull, 0x3fefb5586cf9890full, 0x3fef9301d0125b51ull,
    0x3fef72b83c7d517bull, 0x3fef54873168b9aaull, 0x3fef387a6e756238ull, 0x3fef1e9df51fdee1ull,
    0x3fef06fe0a31b715ull, 0x3feef1a7373aa9cbull, 0x3feedea64c123422ull, 0x3feece086061892dull,
    0x3feebfdad5362a27ull, 0x3feeb42b569d4f82ull, 0x3feeab07dd485429ull, 0x3feea47eb03a5585ull,
    0x3feea09e667f3bcdull, 0x3fee9f75e8ec5f74ull, 0x3feea11473eb0187ull, 0x3feea589994cce13ull,
    0x3feeace5422aa0dbull, 0x3feeb737b0cdc5e5ull, 0x3feec49182a3f090ull, 0x3feed503b23e255dull,
    0x3feee89f995ad3adull, 0x3feeff76f2fb5e47ull, 0x3fef199bdd85529cull, 0x3fef3720dcef9069ull,
    0x3fef5818dcfba487ull, 0x3fef7c97337b9b5full, 0x3fefa4afa2a490daull, 0x3fefd0765b6e4540ull};

__device__ __constant__ double kShift = 0x1.8p+52;
__device__ __constant__ double kInvLn2N = 0x1.71547652b82fep+0 * 32.0;
__device__ __constant__ double kC0 = 0x1.c6af84b912394p-5 / 32.0 / 32.0 / 32.0;
__device__ __constant__ double kC1 = 0x1.ebfce50fac4f3p-3 / 32.0 / 32.0;
__device__ __constant__ double kC2 = 0x1.62e42ff0c52d6p-1 / 32.0;

__device__ __forceinline__ double dmul(double a, double b) { return __dmul_rn(a, b); }
__device__ __forceinline__ double dadd(double a, double b) { return __dadd_rn(a, b); }
__device__ __forceinline__ double dsub(double a, double b) { return __dadd_rn(a, -b); }

/// glibc 2.36's expf as this host actually runs it: the FMA variant (x86-64/fpu/multiarch/e_expf-fma.c, the same
/// source built with -mfma, selected here because this CPU has FMA).  Transcribed from the disassembly of the
/// resolved target, not from the C source, because the two differ: GCC kept `InvLn2N * xd + SHIFT` and
/// `InvLn2N * xd - k` fused, so r is one rounding of the exact product difference rather than (rounded product) - k.
__device__ __forceinline__ float expf_port(float x) {
    const uint32_t ux = __float_as_uint(x);
    const uint32_t abstop = (ux >> 20) & 0x7ff;
    if (abstop > 0x42a) {                      // > top12(88.0f): the special path (as the binary compares it)
        if (ux == 0xff800000u) return 0.0f;    // -inf
        if (abstop > 0x7f7) return x + x;      // |x| >= inf, or nan
        if (x > 0x1.62e42ep6f) return __int_as_float(0x7f800000);   // log(0x1p128): overflow
        if (x < -0x1.9d1d9ep6f) return 0.0f;                        // log(0x1p-149): underflow, in either form
    }
    const double xd = (double) x;
    const double kd = __fma_rn(kInvLn2N, xd, kShift);   // vfmadd132sd: the bias add is fused
    const unsigned long long ki = __double_as_longlong(kd);
    const double k = __dadd_rn(kd, -kShift);            // vsubsd
    const double r = __fma_rn(kInvLn2N, xd, -k);        // vfmsub132sd: r is one rounding, not (product) - k
    unsigned long long t = kTab[ki & 31];
    t += ki << (52 - 5);
    const double s = __longlong_as_double(t);
    const double r2 = __dmul_rn(r, r);
    const double z1 = __fma_rn(kC0, r, kC1);
    const double y1 = __fma_rn(kC2, r, 1.0);
    const double y2 = __fma_rn(z1, r2, y1);
    return (float) __dmul_rn(y2, s);
}

__global__ void sweep(const uint32_t* __restrict__ bits, const float* __restrict__ ref, int n,
                      long long* __restrict__ cnt, uint32_t* __restrict__ first) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x;
    __builtin_memcpy(&x, &bits[i], 4);
    const uint32_t want = __float_as_uint(ref[i]);
    // NaN payloads are canonicalised differently by the two ISAs (x86 quiets and keeps them, NVIDIA does not), and
    // this path cannot produce a NaN input, so they are counted apart instead of pretending to match.
    if ((__float_as_uint(x) & 0x7f800000u) == 0x7f800000u && (__float_as_uint(x) & 0x007fffffu) != 0u) {
        atomicAdd((unsigned long long*) &cnt[2], 1ull);
        return;
    }
    const float got = expf_port(x);
    if (__float_as_uint(got) != want) {
        atomicAdd((unsigned long long*) &cnt[0], 1ull);
        const uint32_t mag = bits[i] & 0x7fffffffu;   // compare magnitudes: negative x and positive x are symmetric
        atomicMin(&first[0], mag);
        atomicMax(&first[1], mag);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const int chunk_log2 = argc > 1 ? std::atoi(argv[1]) : 24;
    const uint32_t chunk = 1u << chunk_log2;
    const uint64_t total = 1ull << 32;
    const unsigned nthreads = std::thread::hardware_concurrency() > 1 ? std::thread::hardware_concurrency() - 1 : 1;

    std::vector<uint32_t> bits(chunk);
    std::vector<float> ref(chunk);
    uint32_t* dbits = nullptr;
    float* dref = nullptr;
    long long* dcnt = nullptr;
    uint32_t* dfirst = nullptr;
    cudaMalloc(&dbits, chunk * 4);
    cudaMalloc(&dref, chunk * 4);
    cudaMalloc(&dcnt, kVariants * 8);
    cudaMalloc(&dfirst, kVariants * 4);
    long long cnt[kVariants] = {0};
    uint32_t first[kVariants];
    for (int v = 0; v < kVariants; ++v) first[v] = (v == 2) ? 0u : 0xffffffffu;
    cudaMemset(dcnt, 0, kVariants * 8);
    cudaMemset(dfirst, 0xff, kVariants * 4);

    std::printf("a port of this host's expf (FMA variant, from its disassembly) against it, over every one of %llu\n"
                "float bit patterns (chunk 2^%d, %u host threads)\n",
                (unsigned long long) total, chunk_log2, nthreads);

    for (uint64_t base = 0; base < total; base += chunk) {
        for (uint32_t i = 0; i < chunk; ++i) bits[i] = (uint32_t) (base + i);
        const unsigned per = (chunk + nthreads - 1) / nthreads;
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < nthreads; ++t)
            pool.emplace_back([&, t] {
                const uint32_t b0 = t * per, b1 = (b0 + per) < chunk ? (b0 + per) : chunk;
                for (uint32_t i = b0; i < b1; ++i) {
                    float x;
                    std::memcpy(&x, &bits[i], 4);
                    ref[i] = std::exp(x);   // the host's expf: the reference
                }
            });
        for (std::thread& th : pool) th.join();

        cudaMemcpy(dbits, bits.data(), chunk * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dref, ref.data(), chunk * 4, cudaMemcpyHostToDevice);
        sweep<<<(chunk + 255) / 256, 256>>>(dbits, dref, (int) chunk, dcnt, dfirst);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::fprintf(stderr, "kernel: %s\n", cudaGetErrorString(cudaGetLastError()));
            return 1;
        }
        if (((base / chunk) & 31) == 31) {
            long long c[kVariants];
            cudaMemcpy(c, dcnt, kVariants * 8, cudaMemcpyDeviceToHost);
            std::printf("  ... %5.1f %%  ", 100.0 * (double) (base + chunk) / total);
            for (int v = 0; v < kVariants; ++v) std::printf("v%d %lld  ", v, c[v]);
            std::printf("\n");
            std::fflush(stdout);
        }
    }
    cudaMemcpy(cnt, dcnt, kVariants * 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(first, dfirst, kVariants * 4, cudaMemcpyDeviceToHost);
    std::printf("\nover all %llu patterns, per variant:\n", (unsigned long long) total);
    {
        float lo, hi;
        std::memcpy(&lo, &first[1], 4);
        std::memcpy(&hi, &first[2], 4);
        std::printf("  differing |x| from %.9g to %.9g\n", (double) lo, (double) hi);
    }
    std::printf("  the port differing          %12lld", cnt[0]);
    if (cnt[0]) {
        float x;
        std::memcpy(&x, &first[0], 4);
        std::printf("   first pattern 0x%08x (x = %.9g)", first[0], (double) x);
    }
    std::printf("\n  NaN inputs (counted apart)  %12lld\n", cnt[2]);
    cudaFree(dbits);
    cudaFree(dref);
    cudaFree(dcnt);
    cudaFree(dfirst);
    return 0;
}
