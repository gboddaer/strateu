// tools/kolibri/grid_shape_bench.cu — plan 2026-10-28-kolibri-r9700-speed.md, Task 2.
//
// Prices the shape questions the Kolibri decode and prefill answer with launches, on whatever card is in front of
// you.  Nothing here links the engine: the dequant and the per-row dot are copies of kolibri_engine.cu's own
// (`deq32<T_Q4_K>`, `k_gemv_f16`, `block_sum`), so a number measured here transfers to the engine's kernels by
// construction rather than by hope.
//
// Arms (each prints µs/call, GB/s, and a bit pattern for the two-run self-gate):
//   A  F16 GEMV [128000 x 2560]  block-per-row (grid = n_out) vs grid-stride (grid = 1024)
//   B  Q4_K GEMV [2560 x 2560]   same pair, scalar deq32
//   C  Q4_K deq32                scalar (branch per element, byte loads) vs packed (32-bit loads, no branch)
//   D  int8 dot (emulated MAD)   vs scalar fp32 dot, [2560 x 2560] — speed probe, NOT bitwise (gfx1201 has no
//                                dot1-insts, so a real dp4a arm cannot be built on this card at all)
//   E  launch latency            18 tiny launches (6 picks x 3 slices) vs 1 launch covering the same work
//
// A, B and C are bitwise by construction: the same per-row algorithm, the same accumulation order per row, the
// same float expressions.  Only the grid (A, B) or the nibble-extraction path (C) changes.
//
// Usage: grid_shape_bench [--selftest] [--reps N]
//   --selftest  fewer repetitions, asserts the A/B/C pairs agree bitwise, exits 1 on any disagreement (CTest)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#define KC(x)                                                                                                  \
    do {                                                                                                       \
        hipError_t e_ = (x);                                                                                   \
        if (e_ != hipSuccess) {                                                                                \
            std::fprintf(stderr, "grid_shape_bench: %s failed: %s (%s:%d)\n", #x, hipGetErrorString(e_),       \
                         __FILE__, __LINE__);                                                                  \
            std::exit(1);                                                                                      \
        }                                                                                                      \
    } while (0)

constexpr int kThreads = 128;   // kolibri_engine.cu:544
enum : int { T_Q4_K = 12 };
static constexpr int kQ4KBlock = 144;   // one 256-value superblock: 2 halves + 12 packed bytes + 128 nibbles

// ---------------------------------------------------------------- the engine's dequant, copied verbatim in spirit
__device__ __forceinline__ float h2f(uint16_t h) {
    const uint32_t s = (uint32_t) (h >> 15) << 31, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e == 0) { const float v = ldexpf((float) m, -24); return s ? -v : v; }
    const uint32_t f = e == 31 ? (s | 0x7f800000u | (m << 13)) : (s | ((e + 112) << 23) | (m << 13));
    return __uint_as_float(f);
}
__device__ __forceinline__ uint16_t ld16(const uint8_t* p) { return (uint16_t) (p[0] | (p[1] << 8)); }
__device__ __forceinline__ void scale_min_k4(int j, const uint8_t* q, int& d, int& m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
/// Scalar: byte loads, one branch per element.  This is the engine's shape (kolibri_engine.cu, deq32<T_Q4_K>).
__device__ void deq32_q4k_scalar(const uint8_t* row, int c, float* out) {
    const uint8_t* b = row + kQ4KBlock * (c >> 3);
    const int j = c & 7;
    const float d = h2f(ld16(b)), dmin = h2f(ld16(b + 2));
    int sc, m;
    scale_min_k4(j, b + 4, sc, m);
    const float d1 = d * (float) sc, m1 = dmin * (float) m;
    const uint8_t* qs = b + 16 + 32 * (j >> 1);
    for (int l = 0; l < 32; ++l) out[l] = d1 * (float) ((j & 1) ? (qs[l] >> 4) : (qs[l] & 0xF)) - m1;
}
/// Packed: 32-bit loads, no per-element branch.  Same float expression per element, so bitwise identical.
__device__ void deq32_q4k_packed(const uint8_t* row, int c, float* out) {
    const uint8_t* b = row + kQ4KBlock * (c >> 3);
    const int j = c & 7;
    const float d = h2f(ld16(b)), dmin = h2f(ld16(b + 2));
    int sc, m;
    scale_min_k4(j, b + 4, sc, m);
    const float d1 = d * (float) sc, m1 = dmin * (float) m;
    const uint32_t* qs = (const uint32_t*) (b + 16 + 32 * (j >> 1));
    const bool hi = (j & 1) != 0;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        const uint32_t v = qs[w];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t byte = (v >> (8 * k)) & 0xFFu;
            out[w * 4 + k] = d1 * (float) (hi ? (int) (byte >> 4) : (int) (byte & 0xF)) - m1;
        }
    }
}
__device__ float block_sum(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}

// ---------------------------------------------------------------- arm A: F16 GEMV [n_out x n_in]
__device__ __forceinline__ float row_dot_f16(const __half* r, const __half* x, int n_in) {
    float s = 0.f;
    for (int i = threadIdx.x; i < n_in; i += blockDim.x) s += __half2float(r[i]) * __half2float(x[i]);
    return s;
}
__global__ void k_f16_row(const __half* __restrict__ W, const __half* __restrict__ x, int n_in, float* y) {
    __shared__ float sh[kThreads];
    const int o = blockIdx.x;
    y[o] = block_sum(row_dot_f16(W + (size_t) o * n_in, x, n_in), sh);
}
__global__ void k_f16_gs(const __half* __restrict__ W, const __half* __restrict__ x, int n_in, float* y, int n_out) {
    __shared__ float sh[kThreads];
    for (int o = blockIdx.x; o < n_out; o += gridDim.x)
        y[o] = block_sum(row_dot_f16(W + (size_t) o * n_in, x, n_in), sh);
}

// ---------------------------------------------------------------- arm B/C: Q4_K GEMV [n_out x n_in]
template <bool PACKED>
__device__ __forceinline__ float row_dot_q4k(const uint8_t* row, const float* x, int n_in) {
    float s = 0.f;
    for (int c = threadIdx.x; c < n_in / 32; c += blockDim.x) {
        float w[32];
        if constexpr (PACKED) deq32_q4k_packed(row, c, w);
        else deq32_q4k_scalar(row, c, w);
        const float* xr = x + 32 * c;
        for (int l = 0; l < 32; ++l) s += w[l] * xr[l];
    }
    return s;
}
template <bool PACKED>
__global__ void k_q4k_row(const uint8_t* __restrict__ W, const float* __restrict__ x, int n_in, float* y) {
    __shared__ float sh[kThreads];
    const int row_bytes = (n_in / 256) * kQ4KBlock;
    const int o = blockIdx.x;
    y[o] = block_sum(row_dot_q4k<PACKED>(W + (size_t) o * row_bytes, x, n_in), sh);
}
template <bool PACKED>
__global__ void k_q4k_gs(const uint8_t* __restrict__ W, const float* __restrict__ x, int n_in, float* y, int n_out) {
    __shared__ float sh[kThreads];
    const int row_bytes = (n_in / 256) * kQ4KBlock;
    for (int o = blockIdx.x; o < n_out; o += gridDim.x)
        y[o] = block_sum(row_dot_q4k<PACKED>(W + (size_t) o * row_bytes, x, n_in), sh);
}

// ---------------------------------------------------------------- arm D: q8-style dp4a vs scalar fp32
// The int8 side is a stand-in for ggml's q4_K x q8_1 kernel: per 32-value group, the weight is its dequantized float
// rounded to int8 by the group's own scale, the activation is rounded the same way.  It is a SPEED probe - its
// numbers are its own, not the shipped path's, and it is not bitwise against anything.
// gfx1201 has NO dot1-insts: `__builtin_amdgcn_sdot4` refuses to compile for it ("needs target feature
// dot1-insts"), so a dp4a arm cannot exist on this card.  The int8 path here is the emulated one - 32 integer
// multiply-adds per 32-value group - which is what any gfx12 int8 plan would actually pay.  Measured, not assumed.
__device__ __forceinline__ int i8_group_dot(const int8_t* w, const int8_t* a) {
    int acc = 0;
    for (int k = 0; k < 32; ++k) acc += (int) w[k] * (int) a[k];
    return acc;
}
__global__ void k_dp4a_dot(const uint8_t* __restrict__ W, const int8_t* __restrict__ aq, const float* __restrict__ ad,
                           int n_in, float* y) {
    __shared__ float sh[kThreads];
    const int row_bytes = (n_in / 256) * kQ4KBlock;
    const uint8_t* row = W + (size_t) blockIdx.x * row_bytes;
    float s = 0.f;
    for (int c = threadIdx.x; c < n_in / 32; c += blockDim.x) {
        float w[32];
        deq32_q4k_scalar(row, c, w);
        int8_t wq[32];
        float mx = 0.f;
        for (int l = 0; l < 32; ++l) mx = fmaxf(mx, fabsf(w[l]));
        const float wd = mx / 127.f;
        const float inv = mx > 0.f ? 127.f / mx : 0.f;
        for (int l = 0; l < 32; ++l) wq[l] = (int8_t) lrintf(w[l] * inv);
        const int acc = i8_group_dot(wq, aq + 32 * c);
        s += (float) acc * wd * ad[c];
    }
    y[blockIdx.x] = block_sum(s, sh);
}
__global__ void k_quant_a(const float* __restrict__ x, int n_in, int8_t* aq, float* ad) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_in / 32) return;
    const float* xr = x + 32 * c;
    float mx = 0.f;
    for (int l = 0; l < 32; ++l) mx = fmaxf(mx, fabsf(xr[l]));
    const float inv = mx > 0.f ? 127.f / mx : 0.f;
    for (int l = 0; l < 32; ++l) aq[32 * c + l] = (int8_t) lrintf(xr[l] * inv);
    ad[c] = mx / 127.f;
}

// ---------------------------------------------------------------- arm E: launch latency
__global__ void k_touch(float* p, int n) { if (threadIdx.x < n) p[blockIdx.x * blockDim.x + threadIdx.x] += 1.f; }
__global__ void k_touch_batched(float* p, int picks, int slices) {
    const int pick = blockIdx.x % picks, slice = blockIdx.x / picks;
    if (threadIdx.x < slices) p[(pick * slices + slice) * kThreads + threadIdx.x] += 1.f;
}

// ---------------------------------------------------------------- harness
static uint64_t fnv(const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*) p;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
static float bench(std::function<void()> fn, int reps) {
    std::vector<hipEvent_t> a(2);
    for (auto& e : a) KC(hipEventCreate(&e));
    for (int i = 0; i < 3; ++i) fn();   // warm
    KC(hipDeviceSynchronize());
    KC(hipEventRecord(a[0]));
    for (int i = 0; i < reps; ++i) fn();
    KC(hipEventRecord(a[1]));
    KC(hipEventSynchronize(a[1]));
    float ms = 0.f;
    KC(hipEventElapsedTime(&ms, a[0], a[1]));
    for (auto& e : a) KC(hipEventDestroy(e));
    return ms / (float) reps;
}
static void report(const char* arm, float ms, double bytes) {
    std::printf("%-42s %9.4f ms  %8.1f GB/s\n", arm, ms, bytes / (ms * 1e6));
    std::fflush(stdout);
}

int main(int argc, char** argv) {
    bool selftest = false;
    int reps = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--selftest") == 0) selftest = true;
        else if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc) reps = std::atoi(argv[++i]);
    }
    int dev = 0;
    KC(hipGetDevice(&dev));
    hipDeviceProp_t prop{};
    KC(hipGetDeviceProperties(&prop, dev));
    std::printf("grid_shape_bench on %s (%s), %d CUs\n", prop.name, prop.gcnArchName, prop.multiProcessorCount);

    const int n_out_f16 = 128000, n_in_f16 = 2560;    // the LM head's shape
    const int n_out_q4k = 2560, n_in_q4k = 2560;      // a dense projection
    const size_t f16_bytes = (size_t) n_out_f16 * n_in_f16 * 2;
    const size_t q4k_row_bytes = (size_t) (n_in_q4k / 256) * kQ4KBlock;
    const size_t q4k_bytes = q4k_row_bytes * n_out_q4k;

    if (reps <= 0) reps = selftest ? 8 : 30;
    const int reps_f16 = selftest ? 4 : 20, reps_q4k = reps * 8;

    __half *W16 = nullptr, *x16 = nullptr;
    float *y1 = nullptr, *y2 = nullptr, *y3 = nullptr, *x = nullptr;
    uint8_t* Wq = nullptr;
    int8_t* aq = nullptr;
    float* ad = nullptr;
    KC(hipMalloc(&W16, f16_bytes));
    KC(hipMalloc(&x16, (size_t) n_in_f16 * 2));
    KC(hipMalloc(&y1, (size_t) n_out_f16 * 4));
    KC(hipMalloc(&y2, (size_t) n_out_f16 * 4));
    KC(hipMalloc(&y3, (size_t) n_out_f16 * 4));
    KC(hipMalloc(&x, (size_t) n_in_f16 * 4));
    KC(hipMalloc(&Wq, q4k_bytes));
    KC(hipMalloc(&aq, (size_t) (n_in_q4k / 32) * 32));
    KC(hipMalloc(&ad, (size_t) (n_in_q4k / 32) * 4));

    {   // deterministic filler: no RNG dependence, same data every run
        std::vector<__half> h((size_t) n_out_f16 * n_in_f16);
        for (size_t i = 0; i < h.size(); ++i) h[i] = __float2half_rn(((int) (i * 2654435761u % 1000) - 500) / 500.f);
        KC(hipMemcpy(W16, h.data(), f16_bytes, hipMemcpyHostToDevice));
        std::vector<__half> hx(n_in_f16);
        for (int i = 0; i < n_in_f16; ++i) hx[i] = __float2half_rn(((int) (i * 40503u % 200) - 100) / 100.f);
        KC(hipMemcpy(x16, hx.data(), (size_t) n_in_f16 * 2, hipMemcpyHostToDevice));
        std::vector<float> fx(n_in_f16);
        for (int i = 0; i < n_in_f16; ++i) fx[i] = ((int) (i * 40503u % 200) - 100) / 100.f;
        KC(hipMemcpy(x, fx.data(), (size_t) n_in_f16 * 4, hipMemcpyHostToDevice));
        std::vector<uint8_t> hw((size_t) q4k_bytes);
        for (size_t i = 0; i < hw.size(); ++i) hw[i] = (uint8_t) (i * 1664525u + 1013904223u >> 13);
        for (size_t b = 0; b < q4k_bytes / kQ4KBlock; ++b) {   // valid fp16 scales: 0.05 .. 0.2, no NaN/Inf
            const size_t o = b * kQ4KBlock;
            const __half d = __float2half_rn(0.05f + (float) (b % 16) * 0.01f);
            std::memcpy(&hw[o], &d, 2);
            std::memcpy(&hw[o + 2], &d, 2);
        }
        KC(hipMemcpy(Wq, hw.data(), q4k_bytes, hipMemcpyHostToDevice));
    }
    KC(hipMemset(y1, 0, (size_t) n_out_f16 * 4));
    KC(hipMemset(y2, 0, (size_t) n_out_f16 * 4));
    KC(hipMemset(y3, 0, (size_t) n_out_f16 * 4));
    KC(hipMemset(ad, 0, (size_t) (n_in_q4k / 32) * 4));
    KC(hipMemset(aq, 0, (size_t) (n_in_q4k / 32) * 32));
    k_quant_a<<<(n_in_q4k / 32 + 127) / 128, 128>>>(x, n_in_q4k, aq, ad);
    KC(hipGetLastError());
    KC(hipDeviceSynchronize());

    // ---- A: F16 GEMV, the LM head's shape
    const float a_row = bench([&] { k_f16_row<<<n_out_f16, kThreads>>>(W16, x16, n_in_f16, y1); }, reps_f16);
    const float a_gs = bench([&] { k_f16_gs<<<1024, kThreads>>>(W16, x16, n_in_f16, y2, n_out_f16); }, reps_f16);
    report("A1 f16 gemv block-per-row (128000 blk)", a_row, (double) f16_bytes);
    report("A2 f16 gemv grid-stride (1024 blk)", a_gs, (double) f16_bytes);
    KC(hipDeviceSynchronize());
    std::vector<float> ha(n_out_f16), hb(n_out_f16);
    KC(hipMemcpy(ha.data(), y1, (size_t) n_out_f16 * 4, hipMemcpyDeviceToHost));
    KC(hipMemcpy(hb.data(), y2, (size_t) n_out_f16 * 4, hipMemcpyDeviceToHost));
    const bool a_ok = std::memcmp(ha.data(), hb.data(), (size_t) n_out_f16 * 4) == 0;
    std::printf("CHECK A bitwise (row vs grid-stride): %s   HASH %016llx\n", a_ok ? "identical" : "DIFFER",
                (unsigned long long) fnv(ha.data(), (size_t) n_out_f16 * 4));

    // ---- B: Q4_K GEMV, scalar dequant
    const float b_row = bench([&] { k_q4k_row<false><<<n_out_q4k, kThreads>>>(Wq, x, n_in_q4k, y1); }, reps_q4k);
    const float b_gs = bench([&] { k_q4k_gs<false><<<256, kThreads>>>(Wq, x, n_in_q4k, y2, n_out_q4k); }, reps_q4k);
    report("B1 q4k gemv block-per-row (2560 blk)", b_row, (double) q4k_bytes);
    report("B2 q4k gemv grid-stride (256 blk)", b_gs, (double) q4k_bytes);
    KC(hipDeviceSynchronize());
    std::vector<float> qa(n_out_q4k), qb(n_out_q4k);
    KC(hipMemcpy(qa.data(), y1, (size_t) n_out_q4k * 4, hipMemcpyDeviceToHost));
    KC(hipMemcpy(qb.data(), y2, (size_t) n_out_q4k * 4, hipMemcpyDeviceToHost));
    const bool b_ok = std::memcmp(qa.data(), qb.data(), (size_t) n_out_q4k * 4) == 0;
    std::printf("CHECK B bitwise (row vs grid-stride): %s   HASH %016llx\n", b_ok ? "identical" : "DIFFER",
                (unsigned long long) fnv(qa.data(), (size_t) n_out_q4k * 4));

    // ---- C: the dequant itself, packed vs scalar, through the same GEMV shape
    const float c_packed = bench([&] { k_q4k_row<true><<<n_out_q4k, kThreads>>>(Wq, x, n_in_q4k, y3); }, reps_q4k);
    report("C1 q4k gemv, packed deq32 (2560 blk)", c_packed, (double) q4k_bytes);
    KC(hipDeviceSynchronize());
    std::vector<float> qc(n_out_q4k);
    KC(hipMemcpy(qc.data(), y3, (size_t) n_out_q4k * 4, hipMemcpyDeviceToHost));
    const bool c_ok = std::memcmp(qa.data(), qc.data(), (size_t) n_out_q4k * 4) == 0;
    std::printf("CHECK C bitwise (scalar vs packed deq32): %s   HASH %016llx\n", c_ok ? "identical" : "DIFFER",
                (unsigned long long) fnv(qc.data(), (size_t) n_out_q4k * 4));

    // ---- D: dp4a vs scalar fp32 (speed probe; its numerics are its own)
    const float d_dp4a = bench([&] { k_dp4a_dot<<<n_out_q4k, kThreads>>>(Wq, aq, ad, n_in_q4k, y3); }, reps_q4k);
    report("D1 int8 MAD dot, dot1 emulated (2560 blk)", d_dp4a, (double) q4k_bytes);
    KC(hipDeviceSynchronize());
    std::vector<float> qd(n_out_q4k);
    KC(hipMemcpy(qd.data(), y3, (size_t) n_out_q4k * 4, hipMemcpyDeviceToHost));
    double max_rel = 0, ref = 0;
    for (int i = 0; i < n_out_q4k; ++i) {
        const double dd = std::fabs((double) qd[i] - (double) qa[i]);
        const double r = std::fabs((double) qa[i]) + 1e-6;
        max_rel = std::max(max_rel, dd / r);
        ref = std::max(ref, std::fabs((double) qa[i]));
    }
    std::printf("NOTE D numerics: max rel diff %.3g against the fp32 reference (max |y| %.3g) - a probe, not a "
                "candidate for a bitwise knob\n", max_rel, ref);

    // ---- E: what a launch costs when the work per launch is tiny (the 6 picks x 3 slices pattern)
    float* touch = nullptr;
    KC(hipMalloc(&touch, (size_t) 6 * 3 * kThreads * 4));
    KC(hipMemset(touch, 0, (size_t) 6 * 3 * kThreads * 4));
    const int e_reps = selftest ? 50 : 200;
    const float e18 = bench([&] {
        for (int p = 0; p < 6; ++p)
            for (int s = 0; s < 3; ++s) k_touch<<<1, kThreads>>>(touch + (p * 3 + s) * kThreads, kThreads);
    }, e_reps);
    const float e1 = bench([&] { k_touch_batched<<<18, kThreads>>>(touch, 6, 3); }, e_reps);
    report("E1 18 tiny launches (6 picks x 3 slices)", e18, 0.0);
    report("E2 1 batched launch (18 blocks)", e1, 0.0);
    std::printf("NOTE E launch overhead: %.2f us per launch (18-launch arm minus batched arm, / 17)\n",
                (e18 - e1) * 1e3 / 17.0);

    const bool ok = a_ok && b_ok && c_ok;
    std::printf("%s\n", ok ? "SELFTEST PASS: A, B and C bitwise" : "SELFTEST FAIL");
    KC(hipFree(touch));
    return ok ? 0 : 1;
}
