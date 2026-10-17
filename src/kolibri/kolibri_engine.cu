// src/kolibri/kolibri_engine.cu - the engine path for Kolibri 1 (GGUF architecture kolibri1): a driver of its
// own beside Flash-Next's, on the src/qwen35 template (plan docs/superpowers/plans/2026-10-16-kolibri1.md).
// It grows task by task: T3 the stub, T6 the reference forward (kolibriref.cu), T7 this file - the weight load,
// the expert repack and the hybrid placement; T8 the layer driver and serve protocol.  `strata` switches to it
// from the file's architecture at the top of its main(); strata-kolibri wraps the same entry point.
//
// The model (docs/KOLIBRI1.md): 50 layers, every layer MoE (384 routed experts top-6 + 1 ungated shared,
// FFN 512, n_embd 2560); the router selects on `logits + expert_bias` and weights by the unbiased
// `sigmoid(logits)`, no renorm (expert_gating_func 5); GQA 48/4 x 128 with q/k RMSNorm; per-layer
// four sliding-window (513, RoPE) to one full-attention (NoPE); sandwich norms around both branches.
// No GDN, no hyper-connections, no indexer, no MTP, no PLE.
//
// The weights, as the plan's Global Constraints fix them: the dense tensors (attention, router, shared expert,
// norms, embeddings) sit on the GPU in their GGUF types (the MMVQ kernels read them as is); the 19,200 routed
// experts are repacked once at load into native [gate|up|down] blobs (native_expert_layout) in a pinned host
// table - the GPU is the expert cache, the model is bigger than any PC's VRAM (Q4_K_M: 42.6 GiB of experts,
// 3.5 GiB of dense).  --load-only runs just the load and reports.
#include "strata/kolibri/engine.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/family_layout.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <functional>
#include "strata/kernels/native_router.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "strata/prefill/gemm.hpp"
#include <cuda_fp16.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#define KC(x)                                                                                                             \
    do {                                                                                                                  \
        cudaError_t e_ = (x);                                                                                             \
        if (e_ != cudaSuccess) {                                                                                          \
            std::fprintf(stderr, "strata-kolibri: %s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__,         \
                         __LINE__);                                                                                       \
            std::exit(1);                                                                                                 \
        }                                                                                                                 \
    } while (0)

namespace {

size_t row_bytes(int type, int64_t n) {
    int be = 0, bb = 0;
    if (!strata::block_geometry((uint32_t) type, be, bb)) return 0;
    return (size_t) (n / be) * (size_t) bb;
}
size_t lcm(size_t a, size_t b) { size_t x = a, y = b; while (y) { size_t t = x % y; x = y; y = t; } return a / x * b; }

struct Dense {
    const uint8_t* d = nullptr;   // device
    int type = -1;
    std::vector<int64_t> shape;
    size_t row_bytes = 0;
};

struct LayerBlobs {
    strata::kernels::NativeExpertLayout L;
    size_t stride = 0;              // blob slot bytes, block-aligned (the lcm of the types' super-blocks)
    uint8_t* table = nullptr;       // pinned host: n_expert slots + 1 zero slot (the no-pool overflow path)
    uint8_t* dev = nullptr;         // --expert-slots: device table, slots slots + 1 zero slot
    bool native_ok = false;         // native_expert_supported for this layer's (gu, d) at these dims
    strata::kernels::cpu::NativeFmt fmt;  // the CPU pool's view of the same blob (native_fmt at load)
};

/// A persistent worker pool: run(f, n) splits [0, n) over n shards across the pool (caller runs shard 0) and
/// returns when all shards are done.  --cpu-threads sizes it; the CPU expert pool is its user.
struct Pool {
    int n_threads = 0;
    std::vector<std::thread> workers;
    std::mutex mu;
    std::condition_variable work_cv, done_cv;
    std::function<void(int)> fn;
    unsigned long gen = 0;          // one generation per run(): a worker consumes each generation once
    std::atomic<int> left{0};
    int n_shards = 0;
    bool quit = false;
    void start(int t) {
        n_threads = t;
        for (int i = 1; i < t; ++i) workers.emplace_back([this, i] {
            unsigned long my_gen = 0;
            for (;;) {
                std::unique_lock<std::mutex> lk(mu);
                work_cv.wait(lk, [&] { return quit || gen > my_gen; });
                if (quit) return;
                my_gen = gen;
                const int n = n_shards;   // a run with fewer shards than workers leaves the idle ones waiting
                if (i >= n) continue;
                lk.unlock();
                fn(i);
                lk.lock();
                if (--left == 0) done_cv.notify_one();
            }
        });
    }
    ~Pool() { { std::lock_guard<std::mutex> lk(mu); quit = true; } work_cv.notify_all(); for (auto& w : workers) w.join(); }
    void run(std::function<void(int)> f, int n) {
        if (n <= 1) { f(0); return; }
        { std::lock_guard<std::mutex> lk(mu); fn = std::move(f); n_shards = n; left = n - 1; ++gen; }
        work_cv.notify_all();
        fn(0);
        std::unique_lock<std::mutex> lk(mu);
        done_cv.wait(lk, [&] { return left.load() == 0; });
    }
};

struct Engine {
    strata::core::ModelGeometry g;
    int64_t top_k = 6;
    float eps = 1e-6f, rope_base = 1e4f;
    int cpu_threads = 0;            // --cpu-threads: routed experts on the CPU pool (ggml-cpu vec_dot, no PCIe)
    int slots = 0;                  // --expert-slots: experts per layer resident in VRAM (the GPU expert cache)
    int grouped = 0;                // --grouped: native_expert_grouped for the routed experts (one launch per layer)
    int graphs = 0;                 // --grouped + graphs (default on with --grouped): per-layer CUDA graphs
    int prefill = 0;                // --prefill: chunked prompt processing (GEMM projections, FP16 attention)
    cublasHandle_t blas = nullptr;
    strata::prefill::Gemm gemm;
    std::vector<unsigned long long> gexec;   // cudaGraphExec_t per layer (0 = the layer is not capturable)
    mutable Pool pool;              // the worker pool behind the CPU expert path
    std::map<std::string, Dense> dense;
    std::map<std::string, float*> f32;
    std::vector<LayerBlobs> blobs;   // per layer
};

// ------------------------------------------------------------------ the forward kernels (the kolibriref set, plain:
// exactness first; the native grouped-expert path over the blob table arrives with the placement work in T8+)
enum : int { T_F32 = 0, T_F16 = 1, T_Q8_0 = 8, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14, T_BF16 = 30 };

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
template <int T> __device__ void deq32(const uint8_t* row, int c, float* out) {
    if constexpr (T == T_F32) { const float* r = (const float*) row + 32 * c; for (int l = 0; l < 32; ++l) out[l] = r[l]; }
    else if constexpr (T == T_F16) { const uint8_t* r = row + 64 * c; for (int l = 0; l < 32; ++l) out[l] = h2f(ld16(r + 2 * l)); }
    else if constexpr (T == T_BF16) { const uint8_t* r = row + 64 * c; for (int l = 0; l < 32; ++l) out[l] = __uint_as_float((uint32_t) ld16(r + 2 * l) << 16); }
    else if constexpr (T == T_Q8_0) { const uint8_t* b = row + 34 * c; const float d = h2f(ld16(b)); const int8_t* q = (const int8_t*) (b + 2); for (int l = 0; l < 32; ++l) out[l] = d * (float) q[l]; }
    else if constexpr (T == T_Q4_K) {
        const uint8_t* b = row + 144 * (c >> 3); const int j = c & 7;
        const float d = h2f(ld16(b)), dmin = h2f(ld16(b + 2));
        int sc, m; scale_min_k4(j, b + 4, sc, m);
        const float d1 = d * (float) sc, m1 = dmin * (float) m;
        const uint8_t* qs = b + 16 + 32 * (j >> 1);
        for (int l = 0; l < 32; ++l) out[l] = d1 * (float) ((j & 1) ? (qs[l] >> 4) : (qs[l] & 0xF)) - m1;
    } else if constexpr (T == T_Q5_K) {
        const uint8_t* b = row + 176 * (c >> 3); const int j = c & 7;
        const float d = h2f(ld16(b)), dmin = h2f(ld16(b + 2));
        int sc, m; scale_min_k4(j, b + 4, sc, m);
        const float d1 = d * (float) sc, m1 = dmin * (float) m;
        const uint8_t* qh = b + 16; const uint8_t* qs = b + 48 + 32 * (j >> 1);
        for (int l = 0; l < 32; ++l) { const int q = ((j & 1) ? (qs[l] >> 4) : (qs[l] & 0xF)) + ((qh[l] >> j) & 1 ? 16 : 0); out[l] = d1 * (float) q - m1; }
    } else if constexpr (T == T_Q6_K) {
        const uint8_t* b = row + 210 * (c >> 3); const int j = c & 7, n = j >> 2, r = j & 3;
        const uint8_t* ql = b + 64 * n; const uint8_t* qh = b + 128 + 32 * n;
        const int8_t* sc = (const int8_t*) (b + 192) + 8 * n;
        const float d = h2f(ld16(b + 208));
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16; int q;
            if (r == 0) q = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
            else if (r == 1) q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
            else if (r == 2) q = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
            else q = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
            out[l] = d * (float) sc[is + 2 * r] * (float) (q - 32);
        }
    }
}
__device__ float block_sum(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if ((int) threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s]; __syncthreads(); }
    const float r = sh[0];
    __syncthreads();
    return r;
}
constexpr int kThreads = 128;
cudaStream_t g_st = nullptr;   // the router kernels require a created stream (kolibriref does the same)
template <int T>
__global__ void k_gemv(const uint8_t* W, size_t row_bytes, size_t expert_bytes, const int* ids, int n_in, const float* x,
                       size_t x_stride, float* y, int n_out) {
    __shared__ float sh[kThreads];
    const int o = blockIdx.x, sel = blockIdx.y;
    const int e = ids ? ids[sel] : 0;
    const uint8_t* row = W + (size_t) e * expert_bytes + (size_t) o * row_bytes;
    const float* xs = x + (size_t) sel * x_stride;
    float acc = 0.f, v[32];
    for (int c = threadIdx.x; c < n_in / 32; c += blockDim.x) {
        deq32<T>(row, c, v);
        const float* xc = xs + 32 * c;
        for (int l = 0; l < 32; ++l) acc += v[l] * xc[l];
    }
    acc = block_sum(acc, sh);
    if (threadIdx.x == 0) y[(size_t) sel * n_out + o] = acc;
}
template <int T> __device__ void k_row_body(const uint8_t* W, size_t row_bytes, int row, int n, float* out) {
    const uint8_t* r = W + (size_t) row * row_bytes;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < n / 32; c += gridDim.x * blockDim.x) deq32<T>(r, c, out + 32 * c);
}
template <int T> __global__ void k_row(const uint8_t* W, size_t row_bytes, int row, int n, float* out) {
    k_row_body<T>(W, row_bytes, row, n, out);
}
/// graph decode: the row (token id) from device memory
template <int T> __global__ void k_row_p(const uint8_t* W, size_t row_bytes, const int* row_p, int n, float* out) {
    k_row_body<T>(W, row_bytes, *row_p, n, out);
}

__global__ void k_rms(const float* x, const float* w, float* y, int n, float eps) {
    __shared__ float sh[kThreads];
    const float* xg = x + (size_t) blockIdx.x * n;
    float* yg = y + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xg[i] * xg[i];
    s = block_sum(s, sh);
    const float r = 1.0f / sqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yg[i] = xg[i] * r * (w ? w[i] : 1.0f);
}
__device__ __forceinline__ float silu(float v) { return v / (1.0f + expf(-v)); }
__global__ void k_swiglu(const float* g, const float* u, float* h, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = silu(g[i]) * u[i];
}
__global__ void k_add(float* x, const float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}
static inline dim3 grid1(int64_t n, int block = kThreads) { return dim3((unsigned) ((n + block - 1) / block)); }
__device__ float block_max(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    return sh[0];
}
__global__ void k_pos_add(int* pos) { *pos += 1; }
__global__ void k_to_f16(const float* x, __half* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}
/// The prompt attention's KV images: the P cells the chunk reads, FP16.  Full layers: rows = ctx, cells 0..P-1.
__global__ void k_kv_f16_full(const float* kc, const float* vc, __half* k16, __half* v16, int64_t ctx, int P, int HD,
                              int HKV) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t per = (int64_t) P * HD;
    if (i >= (int64_t) HKV * per) return;
    const int64_t kvh = i / per, rest = i % per;
    k16[i] = __float2half(kc[(size_t) kvh * ctx * HD + rest]);
    v16[i] = __float2half(vc[(size_t) kvh * ctx * HD + rest]);
}
/// Sliding layers: the ring's last n_vis cells (global first..first+n_vis-1, slot gp % swa).
__global__ void k_kv_f16_swa(const float* kc, const float* vc, __half* k16, __half* v16, int64_t swa, int n_vis,
                             int64_t first, int HD, int HKV) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t per = (int64_t) n_vis * HD;
    if (i >= (int64_t) HKV * per) return;
    const int64_t kvh = i / per, rest = i % per;
    const int64_t c = rest / HD, d = rest % HD;
    const int64_t slot = (first + c) % swa;
    k16[i] = __float2half(kc[((size_t) kvh * swa + slot) * HD + d]);
    v16[i] = __float2half(vc[((size_t) kvh * swa + slot) * HD + d]);
}
/// The prompt's causal softmax (window-masked for sliding layers: cells before pos0+t+1-swa are zero): row (h, t)
/// of S (ld floats, P cells) -> softmax(S * scale) over [lo, pos0+t+1), FP16 out.  grid (T, H), block 256.
__global__ void k_attn_softmax16(const float* S, __half* S16, int ld, int T, int P, int pos0, float scale,
                                 int64_t swa) {
    __shared__ float sh[kThreads];
    const int t = blockIdx.x, h = blockIdx.y;
    const float* row = S + ((size_t) h * T + t) * ld;
    __half* out = S16 + ((size_t) h * T + t) * ld;
    const int n = min(P, pos0 + t + 1);
    const int lo = swa > 0 ? (pos0 + t + 1 > (int) swa ? pos0 + t + 1 - (int) swa : 0) : 0;
    float mx = -INFINITY;
    for (int p = threadIdx.x + lo; p < n; p += blockDim.x) mx = fmaxf(mx, row[p] * scale);
    mx = block_max(mx, sh);
    float sum = 0.f;
    for (int p = threadIdx.x + lo; p < n; p += blockDim.x) sum += expf(row[p] * scale - mx);
    sum = block_sum(sum, sh);
    const float inv = 1.0f / sum;
    for (int p = threadIdx.x; p < P; p += blockDim.x)
        out[p] = __float2half(p >= lo && p < n ? expf(row[p] * scale - mx) * inv : 0.f);
}
/// rms over T rows: row t of x -> row t of out.  grid (T), block kThreads.
__global__ void k_rms_rows(const float* x, const float* w, float* out, int n, float eps) {
    __shared__ float sh[kThreads];
    const float* src = x + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += src[i] * src[i];
    s = block_sum(s, sh);
    const float r = 1.0f / sqrtf(s / (float) n + eps);
    float* dst = out + (size_t) blockIdx.x * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i] * r * w[i];
}
__global__ void k_add_rows(float* x, const float* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}
__global__ void k_swiglu_rows(const float* g, const float* u, float* out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (g[i] / (1.f + expf(-g[i]))) * u[i];
}
/// moe[t] = sy[t] + sum_k wts[t*K+k] * gout[(t*K+k)]: grid (T, ceil(N/B)), block B.
__global__ void k_moe_sum_rows(float* moe, const float* gout, const float* wts, const float* sy, int K, int N) {
    const int t = blockIdx.x;
    const int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float acc = sy[(size_t) t * N + i];
    for (int k = 0; k < K; ++k) acc += wts[(size_t) t * K + k] * gout[((size_t) t * K + k) * (size_t) N + i];
    moe[(size_t) t * N + i] = acc;
}
/// token embedding rows: row toks[t] of W -> row t of out.  grid (T, 4), block 128.
template <int TY> __global__ void k_row_multi(const uint8_t* W, size_t row_bytes, const int* toks, int n, float* out) {
    const uint8_t* r = W + (size_t) toks[blockIdx.x] * row_bytes;
    float* dst = out + (size_t) blockIdx.x * n;
    for (int c = blockIdx.y * blockDim.x + threadIdx.x; c < n / 32; c += 4 * blockDim.x) deq32<TY>(r, c, dst + 32 * c);
}
/// multi-token q/k/v prep: block (head, t); pos = p0 + t.  q rows T x H x D, k/v rows T x HKV x D.
__global__ void k_attn_prep_multi(const float* qx, const float* kx, const float* vx, const float* qw, const float* kw,
                                  float* q, float* kc, float* vc, int H, int HKV, int D, int64_t p0, int64_t swa,
                                  int ctx, float base, float eps, bool rope) {
    const int t = blockIdx.y;
    const int head = blockIdx.x;
    const bool isq = head < H;
    const int kvh = head - H;
    const float* src = isq ? qx + ((size_t) t * H + head) * D : kx + ((size_t) t * HKV + kvh) * D;
    const int pos = (int) (p0 + t);
    const int64_t rows = rope ? swa : ctx;
    const int64_t slot = rope ? (int64_t) pos % swa : (int64_t) pos;
    float s = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) s += src[i] * src[i];
    __shared__ float sh[kThreads];
    s = block_sum(s, sh);
    const float r = 1.0f / sqrtf(s / (float) D + eps);
    __shared__ float buf[512];
    for (int i = threadIdx.x; i < D; i += blockDim.x) buf[i] = src[i] * r * (isq ? qw[i] : kw[i]);
    __syncthreads();
    float* dst = isq ? q + ((size_t) t * H + head) * D : kc + ((size_t) kvh * rows + slot) * D;
    const int half = D / 2;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = buf[i];
        if (rope) {
            const int pp = i < half ? i : i - half;
            const float theta = (float) pos * powf(base, -2.0f * (float) pp / (float) D);
            const float cs = cosf(theta), sn = sinf(theta);
            v = i < half ? buf[i] * cs - buf[i + half] * sn : buf[i - half] * sn + buf[i] * cs;
        }
        dst[i] = v;
    }
    if (!isq)
        for (int i = threadIdx.x; i < D; i += blockDim.x)
            vc[((size_t) kvh * rows + slot) * D + i] = vx[((size_t) t * HKV + kvh) * D + i];
}
/// The routed entries (K picks for one token, entry e = k) grouped per expert on the device: one group per
/// expert used (ascending id), its blob address, its entries in entry order (q36_engine.cu's k_group).
__global__ void k_group(const int32_t* ids, int n_ent, int K_, int n_expert, const uint8_t* blobs,
                        unsigned long long blob, unsigned long long* grp_ptr, int32_t* grp_start, int32_t* n_groups,
                        int32_t* ent_dst, int32_t* ent_tok) {
    __shared__ int cnt[512], off[512];
    for (int x = threadIdx.x; x < n_expert; x += blockDim.x) cnt[x] = 0;
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x) atomicAdd(&cnt[ids[e]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int g = 0, pos = 0;
        for (int x = 0; x < n_expert; ++x) {
            off[x] = pos;
            if (cnt[x] == 0) continue;
            grp_ptr[g] = (unsigned long long) (blobs + (size_t) x * blob);
            grp_start[g] = pos;
            pos += cnt[x];
            ++g;
        }
        grp_start[g] = pos;
        *n_groups = g;
        for (int e = 0; e < n_ent; ++e) {
            const int q = off[ids[e]]++;
            ent_dst[q] = e;
            ent_tok[q] = e / K_;
        }
    }
}

__global__ void k_moe_sum(float* moe, const float* y, const float* w, int k, const float* ys, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = ys[i];
    for (int s = 0; s < k; ++s) acc += w[s] * y[(size_t) s * n + i];
    moe[i] = acc;
}
// the slots path's MoE sum: each selection's row is the sum of its GPU-slot row and its overflow row (the CPU
// pool's or the pinned table's; the side that did not compute a selection leaves its row zero).
__global__ void k_moe_sum2(float* moe, const float* y, const float* yc, const float* w, int k, const float* ys, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = ys[i];
    for (int s = 0; s < k; ++s) acc += w[s] * (y[(size_t) s * n + i] + yc[(size_t) s * n + i]);
    moe[i] = acc;
}
__device__ void k_attn_prep_body(const float* qx, const float* kx, const float* vx, const float* qw, const float* kw,
                            float* q, float* kc, float* vc, int H, int D, int pos, int64_t swa, int ctx, float base,
                            float eps, bool rope) {
    __shared__ float sh[kThreads];
    __shared__ float buf[512];
    const int head = blockIdx.x;
    const bool isq = head < H;
    const int kvh = head - H;
    const float* src = isq ? qx + (size_t) head * D : kx + (size_t) kvh * D;
    const int64_t rows = rope ? swa : ctx;
    const int64_t slot = rope ? (int64_t) pos % swa : (int64_t) pos;
    float s = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) s += src[i] * src[i];
    s = block_sum(s, sh);
    const float r = 1.0f / sqrtf(s / (float) D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x) buf[i] = src[i] * r * (isq ? qw[i] : kw[i]);
    __syncthreads();
    float* dst = isq ? q + (size_t) head * D : kc + ((size_t) kvh * rows + slot) * D;
    const int half = D / 2;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = buf[i];
        if (rope) {
            const int p = i < half ? i : i - half;
            const float theta = (float) pos * powf(base, -2.0f * (float) p / (float) D);
            const float cs = cosf(theta), sn = sinf(theta);
            v = i < half ? buf[i] * cs - buf[i + half] * sn : buf[i - half] * sn + buf[i] * cs;
        }
        dst[i] = v;
    }
    if (!isq)
        for (int i = threadIdx.x; i < D; i += blockDim.x)
            vc[((size_t) kvh * rows + slot) * D + i] = vx[(size_t) kvh * D + i];
}
__global__ void k_attn_prep(const float* qx, const float* kx, const float* vx, const float* qw, const float* kw,
                            float* q, float* kc, float* vc, int H, int D, int pos, int64_t swa, int ctx, float base,
                            float eps, bool rope) {
    k_attn_prep_body(qx, kx, vx, qw, kw, q, kc, vc, H, D, pos, swa, ctx, base, eps, rope);
}
/// graph decode: the position from device memory
__global__ void k_attn_prep_p(const float* qx, const float* kx, const float* vx, const float* qw, const float* kw,
                              float* q, float* kc, float* vc, int H, int D, const int* pos_p, int64_t swa, int ctx,
                              float base, float eps, bool rope) {
    k_attn_prep_body(qx, kx, vx, qw, kw, q, kc, vc, H, D, *pos_p, swa, ctx, base, eps, rope);
}

__device__ void k_attn_body(const float* q, const float* kc, const float* vc, float* out, int H, int HKV, int D, int pos,
                       int64_t swa, int ctx, float scale, bool sliding, bool force_2pass = false) {
    extern __shared__ float sc[];
    __shared__ float sh[kThreads];
    const int h = blockIdx.x, kvh = h / (H / HKV);
    const int64_t rows = sliding ? swa : ctx;
    const int n_pos = sliding ? (pos + 1 < (int) swa ? pos + 1 : (int) swa) : pos + 1;
    const int64_t first = sliding ? (int64_t) pos + 1 - n_pos : 0;
    const float* qh = q + (size_t) h * D;
    // the scores live in dynamic shared memory up to the 48 KB a T4-class block gets; past that (a 32K full-
    // attention layer) the kernel runs two passes and recomputes each score - slower, and it works
    if (!force_2pass && n_pos <= 12000) {
    for (int p = threadIdx.x; p < n_pos; p += blockDim.x) {
        const int64_t gp = first + p, slot = sliding ? gp % swa : gp;
        const float* kp = kc + ((size_t) kvh * rows + slot) * D;
        float s = 0.f;
        for (int i = 0; i < D; ++i) s += qh[i] * kp[i];
        sc[p] = s * scale;
    }
    __syncthreads();
    float mx = -INFINITY;
    for (int p = threadIdx.x; p < n_pos; p += blockDim.x) mx = fmaxf(mx, sc[p]);
    sh[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if ((int) threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]); __syncthreads(); }
    mx = sh[0];
    __syncthreads();
    float sum = 0.f;
    for (int p = threadIdx.x; p < n_pos; p += blockDim.x) { sc[p] = expf(sc[p] - mx); sum += sc[p]; }
    sum = block_sum(sum, sh);
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float acc = 0.f;
        for (int p = 0; p < n_pos; ++p) {
            const int64_t gp = first + p, slot = sliding ? gp % swa : gp;
            acc += sc[p] * vc[((size_t) kvh * rows + slot) * D + i];
        }
        out[(size_t) h * D + i] = acc / sum;
    }
    } else {
        // pass 1: the max and the softmax denominator, scores recomputed from the KV (no shared copy)
        auto score = [&](int p) -> float {
            const int64_t gp = first + p, slot = sliding ? gp % swa : gp;
            const float* kp = kc + ((size_t) kvh * rows + slot) * D;
            float s = 0.f;
            for (int i = 0; i < D; ++i) s += qh[i] * kp[i];
            return s * scale;
        };
        float mx = -INFINITY;
        for (int p = threadIdx.x; p < n_pos; p += blockDim.x) mx = fmaxf(mx, score(p));
        sh[threadIdx.x] = mx;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) { if ((int) threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]); __syncthreads(); }
        mx = sh[0];
        __syncthreads();
        float sum = 0.f;
        for (int p = threadIdx.x; p < n_pos; p += blockDim.x) sum += expf(score(p) - mx);
        sum = block_sum(sum, sh);
        // pass 2: the weighted V sum; each warp of the block takes a slice of D and walks all positions
        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            float acc = 0.f;
            for (int p = 0; p < n_pos; ++p) {
                const int64_t gp = first + p, slot = sliding ? gp % swa : gp;
                acc += expf(score(p) - mx) * vc[((size_t) kvh * rows + slot) * D + i];
            }
            out[(size_t) h * D + i] = acc / sum;
        }
    }
}
__global__ void k_attn(const float* q, const float* kc, const float* vc, float* out, int H, int HKV, int D, int pos,
                       int64_t swa, int ctx, float scale, bool sliding) {
    const int n_vis = sliding ? (pos + 1 < (int) swa ? pos + 1 : (int) swa) : pos + 1;
    k_attn_body(q, kc, vc, out, H, HKV, D, pos, swa, ctx, scale, sliding);
}
/// graph decode: the position from device memory; the dynamic shared memory is the two-pass size (0) so one
/// capture serves every position (the smem path bakes n_vis into the launch and cannot be replayed across pos).
__global__ void k_attn_p(const float* q, const float* kc, const float* vc, float* out, int H, int HKV, int D,
                         const int* pos_p, int64_t swa, int ctx, float scale, bool sliding) {
    k_attn_body(q, kc, vc, out, H, HKV, D, *pos_p, swa, ctx, scale, sliding, true);
}


float* dalloc(size_t n) {
    float* p = nullptr;
    KC(cudaMalloc(&p, n * 4));
    KC(cudaMemset(p, 0, n * 4));
    return p;
}

void* g_q8 = nullptr;            // the q8_1 activation buffer the native mmvq path quantizes x into
std::vector<int> g_ids_host;     // the router's picks, on the host (the native path indexes experts per selection)
constexpr size_t kActBuf = 1 << 16;

/// One matmul, bitwise kolibriref's gemv: quantized weights go through native_mmvq on a q8_1 copy of the
/// activation (one call per selection, expert ids[sel] from the host picks); F32/F16/BF16 through plain k_gemv.
/// `base`/`eb` point at an expert blob table (pinned host, UVA-read); null for the dense GPU tensors.
void gemv(const Dense& w, const float* x, float* y, const int* ids, int n_sel, size_t x_stride, const uint8_t* base,
          size_t eb, cudaStream_t st = 0) {
    const int n_in = (int) w.shape[0], n_out = (int) w.shape[1];
    const uint8_t* W = base ? base : w.d;
    const bool quant = w.type == T_Q8_0 || w.type == T_Q4_K || w.type == T_Q5_K || w.type == T_Q6_K;
    if (quant) {
        const int cols = x_stride ? n_sel : 1;
        strata::kernels::native_quantize_q8_1(x, g_q8, n_in, cols, g_st);
        const size_t qb = strata::kernels::native_q8_1_bytes(n_in, 1);
        for (int sel = 0; sel < n_sel; ++sel) {
            const uint8_t* we = W + (ids ? (size_t) g_ids_host[(size_t) sel] * eb : 0);
            strata::kernels::native_mmvq(w.type, we, (const char*) g_q8 + (x_stride ? (size_t) sel * qb : 0),
                                         y + (size_t) sel * n_out, n_in, n_out, 1, g_st);
        }
        KC(cudaGetLastError());
        return;
    }
    const size_t row_bytes = w.row_bytes;
    switch (w.type) {
    case T_F32: k_gemv<T_F32><<<dim3(n_out, n_sel), kThreads, 0, st>>>(W, row_bytes, eb, ids, n_in, x, x_stride, y, n_out); break;
    case T_F16: k_gemv<T_F16><<<dim3(n_out, n_sel), kThreads, 0, st>>>(W, row_bytes, eb, ids, n_in, x, x_stride, y, n_out); break;
    case T_BF16: k_gemv<T_BF16><<<dim3(n_out, n_sel), kThreads, 0, st>>>(W, row_bytes, eb, ids, n_in, x, x_stride, y, n_out); break;
    default:
        std::fprintf(stderr, "strata-kolibri: no kernel for ggml type %d\n", w.type);
        std::exit(1);
    }
    KC(cudaGetLastError());
}

struct Scratch {
    int ctx = 8192, pos = 0;
    float *x, *h, *y, *q, *k, *v, *att, *router, *eg, *eu, *eh, *ey, *sg, *su, *sh, *sy, *moe, *wts, *logits;
    int* ids;
    std::vector<float*> kc, vc;
    // the CPU expert path's pinned host side: the layer's h, its quantized activations, the FF rows, the down rows
    float* h_host = nullptr; char* act = nullptr; float* ff = nullptr; char* hq = nullptr; float* ey_h = nullptr;
    std::vector<int32_t> exp_cnt, exp_ord;   // prefill pool: the per-expert grouping
    // --expert-slots: the GPU-slot experts' rows and the overflow rows (CPU pool or pinned table)
    float *eg2 = nullptr, *eu2 = nullptr, *eh2 = nullptr, *ey2 = nullptr, *ey_cpu = nullptr;
    float* yc = nullptr;            // the slots path's overflow rows: ey_cpu (pool) or ey (pinned table)
    std::vector<int> slot_ids, ovf_ids;
    // --grouped: the per-layer grouped-expert call's device side
    unsigned long long* grp_ptr = nullptr; int32_t *grp_start = nullptr, *n_groups = nullptr;
    int32_t *ent_dst = nullptr, *ent_tok = nullptr; char* xq8 = nullptr; char* scratch = nullptr; float* gout = nullptr;
    int* pos_d = nullptr; int* tok_d = nullptr;   // graph decode: position and token id on the device
    // --prefill: the chunk's rows (tmax tokens at once)
    static constexpr int kTmax = 512;
    int* toks_d = nullptr;
    float *x_m = nullptr, *h_m = nullptr, *y_m = nullptr, *q_m = nullptr, *k_m = nullptr, *v_m = nullptr;
    float *att_m = nullptr, *router_m = nullptr, *moe_m = nullptr, *sy_m = nullptr;
    float *sg_m = nullptr, *su_m = nullptr, *sh_m = nullptr;
    float *gout_m = nullptr, *eg_m = nullptr, *eu_m = nullptr, *eh_m = nullptr;
    int *ids_m = nullptr, *grp_start_m = nullptr, *n_groups_m = nullptr, *ent_dst_m = nullptr, *ent_tok_m = nullptr;
    float* wts_m = nullptr;
    unsigned long long* grp_ptr_m = nullptr;
    char *xq8_m = nullptr, *scratch_m = nullptr;
    __half *x16 = nullptr, *q16 = nullptr, *att16 = nullptr, *k16 = nullptr, *v16 = nullptr, *p16 = nullptr;
    float* att_s = nullptr;
};


std::string L(int64_t l, const char* s) { return "blk." + std::to_string(l) + "." + s; }

/// One token through the engine at position s.pos.  Dense matmuls on the GPU tensors; the routed experts read
/// their [gate|up|down] blob from the pinned host table over UVA (the plain path: exactness first, the native
/// grouped path with VRAM slots arrives with the placement work).  Bitwise-equal to strata-kolibriref by
/// construction: the same kernels, the same order, the same weights.  `layers_host` (optional, 50 x n_embd)
/// receives each layer's output for this position, kolibriref's --dump-layers record.

/// One decoder step for layer l.  graph=true: everything device-side (position and token id read from
/// device pointers, grouped experts, two-pass attention - capturable as a CUDA graph); graph=false: the
/// plain path (value position, host ids for the mmvq loop, layer dumps).  st = 0 (legacy stream) keeps the
/// plain launches byte-identical; the graph path runs on g_st.
void layer_step(const Engine& e, Scratch& s, int64_t l, bool graph, cudaStream_t st, float* layers_host) {
    const auto& g = e.g;
    const int N = (int) g.n_embd, K = (int) e.top_k, FF = (int) g.n_ff, FS = (int) g.n_ff_shexp;
    const int H = (int) g.n_head, HKV = (int) g.n_head_kv, D = (int) g.head_dim;
    const int B = 256;
    (void) layers_host;

        const bool sliding = strata::core::is_swa_layer(g, l);
        const LayerBlobs& b = e.blobs[(size_t) l];
        k_rms<<<1, kThreads, 0, st>>>(s.x, e.f32.at(L(l, "attn_norm.weight")), s.h, N, e.eps);
        gemv(e.dense.at(L(l, "attn_q.weight")), s.h, s.q, nullptr, 1, 0, nullptr, 0, st);
        gemv(e.dense.at(L(l, "attn_k.weight")), s.h, s.k, nullptr, 1, 0, nullptr, 0, st);
        gemv(e.dense.at(L(l, "attn_v.weight")), s.h, s.v, nullptr, 1, 0, nullptr, 0, st);
        if (graph)
            k_attn_prep_p<<<H + HKV, kThreads, 0, st>>>(s.q, s.k, s.v, e.f32.at(L(l, "attn_q_norm.weight")),
                                           e.f32.at(L(l, "attn_k_norm.weight")), s.q, s.kc[l], s.vc[l], H, D, s.pos_d,
                                           g.n_swa, s.ctx, e.rope_base, e.eps, sliding);
        else
            k_attn_prep<<<H + HKV, kThreads, 0, st>>>(s.q, s.k, s.v, e.f32.at(L(l, "attn_q_norm.weight")),
                                           e.f32.at(L(l, "attn_k_norm.weight")), s.q, s.kc[l], s.vc[l], H, D, s.pos,
                                           g.n_swa, s.ctx, e.rope_base, e.eps, sliding);
        const int n_vis = (int) (s.pos + 1 < (int64_t) (sliding ? g.n_swa : s.ctx) ? s.pos + 1
                                                                            : (int64_t) (sliding ? g.n_swa : s.ctx));
        if (graph)
            k_attn_p<<<H, kThreads, 0, st>>>(
                s.q, s.kc[l], s.vc[l], s.att, H, HKV, D, s.pos_d, g.n_swa, s.ctx, 1.0f / sqrtf((float) D), sliding);
        else
            k_attn<<<H, kThreads, n_vis <= 12000 ? (size_t) n_vis * sizeof(float) : 0, st>>>(
                s.q, s.kc[l], s.vc[l], s.att, H, HKV, D, s.pos, g.n_swa, s.ctx, 1.0f / sqrtf((float) D), sliding);
        gemv(e.dense.at(L(l, "attn_output.weight")), s.att, s.y, nullptr, 1, 0, nullptr, 0, st);
        k_rms<<<1, kThreads, 0, st>>>(s.y, e.f32.at(L(l, "post_attention_norm.weight")), s.y, N, e.eps);
        k_add<<<(N + B - 1) / B, B, 0, st>>>(s.x, s.y, N);
        k_rms<<<1, kThreads, 0, st>>>(s.x, e.f32.at(L(l, "ffn_norm.weight")), s.h, N, e.eps);
        gemv(e.dense.at(L(l, "ffn_gate_inp.weight")), s.h, s.router, nullptr, 1, 0, nullptr, 0, st);
        strata::kernels::native_router_sigmoid_logit_add(s.router, e.f32.at(L(l, "exp_probs_b.bias")), s.ids, s.wts, 1,
                                                         g_st);
        if (!graph) {
            g_ids_host.resize((size_t) K);
            KC(cudaMemcpy(g_ids_host.data(), s.ids, K * sizeof(int), cudaMemcpyDeviceToHost));
        }
        Dense gu; gu.type = b.L.gu_type; gu.shape = {g.n_embd, g.n_ff, g.n_expert}; gu.row_bytes = b.L.gu_row;
        Dense dn; dn.type = b.L.d_type; dn.shape = {g.n_ff, g.n_embd, g.n_expert}; dn.row_bytes = b.L.d_row;
        // --grouped: one native_expert_grouped launch per layer reads the blobs from the pinned table over UVA
        // (the per-selection mmvq loop's 18 launches per layer become one; the grouped kernel's rows are ggml's
        // own arithmetic, native_grouped_parity).  Unsupported layers (Q6_K down) keep the mmvq loop.
        if (e.grouped > 0 && b.native_ok) {
            k_group<<<1, 1024, 0, st>>>(s.ids, K, K, (int) g.n_expert, b.table, (unsigned long long) b.stride,
                                 s.grp_ptr, s.grp_start, s.n_groups, s.ent_dst, s.ent_tok);
            strata::kernels::native_quantize_q8_1(s.h, s.xq8, (int) g.n_embd, 1, g_st);
            strata::kernels::native_expert_grouped(b.L, s.grp_ptr, s.grp_start, s.n_groups, s.ent_dst, s.ent_tok,
                                                   std::min<int64_t>(K, g.n_expert) + 1, K, s.xq8, s.scratch,
                                                   s.gout, g_st);
        } else
        // --expert-slots: the hot experts live in VRAM (the GPU expert cache); the rest overflow to the CPU
        // pool (--cpu-threads) or to the pinned table over UVA.  Bitwise: the same native_mmvq kernel reads the
        // same bytes wherever the blob sits; the side that skips a selection leaves its row zero (k_moe_sum2).
        if (!(e.grouped > 0 && b.native_ok)) {
        if (e.slots > 0) {
            const int S = e.slots;
            s.slot_ids.resize((size_t) K); s.ovf_ids.resize((size_t) K);
            for (int k = 0; k < K; ++k) {
                const int id = g_ids_host[(size_t) k];
                s.slot_ids[k] = id < S ? id : S;                                        // dev zero slot
                s.ovf_ids[k] = id < S ? (e.cpu_threads > 0 ? id : (int) g.n_expert) : id;  // pinned zero slot
            }
            g_ids_host = s.slot_ids;
            gemv(gu, s.h, s.eg2, s.ids, K, 0, b.dev, b.stride);
            gemv(gu, s.h, s.eg2, s.ids, K, 0, b.dev, b.stride);
            gemv(gu, s.h, s.eu2, s.ids, K, 0, b.dev + b.L.up_off, b.stride);
            k_swiglu<<<(K * FF + B - 1) / B, B, 0, st>>>(s.eg2, s.eu2, s.eh2, K * FF);
            gemv(dn, s.eh2, s.ey2, s.ids, K, FF, b.dev + b.L.down_off, b.stride);
            KC(cudaStreamSynchronize(g_st));   // the slot gemv run on g_st; k_moe_sum2 below reads ey2 on the default stream
            float* yc;
            if (e.cpu_threads > 0 && b.native_ok) {
                g_ids_host.assign(g_ids_host.size(), 0);   // restored below from the device ids
                KC(cudaMemcpy(g_ids_host.data(), s.ids, K * sizeof(int), cudaMemcpyDeviceToHost));
                KC(cudaMemcpy(s.h_host, s.h, N * 4, cudaMemcpyDeviceToHost));
                strata::kernels::cpu::native_quant_act(b.fmt, s.h_host, s.act);
                std::memset(s.ey_h, 0, (size_t) K * N * 4);
                const int T = std::min(e.cpu_threads, K);
                e.pool.run([&](int shard) {
                    for (int sel = shard; sel < K; sel += T) {
                        if (g_ids_host[sel] < S) continue;   // in-slot: the GPU computed it
                        const uint8_t* blob = b.table + (size_t) g_ids_host[sel] * b.stride;
                        float* ff = s.ff + (size_t) sel * FF;
                        strata::kernels::cpu::native_gu_rows(b.fmt, blob, (const void* const*) &s.act, 1, &ff, 0, FF);
                        char* hq = s.hq + (size_t) sel * b.fmt.h_bytes;
                        strata::kernels::cpu::native_quant_h(b.fmt, ff, hq);
                        float* ey = s.ey_h + (size_t) sel * N;
                        strata::kernels::cpu::native_down_rows(b.fmt, blob, (const void* const*) &hq, 1, &ey, 0, N);  // adds fmt.down_off itself
                    }
                }, T);
                KC(cudaMemcpy(s.ey_cpu, s.ey_h, (size_t) K * N * 4, cudaMemcpyHostToDevice));
                yc = s.ey_cpu;
            } else {
                g_ids_host = s.ovf_ids;
                gemv(gu, s.h, s.eg, s.ids, K, 0, b.table, b.stride);
                gemv(gu, s.h, s.eu, s.ids, K, 0, b.table + b.L.up_off, b.stride);
                k_swiglu<<<(K * FF + B - 1) / B, B, 0, st>>>(s.eg, s.eu, s.eh, K * FF);
                gemv(dn, s.eh, s.ey, s.ids, K, FF, b.table + b.L.down_off, b.stride);
                yc = s.ey;
            }
            KC(cudaStreamSynchronize(g_st));   // the shared-expert gemv ran on g_st; the sum below reads sy on the default stream
            s.yc = yc;   // the MoE sum runs in the shared code, after the shared-expert gemv
        } else if (e.cpu_threads > 0 && b.native_ok) {
            KC(cudaMemcpy(s.h_host, s.h, N * 4, cudaMemcpyDeviceToHost));
            strata::kernels::cpu::native_quant_act(b.fmt, s.h_host, s.act);
            const int T = std::min(e.cpu_threads, K);
            e.pool.run([&](int shard) {
                for (int sel = shard; sel < K; sel += T) {
                    const uint8_t* blob = b.table + (size_t) g_ids_host[sel] * b.stride;
                    float* ff = s.ff + (size_t) sel * FF;
                    strata::kernels::cpu::native_gu_rows(b.fmt, blob, (const void* const*) &s.act, 1, &ff, 0, FF);
                    char* hq = s.hq + (size_t) sel * b.fmt.h_bytes;
                    strata::kernels::cpu::native_quant_h(b.fmt, ff, hq);
                    float* ey = s.ey_h + (size_t) sel * N;
                    strata::kernels::cpu::native_down_rows(b.fmt, blob, (const void* const*) &hq, 1, &ey, 0, N);  // adds fmt.down_off itself
                }
            }, T);
            KC(cudaMemcpy(s.ey, s.ey_h, (size_t) K * N * 4, cudaMemcpyHostToDevice));
        } else {
        gemv(gu, s.h, s.eg, s.ids, K, 0, b.table, b.stride);
        gemv(gu, s.h, s.eu, s.ids, K, 0, b.table + b.L.up_off, b.stride);
        k_swiglu<<<(K * FF + B - 1) / B, B, 0, st>>>(s.eg, s.eu, s.eh, K * FF);
        gemv(dn, s.eh, s.ey, s.ids, K, FF, b.table + b.L.down_off, b.stride);
        }
        }
        gemv(e.dense.at(L(l, "ffn_gate_shexp.weight")), s.h, s.sg, nullptr, 1, 0, nullptr, 0, st);
        gemv(e.dense.at(L(l, "ffn_up_shexp.weight")), s.h, s.su, nullptr, 1, 0, nullptr, 0, st);
        k_swiglu<<<(FS + B - 1) / B, B, 0, st>>>(s.sg, s.su, s.sh, FS);
        gemv(e.dense.at(L(l, "ffn_down_shexp.weight")), s.sh, s.sy, nullptr, 1, 0, nullptr, 0, st);
        if (e.grouped > 0 && b.native_ok) k_moe_sum<<<(N + B - 1) / B, B, 0, st>>>(s.moe, s.gout, s.wts, K, s.sy, N);
        else if (e.slots > 0) k_moe_sum2<<<(N + B - 1) / B, B, 0, st>>>(s.moe, s.ey2, s.yc, s.wts, K, s.sy, N);
        else k_moe_sum<<<(N + B - 1) / B, B, 0, st>>>(s.moe, s.ey, s.wts, K, s.sy, N);
        k_rms<<<1, kThreads, 0, st>>>(s.moe, e.f32.at(L(l, "post_ffw_norm.weight")), s.moe, N, e.eps);
        k_add<<<(N + B - 1) / B, B, 0, st>>>(s.x, s.moe, N);
        if (layers_host && !graph) KC(cudaMemcpy(layers_host + (size_t) l * N, s.x, N * 4, cudaMemcpyDeviceToHost));
        KC(cudaGetLastError());
    
}


/// Chunked prompt processing (--prefill): T rows at once.  Dense projections run as GEMMs (FP16 activations, the
/// weights dequantized to FP16 in the Gemm scratch - what llama.cpp's batched CUDA path does), the attention
/// scores and P.V as batched FP16 GEMMs with the causal (and window) softmax in between, the routed experts as
/// one grouped launch per layer (T*K entries), the Q6_K-down layers through the mmvq loop with host ids.
/// Bitwise vs the per-token plain path is not expected (FP16 GEMM scores); the gates are the KL parity and the
/// serve==tokens agreement.
void forward_multi(Engine& e, Scratch& s, int T, int64_t p0) {
    const auto& g = e.g;
    const int N = (int) g.n_embd, K = (int) e.top_k, FF = (int) g.n_ff, FS = (int) g.n_ff_shexp;
    const int H = (int) g.n_head, HKV = (int) g.n_head_kv, D = (int) g.head_dim;
    const int B = 256;
    const int NK = T * K;
    cublasHandle_t blas = e.blas;
    const float one = 1.0f, zero = 0.0f;
    const Dense& emb = e.dense.at("token_embd.weight");
    switch (emb.type) {
    case T_Q8_0: k_row_multi<T_Q8_0><<<dim3(T, 4), 128>>>(emb.d, emb.row_bytes, s.toks_d, N, s.x_m); break;
    case T_Q4_K: k_row_multi<T_Q4_K><<<dim3(T, 4), 128>>>(emb.d, emb.row_bytes, s.toks_d, N, s.x_m); break;
    case T_Q6_K: k_row_multi<T_Q6_K><<<dim3(T, 4), 128>>>(emb.d, emb.row_bytes, s.toks_d, N, s.x_m); break;
    case T_F16: k_row_multi<T_F16><<<dim3(T, 4), 128>>>(emb.d, emb.row_bytes, s.toks_d, N, s.x_m); break;
    default: std::fprintf(stderr, "strata-kolibri: token_embd type %d\n", emb.type); std::exit(1);
    }
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const bool sliding = strata::core::is_swa_layer(g, l);
        const LayerBlobs& b = e.blobs[(size_t) l];
        k_rms_rows<<<T, kThreads>>>(s.x_m, e.f32.at(L(l, "attn_norm.weight")), s.h_m, N, e.eps);
        strata::kernels::native_quantize_q8_1(s.h_m, s.xq8_m, N * T, 1, g_st);   // T contiguous rows = one column (q36)
        k_to_f16<<<grid1((int64_t) T * N), kThreads>>>((const float*) s.h_m, s.x16, (int64_t) T * N);
        {
            const Dense& wq = e.dense.at(L(l, "attn_q.weight"));
            e.gemm.native((const uint16_t*) s.x16, wq.type, wq.d, s.q_m, T, H * D, N);
            const Dense& wk = e.dense.at(L(l, "attn_k.weight"));
            e.gemm.native((const uint16_t*) s.x16, wk.type, wk.d, s.k_m, T, HKV * D, N);
            const Dense& wv = e.dense.at(L(l, "attn_v.weight"));
            e.gemm.native((const uint16_t*) s.x16, wv.type, wv.d, s.v_m, T, HKV * D, N);
        }
        k_attn_prep_multi<<<dim3(H + HKV, T), kThreads>>>(s.q_m, s.k_m, s.v_m, e.f32.at(L(l, "attn_q_norm.weight")),
                                                          e.f32.at(L(l, "attn_k_norm.weight")), s.q_m, s.kc[l],
                                                          s.vc[l], H, HKV, D, p0, g.n_swa, s.ctx, e.rope_base,
                                                          e.eps, sliding);
        // the chunk reads positions 0..p0+T-1 (full) or the ring's last n_vis (sliding)
        const int P = sliding ? (int) std::min<int64_t>(p0 + T, g.n_swa) : (int) (p0 + T);
        const int64_t first = sliding ? p0 + T - P : 0;
        if (sliding)
            k_kv_f16_swa<<<grid1((int64_t) HKV * P * D), kThreads>>>(s.kc[l], s.vc[l], s.k16, s.v16, g.n_swa, P,
                                                                     first, D, HKV);
        else
            k_kv_f16_full<<<grid1((int64_t) HKV * P * D), kThreads>>>(s.kc[l], s.vc[l], s.k16, s.v16, s.ctx, P, D,
                                                                      HKV);
        k_to_f16<<<grid1((int64_t) T * H * D), kThreads>>>(s.q_m, s.q16, (int64_t) T * H * D);
        const int ldp = (P + 7) / 8 * 8;
        const int G = H / HKV;
        const size_t kAttScoreFloats = 12u << 20;
        const int tq = (int) std::max<int64_t>(1, std::min<int64_t>(T, (int64_t) (kAttScoreFloats / ((size_t) H * ldp))));
        for (int t0 = 0; t0 < T; t0 += tq) {
            const int nt = std::min(tq, T - t0);
            for (int kvh = 0; kvh < HKV; ++kvh) {
                const __half* Kh = s.k16 + (size_t) kvh * P * D;
                const __half* Qh = s.q16 + ((size_t) t0 * H + kvh * G) * D;
                float* Sh = s.att_s + (size_t) kvh * G * nt * ldp;
                if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, P, nt, D, &one, Kh, CUDA_R_16F, D, 0,
                                               Qh, CUDA_R_16F, H * D, D, &zero, Sh, CUDA_R_32F, ldp,
                                               (long long) nt * ldp, G, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) !=
                    CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "strata-kolibri: scores GEMM failed\n"); std::exit(1); }
            }
            k_attn_softmax16<<<dim3(nt, H), kThreads>>>(s.att_s, s.p16, ldp, nt, P, (int) p0 + t0,
                                                        1.0f / sqrtf((float) D), sliding ? g.n_swa : 0);
            for (int kvh = 0; kvh < HKV; ++kvh) {
                const __half* Vh = s.v16 + (size_t) kvh * P * D;
                const __half* Ph = s.p16 + (size_t) kvh * G * nt * ldp;
                float* Oh = s.att_m + ((size_t) t0 * H + kvh * G) * D;
                if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, D, nt, P, &one, Vh, CUDA_R_16F, D, 0,
                                               Ph, CUDA_R_16F, ldp, (long long) nt * ldp, &zero, Oh, CUDA_R_32F,
                                               H * D, D, G, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) !=
                    CUBLAS_STATUS_SUCCESS) {
                    std::fprintf(stderr, "strata-kolibri: PV GEMM failed (t0=%d nt=%d P=%d ldp=%d ctx=%d)\n", t0, nt,
                                 P, ldp, s.ctx);
                    std::exit(1);
                }
            }
        }
        k_to_f16<<<grid1((int64_t) T * H * D), kThreads>>>(s.att_m, s.att16, (int64_t) T * H * D);
        {
            const Dense& wo = e.dense.at(L(l, "attn_output.weight"));
            e.gemm.native((const uint16_t*) s.att16, wo.type, wo.d, s.y_m, T, N, H * D);
        }
        k_rms_rows<<<T, kThreads>>>(s.y_m, e.f32.at(L(l, "post_attention_norm.weight")), s.y_m, N, e.eps);
        k_add_rows<<<grid1((int64_t) T * N), kThreads>>>(s.x_m, s.y_m, (int64_t) T * N);
        k_rms_rows<<<T, kThreads>>>(s.x_m, e.f32.at(L(l, "ffn_norm.weight")), s.h_m, N, e.eps);
        {
            const Dense& wr = e.dense.at(L(l, "ffn_gate_inp.weight"));
            k_gemv<T_F32><<<dim3(g.n_expert, T), kThreads>>>(wr.d, wr.row_bytes, 0, nullptr, N, s.h_m, (size_t) N,
                                                             s.router_m, (int) g.n_expert);
            strata::kernels::native_router_sigmoid_logit_add(s.router_m, e.f32.at(L(l, "exp_probs_b.bias")),
                                                             s.ids_m, s.wts_m, T, g_st);
        }
        strata::kernels::native_quantize_q8_1(s.h_m, s.xq8_m, N * T, 1, g_st);   // T contiguous rows = one column (q36)
        k_to_f16<<<grid1((int64_t) T * N), kThreads>>>((const float*) s.h_m, s.x16, (int64_t) T * N);
        static const bool pool_all = std::getenv("STRATA_KOLIBRI_POOL_ALL") != nullptr;   // A/B: the pool for every layer
        if (b.native_ok && !(pool_all && e.cpu_threads > 0)) {
            k_group<<<1, 1024>>>(s.ids_m, NK, K, (int) g.n_expert, b.table, (unsigned long long) b.stride,
                                 s.grp_ptr_m, s.grp_start_m, s.n_groups_m, s.ent_dst_m, s.ent_tok_m);
            strata::kernels::native_expert_grouped(b.L, s.grp_ptr_m, s.grp_start_m, s.n_groups_m, s.ent_dst_m,
                                                   s.ent_tok_m, std::min<int64_t>(NK, g.n_expert) + 1, NK, s.xq8_m,
                                                   s.scratch_m, s.gout_m, g_st);
        } else {
            g_ids_host.resize((size_t) NK);
            KC(cudaMemcpy(g_ids_host.data(), s.ids_m, NK * sizeof(int), cudaMemcpyDeviceToHost));
            if (e.cpu_threads > 0) {   // pool_all reaches here for the native_ok layers too (the ids are copied above)   // the CPU pool: ggml's own expert arithmetic over the pinned blobs, the
                                       // chunk's NK entries split over the pool's threads (bandwidth-bound,
                                       // ~10x the launch-bound mmvq loop on this box)
                KC(cudaMemcpy(s.h_host, s.h_m, (size_t) T * N * 4, cudaMemcpyDeviceToHost));
                for (int t = 0; t < T; ++t)
                    strata::kernels::cpu::native_quant_act(b.fmt, s.h_host + (size_t) t * N,
                                                           s.act + (size_t) t * b.fmt.act_bytes);
                // group the chunk's entries by expert: one blob read per expert (the box's RAM bandwidth is the
                // floor; per-entry reads re-stream 2.2 MB per row), all of an expert's rows in one call
                std::vector<int32_t>& cnt = s.exp_cnt;
                std::fill(cnt.begin(), cnt.begin() + g.n_expert + 1, 0);
                for (int sel = 0; sel < NK; ++sel) cnt[g_ids_host[(size_t) sel] + 1]++;
                for (int x = 0; x < (int) g.n_expert; ++x) cnt[x + 1] += cnt[x];
                std::vector<int32_t>& ord = s.exp_ord;
                {
                    std::vector<int32_t> cur(cnt.begin(), cnt.begin() + g.n_expert + 1);
                    for (int sel = 0; sel < NK; ++sel) ord[cur[g_ids_host[(size_t) sel]]++] = sel;
                }
                const int NS = std::min((int) g.n_expert, e.cpu_threads);
                e.pool.run([&](int shard) {
                    const void* acts[64]; float* ffs[64]; const void* hqs[64]; float* eys[64];
                    for (int x = shard; x < (int) g.n_expert; x += NS) {
                        const int lo = cnt[x], hi = cnt[x + 1], m = hi - lo;
                        if (m <= 0) continue;
                        const uint8_t* blob = b.table + (size_t) x * b.stride;
                        for (int a = 0; a < m; a += 64) {
                            const int mm = std::min(64, m - a);
                            for (int k = 0; k < mm; ++k) {
                                const int sel = ord[(size_t) lo + a + k];
                                acts[k] = s.act + (size_t) (sel / K) * b.fmt.act_bytes;
                                ffs[k] = s.ff + (size_t) sel * FF;
                            }
                            strata::kernels::cpu::native_gu_rows(b.fmt, blob, acts, mm, ffs, 0, FF);
                            for (int k = 0; k < mm; ++k) {
                                const int sel = ord[(size_t) lo + a + k];
                                char* hqp = s.hq + (size_t) sel * b.fmt.h_bytes;
                                strata::kernels::cpu::native_quant_h(b.fmt, ffs[k], hqp);
                                hqs[k] = hqp;
                                eys[k] = s.ey_h + (size_t) sel * N;
                            }
                            strata::kernels::cpu::native_down_rows(b.fmt, blob, hqs, mm, eys, 0, N);
                        }
                    }
                }, NS);
                KC(cudaMemcpy(s.gout_m, s.ey_h, (size_t) NK * N * 4, cudaMemcpyHostToDevice));
            } else
            for (int t = 0; t < T; ++t) {
                strata::kernels::native_quantize_q8_1(s.h_m + (size_t) t * N, g_q8, N, 1, g_st);
                for (int k = 0; k < K; ++k) {
                    const int sel = t * K + k;
                    const uint8_t* blob = b.table + (size_t) g_ids_host[(size_t) sel] * b.stride;
                    strata::kernels::native_mmvq(b.L.gu_type, blob, (const char*) g_q8, s.eg_m + (size_t) sel * FF,
                                                 N, FF, 1, g_st);
                    strata::kernels::native_mmvq(b.L.gu_type, blob + b.L.up_off, (const char*) g_q8,
                                                 s.eu_m + (size_t) sel * FF, N, FF, 1, g_st);
                }
            }
            k_swiglu_rows<<<grid1((int64_t) NK * FF), kThreads>>>(s.eg_m, s.eu_m, s.eh_m, (int64_t) NK * FF);
            for (int sel = 0; sel < NK; ++sel) {
                strata::kernels::native_quantize_q8_1(s.eh_m + (size_t) sel * FF, g_q8, FF, 1, g_st);
                const uint8_t* blob = b.table + (size_t) g_ids_host[(size_t) sel] * b.stride + b.L.down_off;
                strata::kernels::native_mmvq(b.L.d_type, blob, (const char*) g_q8, s.gout_m + (size_t) sel * N,
                                             FF, N, 1, g_st);
            }
        }
        {
            const Dense& wg = e.dense.at(L(l, "ffn_gate_shexp.weight"));
            e.gemm.native((const uint16_t*) s.x16, wg.type, wg.d, s.sg_m, T, FS, N);
            const Dense& wu = e.dense.at(L(l, "ffn_up_shexp.weight"));
            e.gemm.native((const uint16_t*) s.x16, wu.type, wu.d, s.su_m, T, FS, N);
            k_swiglu_rows<<<grid1((int64_t) T * FS), kThreads>>>(s.sg_m, s.su_m, s.sh_m, (int64_t) T * FS);
            k_to_f16<<<grid1((int64_t) T * FS), kThreads>>>(s.sh_m, s.x16, (int64_t) T * FS);
            const Dense& wd = e.dense.at(L(l, "ffn_down_shexp.weight"));
            e.gemm.native((const uint16_t*) s.x16, wd.type, wd.d, s.sy_m, T, N, FS);
        }
        k_moe_sum_rows<<<dim3(T, (N + B - 1) / B), B>>>(s.moe_m, s.gout_m, s.wts_m, s.sy_m, K, N);
        k_rms_rows<<<T, kThreads>>>(s.moe_m, e.f32.at(L(l, "post_ffw_norm.weight")), s.moe_m, N, e.eps);
        k_add_rows<<<grid1((int64_t) T * N), kThreads>>>(s.x_m, s.moe_m, (int64_t) T * N);
        KC(cudaGetLastError());
    }
    k_rms<<<1, kThreads>>>(s.x_m + (size_t) (T - 1) * N, e.f32.at("output_norm.weight"), s.h, N, e.eps);
    gemv(e.dense.at("output.weight"), s.h, s.logits, nullptr, 1, 0, nullptr, 0);
    KC(cudaDeviceSynchronize());
}

void forward(const Engine& e, Scratch& s, int tok, float* layers_host) {
    const auto& g = e.g;
    const int N = (int) g.n_embd, K = (int) e.top_k, FF = (int) g.n_ff, FS = (int) g.n_ff_shexp;
    const int H = (int) g.n_head, HKV = (int) g.n_head_kv, D = (int) g.head_dim;
    const int B = 256;
    const Dense& emb = e.dense.at("token_embd.weight");
        if (e.graphs > 0 && layers_host == nullptr) {
        cudaMemcpyAsync(s.tok_d, &tok, 4, cudaMemcpyHostToDevice, g_st);
        switch (emb.type) {
        case T_Q8_0: k_row_p<T_Q8_0><<<4, 128, 0, g_st>>>(emb.d, emb.row_bytes, s.tok_d, N, s.x); break;
        case T_Q4_K: k_row_p<T_Q4_K><<<4, 128, 0, g_st>>>(emb.d, emb.row_bytes, s.tok_d, N, s.x); break;
        case T_Q6_K: k_row_p<T_Q6_K><<<4, 128, 0, g_st>>>(emb.d, emb.row_bytes, s.tok_d, N, s.x); break;
        case T_F16: k_row_p<T_F16><<<4, 128, 0, g_st>>>(emb.d, emb.row_bytes, s.tok_d, N, s.x); break;
        default: std::fprintf(stderr, "strata-kolibri: token_embd type %d\n", emb.type); std::exit(1);
        }
        for (int64_t l = 0; l < g.n_layers; ++l)
            if (e.gexec[(size_t) l] != 0) cudaGraphLaunch((cudaGraphExec_t) e.gexec[(size_t) l], g_st);
            else layer_step(e, s, l, false, 0, nullptr);
        k_pos_add<<<1, 1, 0, g_st>>>(s.pos_d);
    } else {
        switch (emb.type) {
        case T_Q8_0: k_row<T_Q8_0><<<4, 128>>>(emb.d, emb.row_bytes, tok, N, s.x); break;
        case T_Q4_K: k_row<T_Q4_K><<<4, 128>>>(emb.d, emb.row_bytes, tok, N, s.x); break;
        case T_Q6_K: k_row<T_Q6_K><<<4, 128>>>(emb.d, emb.row_bytes, tok, N, s.x); break;
        case T_F16: k_row<T_F16><<<4, 128>>>(emb.d, emb.row_bytes, tok, N, s.x); break;
        default: std::fprintf(stderr, "strata-kolibri: token_embd type %d\n", emb.type); std::exit(1);
        }
        for (int64_t l = 0; l < g.n_layers; ++l) layer_step(e, s, l, false, 0, layers_host);
    }
    k_rms<<<1, kThreads>>>(s.x, e.f32.at("output_norm.weight"), s.h, N, e.eps);
    gemv(e.dense.at("output.weight"), s.h, s.logits, nullptr, 1, 0, nullptr, 0);
    KC(cudaDeviceSynchronize());
    ++s.pos;
}

void init_scratch(const Engine& e, Scratch& s) {
    const auto& g = e.g;
    const int N = (int) g.n_embd, FF = (int) g.n_ff, K = (int) e.top_k;
    s.x = dalloc(N); s.h = dalloc(N); s.y = dalloc(N); s.moe = dalloc(N); s.sy = dalloc(N);
    s.q = dalloc(g.n_head * g.head_dim); s.k = dalloc(g.n_head_kv * g.head_dim); s.v = dalloc(g.n_head_kv * g.head_dim);
    s.att = dalloc(g.n_head * g.head_dim);
    s.router = dalloc(g.n_expert);
    s.eg = dalloc((size_t) K * FF); s.eu = dalloc((size_t) K * FF); s.eh = dalloc((size_t) K * FF); s.ey = dalloc((size_t) K * N);
    s.sg = dalloc(g.n_ff_shexp); s.su = dalloc(g.n_ff_shexp); s.sh = dalloc(g.n_ff_shexp);
    s.wts = dalloc(K);
    KC(cudaMalloc((void**) &s.pos_d, 4));
    KC(cudaMalloc((void**) &s.tok_d, 4));
    if (e.prefill > 0) {   // the chunk's rows: tmax tokens, tmax*K expert entries, the FP16 attention images
        const int T = Scratch::kTmax, NK = T * K;
        s.exp_cnt.resize((size_t) g.n_expert + 1);
        s.exp_ord.resize((size_t) NK);
        const size_t hd = (size_t) g.n_head * g.head_dim, hkd = (size_t) g.n_head_kv * g.head_dim;
        KC(cudaMalloc((void**) &s.toks_d, T * 4));
        KC(cudaMalloc((void**) &s.x_m, (size_t) T * N * 4));
        KC(cudaMalloc((void**) &s.h_m, (size_t) T * N * 4));
        KC(cudaMalloc((void**) &s.y_m, (size_t) T * N * 4));
        KC(cudaMalloc((void**) &s.q_m, (size_t) T * hd * 4));
        KC(cudaMalloc((void**) &s.k_m, (size_t) T * hkd * 4));
        KC(cudaMalloc((void**) &s.v_m, (size_t) T * hkd * 4));
        KC(cudaMalloc((void**) &s.att_m, (size_t) T * hd * 4));
        KC(cudaMalloc((void**) &s.router_m, (size_t) T * g.n_expert * 4));
        KC(cudaMalloc((void**) &s.moe_m, (size_t) T * N * 4));
        KC(cudaMalloc((void**) &s.sy_m, (size_t) T * N * 4));
        KC(cudaMalloc((void**) &s.sg_m, (size_t) T * g.n_ff_shexp * 4));
        KC(cudaMalloc((void**) &s.su_m, (size_t) T * g.n_ff_shexp * 4));
        KC(cudaMalloc((void**) &s.sh_m, (size_t) T * g.n_ff_shexp * 4));
        KC(cudaMalloc((void**) &s.gout_m, (size_t) NK * N * 4));
        KC(cudaMalloc((void**) &s.eg_m, (size_t) NK * g.n_ff * 4));
        KC(cudaMalloc((void**) &s.eu_m, (size_t) NK * g.n_ff * 4));
        KC(cudaMalloc((void**) &s.eh_m, (size_t) NK * g.n_ff * 4));
        KC(cudaMalloc((void**) &s.ids_m, NK * 4));
        KC(cudaMalloc((void**) &s.wts_m, NK * 4));
        KC(cudaMalloc((void**) &s.grp_ptr_m, ((size_t) std::min<int64_t>(NK, g.n_expert) + 1) * sizeof(unsigned long long)));
        KC(cudaMalloc((void**) &s.grp_start_m, ((size_t) std::min<int64_t>(NK, g.n_expert) + 1) * 4));
        KC(cudaMalloc((void**) &s.n_groups_m, 4));
        KC(cudaMalloc((void**) &s.ent_dst_m, NK * 4));
        KC(cudaMalloc((void**) &s.ent_tok_m, NK * 4));
        KC(cudaMalloc((void**) &s.xq8_m, strata::kernels::native_q8_1_bytes((int) g.n_embd, 1) * (size_t) T));
        KC(cudaMalloc((void**) &s.scratch_m, strata::kernels::native_expert_scratch_bytes(NK, g.n_ff)));
        const size_t f16 = 2;
        KC(cudaMalloc((void**) &s.x16, (size_t) T * std::max<int64_t>(N, std::max<int64_t>(g.n_ff, (int64_t) hd)) * f16));
        KC(cudaMalloc((void**) &s.q16, (size_t) T * hd * f16));
        KC(cudaMalloc((void**) &s.att16, (size_t) T * hd * f16));
        KC(cudaMalloc((void**) &s.k16, (size_t) g.n_head_kv * s.ctx * g.head_dim * f16));
        KC(cudaMalloc((void**) &s.v16, (size_t) g.n_head_kv * s.ctx * g.head_dim * f16));
        // the scores slice is H * tq * ldp; the runtime tq uses the chunk's ldp (>= 512), so the worst case is
        // the smallest ldp: tq = 12M / (H * 512), and tq * ldp <= 12M / H always
        const size_t tq_w = std::max<int64_t>(1, std::min<int64_t>(T, (int64_t) ((12u << 20) / ((size_t) g.n_head * 512))));
        KC(cudaMalloc((void**) &s.att_s, (size_t) g.n_head * tq_w * 512 * 4 + (size_t) g.n_head * 8 * 4));
        KC(cudaMalloc((void**) &s.p16, ((size_t) g.n_head * tq_w * 512 + (size_t) g.n_head * 8) * f16));
    }
    if (e.grouped > 0) {   // one grouped launch per layer: K entries, K groups at most, q8_1 acts for one token
        const int64_t cap_g = std::min<int64_t>(K, g.n_expert) + 1, cap_e = K;
        KC(cudaMalloc((void**) &s.grp_ptr, cap_g * sizeof(unsigned long long)));
        KC(cudaMalloc((void**) &s.grp_start, cap_g * sizeof(int32_t)));
        KC(cudaMalloc((void**) &s.n_groups, sizeof(int32_t)));
        KC(cudaMalloc((void**) &s.ent_dst, cap_e * sizeof(int32_t)));
        KC(cudaMalloc((void**) &s.ent_tok, cap_e * sizeof(int32_t)));
        KC(cudaMalloc((void**) &s.xq8, strata::kernels::native_q8_1_bytes((int) g.n_embd, 1)));
        KC(cudaMalloc((void**) &s.scratch, strata::kernels::native_expert_scratch_bytes(cap_e, g.n_ff)));
        KC(cudaMalloc((void**) &s.gout, (size_t) K * N * 4));
    }
    if (e.slots > 0) {   // the GPU expert cache's rows: the in-slot side and the overflow side
        KC(cudaMalloc((void**) &s.eg2, (size_t) K * FF * 4));
        KC(cudaMalloc((void**) &s.eu2, (size_t) K * FF * 4));
        KC(cudaMalloc((void**) &s.eh2, (size_t) K * FF * 4));
        KC(cudaMalloc((void**) &s.ey2, (size_t) K * N * 4));
        KC(cudaMalloc((void**) &s.ey_cpu, (size_t) K * N * 4));
    }
    if (e.cpu_threads > 0) {   // the CPU expert path's pinned host side (one layer's working set); the quantized
                               // activations are sized from the layers' fmt (kolibri's 2560-as-Q8_K is 23,360 B)
        size_t act_b = 0, h_b = 0;
        for (const auto& b : e.blobs) if (b.native_ok) { act_b = std::max(act_b, b.fmt.act_bytes); h_b = std::max(h_b, b.fmt.h_bytes); }
        const int64_t rows = e.prefill > 0 ? Scratch::kTmax : 1;          // prefill: a whole chunk on the host
        const int64_t ents = e.prefill > 0 ? (int64_t) Scratch::kTmax * K : K;
        KC(cudaHostAlloc((void**) &s.h_host, (size_t) rows * N * 4, cudaHostAllocDefault));
        KC(cudaHostAlloc((void**) &s.act, (size_t) rows * act_b, cudaHostAllocDefault));
        KC(cudaHostAlloc((void**) &s.ff, (size_t) ents * FF * 4, cudaHostAllocDefault));
        KC(cudaHostAlloc((void**) &s.hq, (size_t) ents * h_b, cudaHostAllocDefault));
        KC(cudaHostAlloc((void**) &s.ey_h, (size_t) ents * N * 4, cudaHostAllocDefault));
    }
    s.logits = dalloc(g.n_vocab);
    KC(cudaMalloc(&s.ids, K * sizeof(int)));
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const int64_t rows = strata::core::is_swa_layer(g, l) ? g.n_swa : s.ctx;
        s.kc.push_back(dalloc((size_t) g.n_head_kv * rows * g.head_dim));
        s.vc.push_back(dalloc((size_t) g.n_head_kv * rows * g.head_dim));
    }
}

/// Per-layer CUDA graphs (--grouped + graphs): a grouped layer's decode step is fully device-side (position and
/// token id from device pointers, expert blobs from the pinned table over UVA, two-pass attention so the launch
/// carries no position-dependent shared memory) - one capture per layer, replayed per token.  The Q6_K-down
/// layers keep the mmvq loop (the blob pointer is host-computed) and are launched normally.  Warmup runs each
/// layer once (module load, allocations) before capture; the KV rows the warmup touches are pos 0, which the
/// first real token overwrites.
void capture_graphs(Engine& e, Scratch& s) {
    const int zero = 0;
    cudaMemcpy(s.pos_d, &zero, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(s.tok_d, &zero, 4, cudaMemcpyHostToDevice);
    e.gexec.assign((size_t) e.g.n_layers, 0);
    for (int64_t l = 0; l < e.g.n_layers; ++l)
        if (e.grouped > 0 && e.blobs[(size_t) l].native_ok)
            layer_step(e, s, l, true, g_st, nullptr);
    cudaDeviceSynchronize();
    for (int64_t l = 0; l < e.g.n_layers; ++l) {
        if (!(e.grouped > 0 && e.blobs[(size_t) l].native_ok)) continue;
        cudaGraph_t gr = nullptr; cudaGraphExec_t ex = nullptr;
        KC(cudaStreamBeginCapture(g_st, cudaStreamCaptureModeRelaxed));
        layer_step(e, s, l, true, g_st, nullptr);
        KC(cudaStreamEndCapture(g_st, &gr));
        KC(cudaGraphInstantiate(&ex, gr, 0));
        cudaGraphDestroy(gr);
        e.gexec[(size_t) l] = (unsigned long long) ex;
    }
    cudaDeviceSynchronize();
    s.pos = 0;
    cudaMemcpy(s.pos_d, &zero, 4, cudaMemcpyHostToDevice);
    std::printf("kolibri: %d / %d layers captured as CUDA graphs (the rest use the mmvq loop)\n",
                (int) (std::count(e.gexec.begin(), e.gexec.end(), 0ull) == 0 ? e.g.n_layers
                    : e.g.n_layers - std::count(e.gexec.begin(), e.gexec.end(), 0ull)), (int) e.g.n_layers);
    std::fflush(stdout);
}

/// Drop the held state: position 0 and empty KV caches (the serve protocol's fresh-request path).
void reset_state(Engine& e, Scratch& s) {
    s.pos = 0;
    for (size_t l = 0; l < s.kc.size(); ++l) {
        const int64_t rows = strata::core::is_swa_layer(e.g, (int64_t) l) ? e.g.n_swa : s.ctx;
        const size_t bytes = (size_t) e.g.n_head_kv * rows * e.g.head_dim * 4;
        KC(cudaMemset(s.kc[l], 0, bytes));
        KC(cudaMemset(s.vc[l], 0, bytes));
    }
}

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// --------------------------------------------------------------------------- sampling (q36's, verbatim semantics)
struct Sampling {
    float temperature = 0.f, top_p = 1.f, min_p = 0.f, repeat = 1.f, freq = 0.f, present = 0.f;
    int top_k = 20, last_n = 0;
    unsigned long long seed = 0;
    bool needs_logits() const { return temperature > 0.f || repeat != 1.f || freq != 0.f || present != 0.f; }
};

int sample(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist, std::mt19937_64& rng) {
    if (s.last_n != 0 && (s.repeat != 1.f || s.freq != 0.f || s.present != 0.f)) {
        std::map<int, int> cnt;
        const size_t from = s.last_n < 0 || (size_t) s.last_n >= hist.size() ? 0 : hist.size() - (size_t) s.last_n;
        for (size_t i = from; i < hist.size(); ++i) ++cnt[hist[i]];
        for (const auto& [id, c] : cnt) {
            float& v = lg[(size_t) id];
            if (s.repeat != 1.f) v = v > 0 ? v / s.repeat : v * s.repeat;
            v -= (float) c * s.freq + s.present;
        }
    }
    if (s.temperature <= 0.f) return (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
    const int kk = std::max(1, std::min(s.top_k > 0 ? s.top_k : 64, (int) lg.size()));
    std::vector<int> idx(lg.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    idx.resize((size_t) kk);
    std::vector<double> p(idx.size());
    const double mx = lg[(size_t) idx[0]];
    double sum = 0;
    for (size_t i = 0; i < idx.size(); ++i) sum += (p[i] = std::exp((lg[(size_t) idx[i]] - mx) / s.temperature));
    size_t keep = idx.size();
    double cum = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (p[i] / sum < s.min_p * (p[0] / sum)) { keep = i; break; }
        cum += p[i] / sum;
        if (cum >= s.top_p) { keep = i + 1; break; }
    }
    keep = std::max<size_t>(1, keep);
    double tot = 0;
    for (size_t i = 0; i < keep; ++i) tot += p[i];
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(rng);
    for (size_t i = 0; i < keep; ++i) {
        if ((r -= p[i]) <= 0) return idx[i];
    }
    return idx[keep - 1];
}

std::vector<int32_t> parse_ids(const char* s) {
    std::vector<int32_t> v;
    while (*s) {
        while (*s == ',' || *s == ' ' || *s == '\n') ++s;
        if (!*s) break;
        char* e = nullptr;
        const long x = std::strtol(s, &e, 10);
        if (e == s) break;
        v.push_back((int32_t) x);
        s = e;
    }
    return v;
}

bool is_eos(int t) { return t == 127906; }   // kolibri1: im_end

/// The serve protocol (serve/server.py's engine lines, q36's shape): READY, then per request GEN <max_new>
/// [key=value ...] <ids>; PP lines as the prompt is consumed; T <id> per token; DONE at the end.  STOP cancels
/// the running request.  A request whose ids extend the held prefix continues it (the serve `live` logic).
int serve(Engine& e, Scratch& s) {
    std::printf("READY %d stop\n", s.ctx);
    std::fflush(stdout);
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
    std::atomic<bool> stop{false};
    std::thread reader([&] {
        std::string l;
        while (std::getline(std::cin, l)) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            if (l == "STOP") { stop.store(true); continue; }
            std::lock_guard<std::mutex> lk(mu);
            lines.push_back(l);
            cv.notify_one();
        }
        std::lock_guard<std::mutex> lk(mu);
        eof = true;
        cv.notify_one();
    });
    reader.detach();
    std::vector<int32_t> live;   // the tokens whose state the engine holds
    std::vector<float> lg((size_t) e.g.n_vocab);
    for (;;) {
        std::string line;
        {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&] { return eof || !lines.empty(); });
            if (lines.empty()) return 0;
            line = lines.front();
            lines.pop_front();
        }
        if (line.empty()) continue;
        if (line.rfind("GEN ", 0) != 0) {
            std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...>\n");
            std::fflush(stdout);
            continue;
        }
        char* p = nullptr;
        const long long max_new = std::strtoll(line.c_str() + 4, &p, 10);
        Sampling smp;
        for (;;) {
            while (*p == ' ') ++p;
            const char* st0 = p;
            while (*p && *p != ' ') ++p;
            const std::string tk(st0, (size_t) (p - st0));
            const size_t eq = tk.find('=');
            if (tk.empty() || eq == std::string::npos) { p = const_cast<char*>(st0); break; }
            const std::string key = tk.substr(0, eq);
            const char* v = tk.c_str() + eq + 1;
            if (key == "temperature") smp.temperature = std::strtof(v, nullptr);
            else if (key == "top_p") smp.top_p = std::strtof(v, nullptr);
            else if (key == "top_k") smp.top_k = std::atoi(v);
            else if (key == "min_p") smp.min_p = std::strtof(v, nullptr);
            else if (key == "penalty_last_n") smp.last_n = std::atoi(v);
            else if (key == "penalty_repeat") smp.repeat = std::strtof(v, nullptr);
            else if (key == "penalty_freq") smp.freq = std::strtof(v, nullptr);
            else if (key == "penalty_present") smp.present = std::strtof(v, nullptr);
            else if (key == "seed") smp.seed = std::strtoull(v, nullptr, 10);
        }
        const std::vector<int32_t> ids = parse_ids(p);
        if (max_new < 1 || ids.empty()) {
            std::printf("ERR bad request: max_new or the ids\n");
            std::fflush(stdout);
            continue;
        }
        if ((int64_t) ids.size() + max_new > s.ctx) {
            std::printf("ERR prompt (%zu tokens) + max_new (%lld) exceeds the context (%d)\n", ids.size(), max_new, s.ctx);
            std::fflush(stdout);
            continue;
        }
        bool bad = false;
        for (int32_t t : ids) bad |= t < 0 || t >= (int32_t) e.g.n_vocab;
        if (bad) { std::printf("ERR a token id is outside the vocabulary\n"); std::fflush(stdout); continue; }
        stop.store(false);
        // the held state continues when the request extends it (the KV rings cannot step back)
        size_t reuse = 0;
        if (live.size() < ids.size() && std::equal(live.begin(), live.end(), ids.begin())) reuse = live.size();
        else {
            reset_state(e, s);
            live.clear();
        }
        const auto t0 = Clock::now();
        const int64_t fresh = (int64_t) (ids.size() - reuse);
        if (e.prefill > 0) {   // chunked prefill: 512 rows per pass, progress per chunk
            int64_t i = reuse;
            const int64_t n = fresh - 1 - reuse;   // the prompt minus its last token
            while (i - reuse < n) {
                const int T = (int) std::min<int64_t>(Scratch::kTmax, n - (i - reuse));
                KC(cudaMemcpy(s.toks_d, ids.data() + i, (size_t) T * 4, cudaMemcpyHostToDevice));
                forward_multi(e, s, T, s.pos);
                s.pos += T;
                i += T;
                std::printf("PP %lld %lld %.0f %.1f\n", (long long) (i - reuse), (long long) fresh, ms_since(t0),
                            (double) (i - reuse) / std::max(1e-9, ms_since(t0) / 1000.0));
                std::fflush(stdout);
            }
        } else
        for (int64_t t = (int64_t) reuse + 1; t < fresh; ++t) {   // the prompt minus its last token
            forward(e, s, ids[t - 1], nullptr);
            std::printf("PP %lld %lld %.0f %.1f\n", (long long) (t - reuse), (long long) fresh, ms_since(t0),
                        (double) (t - reuse) / std::max(1e-9, ms_since(t0) / 1000.0));
            std::fflush(stdout);
        }
        if (e.graphs > 0) KC(cudaMemcpy(s.pos_d, &s.pos, 4, cudaMemcpyHostToDevice));
        live.insert(live.end(), ids.begin() + (long) reuse, ids.end());
        const double prompt_ms = ms_since(t0);
        std::mt19937_64 rng(smp.seed);
        std::vector<int32_t> hist = ids;
        const auto d0 = Clock::now();
        long long produced = 0;
        const char* finish = "length";
        int32_t tok = ids.back();   // the last prompt token: the first decode step runs it
        for (;;) {
            forward(e, s, tok, nullptr);
            KC(cudaMemcpy(lg.data(), s.logits, lg.size() * 4, cudaMemcpyDeviceToHost));
            const int32_t nx = smp.needs_logits() ? sample(lg, smp, hist, rng)
                                                  : (int32_t) (std::max_element(lg.begin(), lg.end()) - lg.begin());
            std::printf("T %d\n", nx);
            std::fflush(stdout);
            ++produced;
            hist.push_back(nx);
            live.push_back(nx);
            if (is_eos(nx)) { finish = "stop"; break; }
            if (produced >= max_new) break;
            if (stop.load()) { finish = "cancel"; break; }
            tok = nx;
        }
        const double decode_ms = ms_since(d0);
        std::printf("DONE %lld %zu %.1f %.1f %s 0 0 %zu 0 0 0 0 0.0 %lld 0\n", produced, ids.size(), prompt_ms,
                    decode_ms, finish, reuse, (long long) fresh);
        std::fflush(stdout);
    }
}

/// Load: geometry + layout from the header; dense tensors to the GPU; routed experts repacked to pinned blob
/// tables.  Split files: every tensor lives in one shard; the shard's mmap pointer is the source.
bool load(const std::string& path, Engine& e, std::string& err) {
    const auto t0 = std::chrono::steady_clock::now();
    const strata::GgufModel gm = strata::GgufModel::open(path);
    if (!strata::core::geometry_from_gguf(gm.meta(), e.g, e.top_k, err)) return false;
    if (e.g.family != strata::ModelFamily::Kolibri1) { err = "strata-kolibri runs kolibri1 files only"; return false; }
    size_t checked = 0;
    if (!strata::core::check_gguf_layout(gm, e.g, err, &checked)) return false;
    const std::string p = "kolibri1.";
    if (const strata::MetaValue* v = gm.meta().get(p + "attention.layer_norm_rms_epsilon")) e.eps = (float) v->num();
    if (const strata::MetaValue* v = gm.meta().get(p + "rope.freq_base")) e.rope_base = (float) v->num();

    // pass 1: the routed experts, layer by layer, repacked into pinned [gate|up|down] blob tables
    e.blobs.resize((size_t) e.g.n_layers);
    size_t expert_bytes = 0;
    for (int64_t l = 0; l < e.g.n_layers; ++l) {
        const strata::TensorInfo *gate = nullptr, *up = nullptr, *down = nullptr;
        const strata::GgufFile* gf = nullptr;
        for (size_t s = 0; s < gm.size(); ++s)
            for (const strata::TensorInfo& ti : gm.shard(s).tensors()) {
                if (ti.name == L(l, "ffn_gate_exps.weight")) { gate = &ti; gf = &gm.shard(s); }
                if (ti.name == L(l, "ffn_up_exps.weight")) up = &ti;
                if (ti.name == L(l, "ffn_down_exps.weight")) down = &ti;
            }
        if (!gate || !up || !down) { err = "layer " + std::to_string(l) + ": expert tensors missing"; return false; }
        LayerBlobs& b = e.blobs[(size_t) l];
        b.L = strata::kernels::native_expert_layout((int) gate->type, (int) down->type, e.g.n_embd, e.g.n_ff);
        b.native_ok = strata::kernels::native_expert_supported((int) gate->type, (int) down->type, e.g.n_embd, e.g.n_ff);
        {
            std::string ferr;
            // the CPU pool's fmt is needed for every layer (the pool covers the Q6_K-down layers the GPU
            // grouped kernel cannot); && would short-circuit on the GPU check and leave fmt unfilled
            const bool fmt_ok = strata::kernels::cpu::native_fmt((int) gate->type, (int) down->type,
                                                                 e.g.n_embd, e.g.n_ff, b.fmt, ferr);
            b.native_ok = b.native_ok && fmt_ok;
        }
        const size_t sb = lcm(lcm((size_t) 144, (size_t) 210), (size_t) 34);   // Q4_K / Q6_K / Q8_0 super-blocks
        b.stride = ((b.L.bytes + sb - 1) / sb) * sb;
        KC(cudaHostAlloc((void**) &b.table, b.stride * (size_t) (e.g.n_expert + 1), cudaHostAllocDefault));
        std::memset(b.table + b.stride * (size_t) e.g.n_expert, 0, b.stride);  // the slots path's zero slot
        const size_t gu_rows = (size_t) e.g.n_ff, d_rows = (size_t) e.g.n_embd;
        const size_t gu_eb = gu_rows * b.L.gu_row, d_eb = d_rows * b.L.d_row;
        const uint8_t* gs = gf->tensor_data(*gate);
        // the up / down tensors may live in other shards: find their file pointers
        const uint8_t* us = nullptr;
        const uint8_t* ds = nullptr;
        for (size_t s = 0; s < gm.size(); ++s) {
            for (const strata::TensorInfo& ti : gm.shard(s).tensors()) {
                if (ti.name == L(l, "ffn_up_exps.weight")) us = gm.shard(s).tensor_data(ti);
                if (ti.name == L(l, "ffn_down_exps.weight")) ds = gm.shard(s).tensor_data(ti);
            }
        }
        for (int64_t ex = 0; ex < e.g.n_expert; ++ex) {
            uint8_t* blob = b.table + (size_t) ex * b.stride;
            std::memcpy(blob, gs + (size_t) ex * gu_eb, gu_rows * b.L.gu_row);
            std::memcpy(blob + b.L.up_off, us + (size_t) ex * gu_eb, gu_rows * b.L.gu_row);
            std::memcpy(blob + b.L.down_off, ds + (size_t) ex * d_eb, d_rows * b.L.d_row);
        }
        expert_bytes += b.stride * (size_t) e.g.n_expert;
    }

    // pass 2: the dense tensors to the GPU (MMVQ types as is; 1-D vectors as F32)
    size_t dense_bytes = 0;
    for (size_t s = 0; s < gm.size(); ++s) {
        const strata::GgufFile& f = gm.shard(s);
        for (const strata::TensorInfo& ti : f.tensors()) {
            if (ti.shape.size() == 3) continue;   // the routed experts: repacked above
            const uint64_t nb = strata::tensor_payload_bytes(ti);
            const uint8_t* src = f.tensor_data(ti);
            if (ti.shape.size() == 1) {
                if (ti.type != 0) { err = ti.name + ": a 1-D tensor of type " + ti.type_name(); return false; }
                float* d = nullptr;
                KC(cudaMalloc(&d, nb));
                KC(cudaMemcpy(d, src, nb, cudaMemcpyHostToDevice));
                e.f32[ti.name] = d;
                dense_bytes += nb;
                continue;
            }
            Dense w;
            w.type = (int) ti.type;
            for (uint64_t d : ti.shape) w.shape.push_back((int64_t) d);
            w.row_bytes = row_bytes(w.type, w.shape[0]);
            uint8_t* d = nullptr;
            KC(cudaMalloc(&d, nb));
            KC(cudaMemcpy(d, src, nb, cudaMemcpyHostToDevice));
            w.d = d;
            e.dense[ti.name] = w;
            dense_bytes += nb;
        }
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    int native_layers = 0;
    for (const LayerBlobs& b : e.blobs) native_layers += b.native_ok ? 1 : 0;
    std::fprintf(stderr,
                 "strata-kolibri: %s: %zu tensors (layout checked); experts %zu x %zu blobs repacked, %.2f GiB "
                 "pinned; dense %.2f GiB on the GPU; native expert kernels for %zu / %zu layers; %.1f s\n",
                 path.c_str(), checked, e.blobs.size(), (size_t) e.g.n_expert, (double) expert_bytes / (1 << 30),
                 (double) dense_bytes / (1 << 30), (size_t) native_layers, e.blobs.size(), secs);
    return true;
}

}  // namespace

int strata::kolibri::run(int argc, char** argv) {
    std::string model, tokens, dump_layers, dump_logits, dump_full, force;
    bool load_only = false, serve_mode = false;
    int n_gen = 8, ctx = 8192, cpu_threads = 0, slots = 0, grouped = 0, no_graph = 0, prefill = 0;
    double expert_vram = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "--model" || a == "--native") && i + 1 < argc) model = argv[++i];
        else if (a == "--tokens" && i + 1 < argc) tokens = argv[++i];
        else if (a == "--n" && i + 1 < argc) n_gen = std::atoi(argv[++i]);
        else if (a == "--ctx" && i + 1 < argc) ctx = std::atoi(argv[++i]);
        else if (a == "--dump-layers" && i + 1 < argc) dump_layers = argv[++i];
        else if (a == "--dump-logits" && i + 1 < argc) dump_logits = argv[++i];
        else if (a == "--dump-full-logits" && i + 1 < argc) dump_full = argv[++i];
        else if (a == "--force" && i + 1 < argc) force = argv[++i];
        else if (a == "--load-only") load_only = true;
        else if (a == "--serve") serve_mode = true;
        else if (a == "--cpu-threads" && i + 1 < argc) cpu_threads = std::atoi(argv[++i]);
        else if (a == "--expert-slots" && i + 1 < argc) slots = std::atoi(argv[++i]);
        else if (a == "--expert-vram" && i + 1 < argc) expert_vram = std::atoi(argv[++i]);
        else if (a == "--grouped") grouped = 1;
        else if (a == "--no-graph") no_graph = 1;
        else if (a == "--prefill") prefill = 1;
    }
    if (model.empty()) {
        std::fprintf(stderr, "strata-kolibri: the Kolibri 1 engine path (plan T7); --model GGUF [--load-only] for now\n");
        return 2;
    }
    Engine e;
    std::string err;
    KC(cudaStreamCreate(&g_st));
    KC(cudaMalloc(&g_q8, strata::kernels::native_q8_1_bytes((int) kActBuf, 1)));
    if (!load(model, e, err)) {
        std::fprintf(stderr, "strata-kolibri: %s\n", err.c_str());
        return 1;
    }
    e.cpu_threads = cpu_threads;
    e.grouped = grouped;
    // CUDA-graph decode rides the grouped path (device-side expert pointers); the slots/pool paths read host
    // ids per layer and are not capturable.  --no-graph keeps the plain launches.
    e.graphs = grouped && !no_graph && slots == 0 && cpu_threads == 0 ? 1 : 0;
    e.prefill = prefill;
    if (e.prefill > 0) {
        if (cublasCreate(&e.blas) != CUBLAS_STATUS_SUCCESS || cublasSetStream(e.blas, g_st) != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "strata-kolibri: cuBLAS init failed\n");
            return 1;
        }
        const int64_t big = std::max(e.g.n_head * e.g.head_dim, e.g.n_embd) * (int64_t) e.g.n_embd;
        if (!e.gemm.init(g_st, big, err)) {
            std::fprintf(stderr, "strata-kolibri: prompt GEMM init failed: %s\n", err.c_str());
            return 1;
        }
    }
    if (expert_vram > 0 && slots == 0)
        slots = (int) (expert_vram * (1 << 30) / (e.blobs[0].stride * (double) e.g.n_layers));
    e.slots = std::min(slots, (int) e.g.n_expert);
    if (e.slots > 0) {   // the GPU expert cache: slots blobs + 1 zero slot per layer, copied up front
        for (auto& b : e.blobs) {
            KC(cudaMalloc((void**) &b.dev, b.stride * (size_t) (e.slots + 1)));
            KC(cudaMemcpy(b.dev, b.table, b.stride * (size_t) e.slots, cudaMemcpyHostToDevice));
            KC(cudaMemset(b.dev + b.stride * (size_t) e.slots, 0, b.stride));
        }
        std::fprintf(stderr, "strata-kolibri: expert cache: %d slots/layer = %.2f GiB VRAM\n",
                     e.slots, (double) e.slots * e.blobs[0].stride * e.g.n_layers / (1 << 30));
    }
    if (e.cpu_threads > 1) e.pool.start(e.cpu_threads);
    if (load_only) return 0;
    if (serve_mode) {
        Scratch s;
        s.ctx = ctx;
        init_scratch(e, s);
        if (e.graphs > 0 && e.gexec.empty()) capture_graphs(e, s);
        return serve(e, s);
    }
    if (tokens.empty()) {
        std::fprintf(stderr, "strata-kolibri: --tokens ID,ID,... (the layer driver is plain; serve arrives in plan T8)\n");
        return 2;
    }
    Scratch s;
    s.ctx = ctx;
    init_scratch(e, s);
    if (e.graphs > 0 && e.gexec.empty()) capture_graphs(e, s);
    std::vector<float> lg((size_t) e.g.n_vocab);
    std::vector<float> layers((size_t) e.g.n_layers * (size_t) e.g.n_embd);
    std::FILE* fl = dump_layers.empty() ? nullptr : std::fopen(dump_layers.c_str(), "wb");
    std::FILE* fg = dump_logits.empty() ? nullptr : std::fopen(dump_logits.c_str(), "wb");
    std::FILE* ff = dump_full.empty() ? nullptr : std::fopen(dump_full.c_str(), "wb");
    auto parse_ids = [](const std::string& s) {
        std::vector<int> v;
        for (size_t i = 0; i < s.size(); ++i) {
            while (i < s.size() && (s[i] == ',' || s[i] == ' ' || s[i] == '\n')) ++i;
            if (i >= s.size()) break;
            v.push_back(std::atoi(s.c_str() + i));
            while (i < s.size() && s[i] != ',' && s[i] != ' ' && s[i] != '\n') ++i;
        }
        return v;
    };
    std::vector<int> ids = parse_ids(tokens);
    std::vector<int> forced = parse_ids(force);
    int tok = ids[0];
    if (e.prefill > 0 && fl == nullptr) {   // chunked prefill: the whole prompt minus its last token (layer dumps stay per-token)
        int64_t i = 0;
        const int64_t n = (int64_t) ids.size() - 1;
        const auto tp0 = Clock::now();
        while (i < n) {
            const int T = (int) std::min<int64_t>(Scratch::kTmax, n - i);
            KC(cudaMemcpy(s.toks_d, ids.data() + i, (size_t) T * 4, cudaMemcpyHostToDevice));
            forward_multi(e, s, T, s.pos);
            s.pos += T;
            i += T;
            std::printf("PFchunk %lld/%lld %.0fms\n", (long long) i, (long long) n, ms_since(tp0));
            std::fflush(stdout);
        }
        tok = ids.back();
    } else
    for (size_t t = 1; t < ids.size(); ++t) {
        forward(e, s, ids[t - 1], fl ? layers.data() : nullptr);
        if (fl) {
            for (int64_t l = 0; l < e.g.n_layers; ++l)
                std::fwrite(layers.data() + (size_t) l * e.g.n_embd, 4, e.g.n_embd, fl);
        }
        tok = ids[t];
    }
    if (e.graphs > 0) KC(cudaMemcpy(s.pos_d, &s.pos, 4, cudaMemcpyHostToDevice));
    // the last prompt token and every step after it: run, argmax, feed back (a --force sequence replaces the
    // greedy picks - kl.py's teacher-forced KL path, kolibriref's semantics)
    const int n_steps = (int) forced.empty() ? n_gen : (int) forced.size() + 1;
    std::vector<int> gen;
    std::printf("gen:");
    for (int i = 0; i < n_steps; ++i) {
        forward(e, s, tok, fl ? layers.data() : nullptr);
        if (fl)
            for (int64_t l = 0; l < e.g.n_layers; ++l)
                std::fwrite(layers.data() + (size_t) l * e.g.n_embd, 4, e.g.n_embd, fl);
        KC(cudaMemcpy(lg.data(), s.logits, lg.size() * 4, cudaMemcpyDeviceToHost));
        int a = 0, c = -1;
        for (int j = 1; j < (int) lg.size(); ++j) if (lg[j] > lg[a]) a = j;
        for (int j = 0; j < (int) lg.size(); ++j) if (j != a && (c < 0 || lg[j] > lg[c])) c = j;
        std::printf(" %d", a);
        if (fg) { const float rec[4] = {(float) a, lg[a], (float) c, lg[c]}; std::fwrite(rec, 4, 4, fg); }
        if (ff) std::fwrite(lg.data(), 4, lg.size(), ff);
        gen.push_back(a);
        tok = i + 1 == n_steps ? a : (i < (int) forced.size() ? forced[i] : a);
    }
    std::printf("\n");
    if (fl) std::fclose(fl);
    if (fg) std::fclose(fg);
    if (ff) std::fclose(ff);
    return 0;
}
