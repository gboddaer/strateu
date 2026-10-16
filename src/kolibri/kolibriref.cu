// src/kolibri/kolibriref.cu - the reference forward pass for Kolibri 1 (GGUF architecture kolibri1), exactness
// first (plan docs/superpowers/plans/2026-10-16-kolibri1.md T6, on the src/qwen35/q36ref.cu template): one GPU, the
// whole model resident (Q4_K_M: 44 GiB - the routed experts stay in RAM here, this path keeps them on the GPU), one
// token at a time (the prompt too), fp32 activations, every weight read in its GGUF form.  Two kernel sets:
// --kernels plain (this file's own kernels, the reference) and --kernels strata (default, Strata's native MMVQ for
// the quantized matmuls and the sigmoid_logit_add router).  No expert cache, no graphs: checked against llama.cpp
// (per-layer hidden states, greedy tokens, teacher-forced KL: tools/kolibri/).
//
// The forward pass, as llama.cpp's src/models/kolibri1.cpp builds it (docs/KOLIBRI1.md):
//   x = token_embd[tok]
//   per layer l < 50:  h = rms(x) * attn_norm
//       q = W_q h (48 x 128), k = W_k h (4 x 128), v = W_v h (4 x 128)
//       q, k = rms(.) * q_norm/k_norm per head; RoPE (base 10000, all 128 dims) on the SLIDING layers only -
//            the full-attention layers (l % 5 == 4) carry no positional encoding (NoPE)
//       attention: GQA 48 / 4; sliding layers (window 513) attend to the last 513 positions (ring buffer),
//            full layers to every position; y = W_o softmax(q k^T / sqrt(128)) v
//       x += rms(y) * attn_post_norm                       (the sandwich norm after attention)
//       h2 = rms(x) * ffn_norm
//       router: select top-6 of 384 on `logits + exp_probs_b` (ties -> lower index), weights = sigmoid(logits)
//            of the picks, unbiased, NOT renormalised
//       moe = sum_e w_e W_down_e (silu(W_gate_e h2) * W_up_e h2) + W_down_s (silu(W_gate_s h2) * W_up_s h2)
//            (the shared expert is UNGATED)
//       x += rms(moe) * ffn_post_norm                      (the sandwich norm after MoE)
//   logits = W_output (rms(x) * output_norm)
//
// Usage: strata-kolibriref --model Kolibri-1-Q4_K_M.gguf --tokens 127904,1646,... [--n 64] [--ctx 8192]
//                          [--dump-layers FILE] [--dump-logits FILE] [--act q8|f32] [--kernels strata|plain]
//                          [--force IDS] [--dump-full-logits F]
//   prints the greedy ids ("gen: id id ..."); --dump-layers writes every processed position's 50 layer outputs
//   (f32, [pos][layer][2560]); --dump-logits each greedy step's top-2 (id, logit); --force feeds the given
//   continuation instead of the greedy ids and --dump-full-logits writes every step's logits (f32, [step][vocab]).
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/family_layout.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_router.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

#define CK(x)                                                                                                          \
    do {                                                                                                               \
        cudaError_t e_ = (x);                                                                                          \
        if (e_ != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "kolibriref: %s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

enum : int { T_F32 = 0, T_F16 = 1, T_Q8_0 = 8, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14, T_BF16 = 30 };

__device__ __forceinline__ float h2f(uint16_t h) {
    const uint32_t s = (uint32_t) (h >> 15) << 31, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e == 0) {
        const float v = ldexpf((float) m, -24);
        return s ? -v : v;
    }
    const uint32_t f = e == 31 ? (s | 0x7f800000u | (m << 13)) : (s | ((e + 112) << 23) | (m << 13));
    return __uint_as_float(f);
}
__device__ __forceinline__ uint16_t ld16(const uint8_t* p) { return (uint16_t) (p[0] | (p[1] << 8)); }

__device__ __forceinline__ void scale_min_k4(int j, const uint8_t* q, int& d, int& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

/// The 32 values of chunk `c` of one row (`row` = the row's first byte).  ggml-quants.c's dequantize_row_*.
template <int T> __device__ void deq32(const uint8_t* row, int c, float* out) {
    if constexpr (T == T_F32) {
        const float* r = (const float*) row + 32 * c;
        for (int l = 0; l < 32; ++l) out[l] = r[l];
    } else if constexpr (T == T_F16) {
        const uint8_t* r = row + 64 * c;
        for (int l = 0; l < 32; ++l) out[l] = h2f(ld16(r + 2 * l));
    } else if constexpr (T == T_BF16) {
        const uint8_t* r = row + 64 * c;
        for (int l = 0; l < 32; ++l) out[l] = __uint_as_float((uint32_t) ld16(r + 2 * l) << 16);
    } else if constexpr (T == T_Q8_0) {
        const uint8_t* b = row + 34 * c;
        const float d = h2f(ld16(b));
        const int8_t* q = (const int8_t*) (b + 2);
        for (int l = 0; l < 32; ++l) out[l] = d * (float) q[l];
    } else if constexpr (T == T_Q4_K) {
        const uint8_t* b = row + 144 * (c >> 3);
        const int j = c & 7;
        const float d = h2f(ld16(b)), dmin = h2f(ld16(b + 2));
        int sc, m;
        scale_min_k4(j, b + 4, sc, m);
        const float d1 = d * (float) sc, m1 = dmin * (float) m;
        const uint8_t* qs = b + 16 + 32 * (j >> 1);
        for (int l = 0; l < 32; ++l) out[l] = d1 * (float) ((j & 1) ? (qs[l] >> 4) : (qs[l] & 0xF)) - m1;
    } else if constexpr (T == T_Q5_K) {
        const uint8_t* b = row + 176 * (c >> 3);
        const int j = c & 7;
        const float d = h2f(ld16(b)), dmin = h2f(ld16(b + 2));
        int sc, m;
        scale_min_k4(j, b + 4, sc, m);
        const float d1 = d * (float) sc, m1 = dmin * (float) m;
        const uint8_t* qh = b + 16;
        const uint8_t* qs = b + 48 + 32 * (j >> 1);
        for (int l = 0; l < 32; ++l) {
            const int q = ((j & 1) ? (qs[l] >> 4) : (qs[l] & 0xF)) + ((qh[l] >> j) & 1 ? 16 : 0);
            out[l] = d1 * (float) q - m1;
        }
    } else if constexpr (T == T_Q6_K) {
        const uint8_t* b = row + 210 * (c >> 3);
        const int j = c & 7, n = j >> 2, r = j & 3;
        const uint8_t* ql = b + 64 * n;
        const uint8_t* qh = b + 128 + 32 * n;
        const int8_t* sc = (const int8_t*) (b + 192) + 8 * n;
        const float d = h2f(ld16(b + 208));
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            int q;
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
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}

constexpr int kThreads = 128;

/// y[sel][o] = W_e[o] . x[sel] for row o (grid.x), selection `sel` (grid.y): expert ids[sel] (or 0 without ids).
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

template <int T> __global__ void k_row(const uint8_t* W, size_t row_bytes, int row, int n, float* out) {
    const uint8_t* r = W + (size_t) row * row_bytes;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < n / 32; c += gridDim.x * blockDim.x) deq32<T>(r, c, out + 32 * c);
}

/// y = x / sqrt(mean(x^2) + eps) * w, per group of n (grid = groups); w (length n) may be null.
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
__device__ __forceinline__ float sigm(float v) { return 1.0f / (1.0f + expf(-v)); }

__global__ void k_swiglu(const float* g, const float* u, float* h, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = silu(g[i]) * u[i];
}
__global__ void k_add(float* x, const float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}
/// moe[i] = sum_s w[s] * y[s][i] + ys[i]  (the routed experts' weighted sum plus the UNGATED shared expert)
__global__ void k_moe_sum(float* moe, const float* y, const float* w, int k, const float* ys, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = ys[i];
    for (int s = 0; s < k; ++s) acc += w[s] * y[(size_t) s * n + i];
    moe[i] = acc;
}

// ------------------------------------------------------------------------------------------------ the model
struct Tensor {
    const uint8_t* d = nullptr;   // device
    int type = -1;
    std::vector<int64_t> shape;
    size_t row_bytes = 0;
};

size_t row_bytes_of(int type, int64_t n) {
    int be = 0, bb = 0;
    if (!strata::block_geometry((uint32_t) type, be, bb)) return 0;
    return (size_t) (n / be) * (size_t) bb;
}

__global__ void k_act_q8(const float* x, size_t stride, int n, float* out) {
    const int sel = blockIdx.y, blk = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, l = threadIdx.x % 32;
    if (blk * 32 >= n) return;
    const float v = x[(size_t) sel * stride + blk * 32 + l];
    __shared__ float am[kThreads];
    am[threadIdx.x] = fabsf(v);
    __syncthreads();
    const int base = threadIdx.x - l;
    float amax = 0.f;
    for (int i = 0; i < 32; ++i) amax = fmaxf(amax, am[base + i]);
    const float d = amax / 127.0f;
    const float q = amax == 0.0f ? 0.0f : roundf(v / d);
    out[(size_t) sel * n + blk * 32 + l] = q * d;
}

bool g_act_q8 = true;
float* g_act_buf = nullptr;
constexpr size_t kActBuf = 1 << 16;

bool g_strata = true;
cudaStream_t g_st = nullptr;
void* g_q8 = nullptr;
std::vector<int> g_ids_host;

template <int T>
void gemv_t(const Tensor& w, size_t expert_bytes, const int* ids, int n_sel, const float* x, size_t x_stride, float* y) {
    const int n_in = (int) w.shape[0], n_out = (int) w.shape[1];
    k_gemv<T><<<dim3(n_out, n_sel), kThreads>>>(w.d, w.row_bytes, expert_bytes, ids, n_in, x, x_stride, y, n_out);
}

void gemv(const Tensor& w, const float* x, float* y, const int* ids = nullptr, int n_sel = 1, size_t x_stride = 0) {
    const size_t eb = w.shape.size() == 3 ? w.row_bytes * (size_t) w.shape[1] : 0;
    const bool quant = w.type == T_Q8_0 || w.type == T_Q4_K || w.type == T_Q5_K || w.type == T_Q6_K;
    if (quant && g_strata) {
        const int n_in = (int) w.shape[0], n_out = (int) w.shape[1], cols = x_stride ? n_sel : 1;
        if (x_stride && x_stride != (size_t) n_in) {
            std::fprintf(stderr, "kolibriref: the native path wants contiguous input columns\n");
            std::exit(1);
        }
        strata::kernels::native_quantize_q8_1(x, g_q8, n_in, cols, g_st);
        const size_t qb = strata::kernels::native_q8_1_bytes(n_in, 1);
        for (int sel = 0; sel < n_sel; ++sel) {
            const uint8_t* we = w.d + (ids ? (size_t) g_ids_host[(size_t) sel] * eb : 0);
            strata::kernels::native_mmvq(w.type, we, (const char*) g_q8 + (x_stride ? (size_t) sel * qb : 0),
                                         y + (size_t) sel * n_out, n_in, n_out, 1, g_st);
        }
        CK(cudaGetLastError());
        return;
    }
    if (quant && g_act_q8) {
        const int n = (int) w.shape[0], sel = x_stride ? n_sel : 1;
        if ((size_t) n * sel > kActBuf) {
            std::fprintf(stderr, "kolibriref: activation scratch too small\n");
            std::exit(1);
        }
        k_act_q8<<<dim3((n / 32 + kThreads / 32 - 1) / (kThreads / 32), sel), kThreads>>>(x, x_stride, n, g_act_buf);
        x = g_act_buf;
        x_stride = x_stride ? (size_t) n : 0;
    }
    switch (w.type) {
    case T_F32: gemv_t<T_F32>(w, eb, ids, n_sel, x, x_stride, y); break;
    case T_F16: gemv_t<T_F16>(w, eb, ids, n_sel, x, x_stride, y); break;
    case T_BF16: gemv_t<T_BF16>(w, eb, ids, n_sel, x, x_stride, y); break;
    case T_Q8_0: gemv_t<T_Q8_0>(w, eb, ids, n_sel, x, x_stride, y); break;
    case T_Q4_K: gemv_t<T_Q4_K>(w, eb, ids, n_sel, x, x_stride, y); break;
    case T_Q5_K: gemv_t<T_Q5_K>(w, eb, ids, n_sel, x, x_stride, y); break;
    case T_Q6_K: gemv_t<T_Q6_K>(w, eb, ids, n_sel, x, x_stride, y); break;
    default:
        std::fprintf(stderr, "kolibriref: no kernel for ggml type %d (%s)\n", w.type, strata::ggml_type_name((uint32_t) w.type));
        std::exit(1);
    }
    CK(cudaGetLastError());
}

/// A 1-D tensor as f32 on the device (F32 as is, BF16/F16 widened at load).
struct Model {
    strata::core::ModelGeometry g;
    int64_t top_k = 6;
    float eps = 1e-6f, rope_base = 1e4f;
    std::map<std::string, Tensor> t;
    std::map<std::string, float*> f32;
    const Tensor& W(const std::string& n) const {
        auto it = t.find(n);
        if (it == t.end()) { std::fprintf(stderr, "kolibriref: missing tensor %s\n", n.c_str()); std::exit(1); }
        return it->second;
    }
    const float* F(const std::string& n) const {
        auto it = f32.find(n);
        if (it == f32.end()) { std::fprintf(stderr, "kolibriref: missing vector %s\n", n.c_str()); std::exit(1); }
        return it->second;
    }
};

float h2f_host(uint16_t h) {
    const uint32_t s = (uint32_t) (h >> 15) << 31, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e == 0) return (s ? -1.f : 1.f) * std::ldexp((float) m, -24);
    const uint32_t f = e == 31 ? (s | 0x7f800000u | (m << 13)) : (s | ((e + 112) << 23) | (m << 13));
    float r;
    std::memcpy(&r, &f, 4);
    return r;
}

bool load(const std::string& path, Model& m, std::string& err) {
    const auto t0 = std::chrono::steady_clock::now();
    const strata::GgufModel gm = strata::GgufModel::open(path);
    if (!strata::core::geometry_from_gguf(gm.meta(), m.g, m.top_k, err)) return false;
    if (m.g.family != strata::ModelFamily::Kolibri1) {
        err = "this reference path is for Kolibri 1 (kolibri1) only";
        return false;
    }
    size_t checked = 0;
    if (!strata::core::check_gguf_layout(gm, m.g, err, &checked)) return false;
    const std::string p = "kolibri1.";
    if (const strata::MetaValue* v = gm.meta().get(p + "attention.layer_norm_rms_epsilon")) m.eps = (float) v->num();
    if (const strata::MetaValue* v = gm.meta().get(p + "rope.freq_base")) m.rope_base = (float) v->num();
    uint64_t bytes = 0;
    for (size_t s = 0; s < gm.size(); ++s) {
        const strata::GgufFile& f = gm.shard(s);
        for (const strata::TensorInfo& ti : f.tensors()) {
            const uint64_t nb = strata::tensor_payload_bytes(ti);
            const uint8_t* src = f.tensor_data(ti);
            if (ti.shape.size() == 1) {
                std::vector<float> h(ti.elements());
                if (ti.type == T_F32) std::memcpy(h.data(), src, h.size() * 4);
                else if (ti.type == T_BF16)
                    for (size_t i = 0; i < h.size(); ++i) {
                        const uint32_t u = (uint32_t) (src[2 * i] | (src[2 * i + 1] << 8)) << 16;
                        std::memcpy(&h[i], &u, 4);
                    }
                else if (ti.type == T_F16)
                    for (size_t i = 0; i < h.size(); ++i) h[i] = h2f_host((uint16_t) (src[2 * i] | (src[2 * i + 1] << 8)));
                else { err = ti.name + ": a 1-D tensor of type " + ti.type_name(); return false; }
                float* d = nullptr;
                CK(cudaMalloc(&d, h.size() * 4));
                CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
                m.f32[ti.name] = d;
                bytes += h.size() * 4;
                continue;
            }
            Tensor w;
            w.type = (int) ti.type;
            for (uint64_t d : ti.shape) w.shape.push_back((int64_t) d);
            w.row_bytes = row_bytes_of(w.type, w.shape[0]);
            uint8_t* d = nullptr;
            // the routed experts (3-D) in managed memory: 44 GiB does not fit a 16 GB card, the pages come from
            // system RAM on touch (this is the reference path: exactness first, speed second)
            CK(ti.shape.size() == 3 ? cudaMallocManaged(&d, nb, cudaMemAttachGlobal) : cudaMalloc(&d, nb));
            CK(cudaMemcpy(d, src, nb, cudaMemcpyHostToDevice));
            w.d = d;
            m.t[ti.name] = w;
            bytes += nb;
        }
    }
    std::fprintf(stderr, "kolibriref: %s: %zu tensors (layout checked), %.2f GiB on the GPU in %.1f s\n", path.c_str(),
                 checked, (double) bytes / (1 << 30),
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return true;
}

struct State {
    int ctx = 8192, pos = 0;
    float *x, *h, *y, *h2, *q, *k, *v, *att, *router, *eg, *eu, *eh, *ey, *sg, *su, *sh, *sy, *moe, *wts, *logits;
    int* ids;
    std::vector<float*> kc, vc;   // per layer: full layers [ctx][HKV*D], sliding layers the 513-slot ring
};

float* dalloc(size_t n) {
    float* p = nullptr;
    CK(cudaMalloc(&p, n * 4));
    CK(cudaMemset(p, 0, n * 4));
    return p;
}

void init_state(const Model& m, State& s) {
    const auto& g = m.g;
    const int N = (int) g.n_embd, FF = (int) g.n_ff, K = (int) m.top_k;
    s.x = dalloc(N); s.h = dalloc(N); s.y = dalloc(N); s.h2 = dalloc(N); s.moe = dalloc(N); s.sy = dalloc(N);
    s.q = dalloc(g.n_head * g.head_dim); s.k = dalloc(g.n_head_kv * g.head_dim); s.v = dalloc(g.n_head_kv * g.head_dim);
    s.att = dalloc(g.n_head * g.head_dim);
    s.router = dalloc(g.n_expert);
    s.eg = dalloc((size_t) K * FF); s.eu = dalloc((size_t) K * FF); s.eh = dalloc((size_t) K * FF); s.ey = dalloc((size_t) K * N);
    s.sg = dalloc(g.n_ff_shexp); s.su = dalloc(g.n_ff_shexp); s.sh = dalloc(g.n_ff_shexp);
    s.wts = dalloc(K);
    s.logits = dalloc(g.n_vocab);
    CK(cudaMalloc(&s.ids, K * sizeof(int)));
    g_act_buf = dalloc(kActBuf);
    CK(cudaStreamCreate(&g_st));
    CK(cudaMalloc(&g_q8, strata::kernels::native_q8_1_bytes((int) kActBuf, 1)));
    const int64_t kv_rows = strata::core::is_swa_layer(g, 0) ? g.n_swa : s.ctx;   // every layer is MoE; the pattern decides the cache
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const int64_t rows = strata::core::is_swa_layer(g, l) ? g.n_swa : s.ctx;
        s.kc.push_back(dalloc((size_t) g.n_head_kv * rows * g.head_dim));
        s.vc.push_back(dalloc((size_t) g.n_head_kv * rows * g.head_dim));
        (void) kv_rows;
    }
}

std::string L(int64_t l, const char* s) { return "blk." + std::to_string(l) + "." + s; }

void embed(const Model& m, State& s, int tok) {
    const Tensor& e = m.W("token_embd.weight");
    const int N = (int) m.g.n_embd;
    switch (e.type) {
    case T_Q8_0: k_row<T_Q8_0><<<4, 128>>>(e.d, e.row_bytes, tok, N, s.x); break;
    case T_Q4_K: k_row<T_Q4_K><<<4, 128>>>(e.d, e.row_bytes, tok, N, s.x); break;
    case T_Q6_K: k_row<T_Q6_K><<<4, 128>>>(e.d, e.row_bytes, tok, N, s.x); break;
    case T_BF16: k_row<T_BF16><<<4, 128>>>(e.d, e.row_bytes, tok, N, s.x); break;
    case T_F16: k_row<T_F16><<<4, 128>>>(e.d, e.row_bytes, tok, N, s.x); break;
    default:
        std::fprintf(stderr, "kolibriref: token_embd type %d has no row reader\n", e.type);
        std::exit(1);
    }
    CK(cudaGetLastError());
}

/// The layer's attention preparation: q and k RMS-normed per head with their weights, NEOX RoPE over all 128 dims
/// on the sliding layers (the full layers are NoPE), k and v into the cache (the sliding layers' ring: slot pos % 513).
__global__ void k_attn_prep(const float* qx, const float* kx, const float* vx, const float* qw, const float* kw,
                            float* q, float* kc, float* vc, int H, int D, int pos, int64_t swa, int ctx, float base,
                            float eps, bool rope) {
    __shared__ float sh[kThreads];
    __shared__ float buf[512];
    const int head = blockIdx.x;   // 0..H-1: query heads, H..H+HKV-1: key heads (HKV grid rows), then v heads
    const bool isq = head < H;
    const int kvh = head - H;
    const float* src = isq ? qx + (size_t) head * D : kx + (size_t) kvh * D;
    const int64_t rows = rope ? swa : ctx;   // the sliding layers keep the 513-slot ring
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

/// softmax(q k^T * scale) v for one query head per block (GQA: KV head = head / (H / HKV)).  The sliding layers
/// attend to the last `swa` positions through the ring; the full layers to every position.  No output gate.
__global__ void k_attn(const float* q, const float* kc, const float* vc, float* out, int H, int HKV, int D, int pos,
                       int64_t swa, int ctx, float scale, bool sliding) {
    extern __shared__ float sc[];
    __shared__ float sh[kThreads];
    const int h = blockIdx.x, kvh = h / (H / HKV);
    const int64_t rows = sliding ? swa : ctx;
    const int n_pos = sliding ? (pos + 1 < (int) swa ? pos + 1 : (int) swa) : pos + 1;
    const int64_t first = sliding ? (int64_t) pos + 1 - n_pos : 0;
    const float* qh = q + (size_t) h * D;
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
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    mx = sh[0];
    __syncthreads();
    float sum = 0.f;
    for (int p = threadIdx.x; p < n_pos; p += blockDim.x) {
        sc[p] = expf(sc[p] - mx);
        sum += sc[p];
    }
    sum = block_sum(sum, sh);
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float acc = 0.f;
        for (int p = 0; p < n_pos; ++p) {
            const int64_t gp = first + p, slot = sliding ? gp % swa : gp;
            acc += sc[p] * vc[((size_t) kvh * rows + slot) * D + i];
        }
        out[(size_t) h * D + i] = acc / sum;
    }
}

/// One token through the model at position s.pos; the layer outputs into `layers` (host, n_layers x N) when given.
void forward(const Model& m, State& s, int tok, float* layers) {
    const auto& g = m.g;
    const int N = (int) g.n_embd, K = (int) m.top_k, FF = (int) g.n_ff, FS = (int) g.n_ff_shexp;
    const int H = (int) g.n_head, HKV = (int) g.n_head_kv, D = (int) g.head_dim;
    const int B = 256;
    embed(m, s, tok);
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const bool sliding = strata::core::is_swa_layer(g, l);
        const int64_t rows = sliding ? g.n_swa : s.ctx;
        k_rms<<<1, kThreads>>>(s.x, m.F(L(l, "attn_norm.weight")), s.h, N, m.eps);
        gemv(m.W(L(l, "attn_q.weight")), s.h, s.q);
        gemv(m.W(L(l, "attn_k.weight")), s.h, s.k);
        gemv(m.W(L(l, "attn_v.weight")), s.h, s.v);
        k_attn_prep<<<H + HKV, kThreads>>>(s.q, s.k, s.v, m.F(L(l, "attn_q_norm.weight")), m.F(L(l, "attn_k_norm.weight")),
                                           s.q, s.kc[l], s.vc[l], H, D, s.pos, g.n_swa, s.ctx, m.rope_base, m.eps, sliding);
        k_attn<<<H, kThreads, (size_t) std::min<int64_t>((int64_t) s.pos + 1, sliding ? g.n_swa : s.ctx) * sizeof(float)>>>(
            s.q, s.kc[l], s.vc[l], s.att, H, HKV, D, s.pos, g.n_swa, s.ctx, 1.0f / sqrtf((float) D), sliding);
        gemv(m.W(L(l, "attn_output.weight")), s.att, s.y);
        CK(cudaGetLastError());
        k_rms<<<1, kThreads>>>(s.y, m.F(L(l, "attn_post_norm.weight")), s.y, N, m.eps);   // the sandwich norm
        k_add<<<(N + B - 1) / B, B>>>(s.x, s.y, N);
        k_rms<<<1, kThreads>>>(s.x, m.F(L(l, "ffn_norm.weight")), s.h, N, m.eps);
        // router: select top-6 of 384 on `logits + exp_probs_b` (ties -> lower index); weights = sigmoid(logits),
        // unbiased, no renorm (kolibri1.expert_gating_func 5)
        gemv(m.W(L(l, "ffn_gate_inp.weight")), s.h, s.router);
        if (g_strata) {
            strata::kernels::native_router_sigmoid_logit_add(s.router, m.F(L(l, "exp_probs_b.bias")), s.ids, s.wts, 1, g_st);
            g_ids_host.resize((size_t) K);
            CK(cudaMemcpy(g_ids_host.data(), s.ids, K * sizeof(int), cudaMemcpyDeviceToHost));
        } else {
            std::vector<float> r((size_t) g.n_expert), b((size_t) g.n_expert);
            CK(cudaMemcpy(r.data(), s.router, r.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(b.data(), m.F(L(l, "exp_probs_b.bias")), b.size() * 4, cudaMemcpyDeviceToHost));
            std::vector<int> idx((size_t) g.n_expert);
            for (int i = 0; i < (int) idx.size(); ++i) idx[i] = i;
            std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                              [&](int a, int c) { return r[a] + b[a] > r[c] + b[c] || (r[a] + b[a] == r[c] + b[c] && a < c); });
            std::vector<float> w(K);
            for (int i = 0; i < K; ++i) w[i] = 1.0f / (1.0f + std::exp(-r[idx[i]]));
            g_ids_host.assign(idx.begin(), idx.begin() + K);
            CK(cudaMemcpy(s.ids, idx.data(), K * sizeof(int), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(s.wts, w.data(), K * 4, cudaMemcpyHostToDevice));
        }
        gemv(m.W(L(l, "ffn_gate_exps.weight")), s.h, s.eg, s.ids, K, 0);
        gemv(m.W(L(l, "ffn_up_exps.weight")), s.h, s.eu, s.ids, K, 0);
        k_swiglu<<<(K * FF + B - 1) / B, B>>>(s.eg, s.eu, s.eh, K * FF);
        gemv(m.W(L(l, "ffn_down_exps.weight")), s.eh, s.ey, s.ids, K, FF);
        gemv(m.W(L(l, "ffn_gate_shexp.weight")), s.h, s.sg);
        gemv(m.W(L(l, "ffn_up_shexp.weight")), s.h, s.su);
        k_swiglu<<<(FS + B - 1) / B, B>>>(s.sg, s.su, s.sh, FS);
        gemv(m.W(L(l, "ffn_down_shexp.weight")), s.sh, s.sy);
        k_moe_sum<<<(N + B - 1) / B, B>>>(s.moe, s.ey, s.wts, K, s.sy, N);   // the shared expert is UNGATED
        CK(cudaGetLastError());
        k_rms<<<1, kThreads>>>(s.moe, m.F(L(l, "ffn_post_norm.weight")), s.moe, N, m.eps);   // the sandwich norm
        k_add<<<(N + B - 1) / B, B>>>(s.x, s.moe, N);
        CK(cudaGetLastError());
        if (layers) CK(cudaMemcpy(layers + (size_t) l * N, s.x, (size_t) N * 4, cudaMemcpyDeviceToHost));
        (void) rows;
    }
    k_rms<<<1, kThreads>>>(s.x, m.F("output_norm.weight"), s.h, N, m.eps);
    gemv(m.W("output.weight"), s.h, s.logits);
    CK(cudaDeviceSynchronize());
    ++s.pos;
}

std::vector<int> parse_ids(const std::string& s) {
    std::vector<int> v;
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && (s[i] == ',' || s[i] == ' ' || s[i] == '\n' || s[i] == '\r')) ++i;
        if (i >= s.size()) break;
        v.push_back(std::atoi(s.c_str() + i));
        while (i < s.size() && s[i] != ',' && s[i] != ' ' && s[i] != '\n') ++i;
    }
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model, tokens, dump_layers, dump_logits, force, dump_full;
    int n_gen = 64, ctx = 8192;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "kolibriref: %s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--model") model = next();
        else if (a == "--tokens") tokens = next();
        else if (a == "--tokens-file") {
            FILE* f = std::fopen(next().c_str(), "rb");
            if (!f) { std::fprintf(stderr, "kolibriref: cannot read the tokens file\n"); return 2; }
            char buf[4096];
            size_t n;
            while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) tokens.append(buf, n);
            std::fclose(f);
        } else if (a == "--n") n_gen = std::atoi(next().c_str());
        else if (a == "--ctx") ctx = std::atoi(next().c_str());
        else if (a == "--dump-layers") dump_layers = next();
        else if (a == "--dump-logits") dump_logits = next();
        else if (a == "--force") force = next();
        else if (a == "--kernels") {
            const std::string v = next();
            if (v != "strata" && v != "plain") { std::fprintf(stderr, "kolibriref: --kernels strata or plain\n"); return 2; }
            g_strata = v == "strata";
        } else if (a == "--dump-full-logits") dump_full = next();
        else if (a == "--act") {
            const std::string v = next();
            if (v != "q8" && v != "f32") { std::fprintf(stderr, "kolibriref: --act q8 or f32\n"); return 2; }
            g_act_q8 = v == "q8";
        } else {
            std::fprintf(stderr, "kolibriref: usage: --model GGUF --tokens ID,ID,... [--n 64] [--ctx 8192] "
                                 "[--dump-layers FILE] [--dump-logits FILE] [--act q8|f32] [--kernels strata|plain] "
                                 "[--force IDS] [--dump-full-logits F]\n");
            return 2;
        }
    }
    if (model.empty() || tokens.empty()) {
        std::fprintf(stderr, "kolibriref: --model and --tokens are required\n");
        return 2;
    }
    Model m;
    std::string err;
    if (!load(model, m, err)) {
        std::fprintf(stderr, "kolibriref: %s\n", err.c_str());
        return 1;
    }
    State s;
    s.ctx = ctx;
    init_state(m, s);
    const std::vector<int> prompt = parse_ids(tokens), forced = parse_ids(force);
    FILE* fl = dump_layers.empty() ? nullptr : std::fopen(dump_layers.c_str(), "wb");
    FILE* fg = dump_logits.empty() ? nullptr : std::fopen(dump_logits.c_str(), "wb");
    FILE* ff = dump_full.empty() ? nullptr : std::fopen(dump_full.c_str(), "wb");
    const int V = (int) m.g.n_vocab;
    std::vector<float> layers((size_t) m.g.n_layers * m.g.n_embd);
    std::vector<float> lg;
    int tok = prompt[0];
    for (size_t t = 1; t < prompt.size(); ++t) {
        forward(m, s, (int) prompt[t - 1], fl ? layers.data() : nullptr);
        if (fl) {
            for (int64_t l = 0; l < m.g.n_layers; ++l) std::fwrite(layers.data() + (size_t) l * m.g.n_embd, 4, m.g.n_embd, fl);
            // the prompt's first token was embedded but not run: position 0 is the FIRST processed token, so the
            // loop above runs prompt[1..]; the dump covers positions 0..n-2 (the same rows llama.cpp dumps)
        }
        tok = (int) prompt[t];
    }
    // the last prompt token (and every generated one): run it, take the argmax, feed it back
    const int n_steps = (int) forced.empty() ? n_gen : (int) forced.size() + 1;
    std::vector<int> gen;
    for (int i = 0; i < n_steps; ++i) {
        forward(m, s, tok, fl ? layers.data() : nullptr);
        if (fl)
            for (int64_t l = 0; l < m.g.n_layers; ++l)
                std::fwrite(layers.data() + (size_t) l * m.g.n_embd, 4, m.g.n_embd, fl);
        lg.resize((size_t) V);
        CK(cudaMemcpy(lg.data(), s.logits, (size_t) V * 4, cudaMemcpyDeviceToHost));
        int a = 0, c = -1;
        for (int j = 1; j < V; ++j) if (lg[j] > lg[a]) a = j;
        for (int j = 0; j < V; ++j) if (j != a && (c < 0 || lg[j] > lg[c])) c = j;
        if (fg) { const float rec[4] = {(float) a, lg[a], (float) c, lg[c]}; std::fwrite(rec, 4, 4, fg); }
        if (ff) std::fwrite(lg.data(), 4, V, ff);
        gen.push_back(a);
        tok = i + 1 == n_steps ? a : (i < (int) forced.size() ? forced[i] : a);
    }
    std::printf("gen:");
    for (int t : gen) std::printf(" %d", t);
    std::printf("\n");
    if (fl) std::fclose(fl);
    if (fg) std::fclose(fg);
    if (ff) std::fclose(ff);
    return 0;
}
