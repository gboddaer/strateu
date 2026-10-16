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

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
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
    uint8_t* table = nullptr;       // pinned host: n_expert slots
    bool native_ok = false;         // native_expert_supported for this layer's (gu, d) at these dims
};

struct Engine {
    strata::core::ModelGeometry g;
    int64_t top_k = 6;
    float eps = 1e-6f, rope_base = 1e4f;
    std::map<std::string, Dense> dense;
    std::map<std::string, float*> f32;
    std::vector<LayerBlobs> blobs;   // per layer
};

std::string L(int64_t l, const char* s) { return "blk." + std::to_string(l) + "." + s; }

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
        const size_t sb = lcm(lcm((size_t) 144, (size_t) 210), (size_t) 34);   // Q4_K / Q6_K / Q8_0 super-blocks
        b.stride = ((b.L.bytes + sb - 1) / sb) * sb;
        KC(cudaHostAlloc((void**) &b.table, b.stride * (size_t) e.g.n_expert, cudaHostAllocDefault));
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
    std::string model;
    bool load_only = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "--model" || a == "--native") && i + 1 < argc) model = argv[++i];
        else if (a == "--load-only") load_only = true;
    }
    if (model.empty()) {
        std::fprintf(stderr, "strata-kolibri: the Kolibri 1 engine path (plan T7); --model GGUF [--load-only] for now\n");
        return 2;
    }
    Engine e;
    std::string err;
    if (!load(model, e, err)) {
        std::fprintf(stderr, "strata-kolibri: %s\n", err.c_str());
        return 1;
    }
    if (load_only) return 0;
    std::fprintf(stderr, "strata-kolibri: the layer driver and serve protocol arrive in plan T8; --load-only runs "
                         "the load\n");
    return 1;
}
