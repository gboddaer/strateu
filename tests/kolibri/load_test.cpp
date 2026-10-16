// tests/kolibri/load_test.cpp - the GPU-free half of Task 7 (plan docs/superpowers/plans/2026-10-16-kolibri1.md):
// the geometry, the layout, the per-layer expert-blob arithmetic and the dtype census, straight from the GGUF
// header.  The engine (src/kolibri/kolibri_engine.cu) repacks the routed experts into [gate|up|down] blobs; this
// test pins the arithmetic the repack and the CPU-pool table must agree with: per-layer dtypes (the Q4_K_M files
// alternate the down projection Q4_K / Q6_K), the blob bytes, the stride (a whole number of super-blocks, so any
// blob start is block-aligned), and the 50 x 384 blob count.  No CUDA, no model resident: it runs on the header.
//
//   g++ -std=c++17 -Iinclude tests/kolibri/load_test.cpp -o /tmp/kl_load && /tmp/kl_load [GGUF...]
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/family_layout.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef __linux__
#include <sys/resource.h>
static long rss_mib() {
    std::FILE* f = std::fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256];
    long v = -1;
    while (std::fgets(line, sizeof line, f))
        if (std::sscanf(line, "VmRSS: %ld kB", &v) == 1) { std::fclose(f); return v / 1024; }
    std::fclose(f);
    return v;
}
#endif

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-72s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

size_t row_bytes(int type, int64_t n) {
    int be = 0, bb = 0;
    if (!strata::block_geometry((uint32_t) type, be, bb)) return 0;
    return (size_t) (n / be) * (size_t) bb;
}

/// The blob of one expert: gate rows (n_embd wide), up rows, down rows (n_ff wide), laid out in that order.
/// The stride rounds the sum up to the lcm of the three types' super-block sizes so every blob start and every
/// row inside it is block-aligned.  Super-blocks: Q4_K 144 B / 256 values, Q6_K 210 / 256, Q8_0 34 / 32, F32 4 / 1.
struct BlobGeometry {
    size_t gate_row = 0, up_row = 0, down_row = 0, bytes = 0, stride = 0, sb = 1;
};
size_t lcm(size_t a, size_t b) { size_t x = a, y = b; while (y) { size_t t = x % y; x = y; y = t; } return a / x * b; }
BlobGeometry blob_geometry(int gate_t, int up_t, int down_t, int64_t n_embd, int64_t n_ff) {
    BlobGeometry b;
    b.gate_row = row_bytes(gate_t, n_embd);
    b.up_row = row_bytes(up_t, n_embd);
    b.down_row = row_bytes(down_t, n_ff);
    b.bytes = (size_t) n_ff * (b.gate_row + b.up_row) + (size_t) n_embd * b.down_row;
    auto sb_of = [](int t) { return t == 12 ? (size_t) 144 : t == 14 ? (size_t) 210 : t == 8 ? (size_t) 34 : (size_t) 4; };
    b.sb = lcm(lcm(sb_of(gate_t), sb_of(up_t)), sb_of(down_t));
    b.stride = ((b.bytes + b.sb - 1) / b.sb) * b.sb;
    return b;
}

int run(const std::string& path) {
    std::printf("%s\n", path.c_str());
    const auto t0 = std::chrono::steady_clock::now();
    std::string err;
    const strata::GgufModel gm = strata::GgufModel::open(path);
    strata::core::ModelGeometry g;
    int64_t top_k = 0;
    if (!strata::core::geometry_from_gguf(gm.meta(), g, top_k, err)) {
        std::printf("  geometry: %s\n", err.c_str());
        return 1;
    }
    check(g.family == strata::ModelFamily::Kolibri1, "family kolibri1");
    check(g.n_layers == 50 && g.n_embd == 2560 && g.n_head == 48 && g.n_head_kv == 4 && g.head_dim == 128 &&
              g.n_expert == 384 && top_k == 6 && g.n_ff == 512 && g.n_ff_shexp == 512 && g.n_swa == 513 && g.gating == 5,
          "geometry: 50 x 2560, 48/4 x 128, 384 top-6 FF 512 shared 512, swa 513, gating 5");
    size_t checked = 0;
    check(strata::core::check_gguf_layout(gm, g, err, &checked), ("layout table complete (" + std::to_string(checked) + " tensors)").c_str());

    // the dtype census per layer, straight from the tensor infos: the Q4_K_M files alternate the down projection
    // Q4_K (12) / Q6_K (14) across layers; the Q8_0 file is Q8_0 (8) throughout
    int down_q4 = 0, down_q6 = 0, down_q8 = 0, other = 0, missing = 0;
    int first_down_type = -1;
    std::vector<BlobGeometry> blobs;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const strata::TensorInfo* gate = nullptr;
        const strata::TensorInfo* down = nullptr;
        for (size_t s = 0; s < gm.size(); ++s)                      // split files: the layers span the shards
            for (const strata::TensorInfo& ti : gm.shard(s).tensors()) {
                if (ti.name == "blk." + std::to_string(l) + ".ffn_gate_exps.weight") gate = &ti;
                if (ti.name == "blk." + std::to_string(l) + ".ffn_down_exps.weight") down = &ti;
            }
        if (!gate || !down) { ++missing; continue; }
        if (l == 0) first_down_type = (int) down->type;
        if (down->type == 12) ++down_q4;
        else if (down->type == 14) ++down_q6;
        else if (down->type == 8) ++down_q8;
        else ++other;
        blobs.push_back(blob_geometry((int) gate->type, (int) gate->type, (int) down->type, g.n_embd, g.n_ff));
    }
    std::printf("  dtype census: down Q4_K %d, Q6_K %d, Q8_0 %d, other %d (layer 0 down type %d)\n", down_q4, down_q6,
                down_q8, other, first_down_type);
    check(missing == 0 && blobs.size() == (size_t) g.n_layers, "every layer's gate/up/down expert tensors found (all shards)");
    check(other == 0, "every layer's down projection is Q4_K, Q6_K or Q8_0");
    // the blob arithmetic: gate/up rows are n_embd wide (Q4_K: 2560/256 * 144 = 1440 B, Q8_0: 2560/32 * 34 = 2720),
    // down rows n_ff wide (Q4_K 288, Q6_K 420, Q8_0 544)
    const size_t want_gate = first_down_type == 8 ? 2720 : 1440;
    check(blobs[0].gate_row == want_gate && blobs[0].up_row == want_gate,
          "gate/up row: Q4_K 1440 B, Q8_0 2720 B (2560 values)");
    const size_t want_down = first_down_type == 12 ? 288 : first_down_type == 14 ? 420 : 544;
    check(blobs[0].down_row == want_down, "down row: Q4_K 288 B, Q6_K 420 B, Q8_0 544 B (512 values)");
    check(blobs[0].bytes == (size_t) g.n_ff * (blobs[0].gate_row + blobs[0].up_row) + (size_t) g.n_embd * blobs[0].down_row,
          "blob bytes = n_ff*(gate+up rows) + n_embd*down row");
    check(blobs[0].stride % blobs[0].sb == 0 && blobs[0].stride >= blobs[0].bytes,
          "blob stride is a whole number of the types' lcm super-blocks");
    // the mixed layers: a Q6_K down blob is larger than a Q4_K one
    if (down_q4 > 0 && down_q6 > 0) {
        size_t b4 = 0, b6 = 0;
        for (size_t l = 0; l < blobs.size(); ++l) (blobs[l].down_row == 288 ? b4 : b6) = blobs[l].bytes;
        check(b6 > b4, "a Q6_K-down blob is larger than a Q4_K-down blob");
    }
    // the whole model: 50 x 384 blobs, the RAM table's total bytes
    size_t total = 0;
    for (const BlobGeometry& b : blobs) total += (size_t) g.n_expert * b.stride;
    std::printf("  blobs: %zu x %lld = %zu, table %.2f GiB (stride-padded)\n", blobs.size(), (long long) g.n_expert,
                blobs.size() * (size_t) g.n_expert, (double) total / (1 << 30));
    check(blobs.size() * (size_t) g.n_expert == 19200, "19,200 expert blobs");
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
#ifdef __linux__
    std::printf("  load: %.2f s header, RSS %ld MiB\n", secs, rss_mib());
#else
    std::printf("  load: %.2f s header\n", secs);
#endif
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> models;
    for (int i = 1; i < argc; ++i) models.push_back(argv[i]);
    if (models.empty())
        models = {"/data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf", "/data4/models/Kolibri-1-Q8_0/Kolibri-1-Q8_0-00001-of-00002.gguf"};
    int rc = 0;
    for (const std::string& m : models) rc |= run(m);
    std::printf("%s (%d failures)\n", failures ? "load_test FAILED" : "load_test OK", failures);
    return failures ? 1 : 0;
}
