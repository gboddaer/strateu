// tools/kolibri/kolibri_llama_dump.cpp - llama.cpp's side of the Kolibri 1 parity check (plan
// docs/superpowers/plans/2026-10-16-kolibri1.md T5): the same GGUF, the same token ids, llama.cpp's own graph.
// Prints the prompt's ids and the greedy ids, and dumps every processed position's layer outputs ("l_out", the
// kolibri1 graph names them per layer) in strata-kolibriref's format ([pos][layer][n_embd] f32) plus each greedy
// step's top-2 (id, logit).  Model-agnostic: n_embd / n_layer come from the file.
//
// Built against a llama.cpp checkout with the kolibri1 patch (not part of Strata's build):
//   g++ -O2 -std=c++17 kolibri_llama_dump.cpp -I<llama>/include -I<llama>/ggml/include -L<llama>/build/bin -lllama
//       -lggml -lggml-base -Wl,-rpath,<llama>/build/bin -o kolibri_llama_dump
// Usage: kolibri_llama_dump --model GGUF (--prompt TEXT | --tokens ID,ID,...) [--n 64] [--ngl 99] [--cpu-moe 50]
//                           [--dump-layers F] [--dump-logits F] [--force ID,ID,...] [--dump-full-logits F]
//   --cpu-moe N keeps the MoE expert weights of the first N layers in system RAM (llama.cpp's --n-cpu-moe, via
//   tensor_buft_overrides): a 47 GB file on a 16 GB card runs with attention, router, shared experts and KV on
//   the GPU and the 384 routed experts per layer in RAM - the same shape Strata runs.
//   --force feeds that continuation instead of the greedy ids (teacher forcing); --dump-full-logits writes every
//   step's logits (f32, [step][vocab]).
//   --prompt is tokenized with special tokens parsed and no BOS added (Kolibri adds none).
#include "llama.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <map>
#include <string>
#include <vector>

namespace {
struct Capture {
    int n_embd = 0, n_layer = 0, n_tok = 0;
    std::map<int, std::vector<float>> layers;   // layer -> [n_tok][n_embd] of the current batch
};

bool on_tensor(struct ggml_tensor* t, bool ask, void* user) {
    auto* c = (Capture*) user;
    const char* name = ggml_get_name(t);
    if (std::strncmp(name, "l_out-", 6) != 0) return ask ? false : true;
    if (ask) return true;
    const int il = std::atoi(name + 6);
    if (t->type != GGML_TYPE_F32 || t->ne[0] != c->n_embd) return true;
    std::vector<float>& v = c->layers[il];
    v.resize((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    return true;
}

std::vector<int> parse_ids(const std::string& s) {
    std::vector<int> v;
    const char* p = s.c_str();
    while (*p) {
        while (*p == ',' || *p == ' ') ++p;
        if (!*p) break;
        v.push_back(std::atoi(p));
        while (*p && *p != ',' && *p != ' ') ++p;
    }
    return v;
}
}  // namespace

int main(int argc, char** argv) {
    std::string model, prompt, tokens, tokens_file, dump_layers, dump_logits, force, dump_full;
    int n_gen = 64, ngl = 99, n_cpu_moe = 0;
    ggml_type kv_type = GGML_TYPE_F16;   // the default; --kv-type overrides it for the precision floor test
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--model") model = v;
        else if (a == "--prompt") prompt = v;
        else if (a == "--tokens") tokens = v;
        // A 100K-token prompt is ~700 KB of ids, which cannot travel in one argv entry (Linux caps it at 128 KB), so the
        // long-context anchor comparison needs the ids from a file.
        else if (a == "--tokens-file") tokens_file = v;
        else if (a == "--n") n_gen = std::atoi(v.c_str());
        else if (a == "--ngl") ngl = std::atoi(v.c_str());
        else if (a == "--cpu-moe") n_cpu_moe = std::atoi(v.c_str());
        else if (a == "--dump-layers") dump_layers = v;
        else if (a == "--dump-logits") dump_logits = v;
        else if (a == "--force") force = v;
        else if (a == "--dump-full-logits") dump_full = v;
        // --kv-type f32 runs the reference with fp32 K/V instead of the default fp16.  The floor test needs it: if the
        // reference's own answer moves when its KV precision changes, then a marginal token at 98.6K is precision-bound
        // for BOTH engines and the requirement has to be scoped accordingly; if it does not move, the defect is ours.
        else if (a == "--kv-type") {
            if (v == "f32") kv_type = GGML_TYPE_F32;
            else if (v == "f16") kv_type = GGML_TYPE_F16;
            else if (v == "bf16") kv_type = GGML_TYPE_BF16;
            else { std::fprintf(stderr, "--kv-type takes f32, f16 or bf16\n"); return 2; }
        }
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    // STRATA_DUMP_NGL: the GPU layers when the caller asks for the GPU (--ngl > 0) but the file does not fit (the
    // 47 GB Q4_K_M on a 16 GB T4); with --cpu-moe the experts stay in RAM and this stays at the full 50
    if (const char* g = std::getenv("STRATA_DUMP_NGL"); g != nullptr && ngl > 0) ngl = std::atoi(g);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    std::list<std::string> patterns;
    std::vector<llama_model_tensor_buft_override> overrides;
    if (n_cpu_moe > 0) {
        // llama.cpp's --n-cpu-moe: per-layer "blk.<i>\.ffn_(up|down|gate|gate_up)_(ch|)exps" -> CPU buffer
        for (int i = 0; i < n_cpu_moe; ++i) {
            char buf[128];
            std::snprintf(buf, sizeof buf, "blk\\.%d\\.ffn_(up|down|gate|gate_up)_(ch|)exps", i);
            patterns.push_back(buf);
            overrides.push_back({patterns.back().c_str(), ggml_backend_cpu_buffer_type()});
        }
        overrides.push_back({nullptr, nullptr});   // the list is NULL-terminated
        mp.tensor_buft_overrides = overrides.data();
    }
    llama_model* mdl = llama_model_load_from_file(model.c_str(), mp);
    if (!mdl) { std::fprintf(stderr, "cannot load %s\n", model.c_str()); return 1; }
    const llama_vocab* vocab = llama_model_get_vocab(mdl);
    std::vector<llama_token> ids;
    if (!tokens_file.empty()) {   // read the id list OUTSIDE the emptiness test: --tokens-file alone leaves `tokens` empty
        std::FILE* f = std::fopen(tokens_file.c_str(), "rb");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", tokens_file.c_str()); return 1; }
        std::string all;
        char buf[1 << 16];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) all.append(buf, n);
        std::fclose(f);
        tokens = all;
    }
    if (!tokens.empty()) {
        for (int t : parse_ids(tokens)) ids.push_back(t);
    } else {
        ids.resize(prompt.size() + 16);
        const int n = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), ids.data(), (int) ids.size(), false, true);
        if (n < 0) { std::fprintf(stderr, "tokenize failed\n"); return 1; }
        ids.resize(n);
    }
    Capture cap;
    cap.n_embd = llama_model_n_embd(mdl);
    cap.n_layer = llama_model_n_layer(mdl);
    llama_context_params cp = llama_context_default_params();
    // STRATA_DUMP_CTX: a smaller context / batch (the parity sequences are a few hundred tokens) for a file whose
    // weights leave no VRAM for the 4096-token compute buffers
    const char* cx = std::getenv("STRATA_DUMP_CTX");
    const int nctx = cx ? std::atoi(cx) : 4096;
    cp.n_ctx = nctx;
    cp.type_k = kv_type;
    cp.type_v = kv_type;
    // A long-context prompt must be prefilled in BATCHES: with n_batch = n_ubatch = n_ctx a 105K-token prompt makes
    // llama.cpp reserve a compute graph for a 105K batch (measured: "allocating 76463.15 MiB on device 0 ... failed to
    // allocate CUDA0 buffer of size 80177422080").  The context stays nctx; only the batch is bounded.
    cp.n_batch = cp.n_ubatch = std::min(nctx, 2048);
    cp.op_offload = ngl > 0;   // --ngl 0 is the CPU backend: no prompt matmuls offloaded to a GPU either
    cp.cb_eval = on_tensor;
    cp.cb_eval_user_data = &cap;
    llama_context* ctx = llama_init_from_model(mdl, cp);
    if (!ctx) { std::fprintf(stderr, "cannot create the context\n"); return 1; }
    FILE* fl = dump_layers.empty() ? nullptr : std::fopen(dump_layers.c_str(), "wb");
    FILE* fg = dump_logits.empty() ? nullptr : std::fopen(dump_logits.c_str(), "wb");
    FILE* ff = dump_full.empty() ? nullptr : std::fopen(dump_full.c_str(), "wb");
    const std::vector<int> forced = parse_ids(force);
    if (!forced.empty()) n_gen = (int) forced.size() + 1;
    const int V = llama_vocab_n_tokens(vocab);
    auto write_layers = [&](int n_tok) {
        if (!fl) return;
        // a layer may keep fewer rows than the batch (the last layer keeps only the output rows): those are the
        // batch's LAST positions; the positions it did not compute are written as zeros
        std::vector<float> zero(cap.n_embd, 0.f);
        for (int t = 0; t < n_tok; ++t)
            for (int l = 0; l < cap.n_layer; ++l) {
                auto it = cap.layers.find(l);
                const int rows = it == cap.layers.end() ? 0 : (int) (it->second.size() / cap.n_embd);
                const int r = t - (n_tok - rows);
                const float* src = r >= 0 ? it->second.data() + (size_t) r * cap.n_embd : zero.data();
                std::fwrite(src, 4, cap.n_embd, fl);
            }
        cap.layers.clear();
    };
    std::printf("prompt:");
    for (int t : ids) std::printf(" %d", t);
    std::printf("\n");
    std::vector<llama_token> gen;
    // A 100K-token prompt cannot be one llama_batch (that asks llama.cpp for a 100K-token compute graph, measured as a
    // 76 GB allocation failure).  Ingest it in bounded slices; the last slice carries the final token, whose logits are
    // what the first generated token needs.
    {
        const size_t kSlice = 2048;
        for (size_t off = 0; off < ids.size(); off += kSlice) {
            const size_t m = std::min<size_t>(kSlice, ids.size() - off);
            // explicit batch: llama_batch_get_one marks no token for logits, so a chunked ingest ends with n_outputs=0
            // and the first generated token has nothing to read ("get_logits_ith: invalid logits id -1").  Ask for the
            // logits of the FINAL prompt token only.
            llama_batch b = llama_batch_init((int32_t) m, 0, 1);
            for (size_t k = 0; k < m; ++k) {
                const int32_t idx = b.n_tokens++;
                b.token[idx] = ids[off + k];
                b.pos[idx] = (llama_pos) (off + k);
                b.n_seq_id[idx] = 1;
                b.seq_id[idx][0] = 0;
                b.logits[idx] = (off + k + 1 == ids.size()) ? 1 : 0;
            }
            const int rc = llama_decode(ctx, b);
            llama_batch_free(b);
            if (rc != 0) {
                std::fprintf(stderr, "decode failed at prompt token %zu\n", off);
                return 1;
            }
        }
    }
    // BOUNDED: write only the LAST prompt position's per-layer hidden state.  Writing every prompt token at 98.6K is
    // 98,591 x 50 x 2560 x 4 bytes = 50 GB, which is why a layer-by-layer comparison at 100K was impossible.  The lambda
    // takes the batch's LAST rows, so n_tok = 1 is exactly the final prompt token - the state the first generated token
    // is read from - and each generated step then appends one row via write_layers(1) below.
    write_layers(1);
    for (int i = 0; i < n_gen; ++i) {
        const float* lg = llama_get_logits_ith(ctx, -1);
        int a = 0, c = -1;
        for (int j = 1; j < V; ++j) if (lg[j] > lg[a]) a = j;
        for (int j = 0; j < V; ++j) if (j != a && (c < 0 || lg[j] > lg[c])) c = j;
        if (fg) { const float rec[4] = {(float) a, lg[a], (float) c, lg[c]}; std::fwrite(rec, 4, 4, fg); }
        gen.push_back(a);
        if (ff) std::fwrite(lg, 4, V, ff);
        if (i + 1 == n_gen) break;
        llama_token t = i < (int) forced.size() ? forced[i] : a;
        llama_batch one = llama_batch_get_one(&t, 1);
        if (llama_decode(ctx, one) != 0) { std::fprintf(stderr, "decode failed\n"); return 1; }
        write_layers(1);
    }
    std::printf("gen:");
    for (int t : gen) std::printf(" %d", t);
    std::printf("\n");
    if (!dump_layers.empty() || !dump_logits.empty()) {   // --text: the greedy answer decoded, the sanity check
        std::string text;
        for (int t : gen) {
            char buf[256];
            const int n = llama_token_to_piece(vocab, t, buf, sizeof buf, 0, true);
            if (n > 0) text.append(buf, (size_t) n);
        }
        std::printf("text:%s\n", text.c_str());
    }
    if (fl) std::fclose(fl);
    if (fg) std::fclose(fg);
    if (ff) std::fclose(ff);
    llama_free(ctx);
    llama_model_free(mdl);
    llama_backend_free();
    return 0;
}
