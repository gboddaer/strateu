// Diagnostic only: hash valid KV bytes at the already-quiescent prefill/decode boundary.
// No extra GPU kernels and no per-layer decode synchronization. Not a throughput instrument.
static void prefix_fingerprint(const Engine& e, const Scratch& s) {
    const char* path = std::getenv("STRATA_KOLIBRI_PREFIX_FINGERPRINT");
    if (!path || !*path) return;
    std::FILE* file = std::fopen(path, "w");
    if (!file) {
        std::fprintf(stderr, "kolibri: cannot open prefix fingerprint %s\n", path);
        std::exit(1);
    }
    auto hash = [](const std::vector<__half>& values) {
        uint64_t value = 14695981039346656037ULL;
        const auto* bytes = reinterpret_cast<const unsigned char*>(values.data());
        for (size_t i = 0; i < values.size() * sizeof(__half); ++i) {
            value ^= bytes[i];
            value *= 1099511628211ULL;
        }
        return value;
    };
    std::fprintf(file, "position=%lld ctx=%d\n", (long long) s.pos, s.ctx);
    for (size_t l = 0; l < e.blobs.size(); ++l) {
        const bool sliding = strata::core::is_swa_layer(e.g, (int64_t) l);
        const int64_t stride = sliding ? e.g.n_swa : s.ctx;
        const int64_t rows = std::min<int64_t>(s.pos, stride);
        std::vector<__half> k, v;
        if (rows > 0 &&
            (!reuse_gate_copy_kv(s.kc[l], stride, rows, (int) e.g.n_head_kv, (int) e.g.head_dim, k) ||
             !reuse_gate_copy_kv(s.vc[l], stride, rows, (int) e.g.n_head_kv, (int) e.g.head_dim, v))) {
            std::fprintf(stderr, "kolibri: prefix fingerprint copy failed at layer %zu\n", l);
            std::exit(1);
        }
        std::fprintf(file, "%zu rows=%lld K=%016llx V=%016llx\n", l, (long long) rows,
                     (unsigned long long) hash(k), (unsigned long long) hash(v));
    }
    if (std::fclose(file) != 0) std::exit(1);
}
