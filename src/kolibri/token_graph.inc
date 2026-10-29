// Included inside kolibri_engine.cu's private namespace after layer/pool implementations.
// One parent executable: PRE[l] -> bounded ACK wait -> POST[l] -> PRE[l+1].
// CPU arithmetic, expert selection order and GPU kernels are inherited from the segmented arm.
// The host services publications during REPLAY, never as a callback executed during capture.
static bool run_token_graph(const Engine& e, Scratch& s, cudaStream_t stream) {
    using strata::kolibri::DecodeWaitOk;
    using strata::kolibri::enqueue_decode_wait;
    const int N = (int) e.g.n_embd, K = (int) e.top_k;
    const size_t layers = e.blobs.size();
    // Measurement only (no behaviour change): split the token into the host's wait for the GPU, its own payload
    // work plus the CPU experts, the launch call, and the token-boundary sync.  Wall clocks only, so this does not
    // perturb the run the way a driver trace does.
    const bool timing = env_on("STRATA_KOLIBRI_DEC_TOKEN_TIMING");
    using Clock = std::chrono::steady_clock;
    const auto tc0 = Clock::now();
    double ms_wait = 0, ms_host = 0, ms_launch = 0, ms_sync = 0;
    auto since = [](Clock::time_point a) {
        return std::chrono::duration<double, std::milli>(Clock::now() - a).count();
    };
    DecGraphArm& cache = dec_graph_arm((int64_t) layers);
    std::vector<int> keys;
    keys.reserve(layers);
    for (size_t l = 0; l < layers; ++l) {
        const bool sliding = strata::core::is_swa_layer(e.g, (int64_t) l);
        if (s.kt[l] != nullptr && !s.kt_ready[l]) return false;  // first-use mirror must be built eagerly
        const int n = (int) std::min<int64_t>(s.pos + 1, sliding ? e.g.n_swa : s.ctx);
        const int key = attn_graph_key(n, g_two_pass_max);
        if (!cache.pre_graph[l].count(key) || cache.post_graph[l] == nullptr) return false;
        keys.push_back(key);
    }
    auto cancelled = [] { return decode_cancel && decode_cancel->load(std::memory_order_acquire); };
    if (cancelled()) {
        KC(cudaStreamSynchronize(stream));  // finish this token's embedding before the caller resets state
        decode_aborted = true;
        return true;
    }
    HandoffBuffers& hb = *handoff_buffers(N, K);
    auto* control = hb.host_control;
    auto load = [](uint32_t& word) { return std::atomic_ref<uint32_t>(word).load(std::memory_order_acquire); };
    auto store = [](uint32_t& word, uint32_t value) {
        std::atomic_ref<uint32_t>(word).store(value, std::memory_order_release);
    };
    uint32_t generation = load(control->generation) + 1u;
    if (generation == 0) generation = 1;
    store(control->generation, generation);
    store(control->acknowledged_generation, 0);
    store(control->acknowledged, 0);
    store(control->abort, 0);
    store(control->status, DecodeWaitOk);

    auto& executable = cache.tokens[keys];
    if (executable == nullptr) {
        // Retained segment graphs are cloned into the parent. Keep their source graphs alive for
        // subsequent bucket variants. Embedding, logits projection and sampling remain outside.
        const uint64_t ticks = dec_wait_ticks();
        cudaGraph_t wait_graph = nullptr;
        cudaGraphExec_t wait_exec = dec_capture_graph(
            [&] { enqueue_decode_wait(hb.control, ticks, stream); }, stream, "token ACK", -1, &wait_graph);
        KC(cudaGraphExecDestroy(wait_exec));
        cudaGraph_t parent = nullptr;
#if defined(__HIPCC__)
        KC(hipGraphCreate(&parent, 0));
        hipGraphNode_t last = nullptr;
        auto append = [&](cudaGraph_t child) {
            hipGraphNode_t node = nullptr;
            KC(hipGraphAddChildGraphNode(&node, parent, last ? &last : nullptr, last ? 1 : 0, child));
            last = node;
        };
#else
        KC(cudaGraphCreate(&parent, 0));
        cudaGraphNode_t last = nullptr;
        auto append = [&](cudaGraph_t child) {
            cudaGraphNode_t node = nullptr;
            KC(cudaGraphAddChildGraphNode(&node, parent, last ? &last : nullptr, last ? 1 : 0, child));
            last = node;
        };
#endif
        for (size_t l = 0; l < layers; ++l) {
            append(cache.pre_graph[l].at(keys[l]));
            append(wait_graph);
            append(cache.post_graph[l]);
        }
        KC(cudaGraphInstantiate(&executable, parent, 0));
        KC(cudaGraphDestroy(parent));
        KC(cudaGraphDestroy(wait_graph));
    }
    std::vector<int> selections((size_t) K);
    for (int k = 0; k < K; ++k) selections[(size_t) k] = k;
    g_ids_host.resize((size_t) K);
    const auto tl0 = Clock::now();
    KC(cudaGraphLaunch(executable, stream));
    ms_launch = since(tl0);
    static bool announced = false;
    if (!announced) {
        std::fprintf(stderr, "kolibri: DEC_TOKEN_GRAPH replay active: %zu layers, CPU experts retained\n", layers);
        announced = true;
    }
    bool aborted = false;
    bool user_cancel = false;
    try {
        for (size_t l = 0; l < layers; ++l) {
            const uint32_t want = ++hb.expect;
            const auto tw0 = Clock::now();
            const auto deadline = std::chrono::steady_clock::now() +
                                  std::chrono::milliseconds(dec_handoff_timeout_ms());
            while (load(control->ready) < want) {
                if (cancelled() || load(control->abort) || std::chrono::steady_clock::now() >= deadline) {
                    aborted = true;
                    break;
                }
                std::this_thread::yield();
            }
            ms_wait += since(tw0);
            if (aborted || cancelled() || load(control->abort) || load(control->ready) != want) {
                aborted = true;
                user_cancel = cancelled();
                break;
            }
            if (l == 0 && env_on("STRATA_KOLIBRI_DEC_TOKEN_TRACE")) {
                std::fprintf(stderr, "DEC_TOKEN_PUBLICATION pos=%lld layer=0 generation=%u sequence=%u\n",
                             (long long) s.pos, generation, want);
                std::fflush(stderr);
            }
            const auto th0 = Clock::now();
            std::memcpy(s.h_host, hb.host_x, (size_t) N * 4);
            for (int k = 0; k < K; ++k) g_ids_host[(size_t) k] = hb.host_ids[(size_t) k];
            strata::kernels::cpu::native_quant_act(e.blobs[l].fmt, s.h_host, s.act);
            pool_experts(e, e.blobs[l], s, g_ids_host, selections);
            if (cancelled() || load(control->abort)) {
                aborted = true;
                user_cancel = cancelled();
                break;
            }
            std::memcpy(hb.h_y, s.ey_h, (size_t) K * N * 4);
            store(control->acknowledged_generation, generation);
            store(control->acknowledged, want);  // result bytes precede this publication
            ms_host += since(th0);
            if (e.counters) {
                ++e.cnt.br_pool;
                ++e.cnt.d2h_n;
                e.cnt.d2h_bytes += (int64_t) (N + K) * 4;
                ++e.cnt.h2d_n;
                e.cnt.h2d_bytes += (int64_t) K * N * 4;
            }
        }
    } catch (...) {
        store(control->abort, 1);
        KC(cudaStreamSynchronize(stream));
        throw;  // producer failure cannot leave a graph waiting on freed storage
    }
    if (aborted) store(control->abort, 1);
    const auto ts0 = Clock::now();
    KC(cudaStreamSynchronize(stream));  // token boundary, never a per-layer drain
    ms_sync = since(ts0);
    if (aborted) hb.expect = load(control->ready);  // cancelled graphs still drain their later publish nodes
    if (aborted || load(control->status) != DecodeWaitOk || load(control->ready) != hb.expect) {
        if (user_cancel) {
            std::fprintf(stderr, "kolibri: DEC_TOKEN_GRAPH cancelled: graph drained; token discarded\n");
            decode_aborted = true;
            return true;  // caller must reset KV/live prefix and emit no token for this step
        }
        std::fprintf(stderr, "kolibri: DEC_TOKEN_GRAPH failed (status %u, ready %u, expected %u); no token emitted\n",
                     load(control->status), load(control->ready), hb.expect);
        std::exit(1);
    }
    if (timing) {
        std::fprintf(stderr, "DEC_TIMING pos=%lld total=%.2f wait=%.2f host=%.2f launch=%.2f sync=%.2f ms\n",
                     (long long) s.pos, since(tc0), ms_wait, ms_host, ms_launch, ms_sync);
        std::fflush(stderr);
    }
    return true;
}
