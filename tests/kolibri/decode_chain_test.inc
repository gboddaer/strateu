// Replay a dependent eight-layer publication/CPU-result chain, not just an isolated wait.
static int chain_test() {
    constexpr int N = 2560, K = 6, L = 8;
    DecodeWaitControl *h = nullptr, *d = nullptr;
    float *hx = nullptr, *dx = nullptr, *hy = nullptr, *hw = nullptr, *dw = nullptr;
    int32_t *hi = nullptr, *di = nullptr;
    CHECK(cudaHostAlloc((void**) &h, sizeof(*h), cudaHostAllocMapped));
    CHECK(cudaHostAlloc((void**) &hx, N * sizeof(float), cudaHostAllocMapped));
    CHECK(cudaHostAlloc((void**) &hy, N * sizeof(float), cudaHostAllocMapped));
    CHECK(cudaHostAlloc((void**) &hw, K * sizeof(float), cudaHostAllocMapped));
    CHECK(cudaHostAlloc((void**) &hi, K * sizeof(int32_t), cudaHostAllocMapped));
    CHECK(cudaHostGetDevicePointer((void**) &d, h, 0));
    float *mapped_x = nullptr, *mapped_w = nullptr;
    int32_t* mapped_i = nullptr;
    CHECK(cudaHostGetDevicePointer((void**) &mapped_x, hx, 0));
    CHECK(cudaHostGetDevicePointer((void**) &mapped_w, hw, 0));
    CHECK(cudaHostGetDevicePointer((void**) &mapped_i, hi, 0));
    CHECK(cudaMalloc((void**) &dx, N * sizeof(float)));
    CHECK(cudaMalloc((void**) &dw, L * K * sizeof(float)));
    CHECK(cudaMalloc((void**) &di, L * K * sizeof(int32_t)));
    cudaStream_t stream{};
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaGraph_t graphs[2]{};
    cudaGraphExec_t execs[2]{};
    for (int mode = 0; mode < 2; ++mode) {
        CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (int l = 0; l < L; ++l) {
            if (mode == 0 || l != 0)
                strata::kernels::doorbell_publish(dx, di + l * K, dw + l * K, N, K,
                    mapped_x, mapped_i, mapped_w, &d->ready, stream);
            enqueue_decode_wait(d, 100000000ULL, stream);
            // The next publication depends on this layer's CPU result, as in the model residual chain.
            CHECK(cudaMemcpyAsync(dx, hy, N * sizeof(float), cudaMemcpyHostToDevice, stream));
        }
        CHECK(cudaStreamEndCapture(stream, &graphs[mode]));
        CHECK(cudaGraphInstantiate(&execs[mode], graphs[mode], 0));
    }
    auto load = decode_control_load;
    std::vector<float> input(N), result(N), weights(L * K);
    std::vector<int32_t> ids(L * K);
    for (uint32_t replay = 0; replay < 38; ++replay) {
        for (int i = 0; i < N; ++i) input[i] = float(replay * N + i);
        for (int i = 0; i < L * K; ++i) {
            ids[i] = int32_t(replay * 100 + i);
            weights[i] = float(i + 1) / 8.0f + float(replay) / 64.0f;
        }
        CHECK(cudaMemcpy(dx, input.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(di, ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dw, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice));
        *h = DecodeWaitControl{};
        h->generation = replay + 1;
        const bool failure = (replay >= 32 && replay <= 34) || replay == 36;
        CHECK(cudaGraphLaunch(execs[replay == 36 ? 1 : 0], stream));
        try {
            for (int l = 0; l < L; ++l) {
                const uint32_t want = uint32_t(l + 1);
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                while (load(h->ready) < want && !load(h->abort)) {
                    if (std::chrono::steady_clock::now() >= deadline) {
                        store(h->abort, 1);
                        std::fprintf(stderr, "FAIL chain host deadline\n");
                        CHECK(cudaStreamSynchronize(stream));
                        return 1;
                    }
                    std::this_thread::yield();
                }
                if (load(h->abort)) break;
                if (load(h->ready) != want) {
                    store(h->abort, 1);
                    CHECK(cudaStreamSynchronize(stream));
                    std::fprintf(stderr, "FAIL chain advanced without exact ACK\n");
                    return 1;
                }
                if (failure && l == 3) {
                    if (replay == 33) store(h->abort, 1);  // cancellation during a wait
                    if (replay == 34) throw std::runtime_error("simulated CPU producer failure");
                    break;  // replay 32 deliberately withholds ACK, without publishing abort
                }
                const float offset = float(l * (l + 1) / 2);
                for (int i = 0; i < N; ++i) {
                    if (hx[i] != input[i] + offset) {
                        store(h->abort, 1);
                        CHECK(cudaStreamSynchronize(stream));
                        std::fprintf(stderr, "FAIL stale chain activation\n");
                        return 1;
                    }
                    hy[i] = hx[i] + float(l + 1);
                }
                for (int k = 0; k < K; ++k) {
                    if (hi[k] != ids[l * K + k] || hw[k] != weights[l * K + k]) {
                        store(h->abort, 1);
                        CHECK(cudaStreamSynchronize(stream));
                        std::fprintf(stderr, "FAIL stale chain routing\n");
                        return 1;
                    }
                }
                if (l == 0 && !failure) {
                    store(h->acknowledged_generation, h->generation);
                    store(h->acknowledged, want + 1);  // future ACK
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    if (load(h->ready) != want) return 1;
                    store(h->acknowledged_generation, h->generation - 1);  // previous replay
                    store(h->acknowledged, want);
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    if (load(h->ready) != want) return 1;
                }
                store(h->acknowledged_generation, h->generation);
                store(h->acknowledged, want);
            }
        } catch (const std::runtime_error&) {
            store(h->abort, 1);
        }
        CHECK(cudaStreamSynchronize(stream));
        const uint32_t expected = replay == 32 || replay == 36 ? DecodeWaitTimedOut : DecodeWaitAborted;
        if (failure) {
            if (h->status != expected) return 1;
        } else {
            if (h->status != DecodeWaitOk || h->ready != L) return 1;
            CHECK(cudaMemcpy(result.data(), dx, N * sizeof(float), cudaMemcpyDeviceToHost));
            for (int i = 0; i < N; ++i) if (result[i] != input[i] + float(L * (L + 1) / 2)) return 1;
        }
    }
    std::puts("PASS dependent eight-layer chain: changing activations/routing, future/stale ACK rejection, "
              "missing publication/ACK, cancel/producer failure, reset after abort");
    for (int mode = 0; mode < 2; ++mode) {
        CHECK(cudaGraphExecDestroy(execs[mode]));
        CHECK(cudaGraphDestroy(graphs[mode]));
    }
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(dx)); CHECK(cudaFree(dw)); CHECK(cudaFree(di));
    CHECK(cudaFreeHost(h)); CHECK(cudaFreeHost(hx)); CHECK(cudaFreeHost(hy));
    CHECK(cudaFreeHost(hw)); CHECK(cudaFreeHost(hi));
    return 0;
}
