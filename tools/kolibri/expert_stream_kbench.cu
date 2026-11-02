// Standalone kernel bench for the streamed expert products (expert_stream_kernels.cuh).
//
// One launch's throughput at the model's own geometries, so a kernel change can be judged before it is taken to the
// engine.  It is not part of any engine path and changes nothing.
//
//   cmake --build build-sram --target kolibri_expert_stream_kbench -j16
//   numactl --cpunodebind=1 --membind=1 ./build-sram/kolibri_expert_stream_kbench 40
//
// The switches (SMEM, GU, PLANES) are read per launch, so every mode is timed in one process on the same data.
// The table is the minimum of `reps` launches per case.
#include "kolibri/expert_stream_kernels.cuh"

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <random>
#include <vector>

using namespace strata::kolibri::stream;

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

struct Case {
    const char* name;
    int type;    ///< 12 Q4_K, 14 Q6_K
    int nb;      ///< blocks per row
    int rows;    ///< output rows of this launch
    int nt;      ///< tokens (selections of one group)
    int gu;      ///< 1 = the paired gate/up entry (rows = n_ff, writes 2 * rows), 0 = the plain row range
};

// Kolibri-1-Q4_K_M: n_embd 2560, n_ff 512.  gate/up = one launch of 2*n_ff rows over n_embd/256 blocks; the down
// projection is n_embd rows over n_ff/256 blocks, Q4_K on some layers and Q6_K on others.
static const Case kCases[] = {
    {"q4k gate/up  nt=1 ", 12, 10, 1024, 1, 0},  {"q4k gate/up  nt=4 ", 12, 10, 1024, 4, 0},
    {"q4k gate/up  nt=8 ", 12, 10, 1024, 8, 0},  {"q4k gate/up  nt=16", 12, 10, 1024, 16, 0},
    {"q4k gate/up  nt=32", 12, 10, 1024, 32, 0}, {"q4k gu-pair  nt=1 ", 12, 10, 512, 1, 1},
    {"q4k gu-pair  nt=4 ", 12, 10, 512, 4, 1},   {"q4k gu-pair  nt=8 ", 12, 10, 512, 8, 1},
    {"q4k gu-pair  nt=16", 12, 10, 512, 16, 1},  {"q4k gu-pair  nt=32", 12, 10, 512, 32, 1},
    {"q6k down     nt=1 ", 14, 2, 2560, 1, 0},   {"q6k down     nt=4 ", 14, 2, 2560, 4, 0},
    {"q6k down     nt=8 ", 14, 2, 2560, 8, 0},   {"q6k down     nt=16", 14, 2, 2560, 16, 0},
    {"q4k down     nt=8 ", 12, 2, 2560, 8, 0},
};

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 40;
    constexpr int kActBytes = 10 * 292;   // the widest Q8_K row this model uses (n_embd = 2560)
    std::mt19937 rng(1234);
    cudaDeviceProp p{};
    ck(cudaGetDeviceProperties(&p, 0), "props");
    std::printf("device: %s, sm_%d%d, %d SMs\n", p.name, p.major, p.minor, p.multiProcessorCount);

    for (const Case& c : kCases) {
        const size_t block = c.type == 12 ? 144 : 210;
        const size_t wbytes = (size_t) (c.gu ? 2 * c.rows : c.rows) * c.nb * block;
        std::vector<uint8_t> hw(wbytes), ha((size_t) c.nt * kActBytes);
        for (auto& b : hw) b = (uint8_t) (rng() & 0xff);
        for (auto& b : ha) b = (uint8_t) (rng() & 0xff);
        // keep the fp16 d fields in a sane range so nothing goes inf/nan
        for (size_t i = 0; i + 1 < wbytes; i += block) {
            hw[i] = 0x00;
            hw[i + 1] = 0x24;
        }
        uint8_t *dw = nullptr, *da = nullptr;
        float* dout = nullptr;
        ck(cudaMalloc(&dw, wbytes), "malloc w");
        ck(cudaMalloc(&da, ha.size()), "malloc a");
        ck(cudaMalloc(&dout, (size_t) c.nt * (c.gu ? 2 * c.rows : c.rows) * sizeof(float)), "malloc out");
        ck(cudaMemcpy(dw, hw.data(), wbytes, cudaMemcpyHostToDevice), "copy w");
        ck(cudaMemcpy(da, ha.data(), ha.size(), cudaMemcpyHostToDevice), "copy a");

        double best[2] = {1e30, 1e30};
        for (int m = 0; m < 2; ++m) {
            setenv("STRATA_KOLIBRI_STREAM_SMEM", m ? "1" : "0", 1);
            for (int r = 0; r < reps; ++r) {
                cudaEvent_t a, b;
                ck(cudaEventCreate(&a), "ev");
                ck(cudaEventCreate(&b), "ev");
                ck(cudaEventRecord(a), "rec");
                setenv("STRATA_KOLIBRI_STREAM_GU", c.gu ? "1" : "0", 1);
                const int rc = c.gu ? es_q4k_gu_rows(dw, c.nb, c.rows, da, kActBytes, c.nt, dout, 2 * c.rows, nullptr)
                                    : (c.type == 12 ? es_q4k_rows(dw, c.nb, c.rows, da, kActBytes, c.nt, dout, c.rows,
                                                                 nullptr)
                                                    : es_q6k_rows(dw, c.nb, c.rows, da, kActBytes, c.nt, dout,
                                                                  c.rows, nullptr));
                if (rc != 0) {
                    std::fprintf(stderr, "refused: %s\n", c.name);
                    return 1;
                }
                ck(cudaEventRecord(b), "rec");
                ck(cudaEventSynchronize(b), "sync");
                float ms = 0.f;
                ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
                if (ms < best[m]) best[m] = ms;
                ck(cudaEventDestroy(a), "ev");
                ck(cudaEventDestroy(b), "ev");
            }
        }
        const double flops = 2.0 * c.rows * c.nb * 256.0 * c.nt;   // MACs x 2
        const double gp0 = flops / (best[0] * 1e6), gp1 = flops / (best[1] * 1e6);
        std::printf("%s  nt=%-2d direct %8.3f ms %6.1f GFLOP/s | smem %8.3f ms %6.1f GFLOP/s | smem/direct %.3f\n",
                    c.name, c.nt, best[0], gp0, best[1], gp1, best[1] / best[0]);
        cudaFree(dw);
        cudaFree(da);
        cudaFree(dout);
    }
    // Q6_K: the block-interleaved blob against the plane-split copy of the same bytes (nb 2, 2560 rows)
    for (int nt : {1, 4, 8, 16, 64}) {
        const int nb = 2, rows = 2560;
        const size_t n = (size_t) rows * nb, wbytes = n * 210;
        std::vector<uint8_t> hw(wbytes);
        for (auto& b : hw) b = (uint8_t) (rng() & 0xff);
        for (size_t i = 0; i + 1 < wbytes; i += 210) { hw[i] = 0x00; hw[i + 1] = 0x24; }
        std::vector<uint8_t> ha((size_t) nt * kActBytes);
        for (auto& b : ha) b = (uint8_t) (rng() & 0xff);
        uint8_t *dw = nullptr, *dp = nullptr, *da = nullptr;
        float* dout = nullptr;
        ck(cudaMalloc(&dw, wbytes), "m");
        ck(cudaMalloc(&dp, wbytes), "m");
        ck(cudaMalloc(&da, ha.size()), "m");
        ck(cudaMalloc(&dout, (size_t) nt * rows * sizeof(float)), "m");
        ck(cudaMemcpy(dw, hw.data(), wbytes, cudaMemcpyHostToDevice), "c");
        ck(cudaMemcpy(da, ha.data(), ha.size(), cudaMemcpyHostToDevice), "c");
        if (es_q6k_pack_planes(dw, dp, nb, rows, nullptr) != 0) return 1;
        ck(cudaDeviceSynchronize(), "sync");
        double best[2] = {1e30, 1e30};
        for (int m = 0; m < 2; ++m) {
            for (int r = 0; r < reps; ++r) {
                cudaEvent_t a, b;
                ck(cudaEventCreate(&a), "ev");
                ck(cudaEventCreate(&b), "ev");
                ck(cudaEventRecord(a), "rec");
                const int rc = m ? es_q6k_plane_rows(dp, nb, rows, da, kActBytes, nt, dout, rows, nullptr)
                                 : es_q6k_rows(dw, nb, rows, da, kActBytes, nt, dout, rows, nullptr);
                if (rc != 0) return 1;
                ck(cudaEventRecord(b), "rec");
                ck(cudaEventSynchronize(b), "sync");
                float ms = 0.f;
                ck(cudaEventElapsedTime(&ms, a, b), "el");
                if (ms < best[m]) best[m] = ms;
                ck(cudaEventDestroy(a), "ev");
                ck(cudaEventDestroy(b), "ev");
            }
        }
        const double flops = 2.0 * rows * nb * 256.0 * nt;
        std::printf("q6k down %-6s nt=%-2d interleaved %8.3f ms %6.1f GFLOP/s | planes %8.3f ms %6.1f GFLOP/s | planes/interleaved %.3f\n",
                    "planes", nt, best[0], flops / (best[0] * 1e6), best[1], flops / (best[1] * 1e6), best[1] / best[0]);
        cudaFree(dw);
        cudaFree(dp);
        cudaFree(da);
        cudaFree(dout);
    }
    return 0;
}
