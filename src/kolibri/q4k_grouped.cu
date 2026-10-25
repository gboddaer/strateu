// src/kolibri/q4k_grouped.cu - campaign II stage G3: the grouped Q4_K-from-VRAM GEMM.
//
// WHAT THIS ARM IS.  The blobs are read from VRAM as Q4_K and the product is taken on the tensor cores, which is
// what the stage asks for.  It is NOT yet the fused kernel: the weights are dequantized to an FP16 scratch in VRAM
// and then handed to the engine's own batched FP16 GEMM (Gemm::f16_batched), so the bytes of FP16 that the fused
// kernel would never write are written here.  That is the whole difference, and it is why this arm is expected to
// land well above the 2.6 ms gate: for the staged set it materialises ~309 MB of FP16 per projection per layer.
// The number it does land is measured (q4k_grouped_vram_parity --bench), not estimated, and it is recorded in the
// ledger with the rest.
//
// WHY THIS ARM IS WORTH HAVING ANYWAY.  It replaces a leg measured at 154 ms/layer (docs/strateu-campaign-report.md
// :42, :289), it is built entirely from primitives the engine already trusts, and it is the A/B arm the fused kernel
// will be compared against.  It also makes the G2 parity gate green by construction rather than by luck: this arm
// IS the reference composition, so any future divergence the fused kernel introduces shows up on that same gate.
//
// ALLOCATION.  The FP16 weight scratch and the device pointer arrays are cached across calls and grown on demand.
// Re-allocating per call would put cudaMalloc in the timed region and make the per-layer number meaningless.  The
// cache is keyed on total bytes; a caller that changes shape upward pays one realloc, steady state pays none.

#include "q4k_grouped.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "strata/kernels/dequant_bf16.hpp"
#include "strata/prefill/gemm.hpp"

namespace strata {
namespace kolibri {
namespace {

/// The scratch and pointer-array cache.  Single-threaded by contract (the engine calls this from one worker).
struct Cache {
    void* wf16 = nullptr;        ///< batch * N * K halfs
    void* warr = nullptr;        ///< device array of `batch` weight pointers (FP16)
    size_t wf16_bytes = 0;
    size_t warr_bytes = 0;
    strata::prefill::Gemm gemm;
    bool gemm_ready = false;
    void* gemm_stream = nullptr;
};

Cache& cache() {
    static Cache c;
    return c;
}

bool grow(void*& p, size_t& have, size_t need, const char* what) {
    if (have >= need) return true;
    if (p) cudaFree(p);
    const cudaError_t e = cudaMalloc(&p, need);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "q4k_grouped: cudaMalloc %zu for %s failed: %s\n", need, what, cudaGetErrorString(e));
        p = nullptr;
        have = 0;
        return false;
    }
    have = need;
    return true;
}

}  // namespace

int q4k_grouped_gemm_vram(const uint8_t* const* Warr, const uint16_t* const* Xarr, float* const* Yarr,
                          int64_t T, int64_t N, int64_t K, int batch, cudaStream_t st) {
    if (Warr == nullptr || Xarr == nullptr || Yarr == nullptr || batch <= 0 || T <= 0 || N <= 0 || K <= 0) {
        return 2;  // bad arguments
    }
    if (K % 256 != 0) {
        std::fprintf(stderr, "q4k_grouped: K=%lld is not a multiple of 256 (Q4_K block size)\n", (long long)K);
        return 2;
    }
    Cache& c = cache();
    const size_t wf16_bytes = size_t(batch) * size_t(N) * size_t(K) * 2;
    const size_t warr_bytes = size_t(batch) * sizeof(uint16_t*);
    if (!grow(c.wf16, c.wf16_bytes, wf16_bytes, "f16 weight scratch")) return 3;
    if (!grow(c.warr, c.warr_bytes, warr_bytes, "weight pointer array")) return 3;

    if (!c.gemm_ready || c.gemm_stream != (void*)st) {
        std::string err;
        if (!c.gemm.init((void*)st, N * K, err)) {
            std::fprintf(stderr, "q4k_grouped: gemm.init failed: %s\n", err.c_str());
            return 3;
        }
        c.gemm_ready = true;
        c.gemm_stream = (void*)st;
    }

    // 1. dequantize every blob in place from VRAM to the VRAM FP16 scratch (one kernel per expert, no host round
    //    trip - this is the part the fused kernel folds into its own loads).
    // Warr / Xarr / Yarr are DEVICE arrays of device pointers (the f16_batched convention).  Reading them from
    // host code is a host read of device memory - it segfaults, which is how this was caught.  Bring the pointer
    // arrays home once per call; they are `batch` words, not the payload.
    std::vector<const uint8_t*> h_wsrc(batch);
    std::vector<const uint16_t*> h_xsrc(batch);
    std::vector<float*> h_ysrc(batch);
    if (cudaMemcpy(h_wsrc.data(), Warr, size_t(batch) * sizeof(void*), cudaMemcpyDeviceToHost) != cudaSuccess) return 3;
    if (cudaMemcpy(h_xsrc.data(), Xarr, size_t(batch) * sizeof(void*), cudaMemcpyDeviceToHost) != cudaSuccess) return 3;
    if (cudaMemcpy(h_ysrc.data(), Yarr, size_t(batch) * sizeof(void*), cudaMemcpyDeviceToHost) != cudaSuccess) return 3;

    // 1. dequantize every blob from VRAM into the VRAM FP16 scratch (the part the fused kernel folds into its own
    //    loads), and build the FP16 pointer array cublas wants on the device.
    std::vector<const uint16_t*> h_w(batch);
    for (int b = 0; b < batch; ++b) {
        uint16_t* dst = (uint16_t*)((char*)c.wf16 + size_t(b) * size_t(N) * size_t(K) * 2);
        strata::kernels::dequant_f16(12, h_wsrc[b], 0, N, K, dst, st);
        h_w[b] = dst;
    }
    if (cudaMemcpyAsync(c.warr, h_w.data(), warr_bytes, cudaMemcpyHostToDevice, st) != cudaSuccess) return 3;


    // 2. the batched tensor-core product, same layout and accumulation as the reference arm.
    c.gemm.f16_batched((const uint16_t* const*)c.warr, Xarr, Yarr, T, N, K, batch);
    if (cudaGetLastError() != cudaSuccess) return 4;
    return 0;
}

}  // namespace kolibri
}  // namespace strata
