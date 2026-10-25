// src/kolibri/q4k_grouped.hpp - the grouped Q4_K-from-VRAM GEMM (campaign II stage G3).
//
// WHAT THIS IS FOR.  In the staged prefill the expert weights are Q4_K blobs that arrive in system RAM and are
// read on the CPU (the pool: native_gu_rows / native_down_rows).  Stage G3's kernel reads those SAME blobs
// directly from VRAM and does the grouped product on the GPU, so a promoted expert costs one plain H2D of the
// blob and no CPU work at all.  Today's CPU path stays the A/B arm behind an env knob.
//
// SHAPE (identical to Gemm::f16_batched, include/strata/prefill/gemm.hpp:47, so the two are drop-in arms):
//
//     Y_i[T, N] = X_i[T, K] . W_i[N, K]^T      row-major, fp32 out, fp32 accumulate
//
// with `batch` independent products of the same shape, and W_i given as Q4_K blocks (GGML type 12, 144 bytes
// per 256 values, so K must be a multiple of 256) instead of FP16.  The f16_batched reference for the same
// weights is dequant_f16(12, ...) followed by f16_batched - which is exactly what the parity test does.
//
// Warr / Xarr / Yarr are DEVICE arrays of DEVICE pointers, same rule as f16_batched (cublas takes device
// pointer arrays; a host array would be read from the wrong address space).
//
// Returns 0 on success.  A non-zero return means the product was NOT computed and the caller must not read Y.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace strata {
namespace kolibri {

/// Y_i[T,N] = X_i[T,K] . W_i[N,K]^T for `batch` experts, W_i read as Q4_K blocks from VRAM.
/// Returns 0 on success, non-zero if the product was not computed (see the red-phase note in q4k_grouped.cu).
int q4k_grouped_gemm_vram(const uint8_t* const* Warr, const uint16_t* const* Xarr, float* const* Yarr,
                          int64_t T, int64_t N, int64_t K, int batch, cudaStream_t st);

}  // namespace kolibri
}  // namespace strata
