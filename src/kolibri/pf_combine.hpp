// src/kolibri/pf_combine.hpp - the prefill MoE combiner (plan
// docs/superpowers/plans/2026-10-24-kolibri-pf-partial-sums.md Task 1).
//
// Under STRATA_KOLIBRI_PF_PARTIAL the chunked prefill stops round-tripping resident-expert rows through host
// RAM.  Per chunk-layer the MoE output is then the fold of three things that live in three places:
//
//   sy    - the shared-expert row, already on device (Tc x N)
//   p     - the CPU pool's weighted partial over the MISS picks, one H2D of Tc x N instead of the whole
//           NKc x N block that goes up today (5.2 MB per chunk-layer instead of 31.5 MB)
//   rr    - the RESIDENT rows, left on device in block order (rrows x N), mapped to (token, pick) by
//           rr_sel[j] = t*K + k, or -1 for a padded block row
//
//     moe[t*N+i] = sy[t*N+i] + p[t*N+i] + sum_{j : rr_sel[j] = t*K+k} wts[rr_sel[j]] * rr[j*N+i]
//
// which is the engine's own definition (kolibri_engine.cu:1086 `k_moe_sum_rows`) with the miss picks folded
// into `p` on the host and the resident picks arriving in block order instead of pick order.  The router
// weights keep their meaning: sigmoid, un-renormalised (docs/KOLIBRI1.md).
//
// INTERFACE NOTE (correction to the plan, found while implementing): the plan's signature has no `Tc`, and the
// launcher cannot size grid.x without it.  `Tc` is therefore an explicit parameter here, ahead of `rrows`.
//
// `moe`, `sy`, `p`, `rr`, `rr_sel` and `wts` are all DEVICE pointers.  `wts` is the Tc x K router-weight array.
// Nothing here allocates; the caller owns every buffer and its lifetime (see the stream-ordering rules in the
// plan: everything this kernel reads must be ordered on `st`).

#pragma once

#include <cuda_runtime.h>

namespace strata {
namespace kolibri {

/// moe[t*N+i] = sy[t*N+i] + p[t*N+i] + sum over the block rows j belonging to token t of
/// wts[rr_sel[j]] * rr[j*N+i], skipping rr_sel[j] < 0.  Launches on `st`.
/// rrows may be 0 (nothing resident); then this is exactly sy + p, bitwise.
void pf_combine_rows(float* moe, const float* sy, const float* p, const float* rr, const int* rr_sel,
                     const float* wts, int Tc, int rrows, int K, int N, cudaStream_t st);

}  // namespace kolibri
}  // namespace strata
