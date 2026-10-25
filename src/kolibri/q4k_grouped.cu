// src/kolibri/q4k_grouped.cu - campaign II stage G3: the grouped Q4_K-from-VRAM GEMM.
//
// RED-PHASE STUB (stage G2).  The parity test `src/kernels/q4k_grouped_vram_parity.cpp` was written and
// committed BEFORE this kernel existed, as the campaign's measurement discipline requires: the test must be
// shown failing against nothing before it is allowed to pass against something.
//
// The stub deliberately does NOT write Y and returns 1, so the test computes real numbers against
// uninitialised output and fails with a measured magnitude rather than a compile error.  A compile-failing
// target was rejected on purpose: `cmake --build build` aborts the whole tree on one bad target (the tree
// already has that defect in attn_k_parity, CMakeLists.txt:908), which would take the engine build down with
// it between the two commits.
//
// G3 replaces the body with the real kernel and returns 0; the parity test is the gate it has to clear.

#include "q4k_grouped.hpp"

namespace strata {
namespace kolibri {

int q4k_grouped_gemm_vram(const uint8_t* const* /*Warr*/, const uint16_t* const* /*Xarr*/,
                          float* const* /*Yarr*/, int64_t /*T*/, int64_t /*N*/, int64_t /*K*/, int /*batch*/,
                          cudaStream_t /*st*/) {
    return 1;  // not implemented yet (G2 red phase)
}

}  // namespace kolibri
}  // namespace strata
