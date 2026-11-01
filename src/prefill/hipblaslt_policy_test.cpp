// hipblaslt_policy_test: when a tuned hipBLASLt solution is given up on, and what the engine says when the
// prompt MMQ path is not used.  Pure: no GPU, no hipBLASLt, no engine.
#include "strata/prefill/hipblaslt_policy.hpp"

#include <cstdio>
#include <string_view>

using strata::prefill::hipblaslt::drop_solution_after_failure;
using strata::prefill::hipblaslt::failure_is_permanent;
using strata::prefill::hipblaslt::mmq_block_reason;

static int g_fail = 0;
static void expect(bool ok, const char* what) {
    if (!ok) {
        ++g_fail;
        std::fprintf(stderr, "FAIL: %s\n", what);
    }
}
static bool says(const char* reason, const char* needle) {
    return reason && std::string_view(reason).find(needle) != std::string_view::npos;
}

int main() {
    // One failed launch does not prove the solution is broken.  The first one can be the library still setting
    // the kernel up (the rocBLAS path waits for three for exactly this reason), and dropping the tuned path on
    // the first failure costs the rest of the session its fast prompt path: a measured R9700 serve logged
    // "hipBLASLt launch failed with status 6" three times, then "MMQ falling back to the pool", and never used
    // the path again although the same startup probe had just verified it.
    expect(!drop_solution_after_failure(6, 1), "INTERNAL_ERROR once: keep the solution, default kernel this time");
    expect(!drop_solution_after_failure(6, 2), "INTERNAL_ERROR twice: still keep it");
    expect(drop_solution_after_failure(6, 3), "INTERNAL_ERROR three times: drop it for this shape");
    expect(drop_solution_after_failure(5, 3), "EXECUTION_FAILED three times: drop it");
    expect(!drop_solution_after_failure(2, 1), "ALLOC_FAILED once: keep it");

    // A status that says the solution cannot do this problem is decided at once, without counting.
    expect(failure_is_permanent(7), "NOT_SUPPORTED is permanent");
    expect(failure_is_permanent(8), "ARCH_MISMATCH is permanent");
    expect(failure_is_permanent(3), "INVALID_VALUE is permanent");
    expect(!failure_is_permanent(6), "INTERNAL_ERROR is not permanent");
    expect(drop_solution_after_failure(7, 1), "NOT_SUPPORTED drops on the first failure");

    // The engine has to name the reason it is not on the MMQ path.  The old sentence named PF_PIPE and the
    // dense overrides even when the user had set neither, which is what made a real fallback undiagnosable.
    expect(mmq_block_reason(false, false, false, false, true) == nullptr, "nothing blocks it: no reason");
    expect(says(mmq_block_reason(true, false, false, false, true), "PF_PIPE"), "PF_PIPE names PF_PIPE");
    expect(says(mmq_block_reason(false, true, false, false, true), "STRATA_DENSE_MMQ"), "the dense override is named");
    expect(says(mmq_block_reason(false, false, true, false, true), "STRATA_PF_GEMM"), "the prefill gemm override is named");
    expect(says(mmq_block_reason(false, false, false, true, true), "STRATA_WMMA_GEMM"), "the WMMA override is named");
    expect(says(mmq_block_reason(false, false, false, false, false), "tuning"),
           "an untuned shape says the direct tuning is not usable");
    expect(!says(mmq_block_reason(false, false, false, false, false), "PF_PIPE"),
           "an untuned shape does not blame PF_PIPE");

    if (!g_fail) std::printf("hipblaslt_policy_test: ok\n");
    return g_fail ? 1 : 0;
}
