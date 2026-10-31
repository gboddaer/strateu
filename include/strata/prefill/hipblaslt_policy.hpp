#pragma once
// hipblaslt_policy: the two plain decisions the HIP prefill makes about a tuned hipBLASLt solution - when a
// failed launch means the solution is given up on, and why the prompt MMQ path is not being used.  Pure, so
// both are testable without a GPU or hipBLASLt (src/prefill/hipblaslt_policy_test.cpp).

namespace strata::prefill::hipblaslt {

/// How many failed launches a tuned solution may have for one shape before it is given up on.  The same count
/// the rocBLAS solution table uses (gemm.cu): the first failure is often the library still setting the kernel
/// up, and dropping the tuned path on the first one costs the whole session its fast prompt path.
constexpr int kTriesBeforeDrop = 3;

/// True when the status says this solution cannot do this problem at all, so counting is pointless.  The values
/// are hipblasStatus_t: INVALID_VALUE 3, NOT_SUPPORTED 7, ARCH_MISMATCH 8, HANDLE_IS_NULLPTR 9, INVALID_ENUM 10.
/// Everything else (ALLOC_FAILED 2, MAPPING_ERROR 4, EXECUTION_FAILED 5, INTERNAL_ERROR 6, NOT_INITIALIZED 1,
/// UNKNOWN 11) says this launch failed, not that the solution is wrong for the shape.
inline bool failure_is_permanent(int status) {
    return status == 3 || status == 7 || status == 8 || status == 9 || status == 10;
}

/// True when the tuned solution is given up on for this shape after a launch failed with `status` and this is
/// its `failures`-th failure.  The default kernel runs for this call either way; this only decides whether the
/// next call tries the tuned solution again.
inline bool drop_solution_after_failure(int status, int failures) {
    return failure_is_permanent(status) || failures >= kTriesBeforeDrop;
}

/// Why the prompt MMQ path is not being used, in one plain clause, or nullptr when it can run.  The caller
/// reads the environment and asks the Gemm whether the shape is tuned, so this stays testable.
inline const char* mmq_block_reason(bool pf_pipe, bool dense_mmq, bool pf_gemm, bool wmma, bool shape_tuned) {
    if (pf_pipe) {
        return "STRATA_KOLIBRI_PF_PIPE is on";
    }
    if (dense_mmq) {
        return "STRATA_DENSE_MMQ is on";
    }
    if (pf_gemm) {
        return "STRATA_PF_GEMM is on";
    }
    if (wmma) {
        return "STRATA_WMMA_GEMM is on";
    }
    if (!shape_tuned) {
        return "no direct hipBLASLt tuning is usable for this shared-expert shape";
    }
    return nullptr;
}

}  // namespace strata::prefill::hipblaslt
