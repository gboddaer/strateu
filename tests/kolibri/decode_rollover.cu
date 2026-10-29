// Test-only TU: real engine with descriptor seeding and narrowly injected external failures.
// No fault-injection knobs or test methods are added to production.
#include <cuda_runtime.h>
#include "strata/kernels/cpu/native_expert.hpp"
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stdexcept>

static bool test_mode(const char* mode) {
    const char* value = std::getenv("KOLIBRI_TEST_FAILURE");
    return value && std::strcmp(value, mode) == 0;
}
static cudaError_t test_host_alloc(void** ptr, size_t bytes, unsigned int flags) {
    static int calls = 0;
    if (++calls == 3 && test_mode("mapped")) {
        std::fprintf(stderr, "TEST mapped allocation failure\n");
        return cudaErrorMemoryAllocation;
    }
    return cudaHostAlloc(ptr, bytes, flags);
}
static cudaError_t test_graph_instantiate(cudaGraphExec_t* exec, cudaGraph_t graph, unsigned long long flags) {
    if (test_mode("parent")) {
        size_t nodes = 0;
#if defined(__HIPCC__)
        const auto status = hipGraphGetNodes(graph, nullptr, &nodes);
#else
        const auto status = cudaGraphGetNodes(graph, nullptr, &nodes);
#endif
        if (status != cudaSuccess) return status;
        if (nodes == 150) {  // 50 layers x (PRE child, ACK-wait child, POST child)
            std::fprintf(stderr, "TEST parent executable allocation failure\n");
            return cudaErrorMemoryAllocation;
        }
    }
    return cudaGraphInstantiate(exec, graph, flags);
}
static bool test_parent_active();
namespace strata::kernels::cpu {
static void test_quant_act(const NativeFmt& format, const float* input, void* output) {
    if (test_mode("producer") && test_parent_active())
        throw std::runtime_error("TEST CPU producer failure after parent publication");
    native_quant_act(format, input, output);
}
}
#undef cudaHostAlloc
#define cudaHostAlloc test_host_alloc
#define cudaGraphInstantiate test_graph_instantiate
#define native_quant_act test_quant_act
#include "../../src/kolibri/kolibri_engine.cu"
#undef cudaHostAlloc
#undef cudaGraphInstantiate
#undef native_quant_act

static bool test_parent_active() {
    const auto* control = handoff_buffers(2560, 6)->host_control;
    return control->generation != 0 && control->generation != UINT32_MAX;
}
int main(int argc, char** argv) {
    auto& buffers = *handoff_buffers(2560, 6);
    buffers.expect = UINT32_MAX - 25;
    buffers.host_control->ready = buffers.expect;
    buffers.host_control->generation = UINT32_MAX;
    try {
        const int result = strata::kolibri::run(argc, argv);
        if (result != 0) return result;
        // Nine forced forwards through 50 layers. Reset must precede the FIRST publication,
        // even though cold segments are not yet eligible for the parent executable.
        if (buffers.expect != 450 || buffers.host_control->ready != 450 ||
            buffers.host_control->generation == 0 || buffers.host_control->generation == UINT32_MAX) {
            std::fprintf(stderr, "FAIL rollover: expected 450 publications after cold-boundary reset, got %u/%u\n",
                         buffers.expect, buffers.host_control->ready);
            return 1;
        }
        std::fprintf(stderr, "PASS cold sequence rollover and generation rollover\n");
        return 0;
    } catch (const std::exception& error) {
        // A query, NOT a drain: the real engine catch must already have aborted and drained the graph.
        if (test_mode("producer") && cudaStreamQuery(dec_stream()) == cudaSuccess &&
            buffers.host_control->status == strata::kolibri::DecodeWaitAborted)
            std::fprintf(stderr, "PASS CPU producer failure graph already quiescent\n");
        std::fprintf(stderr, "rollover test: %s\n", error.what());
        return 1;
    }
}
