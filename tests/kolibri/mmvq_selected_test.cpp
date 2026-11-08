// tests/kolibri/mmvq_selected_test.cpp - exact multi-selection MMVQ (Q4_K, Q6_K), synthetic and bitwise.
//
//     build-kolibri-hip-tests/mmvq_selected_test
//
// Compares ONE native_mmvq_selected call with the same number of independent native_mmvq calls that read the SAME
// Q8_1 bytes.  The only thing under test is the selection dispatch.  Weights are random bytes with finite fp16
// scales (random scale bits can be inf/NaN, and every NaN compares equal on the GPU, which would hide a difference).
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using strata::kernels::native_mmvq;
using strata::kernels::native_mmvq_selected;
using strata::kernels::native_mmvq_weight_bytes;
using strata::kernels::native_q8_1_bytes;
using strata::kernels::native_quantize_q8_1;

struct Case {
    const char* name;
    int type;
    int n_in;
    int n_out;
    int block_elems;
    int block_bytes;
    int scale_at[2];
};

const Case CASES[] = {
    {"Q4_K smallK tail", 12, 1024, 37, 256, 144, {0, 2}},
    {"Q4_K largeK tail", 12, 2048, 37, 256, 144, {0, 2}},
    {"Q6_K smallK tail", 14, 512, 37, 256, 210, {208, -1}},
    {"Q6_K largeK tail", 14, 2048, 37, 256, 210, {208, -1}},
};

int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); ++failures; } } while (0)

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::printf("CUDA: %s: %s\n", what, cudaGetErrorString(e));
    ++failures;
    return false;
}

uint16_t sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}

bool all_finite(const std::vector<float>& v) {
    for (float f : v) if (!std::isfinite(f)) return false;
    return true;
}

// One case: selections distinct, one permutation, shared or per-selection activation bytes.
void run_case(const Case& c, int selections, bool permute, bool per_selection_x, bool same_weights = false) {
    const std::size_t stride = native_mmvq_weight_bytes(c.type, c.n_in, c.n_out);
    const std::size_t wbytes = stride * (same_weights ? 1 : (std::size_t) selections);
    const std::size_t qcol = native_q8_1_bytes(c.n_in, 1);
    const std::size_t qbytes = qcol * (per_selection_x ? (std::size_t) selections : 1);
    const std::size_t ybytes = (std::size_t) selections * c.n_out * sizeof(float);

    std::mt19937 rng(0x51ec7edU + (unsigned) c.type * 131U + (unsigned) selections + (per_selection_x ? 7U : 0U));
    std::vector<uint8_t> w(wbytes);
    for (auto& b : w) b = (uint8_t) (rng() & 0xffu);
    const std::size_t blocks_per_mat = (std::size_t) c.n_out * (std::size_t) (c.n_in / c.block_elems);
    for (std::size_t m = 0; m < (same_weights ? 1u : (std::size_t) selections); ++m)
        for (std::size_t k = 0; k < blocks_per_mat; ++k)
            for (int at : c.scale_at)
                if (at >= 0) {
                    const uint16_t h = sane_half(rng);
                    std::memcpy(&w[m * stride + k * (std::size_t) c.block_bytes + (std::size_t) at], &h, 2);
                }
    std::vector<int> slots((std::size_t) selections);
    for (int s = 0; s < selections; ++s)
        slots[(std::size_t) s] = same_weights ? 0 : (permute ? (s * 3 + 1) % selections : s);

    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((std::size_t) (per_selection_x ? selections : 1) * c.n_in);
    for (auto& v : x) v = nd(rng);

    void* dw = nullptr;
    void* dx = nullptr;
    void* xq = nullptr;
    float* y_ref = nullptr;
    float* y_sel = nullptr;
    bool ok = ck(cudaMalloc(&dw, wbytes), "malloc weights") &&
              ck(cudaMalloc(&dx, x.size() * sizeof(float)), "malloc x") &&
              ck(cudaMalloc(&xq, qbytes), "malloc q8") &&
              ck(cudaMalloc(&y_ref, ybytes), "malloc reference") &&
              ck(cudaMalloc(&y_sel, ybytes), "malloc selected");
    cudaStream_t stream = nullptr;
    ok = ok && ck(cudaStreamCreate(&stream), "stream");
    if (ok) {
        ok = ck(cudaMemcpy(dw, w.data(), wbytes, cudaMemcpyHostToDevice), "copy weights") &&
             ck(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "copy x");
    }
    if (ok) {
        native_quantize_q8_1((const float*) dx, xq, c.n_in, per_selection_x ? selections : 1, stream);
        for (int s = 0; s < selections; ++s) {
            const uint8_t* we = (const uint8_t*) dw + (std::size_t) slots[(std::size_t) s] * (same_weights ? 0 : stride);
            const void* xqs = (const uint8_t*) xq + (per_selection_x ? (std::size_t) s * qcol : 0);
            native_mmvq(c.type, we, xqs, y_ref + (std::size_t) s * c.n_out, c.n_in, c.n_out, 1, stream);
        }
        const bool launched = native_mmvq_selected(c.type, dw, same_weights ? 0 : stride, slots.data(), xq,
                                                   per_selection_x ? qcol : 0, y_sel, c.n_in, c.n_out,
                                                   selections, stream);
        CHECK(launched, "%s: selected API refused a supported call (S=%d)", c.name, selections);
    }
    if (ok && ck(cudaStreamSynchronize(stream), "sync")) {
        std::vector<float> ref(ybytes / sizeof(float)), sel(ybytes / sizeof(float));
        ck(cudaMemcpy(ref.data(), y_ref, ybytes, cudaMemcpyDeviceToHost), "read reference");
        ck(cudaMemcpy(sel.data(), y_sel, ybytes, cudaMemcpyDeviceToHost), "read selected");
        CHECK(all_finite(ref), "%s: the reference itself is non-finite", c.name);
        CHECK(all_finite(sel), "%s: selected output is non-finite", c.name);
        CHECK(std::memcmp(ref.data(), sel.data(), ybytes) == 0,
              "%s: selected outputs are not bit-identical (S=%d permute=%d per-selection-x=%d)",
              c.name, selections, int(permute), int(per_selection_x));
    }
    if (stream) cudaStreamDestroy(stream);
    cudaFree(dw); cudaFree(dx); cudaFree(xq); cudaFree(y_ref); cudaFree(y_sel);
}

// A refused call must launch nothing and leave the output untouched.
void refused_keeps_output(int type, int selections, std::size_t stride, const int* slots, std::size_t xq_stride,
                          const char* what) {
    const int n_in = 512, n_out = 16;
    const std::size_t ybytes = (std::size_t) selections * n_out * sizeof(float);
    std::vector<float> sentinel(ybytes / sizeof(float), 123.5f), seen(ybytes / sizeof(float));
    void* dw = nullptr; void* xq = nullptr; float* y = nullptr;
    bool ok = ck(cudaMalloc(&dw, 1 << 20), "malloc dummy weights") && ck(cudaMalloc(&xq, 1 << 20), "malloc dummy q8") &&
              ck(cudaMalloc(&y, ybytes), "malloc output");
    cudaStream_t stream = nullptr;
    ok = ok && ck(cudaStreamCreate(&stream), "stream");
    if (ok) {
        ok = ck(cudaMemcpy(y, sentinel.data(), ybytes, cudaMemcpyHostToDevice), "seed output");
        const bool launched = native_mmvq_selected(type, dw, stride, slots, xq, xq_stride, y, n_in, n_out, selections, stream);
        CHECK(!launched, "%s: the call was not refused", what);
        ok = ck(cudaStreamSynchronize(stream), "sync") &&
             ck(cudaMemcpy(seen.data(), y, ybytes, cudaMemcpyDeviceToHost), "read output");
        if (ok) CHECK(std::memcmp(seen.data(), sentinel.data(), ybytes) == 0, "%s: a refused call changed the output", what);
    }
    if (stream) cudaStreamDestroy(stream);
    cudaFree(dw); cudaFree(xq); cudaFree(y);
}

// A registered packed-Q6 table must refuse the selected GGUF-layout dispatch even when STRATA_Q6_PACKED is unset.
void packed_q6_refusal() {
    const int n_in = 1024, n_out = 2048, selections = 2;
    const std::size_t stride = native_mmvq_weight_bytes(14, n_in, n_out);
    const std::size_t qcol = native_q8_1_bytes(n_in, 1);
    const std::size_t ybytes = (std::size_t) selections * n_out * sizeof(float);
    std::mt19937 rng(77u);
    std::vector<uint8_t> w(stride);
    for (auto& b : w) b = (uint8_t) (rng() & 0xffu);
    const std::size_t blocks = (std::size_t) n_out * (std::size_t) (n_in / 256);
    for (std::size_t k = 0; k < blocks; ++k) {
        const uint16_t h = sane_half(rng);
        std::memcpy(&w[k * 210 + 208], &h, 2);
    }
    void* dw = nullptr; void* xq = nullptr; float* y = nullptr;
    bool ok = ck(cudaMalloc(&dw, 2 * stride), "malloc q6 weights") && ck(cudaMalloc(&xq, qcol), "malloc q6 q8") &&
              ck(cudaMalloc(&y, ybytes), "malloc q6 output");
    cudaStream_t stream = nullptr;
    ok = ok && ck(cudaStreamCreate(&stream), "q6 stream");
    if (ok) ok = ck(cudaMemcpy(dw, w.data(), stride, cudaMemcpyHostToDevice), "copy q6 weights") &&
                  ck(cudaMemcpy((uint8_t*) dw + stride, w.data(), stride, cudaMemcpyHostToDevice), "copy q6 sentinel");
    if (ok) {
        // Register ONLY the second matrix: the per-pointer loop must catch a later registered selection.
        const bool packed = strata::kernels::native_q6_k_pack((const uint8_t*) dw + stride, n_in, n_out,
                                                              "mmvq_selected_test");
        CHECK(packed, "could not register a packed-Q6 sentinel");
        std::vector<float> sentinel(ybytes / sizeof(float), -7.25f), seen(ybytes / sizeof(float));
        ck(cudaMemcpy(y, sentinel.data(), ybytes, cudaMemcpyHostToDevice), "seed q6 output");
        const int slots[2] = {0, 1};
        const bool launched = native_mmvq_selected(14, dw, stride, slots, xq, 0, y, n_in, n_out, selections, stream);
        CHECK(!launched, "a registered packed-Q6 table was not refused");
        ck(cudaStreamSynchronize(stream), "q6 sync");
        ck(cudaMemcpy(seen.data(), y, ybytes, cudaMemcpyDeviceToHost), "read q6 output");
        CHECK(std::memcmp(seen.data(), sentinel.data(), ybytes) == 0, "packed-Q6 refusal changed the output");
        strata::kernels::native_q6_k_unpack((const uint8_t*) dw + stride);
    }
    if (stream) cudaStreamDestroy(stream);
    cudaFree(dw); cudaFree(xq); cudaFree(y);
}

void expect_base_throw() {
    const int slots[2] = {0, 1};
    bool threw = false;
    try {
        native_mmvq_selected(12, (const void*) (uintptr_t) 0x1001, 4608, slots, (const void*) (uintptr_t) 0x2000,
                             0, (float*) (uintptr_t) 0x3000, 512, 16, 2, (void*) (uintptr_t) 0x4000);
    } catch (const std::invalid_argument&) {
        threw = true;
    } catch (...) {
    }
    CHECK(threw, "unaligned base: expected an invalid_argument");
}

void expect_throw(int type, std::size_t stride, const int* slots, int selections, const char* what) {
    bool threw = false;
    try {
        native_mmvq_selected(type, (const void*) (uintptr_t) 0x1000, stride, slots, (const void*) (uintptr_t) 0x2000,
                             0, (float*) (uintptr_t) 0x3000, 512, 16, selections, (void*) (uintptr_t) 0x4000);
    } catch (const std::invalid_argument&) {
        threw = true;
    } catch (...) {
    }
    CHECK(threw, "%s: expected an invalid_argument", what);
}

}  // namespace

int main(int argc, char** argv) {
    unsetenv("STRATA_Q6_PACKED");   // the sentinel must test the registry, not an inherited env flag
    if (argc > 1 && std::strcmp(argv[1], "--expect-refusal") == 0) {
        const int slots[2] = {0, 1};
        refused_keeps_output(12, 2, native_mmvq_weight_bytes(12, 512, 16), slots, 0, "gfx906 default wave");
        if (failures) {
            std::printf("mmvq_selected: %d failure(s)\n", failures);
            return 1;
        }
        std::printf("mmvq_selected: unsupported dispatch refused\n");
        return 0;
    }
    for (const Case& c : CASES) {
        for (int selections : {2, 3, 5, 8}) {
            run_case(c, selections, false, false);
            run_case(c, selections, true, false);
            run_case(c, selections, false, true);
        }
        run_case(c, 3, false, true, true);   // stride 0: every selection reads one matrix
    }

    const int slots2[2] = {0, 1};
    refused_keeps_output(13, 2, native_mmvq_weight_bytes(13, 512, 16), slots2, 0, "Q5_K unsupported format");
    refused_keeps_output(12, 1, native_mmvq_weight_bytes(12, 512, 16), slots2, 0, "one selection");
    {
        const int nine[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        refused_keeps_output(12, 9, native_mmvq_weight_bytes(12, 512, 16), nine, 0, "nine selections");
    }
    refused_keeps_output(12, 2, native_mmvq_weight_bytes(12, 512, 16), slots2,
                         native_q8_1_bytes(512, 1) + 1, "bad q8 stride");

    packed_q6_refusal();
    refused_keeps_output(12, 2, 2, slots2, 0, "unaligned derived table");
    expect_base_throw();

    const int negative[2] = {-1, 0};
    expect_throw(12, native_mmvq_weight_bytes(12, 512, 16), negative, 2, "negative slot");
    const int overflowing[2] = {0, 2};
    expect_throw(12, SIZE_MAX, overflowing, 2, "overflowing slot offset");

    if (failures) {
        std::printf("mmvq_selected: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("mmvq_selected: all cases bit-identical\n");
    return 0;
}
