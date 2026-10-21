// src/kolibri/attn_k_parity.cu - the decode attention's UNCOALESCED score pass, tested as a LAYOUT change.
//
// THE PROBLEM.  k_attn_body's score dot gives one thread one K row, and the thread reads the whole 128-dim row
// (256 B) for `s += qh[d] * __half2float(kp[d])`: a warp instruction touches 32 separate sectors.  That pass is
// 91% of the kernel (10.98 of 12.05 ms/launch in the ledger; ~295 us/layer on the T4) against a 32.5 TFLOPS fp32
// peak - a load-shape problem, not a math one.
//
// WHY THE EXISTING FIX CANNOT SHIP.  STRATA_KOLIBRI_ATTN_COALESCED makes the dot warp-cooperative (lane l takes
// 4 dims, then a shuffle reduce).  That changes the accumulation ORDER, and it moved the oracle's ids when tried.
// The plain path must stay bitwise.
//
// THE ROUTE THIS TEST OPENS.  Store K TRANSPOSED ("[d][row]" per KV head).  A warp reading the same `d` across 32
// consecutive rows is then contiguous, while every thread keeps the untouched sequential d = 0..127 chain -
// coalesced by layout, bitwise by construction.  Cost: K's layout touches the store kernel, the sliding-window
// ring index (gp % swa) and the GQA sharing (HKV); V stays row-major (its read is already the coalesced side).
//
// HOW IT IS CHECKED (and why it is written this way).  ONE kernel computes both layouts for the same row and
// writes them side by side: a first version used separate kernels into separate buffers and read uninitialised
// memory - the same float appeared in unrelated slots across runs, and a parity test that can return stale bytes
// is worse than none.  The comparison here cannot be confounded by launch order or buffer reuse.
// GPU, synthetic, no model.  Exit 0 = every case bitwise in at least one form.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define CK(x)                                                                                          \
    do {                                                                                               \
        const cudaError_t _e = (x);                                                                    \
        if (_e != cudaSuccess) {                                                                       \
            std::fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(_e));   \
            return 1;                                                                                  \
        }                                                                                              \
    } while (0)

// Both layouts, same row, side by side: out[2p] = row-major dot, out[2p+1] = transposed dot.
// `use_fma` selects an explicit operation so the comparison does not depend on what the compiler contracts.
__global__ void k_both(const float* __restrict__ q, const __half* __restrict__ kc, const __half* __restrict__ kt,
                       float* __restrict__ out, int D, int H, int HKV, int rows, int n_pos, int first, int swa,
                       bool sliding, float scale, int use_fma) {
    const int h = blockIdx.x, kvh = h / (H / HKV);
    const float* qh = q + (size_t) h * D;
    const __half* base = kt + (size_t) kvh * D * rows;
    for (int p = threadIdx.x; p < n_pos; p += blockDim.x) {
        const int gp = first + p;
        const int slot = sliding ? (gp % swa) : gp;
        const __half* kp = kc + ((size_t) kvh * rows + slot) * D;   // row-major [kvh][row][d]
        float s_major = 0.f, s_trans = 0.f;
        if (use_fma) {
            for (int d = 0; d < D; ++d) s_major = __fmaf_rn(qh[d], __half2float(kp[d]), s_major);
            for (int d = 0; d < D; ++d)
                s_trans = __fmaf_rn(qh[d], __half2float(base[(size_t) d * rows + slot]), s_trans);
        } else {
            for (int d = 0; d < D; ++d) s_major = s_major + qh[d] * __half2float(kp[d]);
            for (int d = 0; d < D; ++d) s_trans = s_trans + qh[d] * __half2float(base[(size_t) d * rows + slot]);
        }
        out[2 * p + 0] = s_major * scale;
        out[2 * p + 1] = s_trans * scale;
    }
}

// The variant that must NOT match: warp-cooperative on the transposed layout (lane l takes 4 dims, shuffle tree).
// Contiguous and fast, different summation order - this is why that path stays env-gated.
__global__ void k_warp(const float* __restrict__ q, const __half* __restrict__ kt, float* __restrict__ sc,
                       int D, int H, int HKV, int rows, int n_pos, int first, int swa, bool sliding, float scale) {
    const int h = blockIdx.x, kvh = h / (H / HKV);
    const float* qh = q + (size_t) h * D;
    const __half* base = kt + (size_t) kvh * D * rows;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nwarp = (int) blockDim.x >> 5;
    for (int row = wid; row < n_pos; row += nwarp) {
        const int gp = first + row;
        const int slot = sliding ? (gp % swa) : gp;
        float part = 0.f;
        for (int d = lane * 4; d < (lane + 1) * 4 && d < D; ++d)
            part = __fmaf_rn(qh[d], __half2float(base[(size_t) d * rows + slot]), part);
        for (int off = 16; off > 0; off >>= 1) part += __shfl_down_sync(0xffffffffu, part, off);
        if (lane == 0) sc[row] = part * scale;
    }
}

// Bench only: one layout per launch, so the two rates are directly comparable.
__global__ void k_one(const float* __restrict__ q, const __half* __restrict__ kc, const __half* __restrict__ kt,
                      float* __restrict__ sc, int D, int H, int HKV, int rows, int n_pos, int first, int swa,
                      bool sliding, float scale, int layout) {
    const int h = blockIdx.x, kvh = h / (H / HKV);
    const float* qh = q + (size_t) h * D;
    const __half* base = kt + (size_t) kvh * D * rows;
    for (int p = threadIdx.x; p < n_pos; p += blockDim.x) {
        const int gp = first + p;
        const int slot = sliding ? (gp % swa) : gp;
        float s = 0.f;
        if (layout == 0) {
            const __half* kp = kc + ((size_t) kvh * rows + slot) * D;
            for (int d = 0; d < D; ++d) s += qh[d] * __half2float(kp[d]);
        } else {
            for (int d = 0; d < D; ++d) s += qh[d] * __half2float(base[(size_t) d * rows + slot]);
        }
        sc[p] = s * scale;
    }
}

static __half rhalf(uint32_t& st) {
    st = st * 1664525u + 1013904223u;
    const int r = (int) (st >> 8);
    if ((r & 0xff) == 0) return __float2half(0.f);
    const float v = ((float) ((r >> 8) & 0x1fff) / 8192.f - 0.5f) * 4.f;
    return __float2half(v);
}

struct Case {
    const char* name;
    int D, H, HKV, ctx, swa, pos, n_pos;
    bool sliding;
};

int main(int argc, char** argv) {
    bool bench = false;
    for (int i = 1; i < argc; ++i) if (std::strcmp(argv[i], "--bench") == 0) bench = true;

    const Case cases[] = {
        {"1 row", 128, 48, 4, 8192, 513, 0, 1, false},
        {"33 rows", 128, 48, 4, 8192, 513, 32, 33, false},
        {"sw513 ring wrapped", 128, 48, 4, 8192, 513, 1000, 513, true},
        {"sw513 exact fill", 128, 48, 4, 8192, 513, 512, 513, true},
        {"sw513 partial", 128, 48, 4, 8192, 513, 100, 101, true},
        {"4K full", 128, 48, 4, 8192, 513, 4095, 4096, false},
        {"D=64 sw", 64, 48, 4, 8192, 513, 511, 512, true},
    };
    int failures = 0, cases_run = 0;

    for (const Case& c : cases) {
        const int rows = c.sliding ? c.swa : c.ctx;
        const int first = c.sliding ? (c.pos + 1 - c.n_pos) : 0;
        const float scale = 1.0f / std::sqrt((float) c.D);

        std::vector<float> hq((size_t) c.H * c.D);
        std::vector<__half> hk((size_t) c.HKV * rows * c.D);    // [kvh][row][d]
        std::vector<__half> hkt((size_t) c.HKV * c.D * rows);   // [kvh][d][row]
        uint32_t st = 12345u + (uint32_t) c.D;
        for (auto& v : hq) { v = (float) ((st >> 9) & 0x3ff) / 512.f - 1.f; st = st * 1664525u + 1013904223u; }
        for (auto& v : hk) v = rhalf(st);
        for (int k = 0; k < c.HKV; ++k)
            for (int r = 0; r < rows; ++r)
                for (int d = 0; d < c.D; ++d)
                    hkt[((size_t) k * c.D + d) * rows + r] = hk[((size_t) k * rows + r) * c.D + d];

        // the transposed copy must be the same numbers (a copy bug would look exactly like an arithmetic one)
        bool copy_ok = true;
        for (int k = 0; k < c.HKV && copy_ok; ++k)
            for (int r = 0; r < rows && copy_ok; r += (rows > 17 ? rows / 17 : 1))
                for (int d = 0; d < c.D; ++d)
                    if (hk[((size_t) k * rows + r) * c.D + d] != hkt[((size_t) k * c.D + d) * rows + r]) {
                        copy_ok = false;
                        break;
                    }

        float *dq = nullptr, *dout = nullptr, *dwarp = nullptr, *dsc = nullptr;
        __half *dk = nullptr, *dkt = nullptr;
        CK(cudaMalloc(&dq, hq.size() * 4));
        CK(cudaMalloc(&dk, hk.size() * 2));
        CK(cudaMalloc(&dkt, hkt.size() * 2));
        CK(cudaMalloc(&dout, (size_t) c.n_pos * 2 * 4));
        CK(cudaMalloc(&dwarp, (size_t) c.n_pos * 4));
        CK(cudaMalloc(&dsc, (size_t) c.n_pos * 4));
        CK(cudaMemcpy(dq, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dk, hk.data(), hk.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dkt, hkt.data(), hkt.size() * 2, cudaMemcpyHostToDevice));
        const int threads = 256, n = c.n_pos;

        int same[2] = {0, 0};
        int wdiff = 0;
        for (int form = 0; form < 2; ++form) {
            CK(cudaMemset(dout, 0xff, (size_t) n * 2 * 4));
            k_both<<<c.H, threads>>>(dq, dk, dkt, dout, c.D, c.H, c.HKV, rows, n, first, c.swa, c.sliding, scale, form);
            k_warp<<<c.H, threads>>>(dq, dkt, dwarp, c.D, c.H, c.HKV, rows, n, first, c.swa, c.sliding, scale);
            CK(cudaGetLastError());
            CK(cudaDeviceSynchronize());
            std::vector<float> both((size_t) n * 2), wr((size_t) n);
            CK(cudaMemcpy(both.data(), dout, both.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(wr.data(), dwarp, wr.size() * 4, cudaMemcpyDeviceToHost));
            int ok = 0;
            for (int p = 0; p < n; ++p)
                if (std::memcmp(&both[(size_t) 2 * p], &both[(size_t) 2 * p + 1], 4) == 0) ++ok;
            same[form] = ok;
            if (form == 1) {
                for (int p = 0; p < n; ++p) {
                    const float a = both[(size_t) 2 * p], b = wr[(size_t) p];
                    if (std::memcmp(&a, &b, 4) != 0) ++wdiff;
                }
            }
        }
        ++cases_run;
        const bool bitwise_plain = same[0] == n, bitwise_fma = same[1] == n;
        if (copy_ok && !(bitwise_plain && bitwise_fma)) ++failures;
        std::printf("%-20s D=%3d n=%5d first=%5d rows=%5d copy=%s | plain-form %d/%d same, fma-form %d/%d same | "
                    "warp-coop %d/%d differ -> %s\n",
                    c.name, c.D, n, first, rows, copy_ok ? "ok" : "BAD", same[0], n, same[1], n, wdiff, n,
                    !copy_ok ? "COPY BUG" : (bitwise_plain && bitwise_fma ? "BITWISE" : "DIFFERS"));
        CK(cudaFree(dq)); CK(cudaFree(dk)); CK(cudaFree(dkt));
        CK(cudaFree(dout)); CK(cudaFree(dwarp)); CK(cudaFree(dsc));
    }
    std::printf("\n%d/%d cases bitwise in BOTH forms (the transposed layout is arithmetic-identical, so wiring it "
                "cannot move the oracle - the warp-cooperative variant is the one that does)\n",
                cases_run - failures, cases_run);

    if (bench) {
        const int D = 128, H = 48, HKV = 4, swa = 513, n = 513, rows = swa, first = 1000 + 1 - n;
        const float scale = 1.0f / std::sqrt((float) D);
        std::vector<float> hq((size_t) H * D, 0.5f);
        std::vector<__half> hk((size_t) HKV * rows * D), hkt((size_t) HKV * D * rows);
        uint32_t st = 999u;
        for (auto& v : hk) v = rhalf(st);
        for (int k = 0; k < HKV; ++k)
            for (int r = 0; r < rows; ++r)
                for (int d = 0; d < D; ++d) hkt[((size_t) k * D + d) * rows + r] = hk[((size_t) k * rows + r) * D + d];
        float *dq = nullptr, *sc = nullptr;
        __half *dk = nullptr, *dkt = nullptr;
        CK(cudaMalloc(&dq, hq.size() * 4));
        CK(cudaMalloc(&dk, hk.size() * 2));
        CK(cudaMalloc(&dkt, hkt.size() * 2));
        CK(cudaMalloc(&sc, (size_t) n * 4));
        CK(cudaMemcpy(dq, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dk, hk.data(), hk.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dkt, hkt.data(), hkt.size() * 2, cudaMemcpyHostToDevice));
        const int reps = 2000, threads = 256;
        auto timeit = [&](int layout) {
            cudaEvent_t a, b;
            cudaEventCreate(&a); cudaEventCreate(&b);
            for (int i = 0; i < 100; ++i)
                k_one<<<H, threads>>>(dq, dk, dkt, sc, D, H, HKV, rows, n, first, swa, true, scale, layout);
            cudaDeviceSynchronize();
            cudaEventRecord(a);
            for (int i = 0; i < reps; ++i)
                k_one<<<H, threads>>>(dq, dk, dkt, sc, D, H, HKV, rows, n, first, swa, true, scale, layout);
            cudaEventRecord(b);
            cudaEventSynchronize(b);
            float ms = 0;
            cudaEventElapsedTime(&ms, a, b);
            cudaEventDestroy(a); cudaEventDestroy(b);
            return (double) ms / reps;
        };
        const double ms_major = timeit(0), ms_trans = timeit(1);
        const double kb = (double) H * n * D * 2 / 1e9;
        std::printf("\n--bench, short-decode geometry (H=48 blocks, 256 thr, sw 513, pos 1000), per launch:\n");
        std::printf("  row-major (shipped)   %7.4f ms  %6.1f GB/s\n", ms_major, kb / (ms_major * 1e-3));
        std::printf("  transposed (bitwise)  %7.4f ms  %6.1f GB/s   %.2fx\n", ms_trans, kb / (ms_trans * 1e-3),
                    ms_trans > 0 ? ms_major / ms_trans : 0.0);
        CK(cudaFree(dq)); CK(cudaFree(dk)); CK(cudaFree(dkt)); CK(cudaFree(sc));
    }
    return failures == 0 ? 0 : 1;
}
