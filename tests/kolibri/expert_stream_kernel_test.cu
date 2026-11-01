// tests/kolibri/expert_stream_kernel_test.cu - plan 2026-10-31-kolibri-gpu-streaming, Task 2.
//
// The streamed expert path runs the expert products on the GPU but must return the SAME BITS as the CPU pool.  The
// CPU arithmetic is fixed by src/kernels/cpu/kq_avx2.cpp (Q4_K) and by ggml-cpu's own vec_dot (Q6_K), and the
// already-verified A1 kernels in src/kolibri/kq_a1.cuh reproduce it one token at a time.  The tile this file tests
// processes four tokens per row so the weight bytes are read and decoded once for four tokens instead of four
// times.  That is only allowed if the per-token arithmetic does not move: every token's eight integer lanes, its
// per-block FMA and its final tree must stay in the CPU's order.
//
// Usage: expert_stream_kernel_test [--selftest] [--tails-only]
//   --selftest     the bounded pairwise matrix plus the model's own geometries (default)
//   --tails-only   row/token tails only, small data, for compute-sanitizer runs
//
// A pass is 100 % bit-identical output on every case, all finite, and every sensitivity check discriminating.  A
// relative-error "pass" is not accepted anywhere in this file.

#include "../../src/kolibri/expert_stream_kernels.cuh"
#include "../../src/kolibri/kq_a1.cuh"

#include "ggml-cpu.h"
#include "ggml.h"

#include "strata/kernels/cpu/kq_avx2.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

using strata::kolibri::a1::kQ4KBlock;
using strata::kolibri::a1::kQ6KBlock;
using strata::kolibri::a1::kQ8KBlock;
using strata::kolibri::stream::kTileRows;
using strata::kolibri::stream::kTileTokens;

int g_failures = 0;
int g_cases = 0;

constexpr int kQK = 256;
constexpr int kQ4K = 12;   // ggml type of gate/up and of the Q4_K down layers
constexpr int kQ6K = 14;   // ggml type of the Q6_K down layers

size_t block_bytes(int type) { return type == kQ4K ? (size_t) kQ4KBlock : (size_t) kQ6KBlock; }

uint16_t to_half_bits(float f) {
    const __half h = __float2half_rn(f);
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
}

float from_half_bits(uint16_t u) {
    __half h;
    std::memcpy(&h, &u, 2);
    return __half2float(h);
}

/// Weights with the format's real shape: positive per-block scales, random 6-bit scales/mins/qh/ql bytes.  `extreme`
/// uses the largest finite fp16 scale, `zero_blocks` zeroes every second block's scales (a block that must contribute
/// exactly nothing), and every 7th block's scales are set to 1/1024 to give the sums a small magnitude.
void fill_weights(std::vector<uint8_t>& w, int type, int rows, int nb, uint32_t seed, bool extreme, bool zero_blocks) {
    std::mt19937 g(seed);
    std::uniform_int_distribution<int> byte(0, 255);
    const size_t bb = block_bytes(type);
    for (int r = 0; r < rows; ++r) {
        for (int i = 0; i < nb; ++i) {
            uint8_t* b = w.data() + ((size_t) r * nb + i) * bb;
            for (size_t j = 0; j < bb; ++j) b[j] = (uint8_t) byte(g);
            float d = extreme ? 65504.f : 0.0020f + 0.0001f * float(g() % 97);
            float dm = extreme ? 65504.f : 0.0009f + 0.0001f * float(g() % 89);
            if (zero_blocks && (i % 2) == 0) {
                d = 0.f;
                dm = 0.f;
            } else if ((i % 7) == 6) {
                d *= 1.f / 1024.f;
                dm *= 1.f / 1024.f;
            }
            const uint16_t hd = to_half_bits(d), hm = to_half_bits(dm);
            if (type == kQ4K) {
                std::memcpy(b, &hd, 2);       // d
                std::memcpy(b + 2, &hm, 2);   // dmin
            } else {
                std::memcpy(b + 208, &hd, 2);  // Q6_K keeps its scale at the end
            }
        }
    }
}

/// Q8_K activations produced by ggml's own quantizer, so the integer sums and block sums are the real ones.
struct Acts {
    std::vector<std::vector<uint8_t>> blk;
    std::vector<const void*> ptr;
};

Acts make_acts(int n, int nt, uint32_t seed) {
    Acts a;
    a.blk.assign((size_t) nt, std::vector<uint8_t>((size_t) ggml_row_size(GGML_TYPE_Q8_K, n)));
    a.ptr.assign((size_t) nt, nullptr);
    std::mt19937 g(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(-3.f, 3.f);
    std::vector<float> x((size_t) n);
    const auto* tq8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    for (int t = 0; t < nt; ++t) {
        for (int i = 0; i < n; ++i) x[(size_t) i] = (i % 5 == 0) ? ud(g) : nd(g);   // signed, with a wider tail
        if (t == 1) {                                                              // one all-zero token
            for (auto& v : x) v = 0.f;
        }
        tq8->from_float(x.data(), a.blk[(size_t) t].data(), n);
        a.ptr[(size_t) t] = a.blk[(size_t) t].data();
    }
    return a;
}

/// The CPU oracle in the form the engine's pool uses for Q4_K rows.
void cpu_oracle_q4k(const uint8_t* w, size_t row_bytes, int n, const Acts& a, int nt, std::vector<float>& out, int rows) {
    out.assign((size_t) nt * rows, 0.f);
    std::vector<float*> op((size_t) nt);
    for (int t = 0; t < nt; ++t) op[(size_t) t] = out.data() + (size_t) t * rows;
    strata::kernels::cpu::kq256_rows(kQ4K, w, row_bytes, n, a.ptr.data(), nt, op.data(), 0, rows);
}

/// Q6_K has no multi-token CPU kernel: ggml-cpu's vec_dot is the oracle, one token at a time (kq256_rows refuses 14).
void cpu_oracle_q6k(const uint8_t* w, size_t row_bytes, int n, const Acts& a, int nt, std::vector<float>& out, int rows) {
    out.assign((size_t) nt * rows, 0.f);
    const auto* tc = ggml_get_type_traits_cpu(GGML_TYPE_Q6_K);
    for (int t = 0; t < nt; ++t) {
        for (int r = 0; r < rows; ++r) {
            float v = 0.f;
            tc->vec_dot(n, &v, 0, w + (size_t) r * row_bytes, 0, a.ptr[(size_t) t], 0, 1);
            out[(size_t) t * rows + r] = v;
        }
    }
}

void cpu_oracle(int type, const uint8_t* w, size_t row_bytes, int n, const Acts& a, int nt, std::vector<float>& out,
                int rows) {
    if (type == kQ4K) {
        cpu_oracle_q4k(w, row_bytes, n, a, nt, out, rows);
    } else {
        cpu_oracle_q6k(w, row_bytes, n, a, nt, out, rows);
    }
}

struct Case {
    int type, nb, rows, nt;
    bool extreme, zero_blocks;
    size_t offset;   // byte offset in front of the rows: 144/210 are not 32-byte multiples, so loads are unaligned
};

std::string case_name(const Case& c) {
    char b[192];
    std::snprintf(b, sizeof b, "%s nb=%d rows=%d nt=%d%s%s offset=%zu",
                  c.type == kQ4K ? "Q4_K" : "Q6_K", c.nb, c.rows, c.nt, c.extreme ? " extreme" : "",
                  c.zero_blocks ? " zero-blocks" : "", c.offset);
    return b;
}

/// Every element compared as bits; the first difference is reported with the bits of both.
bool compare_bits(const std::vector<float>& got, const std::vector<float>& want, int rows, int nt, const char* what) {
    size_t diffs = 0;
    int ft = -1, fr = -1;
    for (int t = 0; t < nt; ++t) {
        for (int r = 0; r < rows; ++r) {
            const size_t i = (size_t) t * rows + r;
            if (std::memcmp(&got[i], &want[i], 4) != 0) {
                if (diffs == 0) {
                    ft = t;
                    fr = r;
                }
                ++diffs;
            }
            if (!std::isfinite(got[i])) {
                std::printf("          NOT FINITE: token %d row %d\n", t, r);
                return false;
            }
        }
    }
    if (diffs) {
        uint32_t gb = 0, wb = 0;
        std::memcpy(&gb, &got[(size_t) ft * rows + fr], 4);
        std::memcpy(&wb, &want[(size_t) ft * rows + fr], 4);
        std::printf("          %s: %zu of %d values differ; first token %d row %d got 0x%08x want 0x%08x\n", what, diffs,
                    rows * nt, ft, fr, gb, wb);
        return false;
    }
    return true;
}

/// Runs one geometry: the tiled kernel against the CPU oracle, and a bounded subset against the A1 per-token kernels.
void run_case(const Case& c) {
    ++g_cases;
    const int n = c.nb * kQK;
    const size_t bb = block_bytes(c.type);
    const size_t row_bytes = (size_t) c.nb * bb;
    std::vector<uint8_t> host(c.offset + row_bytes * (size_t) c.rows + 32, 0);
    std::vector<uint8_t> wsrc(host.size() - c.offset - 32);
    fill_weights(wsrc, c.type, c.rows, c.nb, 7000u + (uint32_t) (c.rows * 31 + c.nb * 7 + c.nt), c.extreme,
                 c.zero_blocks);
    std::memcpy(host.data() + c.offset, wsrc.data(), wsrc.size());
    const Acts a = make_acts(n, c.nt, 900 + (uint32_t) c.nt);

    std::vector<float> want;
    cpu_oracle(c.type, host.data() + c.offset, row_bytes, n, a, c.nt, want, c.rows);

    uint8_t* dw = nullptr;
    float* dout = nullptr;
    cudaError_t e = cudaMalloc((void**) &dw, host.size());
    if (e == cudaSuccess) e = cudaMalloc((void**) &dout, sizeof(float) * (size_t) c.rows * (size_t) c.nt);
    std::vector<const uint8_t*> dacts((size_t) c.nt, nullptr);
    if (e == cudaSuccess) {
        std::vector<uint8_t*> tmp((size_t) c.nt, nullptr);
        for (int t = 0; t < c.nt; ++t) {
            if (cudaMalloc((void**) &dacts[(size_t) t], a.blk[(size_t) t].size()) != cudaSuccess) {
                e = cudaGetLastError();
                break;
            }
        }
    }
    if (e == cudaSuccess) {
        e = cudaMemcpy(dw, host.data(), host.size(), cudaMemcpyHostToDevice);
        for (int t = 0; t < c.nt && e == cudaSuccess; ++t)
            e = cudaMemcpy((void*) dacts[(size_t) t], a.blk[(size_t) t].data(), a.blk[(size_t) t].size(),
                           cudaMemcpyHostToDevice);
    }
    bool refused = false;
    std::vector<float> got;
    if (e == cudaSuccess) {
        const int rc = c.type == kQ4K
                           ? strata::kolibri::stream::es_q4k_rows(dw + c.offset, c.nb, c.rows, dacts.data(), c.nt,
                                                                  dout, c.rows, nullptr)
                           : strata::kolibri::stream::es_q6k_rows(dw + c.offset, c.nb, c.rows, dacts.data(), c.nt,
                                                                  dout, c.rows, nullptr);
        if (rc != 0) {
            std::printf("  FAIL %s: launcher refused (%d)\n", case_name(c).c_str(), rc);
            ++g_failures;
            refused = true;
        } else {
            e = cudaDeviceSynchronize();
        }
    }
    if (e == cudaSuccess) {
        got.resize(want.size());
        e = cudaMemcpy(got.data(), dout, sizeof(float) * got.size(), cudaMemcpyDeviceToHost);
    }
    if (refused) {
        cudaGetLastError();
    } else if (e != cudaSuccess) {
        std::printf("  FAIL %s: cuda %s\n", case_name(c).c_str(), cudaGetErrorString(e));
        ++g_failures;
        cudaGetLastError();
    } else if (!compare_bits(got, want, c.rows, c.nt, "tiled vs CPU oracle")) {
        std::printf("  FAIL %s\n", case_name(c).c_str());
        ++g_failures;
    }
    for (int t = 0; t < c.nt; ++t)
        if (dacts[(size_t) t]) cudaFree((void*) dacts[(size_t) t]);
    if (dout) cudaFree(dout);
    if (dw) cudaFree(dw);
}

/// A1 needs a device pointer for the activation too, so the second-opinion check is a separate, explicit run.
void run_a1_agreement(int type, int nb, int rows, int nt) {
    const int n = nb * kQK;
    const size_t row_bytes = (size_t) nb * block_bytes(type);
    std::vector<uint8_t> w(row_bytes * (size_t) rows);
    fill_weights(w, type, rows, nb, 4242, false, false);
    const Acts a = make_acts(n, nt, 31337);
    std::vector<float> want;
    cpu_oracle(type, w.data(), row_bytes, n, a, nt, want, rows);

    uint8_t* dw = nullptr;
    float* dtile = nullptr;
    float* da1 = nullptr;
    std::vector<const uint8_t*> dacts((size_t) nt, nullptr);
    std::vector<uint8_t*> dactp((size_t) nt, nullptr);
    bool ok = cudaMalloc((void**) &dw, w.size()) == cudaSuccess &&
              cudaMalloc((void**) &dtile, sizeof(float) * (size_t) rows * nt) == cudaSuccess &&
              cudaMalloc((void**) &da1, sizeof(float) * (size_t) rows * nt) == cudaSuccess;
    for (int t = 0; ok && t < nt; ++t) {
        ok = cudaMalloc((void**) &dactp[(size_t) t], a.blk[(size_t) t].size()) == cudaSuccess &&
             cudaMemcpy(dactp[(size_t) t], a.blk[(size_t) t].data(), a.blk[(size_t) t].size(), cudaMemcpyHostToDevice) ==
                 cudaSuccess;
        dacts[(size_t) t] = dactp[(size_t) t];
    }
    ok = ok && cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice) == cudaSuccess;
    if (!ok) {
        std::printf("  FAIL A1 agreement (%s): allocation/copy failed\n", type == kQ4K ? "Q4_K" : "Q6_K");
        ++g_failures;
        cudaGetLastError();
        return;
    }
    const int rc = type == kQ4K ? strata::kolibri::stream::es_q4k_rows(dw, nb, rows, dacts.data(), nt, dtile, rows, nullptr)
                                : strata::kolibri::stream::es_q6k_rows(dw, nb, rows, dacts.data(), nt, dtile, rows, nullptr);
    for (int t = 0; t < nt; ++t) {
        if (type == kQ4K) {
            strata::kolibri::a1::kq_a1_dot_rows<<<(rows + 15) / 16, 128>>>(dw, dactp[(size_t) t], nb, rows,
                                                                         da1 + (size_t) t * rows);
        } else {
            strata::kolibri::a1::kq_a1_dot_rows_q6k<<<(rows + 15) / 16, 128>>>(dw, dactp[(size_t) t], nb, rows,
                                                                            da1 + (size_t) t * rows);
        }
    }
    cudaError_t e = cudaDeviceSynchronize();
    std::vector<float> tiled, first;
    if (rc == 0 && e == cudaSuccess) {
        tiled.resize((size_t) rows * nt);
        first.resize(tiled.size());
        e = cudaMemcpy(tiled.data(), dtile, sizeof(float) * tiled.size(), cudaMemcpyDeviceToHost);
        if (e == cudaSuccess) e = cudaMemcpy(first.data(), da1, sizeof(float) * first.size(), cudaMemcpyDeviceToHost);
    }
    ++g_cases;
    if (rc != 0 || e != cudaSuccess) {
        std::printf("  FAIL A1 agreement (%s nb=%d rows=%d nt=%d): rc=%d cuda=%s\n", type == kQ4K ? "Q4_K" : "Q6_K", nb,
                    rows, nt, rc, cudaGetErrorString(e));
        ++g_failures;
        cudaGetLastError();
    } else {
        // Tiled agrees with A1 AND with the CPU oracle: A1-only agreement is not a pass, so both are checked.
        if (!compare_bits(tiled, first, rows, nt, "tiled vs A1")) {
            std::printf("  FAIL A1 agreement (%s nb=%d rows=%d nt=%d)\n", type == kQ4K ? "Q4_K" : "Q6_K", nb, rows, nt);
            ++g_failures;
        } else if (!compare_bits(tiled, want, rows, nt, "tiled vs CPU oracle")) {
            std::printf("  FAIL A1 agreement (%s nb=%d rows=%d nt=%d)\n", type == kQ4K ? "Q4_K" : "Q6_K", nb, rows, nt);
            ++g_failures;
        }
    }
    for (int t = 0; t < nt; ++t)
        if (dactp[(size_t) t]) cudaFree(dactp[(size_t) t]);
    if (da1) cudaFree(da1);
    if (dtile) cudaFree(dtile);
    if (dw) cudaFree(dw);
}

/// The three mistakes this task exists to prevent, each expressed as a change to the WEIGHTS that is exactly
/// equivalent to the kernel making it, and required to change the oracle's output.  A data set that cannot see them
/// proves nothing about the kernel.
void sensitivity_checks() {
    const int nb = 10, rows = 64, nt = 4, n = nb * kQK;
    const size_t row_bytes = (size_t) nb * kQ4KBlock;
    std::vector<uint8_t> w(row_bytes * (size_t) rows);
    fill_weights(w, kQ4K, rows, nb, 555, false, false);
    const Acts a = make_acts(n, nt, 66);

    std::vector<float> base, mut;
    cpu_oracle_q4k(w.data(), row_bytes, n, a, nt, base, rows);

    std::vector<uint8_t> flip = w;
    for (int r = 0; r < rows; ++r)
        for (int i = 0; i < nb; ++i) flip[((size_t) r * nb + i) * kQ4KBlock + 3] ^= 0x80;   // dmin sign bit
    cpu_oracle_q4k(flip.data(), row_bytes, n, a, nt, mut, rows);
    ++g_cases;
    if (base == mut) {
        std::printf("  FAIL sensitivity: flipping dmin's sign changed nothing - a sign error would pass this data\n");
        ++g_failures;
    }

    const size_t row6 = (size_t) nb * kQ6KBlock;
    std::vector<uint8_t> w6(row6 * (size_t) rows);
    fill_weights(w6, kQ6K, rows, nb, 777, false, false);
    cpu_oracle_q6k(w6.data(), row6, n, a, nt, base, rows);

    std::vector<uint8_t> pairoff = w6;
    for (int r = 0; r < rows; ++r)
        for (int i = 0; i < nb; ++i) {
            uint8_t* b = pairoff.data() + ((size_t) r * nb + i) * kQ6KBlock + 192;
            for (int s = 0; s < 16; s += 2) std::swap(b[s], b[s + 1]);   // the scale-pair mix-up
        }
    cpu_oracle_q6k(pairoff.data(), row6, n, a, nt, mut, rows);
    ++g_cases;
    if (base == mut) {
        std::printf("  FAIL sensitivity: swapping Q6_K scale pairs changed nothing - the pair order is untested\n");
        ++g_failures;
    }

    std::vector<uint8_t> qloff = w6;
    for (int r = 0; r < rows; ++r)
        for (int i = 0; i < nb; ++i) {
            uint8_t* b = qloff.data() + ((size_t) r * nb + i) * kQ6KBlock;
            std::rotate(b, b + 32, b + 128);   // (c&1)*32 dropped   // dropping the (c&1)*32 half of the ql index
        }
    cpu_oracle_q6k(qloff.data(), row6, n, a, nt, mut, rows);
    ++g_cases;
    if (base == mut) {
        std::printf("  FAIL sensitivity: shifting Q6_K ql by 32 bytes changed nothing - the ql offset is untested\n");
        ++g_failures;
    }
}

/// The whole expert, end to end, at Kolibri-1's own geometry: the tiled gate/up product, then the CPU's SwiGLU
/// expression and native_quant_h, then the down product in whichever format the layer stores - compared with the
/// pool's own native_gu_rows / native_down_rows, bit for bit, including the hidden's quantized BYTES.
void run_expert_case(int d_type, int nt) {
    using namespace strata::kernels::cpu;
    ++g_cases;
    const int H = 2560, FF = 512;
    NativeFmt f;
    f.gu_type = 12;
    f.d_type = d_type;
    f.gu_act = 15;   // Q8_K
    f.d_act = 15;
    f.n_embd = H;
    f.n_ff = FF;
    f.gu_row = (size_t) 10 * kQ4KBlock;
    f.d_row = (size_t) 2 * (d_type == kQ4K ? kQ4KBlock : kQ6KBlock);
    f.up_off = (size_t) FF * f.gu_row;
    f.down_off = (size_t) 2 * FF * f.gu_row;
    f.bytes = f.down_off + (size_t) H * f.d_row;
    f.act_bytes = (size_t) ggml_row_size(GGML_TYPE_Q8_K, H);
    f.h_bytes = (size_t) ggml_row_size(GGML_TYPE_Q8_K, FF);

    std::vector<uint8_t> blob(f.bytes);
    fill_weights(blob, 12, 2 * FF, 10, 101, false, false);
    fill_weights(blob, d_type, 0, 0, 0, false, false);   // no-op: the down rows are filled next
    {
        std::vector<uint8_t> down((size_t) H * f.d_row);
        fill_weights(down, d_type, H, 2, 202, false, false);
        std::memcpy(blob.data() + f.down_off, down.data(), down.size());
    }

    std::mt19937 g(4321 + (uint32_t) nt);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<std::vector<float>> x((size_t) nt, std::vector<float>((size_t) H));
    std::vector<std::vector<uint8_t>> a((size_t) nt, std::vector<uint8_t>(f.act_bytes));
    std::vector<std::vector<float>> ff((size_t) nt, std::vector<float>((size_t) FF));
    std::vector<std::vector<uint8_t>> hq((size_t) nt, std::vector<uint8_t>(f.h_bytes));
    std::vector<float*> ffp((size_t) nt), outp((size_t) nt);
    std::vector<const void*> ap((size_t) nt), hp((size_t) nt);
    std::vector<float> want((size_t) nt * H);
    for (int t = 0; t < nt; ++t) {
        for (auto& v : x[(size_t) t]) v = (t == 1) ? 0.f : nd(g);
        native_quant_act(f, x[(size_t) t].data(), a[(size_t) t].data());
        ffp[(size_t) t] = ff[(size_t) t].data();
        outp[(size_t) t] = want.data() + (size_t) t * H;
        ap[(size_t) t] = a[(size_t) t].data();
    }
    native_gu_rows(f, blob.data(), ap.data(), nt, ffp.data(), 0, FF);
    for (int t = 0; t < nt; ++t) {
        native_quant_h(f, ff[(size_t) t].data(), hq[(size_t) t].data());
        hp[(size_t) t] = hq[(size_t) t].data();
    }
    native_down_rows(f, blob.data(), hp.data(), nt, outp.data(), 0, H);

    // GPU: gate/up as one 1024-row product (the up block follows the gate block at the same stride)
    uint8_t* dblob = nullptr;
    float* dgu = nullptr;
    float* ddown = nullptr;
    std::vector<const uint8_t*> dacts((size_t) nt, nullptr);
    std::vector<const uint8_t*> dhq((size_t) nt, nullptr);
    bool ok = cudaMalloc((void**) &dblob, f.bytes) == cudaSuccess &&
              cudaMalloc((void**) &dgu, sizeof(float) * (size_t) nt * 2 * FF) == cudaSuccess &&
              cudaMalloc((void**) &ddown, sizeof(float) * (size_t) nt * H) == cudaSuccess &&
              cudaMemcpy(dblob, blob.data(), blob.size(), cudaMemcpyHostToDevice) == cudaSuccess;
    for (int t = 0; ok && t < nt; ++t) {
        uint8_t* p = nullptr;
        ok = cudaMalloc((void**) &p, f.act_bytes) == cudaSuccess &&
             cudaMemcpy(p, a[(size_t) t].data(), f.act_bytes, cudaMemcpyHostToDevice) == cudaSuccess;
        dacts[(size_t) t] = p;
    }
    int rc = ok ? strata::kolibri::stream::es_q4k_rows(dblob, 10, 2 * FF, dacts.data(), nt, dgu, 2 * FF, nullptr) : 9;
    std::vector<float> gu;
    if (rc == 0 && cudaDeviceSynchronize() == cudaSuccess) {
        gu.resize((size_t) nt * 2 * FF);
        ok = cudaMemcpy(gu.data(), dgu, sizeof(float) * gu.size(), cudaMemcpyDeviceToHost) == cudaSuccess;
    } else {
        ok = false;
    }
    // the CPU bridge: native_gu_rows' own SwiGLU expression, then native_quant_h
    std::vector<std::vector<uint8_t>> hq2((size_t) nt, std::vector<uint8_t>(f.h_bytes));
    std::vector<std::vector<float>> ff2((size_t) nt, std::vector<float>((size_t) FF));
    for (int t = 0; ok && t < nt; ++t) {
        for (int r = 0; r < FF; ++r) {
            const float gv = gu[(size_t) t * 2 * FF + r];
            const float uv = gu[(size_t) t * 2 * FF + FF + r];
            ff2[(size_t) t][(size_t) r] = (gv / (1.f + std::exp(-gv))) * uv;
        }
        native_quant_h(f, ff2[(size_t) t].data(), hq2[(size_t) t].data());
    }
    int hdiff = 0;
    for (int t = 0; ok && t < nt; ++t)
        hdiff += std::memcmp(hq2[(size_t) t].data(), hq[(size_t) t].data(), f.h_bytes) != 0;
    for (int t = 0; ok && t < nt; ++t) {
        uint8_t* p = nullptr;
        ok = cudaMalloc((void**) &p, f.h_bytes) == cudaSuccess &&
             cudaMemcpy(p, hq2[(size_t) t].data(), f.h_bytes, cudaMemcpyHostToDevice) == cudaSuccess;
        dhq[(size_t) t] = p;
    }
    if (ok) {
        rc = d_type == kQ4K
                 ? strata::kolibri::stream::es_q4k_rows(dblob + f.down_off, 2, H, dhq.data(), nt, ddown, H, nullptr)
                 : strata::kolibri::stream::es_q6k_rows(dblob + f.down_off, 2, H, dhq.data(), nt, ddown, H, nullptr);
    }
    std::vector<float> got;
    if (rc == 0 && ok && cudaDeviceSynchronize() == cudaSuccess) {
        got.resize((size_t) nt * H);
        ok = cudaMemcpy(got.data(), ddown, sizeof(float) * got.size(), cudaMemcpyDeviceToHost) == cudaSuccess;
    } else {
        ok = false;
    }
    char name[128];
    std::snprintf(name, sizeof name, "expert %s down, %d token(s)", d_type == kQ4K ? "Q4_K" : "Q6_K", nt);
    if (!ok || rc != 0) {
        std::printf("  FAIL %s: setup or launch failed (rc %d, cuda %s)\n", name, rc, cudaGetErrorString(cudaGetLastError()));
        ++g_failures;
    } else if (hdiff) {
        std::printf("  FAIL %s: %d of %d hidden quantizations differ from native_quant_h\n", name, hdiff, nt);
        ++g_failures;
    } else if (!compare_bits(got, want, H, nt, "expert vs native_gu_rows/native_down_rows")) {
        std::printf("  FAIL %s\n", name);
        ++g_failures;
    }
    for (int t = 0; t < nt; ++t) {
        if (dhq[(size_t) t]) cudaFree((void*) dhq[(size_t) t]);
        if (dacts[(size_t) t]) cudaFree((void*) dacts[(size_t) t]);
    }
    if (ddown) cudaFree(ddown);
    if (dgu) cudaFree(dgu);
    if (dblob) cudaFree(dblob);
    cudaGetLastError();
}

bool device_ok() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) return false;
    std::printf("device: %s, sm_%d%d\n", p.name, p.major, p.minor);
    return p.major > 7 || (p.major == 7 && p.minor >= 5);
}

}  // namespace

int main(int argc, char** argv) {
    bool tails_only = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--tails-only") {
            tails_only = true;
        } else if (a != "--selftest") {
            std::printf("usage: %s [--selftest] [--tails-only]\n", argv[0]);
            return 2;
        }
    }
    ggml_cpu_init();
    if (!strata::kernels::cpu::kq256_supported(kQ4K)) {
        std::printf("SKIP: the CPU oracle needs AVX2 and the kq256 Q4_K kernel on this host\n");
        return 0;
    }
    if (!device_ok()) {
        std::printf("FAIL: no CUDA device with sm_75 or newer\n");
        return 1;
    }

    if (tails_only) {
        for (int nb : {1, 10}) {
            for (int rows : {1, 15, 17}) {
                for (int nt : {1, 2, 3, 5}) {
                    run_case({kQ4K, nb, rows, nt, false, false, 0});
                    run_case({kQ4K, nb, rows, nt, false, false, (size_t) kQ4KBlock});
                    run_case({kQ6K, nb, rows, nt, false, false, 0});
                }
            }
        }
    } else {
        // (a) every nb at every row tail, four tokens
        for (int nb : {1, 2, 10, 32})
            for (int rows : {1, 15, 16, 17, 512, 2560}) run_case({kQ4K, nb, rows, 4, false, false, 0});
        // (b) every token count at a row tail and at a full tile boundary
        for (int nb : {1, 10})
            for (int nt : {1, 2, 7, 8, 9, 13, 31, 64, 65})
                for (int rows : {17, 512}) run_case({kQ4K, nb, rows, nt, false, false, 0});
        // (c) the model's own geometries: gate/up Q4_K (nb=10, 512 rows) and both down projections
        for (int nt : {1, 4, 64, 65}) {
            run_case({kQ4K, 10, 512, nt, false, false, 0});
            run_case({kQ4K, 2, 2560, nt, false, false, 0});
            run_case({kQ6K, 2, 2560, nt, false, false, 0});
        }
        // (d) the format's awkward corners: largest fp16 scales, zero blocks, unaligned row starts
        run_case({kQ4K, 10, 17, 4, true, false, 0});
        run_case({kQ4K, 10, 17, 4, false, true, 0});
        run_case({kQ4K, 10, 17, 4, false, false, (size_t) kQ4KBlock});
        run_case({kQ4K, 10, 512, 4, false, false, (size_t) 7 * kQ4KBlock});
        run_case({kQ6K, 2, 2560, 4, true, false, 0});
        run_case({kQ6K, 2, 2560, 4, false, true, 0});
        run_case({kQ6K, 2, 2560, 4, false, false, (size_t) kQ6KBlock});
        run_case({kQ6K, 2, 17, 65, false, false, (size_t) 3 * kQ6KBlock});
        // (e) A1 as a second opinion
        run_a1_agreement(kQ4K, 10, 17, 4);
        run_a1_agreement(kQ4K, 2, 17, 1);
        run_a1_agreement(kQ6K, 2, 17, 4);
        // (f) the whole expert at the model's geometry: gate/up, CPU SwiGLU, native_quant_h, down, in both formats
        for (int nt : {1, 4, 5, 64}) {
            run_expert_case(kQ4K, nt);
            run_expert_case(kQ6K, nt);
        }
        // (g) the sensitivity of the data itself
        sensitivity_checks();
    }

    std::printf("%s: %d case(s), %d failure(s)\n", g_failures ? "FAIL" : "PASS", g_cases, g_failures);
    return g_failures ? 1 : 0;
}
