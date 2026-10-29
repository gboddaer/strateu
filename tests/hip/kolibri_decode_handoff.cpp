// Kolibri decode handoff: can a per-layer graph publish its payload to the CPU, let the CPU compute, and take
// the CPU's answer back - for 50 layers, several tokens, and a payload that changes every launch?
//
// This is the synthetic proof the decode plan (docs/plans/2026-10-29-kolibri-decode-experts-vram.md, Task 2)
// asks for BEFORE any model code is touched, and it is deliberately narrower than `handoff.cpp`: that test
// proves the primitives work at all, this one proves the protocol the Kolibri arm will use, with the payload
// encoding (token, layer) so a capture-time literal cannot pass, and with the CPU's answer flowing back into
// device arithmetic so a stale read cannot pass either.
//
// One graph per ROLE, not per layer: `pre` publishes and rings, `post` copies the CPU's answer and adds it to
// an accumulator.  Launching the same two graphs 50 times per token is a STRONGER test of the property that
// matters - that a replayed graph publishes the CURRENT payload - than 50 distinct graphs would be, because
// every replay reuses the captured launch parameters and only the device-side inputs change.
//
// Everything here goes through library entry points (no __global__ in this file): the host side of the project
// compiles these tests as plain C++ against the platform shim, which is also why the CPU's answer is consumed
// with `add_inplace` rather than a bespoke kernel.
//
// Cases and their exit codes (each case prints its own PASS line):
//   A 50 layers x 3 tokens, payload and CPU answer verified                                  -> 11..19
//   B token reset at a quiescent point, then the same token replayed                         -> 21
//   C ring visibility and host/GPU overlap without a driver call in the spin loop           -> 31
//   D missing ring: the host's bounded wait must fail, and only then may it free            -> 41
//   0  everything passed
//
// Nothing here frees a buffer while a stream could still write it, and no case waits without a deadline.
#include <hip/hip_runtime.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>
#include "strata/kernels/elementwise.hpp"

#define CHECK(x)                                                                                                       \
    do {                                                                                                               \
        hipError_t e_ = (x);                                                                                           \
        if (e_ != hipSuccess) {                                                                                        \
            std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_));                                                \
            return 2;                                                                                                  \
        }                                                                                                              \
    } while (0)

namespace {

constexpr int kN = 2560;      // Kolibri's n_embd, the published activation width
constexpr int kK = 6;         // top-k routed experts
constexpr int kLayers = 50;
constexpr int kTokens = 3;
constexpr int kSeqWords = 9;      // the doorbell region is documented as 9 words; only word 0 is used on HIP
constexpr int kBusyN = 4 << 20;   // floats for the independent GPU stage (~16 MB in, 16 MB out per pass)

// Exact, position-dependent integers: every value is < 2^24 so float holds it exactly and the host can assert
// with memcmp rather than a tolerance.  The accumulator below therefore also stays exact: 150 steps of |y| <=
// 4097 cannot leave the range where every integer is representable.
float enc_x(int token, int layer, int i) { return (float) (((token * 7 + layer * 13 + i * 3) % 4093) - 2048); }
int32_t enc_id(int token, int layer, int k) { return token * 1000 + layer * 10 + k; }   // order matters
float enc_w(int token, int layer, int k) { return (float) (k + 1) * 0.25f + (float) (token * 100 + layer); }
// The CPU's half: an exact integer result the device can add without a rounding question.
float enc_y(int token, int layer, int i) { return (float) ((int) enc_x(token, layer, i) * 2 + 1); }

struct Mapped {
    float* h = nullptr;
    float* d = nullptr;
    void alloc(size_t bytes) {
        if (hipHostMalloc((void**) &h, bytes, hipHostMallocMapped) != hipSuccess) std::exit(3);
        if (hipHostGetDevicePointer((void**) &d, h, 0) != hipSuccess) std::exit(3);
        std::memset(h, 0, bytes);
    }
    void free_() {
        if (h != nullptr) (void) hipHostFree(h);
        h = nullptr;
        d = nullptr;
    }
};

/// The ids are int32: a separate mapped buffer type rather than a cast, so the publish call's types are checked.
struct MappedI32 {
    int32_t* h = nullptr;
    int32_t* d = nullptr;
    void alloc(size_t bytes) {
        if (hipHostMalloc((void**) &h, bytes, hipHostMallocMapped) != hipSuccess) std::exit(3);
        if (hipHostGetDevicePointer((void**) &d, h, 0) != hipSuccess) std::exit(3);
        std::memset(h, 0, bytes);
    }
    void free_() {
        if (h != nullptr) (void) hipHostFree(h);
        h = nullptr;
        d = nullptr;
    }
};

struct Ring {
    uint32_t* h = nullptr;
    uint32_t* d = nullptr;
    void alloc() {
        if (hipHostMalloc((void**) &h, kSeqWords * 4, hipHostMallocMapped) != hipSuccess) std::exit(3);
        if (hipHostGetDevicePointer((void**) &d, h, 0) != hipSuccess) std::exit(3);
        std::memset(h, 0, kSeqWords * 4);
    }
    void free_() {
        if (h != nullptr) (void) hipHostFree(h);
        h = nullptr;
        d = nullptr;
    }
};

/// Two graph segments around a CPU round trip, plus the device buffers they read and write.
struct Rig {
    hipStream_t stream{};
    hipGraph_t gpre{}, gpost{};
    hipGraphExec_t epre{}, epost{};
    Mapped hx{}, hw{}, hy{};           // mapped: device writes x/w, host writes y
    MappedI32 hids{};                 // mapped: device writes the selected expert ids
    Ring ring{};
    float *dx{}, *dw{}, *dout{}, *dseen{}, *dbusy{};
    int32_t* dids{};
    int busy_passes = 0;

    int init(int busy = 0) {
        busy_passes = busy;
        hx.alloc(kN * 4);
        hids.alloc(kK * 4);
        hw.alloc(kK * 4);
        hy.alloc(kN * 4);
        ring.alloc();
        if (hipMalloc((void**) &dx, kN * 4) != hipSuccess) return 3;
        if (hipMalloc((void**) &dids, kK * 4) != hipSuccess) return 3;
        if (hipMalloc((void**) &dw, kK * 4) != hipSuccess) return 3;
        if (hipMalloc((void**) &dout, kN * 4) != hipSuccess) return 3;
        if (hipMalloc((void**) &dseen, kN * 4) != hipSuccess) return 3;
        if (hipMemset(dseen, 0, kN * 4) != hipSuccess) return 3;
        if (busy > 0 && hipMalloc((void**) &dbusy, (size_t) kBusyN * 4) != hipSuccess) return 3;
        if (busy > 0 && hipMemset(dbusy, 0, (size_t) kBusyN * 4) != hipSuccess) return 3;
        if (hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) != hipSuccess) return 3;

        // pre: the layer's payload goes out and the ring is incremented - the whole handoff, in one node.  Then the
        // layer's independent GPU stage (the shared expert, in the real engine) is enqueued AFTER the ring, which
        // is exactly the overlap the arm is for.
        if (hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal) != hipSuccess) return 3;
        strata::kernels::doorbell_publish(dx, dids, dw, kN, kK, hx.d, hids.d, hw.d, ring.d, stream);
        for (int i = 0; i < busy; ++i) strata::kernels::silu_inplace(dbusy, kBusyN, stream);
        if (hipStreamEndCapture(stream, &gpre) != hipSuccess) return 3;
        if (hipGraphInstantiateWithFlags(&epre, gpre, 0) != hipSuccess) return 3;

        // post: the CPU's answer is copied in by a KERNEL (not a copy-engine node) and consumed on the compute
        // queue, which is what the engine does so the handoff does not leave the stream.
        if (hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal) != hipSuccess) return 3;
        strata::kernels::copy_from_mapped(dout, hy.d, kN, stream);
        strata::kernels::add_inplace(dseen, dout, kN, stream);
        if (hipStreamEndCapture(stream, &gpost) != hipSuccess) return 3;
        if (hipGraphInstantiateWithFlags(&epost, gpost, 0) != hipSuccess) return 3;
        return 0;
    }

    /// One layer: publish, bounded wait for the ring, verify the payload, answer, launch post.
    /// Returns 0 on success; a nonzero code names the first mismatch.
    int step(int token, int layer, uint32_t expect_ring) {
        // The layer's inputs are device-resident and change every launch: nothing about (token, layer) is a
        // capture-time constant.
        std::vector<float> x((size_t) kN);
        std::vector<int32_t> ids((size_t) kK);
        std::vector<float> w((size_t) kK);
        for (int i = 0; i < kN; ++i) x[(size_t) i] = enc_x(token, layer, i);
        for (int k = 0; k < kK; ++k) {
            ids[(size_t) k] = enc_id(token, layer, k);
            w[(size_t) k] = enc_w(token, layer, k);
        }
        CHECK(hipMemcpyAsync(dx, x.data(), kN * 4, hipMemcpyHostToDevice, stream));
        CHECK(hipMemcpyAsync(dids, ids.data(), kK * 4, hipMemcpyHostToDevice, stream));
        CHECK(hipMemcpyAsync(dw, w.data(), kK * 4, hipMemcpyHostToDevice, stream));

        CHECK(hipGraphLaunch(epre, stream));

        // Bounded host wait: no driver call in the loop, a deadline that would end the test rather than hang it.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        volatile uint32_t* seq = ring.h;
        while (*seq < expect_ring) {
            if (std::chrono::steady_clock::now() > deadline) return 12;
            std::this_thread::yield();
        }
        std::atomic_thread_fence(std::memory_order_acquire);

        // Publication implies payload: the moment the ring is visible, every byte it announced must be this
        // token's, including the selection ORDER and the weights of this replay.
        for (int i = 0; i < kN; ++i)
            if (std::memcmp(&hx.h[(size_t) i], &x[(size_t) i], 4) != 0) return 13;
        for (int k = 0; k < kK; ++k) {
            if (hids.h[(size_t) k] != ids[(size_t) k]) return 14;
            if (std::memcmp(&hw.h[(size_t) k], &w[(size_t) k], 4) != 0) return 15;
        }

        for (int i = 0; i < kN; ++i) hy.h[(size_t) i] = enc_y(token, layer, i);
        CHECK(hipGraphLaunch(epost, stream));
        return 0;
    }

    int answers_match(int tokens, int layers) {
        CHECK(hipStreamSynchronize(stream));
        std::vector<float> got((size_t) kN);
        CHECK(hipMemcpy(got.data(), dseen, kN * 4, hipMemcpyDeviceToHost));
        for (int i = 0; i < kN; ++i) {
            float want = 0.0f;
            for (int t = 1; t <= tokens; ++t)
                for (int l = 0; l < layers; ++l) want += enc_y(t, l, i);
            if (std::memcmp(&got[(size_t) i], &want, 4) != 0) {
                std::fprintf(stderr, "answer[%d] = %.9g, want %.9g\n", i, (double) got[(size_t) i], (double) want);
                return 16;
            }
        }
        return 0;
    }

    void destroy() {
        // Only after the stream is idle may any of these be released.
        (void) hipStreamSynchronize(stream);
        if (epre) (void) hipGraphExecDestroy(epre);
        if (epost) (void) hipGraphExecDestroy(epost);
        if (gpre) (void) hipGraphDestroy(gpre);
        if (gpost) (void) hipGraphDestroy(gpost);
        if (stream) (void) hipStreamDestroy(stream);
        if (dx) (void) hipFree(dx);
        if (dids) (void) hipFree(dids);
        if (dw) (void) hipFree(dw);
        if (dout) (void) hipFree(dout);
        if (dseen) (void) hipFree(dseen);
        if (dbusy) (void) hipFree(dbusy);
        hx.free_();
        hids.free_();
        hw.free_();
        hy.free_();
        ring.free_();
    }
};

}  // namespace

int main(int argc, char** argv) {
    const bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, 0) != hipSuccess) { std::fprintf(stderr, "no device 0\n"); return 2; }

    // ---- A: 50 layers x 3 tokens, changing payload, CPU answer verified in device arithmetic ----
    {
        Rig r;
        if (int e = r.init(); e != 0) { std::fprintf(stderr, "init failed (%d)\n", e); return e; }
        uint32_t ring = 0;
        for (int t = 1; t <= kTokens; ++t)
            for (int l = 0; l < kLayers; ++l)
                if (int e = r.step(t, l, ++ring); e != 0) {
                    std::fprintf(stderr, "token %d layer %d: step failed (%d)\n", t, l, e);
                    r.destroy();
                    return e;
                }
        if (int e = r.answers_match(kTokens, kLayers); e != 0) { r.destroy(); return e; }
        r.destroy();
        std::printf("PASS A %d layer steps over %d token(s): every published payload matched the current "
                    "(token, layer) bitwise, and the CPU's answer reached %d device accumulators\n",
                    kLayers * kTokens, kTokens, kN);
    }

    // ---- B: reset at a quiescent point, then replay the same token ----
    {
        Rig r;
        if (int e = r.init(); e != 0) return e;
        uint32_t ring = 0;
        for (int l = 0; l < kLayers; ++l)
            if (int e = r.step(1, l, ++ring); e != 0) {
                std::fprintf(stderr, "B: layer %d before the reset: step failed (%d)\n", l, e);
                r.destroy();
                return 21;
            }
        if (int e = r.answers_match(1, kLayers); e != 0) { r.destroy(); return e; }
        // A new request starts at zero.  The stream is idle, so clearing the ring cannot race a publisher.
        CHECK(hipStreamSynchronize(r.stream));
        r.ring.h[0] = 0;
        ring = 0;   // the expected value restarts with the ring, or every later wait would time out on a stale target
        CHECK(hipMemset(r.dseen, 0, kN * 4));
        for (int l = 0; l < kLayers; ++l)
            if (int e = r.step(1, l, ++ring); e != 0) {
                std::fprintf(stderr, "B: layer %d after the reset: step failed (%d)\n", l, e);
                r.destroy();
                return 21;
            }
        if (int e = r.answers_match(1, kLayers); e != 0) { r.destroy(); return e; }
        r.destroy();
        std::printf("PASS B reset to 0 after the stream went idle, then the same %d layers replayed identically\n",
                    kLayers);
    }

    // ---- C: is the ring visible without a driver call, and does the host get to run while the GPU is busy? ----
    {
        auto now = [] { return std::chrono::steady_clock::now(); };
        // Calibrate the independent GPU stage to about 3 ms: one pass first, then the count that scales to it.
        int passes = 1;
        {
            Rig cal;
            if (int e = cal.init(1); e != 0) return e;
            const auto t0 = now();
            CHECK(hipGraphLaunch(cal.epre, cal.stream));
            CHECK(hipStreamSynchronize(cal.stream));
            const double one_ms = std::chrono::duration<double, std::milli>(now() - t0).count();
            cal.destroy();
            if (one_ms > 0.05) passes = (int) std::min(200.0, std::max(1.0, 3.0 / one_ms));
        }

        Rig r;
        if (int e = r.init(passes); e != 0) return e;
        const int rounds = quick ? 4 : 30;
        int seen_pure = 0, overlapped = 0, late = 0;
        uint32_t ring = 0;
        for (int i = 0; i < rounds; ++i) {
            CHECK(hipGraphLaunch(r.epre, r.stream));
            ++ring;
            bool seen = false;
            const auto deadline = now() + std::chrono::milliseconds(200);
            volatile uint32_t* seq = r.ring.h;
            while (now() < deadline) {
                if (*seq >= ring) { seen = true; break; }
                std::this_thread::yield();
            }
            if (seen) {
                ++seen_pure;
                if (hipStreamQuery(r.stream) == hipErrorNotReady) ++overlapped;   // one query, AFTER seeing the ring
            } else {
                ++late;
            }
            CHECK(hipStreamSynchronize(r.stream));
            if (r.ring.h[0] < ring) {
                std::fprintf(stderr, "round %d: the ring never advanced\n", i);
                r.destroy();
                return 31;
            }
        }
        r.destroy();
        if (late != 0) {
            std::fprintf(stderr, "C: %d of %d rings were not visible to a pure host spin within 200 ms; the "
                                 "engine's per-layer poll calls the driver for a reason, and a whole-token graph "
                                 "cannot rely on a host spin alone\n", late, rounds);
            return 31;
        }
        std::printf("PASS C %d/%d rings seen by a host spin with no driver call; the GPU stage (%d pass(es), about "
                    "3 ms) was still running in %d/%d of them\n", seen_pure, rounds, passes, overlapped, rounds);
    }

    // ---- D: a graph that never rings must fail the host's bounded wait, and cleanup must still be safe ----
    {
        hipStream_t stream;
        CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        hipGraph_t g;
        hipGraphExec_t e;
        CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
        strata::kernels::silu_inplace(nullptr, 0, stream);   // an empty graph: no publish node at all
        CHECK(hipStreamEndCapture(stream, &g));
        CHECK(hipGraphInstantiateWithFlags(&e, g, 0));
        Ring ring;
        ring.alloc();
        CHECK(hipGraphLaunch(e, stream));
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
        volatile uint32_t* seq = ring.h;
        bool rung = false;
        while (std::chrono::steady_clock::now() < deadline) {
            if (*seq != 0) { rung = true; break; }
            std::this_thread::yield();
        }
        CHECK(hipStreamSynchronize(stream));   // quiesce BEFORE releasing anything
        if (rung) {
            std::fprintf(stderr, "D: a graph with no publish node rang anyway\n");
            return 41;
        }
        (void) hipGraphExecDestroy(e);
        (void) hipGraphDestroy(g);
        (void) hipStreamDestroy(stream);
        ring.free_();
        std::printf("PASS D a graph with no publish node did not ring; the host's 250 ms deadline fired and the "
                    "buffers were released only after the stream was quiesced\n");
    }

    std::printf("PASS Kolibri decode handoff on %s: %d layer steps, %d tokens, payload and answer bitwise\n",
                prop.gcnArchName, kLayers * kTokens, kTokens);
    return 0;
}
