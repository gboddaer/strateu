// tools/kolibri/expert_stream_replay.cu - Task 1 of plan 2026-10-31-kolibri-gpu-streaming.
//
// The streamed expert prefill's replay harness.  Task 1's half is the CPU control: it replays a capture's expert
// records through the SAME native kernels and the SAME shard fan-out the engine's prefill pool uses, checks the
// bytes against what the engine produced, and times the repeats.  Task 4 adds the GPU arm and the paired gate; the
// flag surface here is already the one that gate will use.
//
//   expert_stream_replay --capture DIR --cpu --cpu-threads 14 --repeat 3 --json OUT
//
// What a replay record contains, and why the CPU control is trustworthy: the engine captured, per (chunk, layer),
// the chunk's published expert ids, its already-quantized token activations, the unique touched blobs in the order
// the pool dealt them, and the expert output rows (ey_h) BEFORE weighting.  Replaying that through
// native_gu_rows/native_quant_h/native_down_rows with the same lane count and the same striding is the same work,
// not a lookalike: the comparison is byte equality, and a single differing float fails the run.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <ggml-cpu.h>

#include "expert_stream_capture.hpp"
#include "kolibri/expert_stream.hpp"
#include "kolibri/expert_stream_plan.hpp"
#include "strata/kernels/cpu/pool.hpp"

namespace capture = strata::kolibri::stream::capture;
namespace stream = strata::kolibri::stream;
namespace cpu = strata::kernels::cpu;

namespace {

struct Options {
    std::string capture_dir;
    std::string json_out;
    bool cpu = false;
    int cpu_threads = 14;
    int repeat = 3;
    bool pin = true;
    size_t blob_align = 2097152;   // 2 MiB: THP-backed like the engine's own (much larger) table
    bool blob_pinned = false;
    bool gpu = false;          ///< Task 4: the streamed GPU arm, timed and gated
    size_t device_budget = 256u << 20;
    int experts_per_batch = 8;
};

/// The engine's prefill fan-out, same contract as `Pool` in kolibri_engine.cu: `run(f, n)` splits [0, n) over n
/// shards, the caller runs shard 0, and the threads are created ONCE (start-up is outside every timed repeat).
struct LanePool {
    int lanes = 1;
    std::vector<int> cores;
    std::vector<std::thread> workers;
    std::mutex mu;
    std::condition_variable work_cv, done_cv;
    std::function<void(int)> fn;
    unsigned long gen = 0;
    int n_shards = 0;
    std::atomic<int> left{0};
    bool quit = false;

    void start(int n_lanes, const std::vector<int>& lane_cores, bool pin) {
        lanes = n_lanes;
        cores = pin ? lane_cores : std::vector<int>();
        for (int i = 1; i < lanes; ++i) {
            workers.emplace_back([this, i] {
                if (!cores.empty()) {
                    const size_t idx = (size_t) (i - 1);
                    if (idx < cores.size() && cores[idx] >= 0) cpu::pin_current_thread(cores[idx]);
                }
                unsigned long my_gen = 0;
                for (;;) {
                    std::unique_lock<std::mutex> lk(mu);
                    work_cv.wait(lk, [&] { return quit || gen > my_gen; });
                    if (quit) return;
                    my_gen = gen;
                    const int n = n_shards;
                    if (i >= n) continue;
                    lk.unlock();
                    fn(i);
                    lk.lock();
                    if (--left == 0) done_cv.notify_one();
                }
            });
        }
    }
    ~LanePool() {
        {
            std::lock_guard<std::mutex> lk(mu);
            quit = true;
        }
        work_cv.notify_all();
        for (auto& w : workers) w.join();
    }
    void run(std::function<void(int)> f, int n) {
        if (n <= 1) {
            f(0);
            return;
        }
        {
            std::lock_guard<std::mutex> lk(mu);
            fn = std::move(f);
            n_shards = n;
            left = n - 1;
            ++gen;
        }
        work_cv.notify_all();
        fn(0);
        std::unique_lock<std::mutex> lk(mu);
        done_cv.wait(lk, [&] { return left.load() == 0; });
    }
};

/// Per-record working buffers, grown to the largest record seen and reused, so no allocation happens in a timed
/// region after the first record.
struct Buffers {
    std::vector<int32_t> cnt;   // experts + 1, the engine's prefix-summed selection counts
    std::vector<int32_t> ord;   // selections, grouped by expert in original selection order
    std::vector<int32_t> cur;
    std::vector<float> ff;      // selections x n_ff
    std::vector<uint8_t> hq;    // selections x h_bytes
    std::vector<float> out;     // selections x n_embd
    // The engine's blobs live in ONE contiguous table at a fixed stride, so consecutive experts are a smooth
    // 2.4 MB stride apart.  The replay reproduces that layout - a single aligned region, blob i at i*stride - rather
    // than one heap block per blob: the difference was measured, and a scattered layout is not the same workload.
    uint8_t* table = nullptr;
    size_t table_bytes = 0;
    size_t stride = 0;
    size_t align = 2097152;
    bool pinned = false;   // A/B only: the engine's table is cudaHostAlloc'd, pageable was faster here

    void ensure_table(size_t n, size_t blob_stride, size_t alignment, bool pin) {
        align = alignment;
        pinned = pin;
        stride = blob_stride;
        const size_t need = n == 0 ? 0 : ((n * blob_stride + alignment - 1) / alignment) * alignment;
        if (need <= table_bytes) return;
        if (table != nullptr) {
            if (pinned) cudaFreeHost(table);
            else std::free(table);
            table = nullptr;
            table_bytes = 0;
        }
        void* p = nullptr;
        if (pinned) {
            if (cudaHostAlloc(&p, need, cudaHostAllocDefault) != cudaSuccess)
                throw std::runtime_error("replay: cannot allocate a pinned blob table");
        } else if (posix_memalign(&p, alignment, need) != 0) {
            throw std::runtime_error("replay: cannot allocate the blob table");
        }
        table = (uint8_t*) p;
        table_bytes = need;
    }
    const uint8_t* blob(size_t i) const { return table + i * stride; }
    ~Buffers() {
        if (table != nullptr) {
            if (pinned) cudaFreeHost(table);
            else std::free(table);
        }
    }

    void ensure(const capture::Record& r) {
        const size_t selections = (size_t) r.selections;
        const size_t experts = (size_t) r.experts + 1;
        if (cnt.size() < experts) cnt.resize(experts);
        if (cur.size() < experts) cur.resize(experts);
        if (ord.size() < selections) ord.resize(selections);
        if (ff.size() < selections * (size_t) r.fmt.n_ff) ff.resize(selections * (size_t) r.fmt.n_ff);
        if (hq.size() < selections * (size_t) r.fmt.h_bytes) hq.resize(selections * (size_t) r.fmt.h_bytes);
        if (out.size() < selections * (size_t) r.fmt.n_embd) out.resize(selections * (size_t) r.fmt.n_embd);
        ensure_table(r.blobs.size(), (size_t) r.weight_stride, align, pinned);
    }
};

/// The GPU arm copies each expert's blob from a table indexed by EXPERT ID at the model's stride, because that is
/// the layout the engine's own pinned expert table has (LayerView::weights).  The CPU arm keeps its packed table:
/// its measurements must not move.  Pinned, so the H2D is genuinely asynchronous; the engine's table is pinned too.
struct ExpertTable {
    uint8_t* table = nullptr;
    size_t bytes = 0;
    size_t stride = 0;
    bool pinned = true;

    void ensure(int experts, size_t stride_) {
        const size_t need = (size_t) experts * stride_;
        if (need <= bytes && stride_ == stride) return;
        release();
        void* p = nullptr;
        cudaError_t e = pinned ? cudaHostAlloc(&p, need, cudaHostAllocDefault) : cudaMallocHost(&p, need);
        if (e != cudaSuccess || p == nullptr) {   // a pinned table of ~1 GB can be refused; report it, don't fall back silently
            throw std::runtime_error(std::string("replay: cannot allocate the expert table: ") + cudaGetErrorString(e));
        }
        table = (uint8_t*) p;
        bytes = need;
        stride = stride_;
    }
    void release() {
        if (table != nullptr) cudaFreeHost(table);
        table = nullptr;
        bytes = 0;
    }
    ~ExpertTable() { release(); }
};

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

std::string hex_float(float v) {
    uint32_t bits = 0;
    std::memcpy(&bits, &v, 4);
    char buf[16];
    std::snprintf(buf, sizeof buf, "0x%08x", bits);
    return buf;
}

/// Byte-compares a replay record's expert output with the engine's captured rows, naming the first disagreement.
bool same_output(const capture::Record& r, const std::vector<float>& out, std::string& why) {
    const size_t n = (size_t) r.selections * (size_t) r.fmt.n_embd;
    if (out.size() < n) {
        why = "the replay buffer is smaller than the record";
        return false;
    }
    if (std::memcmp(out.data(), r.out.data(), n * sizeof(float)) == 0) return true;
    for (int sel = 0; sel < r.selections; ++sel) {
        for (int64_t i = 0; i < r.fmt.n_embd; ++i) {
            const float a = out[(size_t) sel * (size_t) r.fmt.n_embd + (size_t) i];
            const float b = r.out[(size_t) sel * (size_t) r.fmt.n_embd + (size_t) i];
            if (std::memcmp(&a, &b, sizeof(float)) != 0) {
                char buf[256];
                std::snprintf(buf, sizeof buf,
                              "layer %d chunk %lld selection %d embedding %lld: replay %s (%g) != capture %s (%g)",
                              (int) r.layer, (long long) r.chunk, sel, (long long) i, hex_float(a).c_str(),
                              (double) a, hex_float(b).c_str(), (double) b);
                why = buf;
                return false;
            }
        }
    }
    why = "outputs differ but no differing element was found";   // cannot happen; never silently pass
    return false;
}

/// One record through the engine's own prefill leg: grouping, then the shard fan-out, then the 64-selection CPU
/// call chunks the engine uses (`for (a = 0; a < m; a += 64)` in kolibri_engine.cu's pool block).
void replay_record(const capture::Record& r, const cpu::NativeFmt& fmt, int lane, int ns, Buffers& buf) {
    const int top_k = r.top_k;
    const size_t n_ff = (size_t) fmt.n_ff;
    const size_t n_embd = (size_t) fmt.n_embd;
    for (size_t xi = (size_t) lane; xi < r.blobs.size(); xi += (size_t) ns) {
        const int x = r.blobs[xi].expert;
        const int lo = buf.cnt[(size_t) x], hi = buf.cnt[(size_t) x + 1], m = hi - lo;
        if (m <= 0) continue;
        const uint8_t* blob = buf.blob(xi);
        const void* acts[64];
        float* ffs[64];
        const void* hqs[64];
        float* eys[64];
        for (int a = 0; a < m; a += 64) {
            const int mm = std::min(64, m - a);
            for (int k = 0; k < mm; ++k) {
                const int sel = buf.ord[(size_t) lo + a + k];
                acts[k] = r.acts.data() + (size_t) (sel / top_k) * (size_t) fmt.act_bytes;
                ffs[k] = buf.ff.data() + (size_t) sel * n_ff;
            }
            cpu::native_gu_rows(fmt, blob, acts, mm, ffs, 0, (int) n_ff);
            for (int k = 0; k < mm; ++k) {
                const int sel = buf.ord[(size_t) lo + a + k];
                uint8_t* hqp = buf.hq.data() + (size_t) sel * (size_t) fmt.h_bytes;
                cpu::native_quant_h(fmt, ffs[k], hqp);
                hqs[k] = hqp;
                eys[k] = buf.out.data() + (size_t) sel * n_embd;
            }
            cpu::native_down_rows(fmt, blob, hqs, mm, eys, 0, (int) n_embd);
        }
    }
}

/// The engine's grouping, in the same order it builds it: counts, prefix sums, then one pass over the selections.
void build_groups(const capture::Record& r, Buffers& buf) {
    std::fill(buf.cnt.begin(), buf.cnt.begin() + (size_t) r.experts + 1, 0);
    for (int sel = 0; sel < r.selections; ++sel) ++buf.cnt[(size_t) r.ids[(size_t) sel] + 1];
    for (int x = 0; x < r.experts; ++x) buf.cnt[(size_t) x + 1] += buf.cnt[(size_t) x];
    std::copy(buf.cnt.begin(), buf.cnt.begin() + (size_t) r.experts + 1, buf.cur.begin());
    for (int sel = 0; sel < r.selections; ++sel) buf.ord[(size_t) buf.cur[(size_t) r.ids[(size_t) sel]]++] = sel;
}

std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') {
            o.push_back('\\');
            o.push_back(c);
        } else if ((unsigned char) c < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof buf, "\\u%04x", c);
            o += buf;
        } else {
            o.push_back(c);
        }
    }
    return o;
}

double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int usage() {
    std::fprintf(stderr,
                 "usage: expert_stream_replay --capture DIR --cpu [--cpu-threads N] [--repeat N] [--json OUT]\n"
                 "                            [--no-pin]\n"
                 "  --cpu            replay the CPU expert leg (the control arm)\n"
                 "  --cpu-threads N  lanes of the engine's prefill fan-out (default 14)\n"
                 "  --repeat N       timed repeats after one untimed correctness pass (default 3)\n"
                 "  --blob-align N   alignment of the replayed blob table (default 2097152: 2 MiB, THP-backed)\n"
                 "  --blob-pinned    hold the replayed blobs in cudaHostAlloc memory, as the engine's table is\n"
                 "  --json OUT       write the report to OUT (otherwise stdout)\n"
                 "  --gpu            replay the same records through the streamed GPU pipeline (Task 4's arm)\n"
                 "  --experts-per-batch N  GPU arm: groups per slot batch (default 8)\n"
                 "  --device-budget BYTES  GPU arm: cap on the pipeline's new device memory (default 256 MiB)\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "expert_stream_replay: %s needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--capture") {
            opt.capture_dir = next();
        } else if (a == "--cpu") {
            opt.cpu = true;
        } else if (a == "--cpu-threads") {
            opt.cpu_threads = std::atoi(next().c_str());
        } else if (a == "--repeat") {
            opt.repeat = std::atoi(next().c_str());
        } else if (a == "--json") {
            opt.json_out = next();
        } else if (a == "--no-pin") {
            opt.pin = false;
        } else if (a == "--blob-align") {
            opt.blob_align = (size_t) std::atoll(next().c_str());
        } else if (a == "--blob-pinned") {
            opt.blob_pinned = true;
        } else if (a == "--gpu") {
            opt.gpu = true;
        } else if (a == "--experts-per-batch") {
            opt.experts_per_batch = std::atoi(next().c_str());
        } else if (a == "--device-budget") {
            opt.device_budget = (size_t) std::atoll(next().c_str());
        } else if (a == "--help" || a == "-h") {
            return usage();
        } else {
            std::fprintf(stderr, "expert_stream_replay: unknown argument %s\n", a.c_str());
            return usage();
        }
    }
    if (opt.capture_dir.empty() || (!opt.cpu && !opt.gpu)) {
        std::fprintf(stderr, "expert_stream_replay: --capture DIR and --cpu or --gpu are required\n");
        return usage();
    }
    if (opt.cpu_threads < 2) {
        std::fprintf(stderr, "expert_stream_replay: --cpu-threads must be at least 2\n");
        return 2;
    }
    if (opt.repeat < 1) {
        std::fprintf(stderr, "expert_stream_replay: --repeat must be at least 1\n");
        return 2;
    }
    ggml_cpu_init();   // the ggml type traits the native kernels call

    // The engine's lane rule: the host lane plus one per pinnable physical core, capped by --cpu-threads.
    if (opt.gpu) {
        int dev = 0;
        cudaDeviceProp prop{};
        if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess ||
            !(prop.major > 7 || (prop.major == 7 && prop.minor >= 5))) {
            std::fprintf(stderr, "expert_stream_replay: --gpu needs a CUDA device with sm_75 or newer\n");
            return 2;
        }
        std::printf("expert_stream_replay: GPU arm on %s, sm_%d%d\n", prop.name, prop.major, prop.minor);
    }
    const std::vector<int> pf_cores = cpu::physical_cores(true);
    int lanes = opt.cpu_threads;
    const int lanes_cap = pf_cores.empty() ? opt.cpu_threads : (int) pf_cores.size() + 1;
    if (lanes > lanes_cap) lanes = lanes_cap;
    if (lanes < 1) lanes = 1;

    LanePool pool;
    pool.start(lanes, pf_cores, opt.pin);

    capture::Reader reader(opt.capture_dir);
    Buffers buf;
    buf.align = opt.blob_align;
    buf.pinned = opt.blob_pinned;
    std::vector<std::string> record_lines;
    std::vector<std::vector<double>> record_ms;   // [record][repeat]
    std::vector<double> totals((size_t) opt.repeat, 0.0);
    int records = 0;
    int64_t selections_total = 0;
    double load_total = 0.0;
    // GPU arm state: one context for the whole capture (its creation is first-use cost, reported separately), one
    // expert-indexed pinned table, and the per-repeat totals.
    stream::Config gcfg;
    gcfg.experts_per_batch = opt.experts_per_batch;
    gcfg.device_budget_bytes = opt.device_budget;
    std::unique_ptr<stream::Context> gctx;
    double create_ms = 0.0;
    ExpertTable gtable;
    std::vector<float> gpu_out;
    std::vector<std::vector<double>> gpu_record_ms;
    std::vector<double> gpu_totals((size_t) opt.repeat, 0.0);
    stream::Stats gstats{};
    uint64_t gpu_records = 0;
    double gpu_cpu_activation_ms = 0.0;

    // **RECORD-MAJOR REPEATS.**  Each record's blobs are read from the store ONCE and then replayed `repeat` times.
    // A pass-major loop would re-read the whole store per repeat (measured: 26 minutes of ZFS reads for three
    // repeats of the 19 GB short capture), and the disk is outside both arms anyway.  Repeated records see the same
    // ~281 MB working set from RAM - not from L3, which is 60 MB - so the repeats differ from a streaming pass only
    // by that much, and every repeat is reported separately rather than averaged into one number.
    {
        capture::Record rec;
        while (reader.next(rec)) {
            const cpu::NativeFmt fmt = rec.fmt.to_native();
            buf.ensure(rec);
            if (opt.gpu) gtable.ensure(rec.experts, (size_t) rec.weight_stride);
            const auto load_t0 = std::chrono::steady_clock::now();
            if (opt.cpu)
                for (size_t i = 0; i < rec.blobs.size(); ++i)
                    reader.load_into(rec.blobs[i], const_cast<uint8_t*>(buf.blob(i)), (size_t) rec.weight_stride);
            if (opt.gpu)
                for (size_t i = 0; i < rec.blobs.size(); ++i)
                    reader.load_into(rec.blobs[i], gtable.table + (size_t) rec.blobs[i].expert * gtable.stride,
                                     (size_t) rec.weight_stride);
            const double load_ms = ms_since(load_t0);
            load_total += load_ms;
            std::vector<double> times((size_t) opt.repeat, 0.0);
            if (opt.cpu) {
                std::fill(buf.out.begin(), buf.out.begin() + (size_t) rec.selections * (size_t) fmt.n_embd,
                          std::numeric_limits<float>::quiet_NaN());
                const int ns = std::min(rec.experts, lanes);
                for (int rep = 0; rep < opt.repeat; ++rep) {
                    const auto t0 = std::chrono::steady_clock::now();
                    build_groups(rec, buf);
                    pool.run([&](int lane) { replay_record(rec, fmt, lane, ns, buf); }, ns);
                    times[(size_t) rep] = ms_since(t0);
                    totals[(size_t) rep] += times[(size_t) rep];
                    std::string why;
                    if (!same_output(rec, buf.out, why)) {
                        std::fprintf(stderr, "expert_stream_replay: CPU replay does not reproduce the capture: %s\n",
                                     why.c_str());
                        return 1;
                    }
                }
            }
            std::vector<double> gtimes((size_t) opt.repeat, 0.0);
            if (opt.gpu) {
                stream::LayerView lv;
                lv.fmt = fmt;
                lv.weights = gtable.table;
                lv.weight_stride = gtable.stride;
                lv.expert_count = rec.experts;
                lv.token_acts = rec.acts.data();
                lv.expert_ids = rec.ids.data();
                lv.selection_count = rec.selections;
                lv.top_k = rec.top_k;
                gpu_out.assign((size_t) rec.selections * (size_t) fmt.n_embd,
                               std::numeric_limits<float>::quiet_NaN());
                lv.output = gpu_out.data();
                if (!gctx) {
                    const auto c0 = std::chrono::steady_clock::now();
                    gctx = stream::create(lv, gcfg);
                    create_ms += ms_since(c0);   // first-use cost, reported, never inside a timed repeat
                }
                for (int rep = 0; rep < opt.repeat; ++rep) {
                    // poison every element first: a selection the scatter forgets keeps NaN and fails the compare
                    std::fill(gpu_out.begin(), gpu_out.end(), std::numeric_limits<float>::quiet_NaN());
                    stream::Stats st{};
                    const auto t0 = std::chrono::steady_clock::now();
                    try {
                        st = stream::run(*gctx, lv);
                    } catch (const std::exception& ex) {
                        std::fprintf(stderr, "expert_stream_replay: GPU replay failed: %s\n", ex.what());
                        return 1;
                    }
                    gtimes[(size_t) rep] = ms_since(t0);
                    gpu_totals[(size_t) rep] += gtimes[(size_t) rep];
                    gstats.groups += st.groups;
                    gstats.selections += st.selections;
                    gstats.weight_h2d_bytes += st.weight_h2d_bytes;
                    gstats.activation_h2d_bytes += st.activation_h2d_bytes;
                    gstats.gu_d2h_bytes += st.gu_d2h_bytes;
                    gstats.hidden_h2d_bytes += st.hidden_h2d_bytes;
                    gstats.output_d2h_bytes += st.output_d2h_bytes;
                    gstats.kernel_launches += st.kernel_launches;
                    gstats.slot_waits += st.slot_waits;
                    gstats.device_allocations += st.device_allocations;
                    gstats.wall_ms += st.wall_ms;
                    gstats.cpu_activation_ms += st.cpu_activation_ms / (double) opt.repeat;
                    std::string why;
                    if (!same_output(rec, gpu_out, why)) {
                        std::fprintf(stderr, "expert_stream_replay: GPU replay does not reproduce the capture: %s\n",
                                     why.c_str());
                        return 1;
                    }
                }
                gpu_cpu_activation_ms = gstats.cpu_activation_ms;   // the running total, not a sum of totals
                ++gpu_records;
            }
            char line[256];
            std::snprintf(line, sizeof line,
                          "  record %d: layer %d chunk %lld selections %d tokens %d blobs %zu down q%d\n", records,
                          (int) rec.layer, (long long) rec.chunk, rec.selections, rec.tokens, rec.blobs.size(),
                          (int) rec.fmt.d_type);
            record_lines.push_back(line);
            record_ms.push_back(times);
            gpu_record_ms.push_back(gtimes);
            selections_total += rec.selections;
            ++records;
            if (records % 100 == 0)
                std::fprintf(stderr, "expert_stream_replay: %d records replayed (cpu %.0f ms, gpu %.0f ms so far)\n",
                             records, totals[0], gpu_totals[0]);
        }
    }

    if (records == 0) {
        std::fprintf(stderr, "expert_stream_replay: the capture has no records\n");
        return 1;
    }

    std::string json;
    json += "{\n";
    json += "  \"capture\": \"" + json_escape(opt.capture_dir) + "\",\n";
    json += "  \"arm\": \"" + std::string(opt.cpu && opt.gpu ? "cpu+gpu" : (opt.gpu ? "gpu" : "cpu")) + "\",\n";
    json += "  \"records\": " + std::to_string(records) + ",\n";
    json += "  \"selections\": " + std::to_string(selections_total) + ",\n";
    json += "  \"cpu_threads\": " + std::to_string(opt.cpu_threads) + ",\n";
    json += "  \"lanes\": " + std::to_string(lanes) + ",\n";
    json += "  \"pin\": " + std::string(opt.pin ? "true" : "false") + ",\n";
    json += "  \"blob_align\": " + std::to_string(opt.blob_align) + ",\n";
    json += "  \"blob_memory\": \"" + std::string(opt.blob_pinned ? "pinned" : "pageable") + "\",\n";
    json += "  \"repeat\": " + std::to_string(opt.repeat) + ",\n";
    json += "  \"cpu_ms\": [";
    for (size_t i = 0; i < totals.size(); ++i) json += (i ? ", " : "") + std::to_string(opt.cpu ? totals[i] : 0.0);
    json += "],\n";
    json += "  \"cpu_ms_median\": " + std::to_string(opt.cpu ? median_of(totals) : 0.0) + ",\n";
    json += "  \"gpu\": " + std::string(opt.gpu ? "true" : "false") + ",\n";
    json += "  \"experts_per_batch\": " + std::to_string(opt.experts_per_batch) + ",\n";
    json += "  \"device_budget_bytes\": " + std::to_string(opt.device_budget) + ",\n";
    json += "  \"gpu_ms\": [";
    for (size_t i = 0; i < gpu_totals.size(); ++i) json += (i ? ", " : "") + std::to_string(gpu_totals[i]);
    json += "],\n";
    json += "  \"gpu_ms_median\": " + std::to_string(opt.gpu ? median_of(gpu_totals) : 0.0) + ",\n";
    json += "  \"gpu_create_ms\": " + std::to_string(create_ms) + ",\n";
    json += "  \"gpu_weight_h2d_bytes\": " + std::to_string((unsigned long long) gstats.weight_h2d_bytes) + ",\n";
    json += "  \"gpu_activation_h2d_bytes\": " + std::to_string((unsigned long long) gstats.activation_h2d_bytes) +
            ",\n";
    json += "  \"gpu_hidden_h2d_bytes\": " + std::to_string((unsigned long long) gstats.hidden_h2d_bytes) + ",\n";
    json += "  \"gpu_gu_d2h_bytes\": " + std::to_string((unsigned long long) gstats.gu_d2h_bytes) + ",\n";
    json += "  \"gpu_output_d2h_bytes\": " + std::to_string((unsigned long long) gstats.output_d2h_bytes) + ",\n";
    json += "  \"gpu_kernel_launches\": " + std::to_string((unsigned long long) gstats.kernel_launches) + ",\n";
    json += "  \"gpu_slot_waits\": " + std::to_string((unsigned long long) gstats.slot_waits) + ",\n";
    json += "  \"gpu_device_allocations_in_run\": " + std::to_string((unsigned long long) gstats.device_allocations) +
            ",\n";
    json += "  \"gpu_cpu_activation_ms\": " + std::to_string(gpu_cpu_activation_ms) + ",\n";
    json += "  \"gpu_copy_gbps\": " +
            std::to_string(median_of(gpu_totals) > 0
                               ? (double) gstats.weight_h2d_bytes / (double) opt.repeat /
                                     (median_of(gpu_totals) * 1.0e6)
                               : 0.0) +
            ",\n";
    json += "  \"gpu_record_ms\": [";
    for (size_t i = 0; i < gpu_record_ms.size(); ++i)
        json += (i ? ", " : "") + std::to_string(opt.gpu ? median_of(gpu_record_ms[i]) : 0.0);
    json += "],\n";
    json += "  \"load_ms_total\": " + std::to_string(load_total) + ",\n";
    json += "  \"record_ms\": [";
    for (size_t i = 0; i < record_ms.size(); ++i)
        json += (i ? ", " : "") + std::to_string(median_of(record_ms[i]));
    json += "],\n";
    json += "  \"note\": \"record-major repeats: each record's blobs are read once and replayed N times; loader, "
            "blob store reads and thread start-up are outside cpu_ms; grouping, the shard fan-out and every native "
            "call are inside it\",\n";
    json += "  \"manifest\": \"" + json_escape(reader.manifest()) + "\"\n";
    json += "}\n";

    if (opt.cpu)
        std::printf("expert_stream_replay: CPU control reproduces the capture bit for bit; %d records, %lld "
                    "selections, %d lanes, median %.1f ms over %d repeat(s)\n",
                    records, (long long) selections_total, lanes, median_of(totals), opt.repeat);
    if (opt.gpu)
        std::printf("expert_stream_replay: GPU arm reproduces the capture bit for bit; %d records, %llu unique-batch "
                    "weight bytes, median %.1f ms over %d repeat(s), create %.1f ms, CPU activation %.1f ms total\n",
                    (int) gpu_records, (unsigned long long) gstats.weight_h2d_bytes, median_of(gpu_totals),
                    opt.repeat, create_ms, gpu_cpu_activation_ms);
    for (const std::string& line : record_lines) std::fputs(line.c_str(), stdout);
    if (!opt.json_out.empty()) {
        std::ofstream out(opt.json_out, std::ios::trunc);
        if (!out) {
            std::fprintf(stderr, "expert_stream_replay: cannot write %s\n", opt.json_out.c_str());
            return 1;
        }
        out << json;
        if (!out) {
            std::fprintf(stderr, "expert_stream_replay: cannot write %s\n", opt.json_out.c_str());
            return 1;
        }
    } else {
        std::fputs(json.c_str(), stdout);
    }
    return 0;
}
