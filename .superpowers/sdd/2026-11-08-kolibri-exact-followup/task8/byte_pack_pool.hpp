#pragma once
// Bounded two-lane byte packing: the caller copies lane 0, one persistent worker copies lane 1.  No arithmetic, no
// queue growth, no whole-model copy.  pack() returns only after every span is complete, so the source may be
// released and the caller may start DMA.
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace strata::kolibri {

class BytePackPool {
public:
    struct Span {
        const void* src;
        void* dst;
        std::size_t bytes;
    };
    explicit BytePackPool(int total_threads) {
        lanes_ = total_threads < 2 ? 1 : (total_threads > 4 ? 4 : total_threads);
        for (int i = 1; i < lanes_; ++i) workers_.emplace_back([this, i] { worker(i); });
    }
    ~BytePackPool() {
        {
            std::lock_guard<std::mutex> lk(m_);
            quit_ = true;
            work_.notify_all();
        }
        for (auto& t : workers_) t.join();
    }
    BytePackPool(const BytePackPool&) = delete;
    BytePackPool& operator=(const BytePackPool&) = delete;

    void pack(const Span* spans, int n) {
        if (lanes_ == 1 || n <= 0) {
            for (int i = 0; i < n; ++i) std::memcpy(spans[i].dst, spans[i].src, spans[i].bytes);
            return;
        }
        {
            std::unique_lock<std::mutex> lk(m_);
            if (quit_) {   // a destroyed pool must not silently skip bytes
                lk.unlock();
                for (int i = 0; i < n; ++i) std::memcpy(spans[i].dst, spans[i].src, spans[i].bytes);
                return;
            }
            spans_ = spans;
            n_ = n;
            pending_ = lanes_ - 1;
            ++generation_;
            work_.notify_all();
        }
        for (int i = 0; i < n; i += lanes_)   // caller is lane 0
            std::memcpy(spans[i].dst, spans[i].src, spans[i].bytes);
        std::unique_lock<std::mutex> lk(m_);
        done_.wait(lk, [this] { return pending_ == 0; });
    }

private:
    void worker(int id) {
        std::uint64_t seen = 0;
        for (;;) {
            const Span* spans = nullptr;
            int n = 0, lane = id;
            {
                std::unique_lock<std::mutex> lk(m_);
                work_.wait(lk, [this, &seen] { return quit_ || generation_ != seen; });
                if (quit_) return;
                seen = generation_;
                spans = spans_;
                n = n_;
            }
            for (int i = lane; i < n; i += lanes_) std::memcpy(spans[i].dst, spans[i].src, spans[i].bytes);
            {
                std::lock_guard<std::mutex> lk(m_);
                if (--pending_ == 0) done_.notify_all();
            }
        }
    }
    int lanes_ = 1;
    std::mutex m_;
    std::condition_variable work_, done_;
    std::vector<std::thread> workers_;
    const Span* spans_ = nullptr;
    int n_ = 0, pending_ = 0;
    std::uint64_t generation_ = 0;
    bool quit_ = false;
};

}  // namespace strata::kolibri
