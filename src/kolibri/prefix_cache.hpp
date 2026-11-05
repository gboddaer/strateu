#pragma once
// Owned host bytes at canonical prefill boundaries; no device or arithmetic code.
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <new>
#include <utility>
#include <vector>

namespace strata::kolibri::prefix {
struct Span { void* data; size_t bytes; };
class Cache {
    struct Entry {
        int64_t position;
        std::vector<size_t> sizes;
        std::vector<uint8_t> data;
    };
    static void host_copy(void* dst, const void* src, size_t bytes) {
        std::memcpy(dst, src, bytes);
    }
    size_t budget_ = 0, used_ = 0;
    std::deque<Entry> entries_;
    size_t checked_size(const std::vector<Span>& spans) const {
        size_t size = 0;
        for (const auto& span : spans) {
            if (!span.data || !span.bytes || span.bytes > budget_ - size) return 0;
            size += span.bytes;
        }
        return size;
    }
public:
    explicit Cache(size_t budget = 0) : budget_(budget) {}
    void clear() { entries_.clear(); used_ = 0; }
    void set_budget(size_t budget) { clear(); budget_ = budget; }
    size_t budget() const { return budget_; }
    size_t bytes() const { return used_; }
    size_t count() const { return entries_.size(); }
    void prune(int64_t prefix) {
        while (!entries_.empty() && entries_.back().position > prefix) {
            used_ -= entries_.back().data.size();
            entries_.pop_back();
        }
    }
    int64_t best_position(int64_t prefix) const {
        for (auto i = entries_.rbegin(); i != entries_.rend(); ++i)
            if (i->position <= prefix) return i->position;
        return 0;
    }
    using Copy = void (*)(void*, const void*, size_t);
    bool capture(int64_t position, const std::vector<Span>& spans, Copy copy = host_copy) {
        const size_t size = checked_size(spans);
        if (position <= 0 || size == 0) return false;
        prune(position - 1); // replace duplicates, invalidate descendant branches
        while (used_ > budget_ - size) {
            used_ -= entries_.front().data.size();
            entries_.pop_front();
        }
        try {
            Entry e{position, {}, std::vector<uint8_t>(size)};
            e.sizes.reserve(spans.size());
            size_t offset = 0;
            for (const auto& span : spans) {
                e.sizes.push_back(span.bytes);
                copy(e.data.data() + offset, span.data, span.bytes);
                offset += span.bytes;
            }
            entries_.push_back(std::move(e)); // only publish complete snapshots
            used_ += size;
            return true;
        } catch (const std::bad_alloc&) {
            return false; // missing checkpoint means cold fallback, never partial state
        }
    }
    int64_t restore(int64_t prefix, const std::vector<Span>& spans, Copy copy = host_copy) const {
        if (!checked_size(spans)) return 0;
        for (auto i = entries_.rbegin(); i != entries_.rend(); ++i) {
            if (i->position > prefix) continue;
            if (i->sizes.size() != spans.size()) return 0;
            for (size_t n = 0; n < spans.size(); ++n)
                if (spans[n].bytes != i->sizes[n]) return 0;
            size_t offset = 0;
            for (const auto& span : spans) {
                copy(span.data, i->data.data() + offset, span.bytes);
                offset += span.bytes;
            }
            return i->position;
        }
        return 0;
    }
};
} // namespace strata::kolibri::prefix
