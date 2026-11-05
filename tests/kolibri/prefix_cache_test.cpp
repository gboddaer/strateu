#include "kolibri/prefix_cache.hpp"
#include <cassert>
#include <cstdint>
#include <limits>
#include <vector>
using strata::kolibri::prefix::Cache;
using strata::kolibri::prefix::Span;
static int copies = 0;
static void counted_copy(void* dst, const void* src, size_t bytes) {
    ++copies;
    std::memcpy(dst, src, bytes);
}
int main() {
    uint8_t k[8] = {1}, v[8] = {2};
    std::vector<Span> spans{{k, sizeof k}, {v, sizeof v}};
    Cache c(32);
    assert(c.capture(512, spans));
    k[0] = 3; v[0] = 4;
    assert(c.restore(511, spans) == 0);
    assert(c.restore(512, spans) == 512 && k[0] == 1 && v[0] == 2);
    k[0] = 5;
    assert(c.capture(1024, spans));
    assert(c.count() == 2 && c.bytes() == 32);
    k[0] = 6;
    assert(c.capture(1536, spans));
    assert(c.count() == 2 && c.best_position(512) == 512);
    assert(c.best_position(2000) == 1536); // newest survives alongside the early anchor
    assert(c.restore(1200, spans) == 512 && k[0] == 1);
    c.prune(1100);
    assert(c.count() == 1 && c.best_position(2000) == 512);
    c.clear();
    assert(c.capture(1024, spans));
    k[0] = 7;
    assert(c.capture(1024, spans));
    assert(c.count() == 1 && c.bytes() == 16);
    k[0] = 9;
    assert(c.restore(1024, spans) == 1024 && k[0] == 7);
    assert(c.restore(1024, {{k, 7}, {v, 8}}) == 0);
    c.clear();
    assert(c.count() == 0 && c.restore(9999, spans) == 0);
    Cache transport(32);
    assert(transport.capture(512, spans, counted_copy));
    assert(transport.restore(512, spans, counted_copy) == 512);
    assert(copies == 4); // device adapters use both transfer directions, not host dereferences
    Cache failure(32);
    assert(failure.capture(512, spans));
    auto fail_copy = [](void*, const void*, size_t) { throw std::bad_alloc(); };
    assert(!failure.capture(1024, spans, fail_copy));
    assert(failure.count() == 1 && failure.bytes() == 16);
    assert(failure.best_position(2000) == 512); // failed copies never publish partial state
    Cache disabled(0), tiny(15);
    assert(!disabled.capture(512, spans) && !tiny.capture(512, spans));
    assert(!c.capture(0, spans) && !c.capture(512, {}));
    assert(!c.capture(512, {{nullptr, 8}}));
    assert(!c.capture(512, {{k, std::numeric_limits<size_t>::max()}, {v, 8}}));
}
