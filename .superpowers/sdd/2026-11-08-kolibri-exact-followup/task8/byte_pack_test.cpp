// tests/kolibri/byte_pack_test.cpp - the opt-in two-lane packer is byte-exact, bounded and drained.
#include "strata/kolibri/byte_pack_pool.hpp"

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using strata::kolibri::BytePackPool;

int main() {
    int fails = 0;
    auto check = [&](bool ok, const char* what) {
        if (!ok) { std::printf("FAIL: %s\n", what); ++fails; }
    };
    std::mt19937 rng(0x9e37U);
    for (int lanes : {1, 2, 3, 4}) {
        BytePackPool pool(lanes);
        const std::size_t sizes[4] = {0, 1, 37, 4099};
        for (std::size_t n : sizes) {
            std::vector<unsigned char> a(n), b(n + 3), c(n + 11), dst(n + b.size() + c.size() + 32, 0xA5);
            for (auto& x : a) x = (unsigned char) rng();
            for (auto& x : b) x = (unsigned char) rng();
            for (auto& x : c) x = (unsigned char) rng();
            BytePackPool::Span spans[3] = {
                {a.data(), dst.data() + 3, a.size()},
                {b.data(), dst.data() + 3 + a.size() + 5, b.size()},
                {c.data(), dst.data() + 3 + a.size() + 5 + b.size() + 7, c.size()},
            };
            for (int repeat = 0; repeat < 20; ++repeat) {
                std::memset(dst.data(), 0xA5, dst.size());
                pool.pack(spans, 3);
                check(std::memcmp(dst.data(), "\xA5\xA5\xA5", 3) == 0, "prefix padding changed");
                check(std::memcmp(dst.data() + 3, a.data(), a.size()) == 0 || a.empty(), "span a differs");
                check(std::memcmp(dst.data() + 3 + a.size(), "\xA5\xA5\xA5\xA5\xA5", 5) == 0, "gap a/b changed");
                check(std::memcmp(dst.data() + 3 + a.size() + 5, b.data(), b.size()) == 0 || b.empty(), "span b differs");
                check(std::memcmp(dst.data() + 3 + a.size() + 5 + b.size(), "\xA5\xA5\xA5\xA5\xA5\xA5\xA5", 7) == 0,
                      "gap b/c changed");
                check(std::memcmp(dst.data() + 3 + a.size() + 5 + b.size() + 7, c.data(), c.size()) == 0 || c.empty(),
                      "span c differs");
                check(dst.back() == 0xA5, "suffix padding changed");
                // The source may be overwritten as soon as pack returns.
                if (!a.empty()) { std::memset(a.data(), 0xEE, a.size()); }
                pool.pack(spans, 3);
            }
            check(std::memcmp(dst.data() + 3, b.data(), 0) == 0, "no-op");
        }
        // One-span and partial layouts.
        unsigned char src[17], out[19];
        for (auto& x : src) x = (unsigned char) rng();
        std::memset(out, 0, sizeof out);
        BytePackPool::Span one{src, out + 1, 17};
        pool.pack(&one, 1);
        check(std::memcmp(out + 1, src, 17) == 0, "single span differs");
        check(out[0] == 0 && out[18] == 0, "single span padding changed");
    }
    if (fails) { std::printf("byte_pack: %d failure(s)\n", fails); return 1; }
    std::printf("byte_pack: byte-exact, repeated, drained\n");
    return 0;
}
