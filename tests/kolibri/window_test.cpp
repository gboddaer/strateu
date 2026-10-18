// tests/kolibri/window_test.cpp - the K1 gate for the chunked-prefill window math (host-side, no GPU).
//
// The defect: k_attn_softmax16 was handed a GLOBAL pos0 while its score buffer is the gathered KV slice
// [first, first+P-1].  For first > 0 - any prompt longer than n_swa - the causal bound min(P, q+1) let rows
// attend positions beyond q, i.e. future tokens.  These cases are the ones the end-to-end gates cannot reach.
#include <cstdint>
#include <cstdio>

#include "strata/kernels/kolibri_window.hpp"

using strata::kolibri::attn_mask_bounds;

static int fails = 0;
static void check(int P, int pos0, int t, int64_t swa, int first, int lo_want, int n_want, const char* why) {
    int lo = -1, n = -1;
    attn_mask_bounds(P, pos0, t, swa, first, &lo, &n);
    if (lo != lo_want || n != n_want) {
        std::printf("FAIL %-46s P %d pos0 %d t %d swa %lld first %d -> lo %d n %d, want lo %d n %d\n", why, P, pos0,
                    t, (long long) swa, first, lo, n, lo_want, n_want);
        ++fails;
    }
}

int main() {
    constexpr int64_t swa = 513;

    // The worked case from the review: p0 = 1024, T = 512, swa = 513 -> P = 513, first = 1023.
    // Row q = 1024 (t = 0) must see global [1023, 1024] = slice [0, 1]; the pre-fix code gave lo 512, n 513,
    // i.e. slice cell 512 = global 1535, a future token, and nothing else.
    check(513, 1024, 0, swa, 1023, 0, 2, "p0=1024 T=512 row 0 (the K1 case)");
    check(513, 1024, 511, swa, 1023, 0, 513, "p0=1024 T=512 last row q=1535 sees the full window");
    check(513, 1024, 200, swa, 1023, 0, 202, "p0=1024 T=512 row 200 q=1224");

    // A sliding row whose window starts inside the slice: q = first + 600 needs global [q-512, q].
    check(700, 1024, 600, swa, 1023, 89, 602, "window starts mid-slice: q=1624 -> global [1112,1624] = slice [89,601]");

    // first = 0 (the first chunk, and every full-attention layer) must be bit-identical to the old math:
    // n = min(P, q+1), lo = max(0, q+1-swa).
    check(512, 0, 0, 0, 0, 0, 1, "full attn, first chunk, row 0");
    check(512, 0, 511, 0, 0, 0, 512, "full attn, first chunk, last row");
    check(1025, 512, 512, 0, 0, 0, 1025, "full attn, second chunk, last row q=1024");
    check(513, 0, 0, swa, 0, 0, 1, "sliding, first chunk, row 0");
    check(513, 0, 512, swa, 0, 0, 513, "sliding, first chunk, last row q=512 (window exactly full)");
    check(513, 1, 512, swa, 1, 0, 513, "sliding, chunk with first=1: q=513 sees 513 rows");

    // Degenerate: a row entirely before the slice must mask everything (lo == n), never read out of range.
    check(8, 100, 0, swa, 200, 0, 0, "row before the slice -> empty, not negative");

    std::printf("kolibri_window_test: %s\n", fails ? "FAIL" : "ok");
    return fails ? 1 : 0;
}
