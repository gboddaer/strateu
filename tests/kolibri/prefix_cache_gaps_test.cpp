// Former RED tests: required retention properties, now in the regular CPU suite.
// Also run through tools/kolibri/known_gaps.py.
#include "kolibri/prefix_cache.hpp"
#include <cstdio>
#include <cstring>
#include <vector>
using strata::kolibri::prefix::Cache;
using strata::kolibri::prefix::Span;

static int red = 0;
static void check(const char* name, bool ok, const char* detail) {
    std::printf("%s  %s  (%s)\n", ok ? "GREEN" : "RED  ", name, detail);
    if (!ok) ++red;
}

int main() {
    unsigned char k[8] = {1}, v[8] = {2};
    std::vector<Span> spans{{k, sizeof k}, {v, sizeof v}};
    // Three frames of 16 bytes: the default 128 MiB budget with a 40 MiB frame.
    Cache c(48);
    for (int64_t position = 512; position <= 2560; position += 512)
        c.capture(position, spans);
    std::printf("frames held: %zu, bytes: %zu\n", c.count(), c.bytes());

    // Gap 1: a client that re-sends a SHORTER history (a truncated transcript, or a
    // turn regenerated with fewer messages) is an exact prefix of what the engine
    // held, but the only boundary below it was evicted by the newest-first FIFO.
    check("shorter-history-keeps-an-older-boundary", c.best_position(600) == 512,
          "require boundary 512 for a 600-id follow-up");

    // Gap 2: an edit in the middle of the held history (a changed early system
    // message, a retried tool result) can only reuse a boundary below the edit.
    Cache edit(48);
    for (int64_t position = 512; position <= 2560; position += 512)
        edit.capture(position, spans);
    edit.prune(1500); // the edit is at 1500: later boundaries are another branch
    check("edit-in-the-middle-keeps-the-boundary-below-it", edit.best_position(1500) == 1024,
          "require boundary 1024 below an edit at 1500");

    // These are the same two properties the engine-level runner measures end to end;
    // both must keep useful anchors within the unchanged three-frame budget.
    std::printf("%d known gap(s) still open\n", red);
    return red == 0 ? 0 : 1;
}
