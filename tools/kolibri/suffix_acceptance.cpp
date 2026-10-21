// tools/kolibri/suffix_acceptance.cpp - task 13 (P4): the suffix drafter's acceptance on the MODEL's OWN
// outputs.
//
// The flagship's SuffixDrafter is model-free; this helper feeds it (history, target) and reports what the
// drafter would have proposed and how much of it the target actually continues with - the acceptance rate
// that decides the P4 speedup on real model output (the text-only simulator measures the drafter, not the
// model).
//
//   suffix_drafter_accept PROMPT.ids GEN.ids [K]
//
// Output: "<accept_per_proposal> <proposal_rate> <tokens_per_step>" (or "-" when the drafter proposed nothing).
#include "strata/spec/suffix_drafter.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using strata::spec::SuffixDrafter;

static std::vector<int32_t> read_ids(const char* path) {
    std::ifstream f(path);
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    for (char& c : text) if (c == ',') c = ' ';
    std::istringstream in(text);
    std::vector<int32_t> ids;
    for (int32_t v; in >> v;) ids.push_back(v);
    return ids;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: suffix_drafter_accept PROMPT.ids GEN.ids [K]\n");
        return 2;
    }
    const int K = argc > 3 ? std::atoi(argv[3]) : 8;
    const std::vector<int32_t> hist = read_ids(argv[1]);
    const std::vector<int32_t> tgt = read_ids(argv[2]);
    if (hist.empty() || tgt.empty()) {
        std::printf("- - -\n");
        return 0;
    }
    SuffixDrafter d(3, 32, hist.size() + tgt.size() + 16);
    d.append(hist.data(), hist.size());
    long steps = 0, tokens = 0, proposals = 0, accepted = 0;
    std::vector<int32_t> draft(K);
    size_t i = 0;
    while (i < tgt.size()) {
        const int n = d.propose(K, draft.data());
        int a = 0;
        while (a < n && i + a < tgt.size() && draft[a] == tgt[i + a]) ++a;
        if (n > 0) { ++proposals; accepted += a; }
        const size_t commit = std::min(tgt.size() - i, (size_t) a + 1);
        d.append(&tgt[i], commit);
        i += commit;
        tokens += (long) commit;
        ++steps;
    }
    const double acc = proposals ? (double) accepted / proposals : 0.0;
    const double rate = steps ? (double) proposals / steps : 0.0;
    const double tok_step = steps ? (double) tokens / steps : 0.0;
    std::printf("%.4f %.4f %.4f\n", acc, rate, tok_step);
    return 0;
}
