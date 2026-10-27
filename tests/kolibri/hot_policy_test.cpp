// tests/kolibri/hot_policy_test.cpp - the expert-cache residency policy, without a GPU or a model.
//
// Every case here is a rule from include/strata/kernels/kolibri_hot_policy.hpp.  The engine keeps only the execution
// (install the map, copy the blobs), so these are the tests that say whether a round moves the right experts: a wrong
// budget, a decay applied before the decision instead of after, or a tie broken by pointer order all produce a
// plausible token and a hit rate that looks fine on one prompt.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "strata/kernels/kolibri_hot_policy.hpp"

using namespace strata::kolibri;

static int fails = 0;

static int check_double(double got, double want, const char* why) {
    if (std::fabs(got - want) > 1e-9) {
        std::printf("FAIL %-52s got %.9f, want %.9f\n", why, got, want);
        ++fails;
        return 1;
    }
    return 0;
}

struct Expect {
    int layer, victim, expert, slot;
    double gain;
};

static void check_plan(const std::vector<HotSwap>& got, const std::vector<Expect>& want, const char* why) {
    bool bad = got.size() != want.size();
    for (size_t i = 0; !bad && i < want.size(); ++i) {
        bad = got[i].layer != want[i].layer || got[i].victim != want[i].victim || got[i].expert != want[i].expert ||
              got[i].slot != want[i].slot || std::fabs(got[i].gain - want[i].gain) > 1e-9;
    }
    if (!bad) return;
    std::printf("FAIL %s: got %zu move(s), want %zu\n", why, got.size(), want.size());
    for (const HotSwap& s : got)
        std::printf("     got  layer %d victim %3d expert %3d slot %3d gain %.3f\n", s.layer, s.victim, s.expert,
                    s.slot, s.gain);
    for (const Expect& s : want)
        std::printf("     want layer %d victim %3d expert %3d slot %3d gain %.3f\n", s.layer, s.victim, s.expert,
                    s.slot, s.gain);
    ++fails;
}

int main() {
    // ---- Case A: empty slots are filled best-score first, and a cold cache has only those.
    {
        const std::vector<double> score{5, 0, 3, 0,   2, 9, 0, 0};   // layer 0: e0 5, e2 3 | layer 1: e0 2, e1 9
        const std::vector<int32_t> map{-1, -1, -1, -1,   0, -1, -1, -1};
        check_plan(hot_plan(2, 4, 2, score.data(), map.data(), 0, 0.0),
                   {{1, -1, 1, 1, 9.0}, {0, -1, 0, 0, 5.0}, {0, -1, 2, 1, 3.0}},
                   "A: empty slots filled by score, gain-descending");
    }

    // ---- Case B: the budget keeps the best N and nothing else.
    {
        const std::vector<double> score{5, 0, 3, 0,   2, 9, 0, 0};
        const std::vector<int32_t> map{-1, -1, -1, -1,   0, -1, -1, -1};
        check_plan(hot_plan(2, 4, 2, score.data(), map.data(), 2, 0.0), {{1, -1, 1, 1, 9.0}, {0, -1, 0, 0, 5.0}},
                   "B: budget 2 keeps the two largest gains");
    }

    // ---- Case C: displacement needs a strict, per-pair margin.
    {
        const std::vector<double> score{4, 5};
        const std::vector<int32_t> map{0, -1};
        check_plan(hot_plan(1, 2, 1, score.data(), map.data(), 0, 0.0), {{0, 0, 1, 0, 1.0}},
                   "C1: min_gain 0, a strictly better candidate evicts");
        check_plan(hot_plan(1, 2, 1, score.data(), map.data(), 0, 1.5), {}, "C2: min_gain 1.5, 5 is not > 4 + 1.5");
        check_plan(hot_plan(1, 2, 1, score.data(), map.data(), 0, 0.5), {{0, 0, 1, 0, 1.0}},
                   "C3: min_gain 0.5, 5 > 4.5");
        const std::vector<double> tied{4, 4};
        check_plan(hot_plan(1, 2, 1, tied.data(), map.data(), 0, 0.0), {}, "C4: a tie moves no bytes");
    }

    // ---- Case D: the order is total, so two runs and two machines agree.
    {
        const std::vector<double> score{7, 0,   7, 0};
        const std::vector<int32_t> map{-1, -1,   -1, -1};
        check_plan(hot_plan(2, 2, 1, score.data(), map.data(), 0, 0.0),
                   {{0, -1, 0, 0, 7.0}, {1, -1, 0, 0, 7.0}}, "D1: equal gains break by layer");
        const std::vector<double> s2{4, 4};
        const std::vector<int32_t> m2{-1, -1};
        check_plan(hot_plan(1, 2, 2, s2.data(), m2.data(), 0, 0.0),
                   {{0, -1, 0, 0, 4.0}, {0, -1, 1, 1, 4.0}}, "D2: equal gains in one layer break by expert, then slot");
    }

    // ---- Case E: an expert nothing has routed is never admitted (score 0), so a cold map stays cold.
    {
        const std::vector<double> score{0, 0};
        const std::vector<int32_t> map{-1, -1};
        check_plan(hot_plan(1, 2, 1, score.data(), map.data(), 0, 0.0), {}, "E: zero-score candidates are not admitted");
    }

    // ---- Case F: degenerate shapes return an empty plan instead of reading out of range.
    {
        const std::vector<double> score{1, 2};
        const std::vector<int32_t> map{-1, -1};
        check_plan(hot_plan(1, 2, 0, score.data(), map.data(), 0, 0.0), {}, "F1: no slots");
        check_plan(hot_plan(0, 2, 1, score.data(), map.data(), 0, 0.0), {}, "F2: no layers");
        check_plan(hot_plan(1, 2, 1, nullptr, map.data(), 0, 0.0), {}, "F3: no score");
        const std::vector<int32_t> empty_map{-1, -1};
        check_plan(hot_plan(1, 2, 1, score.data(), empty_map.data(), 0, 0.0), {{0, -1, 1, 0, 2.0}},
                   "F4: an empty map with a positive score still fills");
    }

    // ---- Case G: the gate is positional AND pick-floored.
    {
        struct Due {
            int every;
            long long pos, pos_last, picks, picks_last;
            int min_picks;
            bool want;
        } const due[] = {
            {32, 512, 0, 153600, 0, 8192, true},
            {32, 31, 0, 9300, 0, 8192, false},      // too few positions
            {32, 512, 0, 8000, 0, 8192, false},     // the pick floor
            {32, 64, 32, 19200, 9600, 8192, true},  // the round after the one at 32
            {0, 512, 0, 9000, 0, 8192, false},      // off
            {32, 10, 40, 9000, 0, 8192, false},     // a request rewound the position: never a negative wait
        };
        for (const Due& d : due) {
            const bool got = hot_due(d.every, d.pos, d.pos_last, d.picks, d.picks_last, d.min_picks);
            if (got != d.want) {
                std::printf("FAIL G: hot_due(every %d, pos %lld/%lld, picks %lld/%lld, floor %d) = %d, want %d\n", d.every,
                            d.pos, d.pos_last, d.picks, d.picks_last, d.min_picks, (int) got, (int) d.want);
                ++fails;
            }
        }
    }

    // ---- Case H: decay is a plain multiply, and 1.0 is the no-op.
    {
        std::vector<double> s{8.0, 4.0, 0.0};
        hot_decay(s, 0.7);
        check_double(s[0], 5.6, "H1: 8 x 0.7");
        check_double(s[1], 2.8, "H2: 4 x 0.7");
        check_double(s[2], 0.0, "H3: 0 x 0.7");
        std::vector<double> t{1.5, 2.5};
        hot_decay(t, 1.0);
        check_double(t[0], 1.5, "H4: decay 1.0 leaves the score alone");
    }

    // ---- Case I: the round plans with the scores as they stand and decays AFTERWARDS.
    {
        std::vector<double> score{3.0, 4.6};   // e0 resident (victim), e1 candidate
        const std::vector<int32_t> map{0, -1};
        const std::vector<HotSwap> plan = hot_round(1, 2, 1, score, map.data(), 0, 1.5, 0.7);
        check_plan(plan, {{0, 0, 1, 0, 1.6}}, "I1: 4.6 > 3.0 + 1.5, so the round moves it");
        check_double(score[0], 2.1, "I2: the victim's score is decayed after the decision");
        check_double(score[1], 3.22, "I3: the candidate's score is decayed after the decision");
        std::vector<double> decayed{2.1, 3.22};
        check_plan(hot_plan(1, 2, 1, decayed.data(), map.data(), 0, 1.5), {},
                   "I4: decay-then-plan would have refused the move - the order is not cosmetic");
    }

    // ---- Case J: the budget never hides a strictly better move behind a weaker one from another layer.
    {
        // Two layers, one swap each is admissible; budget 1 must take layer 1's larger gain.
        const std::vector<double> score{6, 5,   9, 1};   // both layers: the candidate is weaker than the victim
        const std::vector<int32_t> map{0, -1,   0, -1};
        check_plan(hot_plan(2, 2, 1, score.data(), map.data(), 1, 0.0), {}, "J1: a weaker candidate never evicts");
        const std::vector<double> score2{1, 5,   1, 9};
        check_plan(hot_plan(2, 2, 1, score2.data(), map.data(), 1, 0.0), {{1, 0, 1, 0, 8.0}},
                   "J2: budget 1 takes the largest gain across layers");
    }

    if (fails == 0) {
        std::printf("ok: hot policy (%d rule groups)\n", 10);
        return 0;
    }
    std::printf("%d FAIL\n", fails);
    return 1;
}
