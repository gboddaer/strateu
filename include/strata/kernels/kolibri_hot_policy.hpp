// include/strata/kernels/kolibri_hot_policy.hpp - the expert-cache residency policy, as arithmetic (no CUDA).
//
// **WHY THIS FILE EXISTS.**  Which expert sits in which VRAM slot, and when that changes, is the one part of the
// Kolibri engine's expert cache that can be wrong in a way no end-to-end gate sees: a wrong budget, an off-by-one in
// the decay, or a tie broken by pointer order all produce a plausible token and a hit rate that looks fine on one
// prompt.  The engine (src/kolibri/kolibri_engine.cu, slots_reselect) keeps only the execution: it calls
// hot_due()/hot_plan()/hot_decay() here, installs the answer and copies the blobs.  tests/kolibri/hot_policy_test.cpp
// pins every rule below without a GPU or a model.
//
// **THE SHAPE IS THE QWEN3.8-FLASH-NEXT ENGINE'S ADAPTIVE TIER** (src/program/generate.cpp:8842-8890 there):
//   * a round every N POSITIONS, not every N seconds, so two runs of one prompt see the same rounds;
//   * at most `budget` moves per round, gain-ranked across all layers (generate.cpp:629 `--adapt-swaps`);
//   * a candidate must beat the resident it evicts by `min_gain` (generate.cpp:8865 `--adapt-min-gain 1.5`);
//   * the score it sorts on is decayed after every round (generate.cpp:601 `--adapt-decay 0.7`), so the map follows
//     the conversation instead of freezing on the prompt's own traffic.
//
// Measured on this box before this change (T4, 512-token prompt, two runs): the unbudgeted round moved 2,699 experts,
// 6,174 MiB, 629 ms inside one token boundary; the budgeted round is at most 96 moves = 212 MB = ~21 ms at the
// 10.3 GB/s the same copies measured.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace strata::kolibri {

/// One move: `expert` takes `slot`, whose previous occupant was `victim` (-1 when the slot was empty).
struct HotSwap {
    int layer = 0;
    int victim = -1;
    int expert = 0;
    int slot = 0;
    double gain = 0.0;
};

/// **THE ROUND'S GATE.**  A round runs when `every_positions` positions have gone by since the last one AND the
/// routed-pick floor has been met.  Both conditions are needed: the position count is what makes two runs comparable,
/// the pick floor is what stops a round on a request that has routed nothing yet (a 1-token request, or a reuse hit
/// that read almost no routing).  `every_positions <= 0` means the mid-request tier is off (rounds only after a
/// request, where the caller calls slots_reselect directly).
inline bool hot_due(int every_positions, long long pos, long long pos_at_last, long long picks,
                    long long picks_at_last, int min_picks) {
    if (every_positions <= 0) return false;
    if (pos - pos_at_last < (long long) every_positions) return false;
    return picks - picks_at_last >= (long long) (min_picks > 0 ? min_picks : 0);
}

/// **THE RECENCY WINDOW.**  Every score is multiplied by `factor` once per round, so an expert picked during the
/// prompt loses its place to one the conversation is routing now: with 0.7 a count has a half-life of ~1.9 rounds
/// (61 positions at the default cadence).  1.0 is a no-op and is what "no decay" means; the caller validates the range.
inline void hot_decay(std::vector<double>& score, double factor) {
    if (factor == 1.0) return;
    for (double& v : score) v *= factor;
}

/// **THE DECISION.**  `score` and `map` are `n_layers * n_expert` (row-major, layer by layer); `map[i]` is the slot
/// expert `i` of its layer occupies, or -1.  Returns the moves to make, best gain first.
///
/// Rules, in the order they are applied, each one a test case:
///   1. an EMPTY slot is filled by the best candidate and costs nothing (gain = its score) - a cold cache has only
///      these, and a `min_gain` that applied to them would leave the cache empty forever;
///   2. an OCCUPIED slot is taken only when `score(candidate) > score(occupant) + min_gain` (strict: a tie is not a
///      reason to move bytes, and that is what stops the boundary churn the unbudgeted round showed as repeated
///      200-300 move rounds);
///   3. a candidate with score <= 0 is never admitted (nothing has been routed, so nothing is known about it);
///   4. within a layer the i-th best candidate is paired with the i-th weakest occupant, so a strong candidate is not
///      wasted on a slot that a weak one could take;
///   5. across layers the moves are sorted by gain descending, then (layer, expert, slot) ascending - a total order,
///      so the same scores give the same plan on every run and on every machine;
///   6. `budget > 0` keeps the best `budget` moves; `budget <= 0` is NO CAP (the function always answers "what would you
///      move"). "Off" is the caller's decision: slots_reselect returns before calling this when its budget is 0.
inline std::vector<HotSwap> hot_plan(int n_layers, int n_expert, int slots_per_layer, const double* score,
                                     const int32_t* map, int budget, double min_gain) {
    std::vector<HotSwap> out;
    if (n_layers <= 0 || n_expert <= 0 || slots_per_layer <= 0 || score == nullptr) return out;

    for (int l = 0; l < n_layers; ++l) {
        const size_t row = (size_t) l * (size_t) n_expert;
        std::vector<int> occ((size_t) slots_per_layer, -1);
        for (int e = 0; e < n_expert; ++e) {
            const int s = (map != nullptr) ? map[row + (size_t) e] : -1;
            if (s >= 0 && s < slots_per_layer && occ[(size_t) s] < 0) occ[(size_t) s] = e;
        }
        struct Item {
            double score;
            int idx;   // expert for a candidate, slot for a victim
        };
        std::vector<Item> cand;
        for (int e = 0; e < n_expert; ++e) {
            if (map != nullptr && map[row + (size_t) e] >= 0) continue;
            const double s = score[row + (size_t) e];
            if (s > 0.0) cand.push_back({s, e});
        }
        std::vector<Item> vict;
        for (int s = 0; s < slots_per_layer; ++s)
            if (occ[(size_t) s] >= 0) vict.push_back({score[row + (size_t) occ[(size_t) s]], s});
        std::sort(cand.begin(), cand.end(), [](const Item& a, const Item& b) {
            if (a.score != b.score) return a.score > b.score;
            return a.idx < b.idx;
        });
        std::sort(vict.begin(), vict.end(), [](const Item& a, const Item& b) {
            if (a.score != b.score) return a.score < b.score;
            return a.idx < b.idx;
        });
        std::vector<char> used(cand.size(), 0);
        size_t next = 0;
        for (int s = 0; s < slots_per_layer; ++s) {   // rule 1: empty slots first, in slot order
            if (occ[(size_t) s] >= 0) continue;
            while (next < cand.size() && used[next]) ++next;
            if (next >= cand.size()) break;
            used[next] = 1;
            out.push_back({l, -1, cand[next].idx, s, cand[next].score});
            ++next;
        }
        for (const Item& v : vict) {   // rules 2-4: pay only for a strictly better occupant, best first
            while (next < cand.size() && used[next]) ++next;
            if (next >= cand.size()) break;
            if (!(cand[next].score > v.score + min_gain)) break;
            used[next] = 1;
            out.push_back({l, occ[(size_t) v.idx], cand[next].idx, v.idx, cand[next].score - v.score});
            ++next;
        }
    }
    std::sort(out.begin(), out.end(), [](const HotSwap& a, const HotSwap& b) {   // rule 5
        if (a.gain != b.gain) return a.gain > b.gain;
        if (a.layer != b.layer) return a.layer < b.layer;
        if (a.expert != b.expert) return a.expert < b.expert;
        return a.slot < b.slot;
    });
    if (budget > 0 && (int) out.size() > budget) out.resize((size_t) budget);   // rule 6
    return out;
}

/// **ONE ROUND, IN THE ORDER THAT MATTERS.**  Plan with the scores as they stand, then decay them.  The order is not
/// cosmetic: an expert that is barely ahead of the one it evicts stops being ahead once both are decayed, so a
/// decay-then-plan round would move it and a plan-then-decay round would not.  Composing both here is what lets the test
/// pin the order (the engine calls this, never the two functions separately).
inline std::vector<HotSwap> hot_round(int n_layers, int n_expert, int slots_per_layer, std::vector<double>& score,
                                      const int32_t* map, int budget, double min_gain, double decay) {
    std::vector<HotSwap> plan = hot_plan(n_layers, n_expert, slots_per_layer, score.data(), map, budget, min_gain);
    hot_decay(score, decay);
    return plan;
}

}  // namespace strata::kolibri
