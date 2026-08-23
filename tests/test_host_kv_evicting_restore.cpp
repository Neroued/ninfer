// Deterministic coverage for the deferred host-KV restore transaction and the
// admission's host-KV handling (src/runtime/engine/host_kv_restore_transaction.h),
// the components the executor's admission runs when the probe deferred a host-KV
// restore. A fake program models the pool-fit math with a page counter, so the
// control flow (which lanes get parked, when the selected resident is destroyed,
// which entry is restored, and whether an impossible restore is detected up front)
// is what is under test - not the real pool arithmetic.
//
// The cases cover the review's required deterministic coverage:
// 1. a selected resident with nonzero reuse, kept when the target never fits
//    (the pre-check detects the impossible restore before any lane is touched);
// 2. a target-switch under LRU pressure (the iteration-3 finding-1 trigger):
//    the transaction restores the exact recorded target, not the better match
//    that appears mid-transaction - proven by a mutation mode that recreates
//    the old find(prompt) reselection;
// 3. the LRU victim is the correct (oldest unprotected) entry, the target
//    survives;
// 4. selected-lane park failure -> discard -> restore;
// 5. post-preflight restore failure -> ResidentLost (replan against the empty
//    lane, and the corrupt entry is dropped - the production postcondition);
// 6. a stale deferred id (the target vanished) -> KeptResident, no destruction;
// 7. max_concurrency == 1 (the entry fits once the sole resident is freed);
// 8. the caller-level contract (admit_host_kv_restore): a KeptResident outcome
//    still honors choice.evict_retained (the HIGH geometry - an impossible host
//    target must not skip the ordinary eviction the resident plan requires),
//    and a stale deferred id goes straight to the ordinary admission;
// 9. the capacity-failure touch (a deferred restore keeps and touches the entry,
//    so it survives LRU pressure while an older untouched entry is evicted) -
//    proven load-bearing by a no-touch mutation mode that flips the LRU victim;
// 10. the production reuse-0 guard (park_selected_lane parks the selected
//     resident only when the plan reuses nothing; a nonzero-reuse plan keeps it).

#include "runtime/engine/host_kv_restore_transaction.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <vector>

namespace {

using ninfer::runtime::HostKvRestoreOutcome;
using ninfer::runtime::evicting_restore_transaction;
using ninfer::runtime::admit_host_kv_restore;

struct FakePrompt {};
struct FakePlan {
    std::uint32_t entitlement = 0;  // the request's growth ceiling (full context + output)
    std::uint32_t reuse = 0;        // how much of the prompt the lane's resident serves
};

// A fake program: the pool-fit math (can_admit_lane / can_restore_lane /
// can_evicting_restore_fit) is the real formula (new_pages <= pool -
// (entitled - old - reclaimable)), driven by a page counter; park/restore move
// entries between the lanes and a small LRU host cache so the target-switch and
// LRU-victim cases are reachable. A lane is EITHER retained (inactive, parkable)
// OR active (live request, not parkable), matching the real semantics where an
// active lane's `retained` flag is cleared.
struct FakeProgram {
    struct Lane {
        bool retained = false;  // holds a sequence, no active request (parkable)
        bool active = false;    // has a live request (not parkable)
        std::uint32_t pages = 0;
        std::uint32_t reuse = 0;
    };
    struct Entry {
        std::uint32_t pages = 0;
        std::uint32_t reuse = 0;
        std::uint64_t id = 0;
        std::uint64_t last_used = 0;
    };
    std::array<Lane, 3> lanes{};
    std::uint32_t pool_pages = 12;
    std::vector<Entry> entries;
    std::uint64_t next_id = 1;
    std::uint64_t clock = 0;
    std::size_t budget_entries = 10;  // the host cache holds this many entries
    // Mutation mode: model the iteration-3 bug where can_restore_lane /
    // restore_lane reselected the current best prompt match instead of the
    // exact recorded entry id.
    bool reselect_best_match = false;
    // Simulate a corrupt entry: the restore fails after the preflight passed.
    bool fail_restore = false;
    // Mutation mode: disable the capacity-deferral touch (the production
    // host_kv_->touch call) to prove the touch is what keeps a deferred entry
    // ahead of older entries under LRU pressure.
    bool touch_on_defer = true;

    // Seed an entry through the id allocator so ids stay unique.
    std::uint64_t seed_entry(std::uint32_t pages, std::uint32_t reuse) {
        const std::uint64_t id = next_id++;
        entries.push_back(Entry{.pages = pages, .reuse = reuse, .id = id, .last_used = ++clock});
        return id;
    }
    // The pool's committed pages: every lane that holds KV (retained or active).
    std::uint32_t entitled() const {
        std::uint32_t t = 0;
        for (const auto& l : lanes) {
            if (l.retained || l.active) { t += l.pages; }
        }
        return t;
    }
    bool has_retained_lane(std::uint32_t l) const { return l < lanes.size() && lanes[l].retained; }
    bool can_admit_lane(std::uint32_t l, const FakePlan& plan) const {
        if (l >= lanes.size()) { return false; }
        const std::uint32_t old = lanes[l].retained ? lanes[l].pages : 0;
        return plan.entitlement <= pool_pages - (entitled() - old);
    }
    bool can_restore_lane(std::uint32_t l, std::uint64_t entry_id) const {
        if (l >= lanes.size()) { return false; }
        const Entry* e = reselect_best_match ? best_match() : find_entry(entry_id);
        if (!e) { return false; }
        const std::uint32_t old = lanes[l].retained ? lanes[l].pages : 0;
        return e->pages <= pool_pages - (entitled() - old);
    }
    // The "can it ever fit" pre-check: both the entry and the plan must fit after
    // reclaiming every retained (parkable) lane except the selected one. Active
    // lanes keep their pages.
    bool can_evicting_restore_fit(std::uint32_t l, std::uint64_t entry_id,
                                  const FakePlan& plan) const {
        if (l >= lanes.size()) { return false; }
        const Entry* e = find_entry(entry_id);
        if (!e) { return false; }
        const std::uint32_t old = lanes[l].retained ? lanes[l].pages : 0;
        std::uint32_t reclaimable = 0;
        for (std::uint32_t other = 0; other < lanes.size(); ++other) {
            if (other == l || !lanes[other].retained) { continue; }
            reclaimable += lanes[other].pages;
        }
        const std::uint32_t available = pool_pages - (entitled() - old - reclaimable);
        return e->pages <= available && plan.entitlement <= available;
    }
    bool host_kv_entry_exists(std::uint64_t entry_id) const { return find_entry(entry_id) != nullptr; }
    const Entry* find_entry(std::uint64_t id) const {
        for (const auto& e : entries) {
            if (e.id == id) { return &e; }
        }
        return nullptr;
    }
    Entry* find_entry_mut(std::uint64_t id) {
        for (auto& e : entries) {
            if (e.id == id) { return &e; }
        }
        return nullptr;
    }
    // The current best prompt match (max reuse) - the operation the iteration-3
    // bug reselected through find(prompt).
    const Entry* best_match() const {
        const Entry* best = nullptr;
        for (const auto& e : entries) {
            if (!best || e.reuse > best->reuse) { best = &e; }
        }
        return best;
    }
    bool park_lane(std::uint32_t l, std::uint64_t protect_id = 0) {
        if (l >= lanes.size() || !lanes[l].retained) { return false; }
        // LRU-evict (skipping the protected entry) until the budget holds the
        // new entry; false when every entry is protected.
        while (entries.size() >= budget_entries) {
            std::size_t victim = 0;
            bool have = false;
            for (std::size_t i = 0; i < entries.size(); ++i) {
                if (entries[i].id == protect_id) { continue; }
                if (!have || entries[i].last_used < entries[victim].last_used) {
                    victim = i;
                    have = true;
                }
            }
            if (!have) { return false; }
            entries.erase(entries.begin() + static_cast<std::ptrdiff_t>(victim));
        }
        entries.push_back(
            Entry{.pages = lanes[l].pages, .reuse = lanes[l].reuse, .id = next_id++, .last_used = ++clock});
        lanes[l] = Lane{};
        return true;
    }
    void evict_retained_lane(std::uint32_t l) {
        if (l < lanes.size()) { lanes[l] = Lane{}; }
    }
    bool restore_lane(std::uint32_t l, std::uint64_t entry_id, const FakePrompt&) {
        if (l >= lanes.size() || lanes[l].retained) { return false; }  // the lane must be empty
        const Entry* e = reselect_best_match ? best_match() : find_entry(entry_id);
        if (!e) { return false; }
        if (fail_restore) {
            // Production restore_lane drops the corrupt entry after reservation
            // fails; model that postcondition (the entry is removed, not kept).
            entries.erase(std::remove_if(entries.begin(), entries.end(),
                                         [id = e->id](const Entry& x) { return x.id == id; }),
                          entries.end());
            return false;
        }
        if (e->pages > pool_pages - entitled()) {
            // Production restore_lane touches the entry on a capacity deferral
            // (it survives, so refresh its recency against LRU eviction).
            if (touch_on_defer) {
                if (Entry* em = find_entry_mut(e->id)) { em->last_used = ++clock; }
            }
            return false;
        }
        lanes[l] = Lane{.retained = true, .pages = e->pages, .reuse = e->reuse};
        entries.erase(std::remove_if(entries.begin(), entries.end(),
                                     [id = e->id](const Entry& x) { return x.id == id; }),
                      entries.end());
        return true;
    }
    FakePlan plan_request_for_lane(std::uint32_t l, const FakePrompt&, const FakePlan& base) {
        // The plan's entitlement is the request's growth ceiling (independent of
        // the resident); the reuse is the resident's prefix (0 when the lane is
        // empty).
        return FakePlan{.entitlement = base.entitlement, .reuse = lanes[l].retained ? lanes[l].reuse : 0};
    }
    bool has_entry(std::uint64_t id) const { return find_entry(id) != nullptr; }
};

int failures = 0;
void check(bool ok, const char* what) {
    std::printf("%-72s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok) { ++failures; }
}

}  // namespace

int main() {
    // T1: non-destructive fallback (the iteration-4 MEDIUM-1 fix). The deferred
    // target (A, reuse 100, 10 pages) has greater reuse than the selected
    // resident (S, reuse 90, 7 pages) but its entitlement still cannot fit even
    // after reclaiming every parkable lane (B parked, C active keeps its pages).
    // The pre-check detects this BEFORE any lane is touched: B is NOT destroyed,
    // the resident is kept, and its (nonzero-reuse) plan is used as-is.
    {
        FakeProgram p;
        p.lanes[0] = FakeProgram::Lane{.retained = true, .pages = 7, .reuse = 90};  // S
        p.lanes[1] = FakeProgram::Lane{.retained = true, .pages = 2, .reuse = 110}; // B
        p.lanes[2] = FakeProgram::Lane{.active = true, .pages = 3, .reuse = 0};    // C (active)
        const std::uint64_t a = p.seed_entry(10, 100);
        FakePrompt prompt;
        FakePlan base{.entitlement = 7, .reuse = 90};
        std::optional<FakePlan> plan = base;
        const HostKvRestoreOutcome outcome = evicting_restore_transaction(
            p, 0, prompt, base, plan, a, 3,
            [](std::uint32_t l) { return l == 2; },  // C is active: never parked
            [](std::uint32_t) {});
        check(outcome == HostKvRestoreOutcome::KeptResident,
              "T1: impossible target detected up front -> KeptResident");
        check(p.lanes[0].retained && p.lanes[0].reuse == 90,
              "T1: selected resident survives (reuse 90)");
        check(p.lanes[1].retained && p.lanes[1].reuse == 110,
              "T1: unrelated retained lane B untouched (not destroyed)");
        check(plan->reuse == 90 && plan->entitlement == 7,
              "T1: resident plan used as-is (reuse 90)");
    }
    // T2: target-switch under LRU pressure (the iteration-3 finding-1 trigger).
    // Parking B inserts a host entry with reuse 110 (better than A's 100). The
    // ID-pinned transaction restores the exact recorded target A; the mutation
    // mode (reselect_best_match) recreates the old find(prompt) reselection and
    // restores the better match B instead - proving the test would have caught
    // the iteration-3 bug.
    {
        for (const bool reselect : {false, true}) {
            FakeProgram p;
            p.lanes[0] = FakeProgram::Lane{};  // S empty: no resident to preserve
            p.lanes[1] = FakeProgram::Lane{.retained = true, .pages = 2, .reuse = 110}; // B
            p.lanes[2] = FakeProgram::Lane{.active = true, .pages = 3, .reuse = 0};    // C (active)
            const std::uint64_t a = p.seed_entry(9, 100);
            p.reselect_best_match = reselect;
            FakePrompt prompt;
            FakePlan base{.entitlement = 7, .reuse = 0};
            std::optional<FakePlan> plan = base;
            const HostKvRestoreOutcome outcome = evicting_restore_transaction(
                p, 0, prompt, base, plan, a, 3,
                [](std::uint32_t l) { return l == 2; },
                [](std::uint32_t) {});
            check(outcome == HostKvRestoreOutcome::Restored,
                  reselect ? "T2(old): a restore still happens (the bug)"
                           : "T2(fix): target restored");
            if (reselect) {
                check(p.lanes[0].reuse == 110,
                      "T2(old): the better match (B, 110) was restored - the target-switch");
            } else {
                check(p.lanes[0].reuse == 100,
                      "T2(fix): the exact target (A, 100) restored, not the better match");
                check(plan->reuse == 100, "T2(fix): plan recomputed against the target (reuse 100)");
            }
            // Entry ids stay unique (A seeded through the allocator, B via park).
            std::vector<std::uint64_t> ids;
            for (const auto& e : p.entries) { ids.push_back(e.id); }
            std::sort(ids.begin(), ids.end());
            check(std::adjacent_find(ids.begin(), ids.end()) == ids.end(),
                  "T2: entry ids are unique");
        }
    }
    // T3: normal successful evicting restore (regression guard for the basic
    // path). Parking one other lane frees enough for the entry; the resident is
    // parked and the entry restored.
    {
        FakeProgram p;
        p.lanes[0] = FakeProgram::Lane{.retained = true, .pages = 7, .reuse = 90};  // S
        p.lanes[1] = FakeProgram::Lane{.retained = true, .pages = 2, .reuse = 50};  // B
        p.lanes[2] = FakeProgram::Lane{.retained = true, .pages = 3, .reuse = 0};   // C
        const std::uint64_t a = p.seed_entry(9, 100);
        FakePrompt prompt;
        FakePlan base{.entitlement = 7, .reuse = 90};
        std::optional<FakePlan> plan = base;
        const HostKvRestoreOutcome outcome = evicting_restore_transaction(
            p, 0, prompt, base, plan, a, 3,
            [](std::uint32_t) { return false; },
            [](std::uint32_t) {});
        check(outcome == HostKvRestoreOutcome::Restored, "T3: normal evicting restore succeeds");
        check(p.lanes[0].retained && p.lanes[0].reuse == 100, "T3: target restored (reuse 100)");
        check(plan->reuse == 100, "T3: plan recomputed against the restored target (reuse 100)");
    }
    // T4: LRU victim correctness. Under budget pressure the selected-lane park
    // must evict the oldest UNPROTECTED entry (B), never the protected target
    // (A). The target survives and the restore succeeds.
    {
        FakeProgram p;
        p.lanes[0] = FakeProgram::Lane{.retained = true, .pages = 7, .reuse = 90};  // S
        p.lanes[1] = FakeProgram::Lane{.retained = true, .pages = 2, .reuse = 50};  // B
        p.lanes[2] = FakeProgram::Lane{.retained = true, .pages = 3, .reuse = 0};   // C
        const std::uint64_t a = p.seed_entry(9, 100);  // A, the oldest entry
        p.budget_entries = 2;  // LRU pressure: the selected-lane park evicts one entry
        FakePrompt prompt;
        FakePlan base{.entitlement = 7, .reuse = 90};
        std::optional<FakePlan> plan = base;
        const HostKvRestoreOutcome outcome = evicting_restore_transaction(
            p, 0, prompt, base, plan, a, 3,
            [](std::uint32_t) { return false; },
            [](std::uint32_t) {});
        check(outcome == HostKvRestoreOutcome::Restored, "T4: restore succeeds under LRU pressure");
        check(p.lanes[0].reuse == 100, "T4: target restored (reuse 100)");
        // The LRU victim was B (the oldest unprotected entry), never the
        // protected target A: after the restore, the only surviving host entry
        // is the selected resident S (A was restored into the lane, B evicted).
        check(p.entries.size() == 1 && p.entries[0].reuse == 90,
              "T4: the LRU victim was B (oldest unprotected), not the target A");
    }
    // T5: selected-lane park failure -> discard -> restore. The host budget is
    // full (only the protected target fits), so parking the selected resident
    // fails and the resident is discarded; the restore still succeeds.
    {
        FakeProgram p;
        p.lanes[0] = FakeProgram::Lane{.retained = true, .pages = 7, .reuse = 90};  // S
        p.lanes[2] = FakeProgram::Lane{.active = true, .pages = 3, .reuse = 0};     // C (active)
        const std::uint64_t a = p.seed_entry(9, 100);
        p.budget_entries = 1;  // only the protected target fits: parking S fails
        FakePrompt prompt;
        FakePlan base{.entitlement = 7, .reuse = 90};
        std::optional<FakePlan> plan = base;
        const HostKvRestoreOutcome outcome = evicting_restore_transaction(
            p, 0, prompt, base, plan, a, 3,
            [](std::uint32_t l) { return l == 2; },
            [](std::uint32_t) {});
        check(outcome == HostKvRestoreOutcome::Restored,
              "T5: park failure -> discard resident -> restore succeeds");
        check(p.lanes[0].reuse == 100, "T5: target restored (reuse 100)");
        check(!p.lanes[0].active, "T5: the discarded resident is not active");
    }
    // T6: post-preflight restore failure -> ResidentLost. The preflight proves
    // the target fits, but the restore fails (a corrupt entry). The lane is
    // empty and the plan is recomputed against it (a full reset, reuse 0).
    {
        FakeProgram p;
        p.lanes[0] = FakeProgram::Lane{.retained = true, .pages = 7, .reuse = 90};  // S
        p.lanes[1] = FakeProgram::Lane{.retained = true, .pages = 2, .reuse = 50};  // B
        const std::uint64_t a = p.seed_entry(9, 100);
        p.fail_restore = true;  // the restore fails after the preflight passed
        FakePrompt prompt;
        FakePlan base{.entitlement = 7, .reuse = 90};
        std::optional<FakePlan> plan = base;
        const HostKvRestoreOutcome outcome = evicting_restore_transaction(
            p, 0, prompt, base, plan, a, 3,
            [](std::uint32_t) { return false; },
            [](std::uint32_t) {});
        check(outcome == HostKvRestoreOutcome::ResidentLost,
              "T6: post-preflight restore failure -> ResidentLost");
        check(!p.lanes[0].retained, "T6: the lane is empty after the failed restore");
        check(plan->reuse == 0, "T6: plan replanned against the empty lane (reuse 0)");
        check(!p.has_entry(a),
              "T6: the corrupt entry was dropped (production postcondition)");
    }
    // T7: stale deferred id (the target vanished before admission). The
    // pre-check finds no such entry and returns KeptResident without touching
    // any lane - the resident survives. (In production the executor revalidates
    // with host_kv_entry_exists before calling the transaction; this is the
    // transaction's own defense-in-depth.)
    {
        FakeProgram p;
        p.lanes[0] = FakeProgram::Lane{.retained = true, .pages = 7, .reuse = 90};  // S
        p.lanes[1] = FakeProgram::Lane{.retained = true, .pages = 2, .reuse = 50};  // B
        const std::uint64_t gone = 999;  // an entry id that was LRU-evicted
        FakePrompt prompt;
        FakePlan base{.entitlement = 7, .reuse = 90};
        std::optional<FakePlan> plan = base;
        const HostKvRestoreOutcome outcome = evicting_restore_transaction(
            p, 0, prompt, base, plan, gone, 3,
            [](std::uint32_t) { return false; },
            [](std::uint32_t) {});
        check(outcome == HostKvRestoreOutcome::KeptResident,
              "T7: stale target id -> KeptResident (no destruction)");
        check(p.lanes[0].retained && p.lanes[0].reuse == 90,
              "T7: selected resident survives (reuse 90)");
        check(p.lanes[1].retained, "T7: unrelated lane B untouched");
    }
    // T8: max_concurrency == 1. There are no other lanes to park; the entry
    // fits once the sole resident is freed, so the restore proceeds directly.
    {
        FakeProgram p;
        p.lanes[0] = FakeProgram::Lane{.retained = true, .pages = 7, .reuse = 90};  // S
        const std::uint64_t a = p.seed_entry(9, 100);
        FakePrompt prompt;
        FakePlan base{.entitlement = 7, .reuse = 90};
        std::optional<FakePlan> plan = base;
        const HostKvRestoreOutcome outcome = evicting_restore_transaction(
            p, 0, prompt, base, plan, a, 1,  // max_concurrency == 1
            [](std::uint32_t) { return false; },
            [](std::uint32_t) {});
        check(outcome == HostKvRestoreOutcome::Restored, "T8: C=1 restore proceeds directly");
        check(p.lanes[0].reuse == 100, "T8: target restored (reuse 100)");
    }
    // T9: the HIGH geometry (the iteration-5 finding). The deferred target (A,
    // 8 pages, reuse 100) can never fit (active C keeps 5 pages), so the
    // transaction returns KeptResident. But the resident-based plan (7 pages)
    // still requires ordinary retained-lane eviction (evict_retained) to be
    // admissible. The caller-level helper must run that eviction (preserving
    // the selected resident S), not skip it - otherwise the infeasible plan is
    // fed to start_prefill_lane and the request fails.
    {
        FakeProgram p;
        p.lanes[0] = FakeProgram::Lane{.retained = true, .pages = 2, .reuse = 90};  // S
        p.lanes[1] = FakeProgram::Lane{.retained = true, .pages = 5, .reuse = 50};  // B
        p.lanes[2] = FakeProgram::Lane{.active = true, .pages = 5, .reuse = 0};     // C (active)
        const std::uint64_t a = p.seed_entry(8, 100);
        FakePrompt prompt;
        FakePlan base{.entitlement = 7, .reuse = 90};
        std::optional<FakePlan> plan = base;
        check(!p.can_admit_lane(0, *plan), "T9: the plan is infeasible before eviction");
        const HostKvRestoreOutcome outcome = admit_host_kv_restore(
            p, 0, prompt, base, plan, a, /*evict_retained=*/true, 3,
            [](std::uint32_t l) { return l == 2; },  // C is active
            [](std::uint32_t) {},
            [](std::uint32_t) {});  // park_selected_lane: no-op (the plan reuses 90)
        check(outcome == HostKvRestoreOutcome::KeptResident,
              "T9: impossible target -> KeptResident (host restore abandoned)");
        check(p.can_admit_lane(0, *plan),
              "T9: ordinary eviction ran, so the plan is now admissible");
        check(p.lanes[0].retained && p.lanes[0].reuse == 90,
              "T9: the selected resident S is preserved (reuse 90)");
        check(!p.lanes[1].retained, "T9: the other retained lane B was parked");
    }
    // T11: the production stale-ID branch. The deferred entry was LRU-evicted
    // between the probe and this admission, so the revalidation (host_kv_entry_
    // exists) fails and the helper skips the transaction, going straight to the
    // ordinary admission. The ordinary path still honors choice.evict_retained,
    // so the plan becomes admissible. (T7 tests the transaction's own
    // defense-in-depth for an absent id; this is the production branch.)
    {
        FakeProgram p;
        p.lanes[0] = FakeProgram::Lane{.retained = true, .pages = 2, .reuse = 90};  // S
        p.lanes[1] = FakeProgram::Lane{.retained = true, .pages = 5, .reuse = 50};  // B
        p.lanes[2] = FakeProgram::Lane{.active = true, .pages = 5, .reuse = 0};     // C (active)
        const std::uint64_t stale = 999;  // an entry id that was LRU-evicted
        FakePrompt prompt;
        FakePlan base{.entitlement = 7, .reuse = 90};
        std::optional<FakePlan> plan = base;
        const HostKvRestoreOutcome outcome = admit_host_kv_restore(
            p, 0, prompt, base, plan, stale, /*evict_retained=*/true, 3,
            [](std::uint32_t l) { return l == 2; },
            [](std::uint32_t) {},
            [](std::uint32_t) {});
        check(outcome == HostKvRestoreOutcome::KeptResident,
              "T11: stale deferred id -> ordinary admission (KeptResident)");
        check(p.can_admit_lane(0, *plan),
              "T11: ordinary eviction ran after the stale-ID fallback");
        check(p.lanes[0].retained && p.lanes[0].reuse == 90,
              "T11: the selected resident S is preserved (reuse 90)");
        check(!p.lanes[1].retained, "T11: the other retained lane B was parked");
    }
    // T12: the capacity-failure touch. A restore that fails because the pool
    // cannot reserve the entry's entitlement keeps the entry and refreshes its
    // recency (touch), so under LRU pressure the touched entry survives while an
    // older untouched entry is evicted. The LRU victim is chosen purely by
    // recency (A is NOT protected), so the touch is what flips the victim from
    // A (oldest, last_used=1) to B_entry (last_used=2): running the same
    // scenario with the touch disabled (the production regression the review
    // worries about) evicts A instead, proving the touch is load-bearing.
    for (const bool touch : {true, false}) {
        FakeProgram p;
        p.lanes[0] = FakeProgram::Lane{};  // empty (the restore target lane)
        p.lanes[1] = FakeProgram::Lane{.retained = true, .pages = 10, .reuse = 50}; // B resident
        const std::uint64_t a = p.seed_entry(9, 100);  // A, last_used=1 (oldest)
        const std::uint64_t b = p.seed_entry(2, 50);   // B_entry, last_used=2
        p.touch_on_defer = touch;
        FakePrompt prompt;
        // Force a capacity failure for A: A (9 pages) > free (12 - 10 = 2).
        check(!p.restore_lane(0, a, prompt),
              touch ? "T12(touch): capacity-deferred restore returns false"
                    : "T12(no-touch): capacity-deferred restore returns false");
        // Force LRU pressure: park a lane with a 2-entry budget, protecting
        // NOTHING (protect_id=0), so the victim is the oldest entry by recency.
        p.budget_entries = 2;
        p.lanes[2] = FakeProgram::Lane{.retained = true, .pages = 1, .reuse = 10};
        (void)p.park_lane(2, 0);
        if (touch) {
            // A was touched (last_used=3, newer than B_entry's 2), so B_entry is
            // the LRU victim and A survives.
            check(p.has_entry(a), "T12(touch): the touched target survived the LRU pressure");
            check(!p.has_entry(b), "T12(touch): the older untouched entry was LRU-evicted");
        } else {
            // Without the touch A is still the oldest (last_used=1), so A is the
            // LRU victim and B_entry survives - the touch is what saved A.
            check(!p.has_entry(a),
                  "T12(no-touch): without the touch the target is the LRU victim");
            check(p.has_entry(b),
                  "T12(no-touch): the newer entry survives (the touch is load-bearing)");
        }
    }
    // T13: the production reuse-0 guard (the park_selected_lane callback). The
    // helper's park_selected_lane models the executor's guard: park the selected
    // resident only when the plan reuses nothing (a full reset overwrites it); a
    // plan that reuses the resident's own prefix must not park it out from under
    // itself. T9/T11 pass a no-op callback, so this is the case that exercises
    // the guard's reuse==0 condition in both directions.
    for (const bool reuse0 : {true, false}) {
        FakeProgram p;
        const std::uint32_t reuse = reuse0 ? 0 : 90;
        p.lanes[0] = FakeProgram::Lane{.retained = true, .pages = 2, .reuse = reuse};  // S
        p.lanes[1] = FakeProgram::Lane{.retained = true, .pages = 5, .reuse = 50};  // B
        p.lanes[2] = FakeProgram::Lane{.active = true, .pages = 5, .reuse = 0};     // C (active)
        const std::uint64_t stale = 999;  // stale id -> the ordinary admission
        FakePrompt prompt;
        FakePlan base{.entitlement = 7, .reuse = reuse};
        std::optional<FakePlan> plan = base;
        // Model the production guard: park the resident only when the plan reuses
        // nothing (reusable_prompt_tokens == 0).
        auto park_selected = [&p, &plan](std::uint32_t l) {
            if (p.lanes[l].retained && plan->reuse == 0) {
                (void)p.park_lane(l);
            }
        };
        const HostKvRestoreOutcome outcome = admit_host_kv_restore(
            p, 0, prompt, base, plan, stale, /*evict_retained=*/true, 3,
            [](std::uint32_t l) { return l == 2; },
            [](std::uint32_t) {},
            park_selected);
        check(outcome == HostKvRestoreOutcome::KeptResident,
              reuse0 ? "T13(reuse0): stale id -> ordinary admission"
                     : "T13(reuse>0): stale id -> ordinary admission");
        if (reuse0) {
            check(!p.lanes[0].retained,
                  "T13(reuse0): the plan reuses nothing, so the resident was parked");
        } else {
            check(p.lanes[0].retained && p.lanes[0].reuse == 90,
                  "T13(reuse>0): the plan reuses the prefix, so the resident was kept");
        }
    }
    std::printf("\n%s\n", failures == 0 ? "all ok" : "FAILURES");
    return failures == 0 ? 0 : 1;
}