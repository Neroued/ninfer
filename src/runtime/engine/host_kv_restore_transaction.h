#pragma once

// The deferred host-KV restore transaction, extracted from
// ConcurrentExecutor::admit_planned_request so the decision is testable with a
// fake program (tests/test_host_kv_evicting_restore.cpp).
//
// The probe defers a host-KV restore when the matched entry's stored
// entitlement does not fit the pool: it records the entry's stable id and keeps
// the pool saturated. The admission then runs this transaction: park the OTHER
// retained lanes (protecting the target entry) until BOTH the target entry and
// the plan fit, WITHOUT destroying the selected resident yet; only then empty
// the resident and restore the target. If the target never fits, keep the
// resident and its (possibly nonzero-reuse) plan - no cold prefill.
//
// Every check and the restore address the EXACT deferred entry id, not the
// current best match: parking another lane inserts a host entry that can become
// the better match mid-transaction, and re-selecting through find() is the
// prompt-relative path that destroyed a valid resident and cold-prefilled.

#include <cstdint>
#include <optional>
#include <stdexcept>

namespace ninfer::runtime {

enum class HostKvRestoreOutcome {
    Restored,      // the target entry was restored; the plan was recomputed (an append)
    KeptResident,  // the host restore was not performed (the target never fit, or the
                   // deferred entry was stale and the ordinary admission ran). The
                   // resident plan is used as-is (the resident may have been parked by
                   // the ordinary admission when the plan reuses nothing, and other
                   // retained lanes may have been evicted), so the plan was NOT
                   // recomputed and the caller does NOT resync the plan version
    ResidentLost,  // the target fit but the restore failed (a corrupt entry); the
                   // resident was destroyed and the plan was recomputed (a full reset)
};

// `lane_plan` is the request's current plan for the selected lane (a
// std::optional<Plan>): it is read for can_admit_lane and replaced (an append
// on success, a full reset on the invariant-violating failure) when the lane
// state changes. `lane_active(l)` reports whether lane l holds a live request
// (its slot is occupied); active lanes are never parked. `invalidate_lane_plans`
// is called for every lane whose plan is invalidated by the parking.
template <class Program, class PlanOpt, class Prompt, class BasePlan>
HostKvRestoreOutcome evicting_restore_transaction(
    Program& program,
    std::uint32_t lane,
    const Prompt& prompt,
    const BasePlan& base_plan,
    PlanOpt& lane_plan,
    std::uint64_t target_entry,
    std::uint32_t max_concurrency,
    auto lane_active,
    auto invalidate_lane_plans) {
    // "Can it ever fit" pre-check: if the target entry and the plan cannot both
    // fit even after reclaiming every parkable retained lane, parking (and
    // possibly evicting) those lanes would only destroy unrelated retained
    // state for a restore that can never succeed. Return KeptResident without
    // touching any lane.
    if (!program.can_evicting_restore_fit(lane, target_entry, *lane_plan)) {
        return HostKvRestoreOutcome::KeptResident;
    }
    for (std::uint32_t retained_lane = 0;
         retained_lane < max_concurrency &&
         (!program.can_restore_lane(lane, target_entry) ||
          !program.can_admit_lane(lane, *lane_plan));
         ++retained_lane) {
        if (retained_lane != lane && !lane_active(retained_lane) &&
            program.has_retained_lane(retained_lane)) {
            // Park to host RAM instead of discarding: 0.3s round-trip against a
            // 95s re-prefill at 231k. park_lane() clears the lane itself on
            // success, so only fall through to the discard when no slab was
            // available.
            if (!program.park_lane(retained_lane, target_entry)) {
                program.evict_retained_lane(retained_lane);
            }
            invalidate_lane_plans(retained_lane);
        }
    }
    if (program.can_restore_lane(lane, target_entry) &&
        program.can_admit_lane(lane, *lane_plan)) {
        // The entry and the plan both fit once the selected resident is freed:
        // empty the lane (park or discard the resident, protecting the target)
        // and restore. If no slab is available even with the target protected,
        // discard the resident so the lane is empty and the restore can
        // proceed (mirrors the probe's park-failure branch).
        if (program.has_retained_lane(lane)) {
            if (!program.park_lane(lane, target_entry)) {
                program.evict_retained_lane(lane);
                invalidate_lane_plans(lane);
            }
        }
        if (program.restore_lane(lane, target_entry, prompt)) {
            lane_plan.reset();
            lane_plan.emplace(program.plan_request_for_lane(lane, prompt, base_plan));
            invalidate_lane_plans(lane);
            return HostKvRestoreOutcome::Restored;
        }
        // Post-preflight failure: the exact-ID preflight proved the target
        // fits, so this is a corrupt entry (restore_lane dropped it) or a
        // deeper invariant violation. The lane is empty; the re-plan is the
        // explicit fallback (a cold prefill; the resident, if parked, survives
        // in the host cache).
        lane_plan.reset();
        lane_plan.emplace(program.plan_request_for_lane(lane, prompt, base_plan));
        invalidate_lane_plans(lane);
        return HostKvRestoreOutcome::ResidentLost;
    }
    // The target never fit: keep the resident and its plan.
    return HostKvRestoreOutcome::KeptResident;
}

// The admission's host-KV handling, extracted from
// ConcurrentExecutor::admit_planned_request so the caller-level contract is
// testable with a fake program. It revalidates the deferred entry by id, runs
// the evicting-restore transaction if the entry still exists, and - when the
// restore is abandoned (KeptResident) or the entry is stale - falls back to the
// ordinary admission of the resident plan: park the selected resident only when
// the plan reuses nothing, and run the retained-lane eviction loop if the lane
// choice requires it.
//
// The KeptResident fallback is the critical contract: a KeptResident outcome
// means "do not pursue the impossible host restore", NOT "ignore the lane
// choice's ordinary eviction requirement." If the resident plan still requires
// ordinary retained-lane eviction (evict_retained) to be admissible, that
// eviction must still run (preserving the selected resident), or an infeasible
// plan is fed to start_prefill_lane and the request fails.
//
// `park_selected_lane(l)` is the reuse-0 park: park the selected resident (to
// host RAM) only when the plan reuses nothing (a full reset overwrites it); a
// plan that reuses the resident's own prefix must not park it out from under
// itself. The caller supplies the condition (it reads the plan's reuse).
template <class Program, class PlanOpt, class Prompt, class BasePlan>
HostKvRestoreOutcome admit_host_kv_restore(
    Program& program,
    std::uint32_t lane,
    const Prompt& prompt,
    const BasePlan& base_plan,
    PlanOpt& lane_plan,
    std::uint64_t deferred_entry,
    bool evict_retained,
    std::uint32_t max_concurrency,
    auto lane_active,
    auto invalidate_lane_plans,
    auto park_selected_lane) {
    // Revalidate the deferred entry by id (not best match): a concurrent
    // admission may have LRU-evicted it between the probe and this admission.
    // A stale entry falls through to the ordinary admission below.
    const std::uint64_t target =
        deferred_entry != 0 && program.host_kv_entry_exists(deferred_entry)
            ? deferred_entry
            : 0;
    if (target != 0) {
        const HostKvRestoreOutcome outcome = evicting_restore_transaction(
            program, lane, prompt, base_plan, lane_plan, target, max_concurrency,
            lane_active, invalidate_lane_plans);
        if (outcome != HostKvRestoreOutcome::KeptResident) {
            // Restored or ResidentLost: the transaction handled capacity and
            // recomputed the plan; the caller resyncs the plan version.
            return outcome;
        }
        // KeptResident: the host restore was abandoned (the target can never
        // fit). Fall through to the ordinary admission of the resident plan.
    }
    // Ordinary admission (the stale-entry and KeptResident fallbacks both land
    // here): park the selected resident only when the plan reuses nothing, and
    // run the retained-lane eviction loop if the lane choice requires it.
    park_selected_lane(lane);
    if (evict_retained) {
        for (std::uint32_t retained_lane = 0;
             retained_lane < max_concurrency &&
             !program.can_admit_lane(lane, *lane_plan);
             ++retained_lane) {
            if (retained_lane != lane && !lane_active(retained_lane) &&
                program.has_retained_lane(retained_lane)) {
                // Park to host RAM instead of discarding: 0.3s round-trip
                // against a 95s re-prefill at 231k. park_lane() clears the
                // lane itself on success, so only fall through to the discard
                // when no slab was available.
                if (!program.park_lane(retained_lane)) {
                    program.evict_retained_lane(retained_lane);
                }
                invalidate_lane_plans(retained_lane);
            }
        }
        if (!program.can_admit_lane(lane, *lane_plan)) {
            throw std::logic_error("retained eviction did not make admission feasible");
        }
    }
    // The resident plan is used as-is (a valid, possibly nonzero-reuse append).
    return HostKvRestoreOutcome::KeptResident;
}

}  // namespace ninfer::runtime