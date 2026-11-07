#include "runtime/engine/scheduler.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

struct Request {
    using SequenceHandle = std::uint64_t;
    enum class Phase { Waiting, Materializing, Prefill, Replay, Decode, Control, Finished };

    struct Budget {
        std::uint32_t tokens = 8;

        [[nodiscard]] std::uint32_t remaining() const noexcept { return tokens; }
    };

    struct Output {
        std::uint32_t model_tokens = 8;
        std::vector<ninfer::TokenId> control;

        [[nodiscard]] std::uint32_t model_token_budget_remaining(std::uint32_t total) const {
            return std::min(total, model_tokens);
        }

        [[nodiscard]] std::span<const ninfer::TokenId> pending_control_tokens() const {
            return control;
        }
    };

    [[nodiscard]] bool is_decode_ready() const { return phase == Phase::Decode; }

    [[nodiscard]] bool is_control_ready() const { return phase == Phase::Control; }

    [[nodiscard]] bool is_prefilling() const { return phase == Phase::Prefill; }

    [[nodiscard]] bool is_replaying() const { return phase == Phase::Replay; }

    [[nodiscard]] bool is_materializing() const { return phase == Phase::Materializing; }

    [[nodiscard]] bool is_model_finished() const { return phase == Phase::Finished; }

    std::uint64_t id                 = 0;
    std::uint32_t admission_bypasses = 0;
    Phase phase                      = Phase::Waiting;
    bool capture_pending             = false;
    bool recovery_pending            = false;
    std::optional<Budget> budget     = Budget{};
    std::optional<SequenceHandle> sequence;
    Output output;
};

using Scheduler  = ninfer::runtime::Scheduler<Request>;
using RequestPtr = std::shared_ptr<Request>;
using Slots      = std::array<RequestPtr, ninfer::kMaximumConcurrency>;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

RequestPtr request(std::uint64_t id, Request::Phase phase = Request::Phase::Waiting) {
    auto value      = std::make_shared<Request>();
    value->id       = id;
    value->phase    = phase;
    value->sequence = id;
    return value;
}

void test_fifo_bypass_survives_pressure() {
    Scheduler scheduler;
    const auto head = request(10);
    std::deque<RequestPtr> fresh{head, request(11), request(12), request(13)};

    // Merely inspecting requests that cannot obtain resources never spends bypass credit.
    for (int attempt = 0; attempt < 6; ++attempt) {
        scheduler.begin_admission_scan();
        const auto candidates = scheduler.fresh_candidates(fresh, 2);
        for (const auto& candidate : candidates) {
            scheduler.admission_candidate_checked(candidate->id);
        }
        require(!candidates.empty() && candidates.front() == head && candidates.size() <= 3,
                "fresh scan did not preserve the head and bounded work");
    }
    require(head->admission_bypasses == 0 && fresh[1]->admission_bypasses == 0,
            "failed admission attempts consumed FIFO bypass credit");

    scheduler.admitted(fresh, *fresh[1], 2);
    fresh.erase(fresh.begin() + 1);
    require(head->admission_bypasses == 1, "successful younger admission was not charged");

    scheduler.require_progress(11);
    scheduler.served(11, Scheduler::service_quantum_ns);
    require(scheduler.advance_service(false, false) && !scheduler.progress_owner(),
            "resolved pressure retained the service owner");
    require(!scheduler.fresh_candidates(fresh, 2).empty(),
            "resolved pressure did not reopen fresh candidate scanning");
    require(head->admission_bypasses == 1, "pressure recovery reset a waiting request's credit");

    scheduler.begin_admission_scan();
    const auto candidates = scheduler.fresh_candidates(fresh, 2);
    const auto second     = std::find(candidates.begin(), candidates.end(), fresh[1]);
    require(second != candidates.end(), "second permitted younger request was not considered");
    scheduler.admitted(fresh, **second, 2);
    fresh.erase(fresh.begin() + 1);
    for (std::uint64_t id = 14; id < 20; ++id) {
        fresh.push_back(request(id));
        scheduler.begin_admission_scan();
        const auto blocked = scheduler.fresh_candidates(fresh, 2);
        require(blocked.size() == 1 && blocked.front() == head,
                "new arrivals or completion events bypassed an exhausted FIFO head");
    }

    // Cancelling the blocked head exposes the next request with its own existing credit.
    fresh.pop_front();
    scheduler.begin_admission_scan();
    const auto after_cancel = scheduler.fresh_candidates(fresh, 2);
    require(after_cancel.size() > 1 && after_cancel.size() <= 3 && after_cancel.front()->id == 13 &&
                after_cancel.front()->admission_bypasses == 0,
            "head removal transferred its exhausted credit to the next request");
}

void test_scan_progress_and_waiter_barrier() {
    Scheduler scheduler;
    std::deque<RequestPtr> fresh{request(1), request(2), request(3),
                                 request(4), request(5), request(6)};
    std::vector<std::uint64_t> seen;
    const auto observe = [&](const std::vector<RequestPtr>& candidates) {
        require(!candidates.empty() && candidates.front() == fresh.front() &&
                    candidates.size() <= 3,
                "candidate scan exceeded the head plus bounded younger requests");
        for (const auto& candidate : candidates) {
            require(std::find(fresh.begin(), fresh.end(), candidate) != fresh.end(),
                    "removed request remained a candidate");
            scheduler.admission_candidate_checked(candidate->id);
            if (std::find(seen.begin(), seen.end(), candidate->id) == seen.end()) {
                seen.push_back(candidate->id);
            }
        }
    };
    scheduler.begin_admission_scan();
    const auto first = scheduler.fresh_candidates(fresh, 2);
    observe(first);
    require(scheduler.admission_scan_pending(),
            "failed bounded batch abandoned later requests that could fit");
    const auto next = scheduler.fresh_candidates(fresh, 2);
    require(
        next.size() == 3 && next[1]->id == 4 && next[2]->id == 5,
        "a later fitting request required another arrival or resident completion to be checked");
    observe(next);

    // Remove the last examined ticket and add an arrival. The next event must preserve
    // cursor progress rather than restart indefinitely at the first failed requests.
    std::erase(fresh, next.back());
    fresh.push_back(request(7));
    scheduler.begin_admission_scan();
    const auto after_change = scheduler.fresh_candidates(fresh, 2);
    require(after_change.size() == 3 && after_change[1]->id == 6 && after_change[2]->id == 7,
            "queue changes reset the stable-ticket admission cursor");
    observe(after_change);
    for (std::size_t attempt = 0; scheduler.admission_scan_pending() && attempt < fresh.size();
         ++attempt) {
        observe(scheduler.fresh_candidates(fresh, 2));
    }
    require(!scheduler.admission_scan_pending(), "a complete admission pass kept retrying");
    require(scheduler.fresh_candidates(fresh, 2).empty(),
            "ordinary decode restarted a failed admission pass");
    for (const auto& pending : fresh) {
        require(std::find(seen.begin(), seen.end(), pending->id) != seen.end(),
                "failed leading candidates starved a later admission attempt");
        require(pending->admission_bypasses == 0, "candidate scanning spent admission credit");
    }

    // A skipped waiter retains its own limit even while an older request is still ahead of it.
    fresh[1]->admission_bypasses = 2;
    for (int attempt = 0; attempt < 3; ++attempt) {
        scheduler.begin_admission_scan();
        const auto candidates = scheduler.fresh_candidates(fresh, 2);
        require(candidates.size() == 2 && candidates[1] == fresh[1],
                "rotating scan crossed an exhausted non-head waiter");
        observe(candidates);
        require(!scheduler.admission_scan_pending(),
                "blocked younger requests kept the scan awake behind a FIFO barrier");
    }
}

void test_scan_keeps_unexamined_candidates() {
    Scheduler scheduler;
    std::deque<RequestPtr> fresh{request(1), request(2), request(3), request(4)};
    scheduler.begin_admission_scan();
    const auto first = scheduler.fresh_candidates(fresh, 2);
    require(first.size() == 3, "admission did not expose a bounded candidate batch");
    scheduler.admission_candidate_checked(first[0]->id);
    scheduler.admission_candidate_checked(first[1]->id);
    scheduler.admitted(fresh, *first[1], 2);
    std::erase(fresh, first[1]);
    // Admission returns early, so the last returned candidate has not actually been inspected.
    const auto continued = scheduler.fresh_candidates(fresh, 2);
    require(continued.size() == 3 && continued[1]->id == 3 && continued[2]->id == 4,
            "successful admission discarded candidates that had never been checked");
    for (const auto& candidate : continued) {
        scheduler.admission_candidate_checked(candidate->id);
    }
    require(!scheduler.admission_scan_pending(),
            "completed candidate checks did not retire the pass");
    scheduler.begin_admission_scan();
    require(!scheduler.fresh_candidates(fresh, 2).empty(),
            "a later capacity event did not retry a previously failed request");
}

void admit(Scheduler& scheduler, const RequestPtr& value) {
    scheduler.admitted(std::deque<RequestPtr>{value}, *value, 4);
}

void test_pressure_handoff_and_fair_service() {
    Scheduler scheduler;
    Slots slots{};
    auto a = slots[0] = request(10, Request::Phase::Decode);
    auto b = slots[1] = request(20, Request::Phase::Decode);
    auto c = slots[2] = request(30, Request::Phase::Decode);
    for (const auto& value : {a, b, c}) admit(scheduler, value);

    scheduler.require_progress(a->id);
    require(scheduler.preemption_victim(slots, 4) == 2,
            "capacity handoff did not exclude its beneficiary");
    const std::array excluded{2U};
    require(scheduler.preemption_victim(slots, 4, excluded) == 1,
            "noncontributing victim prevented another release candidate");
    slots[2].reset(); // c paused; the same capacity is still owed to a.
    scheduler.require_progress(c->id);
    require(scheduler.progress_owner() == a->id && !scheduler.advance_service(true, false),
            "a paused victim stole an unfinished capacity handoff");
    scheduler.served(b->id, Scheduler::service_quantum_ns);
    require(!scheduler.advance_service(true, false), "another row's work expired the owner");
    scheduler.served(a->id, Scheduler::service_quantum_ns - 1);
    require(!scheduler.advance_service(true, false), "an incomplete service slice expired");
    scheduler.served(a->id, 1);
    require(scheduler.advance_service(true, false) && scheduler.progress_owner() == b->id,
            "waiting request did not receive the next service turn");
    scheduler.served(b->id, Scheduler::service_quantum_ns);
    require(scheduler.advance_service(true, false) && scheduler.progress_owner() == c->id,
            "suspended request was skipped by service rotation");
    require(scheduler.preemption_victim(slots, 4) == 1,
            "a waiting request could not preempt an older served resident");
    b->recovery_pending = true;
    require(scheduler.preemption_victim(slots, 4) == 0,
            "protected Native recovery was selected as a victim");
    b->recovery_pending = false;
    // Arbitrarily many resource polls/replay boundaries cannot consume c's service turn.
    for (int i = 0; i < 100; ++i)
        require(!scheduler.advance_service(true, true) && scheduler.progress_owner() == c->id,
                "replay exhausted a turn without new progress");
    scheduler.served(c->id, Scheduler::service_quantum_ns);
    require(!scheduler.advance_service(true, true), "Native recovery protection was bypassed");
    require(scheduler.advance_service(true, false) && scheduler.progress_owner() == a->id,
            "fair rotation failed to return to the earlier request");
    scheduler.finished(a->id);
    require(scheduler.progress_owner() == b->id, "cancelled owner retained the handoff");
    scheduler.finished(b->id);
    require(scheduler.progress_owner() == c->id, "terminal owner blocked remaining work");
    scheduler.served(c->id, 1);
    require(scheduler.advance_service(false, false) && !scheduler.progress_owner(),
            "resolved pressure did not reopen fresh admission");
    scheduler.finished(c->id);
}

void test_recovery_cost_has_a_bounded_service_allowance() {
    Scheduler scheduler;
    auto a = request(1), b = request(2);
    admit(scheduler, a);
    admit(scheduler, b);
    scheduler.require_progress(a->id);
    scheduler.recovery_work(a->id, 10 * Scheduler::maximum_service_ns);
    require(!scheduler.advance_service(true, false), "replay was counted as new progress");
    scheduler.served(a->id, Scheduler::service_quantum_ns);
    require(!scheduler.advance_service(true, false), "expensive replay earned no useful service");
    scheduler.served(a->id, Scheduler::maximum_service_ns - Scheduler::service_quantum_ns);
    require(scheduler.advance_service(true, false) && scheduler.progress_owner() == b->id,
            "recovery cost allowed unbounded service");
    scheduler.served(b->id, Scheduler::service_quantum_ns);
    require(scheduler.advance_service(true, false), "recovery cost leaked into another turn");
}

void test_bounded_service_under_continuous_contention() {
    Scheduler scheduler;
    std::array<RequestPtr, 4> requests{request(1), request(2), request(3), request(4)};
    for (const auto& value : requests) admit(scheduler, value);
    scheduler.require_progress(1);
    std::array<unsigned, 4> grants{};
    for (unsigned turn = 0; turn < 40; ++turn) {
        const auto owner = scheduler.progress_owner();
        require(owner && *owner == turn % 4 + 1, "an admitted waiter was bypassed");
        ++grants[*owner - 1];
        for (const auto& value : requests) scheduler.require_progress(value->id);
        scheduler.served(*owner, Scheduler::service_quantum_ns);
        require(scheduler.advance_service(true, false), "contended service failed to rotate");
    }
    require(std::all_of(grants.begin(), grants.end(), [](auto n) { return n == 10; }),
            "continuous contention starved a request");
}

ninfer::runtime::PrefillUnitTokens units_of(std::uint32_t tokens) {
    ninfer::runtime::PrefillUnitTokens units{};
    units.fill(tokens);
    return units;
}

using ninfer::runtime::PrefillPacing;
using ninfer::runtime::PrefillRegime;

// A Throughput pacing with the given token and unit allowance.
PrefillPacing throughput(std::uint32_t tokens, std::uint32_t units) {
    return {.throughput_tokens_per_cycle = tokens, .throughput_units_per_cycle = units};
}

void test_prefill_replay_round_robin() {
    Scheduler scheduler;
    Slots slots{};
    slots[0]         = request(1, Request::Phase::Decode);
    slots[1]         = request(2, Request::Phase::Prefill);
    slots[2]         = request(3, Request::Phase::Replay);
    slots[3]         = request(4, Request::Phase::Prefill);
    const auto units = units_of(16);
    for (const std::uint32_t expected : {1U, 2U, 3U, 1U}) {
        const auto lane = scheduler.next_prefill(slots, 4, units);
        require(lane == expected, "prefill and replay did not receive rotating execution units");
        scheduler.prefill_executed(*lane, 4, 16, 0);
    }
    slots[2]->capture_pending = true;
    require(scheduler.next_prefill(slots, 4, units) == 3,
            "capture dependency blocked unrelated ready prefill work");
    scheduler.prefill_executed(3, 4, 16, 0);
    slots[1]->phase = Request::Phase::Decode;
    slots[3]->phase = Request::Phase::Finished;
    require(!scheduler.next_prefill(slots, 4, units), "unready prefill member was scheduled");
    slots[2]->capture_pending = false;
    require(scheduler.next_prefill(slots, 4, units) == 2,
            "completed capture did not restore replay eligibility");
}

void test_prefill_cycle_budget() {
    Scheduler scheduler;
    scheduler.configure_prefill(throughput(100, 3));
    require(scheduler.prefill_pacing().throughput_units_per_cycle == 3 &&
                scheduler.prefill_regime() == PrefillRegime::Throughput &&
                scheduler.prefill_tokens_remaining() == 100 &&
                scheduler.prefill_units_remaining() == 3 &&
                scheduler.prefill_ns_remaining() == std::numeric_limits<std::uint64_t>::max(),
            "configured pacing did not open a full Throughput cycle");
    Slots slots{};
    slots[1] = request(1, Request::Phase::Prefill);
    slots[2] = request(2, Request::Phase::Replay);
    slots[3] = request(3, Request::Phase::Prefill);

    // The first unit is admitted even when it alone exceeds the token allowance.
    auto units = units_of(0);
    units[1]   = 150;
    units[2]   = 30;
    units[3]   = 30;
    require(scheduler.next_prefill(slots, 4, units) == 1, "first unit of a cycle was refused");
    scheduler.prefill_executed(1, 4, 150, 0);
    require(scheduler.prefill_tokens_remaining() == 0 && scheduler.prefill_units_remaining() == 2,
            "actual tokens were not charged");
    require(!scheduler.next_prefill(slots, 4, units), "an exhausted token budget admitted a unit");

    // Charging follows the tokens the Session reports, not the declared upper bound.
    scheduler.begin_prefill_cycle(false);
    units[1] = units[2] = units[3] = 60;
    require(scheduler.next_prefill(slots, 4, units) == 2, "cursor did not stay after lane 1");
    scheduler.prefill_executed(2, 4, 10, 0);
    require(scheduler.prefill_tokens_remaining() == 90, "short unit was charged its bound");
    require(scheduler.next_prefill(slots, 4, units) == 3, "fitting unit was refused");
    scheduler.prefill_executed(3, 4, 60, 0);
    require(scheduler.prefill_tokens_remaining() == 30, "second unit was not charged");

    // The first lane that does not fit ends the cycle; a later, smaller lane may not jump it.
    units[1] = 40;
    units[2] = 5;
    require(!scheduler.next_prefill(slots, 4, units), "a smaller lane bypassed the blocked lane");
    scheduler.begin_prefill_cycle(false);
    require(scheduler.next_prefill(slots, 4, units) == 1,
            "blocked lane did not lead the next cycle");

    // The unit-count limit applies even when tokens remain.
    scheduler.begin_prefill_cycle(false);
    units = units_of(1);
    for (unsigned n = 0; n < 3; ++n) {
        const auto lane = scheduler.next_prefill(slots, 4, units);
        require(lane.has_value(), "unit within the count limit was refused");
        scheduler.prefill_executed(*lane, 4, 1, 0);
    }
    require(scheduler.prefill_units_remaining() == 0 && !scheduler.next_prefill(slots, 4, units),
            "unit-count limit was exceeded");
}

void test_prefill_candidates_follow_the_same_rule() {
    Scheduler scheduler;
    scheduler.configure_prefill(throughput(100, 8));
    Slots slots{};
    slots[0]   = request(1, Request::Phase::Prefill);
    slots[1]   = request(2, Request::Phase::Decode);
    slots[2]   = request(3, Request::Phase::Prefill);
    slots[3]   = request(4, Request::Phase::Prefill);
    slots[4]   = request(5, Request::Phase::Replay);
    auto units = units_of(0);
    units[0]   = 60;
    units[2]   = 40;
    units[3]   = 5;
    units[4]   = 5;

    // The first unit is free; later units are charged cumulatively; the first misfit stops
    // the list instead of letting lane 3 skip past lane 2.
    auto picked = scheduler.prefill_candidates(slots, 5, units);
    require(picked.size == 2 && picked.lanes[0] == 0 && picked.lanes[1] == 2 &&
                scheduler.next_prefill(slots, 5, units) == picked.lanes[0],
            "candidate list differs from next_prefill ordering and budget");
    scheduler.prefill_executed(0, 5, 70, 0);
    picked = scheduler.prefill_candidates(slots, 5, units);
    require(picked.size == 0 && !scheduler.next_prefill(slots, 5, units),
            "a unit that does not fit was offered");

    // Cursor order: after lane 3 runs, the list starts at lane 4 and wraps to lane 0.
    scheduler.begin_prefill_cycle(false);
    scheduler.prefill_executed(3, 5, 5, 0);
    scheduler.begin_prefill_cycle(false);
    units[2] = 30;
    picked   = scheduler.prefill_candidates(slots, 5, units);
    require(picked.size == 4 && picked.lanes[0] == 4 && picked.lanes[1] == 0 &&
                picked.lanes[2] == 2 && picked.lanes[3] == 3 && picked.lane_span().size() == 4,
            "candidate list ignored the cursor or repeated a lane");
}

// Runs prefill cycles over `lanes` prefilling lanes whose prompts are `left` tokens long. Each
// unit is limited by `chunk`, sized by prefill_unit_caps and costs `cost[lane]` ns per token.
// `regime_of(cycle)` selects the regime. Returns false when a bound or the progress guarantee
// is violated.
struct CycleRun {
    std::uint32_t lanes = 0;
    std::uint32_t chunk = 0;
    std::array<std::uint32_t, 6> left{};
    std::array<std::uint64_t, 6> cost{};
};

void run_cycles(Scheduler& scheduler, CycleRun run, bool (*waiting)(std::uint32_t),
                const char* tag) {
    Slots slots{};
    for (std::uint32_t lane = 0; lane < run.lanes; ++lane)
        slots[lane] = request(lane + 1, Request::Phase::Prefill);
    std::array<std::uint32_t, 6> last{};
    const auto pacing = scheduler.prefill_pacing();
    for (std::uint32_t cycle = 1; cycle <= 600; ++cycle) {
        const bool decode_waiting = waiting(cycle);
        scheduler.begin_prefill_cycle(decode_waiting);
        std::uint32_t tokens = 0;
        std::uint64_t ns     = 0;
        for (;;) {
            auto limits = units_of(0);
            for (std::uint32_t lane = 0; lane < run.lanes; ++lane)
                if (slots[lane]) limits[lane] = std::min(run.chunk, run.left[lane]);
            const auto caps = scheduler.prefill_unit_caps(limits);
            const auto lane = scheduler.next_prefill(slots, run.lanes, caps);
            if (!lane) break;
            require(caps[*lane] != 0 && caps[*lane] <= limits[*lane], tag);
            // The Session may process fewer tokens than the cap.
            const std::uint32_t done    = caps[*lane] > ((cycle + *lane) % 3)
                                              ? caps[*lane] - (cycle + *lane) % 3
                                              : caps[*lane];
            const std::uint64_t elapsed = std::uint64_t{done} * run.cost[*lane];
            run.left[*lane] -= done;
            tokens += done;
            ns += elapsed;
            last[*lane] = cycle;
            scheduler.prefill_executed(*lane, run.lanes, done, elapsed);
            if (run.left[*lane] == 0) {
                slots[*lane].reset();
                scheduler.prefill_lane_released(*lane);
            }
        }
        if (decode_waiting) {
            require(tokens <= pacing.interleave_tokens_per_cycle, tag);
        } else {
            require(tokens <= pacing.throughput_tokens_per_cycle, tag);
        }
        bool live = false;
        for (std::uint32_t lane = 0; lane < run.lanes; ++lane) {
            if (!slots[lane]) continue;
            live = true;
            require(cycle - last[lane] <= run.lanes, tag);
        }
        if (!live) return;
    }
    throw std::runtime_error(tag);
}

void test_prefill_progress_and_decode_gap_bounds() {
    for (const std::uint32_t budget : {16U, 40U, 64U, 100U, 200U, 512U}) {
        for (const std::uint32_t units_cap : {1U, 2U, 5U}) {
            Scheduler scheduler;
            scheduler.configure_prefill(throughput(budget, units_cap));
            CycleRun run{.lanes = 5, .chunk = 64, .left = {300, 64, 10, 200, 130}, .cost = {}};
            run_cycles(
                scheduler, run, [](std::uint32_t) { return false; },
                "Throughput prefill violated its allowance or starved a lane");
        }
    }
}

void test_interleave_bounds_tokens_and_time() {
    constexpr std::uint64_t cost = 20'000; // ns per token
    Scheduler scheduler;
    scheduler.configure_prefill({.throughput_tokens_per_cycle = 4096,
                                 .throughput_units_per_cycle  = 8,
                                 .interleave_ns               = 10'000'000,
                                 .interleave_tokens_per_cycle = 1000,
                                 .interleave_cold_tokens      = 256,
                                 .unit_granularity            = 64});
    scheduler.begin_prefill_cycle(true);
    require(scheduler.prefill_regime() == PrefillRegime::Interleave &&
                scheduler.prefill_tokens_remaining() == 1000 &&
                scheduler.prefill_ns_remaining() == 10'000'000 &&
                scheduler.prefill_units_remaining() == std::numeric_limits<std::uint32_t>::max(),
            "an Interleave cycle did not open with the Interleave allowance");

    Slots slots{};
    for (std::uint32_t lane = 0; lane < 3; ++lane)
        slots[lane] = request(lane + 1, Request::Phase::Prefill);
    // Teach every lane its cost, then start a new cycle.
    for (std::uint32_t lane = 0; lane < 3; ++lane)
        scheduler.prefill_executed(lane, 3, 100, 100 * cost);
    scheduler.begin_prefill_cycle(true);

    // 10 ms at 20 us/token affords 500 tokens: below the 1000-token cap, so time binds, and the
    // unit is a multiple of the 64-token granularity.
    auto limits = units_of(0);
    limits.fill(2048);
    auto caps = scheduler.prefill_unit_caps(limits);
    require(caps[0] == 448, "a time-bound unit was not the largest granule multiple that fits");
    // The lane behind it only gets what is left of the time: 56 tokens is below one granule.
    require(caps[1] == 0 && caps[2] == 0, "a unit below one granule followed the first unit");

    std::uint32_t tokens = 0;
    std::uint64_t ns     = 0;
    while (const auto lane =
               scheduler.next_prefill(slots, 3, scheduler.prefill_unit_caps(limits))) {
        const auto cap = scheduler.prefill_unit_caps(limits)[*lane];
        tokens += cap;
        ns += cap * cost;
        scheduler.prefill_executed(*lane, 3, cap, cap * cost);
    }
    require(tokens <= 1000 && ns <= 10'000'000, "Interleave exceeded its token or time bound");

    // With cheap tokens the token allowance binds, shared in cursor order and exact (not rounded).
    Scheduler cheap;
    cheap.configure_prefill(scheduler.prefill_pacing());
    cheap.begin_prefill_cycle(true);
    for (std::uint32_t lane = 0; lane < 3; ++lane)
        cheap.prefill_executed(lane, 3, 1000, 1000 * 1000);
    cheap.begin_prefill_cycle(true);
    limits    = units_of(0);
    limits[0] = limits[1] = limits[2] = 700;
    caps                              = cheap.prefill_unit_caps(limits);
    require(caps[0] == 700 && caps[1] == 300 && caps[2] == 0,
            "the token allowance was not handed out in cursor order");
    require(cheap.prefill_candidates(slots, 3, caps).size == 2,
            "an Interleave flight did not follow the caps");

    // The first unit of a cycle gets a granule even when it does not fit the time allowance.
    scheduler.begin_prefill_cycle(true);
    scheduler.prefill_executed(0, 3, 1, 5'000'000);
    scheduler.begin_prefill_cycle(true);
    limits    = units_of(0);
    limits[0] = 2048;
    caps      = scheduler.prefill_unit_caps(limits);
    require(caps[0] == 64, "the first unit made no progress under a tight time allowance");
}

void test_throughput_admits_packed_flights() {
    Scheduler scheduler;
    scheduler.configure_prefill(throughput(4096, 6));
    scheduler.begin_prefill_cycle(false);
    Slots slots{};
    for (std::uint32_t lane = 0; lane < 6; ++lane)
        slots[lane] = request(lane + 1, Request::Phase::Prefill);

    // Unequal windows, including 1 and non-multiples of 128, share one flight.
    auto limits = units_of(0);
    limits[0]   = 1000;
    limits[1]   = 300;
    limits[2]   = 1;
    limits[3]   = 777;
    limits[4]   = 2048;
    limits[5]   = 2048;
    auto caps   = scheduler.prefill_unit_caps(limits);
    require(caps[0] == 1000 && caps[1] == 300 && caps[2] == 1 && caps[3] == 777 &&
                caps[4] == 2018 && caps[5] == 0,
            "Throughput caps are not the limits truncated by the remaining allowance");
    const auto flight = scheduler.prefill_candidates(slots, 6, caps);
    require(flight.size == 5 && flight.lanes[0] == 0 && flight.lanes[4] == 4,
            "a Throughput flight did not carry every capped segment");
    std::uint64_t sum = 0;
    for (const auto lane : flight.lane_span()) sum += caps[lane];
    require(sum <= 4096, "a packed flight exceeded the Throughput token allowance");

    // Full windows pack as many lanes as the allowance holds; the unit count binds beyond that.
    scheduler.configure_prefill(throughput(1 << 20, 3));
    limits.fill(512);
    caps = scheduler.prefill_unit_caps(limits);
    require(scheduler.prefill_candidates(slots, 6, caps).size == 3 &&
                std::count_if(caps.begin(), caps.end(), [](auto cap) { return cap != 0; }) == 3,
            "the Throughput unit allowance was not applied to the caps");
}

void test_no_lane_starves_across_regimes() {
    for (std::uint32_t lanes = 2; lanes <= 6; ++lanes) {
        Scheduler scheduler;
        scheduler.configure_prefill({.throughput_tokens_per_cycle = 256,
                                     .throughput_units_per_cycle  = 4,
                                     .interleave_ns               = 3'000'000,
                                     .interleave_tokens_per_cycle = 128,
                                     .interleave_cold_tokens      = 32,
                                     .unit_granularity            = 16});
        CycleRun run{.lanes = lanes,
                     .chunk = 128,
                     .left  = {900, 128, 40, 513, 300, 1024},
                     .cost  = {20'000, 5'000, 80'000, 1'000, 40'000, 10'000}};
        run_cycles(
            scheduler, run, [](std::uint32_t) { return true; },
            "Interleave starved a lane or exceeded its allowance");
        run_cycles(
            scheduler, run, [](std::uint32_t cycle) { return cycle % 3 != 0; },
            "alternating regimes starved a lane or exceeded their allowance");
    }
}

void test_cost_estimate_takes_effect() {
    Scheduler scheduler;
    scheduler.configure_prefill({.interleave_ns               = 8'000'000,
                                 .interleave_tokens_per_cycle = 4096,
                                 .interleave_cold_tokens      = 128,
                                 .unit_granularity            = 1});
    require(!scheduler.prefill_cost_ns_per_token(0), "a lane had a cost before any unit ran");
    scheduler.begin_prefill_cycle(true);
    scheduler.prefill_executed(0, 2, 200, 200 * 10'000); // 10 us/token
    scheduler.prefill_executed(1, 2, 200, 200 * 40'000); // 40 us/token
    require(scheduler.prefill_cost_ns_per_token(0) == 10'000 &&
                scheduler.prefill_cost_ns_per_token(1) == 40'000,
            "the first sample did not become the estimate");

    scheduler.begin_prefill_cycle(true);
    auto limits = units_of(0);
    limits[0]   = 4096;
    limits[1]   = 4096;
    auto caps   = scheduler.prefill_unit_caps(limits);
    require(caps[0] == 800 && caps[1] == 0, "the cap did not follow the lane's cost");
    scheduler.begin_prefill_cycle(true);
    scheduler.prefill_executed(0, 2, 100, 0);
    limits[0] = 0;
    caps      = scheduler.prefill_unit_caps(limits);
    require(caps[1] > 0 && caps[1] <= 8'000'000 / 40'000,
            "an expensive lane exceeded the time its estimate affords");

    // A new sample moves the estimate toward it, and release forgets it.
    scheduler.begin_prefill_cycle(true);
    scheduler.prefill_executed(1, 2, 100, 100 * 20'000);
    const auto blended = scheduler.prefill_cost_ns_per_token(1);
    require(blended && *blended > 20'000 && *blended < 40'000,
            "a new sample did not move the estimate");
    scheduler.prefill_lane_released(1);
    require(!scheduler.prefill_cost_ns_per_token(1), "a released lane kept its estimate");
}

void test_cold_lane_uses_cold_tokens() {
    Scheduler scheduler;
    scheduler.configure_prefill({.interleave_ns               = 50'000'000,
                                 .interleave_tokens_per_cycle = 4096,
                                 .interleave_cold_tokens      = 192,
                                 .unit_granularity            = 64});
    scheduler.begin_prefill_cycle(true);
    auto limits = units_of(0);
    limits[0]   = 2048;
    limits[1]   = 100;
    limits[2]   = 2048;
    auto caps   = scheduler.prefill_unit_caps(limits);
    require(caps[0] == 192 && caps[1] == 100 && caps[2] == 192,
            "lanes without an estimate were not capped at the cold tokens");

    // Warming lane 0 lifts only its own cap; lane 2 stays cold. Throughput ignores cold tokens.
    scheduler.prefill_executed(0, 3, 192, 192 * 1'000);
    scheduler.begin_prefill_cycle(true);
    caps = scheduler.prefill_unit_caps(limits);
    require(caps[0] > 192 && caps[2] == 192, "a warm lane stayed at the cold tokens");
    scheduler.begin_prefill_cycle(false);
    require(scheduler.prefill_unit_caps(limits)[2] == 2048 ||
                scheduler.prefill_unit_caps(limits)[2] > 192,
            "Throughput applied the Interleave cold tokens");
}

void test_decode_and_control_membership() {
    Scheduler scheduler;
    Slots slots{};
    slots[0]                      = request(1, Request::Phase::Decode);
    slots[0]->budget->tokens      = 11;
    slots[0]->output.model_tokens = 3;
    slots[2]                      = request(2, Request::Phase::Decode);
    slots[2]->recovery_pending    = true;
    slots[3]                      = request(3, Request::Phase::Decode);
    slots[3]->capture_pending     = true;
    const auto round              = scheduler.build_round_membership(slots, 4);
    require(round.size == 2 && round.lanes[0] == 0 && round.lanes[1] == 2 &&
                round.sequences[0] == 1 && round.sequences[1] == 2 &&
                round.budgets[0].generated_tokens_remaining == 3 &&
                round.budgets[1].generated_tokens_remaining == 1,
            "compact decode lost output limits, first-unit limits or capture dependencies");

    slots[0]->phase          = Request::Phase::Control;
    slots[0]->output.control = {7, 8, 9};
    slots[2]->phase          = Request::Phase::Control;
    slots[2]->output.control = {10, 11, 12};
    const auto control       = scheduler.build_control_membership(slots, 4);
    require(control.size == 2 && control.row_stride == 3 &&
                control.tokens == std::vector<ninfer::TokenId>({7, 8, 9, 10, 11, 12}) &&
                control.sequences[0] == 1 && control.sequences[1] == 2,
            "compact control membership changed committed control spans or request order");
}

} // namespace

int main() {
    try {
        test_fifo_bypass_survives_pressure();
        test_scan_progress_and_waiter_barrier();
        test_scan_keeps_unexamined_candidates();
        test_pressure_handoff_and_fair_service();
        test_bounded_service_under_continuous_contention();
        test_recovery_cost_has_a_bounded_service_allowance();
        test_prefill_replay_round_robin();
        test_prefill_cycle_budget();
        test_prefill_candidates_follow_the_same_rule();
        test_prefill_progress_and_decode_gap_bounds();
        test_interleave_bounds_tokens_and_time();
        test_throughput_admits_packed_flights();
        test_no_lane_starves_across_regimes();
        test_cost_estimate_takes_effect();
        test_cold_lane_uses_cold_tokens();
        test_decode_and_control_membership();
        std::cout << "ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
