#include "runtime/engine/scheduler.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <iostream>
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

    scheduler.preempted();
    require(!scheduler.fresh_candidates(fresh, 2).empty(),
            "one paused request blocked spare capacity from fresh work");
    scheduler.capacity_released();
    scheduler.restoration_blocked();
    scheduler.paused_queue_empty();
    require(head->admission_bypasses == 1, "pressure recovery reset a waiting request's credit");

    scheduler.begin_admission_scan();
    const auto candidates = scheduler.fresh_candidates(fresh, 2);
    const auto second     = std::find(candidates.begin(), candidates.end(), fresh[1]);
    require(second != candidates.end(), "second permitted younger request was not considered");
    scheduler.admitted(fresh, **second, 2);
    fresh.erase(fresh.begin() + 1);
    for (std::uint64_t id = 14; id < 20; ++id) {
        fresh.push_back(request(id));
        scheduler.capacity_released();
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

void test_pressure_victims_and_first_unit() {
    Scheduler scheduler;
    Slots slots{};
    slots[0] = request(30, Request::Phase::Decode);
    slots[1] = request(10, Request::Phase::Decode);
    slots[2] = request(40, Request::Phase::Decode);
    slots[3] = request(20, Request::Phase::Decode);

    require(scheduler.youngest_victim(slots, 4) == 2,
            "victim selection followed lane order instead of request age");
    const std::array skipped{2U, 0U};
    require(scheduler.youngest_victim(slots, 4, std::nullopt, skipped) == 3,
            "pressure filtering changed age order or selected a noncontributing victim");
    const std::array all_younger{2U, 0U, 3U};
    require(!scheduler.youngest_victim(slots, 4, std::nullopt, all_younger),
            "pressure filtering made the oldest request eligible for preemption");
    // Releasing this request may free no physical page because another request shares them.
    // There is deliberately no synthetic capacity credit: continue with the remaining members.
    slots[2].reset();
    require(scheduler.youngest_victim(slots, 4) == 0,
            "zero page recovery prevented the next younger victim from being selected");
    slots[0].reset();
    require(scheduler.youngest_victim(slots, 4) == 3,
            "victim reduction did not preserve the oldest request");
    slots[3].reset();
    require(!scheduler.youngest_victim(slots, 4), "the last and oldest resident was preempted");

    slots[0]                   = request(30, Request::Phase::Decode);
    slots[0]->recovery_pending = true;
    slots[2]                   = request(40, Request::Phase::Materializing);
    slots[3]                   = request(20, Request::Phase::Decode);
    require(scheduler.youngest_victim(slots, 4) == 3,
            "restoration destination or unserved first unit was chosen as a victim");
    slots[3].reset();
    require(!scheduler.youngest_victim(slots, 4),
            "first-unit protection displaced the oldest resident");
    slots[0]->recovery_pending = false;
    require(scheduler.youngest_victim(slots, 4) == 0,
            "a served resumed request remained permanently unpreemptible");
    slots[0].reset();
    slots[2].reset();
    require(scheduler.youngest_victim(slots, 4, 5) == 1,
            "an older paused request could not reclaim its lane from the last younger borrower");
    require(!scheduler.youngest_victim(slots, 4, 15),
            "a paused request displaced an older resident");
    slots[1]->recovery_pending = true;
    require(!scheduler.youngest_victim(slots, 4, 5),
            "another request preempted the protected complete recovery");
}

void test_restore_events() {
    Scheduler scheduler;
    scheduler.preempted();
    require(!scheduler.should_restore(true, false),
            "preemption immediately reopened the same failed resident set");

    Slots slots{};
    slots[0] = request(1, Request::Phase::Decode);
    (void)scheduler.build_round_membership(slots, 1);
    require(!scheduler.should_restore(true, false),
            "ordinary work without a permanent capacity event triggered restoration");
    scheduler.capacity_released();
    require(scheduler.should_restore(true, false) && !scheduler.should_restore(false, false),
            "permanent capacity event did not wake a paused request");
    scheduler.restoration_blocked();
    require(!scheduler.should_restore(true, false), "failed restore retried without a new event");
    require(scheduler.should_restore(true, true), "empty residents failed to wake paused work");
    scheduler.paused_queue_empty();
    require(!scheduler.should_restore(true, false),
            "drained paused queue kept a stale restoration event");
}

void test_prefill_replay_round_robin() {
    Scheduler scheduler;
    Slots slots{};
    slots[0] = request(1, Request::Phase::Decode);
    slots[1] = request(2, Request::Phase::Prefill);
    slots[2] = request(3, Request::Phase::Replay);
    slots[3] = request(4, Request::Phase::Prefill);
    for (const std::uint32_t expected : {1U, 2U, 3U, 1U}) {
        const auto lane = scheduler.next_prefill(slots, 4);
        require(lane == expected, "prefill and replay did not receive rotating execution units");
        scheduler.prefill_executed(*lane, 4);
    }
    slots[2]->capture_pending = true;
    require(scheduler.next_prefill(slots, 4) == 3,
            "capture dependency blocked unrelated ready prefill work");
    scheduler.prefill_executed(3, 4);
    slots[1]->phase = Request::Phase::Decode;
    slots[3]->phase = Request::Phase::Finished;
    require(!scheduler.next_prefill(slots, 4), "unready prefill member was scheduled");
    slots[2]->capture_pending = false;
    require(scheduler.next_prefill(slots, 4) == 2,
            "completed capture did not restore replay eligibility");
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
        test_pressure_victims_and_first_unit();
        test_restore_events();
        test_prefill_replay_round_robin();
        test_decode_and_control_membership();
        std::cout << "ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
