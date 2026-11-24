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
using Slots      = std::vector<RequestPtr>;

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
        require(!candidates.empty() && candidates.front() == head &&
                    candidates.size() <= 1 + Scheduler::admission_scan_batch,
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
    require(after_cancel.size() > 1 && after_cancel.size() <= 1 + Scheduler::admission_scan_batch &&
                after_cancel.front()->id == 13 && after_cancel.front()->admission_bypasses == 0,
            "head removal transferred its exhausted credit to the next request");
}

void test_scan_progress_and_waiter_barrier() {
    Scheduler scheduler;
    std::deque<RequestPtr> fresh;
    for (std::uint64_t id = 1; id <= 3 * Scheduler::admission_scan_batch + 2; ++id)
        fresh.push_back(request(id));
    std::vector<std::uint64_t> seen;
    const auto observe = [&](const std::vector<RequestPtr>& candidates) {
        require(!candidates.empty() && candidates.front() == fresh.front() &&
                    candidates.size() <= 1 + Scheduler::admission_scan_batch,
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
        next.size() == 1 + Scheduler::admission_scan_batch &&
            next[1]->id == 2 + Scheduler::admission_scan_batch,
        "a later fitting request required another arrival or resident completion to be checked");
    observe(next);

    // Remove the last examined ticket and add an arrival. The next event must preserve
    // cursor progress rather than restart indefinitely at the first failed requests.
    std::erase(fresh, next.back());
    fresh.push_back(request(3 * Scheduler::admission_scan_batch + 3));
    scheduler.begin_admission_scan();
    const auto after_change = scheduler.fresh_candidates(fresh, 2);
    require(after_change.size() == 1 + Scheduler::admission_scan_batch &&
                after_change[1]->id == 2 + 2 * Scheduler::admission_scan_batch,
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
    require(first.size() > 2 && first.size() <= 1 + Scheduler::admission_scan_batch,
            "admission did not expose a bounded candidate batch");
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

template <class Slots>
std::optional<std::uint32_t> victim(const Scheduler& scheduler, const Slots& slots,
                                    std::span<const std::uint32_t> excluded = {}) {
    const auto candidates = scheduler.preemption_candidates(slots);
    for (auto lane : candidates)
        if (std::find(excluded.begin(), excluded.end(), lane) == excluded.end()) return lane;
    return {};
}

void test_pressure_handoff_and_fair_service() {
    Scheduler scheduler;
    Slots slots(4);
    auto a = slots[0] = request(10, Request::Phase::Decode);
    auto b = slots[1] = request(20, Request::Phase::Decode);
    auto c = slots[2] = request(30, Request::Phase::Decode);
    for (const auto& value : {a, b, c}) admit(scheduler, value);

    scheduler.require_progress(a->id);
    require(victim(scheduler, slots) == 2, "capacity handoff did not exclude its beneficiary");
    const std::array excluded{2U};
    require(victim(scheduler, slots, excluded) == 1,
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
    require(victim(scheduler, slots) == 1,
            "a waiting request could not preempt an older served resident");
    b->recovery_pending = true;
    require(victim(scheduler, slots) == 0, "protected Native recovery was selected as a victim");
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

void test_batch_completion_preserves_service_order() {
    Scheduler scheduler;
    for (std::uint64_t id = 1; id <= 128; ++id) admit(scheduler, request(id));
    scheduler.require_progress(1);
    std::vector<std::uint64_t> ended;
    for (std::uint64_t id = 1; id <= 128; id += 2) ended.push_back(id);
    scheduler.finished(ended);
    require(scheduler.progress_owner() == 2, "batch completion lost the next service owner");
    scheduler.served(2, Scheduler::service_quantum_ns);
    require(scheduler.advance_service(true, false) && scheduler.progress_owner() == 4,
            "batch completion reordered surviving requests");
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

using ninfer::runtime::PrefillBudgetPolicy;
using ninfer::runtime::PrefillCandidateInfo;

struct PrefillFixture {
    Scheduler scheduler;
    Slots slots;
    std::vector<PrefillCandidateInfo> infos;
    Scheduler::PrefillMembership membership;
    PrefillBudgetPolicy policy;

    explicit PrefillFixture(std::uint32_t capacity)
        : slots(capacity), infos(capacity), membership(capacity),
          policy{4096, 4096, capacity, capacity, 64} {}

    void text(std::uint32_t lane, std::uint32_t tokens, std::uint32_t placement = 0) {
        slots[lane] = request(lane + 1, Request::Phase::Prefill);
        infos[lane] = {{placement}, tokens, tokens};
    }

    void build(std::optional<Request::SequenceHandle> preferred = {}) {
        scheduler.build_prefill_membership(
            slots, policy, preferred, [](auto) { return true; },
            [&](auto sequence) { return infos.at(sequence - 1); }, membership);
    }

    void check(std::initializer_list<std::uint32_t> lanes,
               std::initializer_list<std::uint32_t> grants, bool replay = false) const {
        require(std::ranges::equal(membership.lanes, lanes) &&
                    membership.sequences.size() == lanes.size() &&
                    membership.grants.size() == grants.size() && membership.is_replay == replay,
                "prefill membership or execution kind differs");
        std::size_t row = 0;
        for (auto grant : grants) {
            require(membership.grants[row].max_tokens == grant &&
                        membership.sequences[row] == membership.lanes[row] + 1,
                    "prefill token grant or sequence alignment differs");
            ++row;
        }
    }
};

void test_prefill_replay_round_robin() {
    PrefillFixture f(4);
    f.policy.max_sequences_per_step = 1;
    f.slots[0]                      = request(1, Request::Phase::Decode);
    f.text(1, 512);
    f.slots[2] = request(3, Request::Phase::Replay);
    f.text(3, 700);
    for (const std::uint32_t expected : {1U, 2U, 3U, 1U}) {
        f.build();
        f.check({expected}, {f.infos[expected].unit_max_tokens}, expected == 2);
        f.scheduler.prefill_executed(expected, 4);
    }
    f.slots[2]->capture_pending = true;
    f.build(3);
    f.check({3}, {700});
    f.scheduler.prefill_executed(3, 4);
    f.slots[1]->phase = Request::Phase::Decode;
    f.slots[3]->phase = Request::Phase::Finished;
    f.build();
    f.check({}, {});
    f.slots[2]->capture_pending = false;
    f.build();
    f.check({2}, {0}, true);
}

void test_prefill_packing_and_alignment() {
    PrefillFixture f(4);
    f.text(0, 300);
    f.text(1, 500);
    f.text(2, 1200);
    f.build();
    f.check({0, 1, 2}, {300, 500, 1200});
    f.text(3, 8192);
    f.build();
    f.check({0, 1, 2, 3}, {300, 500, 1200, 2048});

    f.policy.max_sequences_per_placement = 2;
    f.build();
    f.check({0, 1}, {300, 500});
    f.policy.max_sequences_per_placement = 4;
    f.policy.max_tokens_global           = 32;
    f.build();
    f.check({0}, {32});
}

void test_prefill_water_filling() {
    PrefillFixture f(4);
    f.text(0, 8192, 0);
    f.text(1, 8192, 42);
    f.build();
    f.check({0, 1}, {2048, 2048});
    f.text(1, 512, 42);
    f.build();
    f.check({0, 1}, {3584, 512});
    f.slots[1].reset();
    f.build();
    f.check({0}, {4096});

    f.text(0, 2048);
    f.text(1, 600);
    f.text(2, 512, 42);
    f.build();
    f.check({0, 2, 1}, {2048, 512, 600});

    PrefillFixture dp(8);
    dp.policy.max_tokens_global           = 32768;
    dp.policy.max_sequences_per_placement = 1;
    for (std::uint32_t lane = 0; lane < 8; ++lane) dp.text(lane, 8192, lane);
    dp.build();
    dp.check({0, 1, 2, 3, 4, 5, 6, 7}, {4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096});
}

void test_prefill_exclusive_pressure_and_zero_token() {
    PrefillFixture f(4);
    f.text(0, 8192);
    f.text(1, 320);
    f.infos[1].exclusive = true;
    f.slots[2]           = request(3, Request::Phase::Replay);
    f.text(3, 0, 99);
    f.infos[3].zero_token_sample = true;
    f.build();
    f.check({0, 3}, {4096, 0});
    f.scheduler.prefill_executed(f.membership.lanes.front(), 4);
    f.build();
    f.check({1}, {320});
    f.scheduler.prefill_executed(f.membership.lanes.front(), 4);
    f.build();
    f.check({2}, {0}, true);
    f.build(4);
    f.check({3, 0}, {0, 4096});

    for (const auto& value : f.slots) admit(f.scheduler, value);
    f.scheduler.require_progress(1);
    f.build(4);
    f.check({0}, {4096});
    f.slots[0]->capture_pending = true;
    f.build(4);
    f.check({}, {});
}

void test_decode_and_control_membership() {
    Scheduler scheduler;
    Slots slots(4);
    slots[0]                      = request(1, Request::Phase::Decode);
    slots[0]->budget->tokens      = 11;
    slots[0]->output.model_tokens = 3;
    slots[2]                      = request(2, Request::Phase::Decode);
    slots[2]->recovery_pending    = true;
    slots[3]                      = request(3, Request::Phase::Decode);
    slots[3]->capture_pending     = true;
    Scheduler::RoundMembership round(4);
    scheduler.build_round_membership(slots, [](auto) { return true; }, round);
    require(round.size() == 2 && round.lanes[0] == 0 && round.lanes[1] == 2 &&
                round.sequences[0] == 1 && round.sequences[1] == 2 &&
                round.budgets[0].generated_tokens_remaining == 3 &&
                round.budgets[1].generated_tokens_remaining == 1,
            "compact decode lost output limits, first-unit limits or capture dependencies");

    slots[0]->phase          = Request::Phase::Control;
    slots[0]->output.control = {7, 8, 9};
    slots[2]->phase          = Request::Phase::Control;
    slots[2]->output.control = {10, 11, 12};
    Scheduler::ControlMembership control(4);
    scheduler.build_control_membership(slots, [](auto) { return true; }, control);
    require(control.size() == 2 && control.row_stride == 3 &&
                control.tokens == std::vector<ninfer::TokenId>({7, 8, 9, 10, 11, 12}) &&
                control.sequences[0] == 1 && control.sequences[1] == 2,
            "compact control membership changed committed control spans or request order");
}

void test_large_sparse_membership_and_reuse() {
    constexpr std::uint32_t capacity = 128;
    Scheduler scheduler;
    Slots slots(capacity);
    Scheduler::RoundMembership round(capacity);
    for (std::uint32_t lane = 0; lane < capacity; ++lane) {
        if (lane % 3 == 0) continue;
        slots[lane]                  = request(lane + 1, Request::Phase::Decode);
        slots[lane]->capture_pending = lane % 5 == 0;
    }
    scheduler.build_round_membership(slots, [](auto lane) { return lane % 7 != 0; }, round);
    std::vector<std::uint32_t> expected;
    for (std::uint32_t lane = 0; lane < capacity; ++lane)
        if (lane % 3 && lane % 5 && lane % 7) expected.push_back(lane);
    require(round.lanes == expected && round.size() > 8,
            "large decode membership lost sparse or eligible lanes");
    for (std::size_t row = 0; row < round.size(); ++row)
        require(round.sequences[row] == expected[row] + 1 &&
                    round.budgets[row].generated_tokens_remaining == 8,
                "compact decode membership lost sequence or budget alignment");

    slots.assign(capacity, nullptr);
    slots[127]                   = request(1000, Request::Phase::Decode);
    slots[127]->recovery_pending = true;
    scheduler.build_round_membership(slots, [](auto) { return true; }, round);
    require(round.size() == 1 && round.lanes[0] == 127 && round.sequences[0] == 1000 &&
                round.budgets[0].generated_tokens_remaining == 1,
            "reused membership retained old rows or confused a reused lane");
    scheduler.build_round_membership(slots, [](auto) { return false; }, round);
    require(round.empty() && round.sequence_span().empty() && round.budget_span().empty(),
            "empty membership retained stale row payloads");
}

void test_control_membership_groups_eligible_widths() {
    Scheduler scheduler;
    Slots slots(128);
    for (const auto lane : {0U, 9U, 100U, 127U})
        slots[lane] = request(lane + 1, Request::Phase::Control);
    slots[0]->output.control   = {1};
    slots[9]->output.control   = {2, 3};
    slots[100]->output.control = {4};
    slots[127]->output.control = {5, 6};
    Scheduler::ControlMembership control(128);
    scheduler.build_control_membership(slots, [](auto lane) { return lane != 0; }, control);
    require(control.lanes == std::vector<std::uint32_t>({9, 127}) && control.row_stride == 2 &&
                control.sequences == std::vector<std::uint64_t>({10, 128}) &&
                control.tokens == std::vector<ninfer::TokenId>({2, 3, 5, 6}),
            "ineligible control row set the width or mismatched widths shared a batch");
    scheduler.build_control_membership(slots, [](auto) { return true; }, control);
    require(control.lanes == std::vector<std::uint32_t>({0, 100}) && control.row_stride == 1 &&
                control.tokens == std::vector<ninfer::TokenId>({1, 4}),
            "reused control membership retained a previous width or token payload");
    scheduler.build_control_membership(slots, [](auto) { return false; }, control);
    require(control.empty() && control.tokens.empty() && control.row_stride == 0,
            "empty control membership retained an old width or tokens");
}

} // namespace

int main() {
    try {
        test_fifo_bypass_survives_pressure();
        test_scan_progress_and_waiter_barrier();
        test_scan_keeps_unexamined_candidates();
        test_pressure_handoff_and_fair_service();
        test_batch_completion_preserves_service_order();
        test_bounded_service_under_continuous_contention();
        test_recovery_cost_has_a_bounded_service_allowance();
        test_prefill_replay_round_robin();
        test_prefill_packing_and_alignment();
        test_prefill_water_filling();
        test_prefill_exclusive_pressure_and_zero_token();
        test_decode_and_control_membership();
        test_large_sparse_membership_and_reuse();
        test_control_membership_groups_eligible_widths();
        std::cout << "ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
