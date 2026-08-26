#include "runtime/engine/admission_policy.h"
#include "runtime/engine/scheduler.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

struct SchedulerRequest {
    using SequenceHandle = std::uint64_t;

    struct Budget {
        std::uint32_t tokens = 0;

        [[nodiscard]] std::uint32_t remaining() const noexcept { return tokens; }
    };

    struct Output {
        std::uint32_t model_tokens = 0;
        std::vector<ninfer::TokenId> control;

        [[nodiscard]] std::uint32_t
        model_token_budget_remaining(std::uint32_t total) const noexcept {
            return std::min(total, model_tokens);
        }

        [[nodiscard]] std::span<const ninfer::TokenId> pending_control_tokens() const noexcept {
            return control;
        }
    };

    enum class State : std::uint8_t { Decode, Control };

    [[nodiscard]] bool is_decode_ready() const noexcept { return state == State::Decode; }

    [[nodiscard]] bool is_control_ready() const noexcept { return state == State::Control; }

    std::uint64_t id                              = 0;
    std::uint64_t remaining_service_work          = 1;
    std::uint64_t backfill_epoch                  = 0;
    ninfer::runtime::BackfillClass backfill_class = ninfer::runtime::BackfillClass::None;
    State state                                   = State::Decode;
    bool capture_pending                          = false;
    std::optional<Budget> budget;
    std::optional<SequenceHandle> sequence;
    Output output;
};

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

template <class Function>
bool throws_logic(Function&& function) {
    try {
        function();
    } catch (const std::logic_error&) { return true; }
    return false;
}

} // namespace

int main() {
    using ninfer::runtime::ActiveAdmissionSnapshot;
    using ninfer::runtime::BackfillClass;
    using Scheduler = ninfer::runtime::Scheduler<SchedulerRequest>;

    int failures = 0;
    const std::array<ActiveAdmissionSnapshot, 2> incumbents{
        ActiveAdmissionSnapshot{.request_id = 1},
        ActiveAdmissionSnapshot{.request_id = 2},
    };

    auto protection = ninfer::runtime::make_admission_protection(7, 10, 31, incumbents);
    failures += check(protection.donor_count == 2 && protection.donor_ids[0] == 1 &&
                          protection.donor_ids[1] == 2 && protection.resource_revision == 31,
                      "protection did not freeze every incumbent donor");
    failures += check(ninfer::runtime::protection_has_live_donor(protection, incumbents),
                      "live donor was not recognized");
    failures +=
        check(ninfer::runtime::persistent_backfill_is_authorized(protection, 11, incumbents, 31),
              "matching Program proof did not authorize persistent backfill");
    failures +=
        check(!ninfer::runtime::persistent_backfill_is_authorized(protection, 11, incumbents, 30),
              "stale Program proof authorized persistent backfill");

    const std::array<ActiveAdmissionSnapshot, 3> with_borrower{
        incumbents[0],
        incumbents[1],
        ActiveAdmissionSnapshot{
            .request_id     = 11,
            .backfill_epoch = 7,
            .backfill_class = BackfillClass::Persistent,
        },
    };
    ninfer::runtime::rebind_admission_protection(protection, with_borrower, 32);
    failures += check(
        protection.donor_count == 2 && protection.resource_revision == 32 &&
            ninfer::runtime::persistent_backfill_is_authorized(protection, 12, with_borrower, 32),
        "revision rebind changed the frozen donor partition");

    const std::array<ActiveAdmissionSnapshot, 1> borrower_only{with_borrower[2]};
    ninfer::runtime::rebind_admission_protection(protection, borrower_only, 33);
    failures += check(
        !ninfer::runtime::protection_has_live_donor(protection, borrower_only) &&
            !ninfer::runtime::persistent_backfill_is_authorized(protection, 12, borrower_only, 33),
        "backfill remained open after every frozen donor completed");

    const std::array<ActiveAdmissionSnapshot, 1> unknown_active{
        ActiveAdmissionSnapshot{.request_id = 99},
    };
    failures +=
        check(throws_logic([&] {
                  ninfer::runtime::rebind_admission_protection(protection, unknown_active, 34);
              }),
              "unclassified post-protection active request was accepted as a donor");

    Scheduler scheduler;
    scheduler.observe_fifo_head(10);
    failures += check(scheduler.protect_blocked_head(10, incumbents, 40),
                      "blocked FIFO head did not open a donor epoch");
    auto stale = scheduler.qualify_backfill(11, 50, incumbents, 40);
    failures += check(stale && stale->backfill_class() == BackfillClass::Persistent &&
                          scheduler.validate_grant(*stale),
                      "Program-proved borrower did not receive a persistent grant");
    if (!stale) { return 1; }
    const std::uint64_t epoch = stale->protection_epoch();
    failures += check(scheduler.protect_blocked_head(10, incumbents, 41) &&
                          !scheduler.validate_grant(*stale),
                      "resource revision did not invalidate an uncommitted grant");

    auto first = scheduler.qualify_backfill(11, 50, incumbents, 41);
    failures +=
        check(first && first->protection_epoch() == epoch && first->resource_revision() == 41,
              "revalidated borrower changed the logical protection epoch");
    if (!first) { return 1; }
    scheduler.commit_admission(std::move(*first));

    const std::array<ActiveAdmissionSnapshot, 3> active_after_first{
        incumbents[0],
        incumbents[1],
        ActiveAdmissionSnapshot{
            .request_id     = 11,
            .backfill_epoch = epoch,
            .backfill_class = BackfillClass::Persistent,
        },
    };
    failures += check(scheduler.protect_blocked_head(10, active_after_first, 42),
                      "existing persistent borrower prevented revision revalidation");
    auto second = scheduler.qualify_backfill(12, 1, active_after_first, 42);
    failures += check(second && scheduler.validate_grant(*second),
                      "second Program-proved borrower was not cumulatively admitted");

    scheduler.on_waiting_removed(10);
    scheduler.observe_fifo_head(14);
    auto head_grant = scheduler.grant_head(14, 1);
    failures += check(scheduler.validate_grant(head_grant),
                      "new FIFO head inherited stale protection state");
    scheduler.commit_admission(std::move(head_grant));

    using ExecutionAction = Scheduler::ExecutionAction;
    failures += check(scheduler.should_attempt_admission(true, true, false, false, false) &&
                          !scheduler.should_attempt_admission(false, true, true, true, false) &&
                          !scheduler.should_attempt_admission(true, false, true, true, false) &&
                          !scheduler.should_attempt_admission(true, true, true, false, false) &&
                          scheduler.should_attempt_admission(true, true, true, true, false) &&
                          !scheduler.should_attempt_admission(true, true, false, false, true) &&
                          scheduler.choose_execution(true, false, false) == ExecutionAction::Decode,
                      "admission and GPU-unit fairness gates changed");
    scheduler.set_prefill_lane(0);
    failures +=
        check(!scheduler.should_attempt_admission(true, true, true, true, false) &&
                  scheduler.choose_execution(true, true, false) == ExecutionAction::Decode &&
                  scheduler.choose_execution(true, true, true) == ExecutionAction::Prefill,
              "prefill/decode alternation changed");
    scheduler.clear_prefill_lane(0);

    std::array<std::shared_ptr<SchedulerRequest>, ninfer::kMaximumConcurrency> slots{};
    slots[0]                      = std::make_shared<SchedulerRequest>();
    slots[0]->id                  = 21;
    slots[0]->budget              = SchedulerRequest::Budget{.tokens = 11};
    slots[0]->sequence            = 23;
    slots[0]->output.model_tokens = 3;
    const auto model_membership   = scheduler.build_round_membership(slots, 1);
    failures += check(model_membership.size == 1 &&
                          model_membership.budgets[0].generated_tokens_remaining == 3,
                      "model-token licensing changed");

    const auto active_set = scheduler.active_admission_set(slots, 1);
    failures += check(active_set.size == 1 && active_set.requests[0].request_id == 21,
                      "active logical admission snapshot lost request identity");

    slots[0]->state          = SchedulerRequest::State::Control;
    slots[0]->output.control = {7, 8, 9};
    const auto control       = scheduler.build_control_membership(slots, 1);
    failures += check(control.size == 1 && control.row_stride == 3 &&
                          control.tokens == slots[0]->output.control && control.sequences[0] == 23,
                      "ControlReady membership changed the exact control span");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
