#include "runtime/engine/materialization_budget.h"

#include <iostream>
#include <stdexcept>

using namespace ninfer::runtime;

static void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}

int main() {
    constexpr std::uint64_t ms = 1'000'000;
    try {
        auto idle = PlanningAllowance::boundary(0, 0);
        MaterializationSearchBudget expensive(idle, 0, 80'000 * ms);
        require(expensive.allow(4 * ms, 2 * ms, 3 * ms, 70'000 * ms, true, 1),
                "valuable completion could not cross the initial window");
        require(expensive.granted_ns() == 10 * ms && expensive.renewals() == 1,
                "extension did not retain cumulative accounting");
        require(!expensive.allow(9 * ms, 2 * ms, 2 * ms, ms, true, 2),
                "improved incumbent retained the expensive root's budget");
        require(expensive.stop_reason() ==
                    ninfer::MaterializationStopReason::InsufficientExpectedGain,
                "economic stopping was reported as wall exhaustion");

        MaterializationSearchBudget discovery(idle, 0, 80'000 * ms);
        require(discovery.allow(5 * ms, ms, 4 * ms, 70'000 * ms, false, 1),
                "unknown candidate could not receive bounded discovery");
        require(!discovery.allow(10 * ms, ms, ms, 70'000 * ms, false, 2),
                "unknown candidate repeatedly renewed discovery");
        require(discovery.allow(10 * ms, ms, ms, 70'000 * ms, true, 2),
                "complete prediction could not continue after discovery");

        auto busy = PlanningAllowance::boundary(2, 0);
        MaterializationSearchBudget first(busy, busy.limit_ns - 4 * ms, 80'000 * ms);
        require(first.granted_ns() == 4 * ms, "mandatory work did not consume boundary time");
        MaterializationSearchBudget backfill(busy, busy.limit_ns - ms, 80'000 * ms);
        require(backfill.granted_ns() == ms, "backfill reset the boundary allowance");
        require(!backfill.allow(busy.limit_ns, 1, 1, 70'000 * ms, true, 1),
                "search delayed runnable requests beyond their boundary");
        require(backfill.overshoot(busy.limit_ns + ms) == ms,
                "indivisible overrun was not measured");

        // The concurrent 55K rotation spends ~3.5 ms preparing/validating identities. A useful
        // complete target must still be assessable; discarding the whole cache makes all six
        // subsequent continuations cold. The old 5 ms boundary left only 1.5 ms and failed here.
        MaterializationSearchBudget restore_after_setup(busy, 3 * ms + ms / 2, 80'000 * ms);
        require(restore_after_setup.allow(3 * ms + ms / 2, 2 * ms, 2 * ms, 70'000 * ms, true, 1),
                "mandatory setup starved the first complete reuse assessment in a busy boundary");

        MaterializationSearchBudget cheap(idle, 0, ms);
        require(cheap.granted_ns() == ms / 20, "cheap request received a minimum 5 ms grant");
        require(!cheap.allow(ms / 20, ms, ms, ms, false, 1),
                "discovery ignored the economic cap for a cheap request");
        MaterializationSearchBudget saturated(idle, 0, UINT64_MAX);
        require(saturated.granted_ns() == 0 && !saturated.allow(0, ms, ms, UINT64_MAX, true, 1),
                "saturated cost was used as evidence of unlimited gain");
        MaterializationSearchBudget seeded(idle, 0, 80'000 * ms);
        require(!seeded.allow(5 * ms, ms, ms, 70'000 * ms, false, 1, false),
                "an already-seeded candidate renewed solely on an incomplete optimistic estimate");
        require(seeded.allow(5 * ms, ms, ms, 70'000 * ms, true, 1, false),
                "a complete profitable refinement was denied after seeding");
        std::atomic<bool> cancelled{false};
        auto controlled                = PlanningAllowance::boundary(0, 0);
        controlled.cancellation        = &cancelled;
        controlled.control_deadline_ns = 3 * ms;
        require(controlled.remaining(2 * ms) == ms, "control deadline did not constrain planning");
        cancelled.store(true);
        require(controlled.remaining(0) == 0, "cancelled request kept optional planning headroom");
        MaterializationSearchBudget stalled(idle, 0, 80'000 * ms);
        require(stalled.allow(5 * ms, ms, ms, 70'000 * ms, true, 7), "first forecast grant failed");
        require(!stalled.allow(10 * ms, ms, ms, 70'000 * ms, true, 7),
                "stalled work renewed its allowance");
        std::cout << "ok\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
