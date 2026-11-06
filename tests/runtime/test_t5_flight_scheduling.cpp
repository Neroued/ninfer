// T5 / F3: multi-request prefill scheduling, through Scheduler's public interface only.
// A "flight" is the set of lanes returned by prefill_candidates(); the Engine commits it once
// and reports each segment to prefill_executed() as one unit.
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
    enum class Phase { Waiting, Prefill, Replay, Decode };

    struct Budget {
        [[nodiscard]] std::uint32_t remaining() const noexcept { return 8; }
    };
    struct Output {
        [[nodiscard]] std::uint32_t model_token_budget_remaining(std::uint32_t total) const {
            return total;
        }
        [[nodiscard]] std::span<const ninfer::TokenId> pending_control_tokens() const {
            return {};
        }
    };

    [[nodiscard]] bool is_decode_ready() const { return phase == Phase::Decode; }
    [[nodiscard]] bool is_control_ready() const { return false; }
    [[nodiscard]] bool is_prefilling() const { return phase == Phase::Prefill; }
    [[nodiscard]] bool is_replaying() const { return phase == Phase::Replay; }

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
    if (!condition) throw std::runtime_error(message);
}

RequestPtr request(std::uint64_t id, Request::Phase phase = Request::Phase::Waiting) {
    auto value      = std::make_shared<Request>();
    value->id       = id;
    value->phase    = phase;
    value->sequence = id;
    return value;
}

// Runs one prefill cycle as flights: one candidate list per flight, one prefill_executed per
// segment. Returns tokens executed in the cycle. `left` is each lane's remaining prompt.
std::uint32_t run_cycle_as_flights(Scheduler& scheduler, Slots& slots, std::uint32_t lanes,
                                   std::uint32_t chunk, std::array<std::uint32_t, 6>& left,
                                   std::array<std::uint32_t, 6>& last_progress,
                                   std::uint32_t cycle, std::uint32_t& flights) {
    scheduler.begin_prefill_cycle();
    std::uint32_t executed = 0;
    for (;;) {
        ninfer::runtime::PrefillUnitTokens units{};
        for (std::uint32_t lane = 0; lane < lanes; ++lane)
            if (slots[lane]) units[lane] = std::min(chunk, left[lane]);
        const auto flight = scheduler.prefill_candidates(slots, lanes, units);
        if (flight.empty()) break;
        ++flights;
        for (const std::uint32_t lane : flight.lane_span()) {
            const std::uint32_t done = units[lane];
            left[lane] -= done;
            executed += done;
            last_progress[lane] = cycle;
            scheduler.prefill_executed(lane, lanes, done);
            if (left[lane] == 0) slots[lane].reset();
        }
    }
    return executed;
}

// Each segment of a flight is charged as one unit against both budgets.
void test_flight_segments_are_charged_per_unit() {
    Scheduler scheduler;
    scheduler.configure_prefill({.max_units_per_cycle = 3, .max_tokens_per_cycle = 1000});
    Slots slots{};
    for (std::uint32_t lane = 0; lane < 4; ++lane)
        slots[lane] = request(lane + 1, Request::Phase::Prefill);
    ninfer::runtime::PrefillUnitTokens units{};
    units.fill(64);

    const auto flight = scheduler.prefill_candidates(slots, 4, units);
    require(flight.size == 3, "a flight did not use the whole unit allowance");
    require(flight.lanes[0] == 0 && flight.lanes[1] == 1 && flight.lanes[2] == 2,
            "flight members are not in cursor order");
    for (const std::uint32_t lane : flight.lane_span())
        scheduler.prefill_executed(lane, 4, 64);
    require(scheduler.prefill_units_remaining() == 0, "segments were not each counted as a unit");
    require(scheduler.prefill_tokens_remaining() == 1000 - 3 * 64,
            "segment tokens were not charged");
    require(scheduler.prefill_candidates(slots, 4, units).empty(),
            "a unit beyond the cycle allowance was offered");

    // The next cycle leads with the lane that did not fit.
    scheduler.begin_prefill_cycle();
    const auto next = scheduler.prefill_candidates(slots, 4, units);
    require(next.size == 3 && next.lanes[0] == 3 && next.lanes[1] == 0,
            "the lane left out of the previous flight did not lead the next cycle");
}

// k = 2..6 prefilling lanes, flights per cycle: every lane completes, none waits > k cycles,
// and prefill work between two decode rounds respects max(max_tokens_per_cycle, chunk).
void test_flights_do_not_starve_and_gap_is_bounded() {
    constexpr std::uint32_t chunk = 128;
    for (std::uint32_t lanes = 2; lanes <= 6; ++lanes) {
        for (const std::uint32_t budget : {0U, 128U, 256U, 512U}) {
            Scheduler scheduler;
            scheduler.configure_prefill({.max_tokens_per_cycle = budget});
            Slots slots{};
            std::array<std::uint32_t, 6> left{900, 128, 40, 513, 300, 1024};
            std::array<std::uint32_t, 6> last{};
            for (std::uint32_t lane = 0; lane < lanes; ++lane)
                slots[lane] = request(lane + 1, Request::Phase::Prefill);
            std::uint32_t flights = 0;
            for (std::uint32_t cycle = 1; cycle <= 400; ++cycle) {
                const auto executed =
                    run_cycle_as_flights(scheduler, slots, lanes, chunk, left, last, cycle, flights);
                require(executed <= std::max(budget, chunk),
                        "prefill work between decode rounds exceeded the contract bound");
                bool live = false;
                for (std::uint32_t lane = 0; lane < lanes; ++lane) {
                    if (!slots[lane]) continue;
                    live = true;
                    require(cycle - last[lane] <= lanes, "a prefilling lane was starved");
                }
                if (!live) break;
            }
            for (std::uint32_t lane = 0; lane < lanes; ++lane)
                require(!slots[lane] && left[lane] == 0, "a lane did not finish prefill");
        }
    }
}

// Concurrent lanes advance together in a cycle when the budget allows it.
void test_budget_lets_several_lanes_advance_per_cycle() {
    Scheduler scheduler;
    scheduler.configure_prefill({.max_tokens_per_cycle = 512});
    Slots slots{};
    for (std::uint32_t lane = 0; lane < 4; ++lane)
        slots[lane] = request(lane + 1, Request::Phase::Prefill);
    ninfer::runtime::PrefillUnitTokens units{};
    units.fill(128);
    const auto flight = scheduler.prefill_candidates(slots, 4, units);
    require(flight.size >= 2, "ample budget still limited the cycle to one lane");
}

// Several waiting requests are offered in one scan; FIFO head leads and bypass credit holds.
void test_multiple_fresh_candidates_per_scan() {
    Scheduler scheduler;
    std::deque<RequestPtr> fresh;
    for (std::uint64_t id = 1; id <= 5; ++id) fresh.push_back(request(id));
    scheduler.begin_admission_scan();
    const auto candidates = scheduler.fresh_candidates(fresh, 6);
    require(candidates.size() >= 2, "admission was limited to one candidate per scan");
    require(candidates.front() == fresh.front(), "FIFO head did not lead the candidates");

    // Admitting a younger request charges the older head exactly once.
    scheduler.admitted(fresh, *fresh[1], 6);
    require(fresh[0]->admission_bypasses == 1, "bypass credit not charged to the skipped head");
    scheduler.admitted(fresh, *fresh[0], 6);
    require(fresh[0]->admission_bypasses == 1, "admitting the head consumed bypass credit");
}

} // namespace

int main() {
    try {
        test_flight_segments_are_charged_per_unit();
        test_flights_do_not_starve_and_gap_is_bounded();
        test_budget_lets_several_lanes_advance_per_cycle();
        test_multiple_fresh_candidates_per_scan();
        std::cout << "ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
