// K1 / F5: the scheduling side of the packed-prefill contract, using only the public Scheduler
// and prefill-chunk API. Segments of one packed execution are the lanes the scheduler admits for
// a cycle; their summed token caps must respect the pack capacity, and a lane that does not fit
// falls back to fewer segments (or to a single, truncated one).
#include "models/qwen3_5/prefill_chunk.h"
#include "runtime/engine/scheduler.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>

namespace {

struct Request {
    using SequenceHandle = std::uint64_t;
    enum class Phase { Waiting, Materializing, Prefill, Replay, Decode, Control, Finished };

    [[nodiscard]] bool is_prefilling() const { return phase == Phase::Prefill; }
    [[nodiscard]] bool is_replaying() const { return phase == Phase::Replay; }

    std::uint64_t id     = 0;
    bool capture_pending = false;
    Phase phase      = Phase::Prefill;
    std::optional<SequenceHandle> sequence;
};

using Scheduler = ninfer::runtime::Scheduler<Request>;
using Slots     = std::array<std::shared_ptr<Request>, ninfer::kMaximumConcurrency>;
using ninfer::runtime::PrefillPacing;
using ninfer::runtime::PrefillRegime;
using ninfer::runtime::PrefillUnitTokens;

void require(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(what);
}

Slots prefilling_lanes(std::uint32_t n) {
    Slots slots{};
    for (std::uint32_t i = 0; i < n; ++i) {
        slots[i]           = std::make_shared<Request>();
        slots[i]->id       = i + 1;
        slots[i]->sequence = i + 1;
    }
    return slots;
}

constexpr std::uint32_t kChunk = 2048;
constexpr std::uint32_t kPack  = 4096;

PrefillPacing pacing() {
    PrefillPacing p;
    p.throughput_tokens_per_cycle = kPack;
    p.throughput_units_per_cycle  = 6;
    p.interleave_ns               = 100'000'000;
    p.interleave_tokens_per_cycle = kChunk;
    p.interleave_cold_tokens      = 512;
    p.unit_granularity            = 128;
    return p;
}

// Segments never sum past the pack capacity; capacity exhaustion drops trailing lanes.
void test_throughput_membership_fits_pack() {
    Scheduler scheduler;
    scheduler.configure_prefill(pacing());
    scheduler.begin_prefill_cycle(false);
    require(scheduler.prefill_regime() == PrefillRegime::Throughput, "regime");
    require(scheduler.prefill_tokens_remaining() == kPack, "throughput token allowance");

    const auto slots = prefilling_lanes(3);
    PrefillUnitTokens limits{};
    limits[0] = limits[1] = limits[2] = kChunk;
    const auto caps                   = scheduler.prefill_unit_caps(limits);
    const auto members                = scheduler.prefill_candidates(slots, 3, caps);
    require(!members.empty() && members.size < 3, "capacity must reduce the segment count");
    std::uint64_t sum = 0;
    for (auto lane : members.lane_span()) {
        require(caps[lane] > 0 && caps[lane] <= limits[lane], "segment cap out of range");
        sum += caps[lane];
    }
    require(sum <= kPack, "segments exceed the pack capacity");
    const auto next = scheduler.next_prefill(slots, 3, caps);
    require(next && *next == members.lane_span().front(), "next_prefill != first member");
}

// A request longer than the pack still gets a nonzero, truncated unit: single-segment fallback.
void test_single_lane_truncated_to_pack() {
    Scheduler scheduler;
    scheduler.configure_prefill(pacing());
    scheduler.begin_prefill_cycle(false);
    const auto slots = prefilling_lanes(1);
    PrefillUnitTokens limits{};
    limits[0]        = 3 * kPack;
    const auto caps  = scheduler.prefill_unit_caps(limits);
    const auto picks = scheduler.prefill_candidates(slots, 1, caps);
    require(picks.size == 1 && caps[0] > 0 && caps[0] <= kPack, "single lane not truncated");
}

// Executed units are charged to the cycle allowance; a drained allowance admits nothing more.
void test_charging_and_regimes() {
    Scheduler scheduler;
    scheduler.configure_prefill(pacing());
    scheduler.begin_prefill_cycle(false);
    scheduler.prefill_executed(0, 3, kChunk, 1'000'000);
    require(scheduler.prefill_tokens_remaining() == kPack - kChunk, "tokens not charged");
    scheduler.prefill_executed(1, 3, kChunk, 1'000'000);
    require(scheduler.prefill_tokens_remaining() == 0, "tokens not drained");
    PrefillUnitTokens limits{};
    limits[2]       = kChunk;
    const auto caps = scheduler.prefill_unit_caps(limits);
    require(caps[2] == 0, "drained allowance still admits a unit");

    scheduler.begin_prefill_cycle(true);
    require(scheduler.prefill_regime() == PrefillRegime::Interleave, "interleave regime");
    require(scheduler.prefill_tokens_remaining() == kChunk, "interleave token allowance");
    require(scheduler.prefill_ns_remaining() == 100'000'000, "interleave ns allowance");
    scheduler.prefill_executed(0, 3, 256, 10'000'000);
    require(scheduler.prefill_ns_remaining() == 90'000'000, "ns not charged");
}

void test_cost_estimate_lifecycle() {
    Scheduler scheduler;
    scheduler.configure_prefill(pacing());
    require(!scheduler.prefill_cost_ns_per_token(0), "estimate before any sample");
    scheduler.prefill_executed(0, 2, 1000, 5'000'000);
    const auto cost = scheduler.prefill_cost_ns_per_token(0);
    require(cost && *cost > 0, "no estimate after a sample");
    scheduler.prefill_lane_released(0);
    require(!scheduler.prefill_cost_ns_per_token(0), "estimate kept after release");
}

// Pack capacity is >= one window, 128-aligned, and never beyond the context.
void test_default_chunk_resolution() {
    using namespace ninfer::models::qwen3_5;
    ninfer::EngineOptions options;
    options.max_context = 262144;
    options.prefill_chunk = 0;
    const auto resident = resolve_prefill_chunk(options, ninfer::models::WeightPlacement::Resident);
    require(resident == kResidentPrefillChunk && resident % 128 == 0 && resident <= 262144,
            "resident default");
    require(resolve_prefill_chunk(options, ninfer::models::WeightPlacement::HostExperts) == 4096,
            "HostExperts default must stay 4096");
    options.prefill_chunk = 1024;
    require(resolve_prefill_chunk(options, ninfer::models::WeightPlacement::Resident) == 1024,
            "explicit value must win");
    options.prefill_chunk = 1000;
    bool threw = false;
    try {
        (void)resolve_prefill_chunk(options, ninfer::models::WeightPlacement::Resident);
    } catch (const std::invalid_argument&) { threw = true; }
    require(threw, "non-multiple of 128 accepted");
    options.prefill_chunk = 0;
    options.max_context   = 512;
    require(resolve_prefill_chunk(options, ninfer::models::WeightPlacement::Resident) <= 512,
            "default exceeds max_context");
}

void test_pack_tokens_resolution() {
    using namespace ninfer::models::qwen3_5;
    for (std::uint32_t chunk : {1024U, 2048U, 4096U}) {
        const auto pack = resolve_prefill_pack_tokens(chunk, 262144);
        require(pack >= chunk && pack % 128 == 0, "pack not >= chunk / 128-aligned");
    }
    require(resolve_prefill_pack_tokens(2048, 2048) == 2048, "pack exceeds the context");
    bool threw = false;
    try {
        (void)resolve_prefill_pack_tokens(1000, 262144);
    } catch (const std::invalid_argument&) { threw = true; }
    require(threw, "misaligned chunk accepted");
}

} // namespace

int main() {
    try {
        test_throughput_membership_fits_pack();
        test_single_lane_truncated_to_pack();
        test_charging_and_regimes();
        test_cost_estimate_lifecycle();
        test_default_chunk_resolution();
        test_pack_tokens_resolution();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
