#pragma once

#include <cstdint>
#include <optional>

namespace ninfer {

enum class RequestPhase : std::uint8_t {
    Waiting,
    Materializing,
    Prefill,
    Decode,
    Control,
    Replay,
    Pausing,
    Paused,
    Finishing,
};

enum class EngineStatus : std::uint8_t { Ready, Stopping, Failed };

enum class GenerationRecoveryRoute : std::uint8_t { None, Direct, Snapshot, Replay };

// Session-owned mathematical frontiers for one restoration. A replay rebuilds
// [begin_frontier, target_frontier); frontier advances only with completed work.
// Completing this range does not release recovery protection: the first new unit must commit.
struct RecoveryProgress {
    GenerationRecoveryRoute route = GenerationRecoveryRoute::None;
    std::uint32_t begin_frontier = 0, frontier = 0, target_frontier = 0;
};

[[nodiscard]] constexpr const char* request_phase_name(RequestPhase phase) noexcept {
    switch (phase) {
    case RequestPhase::Waiting:
        return "waiting";
    case RequestPhase::Materializing:
        return "materializing";
    case RequestPhase::Prefill:
        return "prefill";
    case RequestPhase::Decode:
        return "decode";
    case RequestPhase::Control:
        return "control";
    case RequestPhase::Replay:
        return "replay";
    case RequestPhase::Pausing:
        return "pausing";
    case RequestPhase::Paused:
        return "paused";
    case RequestPhase::Finishing:
        return "finishing";
    }
    return "unknown";
}

[[nodiscard]] constexpr const char* recovery_route_name(GenerationRecoveryRoute route) noexcept {
    switch (route) {
    case GenerationRecoveryRoute::None:
        return "none";
    case GenerationRecoveryRoute::Direct:
        return "direct";
    case GenerationRecoveryRoute::Snapshot:
        return "snapshot";
    case GenerationRecoveryRoute::Replay:
        return "replay";
    }
    return "unknown";
}

struct RequestCounters {
    std::uint64_t submitted = 0, completed = 0, cancelled = 0, failed = 0;
};

// Current request facts at an Engine publication boundary. Times use the coordinator's
// monotonic clock, as does EngineSnapshot::stamp. Completed requests are not retained.
struct RequestSnapshot {
    std::uint64_t id   = 0;
    RequestPhase phase = RequestPhase::Waiting;
    std::optional<std::uint32_t> placement;
    std::uint64_t submitted_ns = 0;
    std::optional<std::uint64_t> admitted_ns, first_token_ns;
    std::uint32_t prompt_tokens = 0, reused_prompt_tokens = 0, processed_prompt_tokens = 0;
    std::uint32_t generated_tokens = 0, injected_control_tokens = 0;
    std::uint32_t logical_context_tokens = 0;
    std::optional<std::uint32_t> output_token_limit;
    std::uint64_t preemptions     = 0;
    std::uint64_t replayed_tokens = 0;
    std::optional<RecoveryProgress> recovery;
    std::optional<std::uint64_t> last_progress_ns;
    bool recovery_pending = false;
    bool resident = false, execution_in_flight = false, capture_pending = false;
};

struct SnapshotOptions {
    bool include_requests = false;
};

struct PlacementUsage {
    std::uint32_t id       = 0;
    std::uint32_t resident = 0, prefilling = 0, decoding = 0, replaying = 0, paused = 0;
    std::uint64_t context_tokens = 0;
};

} // namespace ninfer
