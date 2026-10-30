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
    std::uint64_t preemptions = 0;
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
