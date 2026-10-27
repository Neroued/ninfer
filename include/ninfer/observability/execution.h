#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ninfer {

enum class ExecutionKind : std::uint8_t { Prefill, Decode, Replay, Control, Score, Context, Count };
// Main-model traversal. Mixed denotes segmented graphs with Host execution handoffs.
enum class ExecutionPath : std::uint8_t { Eager, Graph, Mixed };

struct ExecutionWork {
    ExecutionKind kind                = ExecutionKind::Context;
    std::uint32_t rows                = 0;
    std::uint64_t input_tokens        = 0;
    std::uint64_t scheduled_positions = 0;
};

struct ExecutionLatency {
    std::optional<std::uint64_t> result_ns;
    std::optional<std::uint64_t> resolution_ns;
    std::optional<std::uint64_t> retirement_ns;
    std::uint64_t total_ns = 0;
};

struct GpuExecutionObservation {
    std::optional<std::uint64_t> result_stream_ns;
    std::optional<std::uint64_t> commit_stream_ns;
    std::optional<std::uint64_t> prefill_stream_ns;
    std::optional<std::uint64_t> vision_stream_ns;

    void add_prefill(std::uint64_t ns) noexcept {
        prefill_stream_ns = prefill_stream_ns.value_or(0) + ns;
    }

    void add_vision(std::uint64_t ns) noexcept {
        vision_stream_ns = vision_stream_ns.value_or(0) + ns;
    }
};

struct ExecutionSample {
    ExecutionWork work;
    ExecutionLatency latency;
    GpuExecutionObservation max_rank_gpu;
};

struct DurationCounter {
    std::uint64_t count = 0, total_ns = 0, maximum_ns = 0;

    void observe(std::uint64_t elapsed) noexcept {
        ++count;
        total_ns += elapsed;
        if (elapsed > maximum_ns) maximum_ns = elapsed;
    }
};

struct ExecutionExposure {
    DurationCounter result, resolution, retirement, total;

    void observe(const ExecutionLatency& latency) noexcept {
        if (latency.result_ns) result.observe(*latency.result_ns);
        if (latency.resolution_ns) resolution.observe(*latency.resolution_ns);
        if (latency.retirement_ns) retirement.observe(*latency.retirement_ns);
        total.observe(latency.total_ns);
    }
};

struct ExecutionCounters {
    std::uint64_t units = 0, rows = 0, input_tokens = 0, scheduled_positions = 0;
    ExecutionExposure exposure;
};

using ExecutionStatistics =
    std::array<ExecutionCounters, static_cast<std::size_t>(ExecutionKind::Count)>;

using RequestExecutionStatistics =
    std::array<ExecutionExposure, static_cast<std::size_t>(ExecutionKind::Count)>;

struct RankHostStatistics {
    std::uint64_t submit_ns = 0, progress_ns = 0, supply_ns = 0, commit_ns = 0, discard_ns = 0;
};

// Payload is the count passed to each collective, in bytes, not estimated network traffic.
// AllReduce, Broadcast, AllGather, ReduceScatter, Send, Receive.
struct CommunicationWork {
    std::array<std::uint64_t, 6> calls{}, payload_bytes{};

    void add(const CommunicationWork& other) noexcept {
        for (std::size_t i = 0; i < calls.size(); ++i) {
            calls[i] += other.calls[i];
            payload_bytes[i] += other.payload_bytes[i];
        }
    }
};

// NCCL's tracked GPU allocations; Host buffers and driver storage are not included.
struct CommunicationMemory {
    std::uint64_t tracked_device_bytes = 0, persistent_device_bytes = 0;
    std::uint64_t suspendable_device_bytes = 0;
};

struct CommunicationStatistics {
    std::uint32_t group = 0, size = 1;
    CommunicationWork eager, graph;
    std::optional<CommunicationMemory> startup_memory;
};

// Fixed-size completion facts. Host counters are cumulative on the rank's execution thread;
// GPU intervals belong to this unit. Logical request tokens are counted by the Engine once.
struct RankExecutionObservation {
    ExecutionPath path = ExecutionPath::Eager;
    RankHostStatistics host;
    GpuExecutionObservation gpu;
    CommunicationWork communication_eager, communication_graph;
    std::uint64_t workspace_peak_bytes = 0, measurement_peak_bytes = 0;
    std::uint64_t expert_upload_bytes = 0, ple_upload_bytes = 0;
};

enum class SpeculativeBackend : std::uint8_t {
    None,
    Mtp,
    DFlash,
    DFlash2,
};

// Request-stage wall timings retained for end-to-end latency/rate reporting. Prefill/decode are
// accepted-to-retired execution exposure; total also includes queueing and other request lifetime.
// Vision is an overlapping stream interval, with coverage in max_rank_gpu. GenerationEngineTiming
// below is the direct, mutually-exclusive Host observation contract.
struct GenerationTimings {
    double prepare_seconds     = 0.0;
    double first_token_seconds = 0.0;
    double vision_seconds      = 0.0;
    double prefill_seconds     = 0.0;
    double decode_seconds      = 0.0;
    double replay_seconds = 0.0, control_seconds = 0.0;
    // Prompt wall time begins at the successful initial binding attempt and ends at the first
    // accepted output token. Generation spans the first through last accepted output token and
    // therefore has N-1 token intervals.
    double prompt_wall_seconds     = 0.0;
    double generation_wall_seconds = 0.0;
    double total_seconds           = 0.0;
};

// Wall elapsed time directly observed in Engine-owned regions. "Exposed" values are latency
// exposure: every active request delayed by one compact-batch unit observes that unit's full
// elapsed time, so values from concurrent requests must not be summed. Scheduler yield/relock is
// reported separately from coordinator calls.
struct GenerationEngineTiming {
    double queue_wait_seconds                     = 0.0;
    double engine_boundary_exposed_seconds        = 0.0;
    double session_call_exposed_seconds           = 0.0;
    double executor_poll_exposed_seconds          = 0.0;
    double engine_commit_output_exposed_seconds   = 0.0;
    double engine_maintenance_exposed_seconds     = 0.0;
    double scheduler_yield_exposed_seconds        = 0.0;
    double executor_flush_exposed_seconds         = 0.0;
    double decode_host_exposed_seconds            = 0.0;
    double decode_scheduler_yield_exposed_seconds = 0.0;
    std::uint64_t prefill_units                   = 0;
    std::uint64_t decode_rounds                   = 0;
    std::uint64_t control_units                   = 0;
};

// Request-owned scheduling observations. Restore counters count completed restorations;
// replayed_tokens counts tokens actually recomputed, including previously generated output;
// these tokens are separate from initial prompt prefill and delivered output. paused_ns covers
// pause preparation, waiting and binding restoration, excluding replay computation.
// Transfer bytes count request-owned payload submitted for binding, capture,
// pause and restore; background cache maintenance remains in the Engine-wide counters.
struct GenerationSchedulingStats {
    std::uint64_t preemptions          = 0;
    std::uint64_t snapshot_restores    = 0;
    std::uint64_t replay_restores      = 0;
    std::uint64_t replayed_tokens      = 0;
    std::uint64_t paused_ns            = 0;
    std::uint64_t device_to_host_bytes = 0;
    std::uint64_t host_to_device_bytes = 0;
};

struct GenerationTransferTiming {
    std::uint64_t bytes = 0;
    // Cumulative copy time across participating ranks, not an operation's wall latency.
    // Peer payload and operation counts include only the sending endpoint.
    double seconds                                          = 0.0;
    std::uint64_t images                                    = 0;
    std::uint64_t page_groups                               = 0;
    std::uint64_t copy_operations                           = 0;
    friend bool operator==(const GenerationTransferTiming&,
                           const GenerationTransferTiming&) = default;
};

// Directions are D2H, H2D, local D2D, H2H and peer D2D.
using ContextTransferSeries = std::array<GenerationTransferTiming, 5>;

// KV entries identify logical history kinds.
struct ContextTransferStats {
    ContextTransferSeries state{};
    std::vector<ContextTransferSeries> kv;
    friend bool operator==(const ContextTransferStats&, const ContextTransferStats&) = default;
};

struct ContextKvPoolUsage {
    std::uint16_t id = 0;
    std::string label;
    std::uint32_t occupied_page_groups                                           = 0;
    friend bool operator==(const ContextKvPoolUsage&, const ContextKvPoolUsage&) = default;
};

// Frozen before the first nonempty text or tool-argument delta; call metadata does not count.
// This differs from the first accepted model token and the client's first HTTP output. Elapsed
// starts at Engine submission, excluding Frontend preparation. Initial queue, initial binding,
// paused time and the remaining resident interval partition this Engine wall time.
struct GenerationFirstOutputTiming {
    double elapsed_seconds         = 0.0;
    double initial_binding_seconds = 0.0;
    GenerationEngineTiming engine;
    ExecutionExposure prefill;
    ExecutionExposure replay;
    std::optional<ExecutionSample> pending_execution;
    GenerationSchedulingStats scheduling;
    std::uint32_t computed_prefill_tokens = 0;
    // Completed request-owned transfers only; background reclamation remains Engine-wide.
    ContextTransferStats context_transfers;
};

struct SpeculativeStats {
    SpeculativeBackend backend           = SpeculativeBackend::None;
    bool enabled                         = false;
    std::uint32_t draft_window           = 0;
    std::uint64_t rounds                 = 0;
    std::uint64_t drafted_tokens         = 0;
    std::uint64_t accepted_tokens        = 0;
    std::uint64_t committed_draft_tokens = 0;
    std::uint64_t fallback_steps         = 0;
    std::vector<std::uint64_t> accepted_per_position;
};

struct ThinkingBudgetStats {
    std::optional<std::uint32_t> configured_budget;
    // Model-origin tokens accepted while capped thinking remained open.
    std::uint32_t model_thinking_tokens = 0;
    // Complete tokenizer-derived target-control suffix committed by Engine.
    std::uint32_t injected_tokens = 0;
    bool applied                  = false;
};

enum class ConstraintCacheAccess : std::uint8_t { Hit, Built, Waited };
enum class ConstraintOutputBranch : std::uint8_t { Undecided, Content, Tools };

// State describes the committed output language, including an assistant continuation prefix.
// Work includes speculative lookahead that was subsequently rolled back. Times are subintervals
// of existing request timings, not extra latency to add to them.
struct ConstraintObservation {
    ConstraintOutputBranch branch   = ConstraintOutputBranch::Undecided;
    bool complete                   = false;
    bool terminated                 = false;
    ConstraintCacheAccess cache     = ConstraintCacheAccess::Hit;
    bool timings_collected          = false;
    double prepare_seconds          = 0.0;
    double mask_seconds             = 0.0;
    double matcher_seconds          = 0.0;
    std::uint64_t mask_positions    = 0;
    std::uint64_t mask_upload_bytes = 0;
};

enum class PrefixReusePath : std::uint8_t {
    Root,
    Checkpoint,
};

enum class AdmissionFallbackReason : std::uint8_t {
    None,
    SourceInvalid,
    SourceRevoked,
    CostChanged,
    CapacityLimit,
    IsolatedCapacity,
};

struct GenerationAdmissionStats {
    std::uint32_t preferred_reused_tokens = 0;
    // Subset of initial queue wait, not additional TTFT. Includes waiting for a lane
    // after a useful source has been selected.
    double source_wait_seconds              = 0.0;
    std::uint32_t revoked_checkpoints       = 0;
    AdmissionFallbackReason fallback_reason = AdmissionFallbackReason::None;
};

} // namespace ninfer
