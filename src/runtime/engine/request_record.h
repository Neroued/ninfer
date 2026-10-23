#pragma once

#include "core/nvtx.h"
#include "ninfer/types.h"
#include "runtime/contract/execution.h"
#include "runtime/engine/generation_budget.h"
#include "runtime/engine/context_cache/types.h"
#include "runtime/executor/contract.h"
#include "runtime/observability/execution.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer::runtime {

enum class RequestEngineHostPhase : std::uint8_t {
    Boundary,
    CommitOutput,
    Maintenance,
};

struct RequestHostTiming {
    std::uint64_t queue_wait_ns                     = 0;
    std::uint64_t engine_boundary_exposed_ns        = 0;
    std::uint64_t session_call_exposed_ns           = 0;
    std::uint64_t executor_poll_exposed_ns          = 0;
    std::uint64_t engine_commit_output_exposed_ns   = 0;
    std::uint64_t engine_maintenance_exposed_ns     = 0;
    std::uint64_t scheduler_yield_exposed_ns        = 0;
    std::uint64_t executor_flush_exposed_ns         = 0;
    std::uint64_t decode_host_exposed_ns            = 0;
    std::uint64_t decode_scheduler_yield_exposed_ns = 0;
    std::uint64_t prefill_units                     = 0;
    std::uint64_t decode_rounds                     = 0;
    std::uint64_t control_units                     = 0;

    void expose_engine(RequestEngineHostPhase phase, std::uint64_t elapsed_ns,
                       bool decode_member) noexcept {
        switch (phase) {
        case RequestEngineHostPhase::Boundary:
            engine_boundary_exposed_ns += elapsed_ns;
            break;
        case RequestEngineHostPhase::CommitOutput:
            engine_commit_output_exposed_ns += elapsed_ns;
            break;
        case RequestEngineHostPhase::Maintenance:
            engine_maintenance_exposed_ns += elapsed_ns;
            break;
        }
        if (decode_member) { decode_host_exposed_ns += elapsed_ns; }
    }

    void expose_call(CoordinatorCall call, std::uint64_t elapsed, bool decode_member) noexcept {
        switch (call) {
        case CoordinatorCall::Session:
            session_call_exposed_ns += elapsed;
            break;
        case CoordinatorCall::Poll:
            executor_poll_exposed_ns += elapsed;
            break;
        case CoordinatorCall::Flush:
            executor_flush_exposed_ns += elapsed;
            break;
        case CoordinatorCall::Yield:
            scheduler_yield_exposed_ns += elapsed;
            break;
        }
        if (decode_member) {
            if (call == CoordinatorCall::Yield)
                decode_scheduler_yield_exposed_ns += elapsed;
            else
                decode_host_exposed_ns += elapsed;
        }
    }

    [[nodiscard]] GenerationEngineTiming public_snapshot() const noexcept {
        constexpr double kNanosecondsToSeconds = 1.0e-9;
        return GenerationEngineTiming{
            .queue_wait_seconds = static_cast<double>(queue_wait_ns) * kNanosecondsToSeconds,
            .engine_boundary_exposed_seconds =
                static_cast<double>(engine_boundary_exposed_ns) * kNanosecondsToSeconds,
            .session_call_exposed_seconds =
                static_cast<double>(session_call_exposed_ns) * kNanosecondsToSeconds,
            .executor_poll_exposed_seconds =
                static_cast<double>(executor_poll_exposed_ns) * kNanosecondsToSeconds,
            .engine_commit_output_exposed_seconds =
                static_cast<double>(engine_commit_output_exposed_ns) * kNanosecondsToSeconds,
            .engine_maintenance_exposed_seconds =
                static_cast<double>(engine_maintenance_exposed_ns) * kNanosecondsToSeconds,
            .scheduler_yield_exposed_seconds =
                static_cast<double>(scheduler_yield_exposed_ns) * kNanosecondsToSeconds,
            .executor_flush_exposed_seconds =
                static_cast<double>(executor_flush_exposed_ns) * kNanosecondsToSeconds,
            .decode_host_exposed_seconds =
                static_cast<double>(decode_host_exposed_ns) * kNanosecondsToSeconds,
            .decode_scheduler_yield_exposed_seconds =
                static_cast<double>(decode_scheduler_yield_exposed_ns) * kNanosecondsToSeconds,
            .prefill_units = prefill_units,
            .decode_rounds = decode_rounds,
            .control_units = control_units,
        };
    }
};

enum class EngineRequestState : std::uint8_t {
    Waiting,
    Materializing,
    Prefill,
    Replay,
    Pausing,
    Paused,
    DecodeReady,
    ControlReady,
    ModelFinished,
};

template <class ModelContract>
struct RequestRecord {
    using Clock          = std::chrono::steady_clock;
    using PreparedPrompt = typename ModelContract::PreparedPrompt;
    using OutputSession  = typename ModelContract::OutputSession;
    using BasePlan       = typename ModelContract::RequestBasePlan;
    using SequenceHandle = typename ModelContract::SequenceHandle;
    using ResumeState    = typename ModelContract::ResumeState;
    using StreamEvent = std::variant<GenerationTimingObservation, GenerationFirstTokenObservation,
                                     OutputEvent, std::unique_ptr<GenerationSchedulingObservation>>;

    RequestRecord(std::uint64_t request_identity, std::uint64_t publication_sequence,
                  PreparedPrompt input, OutputSession output_session, PromptSummary summary,
                  double frontend_seconds, ResolvedRequestOptions request_options,
                  OutputConsumerMode output_consumer, GenerationObservationOptions observation,
                  Clock::time_point limit, Clock::time_point submit_time)
        : generation_range(nvtx::Name::Generate, nvtx::Category::Runtime, request_identity),
          id(request_identity), publication_order(publication_sequence), prompt(std::move(input)),
          output(std::move(output_session)), prompt_summary(std::move(summary)),
          prepare_seconds(frontend_seconds), options(std::move(request_options)),
          consumer_mode(output_consumer), observation(std::move(observation)), deadline(limit),
          submitted(submit_time) {}

    RequestRecord(const RequestRecord&)            = delete;
    RequestRecord& operator=(const RequestRecord&) = delete;

    [[nodiscard]] bool is_prefilling() const noexcept {
        return model_state == EngineRequestState::Prefill;
    }

    [[nodiscard]] bool is_replaying() const noexcept {
        return model_state == EngineRequestState::Replay;
    }

    [[nodiscard]] bool is_materializing() const noexcept {
        return model_state == EngineRequestState::Materializing;
    }

    [[nodiscard]] bool is_decode_ready() const noexcept {
        return model_state == EngineRequestState::DecodeReady;
    }

    [[nodiscard]] bool is_control_ready() const noexcept {
        return model_state == EngineRequestState::ControlReady;
    }

    [[nodiscard]] bool is_model_finished() const noexcept {
        return model_state == EngineRequestState::ModelFinished;
    }

    // The process range follows Request ownership across submit, worker, and consumer threads.
    nvtx::ScopedAsyncRange generation_range;
    const std::uint64_t id;
    const std::uint64_t publication_order;
    PreparedPrompt prompt;
    OutputSession output;
    PromptSummary prompt_summary;
    double prepare_seconds = 0.0;
    ResolvedRequestOptions options;
    const OutputConsumerMode consumer_mode;
    const GenerationObservationOptions observation;
    Clock::time_point deadline;
    Clock::time_point submitted;
    std::optional<Clock::time_point> admitted_at;
    std::optional<Clock::time_point> first_token;
    std::optional<Clock::time_point> last_token;
    bool queue_wait_recorded = false;
    std::optional<GenerationBudget> budget;
    std::optional<BeginSummary> admitted_begin;
    std::optional<BeginSummary> begin;
    std::vector<TokenId> generated;
    std::string content;
    std::string reasoning;
    std::optional<LaneId> lane;
    std::optional<ResidencyId> residency;
    std::optional<SequenceHandle> sequence;
    std::atomic<bool> cancelled{false};
    EngineRequestState model_state        = EngineRequestState::Waiting;
    bool capture_pending                  = false;
    EngineRequestState post_capture_state = EngineRequestState::Prefill;
    std::optional<FinishReason> terminal_reason;
    std::exception_ptr terminal_error;
    bool consumed_by_commit = false;

    std::optional<BasePlan> base_plan;
    std::optional<ResumeState> suspended;
    ContinuationOwnerToken continuation_owner = 0;
    std::uint64_t device_to_host_bytes        = 0;
    std::uint64_t host_to_device_bytes        = 0;
    EngineRequestState resume_phase           = EngineRequestState::Prefill;
    std::uint32_t admission_bypasses          = 0;
    std::uint64_t admission_generation        = 0;
    bool admission_observed                   = false;
    GenerationAdmissionStats admission;
    std::optional<Clock::time_point> source_wait_started;
    bool recovery_pending                  = false;
    std::uint64_t committed_decode_tokens  = 0;
    GenerationRecoveryRoute recovery_route = GenerationRecoveryRoute::None;
    std::uint64_t preemption_count         = 0;
    std::uint64_t replay_restores          = 0;
    std::uint64_t snapshot_restores        = 0;
    std::uint64_t replayed_tokens          = 0;
    std::uint64_t paused_ns                = 0;
    std::optional<Clock::time_point> paused_at;
    std::uint32_t computed_prompt_tokens = 0;
    RequestExecutionStatistics executions;
    std::vector<ExecutionExposure> decode_by_batch;
    std::array<DurationCounter, 4> max_rank_gpu{};
    std::optional<ExecutionSample> pending_execution;
    RequestHostTiming host_timing;
    std::uint64_t initial_binding_ns = 0;
    std::vector<std::array<GenerationTransferTiming, 4>> context_transfers;
    std::optional<GenerationFirstOutputTiming> first_output_timing;
    SpeculativeStats speculative_stats;

    std::mutex mutex;
    std::condition_variable cv;
    std::optional<GenerationStart> stream_start;
    std::optional<PromptProgress> stream_progress;
    std::vector<StreamEvent> events;
    GenerationResult result;
    std::exception_ptr error;
    bool response_done     = false;
    bool consumer_released = false;
    bool capacity_released = false;
};

} // namespace ninfer::runtime
