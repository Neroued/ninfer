#pragma once

#include "core/nvtx.h"
#include "core/transfer_work.h"
#include "ninfer/types.h"

#include <atomic>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>

namespace ninfer::runtime {

using ::ninfer::FinishReason;
using ::ninfer::KvCapacityMode;
using ::ninfer::KvCapacityPolicy;
using ::ninfer::OutputChannel;
using ::ninfer::ResolvedSamplingParameters;
using ::ninfer::StopPolicy;
using ::ninfer::StopString;
using ::ninfer::TokenId;

// One Program execution may alternate between Host submission, a blocking Device completion
// wait, and Host post-processing. The three monotonic components are returned to Engine as part of
// the execution capability; serve never infers them from total request time.
struct ExecutionTiming {
    std::uint64_t submit_host_ns = 0;
    std::uint64_t device_wait_ns = 0;
    std::uint64_t post_host_ns   = 0;

    ExecutionTiming& operator+=(ExecutionTiming other) noexcept {
        submit_host_ns += other.submit_host_ns;
        device_wait_ns += other.device_wait_ns;
        post_host_ns += other.post_host_ns;
        return *this;
    }

    [[nodiscard]] std::uint64_t host_ns() const noexcept { return submit_host_ns + post_host_ns; }

    [[nodiscard]] std::uint64_t elapsed_ns() const noexcept { return host_ns() + device_wait_ns; }
};

enum class ExecutionTimingPhase : std::uint8_t {
    Submit,
    Wait,
    Post,
    Paused,
};

// Fixed-cost coarse recorder used only at Program phase boundaries, never in token/page/layer
// loops. It starts in Submit, permits explicit Submit -> Wait -> Post transitions, and can resume
// Submit for a later segment in the same execution unit.
class ExecutionTimingRecorder {
public:
    using Clock = std::chrono::steady_clock;

    explicit ExecutionTimingRecorder(
        ExecutionTimingPhase initial_phase = ExecutionTimingPhase::Submit,
        ExecutionTiming* abandoned_timing  = nullptr) noexcept
        : started_(Clock::now()), phase_(initial_phase), abandoned_timing_(abandoned_timing) {
        open_range();
    }

    ~ExecutionTimingRecorder() noexcept {
        if (finished_) { return; }
        const ExecutionTiming timing = finish();
        if (abandoned_timing_ != nullptr) { *abandoned_timing_ += timing; }
    }

    ExecutionTimingRecorder(const ExecutionTimingRecorder&)            = delete;
    ExecutionTimingRecorder& operator=(const ExecutionTimingRecorder&) = delete;

    void begin_wait() noexcept { transition(ExecutionTimingPhase::Wait); }

    void end_wait() noexcept { transition(ExecutionTimingPhase::Post); }

    void resume_submit() noexcept { transition(ExecutionTimingPhase::Submit); }

    void resume_post() noexcept { transition(ExecutionTimingPhase::Post); }

    void pause() noexcept { transition(ExecutionTimingPhase::Paused); }

    void include(ExecutionTiming timing) noexcept { timing_ += timing; }

    [[nodiscard]] ExecutionTiming finish() noexcept {
        if (finished_) { return timing_; }
        accumulate(Clock::now());
        range_.reset();
        phase_    = ExecutionTimingPhase::Paused;
        finished_ = true;
        return timing_;
    }

private:
    [[nodiscard]] static nvtx::Name range_name(ExecutionTimingPhase phase) noexcept {
        switch (phase) {
        case ExecutionTimingPhase::Submit:
            return nvtx::Name::ProgramSubmit;
        case ExecutionTimingPhase::Wait:
            return nvtx::Name::DeviceWait;
        case ExecutionTimingPhase::Post:
            return nvtx::Name::ProgramPost;
        case ExecutionTimingPhase::Paused:
            break;
        }
        return nvtx::Name::ProgramSubmit;
    }

    void open_range() noexcept {
        if (phase_ != ExecutionTimingPhase::Paused) {
            range_.emplace(range_name(phase_), nvtx::Category::Runtime);
        }
    }

    void transition(ExecutionTimingPhase next) noexcept {
        if (finished_ || next == phase_) { return; }
        const Clock::time_point now = Clock::now();
        accumulate(now);
        range_.reset();
        phase_   = next;
        started_ = Clock::now();
        open_range();
    }

    void accumulate(Clock::time_point now) noexcept {
        const auto count =
            std::chrono::duration_cast<std::chrono::nanoseconds>(now - started_).count();
        const std::uint64_t elapsed = count > 0 ? static_cast<std::uint64_t>(count) : 0;
        switch (phase_) {
        case ExecutionTimingPhase::Submit:
            timing_.submit_host_ns += elapsed;
            break;
        case ExecutionTimingPhase::Wait:
            timing_.device_wait_ns += elapsed;
            break;
        case ExecutionTimingPhase::Post:
            timing_.post_host_ns += elapsed;
            break;
        case ExecutionTimingPhase::Paused:
            break;
        }
    }

    Clock::time_point started_;
    ExecutionTimingPhase phase_ = ExecutionTimingPhase::Submit;
    ExecutionTiming timing_;
    std::optional<nvtx::ScopedRange> range_;
    ExecutionTiming* abandoned_timing_ = nullptr;
    bool finished_                     = false;
};

// Engine has already selected the registered model/mode preset, applied every explicit override,
// and validated these values before constructing the runtime request.
struct ResolvedExecutionOptions {
    ResolvedSamplingParameters sampling;
    std::uint32_t requested_output_tokens = 0;
    bool allow_prefix_reuse               = true;
    ThinkingControlOptions thinking;
};

struct ResolvedRequestOptions {
    ResolvedExecutionOptions execution;
    StopPolicy stop;
    OutputOptions output;
};

enum class ContinuationAction : std::uint8_t {
    Decode,
    ApplyTargetControl,
};

struct OutputDecision {
    std::uint32_t accepted_tokens   = 0;
    FinishReason finish_reason      = FinishReason::None;
    ContinuationAction continuation = ContinuationAction::Decode;

    [[nodiscard]] bool finished() const noexcept { return finish_reason != FinishReason::None; }
};

struct LaneId {
    std::uint32_t value = 0;

    [[nodiscard]] friend constexpr bool operator==(LaneId, LaneId) noexcept  = default;
    [[nodiscard]] friend constexpr auto operator<=>(LaneId, LaneId) noexcept = default;
};

enum class ConsumeStatus : std::uint8_t {
    Consumed,
    InvariantMismatch,
};

enum class CommitDisposition : std::uint8_t {
    Active,
    Finishable,
    CancelledReleased,
};

// The product Engine only needs statistics for rows whose sequence is released by commit.
// Direct diagnostic callers may temporarily request cumulative snapshots for every row.
enum class CommitObservation : std::uint8_t {
    ReleasedRowsOnly,
    AllRows,
};

struct CommitDecision {
    std::uint32_t accepted_tokens = 0;
    bool terminal                 = false;
    bool cancelled                = false;
};

// Exact features for the startup-selected static prefill cost model. They describe only the
// suffix rebuilt after a selected prefix and remain separate from Scheduler service work.
struct PrefillWork {
    std::uint64_t chunks          = 0;
    std::uint64_t tokens          = 0;
    std::uint64_t attention_pairs = 0;
    std::uint64_t vision_items    = 0;
    std::uint64_t vision_patches  = 0;

    [[nodiscard]] friend constexpr bool operator==(PrefillWork, PrefillWork) noexcept = default;
};

// Exact prefill feature definition for a suffix beginning after prefix_tokens. Attention work is
// prefix*suffix + suffix*(suffix+1)/2 and all arithmetic saturates.
[[nodiscard]] inline PrefillWork make_prefill_work(std::uint64_t prefix_tokens,
                                                   std::uint64_t suffix_tokens,
                                                   std::uint64_t vision_items,
                                                   std::uint64_t vision_patches,
                                                   std::uint32_t prefill_chunk) noexcept {
    PrefillWork result;
    result.chunks =
        suffix_tokens == 0 || prefill_chunk == 0 ? 0 : 1U + (suffix_tokens - 1U) / prefill_chunk;
    result.tokens                       = suffix_tokens;
    result.vision_items                 = vision_items;
    result.vision_patches               = vision_patches;
    const unsigned __int128 suffix      = suffix_tokens;
    const unsigned __int128 linear      = static_cast<unsigned __int128>(prefix_tokens) * suffix;
    const unsigned __int128 triangular  = suffix * (suffix + 1U) / 2U;
    constexpr unsigned __int128 maximum = ~static_cast<unsigned __int128>(0);
    const unsigned __int128 attention =
        triangular > maximum - linear ? maximum : linear + triangular;
    result.attention_pairs = attention > std::numeric_limits<std::uint64_t>::max()
                                 ? std::numeric_limits<std::uint64_t>::max()
                                 : static_cast<std::uint64_t>(attention);
    return result;
}

enum class ContextResourceClass : std::uint8_t {
    State,
    MainKV,
    BackendKV,
};

enum class ContextTransferDirection : std::uint8_t {
    DeviceToHost,
    HostToDevice,
    DeviceToDevice,
};

struct ContextTransferObservation {
    ContextResourceClass resource      = ContextResourceClass::State;
    ContextTransferDirection direction = ContextTransferDirection::DeviceToHost;
    std::uint64_t units                = 0; // State images for State; bytes for typed KV.
    std::uint32_t page_count           = 0;
    TransferWork work;
    std::uint64_t elapsed_ns = 0;
};

struct ContextTransferRequirement {
    ContextResourceClass resource      = ContextResourceClass::State;
    ContextTransferDirection direction = ContextTransferDirection::DeviceToHost;
    std::uint64_t units                = 0;
    std::uint32_t page_count           = 0;
    TransferWork work;

    [[nodiscard]] friend constexpr bool operator==(ContextTransferRequirement,
                                                   ContextTransferRequirement) noexcept = default;
};

struct ContextOperationCounts {
    std::uint64_t state_moves            = 0;
    std::uint64_t state_forks            = 0;
    std::uint64_t state_restores         = 0;
    std::uint64_t pressure_spill_pages   = 0;
    std::uint64_t partial_tail_cow_pages = 0;
    std::uint64_t historical_fork_hits   = 0;
};

enum class Readiness : std::uint8_t {
    Ready,
    NeedsTransfer,
    TemporarilyBlocked,
    PermanentlyInfeasible,
};

// Non-owning cancellation observation used while the worker advances a context transaction. The
// request record owns the flag for longer than Program can retain this view.
struct CancellationFlagView {
    const std::atomic<bool>* flag = nullptr;

    [[nodiscard]] bool requested() const noexcept {
        return flag != nullptr && flag->load(std::memory_order_acquire);
    }
};

enum class ContextTransactionStatus : std::uint8_t {
    InProgress,
    Published,
    Aborted,
};

enum class ContextTransactionReserveStatus : std::uint8_t {
    Reserved,
    Aborted,
};

struct ContextTransactionInProgress {};

enum class ContextTransactionKind : std::uint8_t {
    Materialization,
    ActiveCapture,
};

enum class PreflightStatus : std::uint8_t {
    Ready,
    StalePolicyState,
    InvariantFailure,
};

enum class CheckpointKind : std::uint8_t {
    SessionEndpoint,
    TurnClosure,
    ResponseReplay,
    SharedStablePrefix,
    LongAnchor,
};

enum class CheckpointScope : std::uint8_t {
    Private,
    Shared,
};

enum class ReplicaResidency : std::uint8_t {
    DeviceOnly,
    HostOnly,
    Both,
};

enum class RetentionClass : std::uint8_t {
    SharedStable,
    LiveSession,
    RecentPrivate,
    Disposable,
};

enum class LogicalOwnerKind : std::uint8_t {
    PrivateContinuation,
    SharedPrefix,
};

struct LogicalOwnerKey {
    LogicalOwnerKind kind = LogicalOwnerKind::PrivateContinuation;
    std::uint64_t id      = 0;

    [[nodiscard]] friend constexpr bool operator==(LogicalOwnerKey,
                                                   LogicalOwnerKey) noexcept = default;
};

struct CatalogCapability {
    LogicalOwnerKey owner;
    std::uint32_t slot       = std::numeric_limits<std::uint32_t>::max();
    std::uint64_t generation = 0;

    [[nodiscard]] friend constexpr bool operator==(CatalogCapability,
                                                   CatalogCapability) noexcept = default;
};

struct PlanningOwnerId {
    std::uint32_t value = std::numeric_limits<std::uint32_t>::max();

    [[nodiscard]] friend constexpr bool operator==(PlanningOwnerId,
                                                   PlanningOwnerId) noexcept = default;
};

struct PlanningCandidateId {
    std::uint32_t value = std::numeric_limits<std::uint32_t>::max();

    [[nodiscard]] friend constexpr bool operator==(PlanningCandidateId,
                                                   PlanningCandidateId) noexcept = default;
};

struct ProgramResourceRevision {
    std::uint64_t value = 0;

    [[nodiscard]] friend constexpr bool operator==(ProgramResourceRevision,
                                                   ProgramResourceRevision) noexcept = default;
};

// ResourceManager-owned final schedule policy. Program validates and consumes this view while
// sealing the already assessed physical target; ResourcePlan is immutable after that boundary.
struct FinalScheduleIntent {
    std::span<const std::uint32_t> shared_capture_frontiers;
};

enum class PrivateSourceMode : std::uint8_t {
    Retain,
    ConsumeToActive,
};

enum class VictimDisposition : std::uint8_t {
    Retained,
    Evicted,
};

enum class FinishDisposition : std::uint8_t {
    Catalogued,
    Released,
};

struct CheckpointRef {
    CheckpointKind kind    = CheckpointKind::SessionEndpoint;
    std::uint32_t frontier = 0;
    // Singleton checkpoint kinds use zero. LongAnchor uses a nonzero, per-continuation slot.
    std::uint32_t ordinal = 0;

    [[nodiscard]] friend constexpr bool operator==(CheckpointRef, CheckpointRef) noexcept = default;
};

struct Revision {
    std::uint64_t value = 0;

    [[nodiscard]] friend constexpr bool operator==(Revision, Revision) noexcept = default;
};

struct RequestPlanSummary {
    std::uint32_t prompt_tokens           = 0;
    std::uint32_t reusable_prompt_tokens  = 0;
    std::uint32_t requested_output_tokens = 0;
    std::uint32_t effective_output_tokens = 0;
    FinishReason effective_limit_reason   = FinishReason::None;
    PrefixReusePath prefix_reuse_path     = PrefixReusePath::Root;
    std::uint64_t service_work_quanta     = 0;
    bool publish_continuation             = true;
};

enum class MaterializationPhysicalStatus : std::uint8_t {
    Feasible,
    Infeasible,
    StructuralInvalid,
};

inline constexpr std::size_t kContextTransferDirectionCount = 3;
using CoalescedTransferWork = std::array<TransferWork, kContextTransferDirectionCount>;

// Exact, unpriced machine work for one complete materialization projection. Program owns this
// physical fact; the common search runner applies the immutable planning cost model exactly once.
// `optimistic_candidate_transfers` is ordering evidence only and never proves feasibility.
struct MaterializationMachineWork {
    CoalescedTransferWork pressure_transfers;
    CoalescedTransferWork candidate_transfers;
    CoalescedTransferWork optimistic_candidate_transfers;
    PrefillWork remaining_prefill_work;
    std::uint32_t reused_prompt_tokens = 0;

    [[nodiscard]] friend constexpr bool
    operator==(const MaterializationMachineWork&,
               const MaterializationMachineWork&) noexcept = default;
};

// One exact recovery recipe. Program enumerates every supported alternative; pricing policy
// selects the cheapest alternative without changing physical legality or the target graph.
struct CheckpointRecoveryAlternativeWork {
    CoalescedTransferWork transfers;
    PrefillWork prefill;

    [[nodiscard]] friend constexpr bool
    operator==(const CheckpointRecoveryAlternativeWork&,
               const CheckpointRecoveryAlternativeWork&) noexcept = default;
};

struct IdentityMaterializationAssessment {
    MaterializationPhysicalStatus physical_status =
        MaterializationPhysicalStatus::StructuralInvalid;
    PrivateSourceMode source_mode = PrivateSourceMode::ConsumeToActive;
    MaterializationMachineWork machine_work;
    bool pressure_may_change_machine_work = false;
    bool expandable                       = false;
    std::uint64_t projection_work         = 0;
    std::uint64_t assessment_digest       = 0;

    [[nodiscard]] friend constexpr bool
    operator==(const IdentityMaterializationAssessment&,
               const IdentityMaterializationAssessment&) noexcept = default;
};

struct PressureCheckpointRecoveryImpact {
    PlanningOwnerId owner;
    CheckpointRef checkpoint;
    std::span<const CheckpointRecoveryAlternativeWork> target_recovery_work;
    bool survives = true;

    [[nodiscard]] friend constexpr bool
    operator==(const PressureCheckpointRecoveryImpact&,
               const PressureCheckpointRecoveryImpact&) noexcept = default;
};

struct PressureCheckpointOutcome {
    PlanningOwnerId owner;
    CheckpointRef checkpoint;
    bool survives = true;

    [[nodiscard]] friend constexpr bool operator==(PressureCheckpointOutcome,
                                                   PressureCheckpointOutcome) noexcept = default;
};

struct PressureOwnerOutcome {
    PlanningOwnerId owner;
    VictimDisposition disposition     = VictimDisposition::Retained;
    std::uint32_t degradation_units   = 0;
    std::uint32_t dropped_checkpoints = 0;

    [[nodiscard]] friend constexpr bool operator==(const PressureOwnerOutcome&,
                                                   const PressureOwnerOutcome&) noexcept = default;
};

// Cheap, target-neutral ordering evidence for an unassessed pressure target.  This is deliberately
// not a feasibility certificate: only PressureTargetAssessment may admit or seal a target.  The
// Program owns the physical projection and the common planner combines the owner outcomes with its
// retention policy.
struct PressurePhysicalGuidance {
    std::uint32_t unsatisfied_constraints   = 0;
    std::uint32_t estimated_remaining_steps = 0;
    std::uint64_t normalized_residual_q20   = 0;
};

// The spans are borrowed from a PressurePlanningSession scratch generation and remain valid only
// until the next session operation.  The common planner folds them immediately into owning values.
struct PressureTargetGuidance {
    PressurePhysicalGuidance physical;
    MaterializationMachineWork estimated_machine_work;
    std::span<const PressureOwnerOutcome> owner_outcomes;
    PlanningCandidateId candidate;
    std::uint32_t stable_target_ordinal = 0;
    std::uint32_t degradation_units     = 0;
    std::uint32_t dropped_checkpoints   = 0;
};

// The spans are borrowed from a PressurePlanningSession scratch generation and remain valid only
// until the next session mutation. The common planner folds them immediately into owning values.
struct PressureTargetAssessment {
    MaterializationPhysicalStatus physical_status =
        MaterializationPhysicalStatus::StructuralInvalid;
    PrivateSourceMode source_mode = PrivateSourceMode::ConsumeToActive;
    MaterializationMachineWork machine_work;
    std::span<const PressureOwnerOutcome> owner_outcomes;
    std::span<const PressureCheckpointRecoveryImpact> checkpoint_impacts;
    PlanningCandidateId candidate;
    std::uint32_t stable_target_ordinal = 0;
    std::uint32_t degradation_units     = 0;
    std::uint32_t dropped_checkpoints   = 0;
    std::uint64_t projection_work       = 0;
    std::uint64_t assessment_digest     = 0;
    bool expandable                     = false;
    bool root_maximal                   = false;
};

struct BeginSummary {
    std::uint32_t prompt_tokens        = 0;
    std::uint32_t reused_prompt_tokens = 0;
    PrefixReusePath prefix_reuse_path  = PrefixReusePath::Root;

    [[nodiscard]] friend constexpr bool operator==(BeginSummary, BeginSummary) noexcept = default;
};

struct GeneratedRound {
    std::span<const TokenId> tokens;
};

struct BatchedGeneratedRound {
    std::span<const TokenId> tokens;
    std::span<const std::int32_t> row_counts;
    std::uint32_t row_stride = 1;
    ExecutionTiming timing;
};

struct PrefillStepResult {
    BeginSummary summary;
    GeneratedRound round;
    std::uint32_t processed_prompt_tokens = 0;
    bool complete                         = false;
    ExecutionTiming timing;
};

struct RoundBudget {
    std::uint32_t generated_tokens_remaining = 0;
};

// Target-produced affine reservation curve for one Main KV physical-capacity axis. The byte
// values come from complete target physical layout plans, not from a model geometry formula in
// the common runtime.
struct SequenceCapacityCurve {
    std::uint32_t main_page_tokens                   = 0;
    std::uint32_t minimum_main_page_groups           = 0;
    std::uint32_t maximum_main_page_groups           = 0;
    std::size_t minimum_device_reservation_bytes     = 0;
    std::size_t bytes_per_additional_main_page_group = 0;

    [[nodiscard]] std::size_t reservation_bytes(std::uint32_t main_page_groups) const;
    [[nodiscard]] std::uint32_t resolved_tokens(std::uint32_t main_page_groups) const;
};

struct KvCapacityResolution {
    KvCapacityMode mode                              = KvCapacityMode::Explicit;
    std::uint32_t main_page_groups                   = 0;
    std::uint32_t maximum_main_page_groups           = 0;
    std::uint32_t resolved_tokens                    = 0;
    std::size_t minimum_runtime_reservation_bytes    = 0;
    std::size_t bytes_per_additional_main_page_group = 0;
    std::size_t runtime_reservation_bytes            = 0;
    std::size_t available_after_weights_bytes        = 0;
    std::size_t available_after_startup_bytes        = 0;
    std::size_t automatic_headroom_bytes             = 0;
    std::size_t planned_slack_bytes                  = 0;
};

} // namespace ninfer::runtime
