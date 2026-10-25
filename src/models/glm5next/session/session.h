#pragma once

#include "models/common/context/checkpoint_store.h"
#include "models/common/context/directory.h"
#include "models/common/context/recovery.h"
#include "models/common/context/release_inventory.h"
#include "models/glm5next/frontend/prepared_prompt.h"
#include "models/glm5next/session/planning.h"
#include "models/glm5next/work.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/placement.h"
#include "runtime/executor/executor.h"

namespace ninfer::models::glm5next {
class Session;
class ModelDefinition;
struct PreparedContext;

namespace detail {
class SessionImpl;
struct RequestPlan;
struct Suspended;
} // namespace detail

using ContextDemand     = context::ContextDemand;
using ContextAssessment = context::ContextAssessment;
using ContextReleaseSet = context::ContextReleaseSet;

struct ResourceReservation {
    runtime::UnitReservationStatus status = runtime::UnitReservationStatus::ContextUnavailable;
    ContextAssessment assessment;

    explicit operator bool() const noexcept {
        return status == runtime::UnitReservationStatus::Granted;
    }
};

using CheckpointHandle = context::CheckpointHandle;

struct PrefixShortlistKey {
    std::array<std::uint64_t, 2> digests{};
    std::uint32_t frontier = 0, identity_tag = 0;
    friend bool operator==(PrefixShortlistKey, PrefixShortlistKey) noexcept = default;
};

class RequestBasePlan {
public:
    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;
    [[nodiscard]] const models::frontend::PreparedContextCache& context_cache() const noexcept;
    [[nodiscard]] std::optional<PrefixShortlistKey> prefix_shortlist_key(std::uint32_t) const;
private:
    std::shared_ptr<const detail::RequestPlan> plan_;

    explicit RequestBasePlan(std::shared_ptr<const detail::RequestPlan> p) : plan_(std::move(p)) {}
    friend class detail::SessionImpl;
    friend class Session;
};

using SequenceHandle = runtime::SequenceId;

struct CheckpointMetadata {
    std::uint32_t frontier           = 0;
    runtime::CheckpointRole role     = runtime::CheckpointRole::Continuation;
    bool leased                      = false;
    runtime::CheckpointOrigin origin = runtime::CheckpointOrigin::Semantic;
    runtime::PlacementId placement;
};

struct CheckpointSummary : CheckpointMetadata {
    PrefixShortlistKey key;
};

struct SourceCandidate {
    std::optional<CheckpointHandle> checkpoint;
    std::uint32_t reused_tokens = 0;
    runtime::PrefillWork remaining_work;
    std::vector<runtime::ContextTransferRequirement> transfers;
    bool consume_source = false, take_private = false;
    std::vector<CheckpointHandle> private_points, retired_points;
    runtime::PlacementId placement;
    bool local_device = false;
};

struct BindingEvaluation {
    bool source_valid = true, capacity_possible = true;
    ContextAssessment assessment;
};

struct BindingReservation {
    bool reserved = false, source_valid = true, capacity_possible = true;
    ContextAssessment assessment;
    std::vector<CheckpointHandle> retired_points;
    std::optional<CheckpointHandle> consumed_source;

    explicit operator bool() const noexcept { return reserved; }
};

struct ContextDemotion {
    std::vector<CheckpointHandle> sources;
    std::size_t host_bytes = 0;
    ContextReleaseSet released;
};

struct ContextRelease {
    std::vector<CheckpointHandle> sources;
    ContextReleaseSet released;
};

struct PauseOption {
    runtime::PauseAction action;
    ContextDemand demand;
    ContextReleaseSet released;
    std::optional<context::HostWriteRequirements> host_write;
    std::vector<CheckpointHandle> sources;
};

struct ContextReclaimPlan {
    std::vector<ContextDemotion> demotions;
    std::vector<ContextRelease> releases;
    std::shared_ptr<const context::ReleaseInventory<ContextReleaseSet>> inventory;
};

class ResumeState {
public:
    ResumeState(ResumeState&&) noexcept;
    ResumeState& operator=(ResumeState&&) noexcept;
    ~ResumeState();
    [[nodiscard]] bool has_snapshot() const noexcept;
    [[nodiscard]] std::optional<CheckpointHandle> snapshot_handle() const noexcept;
    [[nodiscard]] std::uint32_t frontier() const noexcept;
private:
    explicit ResumeState(std::unique_ptr<detail::Suspended>);
    std::unique_ptr<detail::Suspended> state_;
    friend class detail::SessionImpl;
    friend class Session;
};

enum class ExecutionUnitKind : std::uint8_t { Prefill, Replay, Decode, Control };

struct ExecutionUnit {
    SequenceHandle sequence;
    ExecutionUnitKind kind = ExecutionUnitKind::Decode;
    std::uint32_t tokens   = 0;
};

class PendingBatch {
public:
    PendingBatch(PendingBatch&&) noexcept = default;
    PendingBatch(const PendingBatch&)     = delete;

    [[nodiscard]] std::size_t row_count() const noexcept { return members_.size(); }

    [[nodiscard]] std::span<const TokenId> tokens() const noexcept { return tokens_; }

    [[nodiscard]] std::span<const std::int32_t> row_counts() const noexcept { return counts_; }

    [[nodiscard]] std::uint32_t row_stride() const noexcept { return stride_; }

    [[nodiscard]] bool constraint_failed(std::size_t row) const { return failed_.at(row); }

    [[nodiscard]] const ExecutionSample& execution_sample() const noexcept { return execution_; }
private:
    PendingBatch() = default;
    runtime::ExecutionTicket ticket_;
    std::vector<SequenceHandle> members_;
    std::vector<TokenId> tokens_;
    std::vector<std::int32_t> counts_;
    std::uint32_t stride_ = 1;
    bool decode_          = false;
    std::array<bool, kMaximumConcurrency> failed_{};
    ExecutionSample execution_;
    friend class detail::SessionImpl;
    friend class Session;
};

struct PrefillProgress {
    runtime::BeginSummary summary;
    std::uint32_t processed_prompt_tokens = 0;
    bool complete                         = false;

    std::optional<PendingBatch> pending;
    bool capture_ready = false;
};

struct CommitRowResult {
    runtime::CommitDisposition disposition = runtime::CommitDisposition::Active;
    SpeculativeStats speculative;
};

struct CommitResult {
    std::array<CommitRowResult, kMaximumConcurrency> rows{};
    std::array<bool, kMaximumConcurrency> capture_ready{};
    std::size_t row_count = 0;
};

struct DiscardResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    std::size_t row_count         = 0;
};

struct FinishResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    SpeculativeStats speculative;
    std::optional<CheckpointHandle> checkpoint;
};

struct AbortResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    SpeculativeStats speculative;
};

struct ReplayProgress {
    bool capture_ready             = false;
    std::uint32_t processed_tokens = 0;
    bool complete                  = false;
};

struct CapturePreparation {
    std::uint32_t frontier = 0;
    bool reserved          = false;
    ContextAssessment assessment;
    context::HostWriteRequirements host;
    runtime::CheckpointOrigin origin = runtime::CheckpointOrigin::Semantic;
    std::vector<CheckpointHandle> retired_points;
};
enum class ContextOperationKind : std::uint8_t { Bind, Capture, Demote, Pause, Maintenance };

struct ContextProgress {
    ContextOperationKind kind = ContextOperationKind::Bind;
    bool complete = false, published = false;
    std::optional<SequenceHandle> sequence;
    std::optional<runtime::ResidencyId> residency;
    std::optional<ResumeState> paused;
    bool replaying = false;
    std::vector<CheckpointHandle> private_points, retired_checkpoints, captured_checkpoints;
    std::vector<runtime::ContextTransferObservation> transfers;
    runtime::ContextOperationCounts operations;
    SpeculativeStats request_speculative;
};

using PhysicalUsageSnapshot = context::ContextUsage;

struct ControlComplete {};

struct ScoreComplete {
    std::vector<float> logprobs;
};

struct SessionEvent {
    runtime::ExecutionTicket ticket;
    std::vector<SequenceHandle> members;
    std::variant<PrefillProgress, PendingBatch, ReplayProgress, ControlComplete, ScoreComplete,
                 ContextProgress, runtime::DecisionRequest, runtime::DecisionConsumed,
                 runtime::ExecutionComplete>
        result;
};

class PreparedSession {
public:
    ~PreparedSession();
    PreparedSession(PreparedSession&&) noexcept;
    PreparedSession& operator=(PreparedSession&&) noexcept;
    PreparedSession(const PreparedSession&)            = delete;
    PreparedSession& operator=(const PreparedSession&) = delete;

private:
    explicit PreparedSession(std::unique_ptr<detail::SessionImpl>);
    std::unique_ptr<detail::SessionImpl> impl_;
    friend class Session;
    friend PreparedSession prepare_session(std::shared_ptr<const ModelDefinition>, SessionPlan,
                                           std::unique_ptr<context::ContextDirectory>);
};

[[nodiscard]] PreparedSession prepare_session(std::shared_ptr<const ModelDefinition>, SessionPlan,
                                              std::unique_ptr<context::ContextDirectory>);

[[nodiscard]] PreparedSession prepare_session(std::shared_ptr<const ModelDefinition>,
                                              PreparedContext, std::uint32_t checkpoint_interval);

class Session {
public:
    [[nodiscard]] std::optional<SequenceHandle> prefill_continuation() const;
    void yield_prefill(SequenceHandle);

    using ContextDemand     = glm5next::ContextDemand;
    using ContextAssessment = glm5next::ContextAssessment;
    using ContextReleaseSet = glm5next::ContextReleaseSet;

    [[nodiscard]] ContextAssessment assess(const ContextDemand&,
                                           const ContextReleaseSet& = {}) const;
    [[nodiscard]] bool releases_metadata(const ContextReleaseSet& released,
                                         const ContextAssessment& assessment) const;
    [[nodiscard]] std::uint64_t released_bytes(const ContextReleaseSet&,
                                               const ContextAssessment&) const;
    [[nodiscard]] std::uint64_t useful_release_bytes(const ContextReleaseSet&,
                                                     const ContextAssessment&) const;
    [[nodiscard]] runtime::ContextStart start_maintenance(const ContextAssessment&);

    Session(PreparedSession&&, runtime::Executor<ExecutionTypes>&);
    ~Session();
    Session(const Session&)            = delete;
    Session& operator=(const Session&) = delete;

    [[nodiscard]] std::span<const runtime::KvPoolDescriptor> kv_pools() const noexcept;
    [[nodiscard]] RequestBasePlan plan_request(PreparedPrompt&&,
                                               const runtime::ResolvedExecutionOptions&);
    [[nodiscard]] std::span<const runtime::PlacementId> placements(const RequestBasePlan&) const;
    [[nodiscard]] runtime::PlacementId placement(SequenceHandle) const;
    [[nodiscard]] std::uint64_t load_tokens(SequenceHandle) const;
    [[nodiscard]] std::optional<SourceCandidate>
    inspect_source(const RequestBasePlan&, std::optional<CheckpointHandle>, bool consume = false,
                   std::span<const CheckpointHandle> carried = {},
                   std::span<const CheckpointHandle> retired = {}, runtime::PlacementId target = {},
                   const ResumeState* resume = nullptr) const;
    [[nodiscard]] PrefixShortlistKey checkpoint_key(CheckpointHandle, std::uint32_t) const;
    [[nodiscard]] CheckpointSummary checkpoint_summary(CheckpointHandle) const;
    [[nodiscard]] CheckpointMetadata checkpoint_metadata(CheckpointHandle) const;
    [[nodiscard]] bool checkpoint_matches(CheckpointHandle, const RequestBasePlan&) const;
    [[nodiscard]] std::uint32_t
    checkpoint_recovery_frontier(CheckpointHandle, const RequestBasePlan&, std::uint32_t) const;
    [[nodiscard]] context::RecoveryEvaluation<>
        evaluate_recovery(std::span<const CheckpointHandle>) const;
    [[nodiscard]] std::size_t checkpoint_capacity() const noexcept;
    [[nodiscard]] std::vector<runtime::ContextTransferRequirement>
        checkpoint_restore_requirements(CheckpointHandle) const;
    [[nodiscard]] bool valid_checkpoint(CheckpointHandle) const noexcept;
    [[nodiscard]] bool release_checkpoint(CheckpointHandle) noexcept;
    [[nodiscard]] bool revoke_snapshot(ResumeState&) noexcept;
    [[nodiscard]] std::size_t host_bytes_released(std::span<const CheckpointHandle>) const;
    using HostRequirement = context::HostWriteRequirements;
    [[nodiscard]] std::vector<PauseOption> pause_options(SequenceHandle) const;
    [[nodiscard]] bool pause_contributes(SequenceHandle, const PauseOption&,
                                         const ContextAssessment&) const;
    [[nodiscard]] bool prepare_pause(SequenceHandle, runtime::PauseAction,
                                     std::span<const CheckpointHandle> host_victims = {});
    [[nodiscard]] HostRequirement demotion_host_requirements(const ContextDemotion&) const;
    // Combine disjoint quotes from one synchronous reclaim plan; start_demote revalidates them.
    [[nodiscard]] ContextDemotion combine_demotions(std::span<const ContextDemotion>) const;
    [[nodiscard]] std::size_t host_write_bytes(const HostRequirement&) const;
    [[nodiscard]] ContextDemand host_demand(const HostRequirement&) const;
    [[nodiscard]] bool can_place_host(const HostRequirement&, const ContextReleaseSet& = {}) const;
    [[nodiscard]] std::size_t release_redundant_host(std::span<const CheckpointHandle>,
                                                     std::optional<SequenceHandle> = {});
    [[nodiscard]] ResourceReservation reserve_units(std::span<const ExecutionUnit>);
    [[nodiscard]] bool reclaim_capture_reservation(const ContextAssessment&);
    void release_units(std::span<const SequenceHandle>) noexcept;
    [[nodiscard]] BindingEvaluation assess_binding(const RequestBasePlan&, runtime::LaneId,
                                                   const SourceCandidate&,
                                                   const ResumeState*   = nullptr,
                                                   ExecutionUnitKind    = ExecutionUnitKind::Decode,
                                                   std::uint32_t tokens = 1) const;
    [[nodiscard]] BindingReservation start_binding(const RequestBasePlan&, runtime::LaneId,
                                                   const SourceCandidate&, ResumeState* = nullptr,
                                                   ExecutionUnitKind    = ExecutionUnitKind::Decode,
                                                   std::uint32_t tokens = 1);
    [[nodiscard]] bool start_capture(SequenceHandle);
    [[nodiscard]] bool capture_is_input(SequenceHandle) const;
    [[nodiscard]] bool capture_is_shared(SequenceHandle) const;
    [[nodiscard]] std::vector<std::uint32_t> capture_frontiers(SequenceHandle) const;
    [[nodiscard]] std::optional<CapturePreparation>
    prepare_capture(SequenceHandle, std::span<const CheckpointHandle> host_victims = {});
    void skip_capture(SequenceHandle,
                      runtime::CaptureSkip reason = runtime::CaptureSkip::Unavailable);
    [[nodiscard]] ContextReclaimPlan plan_reclaim(std::span<const CheckpointHandle>,
                                                  std::span<const CheckpointHandle>,
                                                  const ContextDemand&) const;
    [[nodiscard]] context::ReleasePlan<ContextRelease, ContextReleaseSet>
    plan_releases(std::span<const CheckpointHandle>, std::span<const CheckpointHandle>,
                  const ContextDemand&) const;
    [[nodiscard]] runtime::ContextStart
    start_demote(const ContextDemotion&, std::span<const CheckpointHandle> host_victims = {});
    [[nodiscard]] bool start_pause(SequenceHandle, runtime::PauseAction);
    [[nodiscard]] bool has_context_operation() const noexcept;
    [[nodiscard]] bool context_blocks(SequenceHandle) const noexcept;
    [[nodiscard]] bool recovery_pending(SequenceHandle) const noexcept;
    [[nodiscard]] bool execution_pending(SequenceHandle) const noexcept;
    void cancel_context();

    [[nodiscard]] std::optional<runtime::ExecutionTicket> start_prefill(SequenceHandle,
                                                                        bool constrained = false);
    [[nodiscard]] std::optional<runtime::ExecutionTicket> start_replay(SequenceHandle);
    [[nodiscard]] std::optional<runtime::ExecutionTicket>
    start_decode(std::span<const SequenceHandle>, std::span<const runtime::RoundBudget>,
                 std::span<const bool> constrained = {});
    [[nodiscard]] std::optional<runtime::ExecutionTicket>
    start_control(std::span<const SequenceHandle>, std::span<const TokenId>, std::uint32_t stride,
                  std::span<const std::optional<std::uint32_t>> splits);
    [[nodiscard]] std::optional<runtime::ExecutionTicket> start_score(PreparedPrompt&&,
                                                                      std::uint32_t first_target);
    [[nodiscard]] std::optional<SessionEvent> poll();
    void supply(runtime::ExecutionTicket, runtime::DecisionInput&&);
    [[nodiscard]] CommitResult commit(PendingBatch&&, std::span<const runtime::CommitDecision>);
    [[nodiscard]] DiscardResult discard(PendingBatch&&) noexcept;
    [[nodiscard]] FinishResult finish(SequenceHandle);
    [[nodiscard]] AbortResult abort(SequenceHandle);
    void fail_all_cleanup() noexcept;
    [[nodiscard]] PhysicalUsageSnapshot physical_usage() const noexcept;
    void reset_measurement_peaks();

private:
    std::unique_ptr<detail::SessionImpl> impl_;
};

} // namespace ninfer::models::glm5next
