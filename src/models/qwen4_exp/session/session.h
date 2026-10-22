#pragma once

#include "models/common/context/checkpoint_store.h"
#include "models/common/context/directory.h"
#include "models/common/context/recovery.h"
#include "models/common/context/release_inventory.h"
#include "models/common/qwen/frontend/prepared_prompt.h"
#include "models/qwen4_exp/session/planning.h"
#include "models/qwen4_exp/work.h"
#include "runtime/executor/executor.h"
#include "runtime/contract/execution.h"

namespace ninfer::models::qwen4_exp {
class Session;
class ModelDefinition;

namespace detail {
class SessionImpl;
struct RequestPlan;
struct Suspended;
struct Demotion;
struct DemotionBatch;
struct Release;
struct Reclaim;
} // namespace detail

using ContextResources    = runtime::ContextResources<2>;
using ResourceReservation = runtime::ResourceReservation<ContextResources>;
using CheckpointHandle    = context::CheckpointHandle;

struct PrefixShortlistKey {
    std::array<std::uint64_t, 2> digests{};
    std::uint32_t frontier = 0, identity_tag = 0;
    friend bool operator==(PrefixShortlistKey, PrefixShortlistKey) noexcept = default;
};

class RequestBasePlan {
public:
    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;
    [[nodiscard]] const qwen::PreparedContextCache& context_cache() const noexcept;
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
};

struct CheckpointSummary : CheckpointMetadata {
    PrefixShortlistKey key;
    ContextResources evictable_resources;
};

struct SourceCandidate {
    std::optional<CheckpointHandle> checkpoint;
    std::uint32_t reused_tokens = 0;
    runtime::PrefillWork remaining_work;
    std::vector<runtime::ContextTransferRequirement> transfers;
    bool consume_source = false, take_private = false;
    std::vector<CheckpointHandle> private_points, retired_points;
};

struct BindingReservation {
    bool reserved = false, source_valid = true, capacity_possible = true;
    ContextResources shortage;
    std::vector<CheckpointHandle> retired_points;
    std::optional<CheckpointHandle> consumed_source;

    explicit operator bool() const noexcept { return reserved; }
};

struct ContextDemotion {
    std::vector<CheckpointHandle> sources;
    std::size_t host_bytes = 0;
    ContextResources released;
    std::shared_ptr<const detail::Demotion> impl;
};

struct ContextRelease {
    std::vector<CheckpointHandle> sources;
    ContextResources released;
    std::shared_ptr<const detail::Release> impl;
};

class ContextDemotionBatch {
public:
    [[nodiscard]] bool append(const ContextDemotion&);
    [[nodiscard]] bool covers(const ContextRelease&) const;
    [[nodiscard]] std::optional<ContextDemotion> finish() const;
private:
    std::shared_ptr<detail::DemotionBatch> plan_;

    explicit ContextDemotionBatch(std::shared_ptr<detail::DemotionBatch> p) : plan_(std::move(p)) {}
    friend struct ContextReclaimPlan;
};

struct ContextReclaimPlan {
    std::vector<ContextDemotion> demotions;
    std::vector<ContextRelease> releases;
    [[nodiscard]] ContextDemotionBatch begin_kv_batch(ContextResources) const;
    std::shared_ptr<const detail::Reclaim> impl;
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
    ContextResources shortage;
    context::HostWriteRequirements host;
    runtime::CheckpointOrigin origin = runtime::CheckpointOrigin::Semantic;
};
enum class ContextOperationKind : std::uint8_t { Bind, Capture, Demote, Pause };

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

struct PhysicalUsageSnapshot {
    ContextResources occupied, capacity;
    std::size_t host_reserved_bytes = 0, host_peak_occupied_bytes = 0;
    std::uint32_t host_state_slots = 0;
    std::size_t host_kv_bytes      = 0;
};

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
    friend PreparedSession prepare_session(std::shared_ptr<const ModelDefinition>, SessionPlan);
};

[[nodiscard]] PreparedSession prepare_session(std::shared_ptr<const ModelDefinition>, SessionPlan);

class Session {
public:
    using ContextResources = qwen4_exp::ContextResources;

    Session(PreparedSession&&, runtime::Executor<ExecutionTypes>&);
    ~Session();
    Session(const Session&)            = delete;
    Session& operator=(const Session&) = delete;

    [[nodiscard]] std::span<const runtime::KvPoolDescriptor> kv_pools() const noexcept;
    [[nodiscard]] RequestBasePlan plan_request(qwen::PreparedPrompt&&,
                                               const runtime::ResolvedExecutionOptions&);
    [[nodiscard]] std::optional<SourceCandidate>
    inspect_source(const RequestBasePlan&, std::optional<CheckpointHandle>, bool consume = false,
                   std::span<const CheckpointHandle> carried = {},
                   std::span<const CheckpointHandle> retired = {}) const;
    [[nodiscard]] PrefixShortlistKey checkpoint_key(CheckpointHandle, std::uint32_t) const;
    [[nodiscard]] ContextResources checkpoint_footprint(std::span<const CheckpointHandle>) const;
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
    [[nodiscard]] ContextResources snapshot_resources(const ResumeState&) const;
    [[nodiscard]] std::size_t host_bytes_released(std::span<const CheckpointHandle>) const;
    using HostRequirement = context::HostWriteRequirements;
    [[nodiscard]] std::optional<HostRequirement> pause_host_requirements(SequenceHandle) const;
    [[nodiscard]] bool prepare_pause(SequenceHandle, bool save_snapshot,
                                     std::span<const CheckpointHandle> host_victims = {});
    [[nodiscard]] HostRequirement demotion_host_requirements(const ContextDemotion&) const;
    [[nodiscard]] std::size_t host_write_bytes(const HostRequirement&) const;
    [[nodiscard]] bool can_place_host(const HostRequirement&) const;
    [[nodiscard]] bool can_place_host(const HostRequirement&,
                                      const context::ReleasePlan<ContextRelease, ContextResources>&,
                                      std::span<const std::uint32_t> freed) const;
    [[nodiscard]] std::size_t release_redundant_host(std::span<const CheckpointHandle>,
                                                     std::optional<SequenceHandle> = {});
    [[nodiscard]] ResourceReservation reserve_units(std::span<const ExecutionUnit>);
    [[nodiscard]] bool reclaim_capture_reservation(ContextResources);
    void release_units(std::span<const SequenceHandle>) noexcept;
    [[nodiscard]] BindingReservation start_binding(const RequestBasePlan&, runtime::LaneId,
                                                   const SourceCandidate&, ResumeState* = nullptr,
                                                   ExecutionUnitKind    = ExecutionUnitKind::Decode,
                                                   std::uint32_t tokens = 1);
    [[nodiscard]] bool start_capture(SequenceHandle);
    [[nodiscard]] bool capture_is_input(SequenceHandle) const;
    [[nodiscard]] bool capture_is_shared(SequenceHandle) const;
    [[nodiscard]] std::vector<std::uint32_t> capture_frontiers(SequenceHandle) const;
    [[nodiscard]] std::optional<CapturePreparation> prepare_capture(SequenceHandle);
    void skip_capture(SequenceHandle,
                      runtime::CaptureSkip reason = runtime::CaptureSkip::Unavailable);
    [[nodiscard]] ContextReclaimPlan plan_reclaim(std::span<const CheckpointHandle>,
                                                  std::span<const CheckpointHandle>,
                                                  ContextResources) const;
    [[nodiscard]] context::ReleasePlan<ContextRelease, ContextResources>
        plan_releases(std::span<const CheckpointHandle>, std::span<const CheckpointHandle>,
                      ContextResources) const;
    [[nodiscard]] runtime::ContextStart
    start_demote(const ContextDemotion&, std::span<const CheckpointHandle> host_victims = {});
    [[nodiscard]] bool start_pause(SequenceHandle, bool save);
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
    [[nodiscard]] std::optional<runtime::ExecutionTicket> start_score(qwen::PreparedPrompt&&,
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

} // namespace ninfer::models::qwen4_exp
