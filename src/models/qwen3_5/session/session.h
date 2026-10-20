#pragma once

#include "models/common/context/checkpoint_store.h"
#include "models/common/context/recovery.h"
#include "models/common/context/release_inventory.h"
#include "models/common/qwen/frontend/prepared_prompt.h"
#include "models/qwen3_5/definition.h"
#include "models/qwen3_5/planning.h"
#include "models/qwen3_5/work.h"
#include "runtime/executor/executor.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <variant>
#include <vector>

namespace ninfer::models::qwen3_5 {

using SequenceHandle = runtime::SequenceId;

namespace detail {
struct RequestBasePlanImpl;
struct ResumeStateImpl;
struct DemotionPlan;
struct ReleasePlan;
struct ReclaimPlan;
struct DemotionBatch;
class SessionImpl;
} // namespace detail

class PreparedSession {
public:
    PreparedSession(PreparedSession&&) noexcept;
    PreparedSession& operator=(PreparedSession&&) noexcept;
    ~PreparedSession();
    PreparedSession(const PreparedSession&)            = delete;
    PreparedSession& operator=(const PreparedSession&) = delete;

private:
    explicit PreparedSession(std::unique_ptr<detail::SessionImpl>);
    std::unique_ptr<detail::SessionImpl> impl_;
    friend class Session;
    friend PreparedSession prepare_session(std::shared_ptr<const ModelDefinition>, SessionPlan);
};

[[nodiscard]] PreparedSession prepare_session(std::shared_ptr<const ModelDefinition>, SessionPlan);

struct PrefixShortlistKey {
    std::array<std::uint64_t, 2> digests{};
    std::uint32_t frontier                                                  = 0;
    std::uint32_t identity_tag                                              = 0;
    friend bool operator==(PrefixShortlistKey, PrefixShortlistKey) noexcept = default;
};

class RequestBasePlan {
public:
    RequestBasePlan(RequestBasePlan&&) noexcept;
    RequestBasePlan& operator=(RequestBasePlan&&) noexcept;
    ~RequestBasePlan();

    RequestBasePlan(const RequestBasePlan&)            = delete;
    RequestBasePlan& operator=(const RequestBasePlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;
    [[nodiscard]] const qwen::PreparedContextCache& context_cache() const noexcept;
    [[nodiscard]] std::optional<PrefixShortlistKey>
    prefix_shortlist_key(std::uint32_t frontier) const noexcept;

private:
    explicit RequestBasePlan(std::shared_ptr<detail::RequestBasePlanImpl> impl) noexcept;
    std::shared_ptr<detail::RequestBasePlanImpl> impl_;

    friend class detail::SessionImpl;
};

using CheckpointHandle = context::CheckpointHandle;

struct CheckpointMetadata {
    std::uint32_t frontier           = 0;
    runtime::CheckpointRole role     = runtime::CheckpointRole::Continuation;
    bool leased                      = false;
    runtime::CheckpointOrigin origin = runtime::CheckpointOrigin::Semantic;
};

struct CheckpointSummary {
    PrefixShortlistKey key;
    std::uint32_t frontier       = 0;
    runtime::CheckpointRole role = runtime::CheckpointRole::Continuation;
    bool leased                  = false;
    // Resources relevant to eviction; shared optional aliases are counted per record.
    ContextResources evictable_resources;
};

enum class StateBindingAction : std::uint8_t { Root, Move, Fork, Restore, Split, BackupSplit };

struct SourceCandidate {
    std::optional<CheckpointHandle> checkpoint;
    std::uint32_t reused_tokens = 0;
    runtime::PrefillWork remaining_work;
    std::vector<runtime::ContextTransferRequirement> transfers;
    bool consume_source             = false;
    bool take_private               = false;
    StateBindingAction state_action = StateBindingAction::Root;
    bool move_history               = false;
    // Compatible carry intent. Binding may consume the selected point if its State slot
    // is necessary for execution and no preservation destination can be obtained.
    std::vector<CheckpointHandle> private_points;
    // Authorized retirement, applied only after binding capacity has been checked.
    std::vector<CheckpointHandle> retired_points;
};

struct BindingReservation {
    bool reserved          = false;
    bool source_valid      = true;
    bool capacity_possible = true;
    ContextResources shortage;
    std::vector<CheckpointHandle> retired_points;
    std::optional<CheckpointHandle> consumed_source;

    explicit operator bool() const noexcept { return reserved; }
};

struct ContextDemotion {
    std::vector<CheckpointHandle> sources;
    std::size_t host_bytes = 0;
    ContextResources released;
    std::shared_ptr<const detail::DemotionPlan> impl;
};

struct ContextRelease {
    std::vector<CheckpointHandle> sources;
    // Guaranteed release in the queried shortage pool; other quantities are not inventoried.
    ContextResources released;
    std::shared_ptr<const detail::ReleasePlan> impl;
};

class ContextDemotionBatch {
public:
    ContextDemotionBatch(ContextDemotionBatch&&) noexcept;
    ContextDemotionBatch& operator=(ContextDemotionBatch&&) noexcept;
    ~ContextDemotionBatch();
    ContextDemotionBatch(const ContextDemotionBatch&)            = delete;
    ContextDemotionBatch& operator=(const ContextDemotionBatch&) = delete;

    [[nodiscard]] bool append(const ContextDemotion&);
    [[nodiscard]] bool covers(const ContextRelease&) const;
    [[nodiscard]] std::optional<ContextDemotion> finish() const;

private:
    explicit ContextDemotionBatch(std::unique_ptr<detail::DemotionBatch>);
    std::unique_ptr<detail::DemotionBatch> impl_;
    friend struct ContextReclaimPlan;
};

// Read-only facts for one synchronous evaluation. Discard before any Session mutation or yield.
// start_demote independently revalidates the selected quote before reserving destinations.
struct ContextReclaimPlan {
    std::vector<ContextDemotion> demotions;
    std::vector<ContextRelease> releases;

    [[nodiscard]] ContextDemotionBatch begin_kv_batch(ContextResources shortage) const;

private:
    std::shared_ptr<detail::ReclaimPlan> impl_;
    friend class detail::SessionImpl;
};

class ResumeState {
public:
    ResumeState(ResumeState&&) noexcept;
    ResumeState& operator=(ResumeState&&) noexcept;
    ~ResumeState();
    ResumeState(const ResumeState&)            = delete;
    ResumeState& operator=(const ResumeState&) = delete;
    [[nodiscard]] bool has_snapshot() const noexcept;
    [[nodiscard]] std::optional<CheckpointHandle> snapshot_handle() const noexcept;
    [[nodiscard]] std::uint32_t frontier() const noexcept;

private:
    explicit ResumeState(std::unique_ptr<detail::ResumeStateImpl>) noexcept;
    std::unique_ptr<detail::ResumeStateImpl> impl_;

    friend class detail::SessionImpl;
};
enum class ExecutionUnitKind : std::uint8_t { Prefill, Replay, Decode, Control, Normalize };

struct ExecutionUnit {
    SequenceHandle sequence;
    ExecutionUnitKind kind = ExecutionUnitKind::Decode;
    // Decode: remaining output budget. Control: exact forced span. Other kinds: zero.
    std::uint32_t tokens = 0;
};

class PendingBatch {
public:
    PendingBatch()                                   = default;
    PendingBatch(PendingBatch&&) noexcept            = default;
    PendingBatch& operator=(PendingBatch&&) noexcept = default;
    PendingBatch(const PendingBatch&)                = delete;
    PendingBatch& operator=(const PendingBatch&)     = delete;

    [[nodiscard]] std::size_t row_count() const noexcept { return members_.size(); }

    [[nodiscard]] std::span<const TokenId> tokens() const noexcept { return tokens_; }

    [[nodiscard]] std::span<const std::int32_t> row_counts() const noexcept { return counts_; }

    [[nodiscard]] std::uint32_t row_stride() const noexcept { return stride_; }

    [[nodiscard]] bool constraint_failed(std::size_t row) const { return failed_.at(row); }

    [[nodiscard]] const ExecutionSample& execution_sample() const noexcept { return execution_; }

    [[nodiscard]] runtime::ExecutionTicket ticket() const noexcept { return ticket_; }

private:
    runtime::ExecutionTicket ticket_;
    std::vector<SequenceHandle> members_;
    std::vector<TokenId> tokens_;
    std::vector<std::int32_t> counts_;
    std::vector<bool> failed_;
    std::uint32_t stride_ = 1;
    ExecutionSample execution_;
    friend class detail::SessionImpl;
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
    std::size_t host_bytes           = 0;
    runtime::CheckpointOrigin origin = runtime::CheckpointOrigin::Semantic;
};
enum class ContextOperationKind : std::uint8_t { Bind, Capture, Demote, Pause };

struct ContextProgress {
    ContextOperationKind kind = ContextOperationKind::Bind;
    bool complete             = false;
    bool published            = false;
    std::optional<SequenceHandle> sequence;
    std::optional<runtime::ResidencyId> residency;
    std::optional<ResumeState> paused;
    bool replaying = false;
    std::vector<CheckpointHandle> private_points;
    std::vector<CheckpointHandle> retired_checkpoints;
    std::vector<CheckpointHandle> captured_checkpoints;
    std::vector<runtime::ContextTransferObservation> transfers;
    runtime::ContextOperationCounts operations;
    SpeculativeStats request_speculative;
};

struct PhysicalUsageSnapshot {
    ContextResources occupied;
    ContextResources capacity;
    std::size_t host_reserved_bytes      = 0;
    std::size_t host_peak_occupied_bytes = 0;
    std::uint32_t host_state_slots       = 0;
    std::size_t host_kv_bytes            = 0;
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

class Session {
public:
    using ContextResources = qwen3_5::ContextResources;
    [[nodiscard]] std::span<const runtime::KvPoolDescriptor> kv_pools() const noexcept;
    Session(PreparedSession&&, runtime::Executor<ExecutionTypes>&);
    ~Session() noexcept;
    Session(const Session&)            = delete;
    Session& operator=(const Session&) = delete;
    [[nodiscard]] RequestBasePlan plan_request(qwen::PreparedPrompt&& prompt,
                                               const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] std::optional<SourceCandidate>
    inspect_source(const RequestBasePlan& base, std::optional<CheckpointHandle> checkpoint,
                   bool consume_source                              = false,
                   std::span<const CheckpointHandle> private_points = {},
                   std::span<const CheckpointHandle> retired_points = {}) const;
    [[nodiscard]] PrefixShortlistKey checkpoint_key(CheckpointHandle, std::uint32_t frontier) const;
    [[nodiscard]] ContextResources checkpoint_footprint(std::span<const CheckpointHandle>) const;
    [[nodiscard]] CheckpointSummary checkpoint_summary(CheckpointHandle checkpoint) const;
    [[nodiscard]] CheckpointMetadata checkpoint_metadata(CheckpointHandle checkpoint) const;
    [[nodiscard]] bool checkpoint_matches(CheckpointHandle checkpoint,
                                          const RequestBasePlan& base) const;
    [[nodiscard]] std::uint32_t checkpoint_recovery_frontier(CheckpointHandle retained,
                                                             const RequestBasePlan& base,
                                                             std::uint32_t target) const;
    // Borrows this Session's index; checkpoint publication or retirement invalidates the view.
    [[nodiscard]] context::RecoveryEvaluation<>
    evaluate_recovery(std::span<const CheckpointHandle> advertised) const;
    [[nodiscard]] std::size_t checkpoint_capacity() const noexcept;
    [[nodiscard]] std::vector<runtime::ContextTransferRequirement>
        checkpoint_restore_requirements(CheckpointHandle) const;
    [[nodiscard]] bool valid_checkpoint(CheckpointHandle checkpoint) const noexcept;
    [[nodiscard]] bool release_checkpoint(CheckpointHandle checkpoint) noexcept;
    [[nodiscard]] bool revoke_snapshot(ResumeState& paused) noexcept;
    [[nodiscard]] ContextResources snapshot_resources(const ResumeState& paused) const;
    // Exact Host bytes freed by deleting this fixed set, with physical sharing counted once.
    [[nodiscard]] std::size_t
    host_bytes_released(std::span<const CheckpointHandle> checkpoints) const;
    [[nodiscard]] std::optional<std::size_t> pause_host_bytes(SequenceHandle sequence) const;
    [[nodiscard]] std::size_t
    release_redundant_host(std::span<const CheckpointHandle> excluded,
                           std::optional<SequenceHandle> pending_backup = std::nullopt);
    // Reservations stay owned by their lane through commit. Failure leaves prior permits intact.
    [[nodiscard]] ResourceReservation reserve_units(std::span<const ExecutionUnit> units);
    [[nodiscard]] bool reclaim_capture_reservation(ContextResources shortage);
    void release_units(std::span<const SequenceHandle> sequences) noexcept;
    [[nodiscard]] BindingReservation
    start_binding(const RequestBasePlan& base, runtime::LaneId lane, const SourceCandidate& source,
                  ResumeState* resume           = nullptr,
                  ExecutionUnitKind resume_kind = ExecutionUnitKind::Decode,
                  std::uint32_t resume_tokens   = 1);
    [[nodiscard]] bool start_capture(SequenceHandle sequence);
    [[nodiscard]] bool capture_is_input(SequenceHandle sequence) const;
    [[nodiscard]] bool capture_is_shared(SequenceHandle) const;
    [[nodiscard]] std::vector<std::uint32_t> capture_frontiers(SequenceHandle) const;
    [[nodiscard]] std::optional<CapturePreparation> prepare_capture(SequenceHandle sequence);
    void skip_capture(SequenceHandle sequence,
                      runtime::CaptureSkip reason = runtime::CaptureSkip::Unavailable);
    // Finite physical actions, each including every optional holder needed for its release.
    [[nodiscard]] ContextReclaimPlan plan_reclaim(std::span<const CheckpointHandle> allowed,
                                                  std::span<const CheckpointHandle> excluded,
                                                  ContextResources shortage) const;
    [[nodiscard]] context::ReleasePlan<ContextRelease, ContextResources>
    plan_releases(std::span<const CheckpointHandle> allowed,
                  std::span<const CheckpointHandle> excluded, ContextResources shortage) const;
    [[nodiscard]] bool start_demote(const ContextDemotion& plan);
    [[nodiscard]] bool start_pause(SequenceHandle sequence, bool save_snapshot);
    [[nodiscard]] bool has_context_operation() const noexcept;
    void cancel_context() noexcept;
    [[nodiscard]] bool context_blocks(SequenceHandle sequence) const noexcept;
    [[nodiscard]] bool execution_pending(SequenceHandle sequence) const noexcept;
    // A resumed binding retains its complete recovery capacity until committed new progress.
    [[nodiscard]] bool recovery_pending(SequenceHandle sequence) const noexcept;
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
    [[nodiscard]] CommitResult commit(PendingBatch&& pending,
                                      std::span<const runtime::CommitDecision> decisions);
    [[nodiscard]] DiscardResult discard(PendingBatch&& pending) noexcept;
    [[nodiscard]] FinishResult finish(SequenceHandle sequence) noexcept;
    [[nodiscard]] AbortResult abort(SequenceHandle sequence) noexcept;
    void fail_all_cleanup() noexcept;
    [[nodiscard]] PhysicalUsageSnapshot physical_usage() const noexcept;
    void reset_measurement_peaks();
private:
    std::unique_ptr<detail::SessionImpl> impl_;
};

} // namespace ninfer::models::qwen3_5
