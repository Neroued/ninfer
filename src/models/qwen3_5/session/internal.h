#pragma once

#include "models/qwen3_5/session/session.h"
#include "models/qwen3_5/state/continuation.h"
#include "models/common/context/snapshot.h"
#include "models/common/context/work.h"
#include "models/common/qwen/prefix_identity.h"
#include "models/common/qwen/recovery_store.h"
#include "models/common/qwen/vision/control.h"

#include <array>
#include <deque>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

class SessionImpl;
struct PendingContext;

struct PreparedCaptureBacking {
    qwen::PrefixShortlistDigests digests;
    std::vector<TokenId> ledger;
    qwen::ResidentPrefixIdentity prefix_identity;
};

struct CaptureGroup {
    std::shared_ptr<const PreparedCaptureBacking> identity;
    PrefixShortlistKey key;
    std::uint32_t frontier = 0;
    std::vector<runtime::CheckpointRole> roles;
    runtime::CheckpointOrigin origin = runtime::CheckpointOrigin::Semantic;
};

struct RequestBasePlanImpl {
    std::shared_ptr<const qwen::PreparedPromptData> prompt;
    runtime::RequestPlanSummary summary;
    models::frontend::PreparedContextCache context_cache;
    ResolvedSamplingParameters sampling;
    std::shared_ptr<const qwen::VisionExecutionInput> vision;
    std::vector<CaptureGroup> capture_groups;
    std::shared_ptr<const PreparedCaptureBacking> capture_backing;
    qwen::PrefixShortlistDigests prefix_digests;
    std::uint32_t prefix_identity_tag = 0;
    bool allow_prefix_reuse           = false;

    [[nodiscard]] bool accepts_capture(std::uint32_t frontier) const noexcept;
    [[nodiscard]] CaptureGroup capture_group(std::uint32_t frontier) const;
};
enum class Lifecycle : std::uint8_t {
    Empty,
    Binding,
    Prefilling,
    Replaying,
    Active,
    Pending,
    Finishable,
    Pausing
};

struct UnitDemand {
    ExecutionUnitKind kind         = ExecutionUnitKind::Decode;
    std::uint32_t tokens           = 0;
    std::uint32_t main_frontier    = 0;
    std::uint32_t backend_frontier = 0;
};

struct RecoveryPermit {
    // The active KV histories hold the physical reservation through this coverage.
    // These frontiers distinguish reconstructed work from committed new progress.
    UnitDemand coverage;
    std::uint32_t frontier    = 0;
    std::size_t ledger_tokens = 0;
};

struct KVHistoryView {
    context::HistoryId text;
    std::optional<context::HistoryId> backend;
    friend bool operator==(const KVHistoryView&, const KVHistoryView&) = default;
};

struct SequenceKVBundle {
    context::HistoryOwner text;
    std::optional<context::HistoryOwner> backend;

    [[nodiscard]] KVHistoryView view() const {
        return {text.handle(), backend ? std::optional(backend->handle()) : std::nullopt};
    }
};

struct SequenceState {
    std::optional<SequenceKVBundle> kv;
    Continuation continuation;
    context::StateId pending_write;
    std::uint32_t lane = 0;

    std::vector<TokenId> ledger;
    qwen::ResidentPrefixIdentity prefix_identity;
    qwen::PrefixShortlistDigests prefix_digests;
    SequenceHandle id;
    runtime::ResidencyId residency;

    [[nodiscard]] context::StateId write_state() const noexcept {
        return pending_write.valid() ? pending_write : continuation.image;
    }
};

struct CaptureReservation {
    SessionImpl* owner = nullptr;
    std::vector<std::pair<CheckpointHandle, runtime::CheckpointRole>> points;
    context::StateId state;
    std::optional<context::RecordReservation> host;
    std::uint32_t frontier = 0;
    ~CaptureReservation();
};

struct RequestControl {
    Lifecycle lifecycle = Lifecycle::Empty;
    ResolvedSamplingParameters sampling_host;
    SpeculativeStats speculative_stats;
    std::shared_ptr<const RequestBasePlanImpl> base;
    std::optional<UnitDemand> permit;
    std::optional<RecoveryPermit> recovery;
    std::unique_ptr<CaptureReservation> capture_reservation;
    std::uint32_t replay_target = 0;
    std::uint32_t replay_cursor = 0;
    Lifecycle resume_lifecycle  = Lifecycle::Empty;
    bool publish_continuation   = true;
    std::vector<CaptureGroup> capture_groups;
    std::size_t next_capture = 0;
    std::uint32_t last_saved = 0, last_intermediate_attempt = 0;
    bool capture_pending = false;

    struct Prefill {
        qwen::PreparedPromptData prompt;
        std::uint32_t base          = 0;
        std::uint32_t cursor        = 0;
        std::uint32_t prompt_tokens = 0;
        PrefixReusePath reuse       = PrefixReusePath::Root;
    };

    std::optional<Prefill> prefill;
};

using Snapshot = context::ContextSnapshot<ContinuationMetadata, PreparedCaptureBacking>;

struct CheckpointState {
    Snapshot snapshot;
    runtime::CheckpointRole role = runtime::CheckpointRole::Continuation;
    PrefixShortlistKey key;
    runtime::CheckpointOrigin origin = runtime::CheckpointOrigin::Semantic;

    [[nodiscard]] const Continuation& continuation() const noexcept {
        return snapshot.continuation();
    }

    [[nodiscard]] std::uint32_t frontier() const noexcept {
        return continuation().metadata.execution_frontier;
    }

    [[nodiscard]] qwen::RecoveryIdentityView recovery_identity() const {
        const auto& identity = snapshot.identity();
        return {frontier(),       key.identity_tag,
                identity.ledger,  identity.prefix_identity,
                identity.digests, snapshot.identity_owner().get()};
    }

    [[nodiscard]] std::uint32_t backend_frontier() const noexcept {
        return continuation().metadata.backend_kv_valid;
    }

    [[nodiscard]] const context::HistoryPrefix& text_prefix() const { return snapshot.prefix(0); }

    [[nodiscard]] const context::HistoryPrefix* backend_prefix() const {
        return snapshot.prefixes().size() == 2 ? &snapshot.prefix(1) : nullptr;
    }

    [[nodiscard]] KVHistoryView history() const {
        return {text_prefix().history(),
                backend_prefix() ? std::optional(backend_prefix()->history()) : std::nullopt};
    }
};

using Checkpoints = context::CheckpointStore<CheckpointState>;

struct PreparedCheckpoint {
    Checkpoints::Reservation record;
    std::shared_ptr<const PreparedCaptureBacking> identity;
};

struct ResumeStateImpl {
    SessionImpl* owner = nullptr;
    std::optional<CheckpointHandle> snapshot;
    SequenceState sequence;
    RequestControl control;
    std::uint32_t frontier = 0;
    ~ResumeStateImpl();
};

struct BindOperation {
    static constexpr auto kind = ContextOperationKind::Bind;
    std::uint32_t lane         = 0;
    std::optional<CheckpointHandle> source;
    Checkpoints::Lease source_lease;
    std::vector<CheckpointHandle> carried_points;
    std::vector<Checkpoints::Lease> carried_pins;
    std::vector<std::pair<CheckpointHandle, CheckpointHandle>> carried_clones;
    std::vector<CheckpointHandle> retired_points;
    std::shared_ptr<const RequestBasePlanImpl> base;
    ResumeState* resume            = nullptr;
    std::uint32_t reuse_frontier   = 0;
    std::uint32_t backend_frontier = 0;
    UnitDemand first_unit;
    UnitDemand reservation_demand;
    std::optional<RecoveryPermit> recovery;
    StateBindingAction state_action = StateBindingAction::Root;
    bool consume_source             = false;
    bool take_private               = false;
    bool resume_snapshot            = false;
    bool source_tail_hidden         = false;
};

struct BindingPlan {
    BindingEvaluation evaluation;
    BindOperation operation;
    ContextDemand demand;
    context::ContextDemandId state, main, backend, backup;
};

struct CaptureOperation {
    static constexpr auto kind     = ContextOperationKind::Capture;
    std::uint32_t lane             = 0;
    std::uint32_t frontier         = 0;
    std::uint32_t backend_frontier = 0;
    std::vector<std::pair<CheckpointHandle, runtime::CheckpointRole>> points;
};

struct MaintenanceOperation {
    static constexpr auto kind = ContextOperationKind::Maintenance;
    std::optional<ContextDemotion> demotion;
    std::vector<Checkpoints::Lease> source_leases;
    context::HostDemandIds targets;
    std::optional<SequenceHandle> pause_sequence;
    runtime::PauseAction pause_action;
};

struct DemoteOperation {
    static constexpr auto kind = ContextOperationKind::Demote;
};

struct PauseOperation {
    static constexpr auto kind = ContextOperationKind::Pause;
    std::uint32_t lane;
    ResumeState paused;
    bool preserve_state_device = true;
};

struct PendingContext {
    using Action = std::variant<BindOperation, CaptureOperation, DemoteOperation, PauseOperation,
                                MaintenanceOperation>;

    PendingContext(context::ContextDirectory& directory, Action value)
        : transaction(directory), action(std::move(value)) {}

    PendingContext(context::ContextTransaction&& prepared, Action value)
        : transaction(std::move(prepared)), action(std::move(value)) {}

    context::ContextTransaction transaction;
    Action action;
    std::optional<runtime::ExecutionReservation> reservation;
    std::optional<runtime::ExecutionTicket> ticket;
    std::optional<ContextResult> result;
    ContextWork input;
    context::ContextActionCursor cursor;
    bool submitted  = false;
    bool last_batch = false;
    std::vector<runtime::ContextTransferObservation> preceding_transfers;

    [[nodiscard]] std::optional<std::uint32_t> lane() const noexcept {
        return std::visit(
            [](const auto& value) -> std::optional<std::uint32_t> {
                if constexpr (requires { value.lane; }) return value.lane;
                return std::nullopt;
            },
            action);
    }

    [[nodiscard]] ContextOperationKind kind() const noexcept {
        return std::visit([](const auto& value) { return value.kind; }, action);
    }
};

class SessionImpl {
public:
    SessionImpl(std::shared_ptr<const ModelDefinition>, SessionPlan);
    ~SessionImpl();
    [[nodiscard]] RequestBasePlan plan_request(qwen::PreparedPromptData&& prompt,
                                               const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] std::optional<SourceCandidate>
    inspect_source(const RequestBasePlan& base, std::optional<CheckpointHandle> checkpoint,
                   bool consume_source                              = false,
                   std::span<const CheckpointHandle> private_points = {},
                   std::span<const CheckpointHandle> retired_points = {},
                   runtime::PlacementId target = {}, const ResumeState* resume = nullptr) const;
    [[nodiscard]] PrefixShortlistKey checkpoint_key(CheckpointHandle, std::uint32_t frontier) const;
    [[nodiscard]] CheckpointSummary checkpoint_summary(CheckpointHandle checkpoint) const;
    [[nodiscard]] CheckpointMetadata checkpoint_metadata(CheckpointHandle checkpoint) const;
    [[nodiscard]] bool checkpoint_matches(CheckpointHandle checkpoint,
                                          const RequestBasePlan& base) const;
    [[nodiscard]] std::uint32_t checkpoint_recovery_frontier(CheckpointHandle retained,
                                                             const RequestBasePlan& base,
                                                             std::uint32_t target) const;
    [[nodiscard]] bool valid_checkpoint(CheckpointHandle checkpoint) const noexcept;
    [[nodiscard]] bool release_checkpoint(CheckpointHandle checkpoint) noexcept;
    [[nodiscard]] bool revoke_snapshot(ResumeState& paused) noexcept;
    // Exact Host bytes freed by deleting this fixed set, with physical sharing counted once.
    [[nodiscard]] std::size_t
    host_bytes_released(std::span<const CheckpointHandle> checkpoints) const;
    [[nodiscard]] runtime::CheckpointEvictableResources
    checkpoint_evictable_resources(CheckpointHandle checkpoint) const;
    [[nodiscard]] std::size_t state_image_bytes() const noexcept;
    [[nodiscard]] std::size_t kv_page_bytes(std::uint32_t pool) const noexcept;
    [[nodiscard]] std::uint32_t state_slots_capacity() const noexcept;
    [[nodiscard]] std::optional<context::HostWriteRequirements>
        pause_host_requirements(SequenceHandle) const;
    [[nodiscard]] std::vector<PauseOption> pause_options(SequenceHandle) const;
    [[nodiscard]] bool pause_contributes(SequenceHandle, const PauseOption&,
                                         const ContextAssessment&) const;
    [[nodiscard]] bool prepare_pause(SequenceHandle, runtime::PauseAction,
                                     std::span<const CheckpointHandle> host_victims);
    [[nodiscard]] context::HostWriteRequirements
    demotion_host_requirements(const ContextDemotion&) const;
    [[nodiscard]] ContextDemotion combine_demotions(std::span<const ContextDemotion>) const;
    [[nodiscard]] std::size_t
    release_redundant_host(std::span<const CheckpointHandle> excluded,
                           std::optional<SequenceHandle> pending_backup = std::nullopt);
    // Reservations stay owned by their lane through commit. Failure leaves prior permits intact.
    [[nodiscard]] ResourceReservation reserve_units(std::span<const ExecutionUnit> units);
    [[nodiscard]] runtime::ContextStart start_maintenance(const ContextAssessment&);
    [[nodiscard]] bool reclaim_capture_reservation(const ContextAssessment&);
    void release_units(std::span<const SequenceHandle> sequences) noexcept;
    [[nodiscard]] BindingPlan plan_binding(const RequestBasePlan&, runtime::LaneId,
                                           const SourceCandidate&, const ResumeState*,
                                           ExecutionUnitKind, std::uint32_t) const;
    [[nodiscard]] BindingReservation
    start_binding(const RequestBasePlan& base, runtime::LaneId lane, const SourceCandidate& source,
                  ResumeState* resume           = nullptr,
                  ExecutionUnitKind resume_kind = ExecutionUnitKind::Decode,
                  std::uint32_t resume_tokens   = 1);
    [[nodiscard]] bool start_capture(SequenceHandle sequence);
    [[nodiscard]] bool capture_is_input(SequenceHandle sequence) const;
    [[nodiscard]] std::optional<CapturePreparation>
    prepare_capture(SequenceHandle sequence, std::span<const CheckpointHandle> host_victims = {});
    void skip_capture(SequenceHandle sequence,
                      runtime::CaptureSkip reason = runtime::CaptureSkip::Unavailable);
    // Finite physical actions, each including every optional holder needed for its release.
    [[nodiscard]] ContextReclaimPlan plan_reclaim(std::span<const CheckpointHandle> allowed,
                                                  std::span<const CheckpointHandle> excluded,
                                                  const ContextDemand&) const;
    [[nodiscard]] context::ReleasePlan<ContextRelease, ContextReleaseSet>
    plan_releases(std::span<const CheckpointHandle> allowed,
                  std::span<const CheckpointHandle> excluded, const ContextDemand&) const;
    [[nodiscard]] runtime::ContextStart
    start_demote(const ContextDemotion&, std::span<const CheckpointHandle> host_victims);
    [[nodiscard]] runtime::ContextStart submit_demotion(const ContextDemotion&,
                                                        context::HostDemandIds,
                                                        context::ContextReservation,
                                                        runtime::ExecutionReservation);
    [[nodiscard]] runtime::ContextStart
    defer_demotion(const ContextDemotion&, context::HostDemandIds, context::ContextReservation,
                   const ContextAssessment&, runtime::ExecutionReservation);
    [[nodiscard]] bool start_pause(SequenceHandle sequence, runtime::PauseAction);
    [[nodiscard]] bool has_context_operation() const noexcept;
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
    [[nodiscard]] std::optional<runtime::ExecutionTicket> start_score(qwen::PreparedPromptData&&,
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

    std::shared_ptr<const ModelDefinition> definition;
    SessionPlan plan;
    vision::Window vision_window;
    [[nodiscard]] std::optional<qwen::VisionWork>
    prepare_visual(const SequenceState&, std::uint32_t begin, std::uint32_t end);
    runtime::Executor<ExecutionTypes>* executor_ = nullptr;

    [[nodiscard]] runtime::Executor<ExecutionTypes>& executor() const { return *executor_; }

    context::ContextDirectory directory;
    context::StateDirectory* state_store;
    context::HistoryDirectory* text_kv_histories;
    context::HistoryDirectory* backend_kv_histories;
    context::PageDirectory* text_kv_pages;
    context::PageDirectory* backend_kv_pages;
    const bool host_context_enabled;
    const std::uint32_t capacity;
    const std::uint32_t max_concurrency;
    const std::uint32_t prefill_chunk;
    const std::uint32_t draft_window;
    const SpeculativeBackend speculative_backend;
    const ContextCacheOptions context_cache;
    const bool vision_enabled;
    const bool causal_scoring;
    std::array<SequenceState, kMaximumConcurrency> sequences;
    std::array<RequestControl, kMaximumConcurrency> requests;
    std::array<std::uint64_t, kMaximumConcurrency> lane_epochs{};
    std::uint64_t next_sequence = 1;
    mutable context::RecoveryIndex<> recovery_index{directory.metadata_budget()};
    Checkpoints checkpoints;
    context::MetadataAccount count_storage{directory.metadata_budget()};
    TokenCountBuilder count_builder;
    std::unique_ptr<PendingContext> context_operation_;
    std::optional<ContextAssessment> pending_maintenance_;
    std::optional<SequenceHandle> normalizing_pause_;

    struct PauseReservation {
        SequenceHandle sequence;
        runtime::PauseAction action;
        ContextAssessment placement;
        context::HostDemandIds target_ids;
        context::ContextReservation targets;
        runtime::ExecutionReservation execution;
        std::optional<runtime::ExecutionReservation> normalization;
        std::vector<runtime::ContextTransferObservation> preceding_transfers;
        PreparedCheckpoint checkpoint;
    };

    std::optional<PauseReservation> pause_reservation_;
    bool pause_cancelled_ = false;
    std::deque<SessionEvent> events;
    std::vector<runtime::ResidencyId> releases;
    std::optional<runtime::ExecutionTicket> release_ticket;
    bool release_result = false;
    bool closing        = false;
    std::optional<SessionEvent> after_releases;
    void release_residency(runtime::ResidencyId);
    void submit_releases();

    enum class OperationKind { Prefill, Replay, Decode, Control, Score, Normalize };

    struct Execution {
        OperationKind kind;
        runtime::ExecutionTicket ticket;
        std::vector<SequenceHandle> members;
        std::vector<Continuation> bases;
        std::vector<ContinuationMetadata> completed;
        std::vector<std::uint32_t> columns;
        std::vector<std::vector<TokenId>> inputs;
        std::vector<std::optional<std::uint32_t>> splits;
        std::unique_ptr<context::ContextTransaction> preparation;
        std::unique_ptr<context::ContextTransaction> settlement;
        std::optional<GeneratedResult> generated;
        ExecutionSample observed_result;
        std::optional<SessionEvent> completion;
        std::vector<runtime::CommitDecision> decisions;
        std::uint64_t vision_id = 0;
        bool result_ready       = false;
        bool committed          = false;
        bool discarded          = false;
    };

    std::optional<Execution> execution;

    struct Scoring {
        runtime::ExecutionTicket public_ticket;
        RequestBasePlan base;
        SequenceHandle sequence;
        std::uint32_t first_target = 0;
        std::uint32_t cursor       = 0;
        std::vector<float> logprobs;
    };

    std::optional<Scoring> scoring;

    [[nodiscard]] std::uint32_t lane(SequenceHandle) const;
    [[nodiscard]] bool valid_sequence(SequenceHandle) const noexcept;
    [[nodiscard]] SequenceHandle sequence_handle(std::uint32_t) const noexcept;
    [[nodiscard]] SequenceState& active_sequence(std::uint32_t);
    [[nodiscard]] const SequenceState& active_sequence(std::uint32_t) const;
    void invalidate_lane(std::uint32_t) noexcept;
    [[nodiscard]] UnitDemand prefill_unit(std::uint32_t prompt, std::uint32_t cursor) const;
    [[nodiscard]] UnitDemand next_unit(const SequenceState&, const RequestControl&,
                                       ExecutionUnitKind, std::uint32_t tokens) const;
    void require_unit(std::uint32_t, ExecutionUnitKind, std::uint32_t tokens = 0) const;
    [[nodiscard]] bool prepare_settlement(std::uint32_t lane,
                                          const ContinuationMetadata& continuation,
                                          Lifecycle lifecycle, std::size_t ledger_tokens,
                                          context::ContextTransaction&);
    void adopt_unit(std::uint32_t lane, const ContinuationMetadata& continuation,
                    bool state_written, bool retain_recovery) noexcept;
    [[nodiscard]] std::uint32_t backend_kv_valid(const SequenceState&) const noexcept;
    [[nodiscard]] bool has_backend_pool() const noexcept;
    [[nodiscard]] ExecutionRow execution_row(const SequenceState&) const;
    [[nodiscard]] InputWindow input_window(const SequenceState&, std::uint32_t begin,
                                           std::uint32_t end) const;
    [[nodiscard]] std::uint32_t window_end(const SequenceState&, const RequestControl&,
                                           std::uint32_t begin, std::uint32_t end,
                                           const vision::Window&) const;
    void ensure_sequence_kv_mapped(SequenceState&, std::uint32_t main, std::uint32_t backend,
                                   context::ContextTransaction&);
    void complete_state_write(SequenceState&);
    void release_sequence_state(SequenceState&) noexcept;
    void release_sequence_kv(SequenceState&) noexcept;
    void clear_lane(SequenceState&, RequestControl&) noexcept;

    void receive_result(runtime::ResultReady<ExecutionTypes>&&);
    void retire(runtime::DeviceRetired);
    void commit_completed_unit();
    void advance_score();
    [[nodiscard]] PendingBatch generated_pending(const GeneratedResult&);
    [[nodiscard]] std::optional<runtime::ExecutionTicket>
    submit_execution(Execution&&, runtime::ExecutionReservation, Work&&);

    [[nodiscard]] bool can_release_checkpoint(CheckpointHandle) const noexcept;
    [[nodiscard]] std::optional<ContextReleaseSet>
        checkpoint_release_set(std::span<const CheckpointHandle>) const;
    [[nodiscard]] CheckpointState& checkpoint(CheckpointHandle);
    [[nodiscard]] const CheckpointState& checkpoint(CheckpointHandle) const;
    [[nodiscard]] std::optional<CheckpointHandle> reserve_checkpoint();
    [[nodiscard]] PreparedCheckpoint prepare_checkpoint(SequenceState&);
    [[nodiscard]] std::optional<CheckpointHandle> detach_checkpoint(SequenceState&,
                                                                    PreparedCheckpoint = {});
    void initialize_prefill(RequestControl&, std::uint32_t base);
    void initialize_captures(RequestControl&, std::uint32_t from, std::uint32_t through);
    [[nodiscard]] bool
    reserve_capture_destination(std::uint32_t, std::uint32_t,
                                std::span<const CheckpointHandle> host_victims = {});
    void prepare_capture_boundary(std::uint32_t);
    void abort_context() noexcept;
    void cancel_context() noexcept;
    [[nodiscard]] bool prepare_backup(context::ContextTransaction&, CheckpointState&,
                                      bool shared_device, bool& preserve_state_device,
                                      context::ContextReservation&, const context::HostDemandIds&);
    void install_binding(BindOperation&);
    void complete_binding(BindOperation&, context::ContextTransaction&, ContextProgress&);
    void publish_capture(CaptureOperation&, context::ContextTransaction&,
                         runtime::ContextOperationCounts&);
    [[nodiscard]] bool begin_pause(SequenceHandle, runtime::PauseAction);
    [[nodiscard]] bool continue_pause_ready(SequenceHandle, runtime::PauseAction);
    void continue_pause_after_normalize(SequenceHandle);
    [[nodiscard]] ResumeState complete_pause(PauseOperation&);
    [[nodiscard]] bool submit_context(PendingContext&, ContextWork&&);
    void submit_context_batch(PendingContext&);
    void receive_context_result(runtime::ExecutionTicket, ContextResult&&);
    void retire_context(runtime::DeviceRetired);
    [[nodiscard]] std::optional<runtime::ExecutionTicket>
        start_normalize(SequenceHandle, runtime::ExecutionReservation);
};

} // namespace ninfer::models::qwen3_5::detail
