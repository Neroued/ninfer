#pragma once

#include "models/common/context/snapshot.h"
#include "models/glm5next/definition.h"
#include "models/glm5next/session/session.h"
#include "models/glm5next/state/continuation.h"
#include "models/glm5next/state/identity.h"

#include <deque>

namespace ninfer::models::glm5next::detail {
using Image        = context::StateId;
using Continuation = context::Continuation<ContinuationMetadata>;

struct CapturePoint {
    std::uint32_t frontier;
    std::vector<runtime::CheckpointRole> roles;
    runtime::CheckpointOrigin origin = runtime::CheckpointOrigin::Semantic;
};

using Identity = InputIdentity;

struct RequestPlan {
    PreparedPromptData prompt;
    std::shared_ptr<const vision::ExecutionInput> vision;
    runtime::RequestPlanSummary summary;
    std::shared_ptr<const Identity> identity;
    std::vector<CapturePoint> captures;
    ResolvedSamplingParameters sampling;
    bool reuse = true;

    [[nodiscard]] bool can_checkpoint(std::uint32_t frontier) const;
};

using Snapshot = context::ContextSnapshot<ContinuationMetadata, Identity>;

struct Checkpoint {
    Snapshot snapshot;
    runtime::CheckpointRole role;
    runtime::CheckpointOrigin origin = runtime::CheckpointOrigin::Semantic;

    [[nodiscard]] std::uint32_t frontier() const {
        return snapshot.continuation().metadata.main_frontier;
    }
};

using Checkpoints = context::CheckpointStore<Checkpoint>;

struct PreparedCheckpoint {
    Checkpoints::Reservation record;
    std::shared_ptr<const Identity> identity;
};

enum class Phase { Prefill, Replay, Active, Pending, Finished };

struct CaptureReservation {
    context::StateDirectory* states;
    Checkpoints* checkpoints;
    Image destination;
    std::vector<std::pair<CheckpointHandle, runtime::CheckpointRole>> records;
    ~CaptureReservation();
};

struct Sequence {
    std::uint64_t id = 0, epoch = 1;
    std::uint32_t lane               = 0;
    std::uint64_t binding_generation = 0;
    std::shared_ptr<const RequestPlan> request;
    Identity identity;
    Continuation continuation;
    Image write_destination;
    std::array<context::HistoryOwner, 2> histories;
    Phase phase = Phase::Prefill, resume_phase = Phase::Active;
    std::uint32_t reused = 0, replay_target = 0, recovery_peak = 0;
    std::optional<ExecutionUnit> permit;
    std::uint32_t permitted_frontier = 0;
    std::size_t next_capture         = 0;
    std::vector<CapturePoint> captures;
    std::uint32_t last_saved = 0, last_intermediate_attempt = 0;
    std::unique_ptr<CaptureReservation> capture;
    bool capture_pending = false;
    SpeculativeStats speculative;
};

struct Suspended {
    SessionImpl* owner;
    std::unique_ptr<Sequence> sequence;
    std::optional<CheckpointHandle> snapshot;
    std::uint32_t frontier;
    ~Suspended();
};

enum class StateBindingAction { Root, Move, Fork, Convert };

struct BindOperation {
    static constexpr auto kind = ContextOperationKind::Bind;
    std::uint32_t lane = 0, frontier = 0, target = 0, peak = 0;
    runtime::PlacementId placement;
    std::shared_ptr<const RequestPlan> request;
    ResumeState* resume = nullptr;
    std::unique_ptr<Sequence> binding;
    std::optional<CheckpointHandle> source;
    std::optional<Checkpoints::Lease> source_lease;
    std::vector<Checkpoints::Lease> carried_leases;
    std::vector<CheckpointHandle> carried, retired;
    std::vector<std::pair<CheckpointHandle, CheckpointHandle>> clones;
    Image source_image;
    StateBindingAction state_action = StateBindingAction::Root;
    bool consume_source             = false;
    context::ContextDemandId state_target;
    std::array<context::ContextDemandId, 2> history_targets;
};

struct BindingPlan {
    BindingEvaluation evaluation;
    BindOperation action;
    ContextDemand demand;
    std::vector<CheckpointHandle> retired;
};

struct CaptureOperation {
    static constexpr auto kind = ContextOperationKind::Capture;
    std::uint32_t lane;
    std::unique_ptr<CaptureReservation> reservation;
    std::optional<Snapshot> snapshot;
};

struct RelocationTargets {
    context::ContextDemandId state;
    std::array<context::ContextDemandId, 2> histories;
};

struct PauseOperation {
    static constexpr auto kind = ContextOperationKind::Pause;
    std::uint32_t lane;
    std::unique_ptr<Suspended> paused;
    runtime::PauseAction action;
    PreparedCheckpoint checkpoint;
    RelocationTargets relocation;
};

struct MaintenanceOperation {
    static constexpr auto kind = ContextOperationKind::Maintenance;
};

struct ContextOperation {
    using Action =
        std::variant<BindOperation, CaptureOperation, PauseOperation, MaintenanceOperation>;
    context::ContextTransaction migration;
    Action action;
    runtime::ExecutionReservation reservation;
    std::optional<runtime::ExecutionTicket> ticket;
    std::optional<ContextResult> result;
    bool preparing_resources = false;
    bool actions_prepared    = false;
    bool final_batch         = false;
    std::map<std::uint32_t, context::ContextActionCursor> cursors;
    ContextWork residency;
    std::vector<runtime::ContextTransferObservation> previous_transfers;

    ContextOperation(context::ContextDirectory& directory, Action value,
                     runtime::ExecutionReservation reserved)
        : migration(directory), action(std::move(value)), reservation(std::move(reserved)) {}

    ContextOperation(context::ContextTransaction&& transaction, Action value,
                     runtime::ExecutionReservation reserved)
        : migration(std::move(transaction)), action(std::move(value)),
          reservation(std::move(reserved)) {}

    [[nodiscard]] ContextOperationKind kind() const {
        return std::visit([](const auto& value) { return value.kind; }, action);
    }
};

struct ComputeTransaction {
    enum class Kind { Prefill, Replay, Ordinary, Mtp, Control, Score };

    Kind kind;
    context::ContextTransaction context;
    runtime::ExecutionTicket ticket;
    std::vector<SequenceHandle> members;
    std::vector<WorkRow> rows;
    std::vector<std::uint32_t> draft_extents;
    std::vector<std::optional<std::uint32_t>> splits;
    std::optional<ComputeResult> result;
    ExecutionSample observed_result;
    std::vector<runtime::CommitDecision> decisions;
    std::unique_ptr<context::ContextTransaction> completion;
    std::uint64_t vision_id = 0;
    bool committed          = false;
    bool discarded          = false;
    std::optional<SessionEvent> event;

    ComputeTransaction(Kind kind, context::ContextDirectory& directory)
        : kind(kind), context(directory) {}
};

struct Scoring {
    std::uint32_t first_target = 1;
    std::optional<SequenceHandle> sequence;
    runtime::ExecutionTicket ticket;
    std::vector<float> logprobs;
    bool finishing = false;
};

class SessionImpl {
public:
    struct PauseReservation {
        SequenceHandle sequence;
        runtime::PauseAction action;
        ContextDemand demand;
        RelocationTargets relocation;
        runtime::ExecutionReservation execution;
        PreparedCheckpoint checkpoint;
    };

    SessionImpl(std::shared_ptr<const ModelDefinition>, SessionPlan,
                std::unique_ptr<context::ContextDirectory>);
    ~SessionImpl();

    [[nodiscard]] bool valid(SequenceHandle) const noexcept;
    [[nodiscard]] Sequence& sequence(SequenceHandle) const;
    [[nodiscard]] SequenceHandle handle(const Sequence&) const;
    [[nodiscard]] runtime::ExecutionMember member(const Sequence&) const;
    [[nodiscard]] StateBindings selectors(const Sequence&) const;
    [[nodiscard]] std::uint32_t pool_count() const noexcept;
    [[nodiscard]] std::uint32_t window_end(const Sequence&, std::uint32_t begin,
                                           std::uint32_t end) const;
    [[nodiscard]] std::uint32_t unit_end(const Sequence&, ExecutionUnitKind, std::uint32_t) const;
    [[nodiscard]] BindingPlan plan_binding(const RequestBasePlan&, runtime::LaneId,
                                           const SourceCandidate&, const ResumeState*,
                                           ExecutionUnitKind, std::uint32_t) const;
    [[nodiscard]] bool source_matches(const RequestBasePlan&, CheckpointHandle,
                                      const ResumeState*) const;
    void complete_write(Sequence&);
    void clear(Sequence&) noexcept;
    void settle(Sequence&);
    void settle(Sequence&, context::ContextTransaction&);
    void release_residency(runtime::ResidencyId);
    void submit_releases();
    void cleanup() noexcept;

    [[nodiscard]] std::optional<vision::Work> prepare_visual(const Sequence&, std::uint32_t begin,
                                                             std::uint32_t end);
    [[nodiscard]] WorkRow work_row(const Sequence&, std::uint32_t tokens) const;
    void prepare_write(Sequence&, std::uint32_t end, context::ContextTransaction&);
    [[nodiscard]] ContinuationMetadata prepare_commit(Sequence&, const WorkRow&,
                                                      std::uint32_t input_columns,
                                                      bool retain_recovery,
                                                      context::ContextTransaction&);
    void adopt_commit(Sequence&, const ContinuationMetadata&, bool state_written,
                      bool retain_recovery) noexcept;
    void retire_compute(const runtime::DeviceRetired&);
    void finish_result(ComputeResult);
    void finish_score(ScoreResult);
    [[nodiscard]] ResourceReservation reserve_units(std::span<const ExecutionUnit>);
    [[nodiscard]] bool submit_score();

    [[nodiscard]] ContextActions actions(const context::ContextTransaction&) const;
    void prepare_binding_resources(context::ContextReservation&);
    void prepare_pause_resources(context::ContextReservation&);
    void resume_context(context::ContextReservation);
    [[nodiscard]] runtime::ContextStart try_pending_maintenance();
    void submit_context();
    [[nodiscard]] ContextProgress complete_context();
    void cancel_operation() noexcept;
    [[nodiscard]] ResidencyBinding context_binding(BindOperation&);

    [[nodiscard]] PreparedCheckpoint prepare_checkpoint(Sequence&);
    [[nodiscard]] std::optional<CheckpointHandle> detach(Sequence&, runtime::CheckpointRole,
                                                         PreparedCheckpoint = {});
    [[nodiscard]] std::vector<context::PageId> checkpoint_pages(std::span<const CheckpointHandle>,
                                                                std::uint32_t pool) const;
    [[nodiscard]] std::vector<Image> checkpoint_images(std::span<const CheckpointHandle>) const;
    [[nodiscard]] ContextReleaseSet footprint(std::span<const CheckpointHandle>,
                                              bool released) const;
    [[nodiscard]] ContextReclaimPlan plan_reclaim(std::span<const CheckpointHandle>,
                                                  std::span<const CheckpointHandle>,
                                                  const ContextDemand&) const;
    [[nodiscard]] context::ReleasePlan<ContextRelease, ContextReleaseSet>
    plan_releases(std::span<const CheckpointHandle>, std::span<const CheckpointHandle>,
                  const ContextDemand&) const;
    [[nodiscard]] PhysicalUsageSnapshot usage() const noexcept;
    std::shared_ptr<const ModelDefinition> definition;
    const SessionPlan plan;
    const ExecutionShape& shape;
    runtime::Executor<ExecutionTypes>* executor = nullptr;
    std::unique_ptr<context::ContextDirectory> directory_owner;
    context::ContextDirectory& directory;
    std::vector<runtime::PlacementId> placements;
    mutable context::RecoveryIndex<> recovery_index{directory.metadata_budget()};
    Checkpoints checkpoints;
    std::array<std::unique_ptr<Sequence>, kMaximumConcurrency> sequences;
    std::array<std::uint64_t, kMaximumConcurrency> epochs{};
    std::vector<runtime::KvPoolDescriptor> pools;
    std::unique_ptr<ContextOperation> operation;
    std::optional<ContextAssessment> pending_maintenance;
    std::optional<PauseReservation> pause_reservation;
    models::vision::Window vision_window;
    std::unique_ptr<ComputeTransaction> compute;
    std::unique_ptr<Scoring> scoring;
    std::deque<SessionEvent> events;
    std::vector<runtime::ResidencyId> releases;
    std::optional<runtime::ExecutionTicket> release_ticket;
    std::uint64_t next_sequence       = 1;
    std::uint64_t identity_generation = 0;
};

[[nodiscard]] std::vector<runtime::ContextTransferRequirement>
conversion_requirements(const SessionPlan&, const Snapshot&, std::uint32_t to);

[[nodiscard]] std::shared_ptr<const Identity> capture_identity(const Sequence&);
[[nodiscard]] PrefixShortlistKey key(const Identity&, std::uint32_t);

} // namespace ninfer::models::glm5next::detail
