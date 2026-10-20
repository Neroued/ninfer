#pragma once

#include "models/qwen4_exp/session/session.h"
#include "models/qwen4_exp/state/continuation.h"
#include "models/qwen4_exp/definition.h"
#include "models/common/context/snapshot.h"
#include "models/common/qwen/prefix_identity.h"
#include "models/common/qwen/vision/control.h"

#include <deque>

namespace ninfer::models::qwen4_exp::detail {
using Image        = context::StateId;
using Continuation = context::Continuation<ContinuationMetadata>;

struct CapturePoint {
    std::uint32_t frontier;
    std::vector<runtime::CheckpointRole> roles;
    runtime::CheckpointOrigin origin = runtime::CheckpointOrigin::Semantic;
};

struct Identity {
    std::vector<TokenId> tokens;
    qwen::ResidentPrefixIdentity positions;
    qwen::PrefixShortlistDigests digests;
    std::uint32_t tag = 0;
};

struct RequestPlan {
    qwen::PreparedPromptData prompt;
    runtime::RequestPlanSummary summary;
    std::shared_ptr<const Identity> identity;
    std::vector<CapturePoint> captures;
    std::shared_ptr<const qwen::VisionExecutionInput> vision;
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

    [[nodiscard]] qwen::RecoveryIdentityView recovery_identity() const {
        const auto& identity = snapshot.identity();
        return {frontier(),         identity.tag,     identity.tokens,
                identity.positions, identity.digests, snapshot.identity_owner().get()};
    }
};

using Checkpoints = context::CheckpointStore<Checkpoint>;

enum class Phase { Prefill, Replay, Active, Pending, Finished };

struct CaptureReservation {
    context::StateDirectory* states;
    Checkpoints* checkpoints;
    Image destination;
    std::optional<context::HostAllocation> host;
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

struct Demotion {
    struct PageVersion {
        std::uint64_t epoch;
        std::uint32_t columns;
    };

    const SessionImpl* owner          = nullptr;
    std::uint64_t checkpoint_revision = 0;
    std::vector<Image> images;
    std::vector<std::uint64_t> image_epochs;
    std::array<std::vector<context::PageId>, 2> pages;
    std::array<std::vector<PageVersion>, 2> page_versions;
    std::vector<CheckpointHandle> records;

    [[nodiscard]] bool empty() const {
        return images.empty() && pages[0].empty() && pages[1].empty();
    }
};

struct Reclaim {
    const SessionImpl* owner;
    ContextResources shortage;
};

struct Release {
    const SessionImpl* owner;
    std::array<std::vector<context::PageId>, 2> pages;
};

enum class StateBindingAction { Root, Move, Fork, Restore, Split };

struct BindOperation {
    static constexpr auto kind = ContextOperationKind::Bind;
    std::uint32_t lane = 0, frontier = 0, target = 0, peak = 0;
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
};

struct CaptureOperation {
    static constexpr auto kind = ContextOperationKind::Capture;
    std::uint32_t lane;
    std::unique_ptr<CaptureReservation> reservation;
    std::optional<Snapshot> snapshot;
};

struct PauseOperation {
    static constexpr auto kind = ContextOperationKind::Pause;
    std::uint32_t lane;
    std::unique_ptr<Suspended> paused;
};

struct DemoteOperation {
    static constexpr auto kind = ContextOperationKind::Demote;
    std::shared_ptr<const Demotion> plan;
};

struct ContextOperation {
    using Action = std::variant<BindOperation, CaptureOperation, PauseOperation, DemoteOperation>;
    context::ContextTransaction migration;
    Action action;
    runtime::ExecutionReservation reservation;
    std::optional<runtime::ExecutionTicket> ticket;
    std::optional<ContextResult> result;

    ContextOperation(context::ContextDirectory& directory, Action value,
                     runtime::ExecutionReservation reserved)
        : migration(directory), action(std::move(value)), reservation(std::move(reserved)) {}

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
    std::vector<std::uint32_t> extents;
    std::vector<std::optional<std::uint32_t>> splits;
    std::optional<ComputeResult> result;
    ExecutionSample observed_result;
    std::vector<runtime::CommitDecision> decisions;
    std::unique_ptr<context::ContextTransaction> completion;
    bool committed = false;
    bool discarded = false;
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
    SessionImpl(std::shared_ptr<const ModelDefinition>, SessionPlan);
    ~SessionImpl();

    [[nodiscard]] bool valid(SequenceHandle) const noexcept;
    [[nodiscard]] Sequence& sequence(SequenceHandle) const;
    [[nodiscard]] SequenceHandle handle(const Sequence&) const;
    [[nodiscard]] runtime::ExecutionMember member(const Sequence&) const;
    [[nodiscard]] context::StateSelectors selectors(const Sequence&) const;
    [[nodiscard]] std::uint32_t pool_count() const noexcept;
    [[nodiscard]] std::uint32_t window_end(const Sequence&, std::uint32_t begin,
                                           std::uint32_t end) const;
    [[nodiscard]] std::uint32_t unit_end(const Sequence&, ExecutionUnitKind, std::uint32_t) const;
    void complete_write(Sequence&);
    void clear(Sequence&) noexcept;
    void settle(Sequence&);
    void settle(Sequence&, context::ContextTransaction&);
    void release_residency(runtime::ResidencyId);
    void submit_releases();
    void cleanup() noexcept;

    [[nodiscard]] WorkRow work_row(const Sequence&, std::uint32_t tokens) const;
    void prepare_write(Sequence&, std::uint32_t end, context::ContextTransaction&);
    [[nodiscard]] ContinuationMetadata prepare_commit(Sequence&, const WorkRow&,
                                                      std::span<const TokenId> inputs,
                                                      bool retain_recovery,
                                                      context::ContextTransaction&);
    void adopt_commit(Sequence&, const ContinuationMetadata&, bool state_written,
                      bool retain_recovery) noexcept;
    void retire_compute(const runtime::DeviceRetired&);
    void finish_result(ComputeResult);
    void finish_score(ScoreResult);
    [[nodiscard]] bool submit_score();

    void submit_context();
    [[nodiscard]] ContextProgress complete_context();
    void cancel_operation() noexcept;
    [[nodiscard]] ResidencyBinding context_binding(BindOperation&);

    [[nodiscard]] std::shared_ptr<const qwen::VisionExecutionInput>
    plan_vision(const qwen::PreparedPromptData&) const;
    [[nodiscard]] std::optional<CheckpointHandle> detach(Sequence&, runtime::CheckpointRole);
    [[nodiscard]] std::vector<context::PageId> checkpoint_pages(std::span<const CheckpointHandle>,
                                                                std::uint32_t pool) const;
    [[nodiscard]] std::vector<Image> checkpoint_images(std::span<const CheckpointHandle>) const;
    [[nodiscard]] ContextResources footprint(std::span<const CheckpointHandle>,
                                             bool released) const;
    [[nodiscard]] std::size_t host_released(std::span<const CheckpointHandle>) const;
    [[nodiscard]] std::shared_ptr<Demotion> demotion(std::span<const CheckpointHandle>,
                                                     ContextResources shortage) const;
    [[nodiscard]] ContextDemotion describe_demotion(std::shared_ptr<const Demotion>) const;
    [[nodiscard]] ContextReclaimPlan plan_reclaim(std::span<const CheckpointHandle>,
                                                  std::span<const CheckpointHandle>,
                                                  ContextResources) const;
    [[nodiscard]] context::ReleasePlan<ContextRelease, ContextResources>
        plan_releases(std::span<const CheckpointHandle>, std::span<const CheckpointHandle>,
                      ContextResources) const;
    [[nodiscard]] bool start_demote(const ContextDemotion&);
    [[nodiscard]] PhysicalUsageSnapshot usage() const noexcept;
    bool prepare_demotion(context::ContextTransaction&, const Demotion&);

    std::shared_ptr<const ModelDefinition> definition;
    const SessionPlan plan;
    const ExecutionShape& shape;
    runtime::Executor<ExecutionTypes>* executor = nullptr;
    context::ContextDirectory directory;
    Checkpoints checkpoints;
    mutable context::RecoveryIndex<> recovery_index;
    std::array<std::unique_ptr<Sequence>, kMaximumConcurrency> sequences;
    std::array<std::uint64_t, kMaximumConcurrency> epochs{};
    std::vector<runtime::KvPoolDescriptor> pools;
    std::unique_ptr<ContextOperation> operation;
    std::unique_ptr<ComputeTransaction> compute;
    std::unique_ptr<Scoring> scoring;
    std::deque<SessionEvent> events;
    std::vector<runtime::ResidencyId> releases;
    std::optional<runtime::ExecutionTicket> release_ticket;
    std::uint64_t next_sequence       = 1;
    std::uint64_t identity_generation = 0;
};

[[nodiscard]] std::shared_ptr<const Identity> capture_identity(const Sequence&);
[[nodiscard]] PrefixShortlistKey key(const Identity&, std::uint32_t);

} // namespace ninfer::models::qwen4_exp::detail
