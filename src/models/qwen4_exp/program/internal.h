#pragma once

#include "models/qwen4_exp/program/program.h"
#include "models/qwen4_exp/program/ple_pipeline.h"
#include "models/qwen4_exp/program/graphs.h"
#include "models/qwen4_exp/execution/text.h"
#include "models/common/context/snapshot.h"
#include "models/common/qwen/prefix_identity.h"
#include "models/common/qwen/vision/prefill.h"
#include "ninfer/ops/sampling.h"

#include <chrono>

namespace ninfer::models::qwen4_exp::detail {
using Image = context::StateImageStore::Handle;
using Clock = std::chrono::steady_clock;

inline double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

struct CapturePoint {
    std::uint32_t frontier;
    std::vector<runtime::CheckpointRole> roles;
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
    std::optional<qwen::VisionPrefillPlan> vision;
    ops::SamplingConfig sampling;
    bool reuse = true;

    [[nodiscard]] bool can_checkpoint(std::uint32_t frontier) const;
};

using Snapshot = context::ContextSnapshot<ContinuationMetadata, Identity>;

struct Checkpoint {
    Snapshot snapshot;
    runtime::CheckpointRole role;

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
    context::StateImageStore* states;
    Checkpoints* checkpoints;
    Image destination;
    std::optional<HostContextAllocation> host;
    std::vector<std::pair<CheckpointHandle, runtime::CheckpointRole>> records;
    ~CaptureReservation();
};

struct Sequence {
    std::uint64_t id, epoch;
    std::uint32_t lane;
    std::shared_ptr<const RequestPlan> request;
    Identity identity;
    Continuation continuation;
    Image write_destination;
    std::array<context::KVHistory, 2> histories;
    Phase phase = Phase::Prefill, resume_phase = Phase::Active;
    std::uint32_t reused = 0, replay_target = 0, recovery_peak = 0;
    std::optional<ExecutionUnit> permit;
    std::uint32_t permitted_frontier = 0;
    std::size_t next_capture         = 0;
    std::unique_ptr<CaptureReservation> capture;
    bool capture_pending = false;
    std::unique_ptr<qwen::VisionPrefillSession> vision;
    double observed_vision_seconds = 0;
    GenerationTimings timings;
    SpeculativeStats speculative;
    Clock::time_point started = Clock::now();
    std::optional<Clock::time_point> first_output;
};

struct MtpInputRow {
    std::uint32_t base;
    std::span<const TokenId> ids;
};

struct VerifyRound {
    std::uint32_t width;
    std::vector<std::uint32_t> bases, extents;
    std::vector<TokenId> inputs;
    execution::BatchView batch;
};

struct Suspended {
    ProgramImpl* owner;
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

    const ProgramImpl* owner          = nullptr;
    std::uint64_t checkpoint_revision = 0;
    std::vector<Image> images;
    std::vector<std::uint64_t> image_epochs;
    std::array<std::vector<context::LogicalKVPageHandle>, 2> pages;
    std::array<std::vector<PageVersion>, 2> page_versions;
    std::vector<CheckpointHandle> records;

    [[nodiscard]] bool empty() const {
        return images.empty() && pages[0].empty() && pages[1].empty();
    }
};

struct Reclaim {
    const ProgramImpl* owner;
    ContextResources shortage;
};

struct Release {
    const ProgramImpl* owner;
    std::array<std::vector<context::LogicalKVPageHandle>, 2> pages;
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
    context::MigrationBatch migration;
    Action action;

    ContextOperation(Resources& resources, Action value)
        : migration(*resources.migration, resources.host_extents.get()), action(std::move(value)) {}

    [[nodiscard]] ContextOperationKind kind() const {
        return std::visit([](const auto& value) { return value.kind; }, action);
    }
};

class ProgramImpl {
public:
    ProgramImpl(const execution::Parameters&, ExecutionPlan, DeviceContext&);
    ~ProgramImpl();

    [[nodiscard]] bool valid(SequenceHandle) const noexcept;
    [[nodiscard]] Sequence& sequence(SequenceHandle) const;
    [[nodiscard]] SequenceHandle handle(const Sequence&) const;
    void clear(Sequence&) noexcept;
    void cleanup() noexcept;
    void initialize_graphs();
    void release_graphs();
    void draft_forward(std::uint32_t step, std::int32_t batch, std::uint32_t visible);
    void save_draft(std::uint32_t step, std::int32_t batch);
    void sample_forward(std::int32_t batch, bool first_token);
    void accept_forward(std::int32_t batch);
    [[nodiscard]] context::StateImageSelectors selectors(const Sequence&) const;
    void complete_write(Sequence&);
    [[nodiscard]] std::uint32_t window_end(const Sequence&, std::uint32_t begin,
                                           std::uint32_t end) const;
    [[nodiscard]] std::uint32_t unit_end(const Sequence&, ExecutionUnitKind, std::uint32_t) const;
    void settle(Sequence&);
    [[nodiscard]] Tensor device_tensor(const TensorRegion&);
    [[nodiscard]] Tensor host_tensor(const TensorRegion&);
    [[nodiscard]] Tensor packed_tensor(const TensorRegion&, std::initializer_list<std::int32_t>);
    [[nodiscard]] execution::BatchView mtp_inputs(const MtpBatchLayout&, std::span<Sequence* const>,
                                                  std::uint32_t tokens,
                                                  std::span<const MtpInputRow>);
    void align_mtp(std::span<Sequence* const>, std::uint32_t tokens, const Tensor& previous,
                   const Tensor& conditioning, std::optional<execution::VisualEmbeddingView>);
    [[nodiscard]] ops::QsaTailStateView mtp_working_tail(std::uint32_t slot, std::uint32_t batch);
    [[nodiscard]] PendingBatch mtp_decode(std::span<Sequence* const>,
                                          std::span<const runtime::RoundBudget>,
                                          runtime::TokenMaskProvider*,
                                          runtime::ExecutionTimingRecorder&);
    void commit_verify(std::span<const SequenceHandle>, std::span<const runtime::CommitDecision>,
                       runtime::ExecutionTimingRecorder&);
    [[nodiscard]] Tensor forward(std::span<Sequence* const>, std::uint32_t tokens, bool sequence,
                                 bool logits = false);
    void adopt_forward(std::span<Sequence* const>, std::uint32_t tokens);
    [[nodiscard]] PendingBatch sample(std::span<Sequence* const>, const Tensor&, bool first,
                                      runtime::TokenMaskProvider*,
                                      runtime::ExecutionTimingRecorder&);
    void restore_counts(Sequence&);
    void increment_counts(Sequence&, std::span<const TokenId>);
    [[nodiscard]] qwen::VisionPrefillPlan plan_vision(const qwen::PreparedPromptData&) const;
    [[nodiscard]] std::optional<execution::VisualEmbeddingView>
    prepare_vision(Sequence&, std::uint32_t tokens);
    double collect_vision_timing(Sequence&);
    void retire_vision(Sequence&) noexcept;
    [[nodiscard]] std::size_t active_vision_bytes() const noexcept;
    [[nodiscard]] std::optional<CheckpointHandle> detach(Sequence&, runtime::CheckpointRole);
    [[nodiscard]] std::vector<context::LogicalKVPageHandle>
    checkpoint_pages(std::span<const CheckpointHandle>, std::uint32_t pool) const;
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
    bool prepare_demotion(context::MigrationBatch&, const Demotion&);
    void cancel_operation() noexcept;

    const execution::Parameters& parameters;
    DeviceContext& device;
    CudaExecutionView cuda;
    Resources resources;
    PlePipeline ple;
    const ExecutionShape& shape;
    std::unique_ptr<Graphs> graphs;
    std::optional<GraphMemorySummary> graph_memory;
    Checkpoints checkpoints;
    mutable context::RecoveryIndex<> recovery_index;
    CudaEventTimer prefill_timer;
    qwen::VisionHandoffState vision_handoff;
    std::size_t vision_handoff_peak_bytes = 0, vision_workspace_peak_bytes = 0;
    std::array<std::unique_ptr<Sequence>, kMaximumConcurrency> sequences;
    std::array<std::uint64_t, kMaximumConcurrency> epochs{};
    std::vector<runtime::KvPoolDescriptor> pools;
    std::unique_ptr<ContextOperation> operation;
    std::uint64_t next_sequence = 1, next_transaction = 1, pending_transaction = 0;
    std::vector<SequenceHandle> pending_members;
    std::array<bool, kMaximumConcurrency> pending_failed{};
    std::optional<VerifyRound> pending_verify;
};

[[nodiscard]] ops::SamplingConfig sampling_config(const ResolvedSamplingParameters&);
[[nodiscard]] std::shared_ptr<const Identity> capture_identity(const Sequence&);
[[nodiscard]] PrefixShortlistKey key(const Identity&, std::uint32_t);

} // namespace ninfer::models::qwen4_exp::detail
