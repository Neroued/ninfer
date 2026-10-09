#pragma once
#include "models/qwen3_5/program/internal.h"

#include "core/arena.h"
#include "core/gdn_replay_records.h"
#include "core/host_kv_arena.h"
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/ops/sampling.h"
#include "core/decode_graph.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"

#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/program/storage/draft_context.h"
#include "models/qwen3_5/program/storage/host_kv_store.h"
#include "models/qwen3_5/program/storage/kv_address_space.h"
#include "models/qwen3_5/program/storage/state_store.h"
#include "models/qwen3_5/program/prefix_identity.h"
#include "core/host_context_arena.h"
#include "models/qwen3_5/execution/text.h"
#include "models/qwen3_5/execution/vision.h"
#include "models/qwen3_5/program/vision_prefill.h"

#include <algorithm>
#include <cstdint>
#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer::models::qwen3_5::detail {
using execution::dimension;
using PreparedPromptData = qwen3_5::PreparedPromptData;

struct PreparedCaptureBacking {
    PrefixShortlistDigests digests;
    std::vector<TokenId> ledger;
    qwen3_5::detail::ResidentPrefixIdentity prefix_identity;
};

struct CaptureGroup {
    std::shared_ptr<const PreparedCaptureBacking> identity;
    PrefixShortlistKey key;
    std::uint32_t frontier = 0;
    std::vector<runtime::CheckpointRole> roles;
};

enum class MtpBridgeMode : std::uint8_t {
    None,
    BeforeSuffix,
    AfterExactHit,
};

struct RequestBasePlanImpl {
    std::shared_ptr<const PreparedPromptData> prompt;
    runtime::RequestPlanSummary summary;
    qwen3_5::PreparedContextCache context_cache;
    ops::SamplingConfig sampling;
    std::shared_ptr<const VisionControlPlan> vision_control_plan;
    std::vector<CaptureGroup> capture_groups;
    std::shared_ptr<const PreparedCaptureBacking> capture_backing;
    PrefixShortlistDigests prefix_digests;
    std::uint32_t prefix_identity_tag = 0;
    bool allow_prefix_reuse           = false;

    [[nodiscard]] bool accepts_capture(std::uint32_t frontier) const noexcept;
    [[nodiscard]] CaptureGroup capture_group(std::uint32_t frontier) const;
};
enum class PendingKind : std::uint8_t { None, Begin, Ordinary, Speculative };

struct PendingCandidate {
    PendingKind kind            = PendingKind::None;
    std::uint32_t base_E        = 0;
    std::uint32_t base_S        = 0;
    std::uint32_t prompt_tokens = 0;
    std::uint32_t produced      = 0;
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
    // The active KV addresses hold the physical reservation through this coverage.
    // These frontiers distinguish reconstructed work from committed new progress.
    UnitDemand coverage;
    std::uint32_t frontier    = 0;
    std::size_t ledger_tokens = 0;
};

struct SequenceKVBundle {
    KVAddressSpaceHandle text;
    std::optional<KVAddressSpaceHandle> backend;
};

// One mutable KV directory shared by the internal recovery points of a private history.
// Independent public/branch views own a different history and share only physical prefix pages.
struct KVHistory {
    ProgramImpl* owner = nullptr;
    KVAddressSpaceHandle text;
    std::optional<KVAddressSpaceHandle> backend;
    ~KVHistory();
};

struct DecodeGraphProfile {
    std::uint32_t batch_size             = 1;
    std::uint32_t min_execution_frontier = 0;
    std::uint32_t max_execution_frontier = 0;
    std::uint32_t topology_class         = 0;
    DecodeGraphDefinition definition;
};

struct DecodeGraphTopology {
    std::uint32_t topology_class = 0;
    DecodeGraphExecutable executable;
    std::optional<std::size_t> installed_profile;
};

struct DecodeGraphFamily {
    std::vector<DecodeGraphProfile> profiles;
    std::vector<DecodeGraphTopology> topologies;
    DecodeGraphProfile& select(std::uint32_t batch_size, std::uint32_t frontier);
    DecodeGraphExecutable& install(DecodeGraphProfile& profile);
};

// A Forward stage publishes these IDs before target execution finishes. Program owns the
// pinned storage and external event together; both outlive the graphs that reference them.
struct DFlashDraftHandoff {
    PinnedHostBuffer ids;
    CudaCompletionEvent ready;

    DFlashDraftHandoff(const DeviceContext& device, std::size_t count)
        : ids(count * sizeof(TokenId)), ready(device) {}

    [[nodiscard]] std::span<TokenId> tokens() const noexcept {
        return {static_cast<TokenId*>(ids.data()), ids.size() / sizeof(TokenId)};
    }
};

struct SequenceState {
    std::shared_ptr<KVHistory> kv;
    ActiveStateBinding state;
    Tensor tail_hidden;
    std::uint32_t lane = 0;

    std::uint32_t execution_frontier = 0;
    std::uint32_t ledger_frontier    = 0;
    std::vector<TokenId> ledger;
    qwen3_5::detail::ResidentPrefixIdentity prefix_identity;
    qwen3_5::detail::PrefixShortlistDigests prefix_digests;
    std::int32_t rope_delta               = 0;
    std::uint32_t text_kv_valid           = 0;
    std::uint32_t mtp_kv_valid            = 0;
    std::uint32_t dflash_context_frontier = 0;
    std::array<TokenId, qwen3_5::kMtpDecodeMaximumDrafts> mtp_drafts{};
    std::uint32_t mtp_draft_count = 0;
    bool tail_hidden_valid        = false;
    bool endpoint_valid           = false;
};

struct CaptureReservation {
    ProgramImpl* owner = nullptr;
    std::vector<std::pair<CheckpointHandle, runtime::CheckpointRole>> points;
    StateImageHandle state;
    std::optional<HostContextAllocation> host;
    std::optional<DeviceKVPageReservation> main_tail;
    std::optional<DeviceKVPageReservation> backend_tail;
    std::uint32_t frontier = 0;
    ~CaptureReservation();
};

struct RequestControl {
    Lifecycle lifecycle = Lifecycle::Empty;
    PendingCandidate pending;
    ops::SamplingConfig sampling_host;
    GenerationTimings timings;
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
    bool capture_pending     = false;

    struct Prefill {
        PreparedPromptData prompt;
        std::optional<VisionPrefillPlan> vision_plan;
        std::unique_ptr<execution::VisionPrefillSession> vision;
        std::uint32_t base               = 0;
        std::uint32_t cursor             = 0;
        std::uint32_t prompt_tokens      = 0;
        std::uint32_t initial_mtp_extent = 0;
        double elapsed_seconds           = 0.0;
        double retired_vision_seconds    = 0.0;
        bool prepare_mtp                 = false;
        PrefixReusePath reuse            = PrefixReusePath::Root;
        MtpBridgeMode mtp_bridge         = MtpBridgeMode::None;
    };

    std::optional<Prefill> prefill;
    std::optional<Prefill> replay;
};

struct CheckpointState {
    std::shared_ptr<KVHistory> kv;
    runtime::CheckpointRole role = runtime::CheckpointRole::Continuation;
    StateImageHandle state;
    std::shared_ptr<const PreparedCaptureBacking> identity;
    PrefixShortlistKey key;
    std::uint32_t frontier         = 0;
    std::uint32_t backend_frontier = 0;
    std::int32_t rope_delta        = 0;
    bool tail_hidden_valid         = false;
};

struct CheckpointSlot {
    std::optional<CheckpointState> value;
    std::uint64_t generation = 1;
    std::uint32_t pins       = 0;
    bool reserved            = false;
};

struct ResumeStateImpl {
    ProgramImpl* owner = nullptr;
    std::optional<CheckpointHandle> snapshot;
    SequenceState sequence;
    RequestControl control;
    std::uint32_t frontier = 0;
    ~ResumeStateImpl();
};

class ProgramImpl {
public:
    ProgramImpl(const execution::Parameters&, const SequencePlanImpl&, DeviceContext&,
                const StartupObserver&);
    ~ProgramImpl() noexcept;
    [[nodiscard]] RequestBasePlan plan_request(PreparedPromptData&&,
                                               const runtime::ResolvedExecutionOptions&);
    [[nodiscard]] std::vector<float> causal_score(PreparedPromptData&&, std::uint32_t first_target);
    [[nodiscard]] std::optional<SourceCandidate>
    inspect_source(const RequestBasePlan&, std::optional<CheckpointHandle>, bool = false,
                   std::span<const CheckpointHandle> = {},
                   std::span<const CheckpointHandle> = {}) const;
    [[nodiscard]] PrefixShortlistKey checkpoint_key(CheckpointHandle, std::uint32_t frontier) const;
    [[nodiscard]] runtime::ContextResourceUsage
        checkpoint_footprint(std::span<const CheckpointHandle>) const;
    [[nodiscard]] CheckpointSummary checkpoint_summary(CheckpointHandle) const;
    [[nodiscard]] CheckpointMetadata checkpoint_metadata(CheckpointHandle) const;
    [[nodiscard]] bool checkpoint_matches(CheckpointHandle, const RequestBasePlan&) const;
    [[nodiscard]] std::uint32_t checkpoint_recovery_frontier(CheckpointHandle,
                                                             const RequestBasePlan&,
                                                             std::uint32_t target) const;
    [[nodiscard]] std::uint64_t checkpoint_recovery_loss(std::span<const CheckpointHandle>,
                                                         std::span<const CheckpointHandle>) const;
    [[nodiscard]] bool can_release_checkpoint(CheckpointHandle) const noexcept;
    [[nodiscard]] std::optional<runtime::ContextResourceUsage>
        checkpoint_release_resources(std::span<const CheckpointHandle>,
                                     runtime::ContextResourceUsage) const;
    [[nodiscard]] bool release_checkpoint(CheckpointHandle) noexcept;
    void refresh_history_requirements(const std::shared_ptr<KVHistory>&, bool trim_unused = false);
    [[nodiscard]] bool revoke_snapshot(ResumeState&) noexcept;
    [[nodiscard]] runtime::ContextResourceUsage snapshot_resources(const ResumeState& paused) const;
    // Exact Host bytes freed by deleting this fixed set, with physical sharing counted once.
    [[nodiscard]] std::size_t
    host_bytes_released(std::span<const CheckpointHandle> checkpoints) const;
    [[nodiscard]] std::optional<std::size_t> pause_host_bytes(SequenceHandle sequence) const;
    [[nodiscard]] std::size_t
    release_redundant_host(std::span<const CheckpointHandle> excluded,
                           std::optional<SequenceHandle> pending_backup = std::nullopt);
    [[nodiscard]] runtime::ResourceReservation reserve_units(std::span<const ExecutionUnit>);
    [[nodiscard]] bool reclaim_capture_reservation(runtime::ContextResourceUsage shortage);
    void release_units(std::span<const SequenceHandle>) noexcept;
    [[nodiscard]] BindingReservation start_binding(const RequestBasePlan&, runtime::LaneId,
                                                   const SourceCandidate&, ResumeState*,
                                                   ExecutionUnitKind, std::uint32_t);
    [[nodiscard]] bool start_capture(SequenceHandle);
    [[nodiscard]] bool capture_is_input(SequenceHandle) const;
    [[nodiscard]] std::optional<CapturePreparation> prepare_capture(SequenceHandle);
    void skip_capture(SequenceHandle);
    [[nodiscard]] ContextReclaimPlan plan_reclaim(std::span<const CheckpointHandle>,
                                                  std::span<const CheckpointHandle>,
                                                  runtime::ContextResourceUsage) const;
    [[nodiscard]] std::vector<ContextRelease> plan_releases(std::span<const CheckpointHandle>,
                                                            std::span<const CheckpointHandle>,
                                                            runtime::ContextResourceUsage) const;
    [[nodiscard]] bool start_demote(const ContextDemotion&);
    [[nodiscard]] bool start_pause(SequenceHandle, bool save_snapshot, runtime::ExecutionTiming*);
    [[nodiscard]] ContextProgress poll_context(runtime::CancellationFlagView);

    [[nodiscard]] bool has_context_transaction() const noexcept {
        return context_transaction_.has_value();
    }

    [[nodiscard]] bool context_blocks(SequenceHandle) const noexcept;
    [[nodiscard]] bool recovery_pending(SequenceHandle) const noexcept;

    [[nodiscard]] PrefillProgress advance_prefill(SequenceHandle, runtime::ExecutionTiming*,
                                                  runtime::TokenMaskProvider*);
    [[nodiscard]] BranchScoreResult score_branches(std::span<const BranchScoreRow>,
                                                   const PreparationControl&);
    bool release_branch_cache();
    [[nodiscard]] ReplayProgress advance_replay(SequenceHandle, runtime::ExecutionTiming*);
    [[nodiscard]] PendingBatch decode(std::span<const SequenceHandle>,
                                      std::span<const runtime::RoundBudget>,
                                      runtime::ExecutionTiming*, runtime::TokenMaskProvider*);
    [[nodiscard]] runtime::ExecutionTiming
    append_forced_tokens(std::span<const SequenceHandle>, std::span<const TokenId>, std::uint32_t,
                         std::span<const std::optional<std::uint32_t>>, runtime::ExecutionTiming*);
    [[nodiscard]] CommitResult commit(PendingBatch&&, std::span<const runtime::CommitDecision>,
                                      runtime::CommitObservation, runtime::ExecutionTiming*);
    [[nodiscard]] DiscardResult abort_pending(PendingBatch&&) noexcept;
    [[nodiscard]] FinishResult finish(SequenceHandle) noexcept;
    [[nodiscard]] AbortResult abort(SequenceHandle) noexcept;
    void fail_all_cleanup() noexcept;
    [[nodiscard]] PhysicalUsageSnapshot physical_usage() const noexcept;
    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    void reset_memory_peaks() noexcept;

    const execution::Parameters& parameters;
    DeviceContext& device;
    const std::uint32_t capacity;
    const std::uint32_t kv_capacity;
    const std::uint32_t max_concurrency;
    const ContextCacheOptions context_cache;
    const std::uint32_t prefill_chunk;
    const std::uint32_t draft_window;
    const SpeculativeBackend speculative_backend;
    const KvCacheStorage kv_storage;
    const ProposalHead proposal_head;
    const bool vision_enabled;
    const bool use_cuda_graph;
    const bool causal_scoring;
    const bool decision_scoring;
    const std::size_t kv_payload_bytes;
    const std::size_t graph_allowance_bytes;
    const WorkspacePlan workspace_plan;

    DeviceArena persistent;
    DeviceArena workspace_storage;
    WorkspaceArena work;
    std::unique_ptr<qwen3_5::DecoderState> decoder;
    std::unique_ptr<HostContextArena> host_context_arena;
    std::unique_ptr<HostKVArena> host_kv_arena;
    std::unique_ptr<LogicalKVPageStore> text_kv_pages;
    std::unique_ptr<KVAddressSpaceStore> text_kv_addresses;
    std::unique_ptr<LogicalKVPageStore> backend_kv_pages;
    std::unique_ptr<KVAddressSpaceStore> backend_kv_addresses;
    std::unique_ptr<HostKVExtentStore> host_kv_extents;
    std::size_t text_host_kv_page_stride    = 0;
    std::size_t backend_host_kv_page_stride = 0;
    std::unique_ptr<qwen3_5::StateImageDevicePool> state_images;
    std::unique_ptr<qwen3_5::HostStatePool> host_state_images;
    std::unique_ptr<StateImageStore> state_store;
    std::optional<GdnReplayRecords> replay_records;
    std::optional<ops::GdnReplayFoldPlan> replay_fold;
    std::optional<DFlashPersistentState> dflash;
    qwen3_5::RoundState io;
    Tensor prefill_hidden;
    std::optional<Tensor> score_hidden;
    std::unique_ptr<BranchStorage> branches;
    Tensor sampling_config;
    Tensor grammar_masks_device;
    std::optional<PinnedHostBuffer> grammar_masks_host;
    std::optional<DFlashDraftHandoff> dflash_draft_handoff;
    std::array<std::uint32_t, kMaximumConcurrency> grammar_dead_positions{};
    ops::SamplingMask bind_grammar_mask(runtime::TokenMaskProvider*, std::size_t row);
    ops::SamplingMask fill_grammar_mask(runtime::TokenMaskProvider*, std::size_t row,
                                        std::span<const TokenId> drafts);
    Tensor token_counts;

    VisionHandoffState vision_handoff;
    std::array<SequenceState, kMaximumConcurrency> sequences;
    std::array<RequestControl, kMaximumConcurrency> requests;
    std::array<std::uint64_t, kMaximumConcurrency> lane_epochs{};
    std::vector<CheckpointSlot> checkpoints;

    std::optional<PinnedHostBuffer> round_host;
    std::optional<PinnedHostBuffer> score_logprobs_host;
    TokenId* host_tokens = nullptr;
    std::optional<PinnedHostBuffer> ordinary_host;
    qwen3_5::OrdinaryDecodeIngress* ordinary_host_ingress = nullptr;
    qwen3_5::OrdinaryDecodeEgress* ordinary_host_egress   = nullptr;
    std::optional<PinnedHostBuffer> mtp_host;
    qwen3_5::MtpDecodeIngress* mtp_host_ingress = nullptr;
    qwen3_5::MtpDecodeEgress* mtp_host_egress   = nullptr;
    std::optional<PinnedHostBuffer> dflash_host;
    qwen3_5::DFlashDecodeIngress* dflash_host_ingress          = nullptr;
    qwen3_5::DFlashDecodeEgress* dflash_host_egress            = nullptr;
    qwen3_5::DFlashPrefillIngress* dflash_prefill_host_ingress = nullptr;

    std::size_t workspace_logical_peak_bytes = 0;
    std::size_t vision_handoff_peak_bytes    = 0;

    struct PendingTransaction {
        std::uint64_t id = 0;
        std::array<std::uint32_t, kMaximumConcurrency> lanes{};
        std::array<std::uint64_t, kMaximumConcurrency> epochs{};
        std::size_t size = 0;
    };

    std::optional<PendingTransaction> pending_transaction_;
    std::uint64_t next_transaction_id_ = 1;

    struct KVTransfer {
        LogicalKVPageStore* pages              = nullptr;
        runtime::ContextResourceClass resource = runtime::ContextResourceClass::MainKV;
        std::vector<LogicalKVPageHandle> logical;
        std::vector<DeviceKVPageHandle> physical;
        std::optional<HostKVExtentReservation> host_destination;
        std::optional<DeviceKVPageReservation> device_reservation;
        bool restore     = false;
        bool drop_device = false;
    };

    struct ContextTransaction {
        ContextOperationKind kind = ContextOperationKind::Bind;
        std::uint32_t lane        = 0;
        std::uint64_t epoch       = 0;
        std::optional<CheckpointHandle> source;
        std::vector<std::pair<CheckpointHandle, runtime::CheckpointRole>> capture_points;
        std::vector<CheckpointHandle> carried_points;
        std::vector<CheckpointHandle> carried_pins;
        std::vector<std::pair<CheckpointHandle, CheckpointHandle>> carried_clones;
        std::vector<CheckpointHandle> retired_points;
        std::shared_ptr<const RequestBasePlanImpl> base;
        ResumeState* resume  = nullptr;
        bool resume_snapshot = false;
        std::optional<ResumeState> paused;
        std::optional<StateImageTransfer> state_transfer;
        std::vector<KVTransfer> kv_transfers;
        std::optional<KVPrefixForkReservation> text_fork;
        std::optional<KVPrefixForkReservation> backend_fork;
        std::optional<KVActivationReservation> text_activation;
        std::optional<KVActivationReservation> backend_activation;
        std::optional<KVActivePrefixViewReservation> text_view;
        std::optional<KVActivePrefixViewReservation> backend_view;
        std::optional<StateImageHandle> reserved_state;
        std::optional<DeviceKVPageReservation> main_growth;
        std::optional<DeviceKVPageReservation> backend_growth;
        bool binding_prepared          = false;
        std::uint32_t reuse_frontier   = 0;
        std::uint32_t backend_frontier = 0;
        UnitDemand first_unit;
        UnitDemand reservation_demand;
        std::optional<RecoveryPermit> recovery;
        std::optional<SequenceKVBundle> reserved_kv;
        std::shared_ptr<KVHistory> binding_history;
        std::shared_ptr<KVHistory> source_history;
        std::vector<runtime::ContextTransferRequirement> transfers;
        std::vector<runtime::ContextTransferObservation> observations;
        runtime::ContextOperationCounts operations;
        bool submitted             = false;
        bool preserve_state_device = true;
        bool consume_source        = false;
        bool take_private          = false;
        bool split_state           = false;
        bool backup_state          = false;
        bool adopted               = false;
        bool borrow_text           = false;
        bool borrow_backend        = false;
        bool borrow_state          = false;
        bool source_tail_hidden    = false;
    };

    std::optional<ContextTransaction> context_transaction_;
    CudaCompletionEvent context_source_ready_;
    CudaCompletionEvent context_completion_;
    std::array<CudaEventTimer, 3> context_transfer_timers_;
    CudaEventTimer prefill_gpu_timer_;

    // Captured transfers and external events reference the buffers and events declared above.
    // Families are destroyed first, including when startup throws.
    DecodeGraphFamily ordinary_graphs;
    DecodeGraphFamily speculative_forward_graphs;
    DecodeGraphFamily speculative_finish_graphs;

    [[nodiscard]] std::uint32_t initial_mtp_extent(const RequestBasePlanImpl&) const;
    [[nodiscard]] UnitDemand prefill_unit(std::uint32_t prompt, std::uint32_t cursor,
                                          std::uint32_t mtp_extent) const;
    [[nodiscard]] UnitDemand next_unit(const SequenceState&, const RequestControl&,
                                       ExecutionUnitKind, std::uint32_t tokens) const;
    void require_unit(std::uint32_t lane, ExecutionUnitKind, std::uint32_t tokens = 0) const;
    void settle_unit(std::uint32_t lane) noexcept;
    [[nodiscard]] bool valid_sequence(SequenceHandle) const noexcept;
    [[nodiscard]] bool valid_pending(const PendingBatch&) const noexcept;
    [[nodiscard]] bool valid_checkpoint(CheckpointHandle) const noexcept;
    [[nodiscard]] CheckpointState& checkpoint(CheckpointHandle);
    [[nodiscard]] const CheckpointState& checkpoint(CheckpointHandle) const;
    [[nodiscard]] std::optional<CheckpointHandle> reserve_checkpoint();
    [[nodiscard]] std::optional<CheckpointHandle> detach_checkpoint(SequenceState&);
    void abort_context() noexcept;
    void enqueue_context_transfers(ContextTransaction&);
    void enqueue_state_backup(ContextTransaction&);
    void copy_local_for_context(ContextTransaction&, std::int32_t source, std::int32_t destination);
    void copy_context_tail(ContextTransaction&, LogicalKVPageStore&, DeviceKVPageHandle source,
                           DeviceKVPageHandle destination, runtime::ContextResourceClass);
    void publish_context_transfers(ContextTransaction&);
    [[nodiscard]] bool prepare_backup(ContextTransaction&, CheckpointState&, bool shared_device);
    void install_binding(ContextTransaction&);
    void prepare_binding(ContextTransaction&);
    void complete_binding(ContextTransaction&, ContextProgress&);
    void publish_capture(ContextTransaction&);
    [[nodiscard]] bool reserve_capture_destination(std::uint32_t lane, std::uint32_t frontier);
    void prepare_capture_boundary(std::uint32_t lane);
    [[nodiscard]] ResumeState complete_pause(ContextTransaction&);
    void install_resume_sampling(SequenceState&, RequestControl&);
    void initialize_prefill(std::uint32_t lane, std::uint32_t base);
    void initialize_captures(std::uint32_t lane, std::uint32_t from, std::uint32_t through);
    [[nodiscard]] SequenceHandle sequence_handle(std::uint32_t lane) const noexcept;
    void invalidate_lane(std::uint32_t lane) noexcept;
    [[nodiscard]] SequenceState& active_sequence(std::uint32_t lane);
    [[nodiscard]] const SequenceState& active_sequence(std::uint32_t lane) const;
    void clear_lane(SequenceState&, RequestControl&) noexcept;
    void clear_execution_failure_lanes(std::span<const std::uint32_t>) noexcept;
    void ordered_reset(SequenceState&);
    void refresh_state_views(SequenceState&);
    [[nodiscard]] StateImageSelectors state_selectors(const SequenceState&) const;
    void settle_state_fork(SequenceState&);
    void release_sequence_state(SequenceState&) noexcept;
    void release_sequence_kv(SequenceState&) noexcept;
    void ensure_sequence_kv_mapped(SequenceState&, std::uint32_t main, std::uint32_t backend = 0);
    void trim_sequence_kv(SequenceState&, std::uint32_t main, std::uint32_t backend = 0);
    void commit_sequence_kv(SequenceState&, std::uint32_t main, std::uint32_t backend = 0);
    [[nodiscard]] PagedKVCache* backend_kv_cache() noexcept;
    [[nodiscard]] const PagedKVCache* backend_kv_cache() const noexcept;
    [[nodiscard]] std::uint32_t backend_kv_valid(const SequenceState&) const noexcept;
    [[nodiscard]] PagedKVCacheView text_kv_view(const SequenceState&) const;
    [[nodiscard]] PagedKVCacheView mtp_kv_view(const SequenceState&) const;
    [[nodiscard]] PrefillProgress wrap_prefill(std::uint32_t lane, runtime::PrefillStepResult);
    [[nodiscard]] PendingBatch wrap_pending(std::span<const std::uint32_t>,
                                            const runtime::BatchedGeneratedRound&);
    [[nodiscard]] runtime::PrefillStepResult advance_prefill_raw(std::uint32_t,
                                                                 runtime::ExecutionTiming*);
    [[nodiscard]] runtime::ExecutionTiming resolve_prefill_raw(std::uint32_t, bool,
                                                               runtime::ExecutionTiming*);
    [[nodiscard]] runtime::ExecutionTiming
    resolve_pending_raw(std::span<const std::uint32_t>, std::span<const std::uint32_t>,
                        std::span<const std::uint8_t>, std::span<const std::uint8_t>,
                        std::span<const std::optional<std::uint32_t>>, runtime::ExecutionTiming*);
    void start_context_transfer_timer(runtime::ContextResourceClass);
    void stop_context_transfer_timer(runtime::ContextResourceClass);
    [[nodiscard]] runtime::ContextTransferObservation
        context_transfer_observation(runtime::ContextResourceClass,
                                     runtime::ContextTransferDirection, TransferWork, std::uint32_t,
                                     std::uint64_t) const;
    [[nodiscard]] runtime::BatchedGeneratedRound decode_raw(std::span<const std::uint32_t>,
                                                            std::span<const runtime::RoundBudget>,
                                                            runtime::ExecutionTiming*,
                                                            runtime::TokenMaskProvider*);
    void prepare_graphs();
    void install_sampling(SequenceState& sequence, RequestControl& request,
                          const ops::SamplingConfig& config);
    void set_device_i32(Tensor& tensor, std::int32_t value);
    void copy_tail(SequenceState& sequence, const Tensor& source);
    void copy_round_token();
    void
    commit_generated_prefix_identity(SequenceState& sequence, std::uint32_t base_ledger_frontier,
                                     std::span<const TokenId> accepted_tokens,
                                     std::optional<std::uint32_t> prefix_execution_split_after);
    [[nodiscard]] runtime::ExecutionTiming
    resolve_non_speculative_pending(SequenceState& sequence, RequestControl& request,
                                    std::uint32_t accepted_tokens, bool terminal,
                                    std::optional<std::uint32_t> prefix_execution_split_after,
                                    runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] runtime::PrefillStepResult
    advance_prefill(SequenceState& sequence, RequestControl& request,
                    runtime::ExecutionTiming* failed_timing);
    void enqueue_dflash_context_append(std::span<const std::uint32_t> lanes,
                                       std::span<const std::uint32_t> starts,
                                       std::span<const std::uint32_t> counts);
    void validate_licensed_tokens(std::span<const TokenId> tokens) const;
    void mark_workspace_usage(std::size_t phase_bytes) noexcept;
    [[nodiscard]] runtime::BatchedGeneratedRound decode_ordinary_batch(
        std::span<const std::uint32_t> lanes, std::span<const runtime::RoundBudget> budgets,
        runtime::ExecutionTiming* failed_timing, runtime::TokenMaskProvider* masks);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_mtp_batch(std::span<const std::uint32_t> lanes,
                     std::span<const runtime::RoundBudget> budgets,
                     runtime::ExecutionTiming* failed_timing, runtime::TokenMaskProvider* masks);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_dflash_batch(std::span<const std::uint32_t> lanes,
                        std::span<const runtime::RoundBudget> budgets,
                        runtime::ExecutionTiming* failed_timing, runtime::TokenMaskProvider* masks);
};
} // namespace ninfer::models::qwen3_5::detail
