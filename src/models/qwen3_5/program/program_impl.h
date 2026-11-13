#pragma once
#include "models/qwen3_5/program/internal.h"

#include "core/arena.h"
#include "core/elastic_arena.h"
#include "models/qwen3_5/program/tp2/program_tp2.h"
#include "models/qwen3_5/execution/tp2/gdn_gating_overlap.h"
#include "core/gdn_replay_records.h"
#include "core/host_kv_arena.h"
#include "core/disk_kv_cache.h"
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/ops/sampling.h"
#include "core/decode_graph.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"

#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/program/storage/draft_context.h"
#include "models/qwen3_5/program/storage/host_kv_store.h"
#include "models/qwen3_5/program/storage/kv_store.h"
#include "models/qwen3_5/program/storage/qwen4_elastic.h"
#include "models/qwen3_5/program/storage/state_store.h"
#include "models/qwen3_5/program/prefix_identity.h"
#include "models/qwen3_5/program/qwen4_ple_stager.h"
#include "models/qwen3_5/program/planning/resource_projection.h"
#include "models/qwen3_5/execution/text.h"
#include "models/qwen3_5/execution/qwen4_draft_head.h"
#include "models/qwen3_5/execution/qwen4_experts.h"
#include "models/qwen3_5/execution/qwen4_runtime.h"
#include "models/qwen3_5/execution/qwen4_tfdump.h"
#include "models/qwen3_5/execution/vision.h"
#include "models/qwen3_5/program/vision_prefill.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <array>
#include <functional>
#include <string>
#include <string_view>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

using PreparedPromptData    = qwen3_5::PreparedPromptData;
using RewriteCheckpointKind = qwen3_5::RewriteCheckpointKind;
using RewriteCheckpointSpec = qwen3_5::RewriteCheckpointSpec;

// Device KV is leased on demand: an active request holds a bounded window of its remaining
// output (EngineOptions::kv_lease_window_tokens) rather than the whole client budget, and extends
// it at a decode-round boundary; a full window is asked first, a step-sized extension when the
// pool cannot spare one.

[[nodiscard]] constexpr std::uint32_t kv_pages_for_tokens(std::uint32_t tokens) noexcept {
    return tokens == 0 ? 0U : 1U + (tokens - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

[[nodiscard]] constexpr std::uint32_t kv_tokens_for_pages(std::uint32_t pages) noexcept {
    return pages == 0 ? 0U : (pages - 1U) * static_cast<std::uint32_t>(kPagedKVPageSize) + 1U;
}

using ReusePath = ninfer::PrefixReusePath;

[[nodiscard]] constexpr bool is_rewrite_checkpoint_restore(ReusePath path) noexcept {
    return path == ReusePath::PrivateTurnClosure || path == ReusePath::PrivateResponseReplay;
}

[[nodiscard]] constexpr ReusePath restore_path(RewriteCheckpointKind kind) noexcept {
    return kind == RewriteCheckpointKind::TurnClosure ? ReusePath::PrivateTurnClosure
                                                      : ReusePath::PrivateResponseReplay;
}

[[nodiscard]] constexpr runtime::CheckpointKind
checkpoint_kind(RewriteCheckpointKind kind) noexcept {
    return kind == RewriteCheckpointKind::TurnClosure ? runtime::CheckpointKind::TurnClosure
                                                      : runtime::CheckpointKind::ResponseReplay;
}

enum class RewriteCheckpointDisposition : std::uint8_t {
    RetainExisting,
    ReplaceAtCommittedFrontier,
    DropOptional,
};

struct PreparedCaptureBacking {
    std::vector<TokenId> ledger;
    qwen3_5::detail::ResidentPrefixIdentity prefix_identity;
};

struct PreparedCaptureIdentity {
    std::shared_ptr<const PreparedCaptureBacking> backing;
    qwen3_5::PrefixShortlistKey shortlist_key;
    runtime::PrefillWork rebuild_work;

    [[nodiscard]] std::span<const TokenId> ledger() const noexcept {
        if (!backing || shortlist_key.frontier > backing->ledger.size()) { return {}; }
        return std::span<const TokenId>(backing->ledger).first(shortlist_key.frontier);
    }

    [[nodiscard]] const qwen3_5::detail::ResidentPrefixIdentity* prefix_identity() const noexcept {
        return backing ? &backing->prefix_identity : nullptr;
    }

    [[nodiscard]] bool prefix_equals(const PreparedCaptureIdentity& other) const {
        const std::span<const TokenId> left  = ledger();
        const std::span<const TokenId> right = other.ledger();
        const auto* left_identity            = prefix_identity();
        const auto* right_identity           = other.prefix_identity();
        return left_identity != nullptr && right_identity != nullptr &&
               left.size() == right.size() && std::equal(left.begin(), left.end(), right.begin()) &&
               left_identity->prefix_equals(*right_identity, left.size());
    }
};

struct CaptureGroup {
    std::shared_ptr<const PreparedCaptureIdentity> identity;
    std::optional<RewriteCheckpointKind> rewrite;
    std::uint32_t frontier                  = 0;
    std::uint32_t input_order               = 0;
    bool shared                             = false;
    bool long_anchor                        = false;
    SharedCandidateEvidence shared_evidence = SharedCandidateEvidence::None;
};

enum class MtpBridgeMode : std::uint8_t {
    None,
    BeforeSuffix,
    AfterExactHit,
};

} // namespace ninfer::models::qwen3_5::detail

namespace ninfer::models::qwen3_5::detail {

enum class PressureStateDecision : std::uint8_t {
    None,
    DropEndpointDeviceDuplicate,
    DemoteEndpointToHost,
    DropEndpointHostDuplicate,
    DropRewriteDeviceDuplicate,
    DemoteRewriteToHost,
    DropRewriteHostDuplicate,
    DropSharedDeviceDuplicate,
    DemoteSharedToHost,
    DropSharedHostDuplicate,
};

enum class PressureKVDecisionKind : std::uint8_t {
    None,
    DropDeviceDuplicate,
    DemoteToHost,
    DropHostDuplicate,
};

struct PressureKVDecision {
    std::uint32_t begin_page    = 0;
    std::uint32_t page_count    = 0;
    PressureKVDecisionKind kind = PressureKVDecisionKind::None;

    [[nodiscard]] friend constexpr bool operator==(PressureKVDecision,
                                                   PressureKVDecision) noexcept = default;
};

// One Program-private complete outcome for one eligible owner. It may combine State, Main KV,
// Backend KV, and checkpoint changes; common scheduling never observes these physical decisions.
struct PressureDecision {
    std::uint64_t id = 0;
    std::vector<PressureStateDecision> state_changes;
    std::vector<PressureKVDecision> main_kv_changes;
    std::vector<PressureKVDecision> backend_kv_changes;
    std::vector<runtime::CheckpointRef> dropped_checkpoints;
    PhysicalDelta checkpoint_drop_effect;
    PhysicalDelta effect;
    std::vector<runtime::ContextTransferRequirement> transfer_requirements;
    std::uint32_t checkpoint_drops = 0;
    bool evicts_continuation       = false;
    bool shared_owner              = false;

    [[nodiscard]] friend bool operator==(const PressureDecision&,
                                         const PressureDecision&) noexcept = default;
};

struct PressureCheckpointRecoveryProjection {
    runtime::PlanningOwnerId owner;
    runtime::CheckpointRef checkpoint;
    std::uint32_t alternative_offset = 0;
    std::uint32_t alternative_count  = 0;
    bool survives                    = true;
};

struct CaptureAssessmentImpl {
    PhysicalDemand demand;
    PhysicalDelta active_entitlement_delta;
    PhysicalResources capacity_preparation_removed;
};

struct RequestBasePlanImpl {
    runtime::RequestPlanSummary summary;
    detail::PhysicalDemand root_demand;
    runtime::PrefillWork root_rebuild_work;
    std::uint32_t root_rebuild_tail_begin = 0;
    qwen3_5::PreparedContextCache context_cache;
    ops::SamplingConfig sampling;
    std::uint32_t accepted_length_steps       = 0;
    std::uint32_t text_kv_page_entitlement    = 0;
    std::uint32_t backend_kv_page_entitlement = 0;
    std::shared_ptr<const qwen3_5::VisionControlPlan> vision_control_plan;
    std::optional<qwen3_5::RewriteCheckpointSpec> rewrite_checkpoint;
    std::vector<CaptureGroup> capture_groups;
    std::vector<CaptureGroup> shared_candidates;
    qwen3_5::detail::PrefixShortlistDigests prefix_digests;
    std::uint32_t prefix_identity_tag = 0;
    bool allow_prefix_reuse           = false;
};

// Program-owned physical planning state shared by request materialization and active capture.
// Request scheduling fields never enter this record, and capture never becomes an admission type.
struct ResourceCandidateState {
    runtime::RequestPlanSummary summary;
    runtime::IdentityMaterializationAssessment identity_assessment;
    runtime::ProgramResourceRevision planning_revision;
    // Pressure outcomes are canonicalized against the candidate's identity peak.  Composition
    // rewrites demand to the selected post-pressure peak, so regenerating an outcome from that
    // rewritten demand would compare it against a different problem at seal time.
    detail::PhysicalResources identity_pressure_deficit;
    // A structurally valid pressure target can still be blocked by Host extent geometry even when
    // aggregate free bytes are sufficient. Keep the blocked allocation work explicit so a child
    // target can release Host replicas instead of being mistaken for a structurally invalid node.
    std::size_t blocked_host_allocation_bytes = 0;
    detail::PhysicalDemand demand;
    // Resources released by consuming this private owner alone. Shared aliases are intentionally
    // absent; complete pressure targets settle their joint reference graph separately.
    detail::PhysicalResources source_resources;
    ReusePath reuse                                  = ReusePath::Root;
    std::uint32_t reuse_base                         = 0;
    RewriteCheckpointDisposition rewrite_disposition = RewriteCheckpointDisposition::DropOptional;
    bool has_source                                  = false;
    bool has_shared_source                           = false;
    std::optional<runtime::CheckpointRef> selected_checkpoint;
    std::uint32_t source_index             = 0;
    std::uint64_t source_generation        = 0;
    std::uint32_t shared_source_index      = 0;
    std::uint64_t shared_source_generation = 0;
    runtime::PrefillWork remaining_prefill_work;
    std::vector<runtime::ContextTransferRequirement> transfer_requirements;
    runtime::PrivateSourceMode source_mode = runtime::PrivateSourceMode::ConsumeToActive;
    detail::PhysicalResources active_optional_resources;
    bool state_fork_required          = false;
    bool text_prefix_fork_required    = false;
    bool backend_prefix_fork_required = false;
    bool needs_transfer               = false;
    std::vector<qwen3_5::detail::PressureDecision> pressure_options;
    std::vector<runtime::PlanningOwnerId> pressure_owner_ids;
    std::vector<std::uint32_t> pressure_indices;
    std::vector<std::uint64_t> pressure_generations;
    std::vector<qwen3_5::detail::PressureDecision> shared_pressure_options;
    std::vector<runtime::PlanningOwnerId> shared_pressure_owner_ids;
    std::vector<std::uint32_t> shared_pressure_indices;
    std::vector<std::uint64_t> shared_pressure_generations;
};

struct AdmissionCandidateImpl : ResourceCandidateState {
    MtpBridgeMode mtp_bridge = MtpBridgeMode::None;
    bool prepare_mtp         = false;
    std::optional<VisionPrefillPlan> vision;
    std::vector<CaptureGroup> capture_groups;
    std::vector<CaptureGroup> shared_candidates;
    ops::SamplingConfig sampling;
    std::uint32_t accepted_length_steps       = 0;
    std::uint32_t text_kv_page_entitlement    = 0;
    std::uint32_t backend_kv_page_entitlement = 0;
    runtime::LaneId destination{};
    std::uint64_t destination_epoch = 0;
    runtime::PrefillWork root_rebuild_work;
    std::uint32_t root_rebuild_tail_begin = 0;
    bool text_retained_tail_release       = false;
    bool backend_retained_tail_release    = false;
};

struct CapturePressureCandidateImpl : ResourceCandidateState {};

} // namespace ninfer::models::qwen3_5::detail

namespace ninfer::models::qwen3_5::detail {

using AdmissionCandidateImpl       = qwen3_5::detail::AdmissionCandidateImpl;
using CapturePressureCandidateImpl = qwen3_5::detail::CapturePressureCandidateImpl;
using ResourceCandidateState       = qwen3_5::detail::ResourceCandidateState;
using RequestBasePlanImpl          = qwen3_5::detail::RequestBasePlanImpl;
using CapturePressureCandidate     = qwen3_5::CapturePressureCandidate;

enum class PendingKind : std::uint8_t {
    None,
    Begin,
    Ordinary,
    Speculative,
};

struct PendingCandidate {
    PendingKind kind            = PendingKind::None;
    std::uint32_t base_E        = 0;
    std::uint32_t base_S        = 0;
    std::uint32_t prompt_tokens = 0;
    std::uint32_t produced      = 0;
};

enum class Lifecycle : std::uint8_t {
    Empty,
    Prefilling,
    Active,
    Pending,
    Finishable,
};

enum class ContinuationSlotRole : std::uint8_t {
    Free,
    ReservedMaterialization,
    Active,
    Catalogued,
};

struct ContinuationSlot {
    ContinuationSlotRole role = ContinuationSlotRole::Free;
    std::uint64_t generation  = 1;
};

struct RewriteCheckpoint {
    bool valid                 = false;
    RewriteCheckpointKind kind = RewriteCheckpointKind::TurnClosure;
    std::uint32_t frontier     = 0;
    runtime::PrefillWork rebuild_work;
};

struct LongAnchorCheckpoint {
    StateImageHandle state;
    std::uint32_t frontier = 0;
    std::uint32_t ordinal  = 0;
    runtime::PrefillWork rebuild_work;
};

struct SequenceKVBundle {
    KVAddressSpaceHandle text;
    std::optional<KVAddressSpaceHandle> backend;
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
};

// Target model continuation for one logical sequence. This state remains meaningful after the
// request which produced it has finished, so it is deliberately separate from request lifecycle,
// output, sampling, and round-control state.
struct SequenceState {
    std::optional<SequenceKVBundle> kv;
    ActiveStateBinding state;
    std::optional<StateImageHandle> rewrite_state;
    std::optional<StateImageHandle> reserved_state;
    Tensor tail_hidden;
    Tensor rewrite_checkpoint_hidden;
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
    // T41 calibration (NINFER_QWEN4_MTP_CALIB): the log-probabilities of the pending proposals.
    std::array<float, qwen3_5::kMtpDecodeMaximumDrafts> mtp_draft_logprobs{};
    bool tail_hidden_valid = false;
    bool endpoint_valid    = false;
    RewriteCheckpoint rewrite_checkpoint;
    std::vector<LongAnchorCheckpoint> long_anchors;
    std::vector<std::uint32_t> shared_prefix_references;
    runtime::PrefillWork rebuild_work;
    std::uint32_t rebuild_tail_begin = 0;
};

struct SharedPrefixState {
    std::optional<SequenceKVBundle> kv;
    StateImageHandle state;
    std::shared_ptr<const PreparedCaptureIdentity> identity;
    std::uint32_t frontier         = 0;
    std::uint32_t backend_frontier = 0;
    std::int32_t rope_delta        = 0;
    bool tail_hidden_valid         = false;
    runtime::PrefillWork rebuild_work;
    std::uint32_t active_references = 0;
};

enum class SharedPrefixSlotRole : std::uint8_t {
    Free,
    ReservedCapture,
    ReservedReplacement,
    Catalogued,
};

struct SharedPrefixSlot {
    SharedPrefixSlotRole role = SharedPrefixSlotRole::Free;
    std::uint64_t generation  = 1;
};

// Request/round control is not retained with a reusable SequenceState. A later concurrent Engine
// gives every occupied request slot its own instance of this state.
struct RequestControl {
    Lifecycle lifecycle = Lifecycle::Empty;
    PendingCandidate pending;
    ops::SamplingConfig sampling_host;
    GenerationTimings timings;
    SpeculativeStats speculative_stats;
    // Bound of speculative_stats.accepted_lengths, reserved at install; 0 records nothing.
    std::uint32_t accepted_length_steps = 0;
    detail::PhysicalResources active_resources;
    detail::PhysicalResources optional_resources;
    bool publish_continuation = true;
    // Device KV lease: the largest frontier the lease may ever cover (the request's own output
    // ceiling); whether it can no longer be extended; whether that is for lack of pool space, and
    // the entitlements its full and smallest growth steps asked for.
    std::uint32_t lease_ceiling = 0;
    bool lease_settled          = false;
    bool lease_space_limited    = false;
    DeviceKVPages lease_full_target;
    DeviceKVPages lease_minimum_target;

    struct Prefill {
        PreparedPromptData prompt;
        std::optional<VisionPrefillPlan> vision_plan;
        std::unique_ptr<execution::VisionPrefillSession> vision;
        std::vector<CaptureGroup> capture_groups;
        std::size_t next_capture            = 0;
        std::uint64_t pending_capture_offer = 0;
        std::uint32_t base                  = 0;
        std::uint32_t cursor                = 0;
        std::uint32_t prompt_tokens         = 0;
        std::uint32_t initial_mtp_extent    = 0;
        double elapsed_seconds              = 0.0;
        bool prepare_mtp                    = false;
        ReusePath reuse                     = ReusePath::Root;
        MtpBridgeMode mtp_bridge            = MtpBridgeMode::None;
        // Disk tier (v0.7.2): the resume was tried; the frontier it loaded (0: none). The MTP
        // bridge then runs at this frontier instead of `base` (which stays the admitted reuse).
        bool disk_attempted     = false;
        std::uint32_t disk_base = 0;
    };

    std::optional<Prefill> prefill;
};

class ProgramImpl {
public:
    struct PressureRecoveryScratch {
        struct StatePlacement {
            StateImageHandle state;
            bool device = false;
            bool host   = false;
        };

        struct OwnerProjection {
            const SequenceState* sequence                     = nullptr;
            const SharedPrefixState* shared                   = nullptr;
            const qwen3_5::detail::PressureDecision* decision = nullptr;
            runtime::PlanningOwnerId owner;
        };

        struct CheckpointProjection {
            qwen3_5::CheckpointSummary checkpoint;
            StateImageHandle state;
            bool survives = true;
        };

        std::vector<StatePlacement> state_placements;
        std::vector<OwnerProjection> owners;
        std::vector<CheckpointProjection> checkpoints;
        std::vector<std::optional<runtime::CheckpointRecoveryAlternativeWork>> direct_work;
        qwen3_5::ContinuationSummary continuation_summary;
    };

    // `execution_in` and `peer_parameters_in` are null at tensor-parallel width 1. At width 2
    // `device` is execution_in->dev[0] and `peer_parameters_in` is Parameters(model, 1).
    ProgramImpl(const execution::Parameters& parameters, const SequencePlanImpl& plan,
                DeviceContext& device, const StartupObserver& startup_observer,
                ExecutionContext* execution_in                  = nullptr,
                const execution::Parameters* peer_parameters_in = nullptr);
    ~ProgramImpl() noexcept;

    [[nodiscard]] RequestBasePlan plan_request(const PreparedPromptData& prompt,
                                               const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] std::vector<float> causal_score(PreparedPromptData&& prompt,
                                                  std::uint32_t first_target);
    [[nodiscard]] std::optional<AdmissionCandidate> inspect_admission(
        const PreparedPromptData& prompt, const RequestBasePlan& base, runtime::LaneId destination,
        const ContinuationHandle* source, const SharedPrefixHandle* shared_source,
        std::optional<runtime::CheckpointRef> checkpoint, bool must_retain_private_source);
    [[nodiscard]] std::optional<AdmissionCandidate> seal_materialization(
        const AdmissionCandidate& admission, const PreparedPromptData& prompt,
        std::span<const ContinuationHandle* const> pressure_owners,
        std::span<const runtime::PlanningOwnerId> pressure_owner_ids,
        std::span<const qwen3_5::detail::PressureDecision* const> pressure_options,
        std::span<const SharedPrefixHandle* const> shared_pressure_owners,
        std::span<const runtime::PlanningOwnerId> shared_pressure_owner_ids,
        std::span<const qwen3_5::detail::PressureDecision* const> shared_pressure_options);
    [[nodiscard]] std::unique_ptr<CapturePressureCandidateImpl>
    make_capture_physical_candidate(const CaptureAssessment& assessment) const;
    void select_shared_captures(AdmissionCandidate& candidate, const PreparedPromptData& prompt,
                                std::span<const std::uint32_t> frontiers);
    [[nodiscard]] runtime::PrefillWork
    shared_capture_split_prefill_work(const AdmissionCandidate& candidate,
                                      const PreparedPromptData& prompt,
                                      std::span<const std::uint32_t> frontiers);
    [[nodiscard]] runtime::PreflightStatus
    revalidate_materialization(const AdmissionCandidate& plan,
                               const PreparedPromptData& prompt) const;
    [[nodiscard]] runtime::ContextTransactionReserveStatus
    reserve_materialization(AdmissionCandidate&& plan, PreparedPromptData&& prompt,
                            runtime::CancellationFlagView cancellation);
    [[nodiscard]] bool
    persistent_backfill_safe(const RequestBasePlan& blocked_head,
                             const AdmissionCandidate& candidate,
                             std::span<const SequenceHandle> persistent_borrowers) const;
    [[nodiscard]] ContextTransactionProgress
    progress_context_transaction(runtime::CancellationFlagView cancellation);
    void finalize_context_transaction() noexcept;
    [[nodiscard]] bool has_context_transaction() const noexcept;
    [[nodiscard]] PrefillProgress advance_prefill(SequenceHandle sequence,
                                                  runtime::ExecutionTiming* failed_timing,
                                                  runtime::TokenMaskProvider* masks = nullptr);
    [[nodiscard]] CaptureAssessment
    inspect_capture(const CaptureOffer& offer, const SharedPrefixHandle* exact_shared,
                    const SharedPrefixHandle* replacement,
                    std::optional<runtime::CheckpointRef> private_replacement,
                    bool permit_shared_publication) const;
    [[nodiscard]] std::vector<runtime::CheckpointRecoveryAlternativeWork>
    checkpoint_recovery_work(const ContinuationHandle& owner,
                             runtime::CheckpointRef checkpoint) const;
    [[nodiscard]] std::vector<runtime::CheckpointRecoveryAlternativeWork>
    checkpoint_recovery_work(const SharedPrefixHandle& owner,
                             runtime::CheckpointRef checkpoint) const;
    [[nodiscard]] bool shared_capture_matches(const CaptureOffer& offer,
                                              const SharedPrefixHandle& shared) const;
    void skip_capture(CaptureOffer&& offer);
    [[nodiscard]] runtime::ContextTransactionReserveStatus
    reserve_active_capture(CaptureOffer&& offer, const SharedPrefixHandle* exact_shared,
                           const SharedPrefixHandle* replacement,
                           std::optional<runtime::CheckpointRef> private_replacement,
                           bool permit_shared_publication,
                           runtime::CancellationFlagView cancellation);
    [[nodiscard]] runtime::ContextTransactionReserveStatus reserve_active_capture_with_pressure(
        CaptureOffer&& offer, const SharedPrefixHandle* exact_shared,
        const SharedPrefixHandle* replacement,
        std::optional<runtime::CheckpointRef> private_replacement, bool permit_shared_publication,
        CapturePressureCandidate&& pressure, runtime::CancellationFlagView cancellation);
    [[nodiscard]] PendingBatch decode(std::span<const SequenceHandle> sequences,
                                      std::span<const runtime::RoundBudget> budgets,
                                      runtime::ExecutionTiming* failed_timing,
                                      runtime::TokenMaskProvider* masks = nullptr);
    [[nodiscard]] bool sampling_bypass_hooks() const noexcept;
    [[nodiscard]] runtime::ExecutionTiming
    append_forced_tokens(std::span<const SequenceHandle> sequences,
                         std::span<const TokenId> row_major_tokens, std::uint32_t row_stride,
                         std::span<const std::optional<std::uint32_t>> prefix_execution_splits,
                         runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] CommitResult commit(PendingBatch&& pending,
                                      std::span<const runtime::CommitDecision> decisions,
                                      runtime::CommitObservation observation,
                                      runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] DiscardResult abort_pending(PendingBatch&& pending) noexcept;
    [[nodiscard]] FinishResult finish(SequenceHandle sequence) noexcept;
    [[nodiscard]] FinishResult finish_cancelled(SequenceHandle sequence) noexcept;
    [[nodiscard]] FinishResult finish_terminal(SequenceHandle sequence, bool cancelled) noexcept;
    [[nodiscard]] AbortResult abort(SequenceHandle sequence) noexcept;
    [[nodiscard]] ReleaseResult release_continuation(ContinuationHandle&& continuation) noexcept;
    [[nodiscard]] std::optional<qwen3_5::ContinuationSummary>
    drop_continuation_anchor(const ContinuationHandle& continuation,
                             runtime::CheckpointRef anchor) noexcept;
    [[nodiscard]] ReleaseResult release_shared_prefix(SharedPrefixHandle&& shared) noexcept;
    void fail_all_cleanup() noexcept;
    [[nodiscard]] detail::PhysicalResources admission_capacity() const noexcept;
    [[nodiscard]] bool isolated_request_feasible(const RequestBasePlan& base) const noexcept;

    [[nodiscard]] runtime::ProgramResourceRevision resource_revision() const noexcept {
        return resource_revision_;
    }

    [[nodiscard]] qwen3_5::PhysicalUsageSnapshot physical_usage() const noexcept;

    [[nodiscard]] std::optional<std::uint32_t>
    device_kv_lease_settlement_tokens(SequenceHandle sequence,
                                      std::uint32_t forced_span_tokens) const noexcept;
    [[nodiscard]] std::optional<DeviceKVLeaseShortfall>
    device_kv_lease_shortfall(SequenceHandle sequence) const noexcept;
    [[nodiscard]] bool resume_device_kv_lease(SequenceHandle sequence) noexcept;
    [[nodiscard]] DeviceKVPages
    retained_device_kv_pages(const ContinuationHandle& continuation) const noexcept;
    [[nodiscard]] DeviceKVPages
    retained_device_kv_pages(const SharedPrefixHandle& shared) const noexcept;
    [[nodiscard]] DeviceKVPages idle_device_kv_pages(const ContinuationHandle& continuation) const;
    [[nodiscard]] DeviceKVPages idle_device_kv_pages(const SharedPrefixHandle& shared) const;
    [[nodiscard]] DeviceKVPages idle_device_kv_pages(const SequenceKVBundle& kv) const;
    [[nodiscard]] std::vector<DeviceKVPages>
    releasable_device_kv_pages(std::span<const ContinuationHandle* const> continuations,
                               std::span<const SharedPrefixHandle* const> shared) const;
    [[nodiscard]] DeviceKVPages device_kv_available_pages() const noexcept;
    [[nodiscard]] std::optional<ContinuationSummary>
    demote_retained_kv_for_lease(const ContinuationHandle& continuation,
                                 DeviceKVPages needed) noexcept;
    [[nodiscard]] std::optional<SharedPrefixSummary>
    demote_retained_shared_kv_for_lease(const SharedPrefixHandle& shared,
                                        DeviceKVPages needed) noexcept;
    [[nodiscard]] bool idle_device_kv_offload() const noexcept;
    [[nodiscard]] std::optional<std::size_t>
    idle_device_kv_host_bytes(const ContinuationHandle& continuation) const noexcept;
    [[nodiscard]] std::size_t host_kv_free_bytes() const noexcept;
    DeviceKVCompaction compact_device_kv();

    // Disk tier (v0.7.2, storage/disk_kv.cpp).
    void initialize_disk_cache();
    [[nodiscard]] bool disk_cache_has_unsaved() const noexcept;
    DiskCacheSave disk_cache_save_one(const std::function<bool()>& abort);
    DiskCacheSave disk_cache_save(std::uint32_t slot, const std::function<bool()>& abort);
    void disk_synchronize() const;
    [[nodiscard]] std::string disk_cache_status() const;
    // At the first step of a root prefill: loads the longest saved checkpoint the prompt begins
    // with (KV, MTP KV, StateImage) and moves the cursor there. Returns the tokens loaded (0:
    // none).
    std::uint32_t disk_cache_resume(SequenceState& sequence, RequestControl& request);
    [[nodiscard]] std::uint32_t disk_resume_tokens(const PreparedPromptData& prompt);

    [[nodiscard]] MemorySummary memory_summary() const noexcept;

    void reset_memory_peaks() noexcept;

    friend struct qwen3_5::detail::PressurePlanningSessionImpl;

    const execution::Parameters& parameters;
    DeviceContext& device;
    const std::uint32_t capacity;
    const std::uint32_t kv_capacity;
    const std::uint32_t max_concurrency;
    const ContextCacheOptions context_cache;
    const std::uint32_t continuation_capacity;
    const std::uint32_t shared_prefix_capacity;
    const std::uint32_t prefill_chunk;
    // T42: Qwen4Exp prompt chunks of more than this many columns run on lent workspace and
    // prefill_hidden holds this many columns (0 = off: prefill_hidden holds prefill_chunk).
    const std::uint32_t qwen4_prompt_base_chunk;
    const std::uint32_t kv_lease_window_tokens; // EngineOptions::kv_lease_window_tokens
    const std::uint32_t draft_window;
    // T41 adaptive MTP window (SpeculativeOptions::min_draft_probability; 0 = off).
    const float mtp_min_p;
    const MtpWindowRule mtp_window_rule;
    const SpeculativeBackend speculative_backend;
    const KvCacheStorage kv_storage;
    const ProposalHead proposal_head;
    const bool vision_enabled;
    // The tensor-parallel rank that holds the Vision tower and encodes; 0 at width 1.
    const int vision_rank;
    const bool use_cuda_graph;
    const bool causal_scoring;
    const std::size_t kv_payload_bytes;
    const std::size_t graph_allowance_bytes;
    const WorkspacePlan workspace_plan;
#include "models/qwen3_5/program/tp2/program_impl_public.inc"

    // T64 elastic KV (plan.elastic_kv): the KV page planes and the expert slot pools, in their own
    // VMM arena (PersistentLayout::elastic_bytes) whose KV pages and elastic expert columns are
    // backed on demand (Qwen4ElasticVram); everything else stays in `persistent`.
    std::unique_ptr<ElasticArena> elastic_arena;
    DeviceArena persistent;
    DeviceArena workspace_storage;
    WorkspaceArena work;
    std::unique_ptr<qwen3_5::DecoderState> decoder;
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
    // Disk tier (v0.7.2, storage/disk_kv.cpp): null when off. The pinned scratch holds one Host
    // StateImage, the read buffer two resume read-ahead chunks.
    std::unique_ptr<DiskKVCache> disk_kv;
    std::optional<PinnedHostBuffer> disk_state_scratch;
    std::optional<PinnedHostBuffer> disk_read_buffer;

    // Per continuation slot: the slot generation and the frontier last handed to the disk.
    struct DiskSaved {
        std::uint64_t generation = 0;
        std::uint32_t frontier   = 0;
        std::uint64_t digest     = 0; // xxh64 of the tokens up to `frontier`
    };

    std::vector<DiskSaved> disk_saved;
    std::optional<GdnReplayRecords> replay_records;
    std::optional<ops::GdnReplayFoldPlan> replay_fold;
    std::optional<DFlashPersistentState> dflash;
    qwen3_5::RoundState io;
    Tensor prefill_hidden;
    std::optional<Tensor> score_hidden;
    Tensor sampling_config;
    Tensor token_counts;

    // Structured output (program/grammar_masks.cpp). Device 0 only (rank 0 samples alone):
    // [ceil(V/32), positions, max_concurrency] mask words, one slot per lane; its pinned twin
    // (plus one staged prefill SamplingConfig per lane); the provider of the current Program call
    // and the dead-end position bits of its compact rows.
    Tensor grammar_masks_device;
    std::optional<PinnedHostBuffer> grammar_masks_host;
    runtime::TokenMaskProvider* round_masks = nullptr;
    std::array<std::uint32_t, kMaximumConcurrency> grammar_dead_positions{};

    class RoundMaskScope {
    public:
        RoundMaskScope(ProgramImpl& program, runtime::TokenMaskProvider* masks) noexcept;
        ~RoundMaskScope();
        RoundMaskScope(const RoundMaskScope&)            = delete;
        RoundMaskScope& operator=(const RoundMaskScope&) = delete;

    private:
        ProgramImpl& program_;
    };

    [[nodiscard]] bool row_constrained(std::size_t row) const noexcept;
    [[nodiscard]] std::size_t grammar_mask_positions() const noexcept;

    // Byte offset of the staged prefill configs in grammar_masks_host: after the mask words,
    // rounded up to alignof(SamplingConfig) (the mask bytes are only a multiple of 4).
    [[nodiscard]] static constexpr std::size_t
    grammar_staged_config_offset(std::size_t mask_bytes) noexcept {
        constexpr std::size_t alignment = alignof(ops::SamplingConfig);
        return (mask_bytes + alignment - 1U) / alignment * alignment;
    }

    // Fills the lane's slot with the row's masks for (drafts.size() + 1) positions, uploads it on
    // the device stream (before the launch) and returns the row's SamplingMask; {} unconstrained.
    [[nodiscard]] ops::SamplingMask fill_grammar_mask(std::size_t row, std::uint32_t lane,
                                                      std::span<const TokenId> drafts);
    void bind_prefill_grammar_mask(SequenceState& sequence, RequestControl& request);

    std::vector<SequenceState> continuation_states;
    std::vector<ContinuationSlot> continuation_slots;
    std::vector<SharedPrefixState> shared_prefix_states;
    std::vector<SharedPrefixSlot> shared_prefix_slots;
    std::array<std::uint32_t, kMaximumConcurrency> active_continuations{};
    std::array<RequestControl, kMaximumConcurrency> requests;
    std::array<std::uint64_t, kMaximumConcurrency> lane_epochs{};

    DecodeGraphFamily ordinary_graphs;
    DecodeGraphFamily mtp_graphs;
    DecodeGraphFamily dflash_graphs;

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

    // Qwen4Exp (T20/M1, tp 2 T31): routed-expert backend, n-gram gather and its pinned staging,
    // the validation dump, and the runtime record every Qwen4Exp TextContext reads.
    std::unique_ptr<execution::Qwen4RoutedExperts> qwen4_experts;
    std::uint32_t qwen4_expert_slots    = 0;
    std::size_t qwen4_expert_pool_bytes = 0;
    // T64: the elastic KV <-> expert slot manager (null unless plan.elastic_kv.enabled).
    std::unique_ptr<Qwen4ElasticVram> qwen4_elastic;
    std::optional<execution::Qwen4ExpertBank> qwen4_mtp_bank;
    // Host copy of the MTP bank's expanded weight divisors (gate|up [2E] then down [E]) for the
    // offload backend (CPU path, device divisor table); empty for per-expert banks.
    std::vector<float> qwen4_mtp_host_divisors;
    std::optional<execution::Qwen4ExpertBank> qwen4_peer_mtp_bank; // tp 2: rank 1's MTP bank
    std::unique_ptr<execution::Qwen4MoeProbe> qwen4_mtp_probe;
    // NINFER_QWEN4_TOP2=<file> (validation): one line per decoded position, the target's two
    // largest logits: lane position source(0 ordinary, 1 verify) id1 logit1 id2 logit2.
    std::FILE* qwen4_top2_file = nullptr;
    // NINFER_QWEN4_MTP_CALIB=<file> (measurement): one line per verified MTP round, "extent
    // accepted logp_1 .. logp_K" (the proposals' own softmax log-probabilities).
    std::FILE* qwen4_mtp_calib = nullptr;
    // NINFER_QWEN4_MTP_ORACLE=<file> (validation): one whole token sequence per line (prompt +
    // a reference continuation). A round whose committed ledger is a prefix of a line drafts that
    // line's next tokens instead of the MTP proposals, so verification and the commit of long
    // accepted prefixes are exercised whatever the draft head's quality.
    std::vector<std::vector<TokenId>> qwen4_mtp_oracle;
    // NINFER_QWEN4_FORCE_TOKENS=<file> (validation, same format): a round of a matching sequence
    // commits the file's tokens instead of its own choices (teacher forcing through the decode or
    // the verification route; with NINFER_QWEN4_MTP_NO_DRAFTS one token per round). The logits and
    // the state the round computed are untouched: they depend only on the consumed tokens.
    std::vector<std::vector<TokenId>> qwen4_forced_tokens;
    void force_qwen4_tokens(const SequenceState& sequence, TokenId* tokens,
                            std::int32_t count) const;
    // NINFER_QWEN4_MTP_NO_DRAFTS=1 (validation): every round verifies its anchor alone (extent 0,
    // the fallback step), i.e. greedy decode through the verification route.
    bool qwen4_mtp_no_drafts = false;
    // NINFER_QWEN4_MTP_EXTENT_CAP=<k> (validation/measurement): verify at most k proposals per
    // round in the K-wide window (the cost of a column at fixed drafting work).
    std::uint32_t qwen4_mtp_extent_cap = qwen3_5::kMtpDecodeMaximumDrafts;
    void write_qwen4_top2(const Tensor& logits_column, std::uint32_t lane, std::uint32_t position,
                          int source);
    std::unique_ptr<execution::PleGather> qwen4_ple_gather;
    std::optional<PinnedHostBuffer> qwen4_ple_host;
    std::optional<PinnedHostBuffer> qwen4_decode_ple_buffer;
    // T49: rank 1's commit work (fold) joined into rank 0's stream when the commit does not wait.
    std::optional<CudaCompletionEvent> qwen4_peer_commit_event;
    std::uint16_t* qwen4_decode_ple_host = nullptr;
    // Deferred decode rows (Qwen4Runtime::decode_ple_staged): the mapped words of rank 0 and rank
    // 1, and the round's queued gathers (prepared before the launch, run after it).
    std::optional<PinnedHostBuffer> qwen4_ple_staged_words;
    std::array<std::uint32_t*, 2> qwen4_ple_staged{};

    struct Qwen4PleStage {
        std::uint32_t lane       = 0;
        std::uint32_t frontier   = 0;
        std::uint32_t count      = 0;
        std::size_t first_column = 0;
        std::array<TokenId, qwen3_5::kMtpDecodeMaximumWidth> tokens{};
    };

    std::vector<Qwen4PleStage> qwen4_ple_queue;
    // Runs stage_qwen4_ple_round() of a graph round after its launch (created with the words).
    std::unique_ptr<Qwen4PleStager> qwen4_ple_stager;
    std::unique_ptr<execution::Qwen4TfDump> qwen4_tfdump;
    // T59 fast bench: teacher-forced replay of a recorded decode (NINFER_QWEN4_REPLAY, see
    // qwen4_tfdump.h) and its recorder (NINFER_QWEN4_REPLAY_RECORD=<file>: "req <prompt tokens>"
    // at every first token, "win <frontier> <extent> <accepted> <drafts...>" per MTP round).
    std::unique_ptr<execution::Qwen4Replay> qwen4_replay;
    std::FILE* qwen4_replay_record = nullptr;
    // T59 (NINFER_QWEN4_ROUND_LOG=<file>, measurement): one line per decode round,
    // "<o|m> <frontier> <extent> <accepted> <round us> <host gap us> <missing> <cpu experts>
    // <gpu-share experts> <expert window wall us> <fill wait us> <elastic expert columns>" (gap =
    // previous round's end -> this round's start, -1 for a request's first round; columns -1
    // without elastic KV).
    std::FILE* qwen4_round_log = nullptr;
    std::chrono::steady_clock::time_point qwen4_round_end{};
    bool qwen4_round_follows = false;
    void log_qwen4_round(char kind, std::uint32_t frontier, std::uint32_t extent,
                         std::int32_t accepted, double round_seconds);
    std::optional<execution::Qwen4Runtime> qwen4_runtime_record;
    // T43: the draft shortlist proposal head (Qwen4Runtime::draft_head of rank 0), when planned.
    std::optional<execution::Qwen4DraftHead> qwen4_draft_head;
    // tp 2: rank 1's expert backend (the odd expert IDs) and runtime record (TpExecution::qwen4).
    std::unique_ptr<execution::Qwen4RoutedExperts> qwen4_peer_experts;
    std::optional<execution::Qwen4Runtime> qwen4_peer_runtime_record;

    [[nodiscard]] const execution::Qwen4Runtime* qwen4_runtime() const noexcept {
        return qwen4_runtime_record ? &*qwen4_runtime_record : nullptr;
    }

    void construct_qwen4(const SequencePlanImpl& plan, DeviceSpan backing);
    void construct_qwen4_experts(const SequencePlanImpl& plan, DeviceSpan backing,
                                 execution::Qwen4Runtime& runtime,
                                 execution::Qwen4Runtime& peer_runtime);
    // Expert window of one execution on every rank's backend (qwen4_experts.h window protocol):
    // begin before anything of it is enqueued, observe once it is enqueued, end after the
    // devices completed it (abandon on failure). No-ops without Qwen4Exp experts.
    void qwen4_begin_window(execution::Qwen4Window kind);
    void qwen4_observe_window();
    void qwen4_end_window();
    void qwen4_abandon_window() noexcept;

    // Runs `body` (an execution outside the decode rounds and prefill chunks, e.g. the MTP
    // prefix-reuse bridge) as one prompt window when Qwen4Exp experts are bound: begin, body,
    // observe, device synchronization, end. Without them `body` runs as is.
    template <class Body>
    void in_qwen4_window(Body&& body) {
        if (!qwen4_experts) {
            body();
            return;
        }
        qwen4_begin_window(execution::Qwen4Window::Prompt);
        try {
            body();
            qwen4_observe_window();
            synchronize_devices();
            qwen4_end_window();
        } catch (...) {
            try {
                synchronize_devices();
            } catch (...) {}
            qwen4_abandon_window();
            throw;
        }
    }

    // Qwen4Exp PLE n-gram rows of a decode window into the pinned round buffer: columns
    // [first_column, first_column + tokens.size()) are the tokens `tokens` at positions
    // frontier.., with `ledger` (the committed tokens before `frontier`) as their history.
    // T53: page prefetch of the next round's n-gram rows (an MTP round's anchor and drafts, or
    // an ordinary round's sampled token, are in the egress once this round has synchronized), so
    // the storage reads overlap the commit and the next round's host preparation.
    // NINFER_T53_PLE_PREFETCH=0 disables it.
    void prefetch_qwen4_ple_next_round(std::span<const std::uint32_t> lanes, bool mtp);
    void stage_qwen4_ple_rows(std::span<const TokenId> ledger, std::uint32_t frontier,
                              std::span<const TokenId> tokens, std::size_t first_column);
    // Deferred decode rows. A round: begin_qwen4_ple_round() (no stream waiting: lowers the
    // words), queue_qwen4_ple_rows() per row while preparing, the launch, then
    // stage_qwen4_ple_round() gathers the queued rows into the pinned staging and raises the
    // words (also on failure). raise_qwen4_ple_staged() alone releases an execution whose rows
    // do not matter (warm-up, graph preparation) or that failed after its launch.
    void begin_qwen4_ple_round();
    void queue_qwen4_ple_rows(std::uint32_t lane, std::uint32_t frontier,
                              std::span<const TokenId> tokens, std::size_t first_column);
    void stage_qwen4_ple_round();
    void raise_qwen4_ple_staged() noexcept;
    // Qwen4Exp speculative commit: advances each row's PLE convolution history by its committed
    // prefix of the verified window (the PLE half of the ReplaySSM fold).
    void fold_qwen4_ple(std::span<const ops::GdnReplayFoldRow> rows);
    void fold_qwen4_ple_rank(std::span<const ops::GdnReplayFoldRow> rows, int rank);

private:
    void advance_resource_revision() noexcept {
        if (++resource_revision_.value == 0) { ++resource_revision_.value; }
    }

    runtime::ProgramResourceRevision resource_revision_{.value = 1};
    std::uint32_t pressure_planning_generation_ = 0;
    bool pressure_planning_active_              = false;

    struct PressurePageScratchSlot {
        std::uint32_t generation     = 0;
        std::uint32_t selected_index = std::numeric_limits<std::uint32_t>::max();
        std::uint64_t host_group     = 0;
        bool projected               = false;
        bool device                  = false;
        bool host                    = false;
        bool pressure_targeted       = false;
    };

    struct PressureSelectedPage {
        LogicalKVPageHandle page;
        std::uint32_t references = 0;
    };

    struct PressureSelectedState {
        StateImageHandle state;
        bool device = false;
        bool host   = false;
    };

    mutable std::uint32_t pressure_page_scratch_generation_ = 0;
    mutable std::vector<PressurePageScratchSlot> pressure_text_page_scratch_;
    mutable std::vector<PressurePageScratchSlot> pressure_backend_page_scratch_;
    mutable std::vector<PressureSelectedPage> pressure_text_selected_pages_;
    mutable std::vector<PressureSelectedPage> pressure_backend_selected_pages_;
    mutable std::vector<std::uint8_t> pressure_private_owner_scratch_;
    mutable std::vector<std::uint8_t> pressure_shared_owner_scratch_;
    mutable std::vector<std::vector<runtime::CheckpointRef>> pressure_private_drop_scratch_;
    mutable std::vector<PressureSelectedState> pressure_state_scratch_;

    void begin_pressure_page_scratch() const noexcept;
    [[nodiscard]] PressurePageScratchSlot& pressure_page_scratch(const LogicalKVPageStore& store,
                                                                 LogicalKVPageHandle page) const;
    [[nodiscard]] const PressurePageScratchSlot*
    find_pressure_page_scratch(const LogicalKVPageStore& store, LogicalKVPageHandle page) const;

    struct MaterializationSourceProtection {
        struct StateOwnershipCandidate {
            StateImageHandle state;
            std::uint32_t source_checkpoint_references = 0;
        };

        std::optional<std::uint32_t> private_source_index;
        bool consumed_private_source = false;
        std::optional<StateImageHandle> state;
        std::uint32_t consumed_state_references = 0;
        bool state_fork_required                = false;
        std::vector<StateOwnershipCandidate> state_ownership_candidates;
        std::optional<KVAddressSpaceHandle> text;
        std::uint32_t text_pages          = 0;
        std::uint32_t text_transfer_pages = 0;
        bool text_prefix_fork_required    = false;
        std::optional<KVAddressSpaceHandle> backend;
        std::uint32_t backend_pages          = 0;
        std::uint32_t backend_transfer_pages = 0;
        bool backend_prefix_fork_required    = false;
    };

    struct PendingTransaction {
        std::uint64_t id = 0;
        std::array<std::uint32_t, kMaximumConcurrency> lanes{};
        std::array<std::uint64_t, kMaximumConcurrency> epochs{};
        std::size_t size = 0;
    };

    std::optional<PendingTransaction> pending_transaction_;
    std::uint64_t next_transaction_id_ = 1;

    enum class PressureTransitionPhase : std::uint8_t {
        HostReleases,
        CopyPreparation,
        CopiesInFlight,
        CopyPublication,
        Committed,
    };

    struct PressureTransition {
        PressureTransitionPhase phase = PressureTransitionPhase::HostReleases;
        std::array<TransferWork, 3> transfer_work{};
        std::array<std::uint32_t, 3> transfer_pages{};
        std::uint64_t state_images = 0;
        std::uint8_t timer_mask    = 0;
    };

    struct MaterializationTransaction {
        struct KVRestorePage {
            LogicalKVPageHandle logical;
            HostKVExtentCapability extent;
            std::uint32_t extent_page = 0;
        };

        struct PressureWork {
            struct StateChangeWork {
                std::optional<StateImageTransfer> transfer;
                bool host_released = false;
            };

            struct KVChangeWork {
                std::vector<LogicalKVPageHandle> pages;
                std::vector<DeviceKVPageHandle> sources;
                std::optional<HostKVExtentReservation> backup;
                bool host_released = false;
            };

            qwen3_5::detail::PressureDecision option;
            std::uint32_t continuation_index      = 0;
            std::uint64_t continuation_generation = 0;
            bool shared_owner                     = false;
            std::vector<StateChangeWork> state_changes;
            std::vector<KVChangeWork> main_kv_changes;
            std::vector<KVChangeWork> backend_kv_changes;
            detail::PhysicalDelta committed_delta;
            bool submitted                 = false;
            bool completed                 = false;
            bool checkpoint_drop_published = false;
            bool mutation_published        = false;
            std::uint64_t spill_pages      = 0;
        };

        std::uint64_t id = 0;
        runtime::LaneId destination;
        bool has_source                        = false;
        bool has_shared_source                 = false;
        runtime::PrivateSourceMode source_mode = runtime::PrivateSourceMode::ConsumeToActive;
        std::uint32_t source_index             = 0;
        std::uint64_t source_generation        = 0;
        std::uint32_t shared_source_index      = 0;
        std::uint64_t shared_source_generation = 0;
        std::optional<MaterializationSourceResult> source_result;
        std::optional<MaterializationSharedSourceResult> shared_source_result;
        std::vector<std::uint32_t> victim_indices;
        std::vector<std::uint64_t> victim_generations;
        std::vector<bool> victim_released;
        std::vector<PressureWork> pressure;
        std::vector<MaterializationVictimResult> pressure_results;
        std::size_t pressure_cursor = 0;
        std::size_t victim_count    = 0;
        std::vector<std::uint32_t> shared_victim_indices;
        std::vector<std::uint64_t> shared_victim_generations;
        std::vector<bool> shared_victim_released;
        std::vector<MaterializationSharedVictimResult> shared_pressure_results;
        std::vector<PressureWork> shared_pressure;
        std::size_t shared_pressure_cursor = 0;
        std::size_t shared_victim_count    = 0;
        PressureTransition pressure_transition;
        std::optional<AdmissionCandidate> plan;
        std::optional<std::uint32_t> root_continuation_index;
        bool root_waiting_for_victim = false;
        std::array<StateImageHandle, 2> reserved_states{};
        std::size_t reserved_state_count = 0;
        std::optional<StateImageHandle> state_fork_destination;
        std::optional<KVAddressSpaceHandle> root_text_address;
        std::optional<KVAddressSpaceHandle> root_backend_address;
        std::optional<KVActivationReservation> text_activation;
        std::optional<KVActivationReservation> backend_activation;
        std::optional<DeviceKVPageReservation> text_source_restore_reservation;
        std::optional<DeviceKVPageReservation> backend_source_restore_reservation;
        std::optional<KVPrefixForkReservation> text_prefix_fork;
        std::optional<KVPrefixForkReservation> backend_prefix_fork;
        std::optional<LogicalKVPageHandle> text_retained_tail;
        std::optional<LogicalKVPageHandle> backend_retained_tail;
        std::optional<HostKVExtentReservation> text_retained_tail_backup;
        std::optional<HostKVExtentReservation> backend_retained_tail_backup;
        std::optional<std::uint32_t> text_activation_frontier;
        std::optional<std::uint32_t> backend_activation_frontier;
        std::optional<StateImageTransfer> state_restore;
        bool split_state_identity = false;
        std::vector<KVRestorePage> text_restores;
        std::vector<DeviceKVPageHandle> text_restore_destinations;
        std::vector<KVRestorePage> backend_restores;
        std::vector<DeviceKVPageHandle> backend_restore_destinations;
        std::vector<runtime::ContextTransferObservation> transfer_observations;
        runtime::ContextOperationCounts operations;
        bool state_restored                 = false;
        bool transfer_submitted             = false;
        std::uint8_t transfer_timer_mask    = 0;
        bool prefix_tail_submitted          = false;
        bool retained_tail_backup_submitted = false;
        bool prefix_forks_ready             = false;
        bool source_prepared                = false;
        bool cancel_pending                 = false;
        bool prepared                       = false;
        bool terminal                       = false;
    };

    std::uint64_t next_materialization_id_ = 1;
    CudaCompletionEvent context_source_ready_;
    CudaCompletionEvent context_completion_;
    // Context transfers complete on rank 0's transfer stream and, at tp 2, on rank 1's: these
    // record, poll and await both (tp2/program_impl_tp2.inc).
    void record_context_completion();
    [[nodiscard]] bool context_completion_ready() const;
    void synchronize_context_completion() const;
    void synchronize_context_transfer_streams() const;
    std::vector<TokenId> materialization_ledger_;
    qwen3_5::detail::ResidentPrefixIdentity materialization_identity_;
    qwen3_5::detail::PrefixShortlistDigests materialization_prefix_digests_;

    struct ActiveCaptureTransaction {
        std::uint64_t id         = 0;
        std::uint32_t lane       = 0;
        std::uint64_t lane_epoch = 0;
        CaptureGroup group;
        bool publish_private = false;
        bool publish_shared  = false;
        bool replaces_shared = false;
        std::optional<runtime::CheckpointRef> private_replacement;
        std::optional<std::uint32_t> shared_index;
        std::uint64_t replacement_generation = 0;
        StateImageHandle source_state;
        StateImageHandle destination_state;
        qwen3_5::CaptureStatePlacement state_placement = qwen3_5::CaptureStatePlacement::DeviceFork;
        std::optional<StateImageTransfer> state_snapshot;
        std::optional<KVAddressSpaceHandle> active_text_destination;
        std::optional<KVAddressSpaceHandle> active_backend_destination;
        std::optional<KVActiveSnapshotReservation> text_snapshot;
        std::optional<KVActiveSnapshotReservation> backend_snapshot;
        detail::PhysicalDelta resource_delta;
        detail::PhysicalDelta active_entitlement_delta;
        detail::PhysicalResources capacity_preparation_removed;
        ContinuationSummary active_summary;
        std::vector<runtime::ContextTransferRequirement> transfer_requirements;
        std::vector<runtime::ContextTransferObservation> transfer_observations;
        runtime::ContextOperationCounts operations;
        std::vector<std::uint32_t> victim_indices;
        std::vector<std::uint64_t> victim_generations;
        std::vector<MaterializationTransaction::PressureWork> pressure;
        std::vector<MaterializationVictimResult> pressure_results;
        std::vector<std::uint32_t> shared_victim_indices;
        std::vector<std::uint64_t> shared_victim_generations;
        std::vector<MaterializationTransaction::PressureWork> shared_pressure;
        std::vector<MaterializationSharedVictimResult> shared_pressure_results;
        PressureTransition pressure_transition;
        bool recycles_private_state        = false;
        bool replacement_removed           = false;
        bool prepared                      = false;
        std::uint64_t recycled_state_epoch = 0;
        bool transfer_enqueue_pending      = false;
        bool transfer_submitted            = false;
        std::uint8_t transfer_timer_mask   = 0;
        bool published                     = false;
    };

    std::uint64_t next_capture_offer_id_ = 1;

    using ContextTransaction =
        std::variant<std::monostate, MaterializationTransaction, ActiveCaptureTransaction>;
    ContextTransaction context_transaction_;

    [[nodiscard]] MaterializationResult
    progress_materialization_transaction(runtime::CancellationFlagView cancellation);
    [[nodiscard]] ActiveCaptureResult
    progress_active_capture_transaction(runtime::CancellationFlagView cancellation);
    [[nodiscard]] runtime::ContextTransactionReserveStatus
    reserve_active_capture_impl(CaptureOffer&& offer, const SharedPrefixHandle* exact_shared,
                                const SharedPrefixHandle* replacement,
                                std::optional<runtime::CheckpointRef> private_replacement,
                                bool permit_shared_publication,
                                std::optional<CapturePressureCandidate> pressure,
                                runtime::CancellationFlagView cancellation);

    std::array<CudaEventTimer, 3> context_transfer_timers_;

    [[nodiscard]] std::optional<AdmissionCandidate>
    inspect_lane(std::uint32_t lane, const PreparedPromptData& prompt, const RequestBasePlan& base,
                 const SequenceState* source, const SharedPrefixState* shared_source,
                 std::optional<runtime::CheckpointRef> checkpoint, bool must_retain_private_source);
    [[nodiscard]] StartResult start_request(MaterializationTransaction& transaction);
    void prepare_materialization(MaterializationTransaction& transaction);
    void enqueue_materialization_transfers(MaterializationTransaction& transaction);
    void record_materialization_transfer_observations(MaterializationTransaction& transaction);
    void publish_materialization_transfers(MaterializationTransaction& transaction);
    void prepare_prefix_forks(MaterializationTransaction& transaction);
    void prepare_consumed_source(MaterializationTransaction& transaction);
    void abort_materialization_transfers(MaterializationTransaction& transaction) noexcept;
    void prepare_pressure_bookkeeping(MaterializationTransaction::PressureWork& work);
    void prepare_pressure_work(MaterializationTransaction::PressureWork& work,
                               runtime::ContextResourceClass resource);
    void publish_pressure_host_releases(MaterializationTransaction::PressureWork& work);
    void publish_pressure_work(MaterializationTransaction::PressureWork& work) noexcept;
    void abort_pressure_work(MaterializationTransaction::PressureWork& work) noexcept;
    void start_context_transfer_timer(runtime::ContextResourceClass resource);
    void stop_context_transfer_timer(runtime::ContextResourceClass resource);
    [[nodiscard]] runtime::ContextTransferObservation context_transfer_observation(
        runtime::ContextResourceClass resource, runtime::ContextTransferDirection direction,
        TransferWork work, std::uint32_t page_count = 0, std::uint64_t state_images = 1) const;

    struct PhysicalReleaseResult {
        runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
        detail::PhysicalDelta delta;
    };

    [[nodiscard]] PhysicalReleaseResult
    release_materialization_victim(MaterializationTransaction& transaction, std::size_t position);
    void start_sequence(std::uint32_t lane, SequenceState& sequence,
                        MaterializationTransaction& transaction);
    void release_materialization_staging(MaterializationTransaction& transaction) noexcept;
    [[nodiscard]] runtime::PrefillStepResult
    advance_prefill_raw(std::uint32_t lane, runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_raw(std::span<const std::uint32_t> lanes, std::span<const runtime::RoundBudget> budgets,
               runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] runtime::ExecutionTiming
    resolve_prefill_raw(std::uint32_t lane, bool terminal, runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] runtime::ExecutionTiming resolve_pending_raw(
        std::span<const std::uint32_t> lanes, std::span<const std::uint32_t> accepted_tokens,
        std::span<const std::uint8_t> terminal, std::span<const std::uint8_t> cancelled,
        std::span<const std::optional<std::uint32_t>> prefix_execution_splits,
        runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] bool valid_sequence(SequenceHandle handle) const noexcept;
    [[nodiscard]] bool valid_continuation(const ContinuationHandle& handle) const noexcept;
    [[nodiscard]] bool valid_shared_prefix(const SharedPrefixHandle& handle) const noexcept;
    [[nodiscard]] bool valid_capture_offer(const CaptureOffer& offer) const noexcept;
    [[nodiscard]] bool materialization_pins(std::uint32_t index,
                                            std::uint64_t generation) const noexcept;
    [[nodiscard]] bool has_unsettled_state_fork() const noexcept;
    [[nodiscard]] bool valid_pending(const PendingBatch& pending) const noexcept;
    // Per-owner resources are intentionally distinct from global physical occupancy: an aliased
    // allocation contributes only when removing this owner would release it.
    [[nodiscard]] detail::PhysicalResources
    owner_exclusive_resources(const SequenceState& sequence) const;
    [[nodiscard]] detail::PhysicalResources
    owner_exclusive_resources(const SharedPrefixState& shared) const;
    // An active sequence's KV entitlement leaves out the prefix pages another owner still
    // references (a retained private source, a shared prefix). Releasing that owner outside the
    // request's own materialization makes them exclusive to the active sequence; this adds them
    // to its entitlement, which a later shared capture subtracts by their reference count.
    void adopt_unshared_active_kv() noexcept;
    [[nodiscard]] detail::PhysicalResources physical_occupancy() const noexcept;
    [[nodiscard]] bool physical_peak_fits(detail::PhysicalResources peak) const noexcept;
    [[nodiscard]] StateImageHandle
    selected_state(const SequenceState& sequence, ReusePath reuse,
                   std::optional<runtime::CheckpointRef> checkpoint) const;
    [[nodiscard]] std::uint32_t
    selected_state_consumed_references(const SequenceState& sequence, ReusePath reuse,
                                       RewriteCheckpointDisposition rewrite_disposition,
                                       std::optional<runtime::CheckpointRef> checkpoint,
                                       std::uint32_t reuse_base) const;
    [[nodiscard]] bool
    selected_state_requires_fork(const SequenceState& sequence, ReusePath reuse,
                                 RewriteCheckpointDisposition rewrite_disposition,
                                 std::optional<runtime::CheckpointRef> checkpoint,
                                 std::uint32_t reuse_base) const;
    [[nodiscard]] bool can_retain_rewrite_checkpoint(const PreparedPromptData& prompt,
                                                     const RewriteCheckpointSpec& desired,
                                                     const SequenceState& sequence, ReusePath reuse,
                                                     std::uint32_t reuse_base) const;
    [[nodiscard]] std::uint32_t device_kv_prefix_pages(const KVAddressSpaceStore& addresses,
                                                       KVAddressSpaceHandle address,
                                                       std::uint32_t frontier) const;
    [[nodiscard]] std::uint32_t shared_kv_prefix_pages(const KVAddressSpaceStore& addresses,
                                                       KVAddressSpaceHandle address,
                                                       std::uint32_t frontier) const;
    [[nodiscard]] std::uint32_t shared_device_kv_prefix_pages(const KVAddressSpaceStore& addresses,
                                                              KVAddressSpaceHandle address,
                                                              std::uint32_t frontier) const;
    [[nodiscard]] bool partial_tail_cow_required(const KVAddressSpaceStore& addresses,
                                                 KVAddressSpaceHandle address,
                                                 std::uint32_t frontier) const;
    [[nodiscard]] std::uint32_t
    missing_shared_device_kv_prefix_pages(const KVAddressSpaceStore& addresses,
                                          KVAddressSpaceHandle address,
                                          std::uint32_t frontier) const;
    [[nodiscard]] std::size_t host_kv_prefix_bytes(const KVAddressSpaceStore& addresses,
                                                   KVAddressSpaceHandle address,
                                                   std::uint32_t frontier) const noexcept;
    [[nodiscard]] qwen3_5::CheckpointSummary
    checkpoint_summary(const SequenceState& sequence, runtime::CheckpointRef checkpoint,
                       StateImageHandle state, runtime::PrefillWork rebuild_work) const;
    [[nodiscard]] qwen3_5::ContinuationSummary
    continuation_summary(const SequenceState& sequence) const;
    void populate_continuation_summary(const SequenceState& sequence,
                                       qwen3_5::ContinuationSummary& summary) const;
    [[nodiscard]] qwen3_5::SharedPrefixSummary
    shared_prefix_summary(const SharedPrefixState& shared) const;
    [[nodiscard]] std::optional<MaterializationSourceProtection>
    materialization_source_protection(const ResourceCandidateState& candidate) const;
    [[nodiscard]] detail::PhysicalResources
    materialization_deficit(const ResourceCandidateState& candidate) const;
    [[nodiscard]] detail::PhysicalResources
    guided_materialization_deficit(const ResourceCandidateState& candidate,
                                   const detail::PhysicalDelta& pressure) const;
    [[nodiscard]] bool
    protected_materialization_page(const MaterializationSourceProtection* protection,
                                   const KVAddressSpaceStore& addresses, std::uint32_t page_offset,
                                   LogicalKVPageHandle page, bool backend) const;
    [[nodiscard]] std::optional<qwen3_5::detail::PressureDecision>
    inspect_pressure_option(const SequenceState& sequence, detail::PhysicalResources deficit,
                            const MaterializationSourceProtection* protection           = nullptr,
                            const qwen3_5::TargetKVRequirement* retained_requirement    = nullptr,
                            std::span<const runtime::CheckpointRef> dropped_checkpoints = {},
                            std::span<const StateImageHandle> released_states           = {},
                            const qwen3_5::detail::PressureDecision* current = nullptr) const;
    [[nodiscard]] std::vector<qwen3_5::detail::PressureDecision>
    inspect_pressure_successors(const SequenceState& sequence, detail::PhysicalResources residual,
                                const MaterializationSourceProtection* protection,
                                const qwen3_5::detail::PressureDecision* current = nullptr) const;
    [[nodiscard]] std::vector<qwen3_5::detail::PressureDecision> inspect_shared_pressure_successors(
        const SharedPrefixState& shared, detail::PhysicalResources residual,
        const MaterializationSourceProtection* protection,
        const qwen3_5::detail::PressureDecision* current = nullptr) const;
    [[nodiscard]] std::optional<qwen3_5::detail::PressureDecision> inspect_shared_pressure_option(
        const SharedPrefixState& shared, detail::PhysicalResources deficit,
        const MaterializationSourceProtection* protection = nullptr,
        const qwen3_5::detail::PressureDecision* current  = nullptr) const;
    [[nodiscard]] std::vector<qwen3_5::detail::PressureDecision> inspect_shared_pressure_options(
        const SharedPrefixState& shared, detail::PhysicalResources deficit,
        const MaterializationSourceProtection* protection = nullptr,
        const qwen3_5::detail::PressureDecision* current  = nullptr) const;
    [[nodiscard]] qwen3_5::detail::PressureDecision
    inspect_eviction_option(const SequenceState& sequence) const;
    [[nodiscard]] qwen3_5::detail::PressureDecision
    inspect_shared_eviction_option(const SharedPrefixState& shared) const;
    [[nodiscard]] std::optional<qwen3_5::detail::PressureDecision>
    inspect_checkpoint_drop_option(const SequenceState& sequence,
                                   std::span<const runtime::CheckpointRef> checkpoints) const;
    [[nodiscard]] bool
    pressure_decision_valid(const SequenceState& sequence,
                            const qwen3_5::detail::PressureDecision& decision,
                            const MaterializationSourceProtection* protection) const;
    [[nodiscard]] bool
    shared_pressure_decision_valid(const SharedPrefixState& shared,
                                   const qwen3_5::detail::PressureDecision& decision,
                                   const MaterializationSourceProtection* protection) const;
    [[nodiscard]] std::vector<runtime::ContextTransferRequirement>
    checkpoint_restore_requirements(const SequenceKVBundle& kv,
                                    const qwen3_5::TargetKVRequirement& requirement,
                                    StateImageHandle state) const;
    [[nodiscard]] bool pressure_checkpoint_recovery_impacts(
        const ResourceCandidateState& candidate,
        std::span<const ContinuationHandle* const> private_owners,
        std::span<const qwen3_5::detail::PressureDecision* const> private_decisions,
        std::span<const runtime::PlanningOwnerId> private_owner_ids,
        std::span<const SharedPrefixHandle* const> shared_owners,
        std::span<const qwen3_5::detail::PressureDecision* const> shared_decisions,
        std::span<const runtime::PlanningOwnerId> shared_owner_ids,
        std::vector<qwen3_5::detail::PressureCheckpointRecoveryProjection>& output,
        std::vector<runtime::CheckpointRecoveryAlternativeWork>& alternatives,
        PressureRecoveryScratch& scratch, std::uint64_t& projection_work) const;
    void publish_checkpoint_drop(SequenceState& sequence, runtime::CheckpointRef checkpoint);
    [[nodiscard]] PrefillProgress wrap_prefill(std::uint32_t lane, runtime::PrefillStepResult step);
    [[nodiscard]] PendingBatch wrap_pending(std::span<const std::uint32_t> lanes,
                                            const runtime::BatchedGeneratedRound& round);
    void invalidate_lane(std::uint32_t lane) noexcept;
    [[nodiscard]] SequenceState& active_sequence(std::uint32_t lane);
    [[nodiscard]] const SequenceState& active_sequence(std::uint32_t lane) const;
    [[nodiscard]] std::optional<std::uint32_t> allocate_continuation_slot() noexcept;
    [[nodiscard]] bool can_release_continuation_slot_strict(std::uint32_t index) const;
    void release_continuation_slot_strict(std::uint32_t index) noexcept;
    void release_continuation_slot_best_effort(std::uint32_t index) noexcept;
    void retire_continuation_slot(std::uint32_t index) noexcept;
    void clear_execution_failure_lanes(std::span<const std::uint32_t> lanes) noexcept;
    [[nodiscard]] bool can_clear_lane_strict(const SequenceState& sequence) const;
    [[nodiscard]] bool clear_lane_strict(SequenceState& sequence, RequestControl& request) noexcept;
    void clear_lane_best_effort(SequenceState& sequence, RequestControl& request) noexcept;
    void ordered_reset(SequenceState& sequence);
    [[nodiscard]] StateImageSelectors state_selectors(const SequenceState& sequence) const;
    [[nodiscard]] detail::PhysicalResources
    sequence_exclusive_state_resources(const SequenceState& sequence) const;
    [[nodiscard]] std::uint32_t owned_checkpoint_references(const SequenceState& sequence,
                                                            StateImageHandle state) const noexcept;
    [[nodiscard]] bool state_exclusive_to_sequence(const SequenceState& sequence,
                                                   StateImageHandle state) const noexcept;
    [[nodiscard]] bool compose_pressure_candidate(
        ResourceCandidateState& candidate,
        std::span<const ContinuationHandle* const> pressure_owners,
        std::span<const runtime::PlanningOwnerId> pressure_owner_ids,
        std::span<const qwen3_5::detail::PressureDecision* const> pressure_options,
        std::span<const SharedPrefixHandle* const> shared_pressure_owners,
        std::span<const runtime::PlanningOwnerId> shared_pressure_owner_ids,
        std::span<const qwen3_5::detail::PressureDecision* const> shared_pressure_options);
    [[nodiscard]] std::optional<detail::PressureTargetProjection> evaluate_pressure_target(
        const MaterializationSourceProtection* protection,
        std::span<const ContinuationHandle* const> pressure_owners,
        std::span<const qwen3_5::detail::PressureDecision> pressure_options,
        std::span<const SharedPrefixHandle* const> shared_pressure_owners,
        std::span<const qwen3_5::detail::PressureDecision> shared_pressure_options,
        std::vector<HostKVPageReplicaRelease>* released_host_pages) const;
    void refresh_state_views(SequenceState& sequence);
    void reserve_state_entitlement(SequenceState& sequence, std::uint32_t slots);
    void settle_state_fork(SequenceState& sequence);
    [[nodiscard]] detail::PhysicalResources
    release_checkpoint_reference(StateImageHandle checkpoint) noexcept;
    [[nodiscard]] bool can_release_shared_prefix_state(std::uint32_t index,
                                                       SharedPrefixSlotRole expected_role) const;
    [[nodiscard]] detail::PhysicalResources
    release_shared_prefix_state_strict(std::uint32_t index,
                                       SharedPrefixSlotRole expected_role) noexcept;
    [[nodiscard]] detail::PhysicalResources
    install_private_capture(SequenceState& sequence, const CaptureGroup& group,
                            StateImageHandle checkpoint,
                            std::optional<runtime::CheckpointRef> replacement);
    void prepare_active_capture(ActiveCaptureTransaction& transaction);
    void enqueue_active_capture_transfers(ActiveCaptureTransaction& transaction);
    void abort_active_capture(ActiveCaptureTransaction& transaction) noexcept;
    [[nodiscard]] ActiveCaptureResult publish_active_capture(ActiveCaptureTransaction& transaction);
    void release_active_shared_references_strict(SequenceState& sequence) noexcept;
    void release_active_shared_references(SequenceState& sequence) noexcept;
    void release_active_sequence_state_strict(SequenceState& sequence) noexcept;
    void release_sequence_state_strict(SequenceState& sequence) noexcept;
    void release_sequence_state(SequenceState& sequence) noexcept;
    void prepare_graphs();
    void install_sampling(SequenceState& sequence, RequestControl& request,
                          const ops::SamplingConfig& config, std::uint32_t accepted_length_steps,
                          std::uint32_t output_tokens);
    void set_device_i32(Tensor& tensor, std::int32_t value);
    // Retains column `column` of the final-normed prefill chunk as `sequence`'s target tail
    // hidden. At tensor-parallel width 2 under MTP, rank 1 retains the same column of its own
    // chunk in the same StateImage slot of its mirror pool.
    void copy_tail(SequenceState& sequence, std::int32_t column);
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
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_ordinary_batch(std::span<const std::uint32_t> lanes,
                          std::span<const runtime::RoundBudget> budgets,
                          runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_mtp_batch(std::span<const std::uint32_t> lanes,
                     std::span<const runtime::RoundBudget> budgets,
                     runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_dflash_batch(std::span<const std::uint32_t> lanes,
                        std::span<const runtime::RoundBudget> budgets,
                        runtime::ExecutionTiming* failed_timing);
    void resize_sequence_kv_entitlement(SequenceState& sequence, std::uint32_t text_pages,
                                        std::uint32_t backend_pages);
    void bind_sequence_kv(SequenceState& sequence);
    void unbind_sequence_kv(SequenceState& sequence) noexcept;

    // The output window a lease holds and extends by. 0 (--kv-lease-window 0) reserves the
    // whole max_tokens at admission: the window then covers every answer and nothing grows.
    [[nodiscard]] std::uint32_t kv_lease_growth_margin_tokens() const noexcept {
        return kv_lease_window_tokens == 0 ? std::numeric_limits<std::uint32_t>::max()
                                           : std::max(prefill_chunk, kv_lease_window_tokens);
    }

    // One round's Backend requirement can sit a whole draft window above the frontier the
    // previous round checked, so the cushion absorbs that jump before the lease grows again.
    [[nodiscard]] std::uint32_t kv_lease_cushion_pages() const noexcept {
        const auto page  = static_cast<std::uint32_t>(kPagedKVPageSize);
        const auto slack = 2U * draft_window + 2U;
        return (slack + page - 1U) / page + 1U;
    }

    // The Backend lease also covers the drafts a round may still verify past the output ceiling,
    // and one forced control span.
    [[nodiscard]] std::uint32_t kv_lease_backend_allowance_tokens() const noexcept {
        return draft_window + std::min(draft_window, qwen3_5::kMtpDecodeMaximumDrafts) + 1U;
    }

    [[nodiscard]] std::uint32_t kv_lease_pages_for_tokens(std::uint32_t tokens) const noexcept {
        return kv_pages_for_tokens(tokens) + kv_lease_cushion_pages();
    }

    [[nodiscard]] bool run_lease_demotion(MaterializationTransaction::PressureWork& work) noexcept;
    void ensure_sequence_kv_lease(SequenceState& sequence, std::uint32_t main_tokens,
                                  std::uint32_t backend_tokens);
    void ensure_sequence_kv_mapped(SequenceState& sequence, std::uint32_t main_tokens,
                                   std::uint32_t backend_tokens = 0);
    void trim_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                          std::uint32_t backend_tokens = 0);
    void release_sequence_growth_entitlement(SequenceState& sequence) noexcept;
    void release_active_sequence_kv_strict(SequenceState& sequence) noexcept;
    void release_sequence_kv_strict(SequenceState& sequence) noexcept;
    void release_sequence_kv(SequenceState& sequence) noexcept;
    void commit_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                            std::uint32_t backend_tokens = 0);
    [[nodiscard]] qwen3_5::PagedKVCache* backend_kv_cache() noexcept;
    [[nodiscard]] const qwen3_5::PagedKVCache* backend_kv_cache() const noexcept;
    [[nodiscard]] std::uint32_t backend_kv_valid(const SequenceState& sequence) const noexcept;
    [[nodiscard]] qwen3_5::PagedKVCacheView text_kv_view(const SequenceState& sequence) const;
    [[nodiscard]] qwen3_5::PagedKVCacheView mtp_kv_view(const SequenceState& sequence) const;

#include "models/qwen3_5/program/tp2/program_impl_private.inc"
};

} // namespace ninfer::models::qwen3_5::detail

namespace ninfer::models::qwen3_5::detail {

struct PressurePlanningSessionImpl {
    using Core                         = ProgramImpl;
    using AdmissionCandidate           = qwen3_5::AdmissionCandidate;
    using CapturePressureCandidate     = qwen3_5::CapturePressureCandidate;
    using AdmissionCandidateImpl       = qwen3_5::detail::AdmissionCandidateImpl;
    using CapturePressureCandidateImpl = qwen3_5::detail::CapturePressureCandidateImpl;
    using CandidateState               = qwen3_5::detail::ResourceCandidateState;
    using ContinuationHandle           = qwen3_5::ContinuationHandle;
    using SharedPrefixHandle           = qwen3_5::SharedPrefixHandle;

    struct Owner {
        const ContinuationHandle* private_handle = nullptr;
        const SharedPrefixHandle* shared_handle  = nullptr;
        runtime::PlanningOwnerId id;
        bool shared = false;
    };

    struct PhysicalCandidateBinding {
        const CandidateState* state                 = nullptr;
        const AdmissionCandidateImpl* admission     = nullptr;
        const CapturePressureCandidateImpl* capture = nullptr;
    };

    struct TargetNode {
        std::uint32_t candidate_index      = 0;
        std::uint32_t victim_choice_offset = 0;
        std::uint32_t victim_choice_count  = 0;
        std::optional<detail::PhysicalResources> assessed_residual;
        std::uint32_t next_expansion_owner = 0;
        std::uint32_t stable_ordinal       = 0;
        bool root_maximal                  = false;
    };

    struct CandidateVictimOptions {
        std::uint32_t owner_index = 0;
        std::vector<PressureDecision> decisions;
        std::uint16_t eviction_choice = 0;
    };

    struct CandidateOptions {
        std::vector<CandidateVictimOptions> victims;
        bool populated = false;
    };

    struct PreparedOwnerDecision {
        std::uint32_t candidate_index = 0;
        std::uint32_t victim_index    = 0;
        std::uint16_t choice          = 0;
        PressureDecision decision;
    };

    struct AssessmentSlot {
        std::vector<runtime::PressureOwnerOutcome> owner_outcomes;
        std::vector<runtime::PressureCheckpointRecoveryImpact> checkpoint_impacts;
        std::vector<runtime::CheckpointRecoveryAlternativeWork> recovery_alternatives;
        std::uint32_t generation = 1;
        bool leased              = false;
    };

    struct ConstructionOption {
        std::size_t victim = 0;
        PressureDecision decision;
        bool identity = false;
    };

    struct ConstructionSlot {
        std::vector<std::uint16_t> choices;
        std::vector<ConstructionOption> options;
        std::size_t next_owner        = 0;
        std::size_t next_option       = 0;
        std::uint32_t candidate_index = 0;
        std::uint32_t generation      = 0;
        std::uint32_t scan_generation = 1;
        detail::PhysicalResources residual;
        bool restore = false;
        bool leased  = false;
    };

    PressurePlanningSessionImpl(
        Core& owner, std::span<const PhysicalCandidateBinding> physical_candidates,
        std::span<const runtime::PlanningCandidateId> admission_candidate_ids,
        std::span<const ContinuationHandle* const> private_owners,
        std::span<const runtime::PlanningOwnerId> private_owner_ids,
        std::span<const SharedPrefixHandle* const> shared_owners,
        std::span<const runtime::PlanningOwnerId> shared_owner_ids);
    ~PressurePlanningSessionImpl() noexcept;

    [[nodiscard]] qwen3_5::PressureTargetHandle
    identity_target(runtime::PlanningCandidateId candidate) const;
    [[nodiscard]] qwen3_5::PressureTargetHandle
    root_maximal_target(runtime::PlanningCandidateId root_candidate);
    [[nodiscard]] qwen3_5::PressureTargetHandle
    maximal_target(runtime::PlanningCandidateId candidate);
    [[nodiscard]] qwen3_5::PressureConstructionCursor
    begin_construction(qwen3_5::PressureTargetHandle target, bool restore = false);
    [[nodiscard]] runtime::PressureConstructionStep
    next_construction_option(qwen3_5::PressureConstructionCursor& cursor);
    void choose_construction(qwen3_5::PressureConstructionCursor& cursor,
                             runtime::PressureConstructionOptionId option);
    [[nodiscard]] std::optional<qwen3_5::PressureTargetHandle>
    construction_target(const qwen3_5::PressureConstructionCursor& cursor);
    [[nodiscard]] ConstructionSlot&
    construction_slot(const qwen3_5::PressureConstructionCursor& cursor);
    static void release_construction(const void*, std::uint32_t, std::uint32_t) noexcept;
    [[nodiscard]] runtime::PressureTargetGuidance
    guidance_choices(std::uint32_t candidate_index, std::span<const std::uint16_t> choices,
                     std::uint32_t ordinal,
                     std::optional<std::size_t> override_owner = std::nullopt,
                     const PressureDecision* override_decision = nullptr);
    [[nodiscard]] detail::PhysicalResources
    construction_residual(std::uint32_t candidate_index,
                          std::span<const std::uint16_t> choices) const;
    [[nodiscard]] runtime::PressureTargetGuidance guidance(qwen3_5::PressureTargetHandle target);
    [[nodiscard]] qwen3_5::AssessedPressureTarget assess(qwen3_5::PressureTargetHandle target);
    [[nodiscard]] qwen3_5::PreparedPressureExpansion
    prepare_expansion(qwen3_5::PressureTargetHandle parent,
                      std::uint32_t maximum_owners = std::numeric_limits<std::uint32_t>::max());
    [[nodiscard]] qwen3_5::PressureExpansionView
    commit_expansion(qwen3_5::PreparedPressureExpansion&& prepared);
    void discard_expansion(qwen3_5::PreparedPressureExpansion&& prepared) noexcept;
    [[nodiscard]] runtime::PrefillWork
    shared_capture_split_prefill_work(const qwen3_5::AssessedPressureTarget& assessed,
                                      const PreparedPromptData& prompt,
                                      std::span<const std::uint32_t> frontiers) const;
    [[nodiscard]] std::optional<AdmissionCandidate> seal(qwen3_5::AssessedPressureTarget&& assessed,
                                                         const PreparedPromptData& prompt,
                                                         runtime::FinalScheduleIntent intent);
    [[nodiscard]] std::optional<CapturePressureCandidate>
    seal_capture(qwen3_5::AssessedPressureTarget&& assessed);

    [[nodiscard]] bool valid(qwen3_5::PressureTargetHandle target) const noexcept;
    [[nodiscard]] std::uint32_t candidate_index(runtime::PlanningCandidateId candidate) const;
    [[nodiscard]] std::span<const std::uint16_t> victim_choices(const TargetNode& target) const;
    [[nodiscard]] TargetNode* find_target(std::uint32_t candidate_index,
                                          std::span<const std::uint16_t> choices) noexcept;
    [[nodiscard]] const TargetNode*
    find_target(std::uint32_t candidate_index,
                std::span<const std::uint16_t> choices) const noexcept;
    [[nodiscard]] std::uint32_t intern_target(std::uint32_t candidate_index,
                                              std::span<const std::uint16_t> choices,
                                              bool root_maximal = false);
    void index_target(std::uint32_t target_index);
    void populate_options(std::uint32_t candidate_index);
    [[nodiscard]] std::vector<PressureDecision>
    pressure_successors(const CandidateVictimOptions& victim_options,
                        const detail::PhysicalResources& residual,
                        const Core::MaterializationSourceProtection& protection,
                        const PressureDecision* current) const;
    [[nodiscard]] std::uint32_t acquire_assessment_slot();
    static void release_assessment_slot(const void* owner, std::uint32_t slot,
                                        std::uint32_t generation) noexcept;

    Core* program = nullptr;
    runtime::ProgramResourceRevision resource_revision;
    std::uint32_t generation         = 1;
    std::uint32_t scratch_generation = 1;
    std::vector<PhysicalCandidateBinding> candidates;
    std::vector<runtime::PlanningCandidateId> candidate_ids;
    std::vector<Owner> owners;
    std::vector<CandidateOptions> candidate_options;
    std::vector<TargetNode> targets;
    std::vector<std::uint16_t> target_choice_arena;
    std::vector<std::uint16_t> choice_scratch;
    std::vector<std::uint32_t> target_hash_table;
    std::vector<TargetNode> expansion_scratch;
    std::vector<PreparedOwnerDecision> prepared_owner_decisions;
    std::vector<qwen3_5::PressureTargetHandle> committed_children;
    std::vector<const ContinuationHandle*> selected_private_owners;
    std::vector<runtime::PlanningOwnerId> selected_private_owner_ids;
    std::vector<const PressureDecision*> selected_private_decisions;
    std::vector<const SharedPrefixHandle*> selected_shared_owners;
    std::vector<runtime::PlanningOwnerId> selected_shared_owner_ids;
    std::vector<const PressureDecision*> selected_shared_decisions;
    std::vector<const ContinuationHandle*> recovery_private_owners;
    std::vector<const PressureDecision*> recovery_private_decisions;
    std::vector<runtime::PlanningOwnerId> recovery_private_owner_ids;
    std::vector<const SharedPrefixHandle*> recovery_shared_owners;
    std::vector<const PressureDecision*> recovery_shared_decisions;
    std::vector<runtime::PlanningOwnerId> recovery_shared_owner_ids;
    std::vector<const PressureDecision*> projected_owner_decisions;
    std::vector<runtime::PressureOwnerOutcome> assessment_outcomes;
    std::vector<PressureCheckpointRecoveryProjection> assessment_impact_projections;
    std::vector<runtime::CheckpointRecoveryAlternativeWork> assessment_recovery_alternatives;
    Core::PressureRecoveryScratch recovery_scratch;
    std::vector<runtime::PressureOwnerOutcome> guidance_outcomes;
    std::vector<runtime::PressureCheckpointOutcome> guidance_checkpoint_changes;
    std::vector<runtime::PressureOwnerRecoveryGuidance> guidance_recovery;
    std::array<ConstructionSlot, 4> construction_slots;
    std::uint32_t construction_generation = 0;
    std::array<AssessmentSlot, 2> assessment_slots;
    std::uint32_t prepared_new_count = 0;
    std::uint32_t prepared_owner_end = 0;
    std::size_t scratch_choice_mark  = 0;
    bool scratch_live                = false;
};



} // namespace ninfer::models::qwen3_5::detail
