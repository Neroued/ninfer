#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "core/arena.h"
#include "core/gdn_replay_records.h"
#include "core/host_kv_arena.h"
#include "runtime/engine/context_cost.h"
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/ops/sampling.h"
#include "core/decode_graph.h"
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include "targets/qwen3_6/impl/runtime/layouts.h"
#include "targets/qwen3_6/impl/runtime/dflash_context.h"
#include "targets/qwen3_6/impl/runtime/host_kv_extent_store.h"
#include "targets/qwen3_6/impl/runtime/logical_kv_store.h"
#include "targets/qwen3_6/impl/runtime/state_image_store.h"
#include "targets/qwen3_6/impl/runtime/prefix_identity.h"
#include "targets/qwen3_6/impl/runtime/text_context.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"
#include "targets/qwen3_6/impl/runtime/vision_prefill.h"

#include <algorithm>
#include <cstdint>
#include <array>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using PreparedPromptData    = qwen3_6::PreparedPromptData;
using RewriteCheckpointKind = qwen3_6::RewriteCheckpointKind;
using RewriteCheckpointSpec = qwen3_6::RewriteCheckpointSpec;

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
    qwen3_6::detail::ResidentPrefixIdentity prefix_identity;
};

struct PreparedCaptureIdentity {
    std::shared_ptr<const PreparedCaptureBacking> backing;
    qwen3_6::PrefixShortlistKey shortlist_key;
    runtime::PrefillWork rebuild_work;

    [[nodiscard]] std::span<const TokenId> ledger() const noexcept {
        if (!backing || shortlist_key.frontier > backing->ledger.size()) { return {}; }
        return std::span<const TokenId>(backing->ledger).first(shortlist_key.frontier);
    }

    [[nodiscard]] const qwen3_6::detail::ResidentPrefixIdentity* prefix_identity() const noexcept {
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

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

namespace ninfer::targets::qwen3_6::detail {

// Concrete allocator quantities never cross the Program implementation boundary.
struct PhysicalDeviceResources {
    std::uint32_t active_lanes     = 0;
    std::uint32_t state_slots      = 0;
    std::uint32_t main_kv_pages    = 0;
    std::uint32_t backend_kv_pages = 0;

    [[nodiscard]] friend constexpr bool operator==(PhysicalDeviceResources,
                                                   PhysicalDeviceResources) noexcept = default;
};

struct PhysicalHostResources {
    std::uint32_t state_slots = 0;
    std::size_t kv_bytes      = 0;

    [[nodiscard]] friend constexpr bool operator==(PhysicalHostResources,
                                                   PhysicalHostResources) noexcept = default;
};

struct PhysicalResources {
    PhysicalDeviceResources device;
    PhysicalHostResources host;

    [[nodiscard]] friend constexpr bool operator==(const PhysicalResources&,
                                                   const PhysicalResources&) noexcept = default;
};

struct PhysicalDemand {
    PhysicalResources active_entitlement;
    PhysicalResources reservation_added;
    PhysicalResources reservation_credit;
    PhysicalResources physical_peak_additional;
    PhysicalResources final_removed;
    PhysicalResources final_added;

    [[nodiscard]] friend constexpr bool operator==(const PhysicalDemand&,
                                                   const PhysicalDemand&) noexcept = default;
};

struct PhysicalDelta {
    PhysicalResources removed;
    PhysicalResources added;

    [[nodiscard]] friend constexpr bool operator==(const PhysicalDelta&,
                                                   const PhysicalDelta&) noexcept = default;
};

struct PhysicalPressureEffect {
    PhysicalDelta aggregate_delta;
    PhysicalDelta final_ownership_delta;
    PhysicalDelta active_entitlement_delta;
    PhysicalResources source_optional_resources_added;
    // Pressure can remove the last checkpoint references outside a consumed private source.  The
    // selected StateImage must then be moved into the active lineage instead of forked.  This is a
    // property of the complete post-reference target, not of an individual owner decision.
    std::optional<bool> source_state_fork_required;
    std::optional<bool> source_text_prefix_fork_required;
    std::optional<bool> source_backend_prefix_fork_required;

    [[nodiscard]] friend constexpr bool
    operator==(const PhysicalPressureEffect&, const PhysicalPressureEffect&) noexcept = default;
};

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

struct CaptureAssessmentImpl {
    PhysicalDemand demand;
    PhysicalDelta active_entitlement_delta;
    PhysicalResources capacity_preparation_removed;
};

template <>
struct RequestBasePlanImpl<NINFER_QWEN36_VARIANT> {
    runtime::RequestPlanSummary summary;
    detail::PhysicalDemand root_demand;
    runtime::PrefillWork root_rebuild_work;
    std::uint32_t root_rebuild_tail_begin = 0;
    qwen3_6::PreparedContextCache context_cache;
    ops::SamplingConfig sampling;
    std::uint32_t text_kv_page_entitlement    = 0;
    std::uint32_t backend_kv_page_entitlement = 0;
    std::shared_ptr<const qwen3_6::VisionControlPlan> vision_control_plan;
    std::optional<qwen3_6::RewriteCheckpointSpec> rewrite_checkpoint;
    std::vector<NINFER_QWEN36_RUNTIME_NS::CaptureGroup> capture_groups;
    std::vector<NINFER_QWEN36_RUNTIME_NS::CaptureGroup> shared_candidates;
    qwen3_6::detail::PrefixShortlistDigests prefix_digests;
    std::uint32_t prefix_identity_tag = 0;
    bool allow_prefix_reuse           = false;
};

template <>
struct AdmissionCandidateImpl<NINFER_QWEN36_VARIANT> {
    runtime::RequestPlanSummary summary;
    runtime::IdentityMaterializationAssessment identity_assessment;
    std::uint64_t planning_revision = 0;
    // Pressure outcomes are canonicalized against the candidate's identity peak.  Composition
    // rewrites demand to the selected post-pressure peak, so regenerating an outcome from that
    // rewritten demand would compare it against a different problem at seal time.
    detail::PhysicalResources identity_pressure_deficit;
    // A structurally valid pressure target can still be blocked by Host extent geometry even when
    // aggregate free bytes are sufficient. Keep the blocked allocation work explicit so a child
    // target can release Host replicas instead of being mistaken for a structurally invalid node.
    std::size_t blocked_host_allocation_bytes = 0;
    detail::PhysicalDemand demand;
    detail::PhysicalResources source_resources;
    NINFER_QWEN36_RUNTIME_NS::ReusePath reuse = NINFER_QWEN36_RUNTIME_NS::ReusePath::Root;
    std::uint32_t reuse_base                  = 0;
    NINFER_QWEN36_RUNTIME_NS::MtpBridgeMode mtp_bridge =
        NINFER_QWEN36_RUNTIME_NS::MtpBridgeMode::None;
    bool prepare_mtp = false;
    std::optional<NINFER_QWEN36_RUNTIME_NS::VisionPrefillPlan> vision;
    NINFER_QWEN36_RUNTIME_NS::RewriteCheckpointDisposition rewrite_disposition =
        NINFER_QWEN36_RUNTIME_NS::RewriteCheckpointDisposition::DropOptional;
    std::vector<NINFER_QWEN36_RUNTIME_NS::CaptureGroup> capture_groups;
    std::vector<NINFER_QWEN36_RUNTIME_NS::CaptureGroup> shared_candidates;
    ops::SamplingConfig sampling;
    std::uint32_t text_kv_page_entitlement    = 0;
    std::uint32_t backend_kv_page_entitlement = 0;
    runtime::LaneId destination{};
    std::uint64_t destination_epoch = 0;
    bool has_source                 = false;
    bool has_shared_source          = false;
    std::optional<runtime::CheckpointRef> selected_checkpoint;
    std::uint32_t source_index             = 0;
    std::uint64_t source_generation        = 0;
    std::uint32_t shared_source_index      = 0;
    std::uint64_t shared_source_generation = 0;
    runtime::PrefillWork root_rebuild_work;
    std::uint32_t root_rebuild_tail_begin = 0;
    runtime::PrefillWork remaining_prefill_work;
    std::vector<runtime::ContextTransferRequirement> transfer_requirements;
    runtime::ClaimDisposition source_disposition = runtime::ClaimDisposition::ConsumedToActive;
    detail::PhysicalResources active_optional_resources;
    bool state_fork_required           = false;
    bool text_prefix_fork_required     = false;
    bool backend_prefix_fork_required  = false;
    bool text_retained_tail_release    = false;
    bool backend_retained_tail_release = false;
    bool needs_transfer                = false;
    bool capture_pressure              = false;
    std::vector<qwen3_6::detail::PressureDecision> pressure_options;
    std::vector<std::uint32_t> pressure_indices;
    std::vector<std::uint64_t> pressure_generations;
    std::vector<qwen3_6::detail::PressureDecision> shared_pressure_options;
    std::vector<std::uint32_t> shared_pressure_indices;
    std::vector<std::uint64_t> shared_pressure_generations;
};

} // namespace ninfer::targets::qwen3_6::detail

namespace ninfer::targets::qwen3_6 {

inline CaptureAssessment::CaptureAssessment()
    : implementation(std::make_shared<detail::CaptureAssessmentImpl>()) {}

template <>
AdmissionCandidate<NINFER_QWEN36_VARIANT>::AdmissionCandidate(
    std::unique_ptr<detail::AdmissionCandidateImpl<NINFER_QWEN36_VARIANT>> impl) noexcept
    : impl_(std::move(impl)) {}

template <>
AdmissionCandidate<NINFER_QWEN36_VARIANT>::AdmissionCandidate(AdmissionCandidate&&) noexcept =
    default;

template <>
AdmissionCandidate<NINFER_QWEN36_VARIANT>&
AdmissionCandidate<NINFER_QWEN36_VARIANT>::operator=(AdmissionCandidate&&) noexcept = default;

template <>
AdmissionCandidate<NINFER_QWEN36_VARIANT>::~AdmissionCandidate() = default;

template <>
const runtime::RequestPlanSummary&
AdmissionCandidate<NINFER_QWEN36_VARIANT>::summary() const noexcept {
    static const runtime::RequestPlanSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

template <>
const runtime::IdentityMaterializationAssessment&
AdmissionCandidate<NINFER_QWEN36_VARIANT>::identity_assessment() const noexcept {
    static const runtime::IdentityMaterializationAssessment empty;
    return impl_ != nullptr ? impl_->identity_assessment : empty;
}

} // namespace ninfer::targets::qwen3_6

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using AdmissionCandidateImpl = qwen3_6::detail::AdmissionCandidateImpl<Variant>;
using RequestBasePlanImpl    = qwen3_6::detail::RequestBasePlanImpl<Variant>;

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
    qwen3_6::detail::ResidentPrefixIdentity prefix_identity;
    qwen3_6::detail::PrefixShortlistDigests prefix_digests;
    std::int32_t rope_delta               = 0;
    std::uint32_t text_kv_valid           = 0;
    std::uint32_t mtp_kv_valid            = 0;
    std::uint32_t dflash_context_frontier = 0;
    std::array<TokenId, qwen3_6::kMtpDecodeMaximumDrafts> mtp_drafts{};
    std::uint32_t mtp_draft_count = 0;
    bool tail_hidden_valid        = false;
    bool state_source_retained    = false;
    bool endpoint_valid           = false;
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
    detail::PhysicalResources active_resources;
    detail::PhysicalResources optional_resources;
    bool publish_continuation = true;

    struct Prefill {
        PreparedPromptData prompt;
        std::optional<VisionPrefillPlan> vision_plan;
        std::unique_ptr<schedule::VisionPrefillSession> vision;
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
    };

    std::optional<Prefill> prefill;
};

class ProgramImplCore {
public:
    ProgramImplCore(const LoadedModelData& model, const SequencePlanImpl& plan,
                    DeviceContext& device);
    ~ProgramImplCore() noexcept;

    [[nodiscard]] RequestBasePlan plan_request(const PreparedPromptData& prompt,
                                               const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] std::vector<float> causal_score(PreparedPromptData&& prompt,
                                                  std::uint32_t first_target);
    [[nodiscard]] std::optional<AdmissionCandidate> inspect_admission(
        const PreparedPromptData& prompt, const RequestBasePlan& base, runtime::LaneId destination,
        const ContinuationHandle* source, const SharedPrefixHandle* shared_source,
        std::optional<runtime::CheckpointRef> checkpoint, bool must_retain_private_source,
        const runtime::ContextMachineCostModel& machine_cost);
    [[nodiscard]] std::optional<AdmissionCandidate> seal_materialization(
        const AdmissionCandidate& admission, const PreparedPromptData& prompt,
        std::span<const ContinuationHandle* const> pressure_owners,
        std::span<const qwen3_6::detail::PressureDecision> pressure_options,
        std::span<const SharedPrefixHandle* const> shared_pressure_owners,
        std::span<const qwen3_6::detail::PressureDecision> shared_pressure_options);
    [[nodiscard]] AdmissionCandidate
    make_capture_pressure_candidate(const CaptureAssessment& assessment,
                                    const runtime::ContextMachineCostModel& machine_cost) const;
    void select_shared_captures(AdmissionCandidate& candidate, const PreparedPromptData& prompt,
                                std::span<const std::uint32_t> frontiers);
    [[nodiscard]] std::uint64_t
    shared_capture_split_cost_ns(const AdmissionCandidate& candidate,
                                 const PreparedPromptData& prompt,
                                 std::span<const std::uint32_t> frontiers,
                                 const runtime::ContextMachineCostModel& machine_cost);
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
    [[nodiscard]] ContextTransactionProgress<Variant>
    progress_context_transaction(runtime::CancellationFlagView cancellation);
    void finalize_context_transaction() noexcept;
    [[nodiscard]] bool has_context_transaction() const noexcept;
    [[nodiscard]] PrefillProgress advance_prefill(SequenceHandle sequence,
                                                  runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] CaptureAssessment
    inspect_capture(const CaptureOffer& offer, const SharedPrefixHandle* exact_shared,
                    const SharedPrefixHandle* replacement,
                    std::optional<runtime::CheckpointRef> private_replacement,
                    bool permit_shared_publication,
                    const runtime::ContextMachineCostModel& machine_cost) const;
    [[nodiscard]] std::uint64_t
    checkpoint_recovery_ns(const ContinuationHandle& owner, runtime::CheckpointRef checkpoint,
                           const runtime::ContextMachineCostModel& machine_cost) const;
    [[nodiscard]] std::uint64_t
    checkpoint_recovery_ns(const SharedPrefixHandle& owner, runtime::CheckpointRef checkpoint,
                           const runtime::ContextMachineCostModel& machine_cost) const;
    [[nodiscard]] bool shared_capture_matches(const CaptureOffer& offer,
                                              const SharedPrefixHandle& shared) const;
    void skip_capture(CaptureOffer&& offer);
    [[nodiscard]] runtime::ContextTransactionReserveStatus
    reserve_active_capture(CaptureOffer&& offer, const SharedPrefixHandle* exact_shared,
                           const SharedPrefixHandle* replacement,
                           std::optional<runtime::CheckpointRef> private_replacement,
                           bool permit_shared_publication,
                           const runtime::ContextMachineCostModel& machine_cost,
                           runtime::CancellationFlagView cancellation);
    [[nodiscard]] runtime::ContextTransactionReserveStatus reserve_active_capture_with_pressure(
        CaptureOffer&& offer, const SharedPrefixHandle* exact_shared,
        const SharedPrefixHandle* replacement,
        std::optional<runtime::CheckpointRef> private_replacement, bool permit_shared_publication,
        AdmissionCandidate&& pressure, const runtime::ContextMachineCostModel& machine_cost,
        runtime::CancellationFlagView cancellation);
    [[nodiscard]] PendingBatch decode(std::span<const SequenceHandle> sequences,
                                      std::span<const runtime::RoundBudget> budgets,
                                      runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] runtime::ExecutionTiming
    append_forced_tokens(std::span<const SequenceHandle> sequences,
                         std::span<const TokenId> row_major_tokens, std::uint32_t row_stride,
                         runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] CommitResult commit(PendingBatch&& pending,
                                      std::span<const runtime::CommitDecision> decisions,
                                      runtime::CommitObservation observation,
                                      runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] DiscardResult abort_pending(PendingBatch&& pending) noexcept;
    [[nodiscard]] FinishResult finish(SequenceHandle sequence) noexcept;
    [[nodiscard]] AbortResult abort(SequenceHandle sequence) noexcept;
    [[nodiscard]] ReleaseResult release_continuation(ContinuationHandle&& continuation) noexcept;
    [[nodiscard]] ReleaseResult release_shared_prefix(SharedPrefixHandle&& shared) noexcept;
    void fail_all_cleanup() noexcept;
    [[nodiscard]] detail::PhysicalResources admission_capacity() const noexcept;
    [[nodiscard]] bool isolated_request_feasible(const RequestBasePlan& base) const noexcept;

    [[nodiscard]] std::uint64_t resource_revision() const noexcept { return resource_revision_; }

    [[nodiscard]] qwen3_6::PhysicalUsageSnapshot physical_usage() const noexcept;

    [[nodiscard]] MemorySummary memory_summary() const noexcept;

    void reset_memory_peaks() noexcept;

    friend struct qwen3_6::detail::PressurePlanningSessionImpl<Variant>;

    const LoadedModelData& model;
    DeviceContext& device;
    const std::uint32_t capacity;
    const std::uint32_t kv_capacity;
    const std::uint32_t max_concurrency;
    const ContextCacheOptions context_cache;
    const std::uint32_t continuation_capacity;
    const std::uint32_t shared_prefix_capacity;
    const std::uint32_t prefill_chunk;
    const std::uint32_t draft_window;
    const SpeculativeBackend speculative_backend;
    const DType kv_dtype;
    const std::int32_t kv_quant_group;
    const ProposalHead proposal_head;
    const bool vision_enabled;
    const bool use_cuda_graph;
    const bool causal_scoring;
    const std::size_t kv_payload_bytes;
    const std::size_t graph_allowance_bytes;
    const WorkspacePlan workspace_plan;

    DeviceArena persistent;
    DeviceArena workspace_storage;
    WorkspaceArena work;
    std::unique_ptr<qwen3_6::DecoderState> decoder;
    std::unique_ptr<HostKVArena> host_kv_arena;
    std::unique_ptr<LogicalKVPageStore> text_kv_pages;
    std::unique_ptr<KVAddressSpaceStore> text_kv_addresses;
    std::unique_ptr<LogicalKVPageStore> backend_kv_pages;
    std::unique_ptr<KVAddressSpaceStore> backend_kv_addresses;
    std::unique_ptr<HostKVExtentStore> host_kv_extents;
    std::size_t text_host_kv_page_stride    = 0;
    std::size_t backend_host_kv_page_stride = 0;
    std::unique_ptr<qwen3_6::StateImageDevicePool> state_images;
    std::unique_ptr<qwen3_6::HostStatePool> host_state_images;
    std::unique_ptr<StateImageStore> state_store;
    std::optional<GdnReplayRecords> replay_records;
    std::optional<ops::GdnReplayFoldPlan> replay_fold;
    std::optional<DFlashPersistentState> dflash;
    qwen3_6::RoundState io;
    Tensor prefill_hidden;
    std::optional<Tensor> score_hidden;
    Tensor sampling_config;
    Tensor token_counts;

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

    PinnedHostBuffer round_host;
    std::optional<PinnedHostBuffer> score_logprobs_host;
    TokenId* host_tokens = nullptr;
    std::optional<PinnedHostBuffer> ordinary_host;
    qwen3_6::OrdinaryDecodeIngress* ordinary_host_ingress = nullptr;
    qwen3_6::OrdinaryDecodeEgress* ordinary_host_egress   = nullptr;
    std::optional<PinnedHostBuffer> mtp_host;
    qwen3_6::MtpDecodeIngress* mtp_host_ingress = nullptr;
    qwen3_6::MtpDecodeEgress* mtp_host_egress   = nullptr;
    std::optional<PinnedHostBuffer> dflash_host;
    qwen3_6::DFlashDecodeIngress* dflash_host_ingress = nullptr;
    qwen3_6::DFlashDecodeEgress* dflash_host_egress   = nullptr;

    std::size_t workspace_logical_peak_bytes = 0;
    std::size_t vision_handoff_peak_bytes    = 0;

private:
    void advance_resource_revision() noexcept {
        if (++resource_revision_ == 0) { ++resource_revision_; }
    }

    std::uint64_t resource_revision_            = 1;
    std::uint32_t pressure_planning_generation_ = 0;
    bool pressure_planning_active_              = false;

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

            qwen3_6::detail::PressureDecision option;
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
        bool has_source                              = false;
        bool has_shared_source                       = false;
        runtime::ClaimDisposition source_disposition = runtime::ClaimDisposition::ConsumedToActive;
        std::uint32_t source_index                   = 0;
        std::uint64_t source_generation              = 0;
        std::uint32_t shared_source_index            = 0;
        std::uint64_t shared_source_generation       = 0;
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
        std::size_t shared_pressure_cursor    = 0;
        std::size_t shared_victim_count       = 0;
        bool pressure_host_releases_published = false;
        bool pressure_copies_prepared         = false;
        bool pressure_copies_submitted        = false;
        bool pressure_copies_published        = false;
        std::array<TransferWork, 3> pressure_transfer_work{};
        std::array<std::uint32_t, 3> pressure_transfer_pages{};
        std::uint64_t pressure_state_images = 0;
        std::uint8_t pressure_timer_mask    = 0;
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
    std::vector<TokenId> materialization_ledger_;
    qwen3_6::detail::ResidentPrefixIdentity materialization_identity_;
    qwen3_6::detail::PrefixShortlistDigests materialization_prefix_digests_;

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
        qwen3_6::CaptureStatePlacement state_placement = qwen3_6::CaptureStatePlacement::DeviceFork;
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
        bool pressure_host_releases_published = false;
        bool pressure_copies_prepared         = false;
        bool pressure_copies_submitted        = false;
        bool pressure_copies_published        = false;
        std::array<TransferWork, 3> pressure_transfer_work{};
        std::array<std::uint32_t, 3> pressure_transfer_pages{};
        std::uint64_t pressure_state_images = 0;
        std::uint8_t pressure_timer_mask    = 0;
        bool pressure_committed             = false;
        bool recycles_private_state         = false;
        bool replacement_removed            = false;
        bool prepared                       = false;
        std::uint64_t recycled_state_epoch  = 0;
        bool transfer_enqueue_pending       = false;
        bool transfer_submitted             = false;
        std::uint8_t transfer_timer_mask    = 0;
        bool published                      = false;
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
                                std::optional<AdmissionCandidate> pressure,
                                const runtime::ContextMachineCostModel& machine_cost,
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
    release_materialization_victim(MaterializationTransaction& transaction,
                                   std::size_t position) noexcept;
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
        runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] bool valid_sequence(SequenceHandle handle) const noexcept;
    [[nodiscard]] bool valid_continuation(const ContinuationHandle& handle) const noexcept;
    [[nodiscard]] bool valid_shared_prefix(const SharedPrefixHandle& handle) const noexcept;
    [[nodiscard]] bool valid_capture_offer(const CaptureOffer& offer) const noexcept;
    [[nodiscard]] bool materialization_pins(std::uint32_t index,
                                            std::uint64_t generation) const noexcept;
    [[nodiscard]] bool has_unsettled_state_fork() const noexcept;
    [[nodiscard]] bool valid_pending(const PendingBatch& pending) const noexcept;
    [[nodiscard]] detail::PhysicalResources
    resident_resources(const SequenceState& sequence) const noexcept;
    [[nodiscard]] detail::PhysicalResources
    resident_resources(const SharedPrefixState& shared) const noexcept;
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
    [[nodiscard]] qwen3_6::CheckpointSummary
    checkpoint_summary(const SequenceState& sequence, runtime::CheckpointRef checkpoint,
                       StateImageHandle state, runtime::PrefillWork rebuild_work) const;
    [[nodiscard]] qwen3_6::ContinuationSummary
    continuation_summary(const SequenceState& sequence) const;
    void populate_continuation_summary(const SequenceState& sequence,
                                       qwen3_6::ContinuationSummary& summary) const;
    [[nodiscard]] qwen3_6::SharedPrefixSummary
    shared_prefix_summary(const SharedPrefixState& shared) const;
    [[nodiscard]] std::optional<MaterializationSourceProtection>
    materialization_source_protection(const AdmissionCandidateImpl& admission) const;
    [[nodiscard]] detail::PhysicalResources
    materialization_deficit(const AdmissionCandidateImpl& admission) const;
    [[nodiscard]] bool
    protected_materialization_page(const MaterializationSourceProtection* protection,
                                   const KVAddressSpaceStore& addresses, std::uint32_t page_offset,
                                   LogicalKVPageHandle page, bool backend) const;
    [[nodiscard]] std::optional<qwen3_6::detail::PressureDecision>
    inspect_pressure_option(const SequenceState& sequence, detail::PhysicalResources deficit,
                            const MaterializationSourceProtection* protection           = nullptr,
                            const qwen3_6::TargetKVRequirement* retained_requirement    = nullptr,
                            std::span<const runtime::CheckpointRef> dropped_checkpoints = {},
                            std::span<const StateImageHandle> released_states           = {},
                            const qwen3_6::detail::PressureDecision* current = nullptr) const;
    [[nodiscard]] std::vector<qwen3_6::detail::PressureDecision>
    inspect_pressure_successors(const SequenceState& sequence, detail::PhysicalResources residual,
                                const MaterializationSourceProtection* protection,
                                const qwen3_6::detail::PressureDecision* current = nullptr) const;
    [[nodiscard]] std::vector<qwen3_6::detail::PressureDecision> inspect_shared_pressure_successors(
        const SharedPrefixState& shared, detail::PhysicalResources residual,
        const MaterializationSourceProtection* protection,
        const qwen3_6::detail::PressureDecision* current = nullptr) const;
    [[nodiscard]] std::optional<qwen3_6::detail::PressureDecision> inspect_shared_pressure_option(
        const SharedPrefixState& shared, detail::PhysicalResources deficit,
        const MaterializationSourceProtection* protection = nullptr,
        const qwen3_6::detail::PressureDecision* current  = nullptr) const;
    [[nodiscard]] std::vector<qwen3_6::detail::PressureDecision> inspect_shared_pressure_options(
        const SharedPrefixState& shared, detail::PhysicalResources deficit,
        const MaterializationSourceProtection* protection = nullptr,
        const qwen3_6::detail::PressureDecision* current  = nullptr) const;
    [[nodiscard]] qwen3_6::detail::PressureDecision
    inspect_eviction_option(const SequenceState& sequence) const;
    [[nodiscard]] qwen3_6::detail::PressureDecision
    inspect_shared_eviction_option(const SharedPrefixState& shared) const;
    [[nodiscard]] std::optional<qwen3_6::detail::PressureDecision>
    inspect_checkpoint_drop_option(const SequenceState& sequence,
                                   std::span<const runtime::CheckpointRef> checkpoints) const;
    [[nodiscard]] bool
    pressure_decision_valid(const SequenceState& sequence,
                            const qwen3_6::detail::PressureDecision& decision,
                            const MaterializationSourceProtection* protection) const;
    [[nodiscard]] bool
    shared_pressure_decision_valid(const SharedPrefixState& shared,
                                   const qwen3_6::detail::PressureDecision& decision,
                                   const MaterializationSourceProtection* protection) const;
    [[nodiscard]] std::vector<runtime::ContextTransferRequirement>
    checkpoint_restore_requirements(const SequenceKVBundle& kv,
                                    const qwen3_6::TargetKVRequirement& requirement,
                                    StateImageHandle state) const;
    [[nodiscard]] bool pressure_checkpoint_recovery_impacts(
        const AdmissionCandidateImpl& candidate,
        std::span<const ContinuationHandle* const> private_owners,
        std::span<const qwen3_6::detail::PressureDecision* const> private_decisions,
        std::span<const std::uint32_t> private_ordinals,
        std::span<const SharedPrefixHandle* const> shared_owners,
        std::span<const qwen3_6::detail::PressureDecision* const> shared_decisions,
        std::span<const std::uint32_t> shared_ordinals,
        const runtime::ContextMachineCostModel& machine_cost,
        std::vector<runtime::PressureCheckpointRecoveryImpact>& output,
        std::uint64_t& projection_work) const;
    void publish_checkpoint_drop(SequenceState& sequence, runtime::CheckpointRef checkpoint);
    [[nodiscard]] PrefillProgress wrap_prefill(std::uint32_t lane, runtime::PrefillStepResult step);
    [[nodiscard]] PendingBatch wrap_pending(std::span<const std::uint32_t> lanes,
                                            const runtime::BatchedGeneratedRound& round);
    void invalidate_lane(std::uint32_t lane) noexcept;
    [[nodiscard]] SequenceState& active_sequence(std::uint32_t lane);
    [[nodiscard]] const SequenceState& active_sequence(std::uint32_t lane) const;
    [[nodiscard]] std::optional<std::uint32_t> allocate_continuation_slot() noexcept;
    void release_continuation_slot(std::uint32_t index) noexcept;
    void clear_execution_failure_lanes(std::span<const std::uint32_t> lanes) noexcept;
    void clear_lane(SequenceState& sequence, RequestControl& request) noexcept;
    void ordered_reset(SequenceState& sequence);
    [[nodiscard]] StateImageSelectors state_selectors(const SequenceState& sequence) const;
    [[nodiscard]] std::uint32_t state_footprint(const SequenceState& sequence) const noexcept;
    [[nodiscard]] std::uint32_t owned_checkpoint_references(const SequenceState& sequence,
                                                            StateImageHandle state) const noexcept;
    [[nodiscard]] bool state_exclusive_to_sequence(const SequenceState& sequence,
                                                   StateImageHandle state) const noexcept;
    [[nodiscard]] std::optional<AdmissionCandidate> compose_materialization(
        AdmissionCandidate&& admission, std::span<const ContinuationHandle* const> pressure_owners,
        std::span<const qwen3_6::detail::PressureDecision> pressure_options,
        std::span<const SharedPrefixHandle* const> shared_pressure_owners,
        std::span<const qwen3_6::detail::PressureDecision> shared_pressure_options);
    [[nodiscard]] std::optional<detail::PhysicalPressureEffect> combined_pressure_effect(
        const MaterializationSourceProtection* protection,
        std::span<const ContinuationHandle* const> pressure_owners,
        std::span<const qwen3_6::detail::PressureDecision> pressure_options,
        std::span<const SharedPrefixHandle* const> shared_pressure_owners,
        std::span<const qwen3_6::detail::PressureDecision> shared_pressure_options,
        std::vector<HostKVPageReplicaRelease>* released_host_pages) const;
    void refresh_state_views(SequenceState& sequence);
    void reserve_state_entitlement(SequenceState& sequence, std::uint32_t slots);
    void settle_state_fork(SequenceState& sequence);
    [[nodiscard]] detail::PhysicalResources
    release_checkpoint_reference(StateImageHandle checkpoint) noexcept;
    [[nodiscard]] detail::PhysicalResources
    release_shared_prefix_state(std::uint32_t index, SharedPrefixSlotRole expected_role);
    [[nodiscard]] detail::PhysicalResources
    install_private_capture(SequenceState& sequence, const CaptureGroup& group,
                            StateImageHandle checkpoint,
                            std::optional<runtime::CheckpointRef> replacement);
    void prepare_active_capture(ActiveCaptureTransaction& transaction);
    void enqueue_active_capture_transfers(ActiveCaptureTransaction& transaction);
    void abort_active_capture(ActiveCaptureTransaction& transaction) noexcept;
    [[nodiscard]] ActiveCaptureResult publish_active_capture(ActiveCaptureTransaction& transaction);
    void release_active_shared_references(SequenceState& sequence) noexcept;
    void release_sequence_state(SequenceState& sequence) noexcept;
    void prepare_graphs();
    void install_sampling(SequenceState& sequence, RequestControl& request,
                          const ops::SamplingConfig& config);
    void set_device_i32(Tensor& tensor, std::int32_t value);
    void copy_tail(SequenceState& sequence, const Tensor& source);
    void copy_round_token();
    [[nodiscard]] runtime::ExecutionTiming
    resolve_non_speculative_pending(SequenceState& sequence, RequestControl& request,
                                    std::uint32_t accepted_tokens, bool terminal,
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
    void materialize_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                                 std::uint32_t backend_tokens = 0);
    void trim_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                          std::uint32_t backend_tokens = 0);
    void release_sequence_growth_entitlement(SequenceState& sequence) noexcept;
    void release_sequence_kv(SequenceState& sequence) noexcept;
    void commit_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                            std::uint32_t backend_tokens = 0);
    [[nodiscard]] qwen3_6::PagedKVCache* backend_kv_cache() noexcept;
    [[nodiscard]] const qwen3_6::PagedKVCache* backend_kv_cache() const noexcept;
    [[nodiscard]] std::uint32_t backend_kv_valid(const SequenceState& sequence) const noexcept;
    [[nodiscard]] qwen3_6::PagedKVCacheView text_kv_view(const SequenceState& sequence) const;
    [[nodiscard]] qwen3_6::PagedKVCacheView mtp_kv_view(const SequenceState& sequence) const;
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

namespace ninfer::targets::qwen3_6::detail {

template <>
struct PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT> {
    using Core               = NINFER_QWEN36_RUNTIME_NS::ProgramImplCore;
    using AdmissionCandidate = qwen3_6::AdmissionCandidate<NINFER_QWEN36_VARIANT>;
    using ContinuationHandle = qwen3_6::ContinuationHandle<NINFER_QWEN36_VARIANT>;
    using SharedPrefixHandle = qwen3_6::SharedPrefixHandle<NINFER_QWEN36_VARIANT>;

    struct Owner {
        const ContinuationHandle* private_handle = nullptr;
        const SharedPrefixHandle* shared_handle  = nullptr;
        std::uint32_t ordinal                    = 0;
        bool shared                              = false;
    };

    struct TargetNode {
        std::uint32_t candidate_index = 0;
        std::vector<std::uint16_t> owner_choices;
        std::uint32_t stable_ordinal = 0;
        bool root_maximal            = false;
    };

    struct CandidateOptions {
        std::vector<std::vector<PressureDecision>> owners;
        std::vector<std::uint16_t> eviction_choices;
        bool populated = false;
    };

    struct PreparedOwnerDecision {
        std::uint32_t candidate_index = 0;
        std::uint32_t owner_index     = 0;
        std::uint16_t choice          = 0;
        PressureDecision decision;
    };

    PressurePlanningSessionImpl(Core& owner, const runtime::ContextMachineCostModel& cost,
                                std::span<const AdmissionCandidate* const> admission_candidates,
                                std::span<const ContinuationHandle* const> private_owners,
                                std::span<const std::uint32_t> private_owner_ordinals,
                                std::span<const SharedPrefixHandle* const> shared_owners,
                                std::span<const std::uint32_t> shared_owner_ordinals);
    ~PressurePlanningSessionImpl() noexcept;

    [[nodiscard]] qwen3_6::PressureTargetHandle
    identity_target(const AdmissionCandidate& candidate) const;
    [[nodiscard]] qwen3_6::PressureTargetHandle
    root_maximal_target(const AdmissionCandidate& root_candidate);
    [[nodiscard]] runtime::PressureTargetAssessment assess(qwen3_6::PressureTargetHandle target);
    [[nodiscard]] qwen3_6::PreparedPressureExpansion<NINFER_QWEN36_VARIANT>
    prepare_expansion(qwen3_6::PressureTargetHandle parent);
    [[nodiscard]] qwen3_6::PressureExpansionView
    commit_expansion(qwen3_6::PreparedPressureExpansion<NINFER_QWEN36_VARIANT>&& prepared);
    void discard_expansion(
        qwen3_6::PreparedPressureExpansion<NINFER_QWEN36_VARIANT>&& prepared) noexcept;
    [[nodiscard]] std::optional<AdmissionCandidate>
    seal(qwen3_6::PressureTargetHandle target,
         const NINFER_QWEN36_RUNTIME_NS::PreparedPromptData& prompt);
    [[nodiscard]] std::optional<AdmissionCandidate>
    seal_capture(qwen3_6::PressureTargetHandle target);

    [[nodiscard]] bool valid(qwen3_6::PressureTargetHandle target) const noexcept;
    [[nodiscard]] std::uint32_t candidate_index(const AdmissionCandidate& candidate) const;
    void populate_options(std::uint32_t candidate_index);

    Core* program                                        = nullptr;
    const runtime::ContextMachineCostModel* machine_cost = nullptr;
    std::uint64_t resource_revision                      = 0;
    std::uint32_t generation                             = 1;
    std::uint32_t scratch_generation                     = 1;
    std::vector<const AdmissionCandidate*> candidates;
    std::vector<Owner> owners;
    std::vector<CandidateOptions> candidate_options;
    std::vector<TargetNode> targets;
    std::vector<TargetNode> expansion_scratch;
    std::vector<PreparedOwnerDecision> prepared_owner_decisions;
    std::vector<qwen3_6::PressureTargetHandle> committed_children;
    std::vector<const ContinuationHandle*> selected_private_owners;
    std::vector<PressureDecision> selected_private_decisions;
    std::vector<const SharedPrefixHandle*> selected_shared_owners;
    std::vector<PressureDecision> selected_shared_decisions;
    std::vector<const ContinuationHandle*> recovery_private_owners;
    std::vector<const PressureDecision*> recovery_private_decisions;
    std::vector<std::uint32_t> recovery_private_ordinals;
    std::vector<const SharedPrefixHandle*> recovery_shared_owners;
    std::vector<const PressureDecision*> recovery_shared_decisions;
    std::vector<std::uint32_t> recovery_shared_ordinals;
    std::vector<runtime::PressureOwnerOutcome> assessment_outcomes;
    std::vector<runtime::PressureCheckpointRecoveryImpact> assessment_impacts;
    std::uint32_t prepared_new_count = 0;
    bool scratch_live                = false;
};

template <>
class ProgramImpl<NINFER_QWEN36_VARIANT> final : public NINFER_QWEN36_RUNTIME_NS::ProgramImplCore {
public:
    using NINFER_QWEN36_RUNTIME_NS::ProgramImplCore::ProgramImplCore;
};

} // namespace ninfer::targets::qwen3_6::detail
