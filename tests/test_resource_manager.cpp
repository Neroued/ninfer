#include "runtime/engine/resource_manager.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace {

using ninfer::PrefixReusePath;
using ninfer::RuntimeStats;
using ninfer::runtime::CancellationFlagView;
using ninfer::runtime::CheckpointKind;
using ninfer::runtime::CheckpointRef;
using ninfer::runtime::CheckpointScope;
using ninfer::runtime::ClaimDisposition;
using ninfer::runtime::CommitDisposition;
using ninfer::runtime::ConsumeStatus;
using ninfer::runtime::ContextOperationCounts;
using ninfer::runtime::ContextTransactionInProgress;
using ninfer::runtime::ContextTransactionReserveStatus;
using ninfer::runtime::ContextTransactionStatus;
using ninfer::runtime::ContextTransferObservation;
using ninfer::runtime::ContextTransferRequirement;
using ninfer::runtime::FinishDisposition;
using ninfer::runtime::LaneId;
using ninfer::runtime::PrefillWork;
using ninfer::runtime::Readiness;
using ninfer::runtime::RequestPlanSummary;
using ninfer::runtime::RetentionClass;

int failures = 0;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

template <class Test>
void run_test(const char* name, Test&& test) {
    try {
        test();
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "FAIL " << name << ": " << error.what() << '\n';
    }
}

ninfer::runtime::ContextMachineCostModel test_cost_model() {
    ninfer::runtime::ContextMachineCostModel model;
    for (auto& transfer : model.transfer) {
        transfer.batch_ns        = 1;
        transfer.operation_ns    = 1;
        transfer.ns_per_byte_q32 = ninfer::runtime::kContextCostQ32One;
    }
    model.prefill.token_ns_q32        = 100ULL * ninfer::runtime::kContextCostQ32One;
    model.prefill.vision_item_ns      = 1;
    model.prefill.vision_patch_ns_q32 = ninfer::runtime::kContextCostQ32One;
    return model;
}

struct FakePreparedPrompt {
    std::uint32_t content_key = 0;
};

struct FakeCacheSessionKey {
    std::uint32_t value = 0;

    [[nodiscard]] std::string_view view() const noexcept {
        return {reinterpret_cast<const char*>(&value), sizeof(value)};
    }

    friend bool operator==(FakeCacheSessionKey, FakeCacheSessionKey) = default;
};

struct FakeShortlistKey {
    std::uint32_t digest   = 0;
    std::uint32_t frontier = 0;

    friend bool operator==(FakeShortlistKey, FakeShortlistKey) = default;
};

struct FakeRequiredKV {
    std::uint32_t main_pages    = 1;
    std::uint32_t backend_pages = 0;
};

struct FakeCheckpointSummary {
    CheckpointRef ref;
    CheckpointScope scope = CheckpointScope::Private;
    FakeShortlistKey shortlist_key;
    FakeRequiredKV required_kv;
    PrefillWork rebuild_work;
};

struct FakeContinuationSummary {
    std::optional<FakeCheckpointSummary> endpoint;
    std::optional<FakeCheckpointSummary> rewrite;
    std::vector<FakeCheckpointSummary> long_anchors;
    std::uint32_t active_references = 0;
};

struct FakeSharedPrefixSummary {
    FakeCheckpointSummary checkpoint;
    std::uint32_t active_references = 0;
};

FakeCheckpointSummary endpoint(std::uint32_t digest, std::uint32_t frontier) {
    return FakeCheckpointSummary{
        .ref           = CheckpointRef{.kind     = CheckpointKind::SessionEndpoint,
                                       .frontier = frontier,
                                       .ordinal  = 0},
        .scope         = CheckpointScope::Private,
        .shortlist_key = FakeShortlistKey{.digest = digest, .frontier = frontier},
        .required_kv   = FakeRequiredKV{.main_pages = 1, .backend_pages = 0},
        .rebuild_work  = PrefillWork{.tokens = frontier},
    };
}

FakeCheckpointSummary long_anchor(std::uint32_t digest, std::uint32_t frontier,
                                  std::uint32_t ordinal) {
    return FakeCheckpointSummary{
        .ref           = CheckpointRef{.kind     = CheckpointKind::LongAnchor,
                                       .frontier = frontier,
                                       .ordinal  = ordinal},
        .scope         = CheckpointScope::Private,
        .shortlist_key = FakeShortlistKey{.digest = digest, .frontier = frontier},
        .required_kv   = FakeRequiredKV{.main_pages = 1, .backend_pages = 0},
        .rebuild_work  = PrefillWork{.tokens = frontier},
    };
}

FakeCheckpointSummary shared_checkpoint(std::uint32_t digest, std::uint32_t frontier) {
    return FakeCheckpointSummary{
        .ref           = CheckpointRef{.kind     = CheckpointKind::SharedStablePrefix,
                                       .frontier = frontier,
                                       .ordinal  = 0},
        .scope         = CheckpointScope::Shared,
        .shortlist_key = FakeShortlistKey{.digest = digest, .frontier = frontier},
        .required_kv   = FakeRequiredKV{.main_pages = 1, .backend_pages = 0},
        .rebuild_work  = PrefillWork{.tokens = frontier},
    };
}

struct FakeContextCache {
    struct Opportunity {
        ninfer::PromptCacheMarkerKind kind = ninfer::PromptCacheMarkerKind::SharedStablePrefix;
        ninfer::SharedCandidateEvidence evidence =
            ninfer::SharedCandidateEvidence::ExplicitBoundary;
        std::uint32_t frontier = 0;
    };

    std::optional<FakeCacheSessionKey> session_key;
    RetentionClass retention  = RetentionClass::RecentPrivate;
    bool update_session_index = true;
    std::vector<Opportunity> opportunities;
};

struct FakeRequestBasePlan {
    RequestPlanSummary value;
    FakeContextCache cache;
    std::uint32_t shortlist_digest = 0;
    bool allow_shortlist           = true;
    bool isolated_feasible         = true;

    [[nodiscard]] const RequestPlanSummary& summary() const noexcept { return value; }

    [[nodiscard]] const FakeContextCache& context_cache() const noexcept { return cache; }

    [[nodiscard]] std::optional<FakeShortlistKey>
    prefix_shortlist_key(std::uint32_t frontier) const noexcept {
        if (!allow_shortlist || frontier == 0) { return std::nullopt; }
        return FakeShortlistKey{.digest = shortlist_digest, .frontier = frontier};
    }

    [[nodiscard]] std::optional<PrefillWork>
    shared_candidate_rebuild_work(std::uint32_t frontier) const noexcept {
        const auto found =
            std::find_if(cache.opportunities.begin(), cache.opportunities.end(),
                         [&](const auto& opportunity) { return opportunity.frontier == frontier; });
        return found == cache.opportunities.end()
                   ? std::nullopt
                   : std::optional<PrefillWork>(PrefillWork{.tokens = frontier});
    }
};

FakeRequestBasePlan make_base(std::uint32_t digest,
                              std::optional<FakeCacheSessionKey> session = std::nullopt,
                              RetentionClass retention  = RetentionClass::RecentPrivate,
                              bool update_session_index = true) {
    FakeRequestBasePlan out;
    out.value.prompt_tokens           = 64;
    out.value.requested_output_tokens = 8;
    out.value.effective_output_tokens = 8;
    out.value.service_work_quanta     = 64;
    out.value.publish_continuation    = true;
    out.cache.session_key             = session;
    out.cache.retention               = retention;
    out.cache.update_session_index    = update_session_index;
    out.shortlist_digest              = digest;
    return out;
}

struct FakeContinuationHandle {
    std::uint32_t id          = 0;
    std::uint32_t content_key = 0;

    FakeContinuationHandle() = default;

    FakeContinuationHandle(std::uint32_t id_value, std::uint32_t key_value)
        : id(id_value), content_key(key_value) {}

    FakeContinuationHandle(FakeContinuationHandle&& other) noexcept
        : id(std::exchange(other.id, 0)), content_key(other.content_key) {}

    FakeContinuationHandle& operator=(FakeContinuationHandle&& other) noexcept {
        id          = std::exchange(other.id, 0);
        content_key = other.content_key;
        return *this;
    }

    FakeContinuationHandle(const FakeContinuationHandle&)            = delete;
    FakeContinuationHandle& operator=(const FakeContinuationHandle&) = delete;
};

struct FakeSharedPrefixHandle {
    std::uint32_t id          = 0;
    std::uint32_t content_key = 0;

    FakeSharedPrefixHandle()                                             = default;
    FakeSharedPrefixHandle(FakeSharedPrefixHandle&&) noexcept            = default;
    FakeSharedPrefixHandle& operator=(FakeSharedPrefixHandle&&) noexcept = default;
    FakeSharedPrefixHandle(const FakeSharedPrefixHandle&)                = delete;
    FakeSharedPrefixHandle& operator=(const FakeSharedPrefixHandle&)     = delete;
};

struct FakeSequenceHandle {
    std::uint32_t id = 0;

    friend bool operator==(FakeSequenceHandle, FakeSequenceHandle) = default;
};

struct FakeCaptureOffer {
    std::uint32_t id = 0;
};

struct FakeAdmissionCandidate {
    RequestPlanSummary value;
    PrefillWork remaining;
    std::vector<ContextTransferRequirement> transfers;
    ninfer::runtime::IdentityMaterializationAssessment identity;
    ClaimDisposition disposition    = ClaimDisposition::ConsumedToActive;
    std::uint32_t private_source_id = 0;
    std::uint32_t shared_source_id  = 0;

    [[nodiscard]] const RequestPlanSummary& summary() const noexcept { return value; }

    [[nodiscard]] const ninfer::runtime::IdentityMaterializationAssessment&
    identity_assessment() const noexcept {
        return identity;
    }
};

struct FakeTargetDecision {
    std::uint64_t id                  = 0;
    std::uint64_t immediate_ns        = 100'000'000;
    std::uint32_t degradation_units   = 1;
    std::uint32_t dropped_checkpoints = 0;
    bool evicts_continuation          = false;
    bool shared_owner                 = false;

    friend bool operator==(const FakeTargetDecision&, const FakeTargetDecision&) = default;
};

struct FakePressureTargetHandle {
    std::uint32_t generation = 0;
    std::uint32_t index      = 0;

    friend bool operator==(FakePressureTargetHandle, FakePressureTargetHandle) = default;
};

struct FakePreparedPressureExpansion {
    std::uint32_t generation         = 0;
    std::uint32_t scratch_generation = 0;
    std::uint32_t new_count          = 0;

    FakePreparedPressureExpansion(std::uint32_t generation_value,
                                  std::uint32_t scratch_generation_value,
                                  std::uint32_t new_count_value) noexcept
        : generation(generation_value), scratch_generation(scratch_generation_value),
          new_count(new_count_value) {}

    FakePreparedPressureExpansion(FakePreparedPressureExpansion&&) noexcept            = default;
    FakePreparedPressureExpansion& operator=(FakePreparedPressureExpansion&&) noexcept = default;
    FakePreparedPressureExpansion(const FakePreparedPressureExpansion&)                = delete;
    FakePreparedPressureExpansion& operator=(const FakePreparedPressureExpansion&)     = delete;

    [[nodiscard]] std::uint32_t new_canonical_count() const noexcept { return new_count; }
};

struct FakePressureExpansionView {
    std::span<const FakePressureTargetHandle> children;
    std::uint32_t new_canonical_count = 0;
};

struct FakeResourcePlan {
    FakeAdmissionCandidate admission;
    std::uint64_t revision = 0;
    std::vector<FakeTargetDecision> private_actions;
    std::vector<FakeTargetDecision> shared_actions;
    std::vector<std::uint32_t> private_owner_ids;
    std::vector<std::uint32_t> shared_owner_ids;

    FakeResourcePlan() = default;

    FakeResourcePlan(FakeAdmissionCandidate admission_value, std::uint64_t revision_value)
        : admission(std::move(admission_value)), revision(revision_value) {}

    FakeResourcePlan(FakeResourcePlan&&) noexcept            = default;
    FakeResourcePlan& operator=(FakeResourcePlan&&) noexcept = default;
    FakeResourcePlan(const FakeResourcePlan&)                = delete;
    FakeResourcePlan& operator=(const FakeResourcePlan&)     = delete;

    [[nodiscard]] const RequestPlanSummary& summary() const noexcept { return admission.summary(); }

    [[nodiscard]] bool needs_transfer() const noexcept { return !admission.transfers.empty(); }

    [[nodiscard]] std::uint64_t resource_revision() const noexcept { return revision; }
};

struct FakePersistentBackfillProof {
    std::uint64_t revision = 0;

    [[nodiscard]] std::uint64_t resource_revision() const noexcept { return revision; }
};

struct FakeStartResult {
    FakeSequenceHandle sequence;
};

struct FakeMaterializationVictimResult {
    ClaimDisposition disposition = ClaimDisposition::Retained;
    bool pressure_committed      = false;
    std::optional<FakeContinuationSummary> final_summary;
};

struct FakeMaterializationSharedVictimResult {
    ClaimDisposition disposition = ClaimDisposition::Retained;
    bool pressure_committed      = false;
    std::optional<FakeSharedPrefixSummary> final_summary;
};

struct FakeMaterializationSourceResult {
    ClaimDisposition disposition = ClaimDisposition::Retained;
    std::optional<FakeContinuationSummary> final_summary;
};

struct FakeMaterializationSharedSourceResult {
    ClaimDisposition disposition = ClaimDisposition::Retained;
    std::optional<FakeSharedPrefixSummary> final_summary;
};

struct FakeMaterializationResult {
    ContextTransactionStatus status = ContextTransactionStatus::Aborted;
    std::optional<FakeStartResult> published;
    std::optional<FakeMaterializationSourceResult> source;
    std::optional<FakeMaterializationSharedSourceResult> shared_source;
    std::vector<FakeMaterializationVictimResult> victims;
    std::vector<FakeMaterializationSharedVictimResult> shared_victims;
    std::vector<ContextTransferObservation> transfer_observations;
    ContextOperationCounts operations;
};

struct FakeSharedPrefixPublication {
    FakeSharedPrefixHandle handle;
    FakeSharedPrefixSummary summary;
};

struct FakeActiveCaptureResult {
    ContextTransactionStatus status     = ContextTransactionStatus::Aborted;
    bool capacity_preparation_committed = false;
    FakeContinuationSummary active_summary;
    std::optional<FakeSharedPrefixPublication> shared;
    std::vector<FakeMaterializationVictimResult> victims;
    std::vector<FakeMaterializationSharedVictimResult> shared_victims;
    std::vector<ContextTransferObservation> transfer_observations;
    ContextOperationCounts operations;
};

using FakeContextTransactionProgress =
    std::variant<ContextTransactionInProgress, FakeMaterializationResult, FakeActiveCaptureResult>;

struct FakeCaptureAssessment {
    FakeShortlistKey shortlist_key;
    ninfer::SharedCandidateEvidence shared_evidence = ninfer::SharedCandidateEvidence::None;
    PrefillWork protected_rebuild_work;
    std::vector<ContextTransferRequirement> transfer_requirements;
    std::vector<CheckpointRef> private_replacement_candidates;
    std::uint64_t immediate_ns          = 0;
    std::uint64_t projected_recovery_ns = 0;
    bool publishes_private              = false;
    bool publishes_shared               = false;
    bool needs_transfer                 = false;
    bool physically_feasible            = true;
};

struct FakeTimings {
    std::uint64_t value = 0;
};

struct FakeSpeculativeStats {
    std::uint64_t value = 0;
};

struct FakeFinishResult {
    ConsumeStatus status          = ConsumeStatus::InvariantMismatch;
    FinishDisposition disposition = FinishDisposition::Released;
    FakeTimings timings;
    FakeSpeculativeStats speculative;
    FakeContinuationSummary summary;
    std::optional<FakeContinuationHandle> continuation;
};

struct FakeAbortResult {
    ConsumeStatus status = ConsumeStatus::InvariantMismatch;
    FakeTimings timings;
    FakeSpeculativeStats speculative;
};

struct FakeReleaseResult {
    ConsumeStatus status = ConsumeStatus::InvariantMismatch;
};

struct FakeCommitRowResult {
    CommitDisposition disposition = CommitDisposition::Active;
};

struct FakeCommitResult {
    std::array<FakeCommitRowResult, ninfer::kMaximumConcurrency> rows{};
    std::size_t row_count = 0;
};

struct FakeDiscardResult {
    ConsumeStatus status  = ConsumeStatus::InvariantMismatch;
    std::size_t row_count = 0;
};

struct FakePhysicalUsage {
    std::uint32_t device_state_slots      = 0;
    std::uint32_t host_state_slots        = 0;
    std::uint32_t device_main_kv_pages    = 0;
    std::uint32_t device_backend_kv_pages = 0;
    std::size_t host_kv_bytes             = 0;
};

class FakeProgram;

class FakePressurePlanningSession {
public:
    FakePressurePlanningSession(FakeProgram& program,
                                const ninfer::runtime::ContextMachineCostModel& machine_cost,
                                std::span<const FakeAdmissionCandidate* const> candidates,
                                std::span<const FakeContinuationHandle* const> private_owners,
                                std::span<const std::uint32_t> private_owner_ordinals,
                                std::span<const FakeSharedPrefixHandle* const> shared_owners,
                                std::span<const std::uint32_t> shared_owner_ordinals);

    FakePressurePlanningSession(FakePressurePlanningSession&&) noexcept            = default;
    FakePressurePlanningSession& operator=(FakePressurePlanningSession&&) noexcept = default;
    FakePressurePlanningSession(const FakePressurePlanningSession&)                = delete;
    FakePressurePlanningSession& operator=(const FakePressurePlanningSession&)     = delete;

    [[nodiscard]] FakePressureTargetHandle
    identity_target(const FakeAdmissionCandidate& candidate) const;
    [[nodiscard]] FakePressureTargetHandle
    root_maximal_target(const FakeAdmissionCandidate& candidate);
    [[nodiscard]] ninfer::runtime::PressureTargetAssessment assess(FakePressureTargetHandle target);
    [[nodiscard]] FakePreparedPressureExpansion prepare_expansion(FakePressureTargetHandle parent);
    [[nodiscard]] FakePressureExpansionView
    commit_expansion(FakePreparedPressureExpansion&& prepared);
    void discard_expansion(FakePreparedPressureExpansion&& prepared) noexcept;
    [[nodiscard]] std::optional<FakeResourcePlan> seal(FakePressureTargetHandle target,
                                                       const FakePreparedPrompt& prompt);
    [[nodiscard]] std::optional<FakeResourcePlan> seal_capture(FakePressureTargetHandle target);

private:
    struct Owner {
        const FakeContinuationHandle* private_handle = nullptr;
        const FakeSharedPrefixHandle* shared_handle  = nullptr;
        std::uint32_t ordinal                        = 0;
        bool shared                                  = false;
    };

    struct Target {
        std::uint32_t candidate_index = 0;
        std::vector<std::uint16_t> choices;
        std::uint32_t stable_ordinal = 0;
        bool root_maximal            = false;
    };

    [[nodiscard]] bool valid(FakePressureTargetHandle target) const noexcept;
    [[nodiscard]] std::uint32_t candidate_index(const FakeAdmissionCandidate& candidate) const;
    void populate_options(std::uint32_t candidate_index);
    [[nodiscard]] bool same_target(const Target& left, const Target& right) const noexcept;
    [[nodiscard]] std::vector<FakeTargetDecision> decisions_for(std::uint32_t candidate_index,
                                                                std::size_t owner_index) const;

    FakeProgram* program_                                         = nullptr;
    const ninfer::runtime::ContextMachineCostModel* machine_cost_ = nullptr;
    std::uint64_t revision_                                       = 0;
    std::uint32_t generation_                                     = 0;
    std::uint32_t scratch_generation_                             = 0;
    bool scratch_live_                                            = false;
    std::vector<const FakeAdmissionCandidate*> candidates_;
    std::vector<Owner> owners_;
    std::vector<std::vector<std::vector<FakeTargetDecision>>> options_;
    std::vector<std::uint8_t> options_populated_;
    std::vector<Target> targets_;
    std::vector<Target> expansion_scratch_;
    std::vector<FakePressureTargetHandle> committed_children_;
    std::vector<ninfer::runtime::PressureOwnerOutcome> assessment_outcomes_;
    std::vector<ninfer::runtime::PressureCheckpointRecoveryImpact> assessment_impacts_;
};

class FakeProgram {
public:
    friend class FakePressurePlanningSession;

    enum class TransactionKind : std::uint8_t {
        None,
        Materialization,
        Capture,
    };

    [[nodiscard]] bool isolated_request_feasible(const FakeRequestBasePlan& base) const noexcept {
        return base.isolated_feasible;
    }

    [[nodiscard]] std::optional<FakeAdmissionCandidate>
    inspect_admission(const FakePreparedPrompt& prompt, const FakeRequestBasePlan& base, LaneId,
                      const FakeContinuationHandle* source,
                      const FakeSharedPrefixHandle* shared_source,
                      std::optional<CheckpointRef> checkpoint, bool must_retain_source,
                      const ninfer::runtime::ContextMachineCostModel& machine_cost) {
        ++admission_inspections;
        if (source != nullptr) {
            inspected_private_sources.push_back(source->id);
            if (source->content_key != prompt.content_key || !checkpoint) { return std::nullopt; }
        }
        if (shared_source != nullptr) {
            inspected_shared_sources.push_back(shared_source->id);
            if (shared_source->content_key != prompt.content_key || !checkpoint) {
                return std::nullopt;
            }
        }

        FakeAdmissionCandidate plan;
        plan.value = base.summary();
        if (checkpoint) {
            plan.value.reusable_prompt_tokens = checkpoint->frontier;
            switch (checkpoint->kind) {
            case CheckpointKind::SessionEndpoint:
                plan.value.prefix_reuse_path = PrefixReusePath::PrivateEndpoint;
                break;
            case CheckpointKind::TurnClosure:
                plan.value.prefix_reuse_path = PrefixReusePath::PrivateTurnClosure;
                break;
            case CheckpointKind::ResponseReplay:
                plan.value.prefix_reuse_path = PrefixReusePath::PrivateResponseReplay;
                break;
            case CheckpointKind::LongAnchor:
                plan.value.prefix_reuse_path = PrefixReusePath::PrivateLongAnchor;
                break;
            case CheckpointKind::SharedStablePrefix:
                plan.value.prefix_reuse_path = PrefixReusePath::SharedStablePrefix;
                break;
            }
        } else {
            plan.value.reusable_prompt_tokens = 0;
            plan.value.prefix_reuse_path      = PrefixReusePath::Root;
        }
        plan.remaining.tokens = plan.value.prompt_tokens > plan.value.reusable_prompt_tokens
                                    ? plan.value.prompt_tokens - plan.value.reusable_prompt_tokens
                                    : 0;
        if (source != nullptr) {
            plan.private_source_id = source->id;
            plan.disposition       = must_retain_source ? ClaimDisposition::Retained
                                                        : ClaimDisposition::ConsumedToActive;
        } else if (shared_source != nullptr) {
            plan.shared_source_id = shared_source->id;
            plan.disposition      = ClaimDisposition::Retained;
        }
        plan.identity.machine.minimum_request_ns     = machine_cost.prefill_ns(plan.remaining);
        plan.identity.machine.immediate_ns           = plan.identity.machine.minimum_request_ns;
        plan.identity.machine.remaining_prefill_work = plan.remaining;
        plan.identity.machine.reused_prompt_tokens   = plan.value.reusable_prompt_tokens;
        plan.identity.physical_status =
            target_feasible(std::span<const FakeTargetDecision>{})
                ? ninfer::runtime::MaterializationPhysicalStatus::Feasible
                : ninfer::runtime::MaterializationPhysicalStatus::Infeasible;
        plan.identity.source_disposition = plan.disposition;
        plan.identity.expandable         = plan.identity.physical_status !=
                                   ninfer::runtime::MaterializationPhysicalStatus::Feasible;
        plan.identity.projection_work = 1;
        plan.identity.assessment_digest =
            (static_cast<std::uint64_t>(plan.value.reusable_prompt_tokens) << 32U) ^ revision_;
        return plan;
    }

    [[nodiscard]] std::optional<FakeResourcePlan>
    seal_identity(const FakeAdmissionCandidate& admission, const FakePreparedPrompt&) {
        if (admission.identity.physical_status !=
            ninfer::runtime::MaterializationPhysicalStatus::Feasible) {
            return std::nullopt;
        }
        seal_attempts.emplace_back();
        return FakeResourcePlan(admission, revision_);
    }

    [[nodiscard]] FakePressurePlanningSession
    begin_pressure_planning(const ninfer::runtime::ContextMachineCostModel& machine_cost,
                            std::span<const FakeAdmissionCandidate* const> candidates,
                            std::span<const FakeContinuationHandle* const> private_owners,
                            std::span<const std::uint32_t> private_owner_ordinals,
                            std::span<const FakeSharedPrefixHandle* const> shared_owners,
                            std::span<const std::uint32_t> shared_owner_ordinals);

    void select_shared_captures(FakeResourcePlan&, const FakePreparedPrompt&,
                                std::span<const std::uint32_t> frontiers) {
        selected_shared_capture_frontiers.assign(frontiers.begin(), frontiers.end());
    }

    [[nodiscard]] std::uint64_t
    shared_capture_split_cost_ns(const FakeResourcePlan&, const FakePreparedPrompt&,
                                 std::span<const std::uint32_t> frontiers,
                                 const ninfer::runtime::ContextMachineCostModel&) const noexcept {
        return static_cast<std::uint64_t>(frontiers.size()) * 100U;
    }

    [[nodiscard]] bool
    target_feasible(std::span<const FakeTargetDecision> decisions) const noexcept {
        if (decisions.size() < required_pressure_actions) { return false; }
        const bool has_eviction =
            std::any_of(decisions.begin(), decisions.end(),
                        [](const auto& decision) { return decision.evicts_continuation; });
        if (required_action_id && !has_eviction &&
            std::none_of(decisions.begin(), decisions.end(), [&](const auto& decision) {
                return decision.id == *required_action_id;
            })) {
            return false;
        }
        if (require_evictions &&
            std::any_of(decisions.begin(), decisions.end(),
                        [](const auto& decision) { return !decision.evicts_continuation; })) {
            return false;
        }
        return true;
    }

    [[nodiscard]] ContextTransactionReserveStatus
    start_resource_transaction(FakeResourcePlan&& plan, FakePreparedPrompt&& prompt,
                               CancellationFlagView cancellation) {
        ++start_calls;
        started_source_id          = plan.admission.private_source_id;
        started_source_disposition = plan.admission.disposition;
        started_action_ids.clear();
        for (const auto& action : plan.private_actions) { started_action_ids.push_back(action.id); }
        for (const auto& action : plan.shared_actions) { started_action_ids.push_back(action.id); }
        if (cancellation.requested() || abort_start || plan.revision != revision_) {
            return ContextTransactionReserveStatus::Aborted;
        }
        pending_prompt_ = prompt;
        pending_plan_.emplace(std::move(plan));
        transaction_kind_ = TransactionKind::Materialization;
        advance_revision();
        return ContextTransactionReserveStatus::Reserved;
    }

    [[nodiscard]] std::optional<FakePersistentBackfillProof>
    prove_persistent_backfill(const FakeRequestBasePlan&, const FakeResourcePlan& candidate,
                              std::span<const FakeSequenceHandle>) const {
        if (candidate.resource_revision() != revision_) { return std::nullopt; }
        return FakePersistentBackfillProof{.revision = revision_};
    }

    [[nodiscard]] FakeContextTransactionProgress
    progress_context_transaction(CancellationFlagView cancellation) {
        require(transaction_kind_ != TransactionKind::None,
                "fake Program has no context transaction");
        if (progress_in_progress_once) {
            progress_in_progress_once = false;
            return ContextTransactionInProgress{};
        }
        if (transaction_kind_ == TransactionKind::Capture) {
            FakeActiveCaptureResult result;
            result.status =
                cancellation.requested() ? ContextTransactionStatus::Aborted : capture_status;
            if (pending_plan_) {
                for (std::size_t index = 0; index < pending_plan_->private_actions.size();
                     ++index) {
                    const FakeTargetDecision& action = pending_plan_->private_actions[index];
                    FakeMaterializationVictimResult victim{
                        .disposition        = action.evicts_continuation ? ClaimDisposition::Evicted
                                                                         : ClaimDisposition::Retained,
                        .pressure_committed = action.evicts_continuation ||
                                              result.status == ContextTransactionStatus::Published,
                    };
                    if (!action.evicts_continuation) {
                        const std::uint32_t owner = pending_plan_->private_owner_ids[index];
                        victim.final_summary.emplace();
                        victim.final_summary->endpoint =
                            endpoint(sequence_content_keys_.at(owner), finish_frontier);
                    }
                    result.victims.push_back(std::move(victim));
                }
                for (std::size_t index = 0; index < pending_plan_->shared_actions.size(); ++index) {
                    const FakeTargetDecision& action = pending_plan_->shared_actions[index];
                    FakeMaterializationSharedVictimResult victim{
                        .disposition        = action.evicts_continuation ? ClaimDisposition::Evicted
                                                                         : ClaimDisposition::Retained,
                        .pressure_committed = action.evicts_continuation ||
                                              result.status == ContextTransactionStatus::Published,
                    };
                    if (!action.evicts_continuation) {
                        const std::uint32_t owner = pending_plan_->shared_owner_ids[index];
                        victim.final_summary      = FakeSharedPrefixSummary{
                                 .checkpoint = shared_checkpoint(owner, finish_frontier),
                        };
                    }
                    result.shared_victims.push_back(std::move(victim));
                }
            }
            if (result.status == ContextTransactionStatus::Published) {
                result.active_summary = capture_summary;
                if (pending_capture_publish_shared_) {
                    FakeSharedPrefixHandle handle;
                    handle.id          = next_shared_id_++;
                    handle.content_key = capture_assessment.shortlist_key.digest;
                    result.shared      = FakeSharedPrefixPublication{
                             .handle = std::move(handle),
                             .summary =
                            FakeSharedPrefixSummary{
                                     .checkpoint =
                                    shared_checkpoint(capture_assessment.shortlist_key.digest,
                                                           capture_assessment.shortlist_key.frontier),
                                     .active_references = 1,
                            },
                    };
                }
            }
            return result;
        }

        require(pending_plan_.has_value(), "fake materialization plan disappeared");
        const FakeResourcePlan& plan = *pending_plan_;
        FakeMaterializationResult result;
        result.status = (abort_progress || cancellation.requested())
                            ? ContextTransactionStatus::Aborted
                            : ContextTransactionStatus::Published;
        for (std::size_t index = 0; index < plan.private_actions.size(); ++index) {
            const FakeTargetDecision& action = plan.private_actions[index];
            const bool evicted               = action.evicts_continuation;
            FakeMaterializationVictimResult victim{
                .disposition = evicted ? ClaimDisposition::Evicted : ClaimDisposition::Retained,
                .pressure_committed =
                    evicted || result.status == ContextTransactionStatus::Published,
            };
            if (!evicted) {
                const std::uint32_t owner_id = plan.private_owner_ids.at(index);
                victim.final_summary.emplace();
                victim.final_summary->endpoint =
                    endpoint(sequence_content_keys_.at(owner_id), finish_frontier);
            }
            result.victims.push_back(std::move(victim));
        }
        for (const FakeTargetDecision& action : plan.shared_actions) {
            result.shared_victims.push_back(FakeMaterializationSharedVictimResult{
                .disposition        = action.evicts_continuation ? ClaimDisposition::Evicted
                                                                 : ClaimDisposition::Retained,
                .pressure_committed = action.evicts_continuation ||
                                      result.status == ContextTransactionStatus::Published,
            });
        }
        if (plan.admission.private_source_id != 0) {
            result.source = FakeMaterializationSourceResult{
                .disposition = result.status == ContextTransactionStatus::Aborted
                                   ? ClaimDisposition::Retained
                                   : plan.admission.disposition};
        }
        if (plan.admission.shared_source_id != 0) {
            result.shared_source =
                FakeMaterializationSharedSourceResult{.disposition = ClaimDisposition::Retained};
        }
        if (result.status == ContextTransactionStatus::Published) {
            const std::uint32_t sequence_id        = next_sequence_id_++;
            sequence_content_keys_.at(sequence_id) = pending_prompt_.content_key;
            result.published = FakeStartResult{.sequence = FakeSequenceHandle{sequence_id}};
        }
        return result;
    }

    void finalize_context_transaction() noexcept {
        transaction_kind_ = TransactionKind::None;
        pending_plan_.reset();
        pending_capture_publish_shared_ = false;
    }

    [[nodiscard]] bool has_context_transaction() const noexcept {
        return transaction_kind_ != TransactionKind::None;
    }

    [[nodiscard]] FakeCaptureAssessment
    inspect_capture(const FakeCaptureOffer&, const FakeSharedPrefixHandle*,
                    const FakeSharedPrefixHandle*, std::optional<CheckpointRef>,
                    bool permit_shared_publication,
                    const ninfer::runtime::ContextMachineCostModel& machine_cost) const {
        FakeCaptureAssessment assessment = capture_assessment;
        assessment.immediate_ns          = 0;
        for (const ContextTransferRequirement& transfer : assessment.transfer_requirements) {
            assessment.immediate_ns += machine_cost.transfer_ns(transfer.direction, transfer.work);
        }
        if (!permit_shared_publication) { assessment.publishes_shared = false; }
        return assessment;
    }

    [[nodiscard]] std::uint64_t
    checkpoint_recovery_ns(const FakeContinuationHandle&, CheckpointRef,
                           const ninfer::runtime::ContextMachineCostModel&) const {
        return 0;
    }

    [[nodiscard]] std::uint64_t
    checkpoint_recovery_ns(const FakeSharedPrefixHandle&, CheckpointRef,
                           const ninfer::runtime::ContextMachineCostModel&) const {
        return 0;
    }

    [[nodiscard]] FakeAdmissionCandidate make_capture_pressure_candidate(
        const FakeCaptureAssessment& assessment,
        const ninfer::runtime::ContextMachineCostModel& machine_cost) const {
        FakeAdmissionCandidate candidate;
        candidate.value.prompt_tokens = assessment.shortlist_key.frontier;
        for (const ContextTransferRequirement& transfer : assessment.transfer_requirements) {
            candidate.identity.machine.immediate_ns +=
                machine_cost.transfer_ns(transfer.direction, transfer.work);
            candidate.identity.machine.transferred_bytes += transfer.work.payload_bytes;
            candidate.identity.machine.copy_operations += transfer.work.copy_operations;
        }
        candidate.identity.machine.minimum_request_ns = candidate.identity.machine.immediate_ns;
        candidate.identity.physical_status =
            target_feasible(std::span<const FakeTargetDecision>{})
                ? ninfer::runtime::MaterializationPhysicalStatus::Feasible
                : ninfer::runtime::MaterializationPhysicalStatus::Infeasible;
        candidate.identity.expandable = candidate.identity.physical_status !=
                                        ninfer::runtime::MaterializationPhysicalStatus::Feasible;
        candidate.identity.projection_work = 1;
        candidate.identity.assessment_digest =
            (static_cast<std::uint64_t>(assessment.shortlist_key.frontier) << 32U) ^ revision_;
        return candidate;
    }

    [[nodiscard]] bool shared_capture_matches(const FakeCaptureOffer&,
                                              const FakeSharedPrefixHandle&) const {
        return false;
    }

    void skip_capture(FakeCaptureOffer&&) { ++skipped_captures; }

    [[nodiscard]] ContextTransactionReserveStatus
    reserve_active_capture(FakeCaptureOffer&&, const FakeSharedPrefixHandle*,
                           const FakeSharedPrefixHandle*, std::optional<CheckpointRef>, bool,
                           const ninfer::runtime::ContextMachineCostModel&,
                           CancellationFlagView cancellation) {
        if (cancellation.requested() || abort_capture_start) {
            return ContextTransactionReserveStatus::Aborted;
        }
        pending_plan_.reset();
        pending_capture_publish_shared_ = false;
        transaction_kind_               = TransactionKind::Capture;
        advance_revision();
        return ContextTransactionReserveStatus::Reserved;
    }

    [[nodiscard]] ContextTransactionReserveStatus reserve_active_capture_with_pressure(
        FakeCaptureOffer&&, const FakeSharedPrefixHandle*, const FakeSharedPrefixHandle*,
        std::optional<CheckpointRef>, bool publish_shared, FakeResourcePlan&& pressure,
        const ninfer::runtime::ContextMachineCostModel&, CancellationFlagView cancellation) {
        if (cancellation.requested() || abort_capture_start || pressure.revision != revision_) {
            return ContextTransactionReserveStatus::Aborted;
        }
        started_action_ids.clear();
        for (const auto& action : pressure.private_actions) {
            started_action_ids.push_back(action.id);
        }
        for (const auto& action : pressure.shared_actions) {
            started_action_ids.push_back(action.id);
        }
        pending_plan_.emplace(std::move(pressure));
        pending_capture_publish_shared_ = publish_shared;
        transaction_kind_               = TransactionKind::Capture;
        advance_revision();
        return ContextTransactionReserveStatus::Reserved;
    }

    [[nodiscard]] FakeFinishResult finish(FakeSequenceHandle sequence) noexcept {
        ++finish_calls;
        if (finish_fail_next) {
            finish_fail_next = false;
            return {};
        }
        advance_revision();
        FakeFinishResult result;
        result.status = ConsumeStatus::Consumed;
        if (finish_release) {
            result.disposition = FinishDisposition::Released;
            return result;
        }
        result.disposition      = FinishDisposition::Catalogued;
        const std::uint32_t key = sequence_content_keys_[sequence.id];
        result.summary.endpoint = endpoint(key, finish_frontier);
        result.continuation.emplace(sequence.id, key);
        return result;
    }

    [[nodiscard]] FakeAbortResult abort(FakeSequenceHandle) noexcept {
        ++abort_calls;
        advance_revision();
        return FakeAbortResult{.status      = ConsumeStatus::Consumed,
                               .timings     = FakeTimings{.value = 7},
                               .speculative = FakeSpeculativeStats{.value = 9}};
    }

    [[nodiscard]] FakeReleaseResult
    release_continuation(FakeContinuationHandle&& continuation) noexcept {
        released_continuations.push_back(continuation.id);
        advance_revision();
        return FakeReleaseResult{.status = ConsumeStatus::Consumed};
    }

    [[nodiscard]] std::uint64_t resource_revision() const noexcept { return revision_; }

    [[nodiscard]] FakePhysicalUsage physical_usage() const noexcept { return usage; }

    void invalidate_resources() noexcept { advance_revision(); }

    std::size_t required_pressure_actions           = 0;
    std::uint32_t private_pressure_alternatives     = 1;
    std::uint64_t pressure_action_immediate_ns      = 100'000'000;
    std::uint32_t pressure_action_degradation_units = 1;
    bool include_cumulative_private_target          = false;
    bool combined_target_cancels_pressure_copy      = false;
    std::optional<std::uint64_t> pressure_target_immediate_ns_override;
    std::optional<std::uint64_t> required_action_id;
    bool require_evictions                  = false;
    bool abort_start                        = false;
    bool abort_progress                     = false;
    bool progress_in_progress_once          = false;
    bool finish_fail_next                   = false;
    bool finish_release                     = false;
    bool abort_capture_start                = false;
    ContextTransactionStatus capture_status = ContextTransactionStatus::Published;
    FakeCaptureAssessment capture_assessment;
    FakeContinuationSummary capture_summary;
    FakePhysicalUsage usage;

    std::uint64_t admission_inspections         = 0;
    std::uint64_t pressure_planning_sessions    = 0;
    std::uint64_t start_calls                   = 0;
    std::uint64_t finish_calls                  = 0;
    std::uint64_t abort_calls                   = 0;
    std::uint64_t skipped_captures              = 0;
    std::uint32_t finish_frontier               = 16;
    std::uint32_t started_source_id             = 0;
    ClaimDisposition started_source_disposition = ClaimDisposition::ConsumedToActive;
    std::vector<std::uint32_t> inspected_private_sources;
    std::vector<std::uint32_t> inspected_shared_sources;
    std::vector<std::vector<std::uint64_t>> seal_attempts;
    std::vector<std::uint64_t> started_action_ids;
    std::vector<std::uint32_t> selected_shared_capture_frontiers;
    std::vector<std::uint32_t> released_continuations;

private:
    void advance_revision() noexcept {
        if (++revision_ == 0) { ++revision_; }
    }

    std::uint64_t revision_            = 1;
    std::uint32_t planning_generation_ = 0;
    std::uint32_t next_sequence_id_    = 1;
    std::uint32_t next_shared_id_      = 1;
    std::array<std::uint32_t, 256> sequence_content_keys_{};
    TransactionKind transaction_kind_ = TransactionKind::None;
    FakePreparedPrompt pending_prompt_;
    std::optional<FakeResourcePlan> pending_plan_;
    bool pending_capture_publish_shared_ = false;
};

FakePressurePlanningSession::FakePressurePlanningSession(
    FakeProgram& program, const ninfer::runtime::ContextMachineCostModel& machine_cost,
    std::span<const FakeAdmissionCandidate* const> candidates,
    std::span<const FakeContinuationHandle* const> private_owners,
    std::span<const std::uint32_t> private_owner_ordinals,
    std::span<const FakeSharedPrefixHandle* const> shared_owners,
    std::span<const std::uint32_t> shared_owner_ordinals)
    : program_(&program), machine_cost_(&machine_cost), revision_(program.resource_revision()) {
    require(!candidates.empty(), "fake pressure session has no candidate");
    require(private_owners.size() == private_owner_ordinals.size() &&
                shared_owners.size() == shared_owner_ordinals.size(),
            "fake pressure owner arrays are not aligned");
    candidates_.assign(candidates.begin(), candidates.end());
    for (std::size_t index = 0; index < private_owners.size(); ++index) {
        owners_.push_back(Owner{.private_handle = private_owners[index],
                                .ordinal        = private_owner_ordinals[index],
                                .shared         = false});
    }
    for (std::size_t index = 0; index < shared_owners.size(); ++index) {
        owners_.push_back(Owner{.shared_handle = shared_owners[index],
                                .ordinal       = shared_owner_ordinals[index],
                                .shared        = true});
    }
    std::sort(owners_.begin(), owners_.end(),
              [](const Owner& left, const Owner& right) { return left.ordinal < right.ordinal; });
    options_.resize(candidates_.size());
    options_populated_.resize(candidates_.size());
    for (std::size_t index = 0; index < candidates_.size(); ++index) {
        require(candidates_[index] != nullptr, "fake pressure candidate is null");
        targets_.push_back(Target{
            .candidate_index = static_cast<std::uint32_t>(index),
            .choices         = std::vector<std::uint16_t>(owners_.size(), 0),
            .stable_ordinal  = static_cast<std::uint32_t>(index),
        });
    }
    if (++program.planning_generation_ == 0) { ++program.planning_generation_; }
    ++program.pressure_planning_sessions;
    generation_ = program.planning_generation_;
}

bool FakePressurePlanningSession::valid(FakePressureTargetHandle target) const noexcept {
    return program_ != nullptr && target.generation == generation_ &&
           target.index < targets_.size() && program_->resource_revision() == revision_;
}

std::uint32_t
FakePressurePlanningSession::candidate_index(const FakeAdmissionCandidate& candidate) const {
    const auto found = std::find(candidates_.begin(), candidates_.end(), &candidate);
    require(found != candidates_.end(), "fake pressure candidate is foreign");
    return static_cast<std::uint32_t>(found - candidates_.begin());
}

bool FakePressurePlanningSession::same_target(const Target& left,
                                              const Target& right) const noexcept {
    return left.candidate_index == right.candidate_index && left.choices == right.choices;
}

std::vector<FakeTargetDecision>
FakePressurePlanningSession::decisions_for(std::uint32_t selected_candidate,
                                           std::size_t owner_index) const {
    const FakeAdmissionCandidate& candidate = *candidates_.at(selected_candidate);
    const Owner& owner                      = owners_.at(owner_index);
    if ((!owner.shared && candidate.private_source_id != 0 &&
         candidate.private_source_id == owner.private_handle->id) ||
        (owner.shared && candidate.shared_source_id != 0 &&
         candidate.shared_source_id == owner.shared_handle->id)) {
        return {};
    }

    std::vector<FakeTargetDecision> decisions;
    if (!owner.shared) {
        for (std::uint32_t index = 0; index < program_->private_pressure_alternatives; ++index) {
            decisions.push_back(FakeTargetDecision{
                .id = 1000U + owner.private_handle->id + 10000U * static_cast<std::uint64_t>(index),
                .immediate_ns      = program_->pressure_action_immediate_ns,
                .degradation_units = program_->pressure_action_degradation_units,
            });
        }
        if (program_->include_cumulative_private_target) {
            decisions.push_back(FakeTargetDecision{
                .id                  = 5000U + owner.private_handle->id,
                .degradation_units   = 2,
                .dropped_checkpoints = 1,
            });
        }
        decisions.push_back(FakeTargetDecision{
            .id                  = 2000U + owner.private_handle->id,
            .degradation_units   = 4,
            .dropped_checkpoints = 1,
            .evicts_continuation = true,
        });
    } else {
        decisions.push_back(FakeTargetDecision{
            .id           = 3000U + owner.shared_handle->id,
            .shared_owner = true,
        });
        decisions.push_back(FakeTargetDecision{
            .id                  = 4000U + owner.shared_handle->id,
            .degradation_units   = 4,
            .dropped_checkpoints = 1,
            .evicts_continuation = true,
            .shared_owner        = true,
        });
    }
    return decisions;
}

void FakePressurePlanningSession::populate_options(std::uint32_t selected_candidate) {
    require(selected_candidate < candidates_.size(), "fake pressure candidate index is invalid");
    if (options_populated_[selected_candidate] != 0) { return; }
    options_[selected_candidate].resize(owners_.size());
    for (std::size_t index = 0; index < owners_.size(); ++index) {
        options_[selected_candidate][index] = decisions_for(selected_candidate, index);
    }
    options_populated_[selected_candidate] = 1;
}

FakePressureTargetHandle
FakePressurePlanningSession::identity_target(const FakeAdmissionCandidate& candidate) const {
    return FakePressureTargetHandle{.generation = generation_, .index = candidate_index(candidate)};
}

FakePressureTargetHandle
FakePressurePlanningSession::root_maximal_target(const FakeAdmissionCandidate& candidate) {
    const std::uint32_t selected = candidate_index(candidate);
    populate_options(selected);
    Target maximal{
        .candidate_index = selected,
        .choices         = std::vector<std::uint16_t>(owners_.size(), 0),
        .root_maximal    = true,
    };
    for (std::size_t index = 0; index < owners_.size(); ++index) {
        maximal.choices[index] = static_cast<std::uint16_t>(options_[selected][index].size());
    }
    auto found = std::find_if(targets_.begin(), targets_.end(),
                              [&](const Target& target) { return same_target(target, maximal); });
    if (found != targets_.end()) {
        found->root_maximal = true;
        return FakePressureTargetHandle{
            .generation = generation_,
            .index      = static_cast<std::uint32_t>(found - targets_.begin()),
        };
    }
    maximal.stable_ordinal = static_cast<std::uint32_t>(targets_.size());
    targets_.push_back(std::move(maximal));
    return FakePressureTargetHandle{
        .generation = generation_,
        .index      = static_cast<std::uint32_t>(targets_.size() - 1U),
    };
}

ninfer::runtime::PressureTargetAssessment
FakePressurePlanningSession::assess(FakePressureTargetHandle handle) {
    require(valid(handle) && !scratch_live_, "fake pressure assessment is stale");
    const Target& target = targets_[handle.index];
    populate_options(target.candidate_index);
    const FakeAdmissionCandidate& candidate = *candidates_[target.candidate_index];
    assessment_outcomes_.clear();
    assessment_impacts_.clear();
    std::vector<FakeTargetDecision> selected;
    std::uint32_t degradation_units = 0;
    std::uint32_t dropped           = 0;
    bool expandable                 = false;
    for (std::size_t index = 0; index < owners_.size(); ++index) {
        const std::uint16_t choice = target.choices[index];
        const auto& alternatives   = options_[target.candidate_index][index];
        if (choice == 0) {
            expandable = expandable || !alternatives.empty();
            continue;
        }
        require(choice <= alternatives.size(), "fake pressure target choice is invalid");
        const FakeTargetDecision& decision = alternatives[choice - 1U];
        selected.push_back(decision);
        degradation_units += decision.degradation_units;
        dropped += decision.dropped_checkpoints;
        assessment_outcomes_.push_back(ninfer::runtime::PressureOwnerOutcome{
            .owner_ordinal       = owners_[index].ordinal,
            .disposition         = decision.evicts_continuation ? ClaimDisposition::Evicted
                                                                : ClaimDisposition::Retained,
            .degradation_units   = decision.degradation_units,
            .dropped_checkpoints = decision.dropped_checkpoints,
            .shared              = owners_[index].shared,
        });
        if (decision.dropped_checkpoints != 0) {
            assessment_impacts_.push_back(ninfer::runtime::PressureCheckpointRecoveryImpact{
                .owner_ordinal = owners_[index].ordinal,
                .checkpoint =
                    CheckpointRef{.kind = owners_[index].shared ? CheckpointKind::SharedStablePrefix
                                                                : CheckpointKind::SessionEndpoint,
                                  .frontier = 16,
                                  .ordinal  = 0},
                .baseline_recovery_ns = 0,
                .target_recovery_ns   = 100,
            });
        }
        if (!decision.evicts_continuation) { expandable = true; }
    }

    ninfer::runtime::MaterializationMachineSummary machine = candidate.identity.machine;
    const bool combined_copy_cancelled =
        program_->combined_target_cancels_pressure_copy && selected.size() > 1U &&
        std::none_of(selected.begin(), selected.end(),
                     [](const auto& decision) { return decision.evicts_continuation; });
    if (!selected.empty() && program_->pressure_target_immediate_ns_override) {
        machine.immediate_ns = *program_->pressure_target_immediate_ns_override;
    } else if (combined_copy_cancelled) {
        // The complete target removes a transfer required by each partial target.  This models
        // source Move replacing Fork, a later eviction cancelling an earlier D2H, or two physical
        // actions coalescing into one direct stage.  Exact target cost is therefore intentionally
        // non-monotonic along the search edge.
        ++machine.immediate_ns;
    } else {
        for (const FakeTargetDecision& decision : selected) {
            machine.immediate_ns += decision.immediate_ns;
            ++machine.copy_operations;
        }
    }
    const bool identity  = selected.empty();
    std::uint64_t digest = candidate.identity.assessment_digest;
    if (!identity) {
        digest = 1469598103934665603ULL;
        for (const std::uint16_t choice : target.choices) {
            digest ^= choice;
            digest *= 1099511628211ULL;
        }
        digest ^= target.candidate_index;
    }
    return ninfer::runtime::PressureTargetAssessment{
        .physical_status       = program_->target_feasible(selected)
                                     ? ninfer::runtime::MaterializationPhysicalStatus::Feasible
                                     : ninfer::runtime::MaterializationPhysicalStatus::Infeasible,
        .source_disposition    = candidate.disposition,
        .machine               = machine,
        .owner_outcomes        = assessment_outcomes_,
        .checkpoint_impacts    = assessment_impacts_,
        .candidate_ordinal     = target.candidate_index,
        .stable_target_ordinal = target.stable_ordinal,
        .degradation_units     = degradation_units,
        .dropped_checkpoints   = dropped,
        .projection_work       = 1U + assessment_outcomes_.size(),
        .assessment_digest     = digest,
        .expandable            = expandable,
        .root_maximal          = target.root_maximal,
    };
}

FakePreparedPressureExpansion
FakePressurePlanningSession::prepare_expansion(FakePressureTargetHandle handle) {
    require(valid(handle) && !scratch_live_, "fake pressure expansion is stale");
    const Target& parent = targets_[handle.index];
    populate_options(parent.candidate_index);
    expansion_scratch_.clear();
    for (std::size_t owner = 0; owner < owners_.size(); ++owner) {
        const std::uint16_t current = parent.choices[owner];
        const auto& alternatives    = options_[parent.candidate_index][owner];
        if (current == 0) {
            for (std::size_t choice = 1; choice <= alternatives.size(); ++choice) {
                Target child         = parent;
                child.choices[owner] = static_cast<std::uint16_t>(choice);
                child.root_maximal   = false;
                if (std::none_of(expansion_scratch_.begin(), expansion_scratch_.end(),
                                 [&](const Target& prior) { return same_target(prior, child); })) {
                    expansion_scratch_.push_back(std::move(child));
                }
            }
        } else if (current <= alternatives.size() &&
                   !alternatives[current - 1U].evicts_continuation) {
            Target child         = parent;
            child.choices[owner] = static_cast<std::uint16_t>(alternatives.size());
            child.root_maximal   = false;
            if (std::none_of(expansion_scratch_.begin(), expansion_scratch_.end(),
                             [&](const Target& prior) { return same_target(prior, child); })) {
                expansion_scratch_.push_back(std::move(child));
            }
        }
    }
    std::uint32_t new_count = 0;
    for (const Target& child : expansion_scratch_) {
        if (std::none_of(targets_.begin(), targets_.end(),
                         [&](const Target& prior) { return same_target(prior, child); })) {
            ++new_count;
        }
    }
    if (++scratch_generation_ == 0) { ++scratch_generation_; }
    scratch_live_ = true;
    return FakePreparedPressureExpansion(generation_, scratch_generation_, new_count);
}

FakePressureExpansionView
FakePressurePlanningSession::commit_expansion(FakePreparedPressureExpansion&& prepared) {
    require(scratch_live_ && prepared.generation == generation_ &&
                prepared.scratch_generation == scratch_generation_,
            "fake prepared expansion is stale");
    committed_children_.clear();
    std::uint32_t new_count = 0;
    for (Target& child : expansion_scratch_) {
        auto found          = std::find_if(targets_.begin(), targets_.end(),
                                           [&](const Target& prior) { return same_target(prior, child); });
        std::uint32_t index = 0;
        if (found == targets_.end()) {
            child.stable_ordinal = static_cast<std::uint32_t>(targets_.size());
            targets_.push_back(std::move(child));
            index = static_cast<std::uint32_t>(targets_.size() - 1U);
            ++new_count;
        } else {
            index = static_cast<std::uint32_t>(found - targets_.begin());
        }
        committed_children_.push_back(
            FakePressureTargetHandle{.generation = generation_, .index = index});
    }
    require(new_count == prepared.new_count, "fake expansion count changed before commit");
    expansion_scratch_.clear();
    scratch_live_ = false;
    return FakePressureExpansionView{
        .children            = committed_children_,
        .new_canonical_count = new_count,
    };
}

void FakePressurePlanningSession::discard_expansion(
    FakePreparedPressureExpansion&& prepared) noexcept {
    if (scratch_live_ && prepared.generation == generation_ &&
        prepared.scratch_generation == scratch_generation_) {
        expansion_scratch_.clear();
        scratch_live_ = false;
    }
}

std::optional<FakeResourcePlan> FakePressurePlanningSession::seal(FakePressureTargetHandle handle,
                                                                  const FakePreparedPrompt&) {
    require(valid(handle) && !scratch_live_, "fake pressure seal is stale");
    const Target& target  = targets_[handle.index];
    const auto assessment = assess(handle);
    if (assessment.physical_status != ninfer::runtime::MaterializationPhysicalStatus::Feasible) {
        return std::nullopt;
    }
    FakeResourcePlan plan(*candidates_[target.candidate_index], revision_);
    std::vector<std::uint64_t> action_ids;
    for (std::size_t index = 0; index < owners_.size(); ++index) {
        const std::uint16_t choice = target.choices[index];
        if (choice == 0) { continue; }
        const FakeTargetDecision& decision = options_[target.candidate_index][index][choice - 1U];
        action_ids.push_back(decision.id);
        if (owners_[index].shared) {
            plan.shared_actions.push_back(decision);
            plan.shared_owner_ids.push_back(owners_[index].shared_handle->id);
        } else {
            plan.private_actions.push_back(decision);
            plan.private_owner_ids.push_back(owners_[index].private_handle->id);
        }
    }
    program_->seal_attempts.push_back(std::move(action_ids));
    return plan;
}

std::optional<FakeResourcePlan>
FakePressurePlanningSession::seal_capture(FakePressureTargetHandle handle) {
    return seal(handle, FakePreparedPrompt{});
}

FakePressurePlanningSession
FakeProgram::begin_pressure_planning(const ninfer::runtime::ContextMachineCostModel& machine_cost,
                                     std::span<const FakeAdmissionCandidate* const> candidates,
                                     std::span<const FakeContinuationHandle* const> private_owners,
                                     std::span<const std::uint32_t> private_owner_ordinals,
                                     std::span<const FakeSharedPrefixHandle* const> shared_owners,
                                     std::span<const std::uint32_t> shared_owner_ordinals) {
    return FakePressurePlanningSession(*this, machine_cost, candidates, private_owners,
                                       private_owner_ordinals, shared_owners,
                                       shared_owner_ordinals);
}

struct FakePackage {
    using Program                    = FakeProgram;
    using PreparedPrompt             = FakePreparedPrompt;
    using RequestBasePlan            = FakeRequestBasePlan;
    using AdmissionCandidate         = FakeAdmissionCandidate;
    using ResourcePlan               = FakeResourcePlan;
    using PersistentBackfillProof    = FakePersistentBackfillProof;
    using SequenceHandle             = FakeSequenceHandle;
    using ContinuationHandle         = FakeContinuationHandle;
    using SharedPrefixHandle         = FakeSharedPrefixHandle;
    using CaptureOffer               = FakeCaptureOffer;
    using ContinuationSummary        = FakeContinuationSummary;
    using SharedPrefixSummary        = FakeSharedPrefixSummary;
    using CaptureAssessment          = FakeCaptureAssessment;
    using CapturePressurePlan        = FakeResourcePlan;
    using ActiveCaptureResult        = FakeActiveCaptureResult;
    using ContextTransactionProgress = FakeContextTransactionProgress;
    using MaterializationResult      = FakeMaterializationResult;
    using StartResult                = FakeStartResult;
    using FinishResult               = FakeFinishResult;
    using AbortResult                = FakeAbortResult;
    using PressureTargetHandle       = FakePressureTargetHandle;
    using CommitResult               = FakeCommitResult;
    using DiscardResult              = FakeDiscardResult;
    using CacheSessionKey            = FakeCacheSessionKey;
};

using FakeManager = ninfer::runtime::ResourceManager<FakePackage>;

FakeManager make_manager(std::uint32_t lanes = 1, std::uint32_t private_capacity = 4,
                         std::uint32_t shared_capacity = 0, bool cache_enabled = true) {
    return FakeManager(lanes, private_capacity, shared_capacity, cache_enabled, 2,
                       test_cost_model());
}

struct ActiveRequest {
    LaneId lane;
    FakeSequenceHandle sequence;
};

ActiveRequest start_active(FakeManager& manager, FakeProgram& program, std::uint32_t content_key,
                           const FakeRequestBasePlan& base, std::uint64_t publication_order) {
    auto inspection =
        manager.inspect(program, FakePreparedPrompt{content_key}, base, publication_order);
    require(inspection.choice.has_value(), "request did not produce an admission choice");
    const LaneId lane   = inspection.choice->destination();
    const auto reserved = manager.reserve_materialization(program, std::move(*inspection.choice),
                                                          FakePreparedPrompt{content_key}, {});
    require(reserved == FakeManager::MaterializationReserveResult::Reserved,
            "request materialization was not reserved");
    auto outcome = [&]() -> FakeManager::MaterializationOutcome {
        auto progress = manager.progress_context_transaction(program, {});
        if (!std::holds_alternative<ContextTransactionInProgress>(progress)) {
            return std::get<FakeManager::MaterializationOutcome>(std::move(progress));
        }
        auto completed = manager.progress_context_transaction(program, {});
        return std::get<FakeManager::MaterializationOutcome>(std::move(completed));
    }();
    require(outcome.status == ContextTransactionStatus::Published && outcome.activation,
            "request materialization did not publish");
    auto activation                   = std::move(*outcome.activation);
    const FakeSequenceHandle sequence = activation.sequence();
    manager.adopt(program, std::move(activation));
    require(manager.lane_state(lane) == ninfer::runtime::LogicalLaneState::Active,
            "published lane was not adopted as active");
    return ActiveRequest{.lane = lane, .sequence = sequence};
}

void test_private_portfolio_loss_keeps_checkpoint_identity_fixed() {
    using ninfer::runtime::ContextPortfolioCheckpointValue;
    using ninfer::runtime::ContextPortfolioOwnerPolicy;
    using ninfer::runtime::ContextPortfolioValue;

    const std::array owners{
        ContextPortfolioOwnerPolicy{.ordinal = 0, .private_retention_weight = 4},
    };
    const std::array checkpoints{
        ContextPortfolioCheckpointValue{
            .owner_ordinal        = 0,
            .rebuild_ns           = 1000,
            .baseline_recovery_ns = 100,
            .target_recovery_ns   = 100,
        },
        ContextPortfolioCheckpointValue{
            .owner_ordinal        = 0,
            .rebuild_ns           = 800,
            .baseline_recovery_ns = 100,
            .target_recovery_ns   = 800,
        },
    };
    ContextPortfolioValue value;
    const auto result = value.fold(owners, checkpoints);
    require(result.baseline_public_value == 0 && result.target_public_value == 0 &&
                result.private_transition_loss == 2800 && !result.saturated,
            "a surviving endpoint masked loss of an earlier private checkpoint");
}

void test_portfolio_demand_and_owner_aggregation() {
    using ninfer::runtime::ContextPortfolioCheckpointValue;
    using ninfer::runtime::ContextPortfolioOwnerPolicy;
    using ninfer::runtime::ContextPortfolioValue;

    {
        const std::array owners{ContextPortfolioOwnerPolicy{.ordinal = 0}};
        const std::array checkpoints{
            ContextPortfolioCheckpointValue{
                .owner_ordinal        = 0,
                .demand_mask          = 1,
                .rebuild_ns           = 1000,
                .baseline_recovery_ns = 200,
                .target_recovery_ns   = 500,
            },
            ContextPortfolioCheckpointValue{
                .owner_ordinal        = 0,
                .demand_mask          = 1,
                .rebuild_ns           = 800,
                .baseline_recovery_ns = 200,
                .target_recovery_ns   = 400,
            },
        };
        ContextPortfolioValue value;
        const auto result = value.fold(owners, checkpoints);
        require(result.baseline_public_value == 800 && result.target_public_value == 500 &&
                    result.private_transition_loss == 0,
                "nested checkpoints counted one empirical demand more than once");
    }

    {
        const std::array owners{
            ContextPortfolioOwnerPolicy{.ordinal = 0, .private_retention_weight = 1},
            ContextPortfolioOwnerPolicy{.ordinal = 1, .private_retention_weight = 4},
        };
        const std::array checkpoints{
            ContextPortfolioCheckpointValue{
                .owner_ordinal        = 0,
                .rebuild_ns           = 1000,
                .baseline_recovery_ns = 100,
                .target_recovery_ns   = 400,
            },
            ContextPortfolioCheckpointValue{
                .owner_ordinal        = 1,
                .rebuild_ns           = 1000,
                .baseline_recovery_ns = 100,
                .target_recovery_ns   = 200,
            },
        };
        ContextPortfolioValue value;
        const auto result = value.fold(owners, checkpoints);
        require(result.private_transition_loss == 700,
                "private checkpoint transition losses were not summed across owners");
    }
}

void test_shared_capture_subtracts_private_transition_loss() {
    using Planner = ninfer::runtime::SharedCapturePlanner<FakePackage>;

    FakeProgram program;
    program.required_pressure_actions             = 1;
    program.require_evictions                     = true;
    program.pressure_target_immediate_ns_override = 0;
    FakeCaptureAssessment capture{
        .shared_evidence       = ninfer::SharedCandidateEvidence::ExplicitBoundary,
        .projected_recovery_ns = 0,
        .publishes_shared      = true,
        .physically_feasible   = false,
    };
    FakeContinuationHandle owner{7, 0};
    const std::array<const FakeContinuationHandle*, 1> private_owners{&owner};
    const std::array<std::uint32_t, 1> private_ordinals{0};
    const std::array<Planner::OwnerPolicy, 1> owner_policies{
        Planner::OwnerPolicy{.ordinal = 0, .private_retention_weight = 4},
    };
    const std::array<Planner::CheckpointPolicy, 1> checkpoint_policies{
        Planner::CheckpointPolicy{
            .owner_ordinal        = 0,
            .checkpoint           = CheckpointRef{.kind     = CheckpointKind::SessionEndpoint,
                                                  .frontier = 16,
                                                  .ordinal  = 0},
            .rebuild_ns           = 1000,
            .baseline_recovery_ns = 0,
        },
    };

    Planner planner;
    const auto result = planner.plan(program, test_cost_model(),
                                     Planner::Input{
                                         .capture                = &capture,
                                         .private_owners         = private_owners,
                                         .private_owner_ordinals = private_ordinals,
                                         .shared_owners          = {},
                                         .shared_owner_ordinals  = {},
                                         .owner_policies         = owner_policies,
                                         .checkpoint_policies    = checkpoint_policies,
                                         .candidate_rebuild_ns   = 1000,
                                     });
    require(result && result->baseline_value == 0 && result->target_value == 1000 &&
                result->immediate_ns == 0 && result->net_gain == 600,
            "shared capture gain did not subtract the private capability transition loss");
}

void test_equal_lower_bound_does_not_short_circuit_tie_break() {
    using Planner = ninfer::runtime::MaterializationPlanner<FakePackage>;

    FakeProgram program;
    program.required_pressure_actions         = 1;
    program.pressure_action_immediate_ns      = 0;
    program.pressure_action_degradation_units = 0;

    FakeAdmissionCandidate incumbent;
    incumbent.identity.machine.minimum_request_ns = 1'000'000'000;
    incumbent.identity.machine.immediate_ns       = 1'000'000'000;
    incumbent.identity.machine.copy_operations    = 1;
    incumbent.identity.physical_status = ninfer::runtime::MaterializationPhysicalStatus::Feasible;
    incumbent.identity.source_disposition = ClaimDisposition::ConsumedToActive;
    incumbent.identity.assessment_digest  = 11;

    FakeAdmissionCandidate tied_pressure;
    tied_pressure.identity.machine.minimum_request_ns = 1'000'000'000;
    tied_pressure.identity.machine.immediate_ns       = 1'000'000'000;
    tied_pressure.identity.physical_status =
        ninfer::runtime::MaterializationPhysicalStatus::Infeasible;
    tied_pressure.identity.source_disposition = ClaimDisposition::ConsumedToActive;
    tied_pressure.identity.expandable         = true;
    tied_pressure.identity.assessment_digest  = 22;

    std::array<Planner::CandidateInput, 2> candidates{
        Planner::CandidateInput{
            .candidate = &incumbent, .stable_ordinal = 0, .current_session_binding = false},
        Planner::CandidateInput{
            .candidate = &tied_pressure, .stable_ordinal = 1, .current_session_binding = true},
    };
    FakeContinuationHandle owner{7, 0};
    const std::array<const FakeContinuationHandle*, 1> private_owners{&owner};
    const std::array<std::uint32_t, 1> private_ordinals{0};
    const std::array<ninfer::runtime::MaterializationOwnerPolicy, 1> owner_policy{
        ninfer::runtime::MaterializationOwnerPolicy{.ordinal = 0},
    };
    const std::array<ninfer::runtime::MaterializationCheckpointPolicy, 1> checkpoint_policy{
        ninfer::runtime::MaterializationCheckpointPolicy{
            .owner_ordinal = 0,
            .checkpoint    = CheckpointRef{.kind     = CheckpointKind::SessionEndpoint,
                                           .frontier = 16,
                                           .ordinal  = 0},
            .rebuild_ns    = 100,
        },
    };

    Planner planner;
    const auto logical_goal = [](std::uint32_t, ClaimDisposition,
                                 std::span<const ninfer::runtime::PressureOwnerOutcome>)
        -> std::optional<Planner::LogicalGoal> {
        return Planner::LogicalGoal{.publication_slot = 0};
    };
    const auto pressure_inputs = [&]() -> Planner::PressureInputs {
        return Planner::PressureInputs{
            .private_owners         = private_owners,
            .private_owner_ordinals = private_ordinals,
            .shared_owners          = {},
            .shared_owner_ordinals  = {},
            .owner_policy           = owner_policy,
            .checkpoint_policy      = checkpoint_policy,
        };
    };
    auto result = planner.plan(program, FakePreparedPrompt{}, test_cost_model(), candidates, 0,
                               pressure_inputs, logical_goal, Planner::Clock::now());

    require(result && result->candidate_index == 1 && program.pressure_planning_sessions == 1,
            "equal lower bound bypassed the pressure target that wins the stable tie-break");
}

void test_feasible_identity_expands_when_pressure_can_remove_copy() {
    using Planner = ninfer::runtime::MaterializationPlanner<FakePackage>;

    FakeProgram program;
    program.pressure_action_immediate_ns          = 0;
    program.pressure_action_degradation_units     = 1;
    program.pressure_target_immediate_ns_override = 100'000'000;

    FakeAdmissionCandidate candidate;
    candidate.identity.machine.minimum_request_ns = 100'000'000;
    candidate.identity.machine.immediate_ns       = 1'000'000'000;
    candidate.identity.machine.copy_operations    = 1;
    candidate.identity.physical_status = ninfer::runtime::MaterializationPhysicalStatus::Feasible;
    candidate.identity.source_disposition = ClaimDisposition::ConsumedToActive;
    candidate.identity.assessment_digest  = 17;

    const std::array<Planner::CandidateInput, 1> candidates{
        Planner::CandidateInput{
            .candidate = &candidate, .stable_ordinal = 0, .current_session_binding = false},
    };
    FakeContinuationHandle owner{9, 0};
    const std::array<const FakeContinuationHandle*, 1> private_owners{&owner};
    const std::array<std::uint32_t, 1> private_ordinals{0};
    const std::array<ninfer::runtime::MaterializationOwnerPolicy, 1> owner_policy{
        ninfer::runtime::MaterializationOwnerPolicy{.ordinal = 0},
    };
    const std::array<ninfer::runtime::MaterializationCheckpointPolicy, 1> checkpoint_policy{
        ninfer::runtime::MaterializationCheckpointPolicy{
            .owner_ordinal = 0,
            .checkpoint    = CheckpointRef{.kind     = CheckpointKind::SessionEndpoint,
                                           .frontier = 16,
                                           .ordinal  = 0},
            .rebuild_ns    = 100,
        },
    };

    Planner planner;
    const auto pressure_inputs = [&]() -> Planner::PressureInputs {
        return Planner::PressureInputs{
            .private_owners         = private_owners,
            .private_owner_ordinals = private_ordinals,
            .shared_owners          = {},
            .shared_owner_ordinals  = {},
            .owner_policy           = owner_policy,
            .checkpoint_policy      = checkpoint_policy,
        };
    };
    const auto logical_goal = [](std::uint32_t, ClaimDisposition,
                                 std::span<const ninfer::runtime::PressureOwnerOutcome>)
        -> std::optional<Planner::LogicalGoal> {
        return Planner::LogicalGoal{.publication_slot = 0};
    };
    auto result = planner.plan(program, FakePreparedPrompt{}, test_cost_model(), candidates, 0,
                               pressure_inputs, logical_goal, Planner::Clock::now());

    require(result && result->diagnostics.predicted_now_ns == 100'000'000 &&
                result->diagnostics.selected_degradation_units == 1 &&
                program.pressure_planning_sessions == 1 && !program.seal_attempts.empty() &&
                program.seal_attempts.back() == std::vector<std::uint64_t>{1009},
            "feasible identity suppressed a cheaper complete pressure target");
}

void test_dominating_identity_does_not_build_pressure_graph() {
    using Planner = ninfer::runtime::MaterializationPlanner<FakePackage>;

    FakeProgram program;
    FakeAdmissionCandidate candidate;
    candidate.identity.machine.minimum_request_ns = 100'000'000;
    candidate.identity.machine.immediate_ns       = 100'000'000;
    candidate.identity.physical_status = ninfer::runtime::MaterializationPhysicalStatus::Feasible;
    candidate.identity.source_disposition = ClaimDisposition::ConsumedToActive;
    candidate.identity.assessment_digest  = 23;
    const std::array<Planner::CandidateInput, 1> candidates{
        Planner::CandidateInput{
            .candidate = &candidate, .stable_ordinal = 0, .current_session_binding = false},
    };

    bool pressure_inputs_built = false;
    const auto pressure_inputs = [&]() -> Planner::PressureInputs {
        pressure_inputs_built = true;
        return {};
    };
    const auto logical_goal = [](std::uint32_t, ClaimDisposition,
                                 std::span<const ninfer::runtime::PressureOwnerOutcome>)
        -> std::optional<Planner::LogicalGoal> {
        return Planner::LogicalGoal{.publication_slot = 0};
    };

    Planner planner;
    auto result = planner.plan(program, FakePreparedPrompt{}, test_cost_model(), candidates, 0,
                               pressure_inputs, logical_goal, Planner::Clock::now());
    require(result &&
                result->diagnostics.stop_reason == ninfer::MaterializationStopReason::NoPressure &&
                !pressure_inputs_built && program.pressure_planning_sessions == 0,
            "dominating identity eagerly constructed the pressure graph");
}

FakeFinishResult finish_active(FakeManager& manager, FakeProgram& program, ActiveRequest request,
                               std::uint32_t frontier = 16) {
    program.finish_frontier = frontier;
    manager.mark_terminal_pending(request.lane);
    return manager.finish(program, request.lane, request.sequence);
}

void test_root_lifecycle_and_prefix_reuse() {
    FakeManager manager = make_manager(1, 2);
    FakeProgram program;
    const ActiveRequest first = start_active(manager, program, 7, make_base(7), 1);
    require(program.pressure_planning_sessions == 0,
            "no-pressure root admission created a pressure planning session");
    const FakeFinishResult finish = finish_active(manager, program, first);
    require(finish.status == ConsumeStatus::Consumed &&
                finish.disposition == FinishDisposition::Catalogued,
            "terminal continuation was not catalogued");
    require(manager.lane_state(first.lane) == ninfer::runtime::LogicalLaneState::Free,
            "terminal lane did not return to Free");

    auto reuse = manager.inspect(program, FakePreparedPrompt{7}, make_base(7), 2);
    require(reuse.readiness == Readiness::Ready && reuse.choice,
            "catalogued endpoint was not reusable");
    require(reuse.choice->summary().reusable_prompt_tokens == 16,
            "endpoint reuse frontier was not selected");
    program.abort_start = true;
    const auto status   = manager.reserve_materialization(program, std::move(*reuse.choice),
                                                          FakePreparedPrompt{7}, {});
    require(status == FakeManager::MaterializationReserveResult::Stale &&
                program.started_source_id == first.sequence.id,
            "selected endpoint did not reach the sealed Program plan");
    require(manager.catalog_state(0) == FakeManager::CatalogState::Catalogued,
            "failed start did not roll back its logical source claim");
}

void test_stale_revision_is_retryable() {
    FakeManager manager = make_manager();
    FakeProgram program;
    auto inspection = manager.inspect(program, FakePreparedPrompt{1}, make_base(1), 1);
    require(inspection.choice.has_value(), "root choice was not produced");
    const std::uint64_t start_calls = program.start_calls;
    program.invalidate_resources();
    const auto status = manager.reserve_materialization(program, std::move(*inspection.choice),
                                                        FakePreparedPrompt{1}, {});
    require(status == FakeManager::MaterializationReserveResult::Stale,
            "revision mismatch was not reported as retryable stale work");
    require(program.start_calls == start_calls,
            "known-stale plan was incorrectly passed into Program start");
    require(manager.lane_state(LaneId{0}) == ninfer::runtime::LogicalLaneState::Free,
            "known-stale plan changed logical lane state");
}

void test_materialization_abort_preserves_source() {
    FakeManager manager = make_manager(1, 2);
    FakeProgram program;
    const ActiveRequest seed = start_active(manager, program, 5, make_base(5), 1);
    (void)finish_active(manager, program, seed);

    auto inspection = manager.inspect(program, FakePreparedPrompt{5}, make_base(5), 2);
    require(inspection.choice && inspection.choice->summary().reusable_prompt_tokens == 16,
            "abort test did not select its private source");
    program.abort_progress = true;
    require(manager.reserve_materialization(program, std::move(*inspection.choice),
                                            FakePreparedPrompt{5}, {}) ==
                FakeManager::MaterializationReserveResult::Reserved,
            "abort test could not reserve materialization");
    auto progress = manager.progress_context_transaction(program, {});
    auto outcome  = std::get<FakeManager::MaterializationOutcome>(std::move(progress));
    require(outcome.status == ContextTransactionStatus::Aborted && !outcome.activation,
            "cancelled materialization did not abort before publication");
    require(manager.catalog_state(0) == FakeManager::CatalogState::Catalogued,
            "aborted Move did not restore its source visibility");

    program.abort_progress = false;
    program.abort_start    = true;
    auto retry             = manager.inspect(program, FakePreparedPrompt{5}, make_base(5), 3);
    require(retry.choice && retry.choice->summary().reusable_prompt_tokens == 16,
            "restored source was not reusable after abort");
    (void)manager.reserve_materialization(program, std::move(*retry.choice), FakePreparedPrompt{5},
                                          {});
    require(program.started_source_id == seed.sequence.id,
            "abort restored the wrong source capability");
}

void test_committed_victim_survives_transaction_abort() {
    FakeManager manager = make_manager(1, 2);
    FakeProgram program;
    const ActiveRequest first = start_active(manager, program, 10, make_base(10), 1);
    (void)finish_active(manager, program, first);
    const ActiveRequest second = start_active(manager, program, 20, make_base(20), 2);
    (void)finish_active(manager, program, second);

    auto inspection = manager.inspect(program, FakePreparedPrompt{30}, make_base(30), 3);
    require(inspection.choice.has_value(), "full catalog did not produce an eviction closure");
    program.abort_progress = true;
    require(manager.reserve_materialization(program, std::move(*inspection.choice),
                                            FakePreparedPrompt{30}, {}) ==
                FakeManager::MaterializationReserveResult::Reserved,
            "evicting materialization was not reserved");
    auto progress = manager.progress_context_transaction(program, {});
    auto outcome  = std::get<FakeManager::MaterializationOutcome>(std::move(progress));
    require(outcome.status == ContextTransactionStatus::Aborted,
            "pressure transaction did not take the abort path");
    const std::uint32_t catalogued =
        (manager.catalog_state(0) == FakeManager::CatalogState::Catalogued ? 1U : 0U) +
        (manager.catalog_state(1) == FakeManager::CatalogState::Catalogued ? 1U : 0U);
    require(catalogued == 1,
            "committed victim eviction was incorrectly rolled back with request-local abort");
}

void test_uncommitted_pressure_acknowledgement_is_not_degradation() {
    FakeManager manager = make_manager(1, 2);
    FakeProgram program;
    const ActiveRequest seed = start_active(manager, program, 40, make_base(40), 1);
    (void)finish_active(manager, program, seed);

    program.required_pressure_actions = 1;
    auto inspection = manager.inspect(program, FakePreparedPrompt{41}, make_base(41), 2);
    require(inspection.choice.has_value(),
            "pressure-abort test did not select a preserving pressure target");
    program.abort_progress = true;
    require(manager.reserve_materialization(program, std::move(*inspection.choice),
                                            FakePreparedPrompt{41}, {}) ==
                FakeManager::MaterializationReserveResult::Reserved,
            "pressure-abort test could not reserve materialization");
    require(program.started_action_ids.size() == 1 &&
                program.started_action_ids.front() == 1000U + seed.sequence.id,
            "pressure-abort test selected an eviction instead of preserving pressure");

    auto progress = manager.progress_context_transaction(program, {});
    auto outcome  = std::get<FakeManager::MaterializationOutcome>(std::move(progress));
    RuntimeStats stats;
    manager.populate_runtime_stats(program, stats);
    require(outcome.status == ContextTransactionStatus::Aborted &&
                manager.catalog_state(0) == FakeManager::CatalogState::Catalogued,
            "uncommitted pressure acknowledgement changed owner availability");
    require(stats.pressure_private_owners_degraded == 0 &&
                stats.pressure_private_owners_evicted == 0 &&
                stats.pressure_checkpoints_dropped == 0,
            "uncommitted pressure acknowledgement was counted as a degradation");
    require(stats.pressure_searches == 1,
            "accepted pressure plan was hidden when its request later aborted");
}

void test_aborted_source_selection_does_not_create_hit_history() {
    FakeManager manager = make_manager(1, 3);
    FakeProgram program;
    const ActiveRequest first = start_active(manager, program, 61, make_base(61), 1);
    (void)finish_active(manager, program, first);
    const ActiveRequest second = start_active(manager, program, 62, make_base(62), 2);
    (void)finish_active(manager, program, second);

    auto reuse = manager.inspect(program, FakePreparedPrompt{61}, make_base(61), 3);
    require(reuse.choice && reuse.choice->summary().reusable_prompt_tokens == 16,
            "cancelled-hit test did not select its exact source");
    program.abort_progress = true;
    require(manager.reserve_materialization(program, std::move(*reuse.choice),
                                            FakePreparedPrompt{61}, {}) ==
                FakeManager::MaterializationReserveResult::Reserved,
            "cancelled-hit test could not reserve exact reuse");
    auto progress = manager.progress_context_transaction(program, {});
    auto outcome  = std::get<FakeManager::MaterializationOutcome>(std::move(progress));
    require(outcome.status == ContextTransactionStatus::Aborted,
            "cancelled-hit test unexpectedly published its request");

    program.abort_progress            = false;
    program.required_pressure_actions = 1;
    program.require_evictions         = true;
    auto pressure = manager.inspect(program, FakePreparedPrompt{63}, make_base(63), 4);
    require(pressure.choice.has_value(), "cancelled-hit test could not plan pressure");
    program.abort_start = true;
    (void)manager.reserve_materialization(program, std::move(*pressure.choice),
                                          FakePreparedPrompt{63}, {});
    require(program.started_action_ids.size() == 1 &&
                program.started_action_ids.front() == 2000U + first.sequence.id,
            "aborted source selection incorrectly biased later retention policy");
}

void test_retained_source_is_protected_until_terminal() {
    FakeManager manager = make_manager(2, 3);
    FakeProgram program;
    const FakeCacheSessionKey first_session{1};
    const FakeCacheSessionKey second_session{2};
    const ActiveRequest seed = start_active(
        manager, program, 9, make_base(9, first_session, RetentionClass::LiveSession), 1);
    (void)finish_active(manager, program, seed);

    const ActiveRequest fork = start_active(
        manager, program, 9, make_base(9, second_session, RetentionClass::LiveSession), 2);
    require(program.started_source_disposition == ClaimDisposition::Retained,
            "different-session source was destructively moved");

    program.required_pressure_actions = 1;
    auto blocked = manager.inspect(program, FakePreparedPrompt{77}, make_base(77), 3);
    require(blocked.readiness == Readiness::TemporarilyBlocked && !blocked.choice,
            "active retained source was exposed as a pressure victim");

    (void)manager.abort(program, fork.lane, fork.sequence);
    program.abort_start = true;
    auto available      = manager.inspect(program, FakePreparedPrompt{77}, make_base(77), 4);
    require(available.choice.has_value(),
            "terminal release did not return retained source to pressure policy");
    (void)manager.reserve_materialization(program, std::move(*available.choice),
                                          FakePreparedPrompt{77}, {});
    require(!program.started_action_ids.empty(),
            "released source did not participate in the sealed pressure plan");
}

void test_session_publication_order_controls_tied_source() {
    FakeManager manager = make_manager(2, 3);
    FakeProgram program;
    const FakeCacheSessionKey session{42};
    const FakeRequestBasePlan base = make_base(42, session, RetentionClass::LiveSession, true);

    const ActiveRequest older = start_active(manager, program, 42, base, 10);
    const ActiveRequest newer = start_active(manager, program, 42, base, 20);
    (void)finish_active(manager, program, newer, 16);
    (void)finish_active(manager, program, older, 16);

    auto next = manager.inspect(program, FakePreparedPrompt{42}, base, 30);
    require(next.choice && next.choice->summary().reusable_prompt_tokens == 16,
            "same-session endpoint candidates were not reusable");
    program.abort_start = true;
    (void)manager.reserve_materialization(program, std::move(*next.choice), FakePreparedPrompt{42},
                                          {});
    require(program.started_source_id == newer.sequence.id,
            "older out-of-order finish displaced the current session binding on a tied cost");

    program.abort_start             = false;
    const ActiveRequest replacement = start_active(
        manager, program, 99, make_base(99, session, RetentionClass::LiveSession, true), 40);
    (void)finish_active(manager, program, replacement, 16);
    require(program.released_continuations.empty(),
            "session publication synchronously released an old physical continuation");
    require(manager.catalog_state(0) == FakeManager::CatalogState::Catalogued &&
                manager.catalog_state(1) == FakeManager::CatalogState::Catalogued &&
                manager.catalog_state(2) == FakeManager::CatalogState::Catalogued,
            "session replacement did not retain its old binding as anonymous cache");
}

void test_canonical_pressure_starts_with_disposable_owner() {
    FakeManager manager = make_manager(1, 3);
    FakeProgram program;
    const ActiveRequest disposable = start_active(
        manager, program, 1, make_base(1, std::nullopt, RetentionClass::Disposable), 1);
    (void)finish_active(manager, program, disposable);
    const ActiveRequest live = start_active(
        manager, program, 2, make_base(2, FakeCacheSessionKey{2}, RetentionClass::LiveSession), 2);
    (void)finish_active(manager, program, live);

    program.required_pressure_actions = 1;
    auto inspection = manager.inspect(program, FakePreparedPrompt{3}, make_base(3), 3);
    require(inspection.choice.has_value(), "canonical pressure did not find a feasible prefix");
    program.abort_start = true;
    (void)manager.reserve_materialization(program, std::move(*inspection.choice),
                                          FakePreparedPrompt{3}, {});
    require(program.started_action_ids.size() == 1 &&
                program.started_action_ids.front() == 1000U + disposable.sequence.id,
            "canonical pressure did not degrade Disposable before LiveSession");
}

void test_pressure_tries_every_preserving_alternative_before_eviction() {
    FakeManager manager = make_manager(1, 2);
    FakeProgram program;
    const ActiveRequest seed = start_active(manager, program, 15, make_base(15), 1);
    (void)finish_active(manager, program, seed);

    program.required_pressure_actions     = 1;
    program.private_pressure_alternatives = 2;
    program.required_action_id            = 11000U + seed.sequence.id;
    auto inspection = manager.inspect(program, FakePreparedPrompt{25}, make_base(25), 2);
    require(inspection.choice.has_value(), "second preserving pressure alternative was skipped");

    program.abort_start = true;
    (void)manager.reserve_materialization(program, std::move(*inspection.choice),
                                          FakePreparedPrompt{25}, {});
    require(program.started_action_ids.size() == 1 &&
                program.started_action_ids.front() == *program.required_action_id,
            "pressure escalated before trying the feasible preserving alternative");
}

void test_cumulative_owner_target_closes_pressure_without_eviction() {
    FakeManager manager = make_manager(1, 2);
    FakeProgram program;
    const ActiveRequest seed = start_active(manager, program, 31, make_base(31), 1);
    (void)finish_active(manager, program, seed);

    program.required_pressure_actions         = 1;
    program.include_cumulative_private_target = true;
    program.required_action_id                = 5000U + seed.sequence.id;
    auto inspection = manager.inspect(program, FakePreparedPrompt{32}, make_base(32), 2);
    require(inspection.choice.has_value(),
            "cumulative checkpoint-drop and spill owner target was unreachable");

    program.abort_start = true;
    (void)manager.reserve_materialization(program, std::move(*inspection.choice),
                                          FakePreparedPrompt{32}, {});
    require(program.started_action_ids.size() == 1 &&
                program.started_action_ids.front() == *program.required_action_id,
            "planner replaced a feasible cumulative owner target with eviction");
}

void test_two_owners_jointly_close_pressure() {
    FakeManager manager = make_manager(1, 3);
    FakeProgram program;
    const ActiveRequest first = start_active(manager, program, 41, make_base(41), 1);
    (void)finish_active(manager, program, first);
    const ActiveRequest second = start_active(manager, program, 42, make_base(42), 2);
    (void)finish_active(manager, program, second);

    program.required_pressure_actions = 2;
    auto inspection = manager.inspect(program, FakePreparedPrompt{43}, make_base(43), 3);
    require(inspection.choice.has_value(), "joint two-owner pressure target was unreachable");

    program.abort_start = true;
    (void)manager.reserve_materialization(program, std::move(*inspection.choice),
                                          FakePreparedPrompt{43}, {});
    std::sort(program.started_action_ids.begin(), program.started_action_ids.end());
    const std::vector<std::uint64_t> expected{
        1000U + first.sequence.id,
        1000U + second.sequence.id,
    };
    require(program.started_action_ids == expected,
            "planner did not combine preserving targets from two owners");
}

void test_combined_target_reprices_cancelled_pressure_copy() {
    FakeManager manager = make_manager(1, 3);
    FakeProgram program;
    const ActiveRequest first = start_active(manager, program, 51, make_base(51), 1);
    (void)finish_active(manager, program, first);
    const ActiveRequest second = start_active(manager, program, 52, make_base(52), 2);
    (void)finish_active(manager, program, second);

    program.required_pressure_actions             = 2;
    program.combined_target_cancels_pressure_copy = true;
    auto inspection = manager.inspect(program, FakePreparedPrompt{53}, make_base(53), 3);
    require(inspection.choice.has_value(),
            "non-monotonic complete target cost made the feasible combination unreachable");

    program.abort_start = true;
    (void)manager.reserve_materialization(program, std::move(*inspection.choice),
                                          FakePreparedPrompt{53}, {});
    std::sort(program.started_action_ids.begin(), program.started_action_ids.end());
    const std::vector<std::uint64_t> expected{
        1000U + first.sequence.id,
        1000U + second.sequence.id,
    };
    require(program.started_action_ids == expected,
            "planner accumulated parent transfer cost instead of repricing the complete target");
}

void test_in_progress_adoption_and_private_capture() {
    FakeManager manager = make_manager(1, 2);
    FakeProgram program;
    program.progress_in_progress_once = true;
    const ActiveRequest active        = start_active(manager, program, 12, make_base(12), 1);

    program.capture_assessment = FakeCaptureAssessment{
        .shortlist_key     = FakeShortlistKey{.digest = 12, .frontier = 24},
        .publishes_private = true,
    };
    program.capture_summary.endpoint = endpoint(12, 24);
    program.capture_summary.long_anchors.push_back(long_anchor(12, 16, 1));
    const auto reserved =
        manager.reserve_active_capture(program, active.lane, FakeCaptureOffer{.id = 1}, true, {});
    require(reserved == FakeManager::ActiveCaptureReserveResult::Reserved,
            "private capture was not reserved");
    auto progress = manager.progress_context_transaction(program, {});
    auto outcome  = std::get<FakeManager::ActiveCaptureOutcome>(std::move(progress));
    require(outcome.status == ContextTransactionStatus::Published &&
                !manager.context_transaction_kind(),
            "private capture was not adopted to a stable logical state");
    require(manager.lane_state(active.lane) == ninfer::runtime::LogicalLaneState::Active,
            "private capture disturbed active lane ownership");
    (void)finish_active(manager, program, active, 24);
}

void test_projected_nested_shared_candidates_use_marginal_value() {
    FakeManager manager = make_manager(1, 2, 2);
    FakeProgram program;
    FakeRequestBasePlan base = make_base(61);
    base.cache.opportunities = {
        FakeContextCache::Opportunity{
            .kind     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
            .evidence = ninfer::SharedCandidateEvidence::EngineStructural,
            .frontier = 32,
        },
        FakeContextCache::Opportunity{
            .kind     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
            .evidence = ninfer::SharedCandidateEvidence::EngineStructural,
            .frontier = 64,
        },
    };
    const ActiveRequest active = start_active(manager, program, 61, base, 1);
    require(program.selected_shared_capture_frontiers == std::vector<std::uint32_t>{64},
            "nested shared candidates were selected independently instead of by marginal value");
    (void)finish_active(manager, program, active);
}

void test_observed_shared_candidate_requires_independent_domains() {
    const auto observed_base = [](std::optional<FakeCacheSessionKey> session) {
        FakeRequestBasePlan base = make_base(71, session);
        base.cache.opportunities.push_back(FakeContextCache::Opportunity{
            .kind     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
            .evidence = ninfer::SharedCandidateEvidence::EngineObserved,
            .frontier = 64,
        });
        return base;
    };

    {
        FakeManager manager = make_manager(1, 3, 1);
        FakeProgram program;
        const ActiveRequest first = start_active(manager, program, 71, observed_base({}), 1);
        require(program.selected_shared_capture_frontiers.empty(),
                "first stateless observation was treated as independent reuse");
        (void)finish_active(manager, program, first);
        const ActiveRequest second = start_active(manager, program, 71, observed_base({}), 2);
        require(program.selected_shared_capture_frontiers == std::vector<std::uint32_t>{64},
                "two stateless observations did not establish independent reuse demand");
        (void)finish_active(manager, program, second);
    }
    {
        FakeManager manager = make_manager(1, 3, 1);
        FakeProgram program;
        const FakeCacheSessionKey session{.value = 9};
        const ActiveRequest first = start_active(manager, program, 71, observed_base(session), 1);
        (void)finish_active(manager, program, first);
        const ActiveRequest second = start_active(manager, program, 71, observed_base(session), 2);
        require(program.selected_shared_capture_frontiers.empty(),
                "same-session replay was misclassified as shared fanout demand");
        (void)finish_active(manager, program, second);
    }
}

void test_repeated_private_reuse_selects_zero_prefill_shared_promotion() {
    const auto observed_base = [] {
        FakeRequestBasePlan base = make_base(81);
        base.cache.opportunities.push_back(FakeContextCache::Opportunity{
            .kind     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
            .evidence = ninfer::SharedCandidateEvidence::EngineObserved,
            .frontier = 16,
        });
        return base;
    };

    FakeManager manager = make_manager(1, 3, 1);
    FakeProgram program;
    const ActiveRequest first = start_active(manager, program, 81, observed_base(), 1);
    require(program.selected_shared_capture_frontiers.empty(),
            "first observation selected an unsupported shared promotion");
    (void)finish_active(manager, program, first, 16);

    const ActiveRequest second = start_active(manager, program, 81, observed_base(), 2);
    require(program.selected_shared_capture_frontiers == std::vector<std::uint32_t>{16},
            "repeated private base was not selected for zero-prefill shared promotion");
    (void)finish_active(manager, program, second, 16);
}

void test_shared_capture_combines_two_pressure_owners() {
    FakeManager manager = make_manager(1, 4, 1);
    FakeProgram program;
    const ActiveRequest first = start_active(manager, program, 41, make_base(41), 1);
    (void)finish_active(manager, program, first);
    const ActiveRequest second = start_active(manager, program, 42, make_base(42), 2);
    (void)finish_active(manager, program, second);

    FakeRequestBasePlan shared_request = make_base(43);
    shared_request.cache.opportunities.push_back(FakeContextCache::Opportunity{
        .kind     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
        .evidence = ninfer::SharedCandidateEvidence::ExplicitBoundary,
        .frontier = 64,
    });
    const ActiveRequest active           = start_active(manager, program, 43, shared_request, 3);
    program.required_pressure_actions    = 2;
    program.pressure_action_immediate_ns = 0;
    program.capture_assessment           = FakeCaptureAssessment{
                  .shortlist_key          = FakeShortlistKey{.digest = 43, .frontier = 64},
                  .shared_evidence        = ninfer::SharedCandidateEvidence::ExplicitBoundary,
                  .protected_rebuild_work = PrefillWork{.tokens = 64},
                  .projected_recovery_ns  = 0,
                  .publishes_shared       = true,
                  .physically_feasible    = false,
    };

    const auto reserved =
        manager.reserve_active_capture(program, active.lane, FakeCaptureOffer{.id = 7}, 0, {});
    require(reserved == FakeManager::ActiveCaptureReserveResult::Reserved,
            "shared capture did not reserve a multi-owner pressure target");
    auto progress      = manager.progress_context_transaction(program, {});
    const auto outcome = std::get<FakeManager::ActiveCaptureOutcome>(std::move(progress));
    require(outcome.status == ContextTransactionStatus::Published &&
                program.started_action_ids.size() == 2,
            "shared capture did not publish the selected two-owner target");
    RuntimeStats stats;
    manager.populate_runtime_stats(program, stats);
    require(stats.shared_active_references == 1,
            "shared capture publication did not retain the active owner reference");
    program.required_pressure_actions = 0;
    (void)finish_active(manager, program, active);
}

void test_terminal_fallback_releases_failed_retention() {
    FakeManager manager = make_manager(1, 1);
    FakeProgram program;
    const ActiveRequest active = start_active(manager, program, 4, make_base(4), 1);
    manager.mark_terminal_pending(active.lane);
    program.finish_fail_next      = true;
    const FakeFinishResult result = manager.finish(program, active.lane, active.sequence);
    require(result.status == ConsumeStatus::Consumed &&
                result.disposition == FinishDisposition::Released,
            "failed retention did not converge to a released terminal result");
    require(result.timings.value == 7 && result.speculative.value == 9,
            "terminal fallback lost abort accounting");
    require(program.abort_calls == 1 &&
                manager.lane_state(active.lane) == ninfer::runtime::LogicalLaneState::Free &&
                manager.catalog_state(0) == FakeManager::CatalogState::Vacant,
            "terminal fallback did not free every logical owner");
}

void test_terminal_settlement_waits_for_open_resource_transaction() {
    FakeManager manager = make_manager(2, 3);
    FakeProgram program;
    const ActiveRequest active = start_active(manager, program, 31, make_base(31), 1);

    auto inspection = manager.inspect(program, FakePreparedPrompt{32}, make_base(32), 2);
    require(inspection.choice.has_value(), "concurrent materialization choice was not produced");
    require(manager.reserve_materialization(program, std::move(*inspection.choice),
                                            FakePreparedPrompt{32}, {}) ==
                FakeManager::MaterializationReserveResult::Reserved,
            "concurrent materialization was not reserved");

    const auto capture =
        manager.reserve_active_capture(program, active.lane, FakeCaptureOffer{.id = 9}, true, {});
    require(capture == FakeManager::ActiveCaptureReserveResult::Skipped &&
                program.skipped_captures == 1 &&
                manager.context_transaction_kind() ==
                    ninfer::runtime::ContextTransactionKind::Materialization,
            "optional capture was not skipped behind the open materialization");

    manager.mark_terminal_pending(active.lane);
    bool rejected = false;
    try {
        (void)manager.finish(program, active.lane, active.sequence);
    } catch (const std::logic_error&) { rejected = true; }
    require(rejected && manager.lane_state(active.lane) ==
                            ninfer::runtime::LogicalLaneState::TerminalPending,
            "terminal settlement changed topology during an open resource transaction");
}

void test_commit_and_discard_terminal_states() {
    FakeManager manager = make_manager(2, 2);
    FakeProgram program;
    const ActiveRequest first  = start_active(manager, program, 1, make_base(1), 1);
    const ActiveRequest second = start_active(manager, program, 2, make_base(2), 2);
    const std::array<LaneId, 2> lanes{first.lane, second.lane};
    FakeCommitResult commit;
    commit.row_count           = 2;
    commit.rows[0].disposition = CommitDisposition::Active;
    commit.rows[1].disposition = CommitDisposition::Finishable;
    manager.apply_commit(lanes, commit);
    require(manager.lane_state(first.lane) == ninfer::runtime::LogicalLaneState::Active &&
                manager.lane_state(second.lane) ==
                    ninfer::runtime::LogicalLaneState::TerminalPending,
            "row-aligned commit did not establish terminal pending state");
    (void)manager.finish(program, second.lane, second.sequence);

    FakeDiscardResult discard{.status = ConsumeStatus::Consumed, .row_count = 1};
    const std::array<LaneId, 1> remaining{first.lane};
    manager.apply_discard(remaining, discard);
    require(manager.lane_state(first.lane) == ninfer::runtime::LogicalLaneState::Free,
            "discard did not release cancelled active membership");
}

void test_backfill_proof_and_stats_follow_program_revision() {
    FakeManager manager = make_manager();
    FakeProgram program;
    auto inspection = manager.inspect(program, FakePreparedPrompt{8}, make_base(8), 1);
    require(inspection.choice.has_value(), "backfill test did not produce a candidate plan");
    const std::array<FakeSequenceHandle, 0> borrowers{};
    auto proof =
        manager.prove_persistent_backfill(program, make_base(99), *inspection.choice, borrowers);
    require(proof && proof->resource_revision() == program.resource_revision(),
            "Program did not seal a current-revision persistent proof");
    program.invalidate_resources();
    proof =
        manager.prove_persistent_backfill(program, make_base(99), *inspection.choice, borrowers);
    require(!proof, "resource revision change did not invalidate persistent proof");

    program.usage = FakePhysicalUsage{
        .device_state_slots      = 3,
        .host_state_slots        = 2,
        .device_main_kv_pages    = 11,
        .device_backend_kv_pages = 5,
        .host_kv_bytes           = 4096,
    };
    RuntimeStats stats;
    manager.populate_runtime_stats(program, stats);
    require(stats.device_state_occupied_slots == 3 && stats.host_state_occupied_slots == 2 &&
                stats.device_main_kv_occupied_pages == 11 &&
                stats.device_backend_kv_occupied_pages == 5 && stats.host_kv_occupied_bytes == 4096,
            "runtime physical gauges did not come directly from Program");
}

void test_shortlist_collision_requires_program_exact_verification() {
    FakeManager manager = make_manager(1, 2);
    FakeProgram program;
    const ActiveRequest seed = start_active(manager, program, 55, make_base(123), 1);
    (void)finish_active(manager, program, seed);

    auto collision = manager.inspect(program, FakePreparedPrompt{99}, make_base(55), 2);
    require(collision.choice && collision.choice->summary().reusable_prompt_tokens == 0,
            "shortlist collision bypassed Program exact identity verification");
}

} // namespace

int main() {
    run_test("private checkpoint identity loss",
             test_private_portfolio_loss_keeps_checkpoint_identity_fixed);
    run_test("portfolio demand and owner aggregation", test_portfolio_demand_and_owner_aggregation);
    run_test("shared capture private transition loss",
             test_shared_capture_subtracts_private_transition_loss);
    run_test("equal lower-bound tie-break",
             test_equal_lower_bound_does_not_short_circuit_tie_break);
    run_test("feasible identity pressure improvement",
             test_feasible_identity_expands_when_pressure_can_remove_copy);
    run_test("dominating identity fast path",
             test_dominating_identity_does_not_build_pressure_graph);
    run_test("root lifecycle and prefix reuse", test_root_lifecycle_and_prefix_reuse);
    run_test("stale revision is retryable", test_stale_revision_is_retryable);
    run_test("materialization abort preserves source", test_materialization_abort_preserves_source);
    run_test("committed victim survives abort", test_committed_victim_survives_transaction_abort);
    run_test("uncommitted pressure acknowledgement",
             test_uncommitted_pressure_acknowledgement_is_not_degradation);
    run_test("aborted source is not a hit",
             test_aborted_source_selection_does_not_create_hit_history);
    run_test("retained source protection", test_retained_source_is_protected_until_terminal);
    run_test("session publication order", test_session_publication_order_controls_tied_source);
    run_test("canonical pressure", test_canonical_pressure_starts_with_disposable_owner);
    run_test("all preserving pressure alternatives",
             test_pressure_tries_every_preserving_alternative_before_eviction);
    run_test("cumulative owner target",
             test_cumulative_owner_target_closes_pressure_without_eviction);
    run_test("joint two-owner pressure", test_two_owners_jointly_close_pressure);
    run_test("combined target exact repricing",
             test_combined_target_reprices_cancelled_pressure_copy);
    run_test("in-progress and capture", test_in_progress_adoption_and_private_capture);
    run_test("projected shared marginal value",
             test_projected_nested_shared_candidates_use_marginal_value);
    run_test("observed shared independent domains",
             test_observed_shared_candidate_requires_independent_domains);
    run_test("zero-prefill private promotion",
             test_repeated_private_reuse_selects_zero_prefill_shared_promotion);
    run_test("shared capture multi-owner pressure",
             test_shared_capture_combines_two_pressure_owners);
    run_test("terminal fallback", test_terminal_fallback_releases_failed_retention);
    run_test("terminal waits for resource transaction",
             test_terminal_settlement_waits_for_open_resource_transaction);
    run_test("commit and discard", test_commit_and_discard_terminal_states);
    run_test("backfill proof and stats", test_backfill_proof_and_stats_follow_program_revision);
    run_test("shortlist exact verification",
             test_shortlist_collision_requires_program_exact_verification);
    if (failures != 0) { return 1; }
    std::cout << "ok\n";
    return 0;
}
