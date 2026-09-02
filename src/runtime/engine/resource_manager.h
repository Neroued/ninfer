#pragma once

#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include "runtime/engine/context_cost.h"
#include "runtime/engine/materialization_planner.h"
#include "runtime/engine/shared_capture_planner.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer::runtime {

inline constexpr std::uint32_t kInvalidCatalogSlot = std::numeric_limits<std::uint32_t>::max();

enum class LogicalLaneState : std::uint8_t {
    Free,
    Materializing,
    Active,
    TerminalPending,
};

struct RetentionObservation {
    RetentionClass retention_class   = RetentionClass::RecentPrivate;
    std::uint64_t selected_hit_count = 0;
    std::uint64_t last_hit_epoch     = 0;
};

struct PolicyObservationKey {
    bool shared            = false;
    std::uint32_t slot     = kInvalidCatalogSlot;
    std::uint64_t owner_id = 0;
    std::uint64_t revision = 0;
    CheckpointRef checkpoint;

    [[nodiscard]] friend constexpr bool operator==(const PolicyObservationKey&,
                                                   const PolicyObservationKey&) noexcept = default;
};

struct CheckpointObservation {
    CheckpointRef checkpoint;
    RetentionObservation observation;
};

// ResourceManager owns logical policy only.  Every physical feasibility decision and mutation is
// represented by an opaque Package::ResourcePlan sealed against Program::resource_revision().
template <class Package>
class ResourceManager {
public:
    using Program                 = typename Package::Program;
    using PreparedPrompt          = typename Package::PreparedPrompt;
    using RequestBasePlan         = typename Package::RequestBasePlan;
    using AdmissionCandidate      = typename Package::AdmissionCandidate;
    using ResourcePlan            = typename Package::ResourcePlan;
    using PersistentBackfillProof = typename Package::PersistentBackfillProof;
    using SequenceHandle          = typename Package::SequenceHandle;
    using ContinuationHandle      = typename Package::ContinuationHandle;
    using SharedPrefixHandle      = typename Package::SharedPrefixHandle;
    using CaptureOffer            = typename Package::CaptureOffer;
    using ContinuationSummary     = typename Package::ContinuationSummary;
    using SharedPrefixSummary     = typename Package::SharedPrefixSummary;
    using PrefixShortlistKey =
        std::remove_cvref_t<decltype(std::declval<SharedPrefixSummary>().checkpoint.shortlist_key)>;
    using CaptureAssessment                 = typename Package::CaptureAssessment;
    using ProgramActiveCaptureResult        = typename Package::ActiveCaptureResult;
    using CacheSessionKey                   = typename Package::CacheSessionKey;
    using ProgramContextTransactionProgress = typename Package::ContextTransactionProgress;
    using ProgramMaterializationResult      = typename Package::MaterializationResult;
    using StartResult                       = typename Package::StartResult;
    using FinishResult                      = typename Package::FinishResult;
    using AbortResult                       = typename Package::AbortResult;
    using Planner                           = MaterializationPlanner<Package>;
    using CapturePlanner                    = SharedCapturePlanner<Package>;

private:
    // A transaction capability is a point-in-time structural snapshot. An active edge is a
    // durable logical lease on the owner and deliberately does not freeze that snapshot's
    // generation: another reader may change replica residency while the same owner remains live.
    struct ActiveOwnerEdge {
        LogicalOwnerKey owner;
        std::uint32_t slot = kInvalidCatalogSlot;

        [[nodiscard]] friend constexpr bool operator==(ActiveOwnerEdge,
                                                       ActiveOwnerEdge) noexcept = default;
    };

    struct OwnerClaim {
        PlanningOwnerId planning_id;
        CatalogCapability capability;
        VictimDisposition disposition = VictimDisposition::Retained;
        std::vector<CheckpointRef> dropped_checkpoints;
    };

    struct PlanningOwnerRecord {
        PlanningOwnerId id;
        CatalogCapability capability;
    };

public:
    struct ReuseDomainId {
        std::uint64_t low  = 0;
        std::uint64_t high = 0;

        [[nodiscard]] friend constexpr bool operator==(ReuseDomainId,
                                                       ReuseDomainId) noexcept = default;
    };

    struct PrefixDemandRecord {
        ReuseDomainId domain;
        std::vector<PrefixShortlistKey> candidate_keys;
        std::vector<PrefixShortlistKey> exact_resident_keys;
        std::optional<PrefixShortlistKey> selected_source_key;
    };

    enum class CatalogState : std::uint8_t {
        Vacant,
        Catalogued,
        Claimed,
        ReservedForActive,
    };

    enum class SharedCatalogState : std::uint8_t {
        Vacant,
        Catalogued,
        Claimed,
        ReservedCapture,
    };

    class Choice {
    public:
        Choice(Choice&&) noexcept        = default;
        Choice& operator=(Choice&&)      = delete;
        Choice(const Choice&)            = delete;
        Choice& operator=(const Choice&) = delete;

        [[nodiscard]] const RequestPlanSummary& summary() const noexcept {
            return plan_->summary();
        }

        [[nodiscard]] LaneId destination() const noexcept { return destination_; }

        [[nodiscard]] bool needs_transfer() const noexcept { return plan_->needs_transfer(); }

        [[nodiscard]] ProgramResourceRevision resource_revision() const noexcept {
            return plan_->resource_revision();
        }

    private:
        Choice(LaneId destination, ResourcePlan&& plan, std::uint32_t catalog_capacity,
               std::optional<CacheSessionKey> session, RetentionClass retention,
               bool update_session_index, std::uint64_t publication_order)
            : destination_(destination), plan_(std::move(plan)), session_(std::move(session)),
              retention_(retention), update_session_index_(update_session_index),
              publication_order_(publication_order) {
            private_claims_.reserve(catalog_capacity);
            shared_claims_.reserve(catalog_capacity);
        }

        LaneId destination_{};
        std::optional<ResourcePlan> plan_;
        PrivateSourceMode source_mode_ = PrivateSourceMode::ConsumeToActive;
        std::optional<CatalogCapability> private_source_;
        std::optional<CatalogCapability> shared_source_;
        std::uint32_t publication_slot_ = kInvalidCatalogSlot;
        std::vector<OwnerClaim> private_claims_;
        std::vector<OwnerClaim> shared_claims_;
        std::optional<PolicyObservationKey> selected_observation_;
        std::optional<CacheSessionKey> session_;
        RetentionClass retention_        = RetentionClass::RecentPrivate;
        bool update_session_index_       = true;
        std::uint64_t publication_order_ = 0;
        MaterializationDiagnostics diagnostics_;
        PrefixDemandRecord demand_;

        friend class ResourceManager;
    };

    class PublishedActivation {
    public:
        PublishedActivation(PublishedActivation&& other) noexcept
            : owner_(std::exchange(other.owner_, nullptr)), result_(std::move(other.result_)),
              destination_(other.destination_) {}

        PublishedActivation& operator=(PublishedActivation&&)      = delete;
        PublishedActivation(const PublishedActivation&)            = delete;
        PublishedActivation& operator=(const PublishedActivation&) = delete;

        [[nodiscard]] const SequenceHandle& sequence() const {
            if (!result_) { throw std::logic_error("published activation is empty"); }
            return result_->sequence;
        }

    private:
        PublishedActivation(ResourceManager& owner, StartResult&& result, LaneId destination)
            : owner_(&owner), result_(std::move(result)), destination_(destination) {}

        ResourceManager* owner_ = nullptr;
        std::optional<StartResult> result_;
        LaneId destination_{};

        friend class ResourceManager;
    };

    struct MaterializationOutcome {
        ContextTransactionStatus status = ContextTransactionStatus::Aborted;
        std::optional<PublishedActivation> activation;
        MaterializationDiagnostics diagnostics;
    };

    enum class MaterializationReserveResult : std::uint8_t {
        Reserved,
        Stale,
        Aborted,
    };

    enum class ActiveCaptureReserveResult : std::uint8_t {
        Reserved,
        Skipped,
    };

    struct ActiveCaptureOutcome {
        ContextTransactionStatus status = ContextTransactionStatus::Aborted;
    };

    using ContextTransactionOutcome =
        std::variant<ContextTransactionInProgress, MaterializationOutcome, ActiveCaptureOutcome>;

    struct Inspection {
        Readiness readiness = Readiness::TemporarilyBlocked;
        std::optional<Choice> choice;
    };

    ResourceManager(std::uint32_t lane_count, std::uint32_t private_catalog_capacity,
                    std::uint32_t shared_catalog_capacity, bool cache_enabled,
                    std::uint32_t max_long_anchors, ContextMachineCostModel cost_model)
        : lane_count_(lane_count), catalog_count_(private_catalog_capacity),
          shared_catalog_count_(shared_catalog_capacity), cache_enabled_(cache_enabled),
          catalog_(private_catalog_capacity), shared_catalog_(shared_catalog_capacity),
          session_index_(private_catalog_capacity),
          prefix_index_(checked_prefix_index_capacity(private_catalog_capacity,
                                                      shared_catalog_capacity, max_long_anchors)),
          max_long_anchors_(max_long_anchors), cost_model_(std::move(cost_model)) {
        if (lane_count == 0 || lane_count > kMaximumConcurrency ||
            private_catalog_capacity < lane_count) {
            throw std::invalid_argument("logical resource-manager bounds are invalid");
        }
        const std::size_t observation_capacity = 3U + max_long_anchors_;
        observation_scratch_.reserve(observation_capacity);
        demand_window_.reserve(kDemandWindowCapacity);
        for (CatalogEntry& entry : catalog_) {
            entry.summary.long_anchors.reserve(max_long_anchors_);
            entry.observations.reserve(observation_capacity);
        }
        for (std::uint32_t lane = 0; lane < lane_count_; ++lane) {
            active_[lane].shared_sources.reserve(shared_catalog_capacity);
        }
    }

    [[nodiscard]] Inspection inspect(Program& program, const PreparedPrompt& prompt,
                                     const RequestBasePlan& base, std::uint64_t publication_order) {
        if (!std::holds_alternative<std::monostate>(transaction_) ||
            program.has_context_transaction()) {
            return {.readiness = Readiness::TemporarilyBlocked};
        }
        if (publication_order == 0) {
            throw std::invalid_argument("request publication order is zero");
        }
        if (!program.isolated_request_feasible(base)) {
            return {.readiness = Readiness::PermanentlyInfeasible};
        }
        std::optional<LaneId> destination;
        for (std::uint32_t lane = 0; lane < lane_count_; ++lane) {
            if (lanes_[lane] == LogicalLaneState::Free) {
                destination = LaneId{lane};
                break;
            }
        }
        if (!destination) { return {.readiness = Readiness::TemporarilyBlocked}; }

        const typename Planner::Clock::time_point planning_started = Planner::Clock::now();
        rebuild_prefix_index();
        PrefixDemandRecord provisional_demand;
        provisional_demand.domain =
            reuse_domain(base.context_cache().session_key, publication_order);
        provisional_demand.candidate_keys.reserve(base.context_cache().opportunities.size());
        provisional_demand.exact_resident_keys.reserve(prefix_index_.size());
        if (cache_enabled_) {
            for (const auto& opportunity : base.context_cache().opportunities) {
                if (opportunity.kind != PromptCacheMarkerKind::SharedStablePrefix) { continue; }
                const std::optional<PrefixShortlistKey> key =
                    base.prefix_shortlist_key(opportunity.frontier);
                if (key) { append_unique(provisional_demand.candidate_keys, *key); }
            }
        }
        std::optional<std::size_t> current_session_cell;
        if (cache_enabled_ && base.context_cache().session_key &&
            base.context_cache().update_session_index) {
            current_session_cell = find_session_cell(*base.context_cache().session_key);
        }
        std::vector<Candidate> candidates;
        candidates.reserve(1U + prefix_index_.size());
        std::optional<AdmissionCandidate> root = program.inspect_admission(
            prompt, base, *destination, nullptr, nullptr, std::nullopt, false);
        if (!root) { throw std::logic_error("Program rejected isolated root planning"); }
        candidates.push_back(Candidate{.plan = std::move(*root)});

        if (cache_enabled_) {
            for (const PrefixIndexEntry& index : prefix_index_) {
                if (!valid_prefix_index_entry(index)) { continue; }
                const std::optional<PrefixShortlistKey> incoming =
                    base.prefix_shortlist_key(index.key.frontier);
                if (!incoming || *incoming != index.key) { continue; }

                if (!index.shared) {
                    const CatalogEntry& entry = catalog_[index.slot];
                    if (entry.state != CatalogState::Catalogued || !entry.handle ||
                        private_has_active_edge(index.slot)) {
                        continue;
                    }
                    const bool retain =
                        entry.session && (!base.context_cache().session_key ||
                                          *entry.session != *base.context_cache().session_key ||
                                          !base.context_cache().update_session_index);
                    std::optional<AdmissionCandidate> plan =
                        program.inspect_admission(prompt, base, *destination, &*entry.handle,
                                                  nullptr, index.checkpoint, retain);
                    if (!plan) { continue; }
                    if (plan->summary().reusable_prompt_tokens == 0 ||
                        (retain &&
                         plan->identity_assessment().source_mode != PrivateSourceMode::Retain)) {
                        throw std::logic_error("Program returned an invalid private candidate");
                    }
                    const bool current_session_binding =
                        current_session_cell &&
                        session_index_[*current_session_cell].slot == index.slot &&
                        session_index_[*current_session_cell].owner_id == entry.id &&
                        session_index_[*current_session_cell].revision == entry.revision;
                    append_unique(provisional_demand.exact_resident_keys, index.key);
                    candidates.push_back(Candidate{
                        .plan                    = std::move(*plan),
                        .current_session_binding = current_session_binding,
                        .private_source          = private_capability(index.slot),
                        .selected_observation =
                            PolicyObservationKey{
                                .shared     = false,
                                .slot       = index.slot,
                                .owner_id   = entry.id,
                                .revision   = entry.revision,
                                .checkpoint = index.checkpoint,
                            },
                        .source_key = index.key,
                    });
                    continue;
                }

                const SharedCatalogEntry& entry = shared_catalog_[index.slot];
                if (entry.state != SharedCatalogState::Catalogued || !entry.handle) { continue; }
                std::optional<AdmissionCandidate> plan = program.inspect_admission(
                    prompt, base, *destination, nullptr, &*entry.handle, index.checkpoint, false);
                if (!plan) { continue; }
                if (plan->summary().reusable_prompt_tokens == 0 ||
                    plan->identity_assessment().source_mode != PrivateSourceMode::Retain) {
                    throw std::logic_error("Program returned an invalid shared candidate");
                }
                append_unique(provisional_demand.exact_resident_keys, index.key);
                candidates.push_back(Candidate{
                    .plan          = std::move(*plan),
                    .shared_source = shared_capability(index.slot),
                    .selected_observation =
                        PolicyObservationKey{
                            .shared     = true,
                            .slot       = index.slot,
                            .owner_id   = entry.id,
                            .revision   = entry.revision,
                            .checkpoint = index.checkpoint,
                        },
                    .source_key = index.key,
                });
            }
        }

        std::optional<Choice> selected =
            plan_materialization(program, prompt, base, *destination, candidates, publication_order,
                                 planning_started, provisional_demand);
        if (!selected) { return {.readiness = Readiness::TemporarilyBlocked}; }
        return {
            .readiness = selected->needs_transfer() ? Readiness::NeedsTransfer : Readiness::Ready,
            .choice    = std::move(selected),
        };
    }

    [[nodiscard]] std::optional<PersistentBackfillProof>
    prove_persistent_backfill(Program& program, const RequestBasePlan& blocked_head,
                              const Choice& candidate,
                              std::span<const SequenceHandle> persistent_borrowers) const {
        if (!candidate.plan_ || candidate.resource_revision() != program.resource_revision()) {
            return std::nullopt;
        }
        return program.prove_persistent_backfill(blocked_head, *candidate.plan_,
                                                 persistent_borrowers);
    }

    [[nodiscard]] MaterializationReserveResult
    reserve_materialization(Program& program, Choice&& choice, PreparedPrompt&& prompt,
                            CancellationFlagView cancellation) {
        if (!std::holds_alternative<std::monostate>(transaction_) ||
            program.has_context_transaction()) {
            throw std::logic_error("ResourceManager already owns a resource transaction");
        }
        const ProgramResourceRevision resource_revision = program.resource_revision();
        if (!choice.plan_ || resource_revision.value == 0) {
            throw std::logic_error("resource choice is malformed");
        }
        if (choice.plan_->resource_revision() != resource_revision) {
            return MaterializationReserveResult::Stale;
        }
        validate_choice(choice, resource_revision);
        MaterializationRecord record = take_materialization_record(choice);
        transaction_.template emplace<MaterializationRecord>(std::move(record));
        MaterializationRecord& open = std::get<MaterializationRecord>(transaction_);
        reserve_logical_materialization(open);

        const ContextTransactionReserveStatus status = program.start_resource_transaction(
            std::move(*choice.plan_), std::move(prompt), cancellation);
        choice.plan_.reset();
        if (status == ContextTransactionReserveStatus::Aborted) {
            rollback_logical_materialization(open);
            transaction_.template emplace<std::monostate>();
            return cancellation.requested() ? MaterializationReserveResult::Aborted
                                            : MaterializationReserveResult::Stale;
        }
        observe_planner_diagnostics(open.diagnostics);
        return MaterializationReserveResult::Reserved;
    }

    [[nodiscard]] std::optional<ContextTransactionKind> context_transaction_kind() const noexcept {
        if (std::holds_alternative<MaterializationRecord>(transaction_)) {
            return ContextTransactionKind::Materialization;
        }
        if (std::holds_alternative<ActiveCaptureRecord>(transaction_)) {
            return ContextTransactionKind::ActiveCapture;
        }
        return std::nullopt;
    }

    [[nodiscard]] ContextTransactionOutcome
    progress_context_transaction(Program& program, CancellationFlagView cancellation) {
        if (std::holds_alternative<std::monostate>(transaction_) ||
            !program.has_context_transaction()) {
            throw std::logic_error("ResourceManager has no progressable resource transaction");
        }
        ProgramContextTransactionProgress progress =
            program.progress_context_transaction(cancellation);
        return std::visit(
            [&](auto&& result) -> ContextTransactionOutcome {
                using Result = std::decay_t<decltype(result)>;
                if constexpr (std::is_same_v<Result, ContextTransactionInProgress>) {
                    return ContextTransactionInProgress{};
                } else if constexpr (std::is_same_v<Result, ProgramMaterializationResult>) {
                    return adopt_materialization_progress(program, std::move(result));
                } else if constexpr (std::is_same_v<Result, ProgramActiveCaptureResult>) {
                    return adopt_active_capture_progress(program, std::move(result));
                } else {
                    throw std::logic_error("Program returned an unsupported resource operation");
                }
            },
            std::move(progress));
    }

    void adopt(Program& program, PublishedActivation&& activation) noexcept {
        if (activation.owner_ != this || !activation.result_ ||
            activation.destination_.value >= lane_count_ ||
            lanes_[activation.destination_.value] != LogicalLaneState::Materializing ||
            !active_[activation.destination_.value].occupied ||
            !std::holds_alternative<MaterializationRecord>(transaction_) ||
            !program.has_context_transaction()) {
            std::terminate();
        }
        lanes_[activation.destination_.value] = LogicalLaneState::Active;
        activation.result_.reset();
        activation.owner_ = nullptr;
        transaction_.template emplace<std::monostate>();
        program.finalize_context_transaction();
    }

    [[nodiscard]] ActiveCaptureReserveResult
    reserve_active_capture(Program& program, LaneId lane, CaptureOffer&& offer,
                           std::uint32_t blocked_runnable_requests,
                           CancellationFlagView cancellation) {
        require_lane(lane, LogicalLaneState::Active);
        const bool manager_transaction = !std::holds_alternative<std::monostate>(transaction_);
        const bool program_transaction = program.has_context_transaction();
        if (manager_transaction != program_transaction) {
            throw std::logic_error("capture observes inconsistent transaction ownership");
        }
        if (program_transaction) {
            program.skip_capture(std::move(offer));
            return ActiveCaptureReserveResult::Skipped;
        }
        rebuild_prefix_index();

        CaptureAssessment private_baseline =
            program.inspect_capture(offer, nullptr, nullptr, std::nullopt, false);
        std::optional<CheckpointRef> private_replacement;
        if (!private_baseline.private_replacement_candidates.empty()) {
            private_replacement =
                *std::min_element(private_baseline.private_replacement_candidates.begin(),
                                  private_baseline.private_replacement_candidates.end(),
                                  [](CheckpointRef lhs, CheckpointRef rhs) {
                                      return std::tuple{lhs.kind, lhs.frontier, lhs.ordinal} <
                                             std::tuple{rhs.kind, rhs.frontier, rhs.ordinal};
                                  });
            private_baseline =
                program.inspect_capture(offer, nullptr, nullptr, private_replacement, false);
        }

        CaptureAssessment candidate =
            program.inspect_capture(offer, nullptr, nullptr, private_replacement, true);
        const SharedPrefixHandle* exact_shared = nullptr;
        if (candidate.publishes_shared) {
            for (const PrefixIndexEntry& index : prefix_index_) {
                if (!index.shared || !valid_prefix_index_entry(index) ||
                    index.key != candidate.shortlist_key) {
                    continue;
                }
                SharedCatalogEntry& entry = shared_catalog_[index.slot];
                if (program.shared_capture_matches(offer, *entry.handle)) {
                    exact_shared = &*entry.handle;
                    break;
                }
            }
        }
        if (exact_shared != nullptr) {
            if (!private_baseline.publishes_private || !private_baseline.physically_feasible) {
                program.skip_capture(std::move(offer));
                return ActiveCaptureReserveResult::Skipped;
            }
            transaction_.template emplace<ActiveCaptureRecord>(ActiveCaptureRecord{
                .lane              = lane,
                .publishes_private = true,
            });
            const ContextTransactionReserveStatus reserved = program.reserve_active_capture(
                std::move(offer), exact_shared, nullptr, private_replacement, false, cancellation);
            if (reserved == ContextTransactionReserveStatus::Aborted) {
                transaction_.template emplace<std::monostate>();
                return ActiveCaptureReserveResult::Skipped;
            }
            return ActiveCaptureReserveResult::Reserved;
        }

        struct CaptureScenario {
            CaptureAssessment assessment;
            std::uint32_t publication_slot        = kInvalidCatalogSlot;
            const SharedPrefixHandle* replacement = nullptr;
            std::uint64_t replacement_id          = 0;
            std::uint64_t replacement_revision    = 0;
            std::uint32_t stable_ordinal          = 0;
        };

        struct SelectedCapture {
            CaptureScenario scenario;
            typename CapturePlanner::Result plan;
        };

        std::optional<SelectedCapture> selected;
        std::vector<PlanningOwnerRecord> capture_owner_records;
        if (candidate.publishes_shared) {
            const bool pressure_evidence =
                has_shared_candidate_evidence(candidate.shared_evidence,
                                              SharedCandidateEvidence::ExplicitBoundary) ||
                has_shared_candidate_evidence(candidate.shared_evidence,
                                              SharedCandidateEvidence::RequestedAutomatic) ||
                matching_reuse_domains(candidate.shortlist_key) >= 2U;

            std::vector<CaptureScenario> scenarios;
            scenarios.reserve(static_cast<std::size_t>(shared_catalog_count_) + 1U);
            for (std::uint32_t slot = 0; slot < shared_catalog_count_; ++slot) {
                if (shared_catalog_[slot].state != SharedCatalogState::Vacant) { continue; }
                scenarios.push_back(CaptureScenario{
                    .assessment       = candidate,
                    .publication_slot = slot,
                    .stable_ordinal   = 0,
                });
                break;
            }
            if (pressure_evidence) {
                for (std::uint32_t slot = 0; slot < shared_catalog_count_; ++slot) {
                    SharedCatalogEntry& entry = shared_catalog_[slot];
                    if (entry.state != SharedCatalogState::Catalogued || !entry.handle ||
                        entry.transaction_pins != 0 || shared_active_edge_count(slot) != 0) {
                        continue;
                    }
                    CaptureAssessment assessment = program.inspect_capture(
                        offer, nullptr, &*entry.handle, private_replacement, true);
                    if (!assessment.publishes_shared) { continue; }
                    scenarios.push_back(CaptureScenario{
                        .assessment           = std::move(assessment),
                        .publication_slot     = slot,
                        .replacement          = &*entry.handle,
                        .replacement_id       = entry.id,
                        .replacement_revision = entry.revision,
                        .stable_ordinal       = 1U + slot,
                    });
                }
            }

            std::vector<typename CapturePlanner::OwnerPolicy> owner_policies;
            std::vector<typename CapturePlanner::CheckpointPolicy> checkpoint_policies;
            owner_policies.reserve(catalog_count_ + shared_catalog_count_);
            checkpoint_policies.reserve(prefix_index_.size());
            capture_owner_records.reserve(catalog_count_ + shared_catalog_count_);
            const auto append_private_checkpoint = [&](PlanningOwnerId owner, std::uint32_t slot,
                                                       const auto& checkpoint) {
                const CatalogEntry& entry = catalog_[slot];
                checkpoint_policies.push_back(typename CapturePlanner::CheckpointPolicy{
                    .owner                = owner,
                    .checkpoint           = checkpoint.ref,
                    .demand_mask          = committed_demand_mask_for(checkpoint.shortlist_key),
                    .rebuild_ns           = cost_model_.prefill_ns(checkpoint.rebuild_work),
                    .baseline_recovery_ns = price_checkpoint_recovery_work(
                        cost_model_,
                        program.checkpoint_recovery_work(*entry.handle, checkpoint.ref)),
                });
            };
            for (std::uint32_t slot = 0; slot < catalog_count_; ++slot) {
                const CatalogEntry& entry = catalog_[slot];
                if (entry.state != CatalogState::Catalogued || !entry.handle ||
                    private_has_active_edge(slot)) {
                    continue;
                }
                const PlanningOwnerId owner{
                    .value = static_cast<std::uint32_t>(capture_owner_records.size())};
                capture_owner_records.push_back(PlanningOwnerRecord{
                    .id = owner,
                    .capability =
                        CatalogCapability{
                            .owner =
                                LogicalOwnerKey{
                                    .kind = LogicalOwnerKind::PrivateContinuation,
                                    .id   = entry.id,
                                },
                            .slot       = slot,
                            .generation = entry.revision,
                        },
                });
                owner_policies.push_back(typename CapturePlanner::OwnerPolicy{
                    .owner                    = owner,
                    .private_retention_weight = private_retention_weight(entry.retention),
                });
                if (entry.summary.endpoint) {
                    append_private_checkpoint(owner, slot, *entry.summary.endpoint);
                }
                if (entry.summary.rewrite) {
                    append_private_checkpoint(owner, slot, *entry.summary.rewrite);
                }
                for (const auto& checkpoint : entry.summary.long_anchors) {
                    append_private_checkpoint(owner, slot, checkpoint);
                }
            }
            for (std::uint32_t slot = 0; slot < shared_catalog_count_; ++slot) {
                const SharedCatalogEntry& entry = shared_catalog_[slot];
                if (entry.state != SharedCatalogState::Catalogued || !entry.handle) { continue; }
                const PlanningOwnerId owner{
                    .value = static_cast<std::uint32_t>(capture_owner_records.size())};
                capture_owner_records.push_back(PlanningOwnerRecord{
                    .id = owner,
                    .capability =
                        CatalogCapability{
                            .owner =
                                LogicalOwnerKey{
                                    .kind = LogicalOwnerKind::SharedPrefix,
                                    .id   = entry.id,
                                },
                            .slot       = slot,
                            .generation = entry.revision,
                        },
                });
                owner_policies.push_back(typename CapturePlanner::OwnerPolicy{
                    .owner                    = owner,
                    .private_retention_weight = 0,
                    .explicit_shared_credit   = entry.explicit_credit,
                });
                checkpoint_policies.push_back(typename CapturePlanner::CheckpointPolicy{
                    .owner      = owner,
                    .checkpoint = entry.summary.checkpoint.ref,
                    .demand_mask =
                        committed_demand_mask_for(entry.summary.checkpoint.shortlist_key),
                    .rebuild_ns = cost_model_.prefill_ns(entry.summary.checkpoint.rebuild_work),
                    .baseline_recovery_ns = price_checkpoint_recovery_work(
                        cost_model_, program.checkpoint_recovery_work(
                                         *entry.handle, entry.summary.checkpoint.ref)),
                });
            }

            const std::uint32_t scenario_budget =
                scenarios.empty()
                    ? 0
                    : std::max<std::uint32_t>(1U, CapturePlanner::kTargetBudget /
                                                      static_cast<std::uint32_t>(scenarios.size()));
            for (CaptureScenario& scenario : scenarios) {
                std::vector<const ContinuationHandle*> private_owners;
                std::vector<PlanningOwnerId> private_owner_ids;
                std::vector<const SharedPrefixHandle*> shared_owners;
                std::vector<PlanningOwnerId> shared_owner_ids;
                const auto owner_id_for = [&](LogicalOwnerKind kind,
                                              std::uint32_t slot) -> PlanningOwnerId {
                    const auto found =
                        std::find_if(capture_owner_records.begin(), capture_owner_records.end(),
                                     [&](const auto& record) {
                                         return record.capability.owner.kind == kind &&
                                                record.capability.slot == slot;
                                     });
                    if (found == capture_owner_records.end()) {
                        throw std::logic_error("capture owner has no planning ID");
                    }
                    return found->id;
                };
                if (pressure_evidence) {
                    private_owners.reserve(catalog_count_);
                    private_owner_ids.reserve(catalog_count_);
                    shared_owners.reserve(shared_catalog_count_);
                    shared_owner_ids.reserve(shared_catalog_count_);
                    for (std::uint32_t slot = 0; slot < catalog_count_; ++slot) {
                        const CatalogEntry& entry = catalog_[slot];
                        if (entry.state != CatalogState::Catalogued || !entry.handle ||
                            private_has_active_edge(slot)) {
                            continue;
                        }
                        private_owners.push_back(&*entry.handle);
                        private_owner_ids.push_back(
                            owner_id_for(LogicalOwnerKind::PrivateContinuation, slot));
                    }
                    for (std::uint32_t slot = 0; slot < shared_catalog_count_; ++slot) {
                        const SharedCatalogEntry& entry = shared_catalog_[slot];
                        if (entry.state != SharedCatalogState::Catalogued || !entry.handle ||
                            entry.transaction_pins != 0 || shared_active_edge_count(slot) != 0 ||
                            entry.id == scenario.replacement_id) {
                            continue;
                        }
                        shared_owners.push_back(&*entry.handle);
                        shared_owner_ids.push_back(
                            owner_id_for(LogicalOwnerKind::SharedPrefix, slot));
                    }
                }
                const typename CapturePlanner::Input input{
                    .capture             = &scenario.assessment,
                    .private_owners      = private_owners,
                    .private_owner_ids   = private_owner_ids,
                    .shared_owners       = shared_owners,
                    .shared_owner_ids    = shared_owner_ids,
                    .owner_policies      = owner_policies,
                    .checkpoint_policies = checkpoint_policies,
                    .direct_shared_victim =
                        scenario.replacement == nullptr
                            ? std::nullopt
                            : std::optional<PlanningOwnerId>(owner_id_for(
                                  LogicalOwnerKind::SharedPrefix, scenario.publication_slot)),
                    .candidate_demand_mask =
                        committed_demand_mask_for(scenario.assessment.shortlist_key),
                    .candidate_rebuild_ns =
                        cost_model_.prefill_ns(scenario.assessment.protected_rebuild_work),
                    .private_baseline_immediate_ns = price_context_transfer_requirements(
                        cost_model_, private_baseline.transfer_requirements),
                    .blocked_runnable_requests = blocked_runnable_requests,
                    .stable_scenario_ordinal   = scenario.stable_ordinal,
                    .target_budget             = scenario_budget,
                };
                std::optional<typename CapturePlanner::Result> planned =
                    capture_planner_.plan(program, cost_model_, input);
                if (!planned) { continue; }
                const bool better =
                    !selected || planned->net_gain > selected->plan.net_gain ||
                    (planned->net_gain == selected->plan.net_gain &&
                     std::tie(planned->stable_scenario_ordinal, planned->stable_target_ordinal) <
                         std::tie(selected->plan.stable_scenario_ordinal,
                                  selected->plan.stable_target_ordinal));
                if (better) {
                    selected.emplace(SelectedCapture{
                        .scenario = std::move(scenario),
                        .plan     = std::move(*planned),
                    });
                }
            }
        }

        if (!selected) {
            if (!private_baseline.publishes_private || !private_baseline.physically_feasible) {
                program.skip_capture(std::move(offer));
                return ActiveCaptureReserveResult::Skipped;
            }
            transaction_.template emplace<ActiveCaptureRecord>(ActiveCaptureRecord{
                .lane              = lane,
                .publishes_private = true,
            });
            const ContextTransactionReserveStatus reserved = program.reserve_active_capture(
                std::move(offer), nullptr, nullptr, private_replacement, false, cancellation);
            if (reserved == ContextTransactionReserveStatus::Aborted) {
                transaction_.template emplace<std::monostate>();
                return ActiveCaptureReserveResult::Skipped;
            }
            return ActiveCaptureReserveResult::Reserved;
        }

        ActiveCaptureRecord record{
            .lane                 = lane,
            .publishes_private    = selected->scenario.assessment.publishes_private,
            .publishes_shared     = true,
            .publication_slot     = selected->scenario.publication_slot,
            .replacement_id       = selected->scenario.replacement_id,
            .replacement_revision = selected->scenario.replacement_revision,
            .shared_evidence      = selected->scenario.assessment.shared_evidence,
        };
        for (const PressureOwnerOutcome& outcome : selected->plan.owner_outcomes) {
            const auto owner_record =
                std::find_if(capture_owner_records.begin(), capture_owner_records.end(),
                             [&](const PlanningOwnerRecord& candidate) {
                                 return candidate.id == outcome.owner;
                             });
            if (owner_record == capture_owner_records.end()) {
                throw std::logic_error("shared capture pressure owner ID is invalid");
            }
            const bool shared =
                owner_record->capability.owner.kind == LogicalOwnerKind::SharedPrefix;
            const std::uint32_t slot = owner_record->capability.slot;
            if (!shared) {
                if (slot >= catalog_count_) {
                    throw std::logic_error("shared capture private pressure owner is invalid");
                }
                const CatalogEntry& entry = catalog_[slot];
                if (entry.state != CatalogState::Catalogued || !entry.handle ||
                    entry.id != owner_record->capability.owner.id ||
                    entry.revision != owner_record->capability.generation ||
                    private_has_active_edge(slot)) {
                    throw std::logic_error("shared capture private pressure owner is stale");
                }
                std::vector<CheckpointRef> dropped = selected_checkpoint_drops(
                    outcome.owner, outcome.disposition, outcome.dropped_checkpoints,
                    selected->plan.checkpoint_outcomes,
                    continuation_checkpoint_count(entry.summary), [&](CheckpointRef checkpoint) {
                        return continuation_contains_checkpoint(entry.summary, checkpoint);
                    });
                record.private_claims.push_back(OwnerClaim{
                    .planning_id         = outcome.owner,
                    .capability          = owner_record->capability,
                    .disposition         = outcome.disposition,
                    .dropped_checkpoints = std::move(dropped),
                });
            } else {
                if (slot >= shared_catalog_count_ || slot == record.publication_slot) {
                    throw std::logic_error("shared capture shared pressure owner is duplicated");
                }
                const SharedCatalogEntry& entry = shared_catalog_[slot];
                if (entry.state != SharedCatalogState::Catalogued || !entry.handle ||
                    entry.id != owner_record->capability.owner.id ||
                    entry.revision != owner_record->capability.generation ||
                    entry.transaction_pins != 0 || shared_active_edge_count(slot) != 0) {
                    throw std::logic_error("shared capture shared pressure owner is stale");
                }
                std::vector<CheckpointRef> dropped = selected_checkpoint_drops(
                    outcome.owner, outcome.disposition, outcome.dropped_checkpoints,
                    selected->plan.checkpoint_outcomes, 1U, [&](CheckpointRef checkpoint) {
                        return checkpoint == entry.summary.checkpoint.ref;
                    });
                record.shared_claims.push_back(OwnerClaim{
                    .planning_id         = outcome.owner,
                    .capability          = owner_record->capability,
                    .disposition         = outcome.disposition,
                    .dropped_checkpoints = std::move(dropped),
                });
            }
        }
        if (!selected->plan.pressure) {
            throw std::logic_error("selected shared capture has no pressure plan");
        }
        if (record.publication_slot >= shared_catalog_count_) {
            throw std::logic_error("selected shared publication slot is invalid");
        }
        const SharedCatalogEntry& publication = shared_catalog_[record.publication_slot];
        if (record.replacement_id == 0) {
            if (publication.state != SharedCatalogState::Vacant || publication.id != 0 ||
                publication.handle || publication.transaction_pins != 0 ||
                shared_active_edge_count(record.publication_slot) != 0) {
                throw std::logic_error("selected vacant shared publication slot changed");
            }
        } else if (publication.state != SharedCatalogState::Catalogued || !publication.handle ||
                   publication.id != record.replacement_id ||
                   publication.revision != record.replacement_revision ||
                   publication.transaction_pins != 0 ||
                   shared_active_edge_count(record.publication_slot) != 0) {
            throw std::logic_error("selected shared replacement changed before reservation");
        }

        transaction_.template emplace<ActiveCaptureRecord>(std::move(record));
        ActiveCaptureRecord& open = std::get<ActiveCaptureRecord>(transaction_);
        reserve_logical_active_capture(open);
        const ContextTransactionReserveStatus reserved =
            program.reserve_active_capture_with_pressure(
                std::move(offer), nullptr, selected->scenario.replacement, private_replacement,
                true, std::move(*selected->plan.pressure), cancellation);
        if (reserved == ContextTransactionReserveStatus::Aborted) {
            rollback_logical_active_capture(open);
            transaction_.template emplace<std::monostate>();
            return ActiveCaptureReserveResult::Skipped;
        }
        return ActiveCaptureReserveResult::Reserved;
    }

    void mark_terminal_pending(LaneId lane) {
        require_lane(lane, LogicalLaneState::Active);
        lanes_[lane.value] = LogicalLaneState::TerminalPending;
    }

    [[nodiscard]] FinishResult finish(Program& program, LaneId lane, SequenceHandle sequence) {
        require_lane(lane, LogicalLaneState::TerminalPending);
        if (!std::holds_alternative<std::monostate>(transaction_) ||
            program.has_context_transaction()) {
            throw std::logic_error("terminal finish overlaps an open resource transaction");
        }
        ActiveEntry& active = active_[lane.value];
        FinishResult result = program.finish(sequence);
        if (result.status != ConsumeStatus::Consumed) {
            AbortResult discarded = program.abort(sequence);
            if (discarded.status != ConsumeStatus::Consumed) {
                throw std::logic_error(
                    "Program could neither retain nor discard terminal sequence");
            }
            release_active_references(lane);
            clear_catalog_entry(catalog_.at(active.publication_slot));
            reset_active_entry(active);
            lanes_[lane.value] = LogicalLaneState::Free;

            FinishResult released;
            released.status      = ConsumeStatus::Consumed;
            released.disposition = FinishDisposition::Released;
            released.timings     = discarded.timings;
            released.speculative = std::move(discarded.speculative);
            return released;
        }
        CatalogEntry& publication = catalog_.at(active.publication_slot);
        if (!cache_enabled_ || result.disposition == FinishDisposition::Released) {
            if (result.disposition != FinishDisposition::Released || result.continuation) {
                throw std::logic_error("released finish returned a continuation");
            }
            release_active_references(lane);
            clear_catalog_entry(publication);
            reset_active_entry(active);
            lanes_[lane.value] = LogicalLaneState::Free;
            return result;
        }
        if (result.disposition != FinishDisposition::Catalogued || !result.continuation ||
            !valid_continuation_summary(result.summary) ||
            publication.state != CatalogState::ReservedForActive ||
            publication.id != active.continuation_id) {
            if (result.continuation) {
                (void)program.release_continuation(std::move(*result.continuation));
                result.continuation.reset();
            }
            throw std::logic_error("Program returned an invalid terminal continuation");
        }

        release_active_references(lane);
        publication.state = CatalogState::Catalogued;
        assign_continuation_summary(publication.summary, result.summary);
        publication.handle.emplace(std::move(*result.continuation));
        result.continuation.reset();
        publication.session   = active.session;
        publication.retention = active.retention;
        migrate_observations(publication, result.summary, active.retention);
        advance_revision(publication.revision);
        if (publication.session && active.update_session_index) {
            if (!publish_session(*publication.session, active.publication_slot, publication.id,
                                 publication.revision, active.publication_order)) {
                publication.session.reset();
                publication.retention = RetentionClass::RecentPrivate;
            }
        }
        reset_active_entry(active);
        lanes_[lane.value] = LogicalLaneState::Free;
        return result;
    }

    [[nodiscard]] AbortResult abort(Program& program, LaneId lane, SequenceHandle sequence) {
        if (!std::holds_alternative<std::monostate>(transaction_) ||
            program.has_context_transaction()) {
            throw std::logic_error("terminal abort overlaps an open resource transaction");
        }
        if (lanes_.at(lane.value) == LogicalLaneState::Active) {
            lanes_[lane.value] = LogicalLaneState::TerminalPending;
        }
        require_lane(lane, LogicalLaneState::TerminalPending);
        AbortResult result = program.abort(sequence);
        if (result.status != ConsumeStatus::Consumed) {
            throw std::logic_error("Program did not consume aborted sequence");
        }
        release_active_references(lane);
        clear_catalog_entry(catalog_.at(active_[lane.value].publication_slot));
        reset_active_entry(active_[lane.value]);
        lanes_[lane.value] = LogicalLaneState::Free;
        return result;
    }

    void apply_commit(std::span<const LaneId> lanes, const typename Package::CommitResult& result) {
        if (lanes.size() != result.row_count) {
            throw std::logic_error("commit result membership is not row aligned");
        }
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const LaneId lane = lanes[row];
            require_lane(lane, LogicalLaneState::Active);
            switch (result.rows[row].disposition) {
            case CommitDisposition::Active:
                break;
            case CommitDisposition::Finishable:
                lanes_[lane.value] = LogicalLaneState::TerminalPending;
                break;
            case CommitDisposition::CancelledReleased:
                release_cancelled_lane(lane);
                break;
            }
        }
    }

    void apply_discard(std::span<const LaneId> lanes,
                       const typename Package::DiscardResult& result) {
        if (lanes.size() != result.row_count || result.status != ConsumeStatus::Consumed) {
            throw std::logic_error("pending discard did not consume its membership");
        }
        for (const LaneId lane : lanes) { release_cancelled_lane(lane); }
    }

    void release_failed_commit(std::span<const LaneId> lanes) noexcept {
        for (const LaneId lane : lanes) {
            if (lane.value < lane_count_ && active_[lane.value].occupied) {
                try {
                    release_cancelled_lane(lane);
                } catch (...) {}
            }
        }
    }

    void populate_runtime_stats(Program& program, RuntimeStats& out) const noexcept {
        out.state_moves                        = context_stats_.state_moves;
        out.state_forks                        = context_stats_.state_forks;
        out.state_restores                     = context_stats_.state_restores;
        out.state_d2h_count                    = context_stats_.state_d2h_count;
        out.state_h2d_count                    = context_stats_.state_h2d_count;
        out.state_d2d_count                    = context_stats_.state_d2d_count;
        out.state_d2h_bytes                    = context_stats_.state_d2h_bytes;
        out.state_h2d_bytes                    = context_stats_.state_h2d_bytes;
        out.state_d2d_bytes                    = context_stats_.state_d2d_bytes;
        out.state_d2h_seconds                  = context_stats_.state_d2h_seconds;
        out.state_h2d_seconds                  = context_stats_.state_h2d_seconds;
        out.state_d2d_seconds                  = context_stats_.state_d2d_seconds;
        out.main_kv_d2h_pages                  = context_stats_.main_kv_d2h_pages;
        out.main_kv_h2d_pages                  = context_stats_.main_kv_h2d_pages;
        out.main_kv_d2d_pages                  = context_stats_.main_kv_d2d_pages;
        out.main_kv_d2h_bytes                  = context_stats_.main_kv_d2h_bytes;
        out.main_kv_h2d_bytes                  = context_stats_.main_kv_h2d_bytes;
        out.main_kv_d2d_bytes                  = context_stats_.main_kv_d2d_bytes;
        out.main_kv_d2h_seconds                = context_stats_.main_kv_d2h_seconds;
        out.main_kv_h2d_seconds                = context_stats_.main_kv_h2d_seconds;
        out.main_kv_d2d_seconds                = context_stats_.main_kv_d2d_seconds;
        out.backend_kv_d2h_pages               = context_stats_.backend_kv_d2h_pages;
        out.backend_kv_h2d_pages               = context_stats_.backend_kv_h2d_pages;
        out.backend_kv_d2d_pages               = context_stats_.backend_kv_d2d_pages;
        out.backend_kv_d2h_bytes               = context_stats_.backend_kv_d2h_bytes;
        out.backend_kv_h2d_bytes               = context_stats_.backend_kv_h2d_bytes;
        out.backend_kv_d2d_bytes               = context_stats_.backend_kv_d2d_bytes;
        out.backend_kv_d2h_seconds             = context_stats_.backend_kv_d2h_seconds;
        out.backend_kv_h2d_seconds             = context_stats_.backend_kv_h2d_seconds;
        out.backend_kv_d2d_seconds             = context_stats_.backend_kv_d2d_seconds;
        out.pressure_spill_pages               = context_stats_.pressure_spill_pages;
        out.partial_tail_cow_pages             = context_stats_.partial_tail_cow_pages;
        out.pressure_private_owners_degraded   = context_stats_.pressure_private_owners_degraded;
        out.pressure_private_owners_evicted    = context_stats_.pressure_private_owners_evicted;
        out.pressure_shared_owners_degraded    = context_stats_.pressure_shared_owners_degraded;
        out.pressure_shared_owners_evicted     = context_stats_.pressure_shared_owners_evicted;
        out.pressure_checkpoints_dropped       = context_stats_.pressure_checkpoints_dropped;
        out.pressure_searches                  = context_stats_.pressure_searches;
        out.pressure_search_budget_exhaustions = context_stats_.pressure_search_budget_exhaustions;
        out.pressure_maximal_fallback_selections =
            context_stats_.pressure_maximal_fallback_selections;
        out.historical_fork_hits            = context_stats_.historical_fork_hits;
        out.actual_context_transfer_seconds = context_stats_.actual_context_transfer_seconds;

        const auto usage                     = program.physical_usage();
        out.device_state_occupied_slots      = usage.device_state_slots;
        out.host_state_occupied_slots        = usage.host_state_slots;
        out.device_main_kv_occupied_pages    = usage.device_main_kv_pages;
        out.device_backend_kv_occupied_pages = usage.device_backend_kv_pages;
        out.host_kv_occupied_bytes           = usage.host_kv_bytes;
        std::uint64_t shared_references      = 0;
        for (std::uint32_t lane = 0; lane < lane_count_; ++lane) {
            if (active_[lane].occupied) {
                shared_references += active_[lane].shared_sources.size();
            }
        }
        out.shared_active_references = shared_references > std::numeric_limits<std::uint32_t>::max()
                                           ? std::numeric_limits<std::uint32_t>::max()
                                           : static_cast<std::uint32_t>(shared_references);
    }

    [[nodiscard]] CatalogState catalog_state(std::uint32_t slot) const noexcept {
        return slot < catalog_count_ ? catalog_[slot].state : CatalogState::Vacant;
    }

    [[nodiscard]] LogicalLaneState lane_state(LaneId lane) const noexcept {
        return lane.value < lane_count_ ? lanes_[lane.value] : LogicalLaneState::Free;
    }

    void clear_after_program_cleanup() noexcept {
        transaction_.template emplace<std::monostate>();
        for (CatalogEntry& entry : catalog_) {
            entry.handle.reset();
            clear_catalog_entry(entry);
        }
        for (SharedCatalogEntry& entry : shared_catalog_) {
            entry.handle.reset();
            clear_shared_entry(entry);
        }
        for (SessionIndexEntry& entry : session_index_) { entry = {}; }
        for (std::uint32_t lane = 0; lane < lane_count_; ++lane) {
            lanes_[lane] = LogicalLaneState::Free;
            reset_active_entry(active_[lane]);
        }
        demand_window_.clear();
        demand_epoch_ = 0;
    }

private:
    struct Candidate {
        std::optional<AdmissionCandidate> plan;
        bool current_session_binding = false;
        std::optional<CatalogCapability> private_source;
        std::optional<CatalogCapability> shared_source;
        std::optional<PolicyObservationKey> selected_observation;
        std::optional<PrefixShortlistKey> source_key;
    };

    struct CatalogEntry {
        CatalogState state     = CatalogState::Vacant;
        std::uint64_t id       = 0;
        std::uint64_t revision = 1;
        ContinuationSummary summary;
        std::optional<ContinuationHandle> handle;
        std::optional<CacheSessionKey> session;
        std::vector<CheckpointObservation> observations;
        RetentionClass retention = RetentionClass::RecentPrivate;
    };

    struct SharedCatalogEntry {
        SharedCatalogState state = SharedCatalogState::Vacant;
        std::uint64_t id         = 0;
        std::uint64_t revision   = 1;
        SharedPrefixSummary summary;
        std::optional<SharedPrefixHandle> handle;
        RetentionObservation observation{.retention_class = RetentionClass::SharedStable};
        std::uint32_t transaction_pins    = 0;
        bool explicit_credit              = false;
        std::uint64_t credit_expiry_epoch = 0;
    };

    enum class SessionIndexState : std::uint8_t {
        Empty,
        Occupied,
        Deleted,
    };

    struct SessionIndexEntry {
        SessionIndexState state = SessionIndexState::Empty;
        CacheSessionKey key;
        std::uint32_t slot              = kInvalidCatalogSlot;
        std::uint64_t owner_id          = 0;
        std::uint64_t revision          = 0;
        std::uint64_t publication_order = 0;
    };

    struct PrefixIndexEntry {
        bool occupied = false;
        bool shared   = false;
        PrefixShortlistKey key;
        std::uint32_t slot     = kInvalidCatalogSlot;
        std::uint64_t owner_id = 0;
        std::uint64_t revision = 0;
        CheckpointRef checkpoint;
    };

    struct ActiveEntry {
        bool occupied                  = false;
        std::uint32_t publication_slot = kInvalidCatalogSlot;
        std::uint64_t continuation_id  = 0;
        std::optional<CacheSessionKey> session;
        RetentionClass retention        = RetentionClass::RecentPrivate;
        bool update_session_index       = true;
        std::uint64_t publication_order = 0;
        std::optional<ActiveOwnerEdge> retained_private_source;
        std::vector<ActiveOwnerEdge> shared_sources;
    };

    struct MaterializationRecord {
        LaneId destination;
        std::optional<CatalogCapability> private_source;
        PrivateSourceMode source_mode = PrivateSourceMode::ConsumeToActive;
        std::optional<CatalogCapability> shared_source;
        std::uint32_t publication_slot = kInvalidCatalogSlot;
        std::vector<OwnerClaim> private_claims;
        std::vector<OwnerClaim> shared_claims;
        std::optional<PolicyObservationKey> selected_observation;
        std::optional<CacheSessionKey> session;
        RetentionClass retention        = RetentionClass::RecentPrivate;
        bool update_session_index       = true;
        std::uint64_t publication_order = 0;
        MaterializationDiagnostics diagnostics;
        PrefixDemandRecord demand;
    };

    struct ActiveCaptureRecord {
        LaneId lane;
        bool publishes_private                  = false;
        bool publishes_shared                   = false;
        std::uint32_t publication_slot          = kInvalidCatalogSlot;
        std::uint64_t replacement_id            = 0;
        std::uint64_t replacement_revision      = 0;
        SharedCandidateEvidence shared_evidence = SharedCandidateEvidence::None;
        std::vector<OwnerClaim> private_claims;
        std::vector<OwnerClaim> shared_claims;
    };

    [[nodiscard]] CatalogCapability private_capability(std::uint32_t slot) const {
        const CatalogEntry& entry = catalog_.at(slot);
        return CatalogCapability{
            .owner =
                LogicalOwnerKey{
                    .kind = LogicalOwnerKind::PrivateContinuation,
                    .id   = entry.id,
                },
            .slot       = slot,
            .generation = entry.revision,
        };
    }

    [[nodiscard]] CatalogCapability shared_capability(std::uint32_t slot) const {
        const SharedCatalogEntry& entry = shared_catalog_.at(slot);
        return CatalogCapability{
            .owner =
                LogicalOwnerKey{
                    .kind = LogicalOwnerKind::SharedPrefix,
                    .id   = entry.id,
                },
            .slot       = slot,
            .generation = entry.revision,
        };
    }

    [[nodiscard]] static constexpr ActiveOwnerEdge
    active_edge(CatalogCapability capability) noexcept {
        return ActiveOwnerEdge{.owner = capability.owner, .slot = capability.slot};
    }

    [[nodiscard]] bool private_has_active_edge(std::uint32_t slot) const noexcept {
        return std::any_of(active_.begin(), active_.begin() + lane_count_,
                           [&](const ActiveEntry& active) {
                               return active.occupied && active.retained_private_source &&
                                      active.retained_private_source->slot == slot;
                           });
    }

    [[nodiscard]] std::uint32_t shared_active_edge_count(std::uint32_t slot) const noexcept {
        std::uint32_t count = 0;
        for (std::uint32_t lane = 0; lane < lane_count_; ++lane) {
            const ActiveEntry& active = active_[lane];
            if (!active.occupied) { continue; }
            count += static_cast<std::uint32_t>(
                std::count_if(active.shared_sources.begin(), active.shared_sources.end(),
                              [&](const ActiveOwnerEdge& edge) { return edge.slot == slot; }));
        }
        return count;
    }

    void reset_active_entry(ActiveEntry& active) noexcept {
        active.occupied         = false;
        active.publication_slot = kInvalidCatalogSlot;
        active.continuation_id  = 0;
        active.session.reset();
        active.retention            = RetentionClass::RecentPrivate;
        active.update_session_index = true;
        active.publication_order    = 0;
        active.retained_private_source.reset();
        active.shared_sources.clear();
    }

    [[nodiscard]] static std::size_t checked_prefix_index_capacity(std::uint32_t private_capacity,
                                                                   std::uint32_t shared_capacity,
                                                                   std::uint32_t max_long_anchors) {
        const std::size_t width = static_cast<std::size_t>(max_long_anchors) + 2U;
        if (private_capacity != 0 &&
            width >
                (std::numeric_limits<std::size_t>::max() - shared_capacity) / private_capacity) {
            throw std::overflow_error("prefix index capacity overflow");
        }
        return static_cast<std::size_t>(private_capacity) * width + shared_capacity;
    }

    static void advance_revision(std::uint64_t& revision) noexcept {
        if (++revision == 0) { ++revision; }
    }

    static void saturating_increment(std::uint64_t& value) noexcept {
        if (value != std::numeric_limits<std::uint64_t>::max()) { ++value; }
    }

    static constexpr std::size_t kDemandWindowCapacity = 32U;

    static void append_unique(std::vector<PrefixShortlistKey>& destination,
                              const PrefixShortlistKey& key) {
        if (std::find(destination.begin(), destination.end(), key) == destination.end()) {
            destination.push_back(key);
        }
    }

    [[nodiscard]] static bool demand_matches(const PrefixDemandRecord& demand,
                                             const PrefixShortlistKey& key) noexcept {
        return std::find(demand.candidate_keys.begin(), demand.candidate_keys.end(), key) !=
                   demand.candidate_keys.end() ||
               std::find(demand.exact_resident_keys.begin(), demand.exact_resident_keys.end(),
                         key) != demand.exact_resident_keys.end() ||
               (demand.selected_source_key && *demand.selected_source_key == key);
    }

    [[nodiscard]] static ReuseDomainId reuse_domain(const std::optional<CacheSessionKey>& session,
                                                    std::uint64_t publication_order) noexcept {
        if (!session) {
            return ReuseDomainId{
                .low  = publication_order,
                .high = publication_order ^ 0xD6E8FEB86659FD93ULL,
            };
        }
        std::uint64_t low  = 1469598103934665603ULL;
        std::uint64_t high = 1099511628211ULL ^ 0x9E3779B97F4A7C15ULL;
        for (const unsigned char value : session->view()) {
            low ^= value;
            low *= 1099511628211ULL;
            high ^= static_cast<std::uint64_t>(value) + 0x9E3779B97F4A7C15ULL + (high << 6U) +
                    (high >> 2U);
            high *= 0xD6E8FEB86659FD93ULL;
        }
        return ReuseDomainId{.low = low, .high = high};
    }

    [[nodiscard]] std::uint32_t
    demand_mask_for(const PrefixShortlistKey& key,
                    const PrefixDemandRecord& provisional) const noexcept {
        std::uint32_t mask      = 0;
        std::uint32_t bit       = 0;
        const std::size_t begin = demand_window_.size() == kDemandWindowCapacity ? 1U : 0U;
        for (std::size_t index = begin; index < demand_window_.size(); ++index, ++bit) {
            if (demand_matches(demand_window_[index], key)) { mask |= 1U << bit; }
        }
        if (bit < kDemandWindowCapacity && demand_matches(provisional, key)) { mask |= 1U << bit; }
        return mask;
    }

    [[nodiscard]] std::uint32_t
    committed_demand_mask_for(const PrefixShortlistKey& key) const noexcept {
        std::uint32_t mask = 0;
        for (std::uint32_t bit = 0; bit < demand_window_.size(); ++bit) {
            if (demand_matches(demand_window_[bit], key)) { mask |= 1U << bit; }
        }
        return mask;
    }

    [[nodiscard]] std::size_t matching_reuse_domains(const PrefixShortlistKey& key) const noexcept {
        std::array<ReuseDomainId, kDemandWindowCapacity> domains{};
        std::size_t count = 0;
        for (const PrefixDemandRecord& demand : demand_window_) {
            if (!demand_matches(demand, key) ||
                std::find(domains.begin(), domains.begin() + static_cast<std::ptrdiff_t>(count),
                          demand.domain) != domains.begin() + static_cast<std::ptrdiff_t>(count)) {
                continue;
            }
            domains[count++] = demand.domain;
        }
        return count;
    }

    [[nodiscard]] std::size_t
    matching_reuse_domains(const PrefixShortlistKey& key,
                           const PrefixDemandRecord& provisional) const noexcept {
        std::array<ReuseDomainId, kDemandWindowCapacity> domains{};
        std::size_t count       = 0;
        const std::size_t begin = demand_window_.size() == kDemandWindowCapacity ? 1U : 0U;
        const auto append       = [&](const PrefixDemandRecord& demand) {
            if (!demand_matches(demand, key) ||
                std::find(domains.begin(), domains.begin() + static_cast<std::ptrdiff_t>(count),
                                demand.domain) != domains.begin() + static_cast<std::ptrdiff_t>(count)) {
                return;
            }
            domains[count++] = demand.domain;
        };
        for (std::size_t index = begin; index < demand_window_.size(); ++index) {
            append(demand_window_[index]);
        }
        append(provisional);
        return count;
    }

    [[nodiscard]] static std::uint32_t private_retention_weight(RetentionClass retention) noexcept {
        switch (retention) {
        case RetentionClass::Disposable:
            return 1;
        case RetentionClass::RecentPrivate:
            return 4;
        case RetentionClass::LiveSession:
            return 16;
        case RetentionClass::SharedStable:
            return 0;
        }
        return 0;
    }

    void require_lane(LaneId lane, LogicalLaneState expected) const {
        if (lane.value >= lane_count_ || lanes_[lane.value] != expected ||
            ((expected == LogicalLaneState::Active ||
              expected == LogicalLaneState::TerminalPending) &&
             !active_[lane.value].occupied)) {
            throw std::logic_error("logical lane is not in the required state");
        }
    }

    [[nodiscard]] bool valid_checkpoint_summary(const auto& checkpoint,
                                                CheckpointScope scope) const noexcept {
        return checkpoint.ref.frontier != 0 && checkpoint.ref.ordinal == 0 &&
               checkpoint.scope == scope &&
               checkpoint.shortlist_key.frontier == checkpoint.ref.frontier &&
               checkpoint.required_kv.main_pages != 0 && checkpoint.rebuild_work.tokens != 0;
    }

    [[nodiscard]] bool
    valid_continuation_summary(const ContinuationSummary& summary) const noexcept {
        if ((!summary.endpoint && !summary.rewrite && summary.long_anchors.empty()) ||
            summary.long_anchors.size() > max_long_anchors_) {
            return false;
        }
        if (summary.endpoint &&
            (summary.endpoint->ref.kind != CheckpointKind::SessionEndpoint ||
             !valid_checkpoint_summary(*summary.endpoint, CheckpointScope::Private))) {
            return false;
        }
        if (summary.rewrite &&
            (summary.rewrite->ref.kind == CheckpointKind::SessionEndpoint ||
             summary.rewrite->ref.kind == CheckpointKind::SharedStablePrefix ||
             summary.rewrite->ref.kind == CheckpointKind::LongAnchor ||
             !valid_checkpoint_summary(*summary.rewrite, CheckpointScope::Private))) {
            return false;
        }
        for (std::size_t index = 0; index < summary.long_anchors.size(); ++index) {
            const auto& anchor = summary.long_anchors[index];
            if (anchor.ref.kind != CheckpointKind::LongAnchor || anchor.ref.ordinal == 0 ||
                anchor.ref.ordinal > max_long_anchors_ || anchor.ref.frontier == 0 ||
                anchor.scope != CheckpointScope::Private ||
                anchor.shortlist_key.frontier != anchor.ref.frontier ||
                anchor.required_kv.main_pages == 0 || anchor.rebuild_work.tokens == 0) {
                return false;
            }
            for (std::size_t previous = 0; previous < index; ++previous) {
                if (summary.long_anchors[previous].ref.ordinal == anchor.ref.ordinal) {
                    return false;
                }
            }
        }
        return true;
    }

    [[nodiscard]] static bool
    valid_shared_prefix_summary(const SharedPrefixSummary& summary) noexcept {
        const auto& checkpoint = summary.checkpoint;
        return checkpoint.ref.kind == CheckpointKind::SharedStablePrefix &&
               checkpoint.ref.frontier != 0 && checkpoint.ref.ordinal == 0 &&
               checkpoint.scope == CheckpointScope::Shared &&
               checkpoint.shortlist_key.frontier == checkpoint.ref.frontier &&
               checkpoint.required_kv.main_pages != 0 && checkpoint.rebuild_work.tokens != 0;
    }

    static void assign_continuation_summary(ContinuationSummary& destination,
                                            const ContinuationSummary& source) noexcept {
        if (source.long_anchors.size() > destination.long_anchors.capacity()) { std::terminate(); }
        destination.endpoint          = source.endpoint;
        destination.rewrite           = source.rewrite;
        destination.active_references = 0;
        destination.long_anchors.clear();
        for (const auto& anchor : source.long_anchors) {
            destination.long_anchors.push_back(anchor);
        }
    }

    static RetentionObservation* find_observation(std::vector<CheckpointObservation>& observations,
                                                  CheckpointRef checkpoint) noexcept {
        const auto found = std::find_if(
            observations.begin(), observations.end(),
            [&](const CheckpointObservation& value) { return value.checkpoint == checkpoint; });
        return found == observations.end() ? nullptr : &found->observation;
    }

    static const RetentionObservation*
    find_observation(const std::vector<CheckpointObservation>& observations,
                     CheckpointRef checkpoint) noexcept {
        const auto found = std::find_if(
            observations.begin(), observations.end(),
            [&](const CheckpointObservation& value) { return value.checkpoint == checkpoint; });
        return found == observations.end() ? nullptr : &found->observation;
    }

    void migrate_observations(CatalogEntry& entry, const ContinuationSummary& summary,
                              RetentionClass retention) noexcept {
        observation_scratch_.clear();
        const auto append = [&](const auto& checkpoint) {
            if (observation_scratch_.size() == observation_scratch_.capacity()) {
                std::terminate();
            }
            RetentionObservation observation{.retention_class = retention};
            if (const RetentionObservation* old =
                    find_observation(entry.observations, checkpoint.ref)) {
                observation                 = *old;
                observation.retention_class = retention;
            }
            observation_scratch_.push_back(
                CheckpointObservation{.checkpoint = checkpoint.ref, .observation = observation});
        };
        if (summary.endpoint) { append(*summary.endpoint); }
        if (summary.rewrite) { append(*summary.rewrite); }
        for (const auto& anchor : summary.long_anchors) { append(anchor); }
        entry.observations.clear();
        for (const CheckpointObservation& observation : observation_scratch_) {
            entry.observations.push_back(observation);
        }
    }

    void clear_catalog_entry(CatalogEntry& entry) noexcept {
        entry.state = CatalogState::Vacant;
        entry.id    = 0;
        entry.summary.endpoint.reset();
        entry.summary.rewrite.reset();
        entry.summary.long_anchors.clear();
        entry.summary.active_references = 0;
        entry.handle.reset();
        entry.session.reset();
        entry.observations.clear();
        entry.retention = RetentionClass::RecentPrivate;
        advance_revision(entry.revision);
    }

    void clear_shared_entry(SharedCatalogEntry& entry) noexcept {
        entry.state = SharedCatalogState::Vacant;
        entry.id    = 0;
        entry.handle.reset();
        entry.summary     = {};
        entry.observation = RetentionObservation{.retention_class = RetentionClass::SharedStable};
        entry.transaction_pins    = 0;
        entry.explicit_credit     = false;
        entry.credit_expiry_epoch = 0;
        advance_revision(entry.revision);
    }

    void rebuild_prefix_index() {
        for (PrefixIndexEntry& entry : prefix_index_) { entry = {}; }
        std::size_t cursor = 0;
        const auto append  = [&](bool shared, std::uint32_t slot, std::uint64_t owner_id,
                                std::uint64_t revision, const auto& checkpoint) {
            if (cursor >= prefix_index_.size()) {
                throw std::logic_error("prefix index exceeded fixed capacity");
            }
            prefix_index_[cursor++] = PrefixIndexEntry{
                 .occupied   = true,
                 .shared     = shared,
                 .key        = checkpoint.shortlist_key,
                 .slot       = slot,
                 .owner_id   = owner_id,
                 .revision   = revision,
                 .checkpoint = checkpoint.ref,
            };
        };
        for (std::uint32_t slot = 0; slot < catalog_count_; ++slot) {
            const CatalogEntry& entry = catalog_[slot];
            if (entry.state != CatalogState::Catalogued || !entry.handle) { continue; }
            if (entry.summary.endpoint) {
                append(false, slot, entry.id, entry.revision, *entry.summary.endpoint);
            }
            if (entry.summary.rewrite) {
                append(false, slot, entry.id, entry.revision, *entry.summary.rewrite);
            }
            for (const auto& anchor : entry.summary.long_anchors) {
                append(false, slot, entry.id, entry.revision, anchor);
            }
        }
        for (std::uint32_t slot = 0; slot < shared_catalog_count_; ++slot) {
            const SharedCatalogEntry& entry = shared_catalog_[slot];
            if (entry.state == SharedCatalogState::Catalogued && entry.handle) {
                append(true, slot, entry.id, entry.revision, entry.summary.checkpoint);
            }
        }
    }

    [[nodiscard]] bool valid_prefix_index_entry(const PrefixIndexEntry& index) const noexcept {
        if (!index.occupied) { return false; }
        if (!index.shared) {
            if (index.slot >= catalog_count_) { return false; }
            const CatalogEntry& entry = catalog_[index.slot];
            return entry.state == CatalogState::Catalogued && entry.handle &&
                   entry.id == index.owner_id && entry.revision == index.revision;
        }
        if (index.slot >= shared_catalog_count_) { return false; }
        const SharedCatalogEntry& entry = shared_catalog_[index.slot];
        return entry.state == SharedCatalogState::Catalogued && entry.handle &&
               entry.id == index.owner_id && entry.revision == index.revision;
    }

    [[nodiscard]] std::uint64_t newest_hit_epoch(const CatalogEntry& entry) const noexcept {
        std::uint64_t epoch = 0;
        for (const CheckpointObservation& observation : entry.observations) {
            epoch = std::max(epoch, observation.observation.last_hit_epoch);
        }
        return epoch;
    }

    template <class SplitCostFn>
    [[nodiscard]] std::vector<std::uint32_t> select_materialization_shared_captures(
        Program& program, const RequestBasePlan& base, const Candidate& selected_candidate,
        const RequestPlanSummary& selected_summary, const PrefixDemandRecord& provisional_demand,
        SplitCostFn&& split_cost) const {
        struct ProjectedSharedCandidate {
            PrefixShortlistKey key;
            SharedCandidateEvidence evidence = SharedCandidateEvidence::None;
            std::uint32_t frontier           = 0;
            std::uint32_t demand_mask        = 0;
            std::uint64_t rebuild_ns         = 0;
            bool pressure_capable            = false;
        };

        std::vector<ProjectedSharedCandidate> shared_candidates;
        shared_candidates.reserve(base.context_cache().opportunities.size());
        const std::uint32_t vacant_shared_slots = static_cast<std::uint32_t>(
            std::count_if(shared_catalog_.begin(), shared_catalog_.end(), [](const auto& entry) {
                return entry.state == SharedCatalogState::Vacant;
            }));
        for (const auto& opportunity : base.context_cache().opportunities) {
            if (opportunity.kind != PromptCacheMarkerKind::SharedStablePrefix ||
                opportunity.frontier < selected_summary.reusable_prompt_tokens) {
                continue;
            }
            const std::optional<PrefixShortlistKey> key =
                base.prefix_shortlist_key(opportunity.frontier);
            if (!key) { continue; }
            const bool selected_private_base =
                opportunity.frontier == selected_summary.reusable_prompt_tokens &&
                selected_candidate.private_source && selected_candidate.source_key &&
                *selected_candidate.source_key == *key;
            const bool exact_shared_resident = std::any_of(
                prefix_index_.begin(), prefix_index_.end(), [&](const PrefixIndexEntry& entry) {
                    return entry.shared && valid_prefix_index_entry(entry) && entry.key == *key;
                });
            const bool exact_resident =
                std::find(provisional_demand.exact_resident_keys.begin(),
                          provisional_demand.exact_resident_keys.end(),
                          *key) != provisional_demand.exact_resident_keys.end();
            if (exact_shared_resident || (exact_resident && !selected_private_base)) { continue; }
            const bool declared =
                has_shared_candidate_evidence(opportunity.evidence,
                                              SharedCandidateEvidence::ExplicitBoundary) ||
                has_shared_candidate_evidence(opportunity.evidence,
                                              SharedCandidateEvidence::RequestedAutomatic);
            const bool repeated = matching_reuse_domains(*key, provisional_demand) >= 2U;
            const bool surplus_candidate =
                vacant_shared_slots != 0 &&
                (has_shared_candidate_evidence(opportunity.evidence,
                                               SharedCandidateEvidence::DefaultAutomatic) ||
                 has_shared_candidate_evidence(opportunity.evidence,
                                               SharedCandidateEvidence::EngineStructural));
            if (!declared && !repeated && !surplus_candidate) { continue; }
            const std::optional<PrefillWork> rebuild =
                base.shared_candidate_rebuild_work(opportunity.frontier);
            if (!rebuild) {
                throw std::logic_error("prepared shared candidate has no canonical rebuild work");
            }
            shared_candidates.push_back(ProjectedSharedCandidate{
                .key              = *key,
                .evidence         = opportunity.evidence,
                .frontier         = opportunity.frontier,
                .demand_mask      = demand_mask_for(*key, provisional_demand),
                .rebuild_ns       = cost_model_.prefill_ns(*rebuild),
                .pressure_capable = declared || repeated,
            });
        }

        std::vector<ContextPortfolioOwnerPolicy> projected_owners;
        std::vector<ContextPortfolioCheckpointValue> projected_checkpoints;
        std::uint32_t next_projected_owner = 0;
        projected_owners.reserve(catalog_count_ + shared_catalog_count_ + shared_candidates.size());
        projected_checkpoints.reserve(prefix_index_.size() + shared_candidates.size());
        const auto append_existing = [&](PlanningOwnerId owner, const auto& handle,
                                         const auto& checkpoint) {
            const std::uint64_t rebuild  = cost_model_.prefill_ns(checkpoint.rebuild_work);
            const std::uint64_t recovery = price_checkpoint_recovery_work(
                cost_model_, program.checkpoint_recovery_work(handle, checkpoint.ref));
            projected_checkpoints.push_back(ContextPortfolioCheckpointValue{
                .owner       = owner,
                .demand_mask = demand_mask_for(checkpoint.shortlist_key, provisional_demand),
                .rebuild_ns  = rebuild,
                .baseline_recovery_ns = recovery,
                .target_recovery_ns   = recovery,
            });
        };
        for (std::uint32_t slot = 0; slot < catalog_count_; ++slot) {
            const CatalogEntry& entry = catalog_[slot];
            if (entry.state != CatalogState::Catalogued || !entry.handle ||
                private_has_active_edge(slot) ||
                (selected_candidate.private_source &&
                 slot == selected_candidate.private_source->slot)) {
                continue;
            }
            const PlanningOwnerId owner{.value = next_projected_owner++};
            projected_owners.push_back(ContextPortfolioOwnerPolicy{
                .owner                    = owner,
                .private_retention_weight = private_retention_weight(entry.retention),
            });
            if (entry.summary.endpoint) {
                append_existing(owner, *entry.handle, *entry.summary.endpoint);
            }
            if (entry.summary.rewrite) {
                append_existing(owner, *entry.handle, *entry.summary.rewrite);
            }
            for (const auto& checkpoint : entry.summary.long_anchors) {
                append_existing(owner, *entry.handle, checkpoint);
            }
        }
        for (std::uint32_t slot = 0; slot < shared_catalog_count_; ++slot) {
            const SharedCatalogEntry& entry = shared_catalog_[slot];
            if (entry.state != SharedCatalogState::Catalogued || !entry.handle) { continue; }
            const PlanningOwnerId owner{.value = next_projected_owner++};
            projected_owners.push_back(ContextPortfolioOwnerPolicy{
                .owner                  = owner,
                .explicit_shared_credit = entry.explicit_credit,
            });
            append_existing(owner, *entry.handle, entry.summary.checkpoint);
        }

        std::vector<std::uint32_t> selected_frontiers;
        std::uint64_t selected_gain = 0;
        ContextPortfolioValue projected_value;
        if (shared_candidates.size() > 7U) {
            throw std::logic_error("prepared shared candidates exceeded the fixed subset bound");
        }
        const std::uint32_t subset_count = 1U << shared_candidates.size();
        for (std::uint32_t mask = 1; mask < subset_count; ++mask) {
            const std::uint32_t selected_count = std::popcount(mask);
            if (selected_count > shared_catalog_count_) { continue; }
            std::uint32_t surplus_only_count = 0;
            std::vector<std::uint32_t> frontiers;
            frontiers.reserve(selected_count);
            std::vector<ContextPortfolioOwnerPolicy> owners          = projected_owners;
            std::vector<ContextPortfolioCheckpointValue> checkpoints = projected_checkpoints;
            for (std::size_t index = 0; index < shared_candidates.size(); ++index) {
                if ((mask & (1U << index)) == 0) { continue; }
                const ProjectedSharedCandidate& candidate = shared_candidates[index];
                if (!candidate.pressure_capable) { ++surplus_only_count; }
                frontiers.push_back(candidate.frontier);
                const PlanningOwnerId owner{.value = next_projected_owner +
                                                     static_cast<std::uint32_t>(index)};
                const bool credit =
                    has_shared_candidate_evidence(candidate.evidence,
                                                  SharedCandidateEvidence::ExplicitBoundary) ||
                    has_shared_candidate_evidence(candidate.evidence,
                                                  SharedCandidateEvidence::RequestedAutomatic);
                owners.push_back(ContextPortfolioOwnerPolicy{
                    .owner                  = owner,
                    .explicit_shared_credit = credit,
                });
                checkpoints.push_back(ContextPortfolioCheckpointValue{
                    .owner                = owner,
                    .demand_mask          = candidate.demand_mask,
                    .rebuild_ns           = candidate.rebuild_ns,
                    .baseline_recovery_ns = candidate.rebuild_ns,
                    .target_recovery_ns   = 0,
                });
            }
            if (surplus_only_count > vacant_shared_slots) { continue; }
            std::sort(frontiers.begin(), frontiers.end());
            const ContextPortfolioValueResult value = projected_value.fold(owners, checkpoints);
            const std::uint64_t schedule_cost       = split_cost(frontiers);
            if (value.saturated ||
                value.private_transition_loss >
                    std::numeric_limits<std::uint64_t>::max() - value.baseline_public_value) {
                continue;
            }
            std::uint64_t threshold = value.baseline_public_value + value.private_transition_loss;
            if (schedule_cost > std::numeric_limits<std::uint64_t>::max() - threshold) { continue; }
            threshold += schedule_cost;
            if (value.target_public_value <= threshold) { continue; }
            const std::uint64_t gain = value.target_public_value - threshold;
            const bool better =
                gain > selected_gain ||
                (gain == selected_gain &&
                 (selected_frontiers.empty() || frontiers.size() < selected_frontiers.size() ||
                  (frontiers.size() == selected_frontiers.size() &&
                   std::lexicographical_compare(frontiers.begin(), frontiers.end(),
                                                selected_frontiers.begin(),
                                                selected_frontiers.end()))));
            if (better) {
                selected_gain      = gain;
                selected_frontiers = std::move(frontiers);
            }
        }
        return selected_frontiers;
    }

    [[nodiscard]] std::optional<Choice>
    plan_materialization(Program& program, const PreparedPrompt& prompt,
                         const RequestBasePlan& base, LaneId destination,
                         std::vector<Candidate>& candidates, std::uint64_t publication_order,
                         typename Planner::Clock::time_point planning_started,
                         PrefixDemandRecord& provisional_demand) {
        std::vector<typename Planner::CandidateInput> candidate_inputs;
        std::vector<const ContinuationHandle*> private_owners;
        std::vector<PlanningOwnerId> private_owner_ids;
        std::vector<const SharedPrefixHandle*> shared_owners;
        std::vector<PlanningOwnerId> shared_owner_ids;
        std::vector<PlanningOwnerRecord> owner_records;
        std::vector<MaterializationOwnerPolicy> owner_policies;
        std::vector<MaterializationCheckpointPolicy> checkpoint_policies;
        candidate_inputs.reserve(candidates.size());

        for (std::size_t index = 0; index < candidates.size(); ++index) {
            if (!candidates[index].plan) {
                throw std::logic_error("materialization candidate is empty");
            }
            candidate_inputs.push_back(typename Planner::CandidateInput{
                .candidate      = &*candidates[index].plan,
                .id             = PlanningCandidateId{.value = static_cast<std::uint32_t>(index)},
                .stable_ordinal = static_cast<std::uint32_t>(index),
                .current_session_binding = candidates[index].current_session_binding,
            });
        }

        bool pressure_inputs_built       = false;
        const auto build_pressure_inputs = [&]() -> typename Planner::PressureInputs {
            if (pressure_inputs_built) {
                throw std::logic_error("materialization pressure inputs requested twice");
            }
            pressure_inputs_built = true;
            private_owners.reserve(catalog_count_);
            private_owner_ids.reserve(catalog_count_);
            shared_owners.reserve(shared_catalog_count_);
            shared_owner_ids.reserve(shared_catalog_count_);
            owner_records.reserve(catalog_count_ + shared_catalog_count_);
            owner_policies.reserve(catalog_count_ + shared_catalog_count_);
            checkpoint_policies.reserve(prefix_index_.size());

            for (std::uint32_t slot = 0; slot < catalog_count_; ++slot) {
                const CatalogEntry& entry = catalog_[slot];
                if (entry.state != CatalogState::Catalogued || !entry.handle ||
                    private_has_active_edge(slot)) {
                    continue;
                }
                const PlanningOwnerId owner{.value =
                                                static_cast<std::uint32_t>(owner_records.size())};
                private_owners.push_back(&*entry.handle);
                private_owner_ids.push_back(owner);
                owner_records.push_back(PlanningOwnerRecord{
                    .id = owner,
                    .capability =
                        CatalogCapability{
                            .owner =
                                LogicalOwnerKey{
                                    .kind = LogicalOwnerKind::PrivateContinuation,
                                    .id   = entry.id,
                                },
                            .slot       = slot,
                            .generation = entry.revision,
                        },
                });
                std::uint64_t selected_hits  = 0;
                const auto append_checkpoint = [&](const auto& checkpoint) {
                    const RetentionObservation* observation =
                        find_observation(entry.observations, checkpoint.ref);
                    if (observation == nullptr) {
                        throw std::logic_error("catalogued checkpoint has no policy observation");
                    }
                    selected_hits = std::max(selected_hits, observation->selected_hit_count);
                    checkpoint_policies.push_back(MaterializationCheckpointPolicy{
                        .owner              = owner,
                        .checkpoint         = checkpoint.ref,
                        .retention_class    = observation->retention_class,
                        .selected_hit_count = observation->selected_hit_count,
                        .last_hit_epoch     = observation->last_hit_epoch,
                        .demand_mask =
                            demand_mask_for(checkpoint.shortlist_key, provisional_demand),
                        .rebuild_ns           = cost_model_.prefill_ns(checkpoint.rebuild_work),
                        .baseline_recovery_ns = price_checkpoint_recovery_work(
                            cost_model_,
                            program.checkpoint_recovery_work(*entry.handle, checkpoint.ref)),
                    });
                };
                if (entry.summary.endpoint) { append_checkpoint(*entry.summary.endpoint); }
                if (entry.summary.rewrite) { append_checkpoint(*entry.summary.rewrite); }
                for (const auto& checkpoint : entry.summary.long_anchors) {
                    append_checkpoint(checkpoint);
                }
                owner_policies.push_back(MaterializationOwnerPolicy{
                    .owner                    = owner,
                    .retention_class          = entry.retention,
                    .selected_hit_count       = selected_hits,
                    .last_hit_epoch           = newest_hit_epoch(entry),
                    .private_retention_weight = private_retention_weight(entry.retention),
                });
            }
            for (std::uint32_t slot = 0; slot < shared_catalog_count_; ++slot) {
                const SharedCatalogEntry& entry = shared_catalog_[slot];
                if (entry.state != SharedCatalogState::Catalogued || !entry.handle ||
                    entry.transaction_pins != 0 || shared_active_edge_count(slot) != 0) {
                    continue;
                }
                const PlanningOwnerId owner{.value =
                                                static_cast<std::uint32_t>(owner_records.size())};
                shared_owners.push_back(&*entry.handle);
                shared_owner_ids.push_back(owner);
                owner_records.push_back(PlanningOwnerRecord{
                    .id = owner,
                    .capability =
                        CatalogCapability{
                            .owner =
                                LogicalOwnerKey{
                                    .kind = LogicalOwnerKind::SharedPrefix,
                                    .id   = entry.id,
                                },
                            .slot       = slot,
                            .generation = entry.revision,
                        },
                });
                owner_policies.push_back(MaterializationOwnerPolicy{
                    .owner                    = owner,
                    .retention_class          = RetentionClass::SharedStable,
                    .selected_hit_count       = entry.observation.selected_hit_count,
                    .last_hit_epoch           = entry.observation.last_hit_epoch,
                    .private_retention_weight = 0,
                    .explicit_shared_credit   = entry.explicit_credit,
                });
                checkpoint_policies.push_back(MaterializationCheckpointPolicy{
                    .owner              = owner,
                    .checkpoint         = entry.summary.checkpoint.ref,
                    .retention_class    = RetentionClass::SharedStable,
                    .selected_hit_count = entry.observation.selected_hit_count,
                    .last_hit_epoch     = entry.observation.last_hit_epoch,
                    .demand_mask =
                        demand_mask_for(entry.summary.checkpoint.shortlist_key, provisional_demand),
                    .rebuild_ns = cost_model_.prefill_ns(entry.summary.checkpoint.rebuild_work),
                    .baseline_recovery_ns = price_checkpoint_recovery_work(
                        cost_model_, program.checkpoint_recovery_work(
                                         *entry.handle, entry.summary.checkpoint.ref)),
                });
            }

            return typename Planner::PressureInputs{
                .private_owners    = private_owners,
                .private_owner_ids = private_owner_ids,
                .shared_owners     = shared_owners,
                .shared_owner_ids  = shared_owner_ids,
                .owner_policy      = owner_policies,
                .checkpoint_policy = checkpoint_policies,
            };
        };

        const auto logical_goal = [&](PlanningCandidateId candidate_id,
                                      PrivateSourceMode source_mode,
                                      std::span<const PressureOwnerOutcome> outcomes)
            -> std::optional<typename Planner::LogicalGoal> {
            const auto candidate_record =
                std::find_if(candidate_inputs.begin(), candidate_inputs.end(),
                             [&](const typename Planner::CandidateInput& input) {
                                 return input.id == candidate_id;
                             });
            if (candidate_record == candidate_inputs.end()) { return std::nullopt; }
            const std::size_t candidate_index =
                static_cast<std::size_t>(candidate_record - candidate_inputs.begin());
            const Candidate& candidate = candidates[candidate_index];
            if (candidate.shared_source && source_mode != PrivateSourceMode::Retain) {
                return std::nullopt;
            }
            if (!candidate.private_source && !candidate.shared_source &&
                source_mode == PrivateSourceMode::Retain) {
                return std::nullopt;
            }
            if (candidate.private_source && source_mode != PrivateSourceMode::Retain &&
                source_mode != PrivateSourceMode::ConsumeToActive) {
                return std::nullopt;
            }

            std::uint32_t publication_slot = kInvalidCatalogSlot;
            if (candidate.private_source && source_mode == PrivateSourceMode::ConsumeToActive) {
                publication_slot = candidate.private_source->slot;
            } else {
                for (std::uint32_t slot = 0; slot < catalog_count_; ++slot) {
                    if (catalog_[slot].state == CatalogState::Vacant) {
                        publication_slot = slot;
                        break;
                    }
                }
            }

            for (std::size_t row = 0; row < outcomes.size(); ++row) {
                const PressureOwnerOutcome& outcome = outcomes[row];
                if (outcome.disposition != VictimDisposition::Retained &&
                    outcome.disposition != VictimDisposition::Evicted) {
                    return std::nullopt;
                }
                if (std::find_if(outcomes.begin(), outcomes.begin() + row,
                                 [&](const PressureOwnerOutcome& prior) {
                                     return prior.owner == outcome.owner;
                                 }) != outcomes.begin() + row) {
                    return std::nullopt;
                }
                const auto record = std::find_if(
                    owner_records.begin(), owner_records.end(),
                    [&](const PlanningOwnerRecord& item) { return item.id == outcome.owner; });
                if (record == owner_records.end()) { return std::nullopt; }
                const bool shared = record->capability.owner.kind == LogicalOwnerKind::SharedPrefix;
                if (!shared) {
                    const std::uint32_t slot = record->capability.slot;
                    if (slot >= catalog_count_ ||
                        (candidate.private_source && slot == candidate.private_source->slot)) {
                        return std::nullopt;
                    }
                    const CatalogEntry& entry = catalog_[slot];
                    if (entry.state != CatalogState::Catalogued || !entry.handle ||
                        entry.id != record->capability.owner.id ||
                        entry.revision != record->capability.generation ||
                        private_has_active_edge(slot)) {
                        return std::nullopt;
                    }
                    if (publication_slot == kInvalidCatalogSlot &&
                        outcome.disposition == VictimDisposition::Evicted) {
                        publication_slot = slot;
                    }
                } else {
                    const std::uint32_t slot = record->capability.slot;
                    if (slot >= shared_catalog_count_ ||
                        (candidate.shared_source && slot == candidate.shared_source->slot)) {
                        return std::nullopt;
                    }
                    const SharedCatalogEntry& entry = shared_catalog_[slot];
                    if (entry.state != SharedCatalogState::Catalogued || !entry.handle ||
                        entry.id != record->capability.owner.id ||
                        entry.revision != record->capability.generation ||
                        entry.transaction_pins != 0 || shared_active_edge_count(slot) != 0) {
                        return std::nullopt;
                    }
                }
            }
            if (publication_slot == kInvalidCatalogSlot) { return std::nullopt; }
            return typename Planner::LogicalGoal{.publication_slot = publication_slot};
        };

        const auto final_schedule = [&](PlanningCandidateId candidate_id,
                                        const RequestPlanSummary& summary,
                                        const auto& split_cost) -> std::vector<std::uint32_t> {
            const auto selected = std::find_if(candidate_inputs.begin(), candidate_inputs.end(),
                                               [&](const typename Planner::CandidateInput& input) {
                                                   return input.id == candidate_id;
                                               });
            if (selected == candidate_inputs.end()) {
                throw std::logic_error("final schedule references an unknown candidate");
            }
            const Candidate& candidate =
                candidates[static_cast<std::size_t>(selected - candidate_inputs.begin())];
            return select_materialization_shared_captures(program, base, candidate, summary,
                                                          provisional_demand, split_cost);
        };

        std::optional<typename Planner::Result> planned =
            planner_.plan(program, prompt, cost_model_, candidate_inputs, 0, build_pressure_inputs,
                          logical_goal, final_schedule, planning_started);
        const auto selected_candidate =
            planned ? std::find_if(candidate_inputs.begin(), candidate_inputs.end(),
                                   [&](const typename Planner::CandidateInput& input) {
                                       return input.id == planned->candidate;
                                   })
                    : candidate_inputs.end();
        if (!planned || !planned->plan || selected_candidate == candidate_inputs.end()) {
            return std::nullopt;
        }

        Candidate& candidate =
            candidates[static_cast<std::size_t>(selected_candidate - candidate_inputs.begin())];

        Choice choice(destination, std::move(*planned->plan), catalog_count_,
                      base.context_cache().session_key, base.context_cache().retention,
                      base.context_cache().update_session_index, publication_order);
        choice.private_source_                 = candidate.private_source;
        choice.source_mode_                    = planned->source_mode;
        choice.shared_source_                  = candidate.shared_source;
        choice.publication_slot_               = planned->publication_slot;
        choice.selected_observation_           = candidate.selected_observation;
        choice.diagnostics_                    = planned->diagnostics;
        provisional_demand.selected_source_key = candidate.source_key;
        choice.demand_                         = std::move(provisional_demand);
        for (const PressureOwnerOutcome& outcome : planned->owner_outcomes) {
            const auto record = std::find_if(
                owner_records.begin(), owner_records.end(),
                [&](const PlanningOwnerRecord& item) { return item.id == outcome.owner; });
            if (record == owner_records.end()) {
                throw std::logic_error("selected pressure outcome has no logical owner record");
            }
            const bool shared = record->capability.owner.kind == LogicalOwnerKind::SharedPrefix;
            if (!shared) {
                const CatalogEntry& entry          = catalog_.at(record->capability.slot);
                std::vector<CheckpointRef> dropped = selected_checkpoint_drops(
                    outcome.owner, outcome.disposition, outcome.dropped_checkpoints,
                    planned->checkpoint_outcomes, continuation_checkpoint_count(entry.summary),
                    [&](CheckpointRef checkpoint) {
                        return continuation_contains_checkpoint(entry.summary, checkpoint);
                    });
                choice.private_claims_.push_back(OwnerClaim{
                    .planning_id         = outcome.owner,
                    .capability          = record->capability,
                    .disposition         = outcome.disposition,
                    .dropped_checkpoints = std::move(dropped),
                });
            } else {
                const SharedCatalogEntry& entry    = shared_catalog_.at(record->capability.slot);
                std::vector<CheckpointRef> dropped = selected_checkpoint_drops(
                    outcome.owner, outcome.disposition, outcome.dropped_checkpoints,
                    planned->checkpoint_outcomes, 1U, [&](CheckpointRef checkpoint) {
                        return checkpoint == entry.summary.checkpoint.ref;
                    });
                choice.shared_claims_.push_back(OwnerClaim{
                    .planning_id         = outcome.owner,
                    .capability          = record->capability,
                    .disposition         = outcome.disposition,
                    .dropped_checkpoints = std::move(dropped),
                });
            }
        }
        return choice;
    }

    void validate_choice(const Choice& choice, ProgramResourceRevision revision) const {
        if (!choice.plan_ || choice.destination_.value >= lane_count_ ||
            lanes_[choice.destination_.value] != LogicalLaneState::Free || revision.value == 0 ||
            choice.publication_slot_ >= catalog_count_ || choice.publication_order_ == 0) {
            throw std::logic_error("resource choice is stale or malformed");
        }
        if (choice.private_source_) {
            const CatalogCapability& capability = *choice.private_source_;
            if (capability.owner.kind != LogicalOwnerKind::PrivateContinuation ||
                capability.slot >= catalog_count_) {
                throw std::logic_error("private source slot is invalid");
            }
            const CatalogEntry& source = catalog_[capability.slot];
            if (source.state != CatalogState::Catalogued || !source.handle ||
                source.id != capability.owner.id || source.revision != capability.generation ||
                private_has_active_edge(capability.slot)) {
                throw std::logic_error("private source changed after planning");
            }
        }
        if (choice.shared_source_) {
            const CatalogCapability& capability = *choice.shared_source_;
            if (capability.owner.kind != LogicalOwnerKind::SharedPrefix ||
                capability.slot >= shared_catalog_count_) {
                throw std::logic_error("shared source slot is invalid");
            }
            const SharedCatalogEntry& source = shared_catalog_[capability.slot];
            if (source.state != SharedCatalogState::Catalogued || !source.handle ||
                source.id != capability.owner.id || source.revision != capability.generation) {
                throw std::logic_error("shared source changed after planning");
            }
        }
        for (const OwnerClaim& claim : choice.private_claims_) {
            const std::uint32_t slot = claim.capability.slot;
            if (slot >= catalog_count_ ||
                (choice.private_source_ && slot == choice.private_source_->slot)) {
                throw std::logic_error("private pressure owner is invalid");
            }
            const CatalogEntry& entry = catalog_[slot];
            if (entry.state != CatalogState::Catalogued || !entry.handle ||
                claim.capability.owner.kind != LogicalOwnerKind::PrivateContinuation ||
                entry.id != claim.capability.owner.id ||
                entry.revision != claim.capability.generation || private_has_active_edge(slot) ||
                (claim.disposition != VictimDisposition::Retained &&
                 claim.disposition != VictimDisposition::Evicted)) {
                throw std::logic_error("private pressure owner changed after planning");
            }
        }
        for (const OwnerClaim& claim : choice.shared_claims_) {
            const std::uint32_t slot = claim.capability.slot;
            if (slot >= shared_catalog_count_ ||
                (choice.shared_source_ && slot == choice.shared_source_->slot)) {
                throw std::logic_error("shared pressure owner is invalid");
            }
            const SharedCatalogEntry& entry = shared_catalog_[slot];
            if (entry.state != SharedCatalogState::Catalogued || !entry.handle ||
                claim.capability.owner.kind != LogicalOwnerKind::SharedPrefix ||
                entry.id != claim.capability.owner.id ||
                entry.revision != claim.capability.generation || entry.transaction_pins != 0 ||
                shared_active_edge_count(slot) != 0 ||
                (claim.disposition != VictimDisposition::Retained &&
                 claim.disposition != VictimDisposition::Evicted)) {
                throw std::logic_error("shared pressure owner changed after planning");
            }
        }
        const CatalogEntry& publication = catalog_[choice.publication_slot_];
        const bool source_cell          = choice.private_source_ &&
                                 choice.publication_slot_ == choice.private_source_->slot &&
                                 choice.source_mode_ == PrivateSourceMode::ConsumeToActive;
        const auto victim =
            std::find_if(choice.private_claims_.begin(), choice.private_claims_.end(),
                         [&](const OwnerClaim& claim) {
                             return claim.capability.slot == choice.publication_slot_;
                         });
        const bool victim_cell = victim != choice.private_claims_.end() &&
                                 victim->disposition == VictimDisposition::Evicted;
        if (publication.state != CatalogState::Vacant && !source_cell && !victim_cell) {
            throw std::logic_error("resource choice has no publication cell");
        }
    }

    [[nodiscard]] MaterializationRecord take_materialization_record(Choice& choice) {
        return MaterializationRecord{
            .destination          = choice.destination_,
            .private_source       = choice.private_source_,
            .source_mode          = choice.source_mode_,
            .shared_source        = choice.shared_source_,
            .publication_slot     = choice.publication_slot_,
            .private_claims       = std::move(choice.private_claims_),
            .shared_claims        = std::move(choice.shared_claims_),
            .selected_observation = choice.selected_observation_,
            .session              = std::move(choice.session_),
            .retention            = choice.retention_,
            .update_session_index = choice.update_session_index_,
            .publication_order    = choice.publication_order_,
            .diagnostics          = choice.diagnostics_,
            .demand               = std::move(choice.demand_),
        };
    }

    void reserve_logical_materialization(const MaterializationRecord& record) noexcept {
        lanes_[record.destination.value] = LogicalLaneState::Materializing;
        if (record.private_source) {
            catalog_[record.private_source->slot].state = CatalogState::Claimed;
        }
        if (record.shared_source) {
            ++shared_catalog_[record.shared_source->slot].transaction_pins;
        }
        for (const OwnerClaim& claim : record.private_claims) {
            catalog_[claim.capability.slot].state = CatalogState::Claimed;
        }
        for (const OwnerClaim& claim : record.shared_claims) {
            shared_catalog_[claim.capability.slot].state = SharedCatalogState::Claimed;
        }
        CatalogEntry& publication = catalog_[record.publication_slot];
        if (publication.state == CatalogState::Vacant) {
            publication.state = CatalogState::Claimed;
        }
    }

    void rollback_logical_materialization(const MaterializationRecord& record) noexcept {
        lanes_[record.destination.value] = LogicalLaneState::Free;
        if (record.private_source) {
            catalog_[record.private_source->slot].state = CatalogState::Catalogued;
        }
        if (record.shared_source) {
            SharedCatalogEntry& source = shared_catalog_[record.shared_source->slot];
            if (source.transaction_pins != 0) { --source.transaction_pins; }
        }
        for (const OwnerClaim& claim : record.private_claims) {
            catalog_[claim.capability.slot].state = CatalogState::Catalogued;
        }
        for (const OwnerClaim& claim : record.shared_claims) {
            shared_catalog_[claim.capability.slot].state = SharedCatalogState::Catalogued;
        }
        CatalogEntry& publication = catalog_[record.publication_slot];
        if (publication.id == 0 && !publication.handle) {
            publication.state = CatalogState::Vacant;
        }
    }

    void reserve_logical_active_capture(const ActiveCaptureRecord& record) noexcept {
        for (const OwnerClaim& claim : record.private_claims) {
            catalog_[claim.capability.slot].state = CatalogState::Claimed;
        }
        for (const OwnerClaim& claim : record.shared_claims) {
            shared_catalog_[claim.capability.slot].state = SharedCatalogState::Claimed;
        }
        shared_catalog_[record.publication_slot].state = SharedCatalogState::ReservedCapture;
    }

    void rollback_logical_active_capture(const ActiveCaptureRecord& record) noexcept {
        for (const OwnerClaim& claim : record.private_claims) {
            catalog_[claim.capability.slot].state = CatalogState::Catalogued;
        }
        for (const OwnerClaim& claim : record.shared_claims) {
            shared_catalog_[claim.capability.slot].state = SharedCatalogState::Catalogued;
        }
        shared_catalog_[record.publication_slot].state = record.replacement_id == 0
                                                             ? SharedCatalogState::Vacant
                                                             : SharedCatalogState::Catalogued;
    }

    void observe_selected_hit(const MaterializationRecord& record) noexcept {
        if (record.selected_observation) {
            RetentionObservation* selected = resolve_observation(*record.selected_observation);
            if (selected) {
                saturating_increment(selected->selected_hit_count);
                selected->last_hit_epoch = ++retention_epoch_;
            }
        }
    }

    void commit_demand(PrefixDemandRecord&& demand) noexcept {
        if (demand_window_.capacity() < kDemandWindowCapacity) { std::terminate(); }
        if (demand_window_.size() == kDemandWindowCapacity) {
            demand_window_.erase(demand_window_.begin());
        }
        demand_window_.push_back(std::move(demand));
        saturating_increment(demand_epoch_);
        const PrefixDemandRecord& committed = demand_window_.back();
        for (SharedCatalogEntry& entry : shared_catalog_) {
            if (entry.state != SharedCatalogState::Catalogued || !entry.handle ||
                !entry.explicit_credit) {
                continue;
            }
            if (std::find(committed.exact_resident_keys.begin(),
                          committed.exact_resident_keys.end(),
                          entry.summary.checkpoint.shortlist_key) !=
                committed.exact_resident_keys.end()) {
                entry.explicit_credit     = false;
                entry.credit_expiry_epoch = 0;
                continue;
            }
            if (demand_epoch_ >= entry.credit_expiry_epoch) {
                entry.explicit_credit     = false;
                entry.credit_expiry_epoch = 0;
            }
        }
    }

    void observe_planner_diagnostics(const MaterializationDiagnostics& diagnostics) noexcept {
        if (diagnostics.stop_reason != MaterializationStopReason::NoPressure) {
            saturating_increment(context_stats_.pressure_searches);
        }
        if (diagnostics.budget_exhausted) {
            saturating_increment(context_stats_.pressure_search_budget_exhaustions);
        }
        if (diagnostics.selected_maximal_fallback) {
            saturating_increment(context_stats_.pressure_maximal_fallback_selections);
        }
    }

    [[nodiscard]] RetentionObservation*
    resolve_observation(const PolicyObservationKey& key) noexcept {
        if (!key.shared) {
            if (key.slot >= catalog_count_) { return nullptr; }
            CatalogEntry& entry = catalog_[key.slot];
            if (entry.id != key.owner_id || entry.revision != key.revision) { return nullptr; }
            return find_observation(entry.observations, key.checkpoint);
        }
        if (key.slot >= shared_catalog_count_) { return nullptr; }
        SharedCatalogEntry& entry = shared_catalog_[key.slot];
        if (entry.id != key.owner_id || entry.revision != key.revision ||
            entry.summary.checkpoint.ref != key.checkpoint) {
            return nullptr;
        }
        return &entry.observation;
    }

    [[nodiscard]] static bool continuation_contains_checkpoint(const ContinuationSummary& summary,
                                                               CheckpointRef checkpoint) noexcept {
        if (summary.endpoint && summary.endpoint->ref == checkpoint) { return true; }
        if (summary.rewrite && summary.rewrite->ref == checkpoint) { return true; }
        return std::any_of(summary.long_anchors.begin(), summary.long_anchors.end(),
                           [&](const auto& anchor) { return anchor.ref == checkpoint; });
    }

    [[nodiscard]] static std::uint32_t
    continuation_checkpoint_count(const ContinuationSummary& summary) noexcept {
        const std::size_t count = static_cast<std::size_t>(summary.endpoint.has_value()) +
                                  static_cast<std::size_t>(summary.rewrite.has_value()) +
                                  summary.long_anchors.size();
        return count > std::numeric_limits<std::uint32_t>::max()
                   ? std::numeric_limits<std::uint32_t>::max()
                   : static_cast<std::uint32_t>(count);
    }

    template <class ContainsCheckpoint>
    [[nodiscard]] static std::vector<CheckpointRef> selected_checkpoint_drops(
        PlanningOwnerId owner, VictimDisposition disposition, std::uint32_t expected_drop_count,
        std::span<const PressureCheckpointOutcome> outcomes, std::uint32_t checkpoint_count,
        ContainsCheckpoint&& contains_checkpoint) {
        std::vector<CheckpointRef> dropped;
        dropped.reserve(expected_drop_count);
        std::uint32_t observed = 0;
        for (std::size_t index = 0; index < outcomes.size(); ++index) {
            const PressureCheckpointOutcome& outcome = outcomes[index];
            if (outcome.owner != owner) { continue; }
            if (!contains_checkpoint(outcome.checkpoint) ||
                std::find_if(outcomes.begin(), outcomes.begin() + index,
                             [&](const PressureCheckpointOutcome& prior) {
                                 return prior.owner == owner &&
                                        prior.checkpoint == outcome.checkpoint;
                             }) != outcomes.begin() + index) {
                throw std::logic_error("selected checkpoint outcome is unknown or duplicated");
            }
            ++observed;
            if (!outcome.survives) { dropped.push_back(outcome.checkpoint); }
        }
        if (observed != checkpoint_count || dropped.size() != expected_drop_count ||
            (disposition == VictimDisposition::Evicted) != (dropped.size() == checkpoint_count)) {
            throw std::logic_error("selected checkpoint outcome is incomplete");
        }
        return dropped;
    }

    [[nodiscard]] static bool continuation_matches_claim(
        const ContinuationSummary& before, const std::optional<ContinuationSummary>& after,
        VictimDisposition disposition, std::span<const CheckpointRef> expected_drops) noexcept {
        const std::uint32_t before_count = continuation_checkpoint_count(before);
        if (disposition == VictimDisposition::Evicted) {
            return !after && expected_drops.size() == before_count;
        }
        if (disposition != VictimDisposition::Retained || !after ||
            continuation_checkpoint_count(*after) + expected_drops.size() != before_count) {
            return false;
        }
        const auto expected_drop = [&](CheckpointRef checkpoint) {
            return std::find(expected_drops.begin(), expected_drops.end(), checkpoint) !=
                   expected_drops.end();
        };
        const auto check = [&](CheckpointRef checkpoint) {
            return continuation_contains_checkpoint(*after, checkpoint) !=
                   expected_drop(checkpoint);
        };
        if ((before.endpoint && !check(before.endpoint->ref)) ||
            (before.rewrite && !check(before.rewrite->ref))) {
            return false;
        }
        for (const auto& anchor : before.long_anchors) {
            if (!check(anchor.ref)) { return false; }
        }
        return std::all_of(expected_drops.begin(), expected_drops.end(), [&](CheckpointRef drop) {
            return continuation_contains_checkpoint(before, drop);
        });
    }

    [[nodiscard]] static std::uint32_t
    dropped_checkpoint_count(const ContinuationSummary& before,
                             const std::optional<ContinuationSummary>& after,
                             VictimDisposition disposition) noexcept {
        if (disposition == VictimDisposition::Evicted) {
            return continuation_checkpoint_count(before);
        }
        if (!after) { return 0; }
        std::uint32_t dropped = 0;
        const auto visit      = [&](const auto& checkpoint) {
            if (checkpoint && !continuation_contains_checkpoint(*after, checkpoint->ref)) {
                ++dropped;
            }
        };
        visit(before.endpoint);
        visit(before.rewrite);
        for (const auto& anchor : before.long_anchors) {
            if (!continuation_contains_checkpoint(*after, anchor.ref)) { ++dropped; }
        }
        return dropped;
    }

    static void record_checkpoint_drops(RuntimeStats& stats, std::uint32_t count) noexcept {
        for (std::uint32_t index = 0; index < count; ++index) {
            saturating_increment(stats.pressure_checkpoints_dropped);
        }
    }

    template <class Result>
    void validate_private_action(const OwnerClaim& claim, bool target_committed,
                                 const Result& result) const {
        const std::uint32_t slot = claim.capability.slot;
        if (slot >= catalog_count_) {
            throw std::logic_error("private action result has an invalid slot");
        }
        const CatalogEntry& entry = catalog_[slot];
        if (claim.capability.owner.kind != LogicalOwnerKind::PrivateContinuation ||
            entry.state != CatalogState::Claimed || entry.id != claim.capability.owner.id ||
            entry.revision != claim.capability.generation || !entry.handle) {
            throw std::logic_error("private action owner changed before adoption");
        }
        if (result.final_summary && !valid_continuation_summary(*result.final_summary)) {
            throw std::logic_error("private action returned an invalid final summary");
        }
        const std::uint32_t dropped =
            dropped_checkpoint_count(entry.summary, result.final_summary, result.disposition);
        if (target_committed && !result.pressure_committed) {
            throw std::logic_error("selected private pressure action was not committed");
        }
        if (result.pressure_committed &&
            (result.disposition != claim.disposition ||
             !continuation_matches_claim(entry.summary, result.final_summary, result.disposition,
                                         claim.dropped_checkpoints))) {
            throw std::logic_error("private owner outcome differs from the selected target");
        }
        if (result.disposition == VictimDisposition::Evicted) {
            if (!result.pressure_committed || result.final_summary) {
                throw std::logic_error("private eviction result is malformed");
            }
            return;
        }
        if (result.disposition != VictimDisposition::Retained) {
            throw std::logic_error("private pressure action returned an invalid disposition");
        }
        if (!result.pressure_committed &&
            (result.disposition != VictimDisposition::Retained || dropped != 0 ||
             (result.final_summary &&
              !continuation_matches_claim(entry.summary, result.final_summary,
                                          VictimDisposition::Retained, {})))) {
            throw std::logic_error("uncommitted private pressure action changed checkpoints");
        }
        if (result.pressure_committed && !result.final_summary) {
            throw std::logic_error("committed private pressure action has no final summary");
        }
    }

    template <class Result>
    void validate_shared_action(const OwnerClaim& claim, bool target_committed,
                                const Result& result) const {
        const std::uint32_t slot = claim.capability.slot;
        if (slot >= shared_catalog_count_) {
            throw std::logic_error("shared action result has an invalid slot");
        }
        const SharedCatalogEntry& entry = shared_catalog_[slot];
        if (claim.capability.owner.kind != LogicalOwnerKind::SharedPrefix ||
            entry.state != SharedCatalogState::Claimed || entry.id != claim.capability.owner.id ||
            entry.revision != claim.capability.generation || !entry.handle) {
            throw std::logic_error("shared action owner changed before adoption");
        }
        if (result.final_summary && !valid_shared_prefix_summary(*result.final_summary)) {
            throw std::logic_error("shared action returned an invalid final summary");
        }
        if (target_committed && !result.pressure_committed) {
            throw std::logic_error("selected shared pressure action was not committed");
        }
        const bool exact_committed_outcome =
            result.disposition == VictimDisposition::Evicted
                ? claim.dropped_checkpoints.size() == 1U &&
                      claim.dropped_checkpoints.front() == entry.summary.checkpoint.ref &&
                      !result.final_summary
                : result.disposition == VictimDisposition::Retained &&
                      claim.dropped_checkpoints.empty() && result.final_summary &&
                      result.final_summary->checkpoint.ref == entry.summary.checkpoint.ref;
        if (result.pressure_committed &&
            (result.disposition != claim.disposition || !exact_committed_outcome)) {
            throw std::logic_error("shared owner outcome differs from the selected target");
        }
        if (result.disposition == VictimDisposition::Evicted) {
            if (!result.pressure_committed || result.final_summary) {
                throw std::logic_error("shared eviction result is malformed");
            }
            return;
        }
        if (result.disposition != VictimDisposition::Retained) {
            throw std::logic_error("shared pressure action returned an invalid disposition");
        }
        if (!result.pressure_committed && result.final_summary &&
            result.final_summary->checkpoint.ref != entry.summary.checkpoint.ref) {
            throw std::logic_error(
                "uncommitted shared pressure action changed checkpoint identity");
        }
        if (result.pressure_committed && !result.final_summary) {
            throw std::logic_error("committed shared pressure action has no final summary");
        }
    }

    template <class Result>
    void apply_private_action(const OwnerClaim& claim, bool target_committed,
                              const Result& result) noexcept {
        (void)target_committed;
        const std::uint32_t slot = claim.capability.slot;
        CatalogEntry& entry      = catalog_[slot];
        const std::uint32_t dropped =
            dropped_checkpoint_count(entry.summary, result.final_summary, result.disposition);
        if (result.disposition == VictimDisposition::Evicted) {
            erase_session_if_owner(claim.capability.owner.id);
            clear_catalog_entry(entry);
            saturating_increment(context_stats_.pressure_private_owners_evicted);
            record_checkpoint_drops(context_stats_, dropped);
            return;
        }
        if (!result.pressure_committed) {
            entry.state = CatalogState::Catalogued;
            return;
        }
        assign_continuation_summary(entry.summary, *result.final_summary);
        migrate_observations(entry, *result.final_summary, entry.retention);
        advance_revision(entry.revision);
        refresh_session_owner_revision(claim.capability.owner.id, slot, entry.revision);
        saturating_increment(context_stats_.pressure_private_owners_degraded);
        record_checkpoint_drops(context_stats_, dropped);
        entry.state = CatalogState::Catalogued;
    }

    template <class Result>
    void apply_shared_action(const OwnerClaim& claim, bool target_committed,
                             const Result& result) noexcept {
        (void)target_committed;
        const std::uint32_t slot    = claim.capability.slot;
        SharedCatalogEntry& entry   = shared_catalog_[slot];
        const std::uint32_t dropped = result.disposition == VictimDisposition::Evicted ? 1U : 0U;
        if (result.disposition == VictimDisposition::Evicted) {
            clear_shared_entry(entry);
            saturating_increment(context_stats_.pressure_shared_owners_evicted);
            record_checkpoint_drops(context_stats_, dropped);
            return;
        }
        if (!result.pressure_committed) {
            entry.state = SharedCatalogState::Catalogued;
            return;
        }
        entry.summary = *result.final_summary;
        advance_revision(entry.revision);
        saturating_increment(context_stats_.pressure_shared_owners_degraded);
        record_checkpoint_drops(context_stats_, dropped);
        entry.state = SharedCatalogState::Catalogued;
    }

    void restore_unreported_materialization(const MaterializationRecord& record) noexcept {
        if (record.private_source &&
            catalog_[record.private_source->slot].state == CatalogState::Claimed) {
            catalog_[record.private_source->slot].state = CatalogState::Catalogued;
        }
        if (record.shared_source) {
            SharedCatalogEntry& source = shared_catalog_[record.shared_source->slot];
            if (source.transaction_pins != 0) { --source.transaction_pins; }
        }
        for (const OwnerClaim& claim : record.private_claims) {
            CatalogEntry& entry = catalog_[claim.capability.slot];
            if (entry.state == CatalogState::Claimed) { entry.state = CatalogState::Catalogued; }
        }
        for (const OwnerClaim& claim : record.shared_claims) {
            SharedCatalogEntry& entry = shared_catalog_[claim.capability.slot];
            if (entry.state == SharedCatalogState::Claimed) {
                entry.state = SharedCatalogState::Catalogued;
            }
        }
        CatalogEntry& publication = catalog_[record.publication_slot];
        if (publication.state == CatalogState::Claimed && publication.id == 0 &&
            !publication.handle) {
            publication.state = CatalogState::Vacant;
        }
    }

    [[nodiscard]] MaterializationOutcome
    adopt_materialization_progress(Program& program, ProgramMaterializationResult&& result) {
        MaterializationRecord* record = std::get_if<MaterializationRecord>(&transaction_);
        if (record == nullptr || !program.has_context_transaction()) {
            throw std::logic_error("materialization result has no logical transaction");
        }
        if (result.status == ContextTransactionStatus::InProgress) {
            throw std::logic_error("terminal materialization result is marked in progress");
        }
        if (result.victims.size() != record->private_claims.size() ||
            result.shared_victims.size() != record->shared_claims.size()) {
            throw std::logic_error("materialization result is not action aligned");
        }
        const auto private_result_for = [&](const OwnerClaim& claim) -> const auto& {
            const auto found =
                std::find_if(result.victims.begin(), result.victims.end(),
                             [&](const auto& row) { return row.owner == claim.planning_id; });
            if (found == result.victims.end() ||
                std::find_if(found + 1, result.victims.end(), [&](const auto& row) {
                    return row.owner == claim.planning_id;
                }) != result.victims.end()) {
                throw std::logic_error("materialization private result ID is not unique");
            }
            return *found;
        };
        const auto shared_result_for = [&](const OwnerClaim& claim) -> const auto& {
            const auto found =
                std::find_if(result.shared_victims.begin(), result.shared_victims.end(),
                             [&](const auto& row) { return row.owner == claim.planning_id; });
            if (found == result.shared_victims.end() ||
                std::find_if(found + 1, result.shared_victims.end(), [&](const auto& row) {
                    return row.owner == claim.planning_id;
                }) != result.shared_victims.end()) {
                throw std::logic_error("materialization shared result ID is not unique");
            }
            return *found;
        };

        const bool published = result.status == ContextTransactionStatus::Published;
        if ((!published && result.status != ContextTransactionStatus::Aborted) ||
            published != result.published.has_value()) {
            throw std::logic_error("materialization terminal status is invalid");
        }
        if (record->destination.value >= lane_count_ ||
            lanes_[record->destination.value] != LogicalLaneState::Materializing ||
            active_[record->destination.value].occupied ||
            active_[record->destination.value].retained_private_source ||
            !active_[record->destination.value].shared_sources.empty() ||
            active_[record->destination.value].shared_sources.capacity() < shared_catalog_count_ ||
            record->publication_slot >= catalog_count_ ||
            catalog_[record->publication_slot].state != CatalogState::Claimed) {
            throw std::logic_error("materialization logical destination changed before adoption");
        }
        for (std::size_t row = 0; row < record->private_claims.size(); ++row) {
            const OwnerClaim& claim = record->private_claims[row];
            if ((record->private_source && claim.capability.slot == record->private_source->slot) ||
                std::find_if(record->private_claims.begin(),
                             record->private_claims.begin() + static_cast<std::ptrdiff_t>(row),
                             [&](const OwnerClaim& prior) {
                                 return prior.planning_id == claim.planning_id ||
                                        prior.capability == claim.capability;
                             }) !=
                    record->private_claims.begin() + static_cast<std::ptrdiff_t>(row)) {
                throw std::logic_error("materialization private claim manifest is not unique");
            }
        }
        for (std::size_t row = 0; row < record->shared_claims.size(); ++row) {
            const OwnerClaim& claim = record->shared_claims[row];
            if (std::any_of(record->private_claims.begin(), record->private_claims.end(),
                            [&](const OwnerClaim& prior) {
                                return prior.planning_id == claim.planning_id;
                            })) {
                throw std::logic_error("materialization owner ID changes kind");
            }
            if ((record->shared_source && claim.capability.slot == record->shared_source->slot) ||
                std::find_if(record->shared_claims.begin(),
                             record->shared_claims.begin() + static_cast<std::ptrdiff_t>(row),
                             [&](const OwnerClaim& prior) {
                                 return prior.planning_id == claim.planning_id ||
                                        prior.capability == claim.capability;
                             }) !=
                    record->shared_claims.begin() + static_cast<std::ptrdiff_t>(row)) {
                throw std::logic_error("materialization shared claim manifest is not unique");
            }
        }
        for (const OwnerClaim& claim : record->private_claims) {
            validate_private_action(claim, published, private_result_for(claim));
        }
        for (const OwnerClaim& claim : record->shared_claims) {
            validate_shared_action(claim, published, shared_result_for(claim));
        }

        if (record->private_source) {
            const CatalogCapability& capability = *record->private_source;
            if (capability.owner.kind != LogicalOwnerKind::PrivateContinuation ||
                capability.slot >= catalog_count_) {
                throw std::logic_error("materialization private source capability is malformed");
            }
            const CatalogEntry& source = catalog_[capability.slot];
            if (!result.source || source.state != CatalogState::Claimed ||
                source.id != capability.owner.id || source.revision != capability.generation ||
                (!published && result.source->mode != PrivateSourceMode::Retain) ||
                (published && result.source->mode != record->source_mode) ||
                (result.source->final_summary &&
                 !valid_continuation_summary(*result.source->final_summary)) ||
                (result.source->mode == PrivateSourceMode::ConsumeToActive &&
                 result.source->final_summary) ||
                (published && result.source->mode == PrivateSourceMode::Retain &&
                 private_has_active_edge(capability.slot))) {
                throw std::logic_error("materialization private source result is invalid");
            }
        } else if (result.source) {
            throw std::logic_error("root materialization returned a private source result");
        }

        if (record->shared_source) {
            const CatalogCapability& capability = *record->shared_source;
            if (capability.owner.kind != LogicalOwnerKind::SharedPrefix ||
                capability.slot >= shared_catalog_count_) {
                throw std::logic_error("materialization shared source capability is malformed");
            }
            const SharedCatalogEntry& source       = shared_catalog_[capability.slot];
            const std::uint32_t current_references = shared_active_edge_count(capability.slot);
            if (!result.shared_source || source.transaction_pins == 0 ||
                source.id != capability.owner.id || source.revision != capability.generation ||
                current_references == std::numeric_limits<std::uint32_t>::max()) {
                throw std::logic_error("materialization shared source result is invalid");
            }
            const std::uint32_t expected_references =
                published ? current_references + 1U : current_references;
            if (result.shared_source->final_summary) {
                const SharedPrefixSummary& final = *result.shared_source->final_summary;
                SharedPrefixSummary expected     = source.summary;
                expected.active_references       = expected_references;
                // Source reuse may restore a replica, but it cannot rewrite the checkpoint
                // identity or its recovery contract.
                expected.checkpoint.state_residency = final.checkpoint.state_residency;
                if (!valid_shared_prefix_summary(final) || final != expected) {
                    throw std::logic_error("materialization shared source summary is invalid");
                }
            }
        } else if (result.shared_source) {
            throw std::logic_error("materialization returned an unexpected shared source result");
        }

        if (published) {
            const CatalogEntry& publication = catalog_[record->publication_slot];
            bool publication_released       = !publication.handle && publication.id == 0;
            publication_released =
                publication_released ||
                (record->private_source &&
                 record->publication_slot == record->private_source->slot && result.source &&
                 result.source->mode == PrivateSourceMode::ConsumeToActive);
            for (const OwnerClaim& claim : record->private_claims) {
                if (publication_released || claim.capability.slot != record->publication_slot) {
                    continue;
                }
                const auto& victim = private_result_for(claim);
                publication_released =
                    victim.disposition == VictimDisposition::Evicted && victim.pressure_committed;
            }
            if (!publication_released) {
                throw std::logic_error(
                    "materialization result cannot release its publication cell");
            }
        }

        if (published) { observe_selected_hit(*record); }
        for (const OwnerClaim& claim : record->private_claims) {
            apply_private_action(claim, published, private_result_for(claim));
        }
        for (const OwnerClaim& claim : record->shared_claims) {
            apply_shared_action(claim, published, shared_result_for(claim));
        }

        bool retained_private_source = false;
        if (record->private_source) {
            const CatalogCapability capability = *record->private_source;
            CatalogEntry& source               = catalog_[capability.slot];
            if (result.source->mode == PrivateSourceMode::Retain) {
                if (result.source->final_summary) {
                    assign_continuation_summary(source.summary, *result.source->final_summary);
                    migrate_observations(source, *result.source->final_summary, source.retention);
                    advance_revision(source.revision);
                    refresh_session_owner_revision(capability.owner.id, capability.slot,
                                                   source.revision);
                }
                source.state            = CatalogState::Catalogued;
                retained_private_source = result.status == ContextTransactionStatus::Published;
            } else if (result.source->mode == PrivateSourceMode::ConsumeToActive) {
                erase_session_if_owner(source.id);
                source.handle.reset();
                source.summary.endpoint.reset();
                source.summary.rewrite.reset();
                source.summary.long_anchors.clear();
                source.observations.clear();
                source.session.reset();
            }
        }

        if (record->shared_source) {
            SharedCatalogEntry& source = shared_catalog_[record->shared_source->slot];
            --source.transaction_pins;
            if (result.shared_source->final_summary) {
                SharedPrefixSummary updated = *result.shared_source->final_summary;
                updated.active_references   = 0;
                if (updated != source.summary) {
                    source.summary = std::move(updated);
                    advance_revision(source.revision);
                }
            }
        }

        observe_transfers(result);
        observe_operations(result);

        if (result.status == ContextTransactionStatus::Aborted) {
            restore_unreported_materialization(*record);
            lanes_[record->destination.value] = LogicalLaneState::Free;
            transaction_.template emplace<std::monostate>();
            program.finalize_context_transaction();
            return {.status = ContextTransactionStatus::Aborted};
        }

        CatalogEntry& publication = catalog_[record->publication_slot];
        publication.state         = CatalogState::ReservedForActive;
        publication.id            = next_continuation_id_++;
        publication.session       = record->session;
        publication.retention     = record->retention;
        advance_revision(publication.revision);
        if (publication.id == 0) { publication.id = next_continuation_id_++; }

        ActiveEntry& active         = active_[record->destination.value];
        active.occupied             = true;
        active.publication_slot     = record->publication_slot;
        active.continuation_id      = publication.id;
        active.session              = record->session;
        active.retention            = record->retention;
        active.update_session_index = record->update_session_index;
        active.publication_order    = record->publication_order;
        if (retained_private_source) {
            active.retained_private_source =
                active_edge(private_capability(record->private_source->slot));
        }
        if (record->shared_source) {
            active.shared_sources.push_back(
                active_edge(shared_capability(record->shared_source->slot)));
        }
        StartResult start = std::move(*result.published);
        result.published.reset();
        commit_demand(std::move(record->demand));
        return MaterializationOutcome{
            .status      = ContextTransactionStatus::Published,
            .activation  = PublishedActivation(*this, std::move(start), record->destination),
            .diagnostics = record->diagnostics,
        };
    }

    [[nodiscard]] ActiveCaptureOutcome
    adopt_active_capture_progress(Program& program, ProgramActiveCaptureResult&& result) {
        ActiveCaptureRecord* record = std::get_if<ActiveCaptureRecord>(&transaction_);
        if (record == nullptr || !program.has_context_transaction()) {
            throw std::logic_error("active capture result has no logical transaction");
        }
        if (result.status == ContextTransactionStatus::InProgress) {
            throw std::logic_error("terminal capture result is marked in progress");
        }
        if (result.victims.size() != record->private_claims.size() ||
            result.shared_victims.size() != record->shared_claims.size()) {
            throw std::logic_error("active capture result is not pressure-action aligned");
        }
        const auto private_result_for = [&](const OwnerClaim& claim) -> const auto& {
            const auto found =
                std::find_if(result.victims.begin(), result.victims.end(),
                             [&](const auto& row) { return row.owner == claim.planning_id; });
            if (found == result.victims.end() ||
                std::find_if(found + 1, result.victims.end(), [&](const auto& row) {
                    return row.owner == claim.planning_id;
                }) != result.victims.end()) {
                throw std::logic_error("active capture private result ID is not unique");
            }
            return *found;
        };
        const auto shared_result_for = [&](const OwnerClaim& claim) -> const auto& {
            const auto found =
                std::find_if(result.shared_victims.begin(), result.shared_victims.end(),
                             [&](const auto& row) { return row.owner == claim.planning_id; });
            if (found == result.shared_victims.end() ||
                std::find_if(found + 1, result.shared_victims.end(), [&](const auto& row) {
                    return row.owner == claim.planning_id;
                }) != result.shared_victims.end()) {
                throw std::logic_error("active capture shared result ID is not unique");
            }
            return *found;
        };
        const bool published = result.status == ContextTransactionStatus::Published;
        if (!published && result.status != ContextTransactionStatus::Aborted) {
            throw std::logic_error("active capture terminal status is invalid");
        }
        for (std::size_t row = 0; row < record->private_claims.size(); ++row) {
            const OwnerClaim& claim = record->private_claims[row];
            if (std::find_if(record->private_claims.begin(),
                             record->private_claims.begin() + static_cast<std::ptrdiff_t>(row),
                             [&](const OwnerClaim& prior) {
                                 return prior.planning_id == claim.planning_id ||
                                        prior.capability == claim.capability;
                             }) !=
                record->private_claims.begin() + static_cast<std::ptrdiff_t>(row)) {
                throw std::logic_error("active capture private claim manifest is not unique");
            }
        }
        for (std::size_t row = 0; row < record->shared_claims.size(); ++row) {
            const OwnerClaim& claim = record->shared_claims[row];
            if (std::any_of(record->private_claims.begin(), record->private_claims.end(),
                            [&](const OwnerClaim& prior) {
                                return prior.planning_id == claim.planning_id;
                            })) {
                throw std::logic_error("active capture owner ID changes kind");
            }
            if (claim.capability.slot == record->publication_slot ||
                std::find_if(record->shared_claims.begin(),
                             record->shared_claims.begin() + static_cast<std::ptrdiff_t>(row),
                             [&](const OwnerClaim& prior) {
                                 return prior.planning_id == claim.planning_id ||
                                        prior.capability == claim.capability;
                             }) !=
                    record->shared_claims.begin() + static_cast<std::ptrdiff_t>(row)) {
                throw std::logic_error("active capture shared claim manifest is not unique");
            }
        }
        for (const OwnerClaim& claim : record->private_claims) {
            validate_private_action(claim, published, private_result_for(claim));
        }
        for (const OwnerClaim& claim : record->shared_claims) {
            validate_shared_action(claim, published, shared_result_for(claim));
        }
        if (record->lane.value >= lane_count_ || !active_[record->lane.value].occupied ||
            lanes_[record->lane.value] != LogicalLaneState::Active) {
            throw std::logic_error("active capture owner left its lane");
        }
        if (record->publishes_shared) {
            if (record->publication_slot >= shared_catalog_count_) {
                throw std::logic_error("active capture shared publication slot is invalid");
            }
            const SharedCatalogEntry& publication = shared_catalog_[record->publication_slot];
            if (publication.state != SharedCatalogState::ReservedCapture ||
                shared_active_edge_count(record->publication_slot) != 0 ||
                (record->replacement_id == 0 && (publication.id != 0 || publication.handle ||
                                                 result.capacity_preparation_committed)) ||
                (record->replacement_id != 0 &&
                 (publication.id != record->replacement_id ||
                  publication.revision != record->replacement_revision || !publication.handle ||
                  (published && !result.capacity_preparation_committed)))) {
                throw std::logic_error("active capture publication changed before adoption");
            }
        } else if (record->publication_slot != kInvalidCatalogSlot || record->replacement_id != 0 ||
                   result.capacity_preparation_committed) {
            throw std::logic_error("private capture has shared publication state");
        }
        if (!published) {
            if (result.shared) {
                throw std::logic_error("aborted active capture published a shared prefix");
            }
        } else {
            const ActiveEntry& active = active_[record->lane.value];
            if (record->publishes_private) {
                if (!valid_continuation_summary(result.active_summary)) {
                    throw std::logic_error("active capture returned an invalid private summary");
                }
                const CatalogEntry& publication = catalog_.at(active.publication_slot);
                if (publication.state != CatalogState::ReservedForActive ||
                    publication.id != active.continuation_id) {
                    throw std::logic_error("active private publication changed during capture");
                }
            }
            if (record->publishes_shared) {
                if (!result.shared ||
                    active.shared_sources.size() == active.shared_sources.capacity() ||
                    !valid_shared_prefix_summary(result.shared->summary) ||
                    result.shared->summary.active_references != 1) {
                    throw std::logic_error("active capture returned an invalid shared publication");
                }
            } else if (result.shared) {
                throw std::logic_error("private capture returned an unexpected shared publication");
            }
        }
        for (const OwnerClaim& claim : record->private_claims) {
            apply_private_action(claim, published, private_result_for(claim));
        }
        for (const OwnerClaim& claim : record->shared_claims) {
            apply_shared_action(claim, published, shared_result_for(claim));
        }
        if (result.status == ContextTransactionStatus::Aborted) {
            if (record->publication_slot != kInvalidCatalogSlot) {
                SharedCatalogEntry& publication = shared_catalog_[record->publication_slot];
                if (record->replacement_id != 0 && result.capacity_preparation_committed) {
                    clear_shared_entry(publication);
                } else if (record->replacement_id != 0) {
                    publication.state = SharedCatalogState::Catalogued;
                } else {
                    clear_shared_entry(publication);
                }
            }
            observe_transfers(result);
            observe_operations(result);
            transaction_.template emplace<std::monostate>();
            program.finalize_context_transaction();
            return {.status = ContextTransactionStatus::Aborted};
        }
        ActiveEntry& active = active_[record->lane.value];
        if (record->publishes_private) {
            CatalogEntry& publication = catalog_[active.publication_slot];
            assign_continuation_summary(publication.summary, result.active_summary);
            migrate_observations(publication, result.active_summary, active.retention);
            advance_revision(publication.revision);
        }
        if (record->publishes_shared) {
            SharedCatalogEntry& publication = shared_catalog_[record->publication_slot];
            publication.handle.reset();
            publication.state = SharedCatalogState::Catalogued;
            publication.id    = next_shared_prefix_id_++;
            if (publication.id == 0) { publication.id = next_shared_prefix_id_++; }
            publication.summary                   = result.shared->summary;
            publication.summary.active_references = 0;
            publication.handle.emplace(std::move(result.shared->handle));
            publication.observation =
                RetentionObservation{.retention_class = RetentionClass::SharedStable};
            publication.transaction_pins = 0;
            publication.explicit_credit =
                has_shared_candidate_evidence(record->shared_evidence,
                                              SharedCandidateEvidence::ExplicitBoundary) ||
                has_shared_candidate_evidence(record->shared_evidence,
                                              SharedCandidateEvidence::RequestedAutomatic);
            publication.credit_expiry_epoch =
                publication.explicit_credit
                    ? (demand_epoch_ >
                               std::numeric_limits<std::uint64_t>::max() - kDemandWindowCapacity
                           ? std::numeric_limits<std::uint64_t>::max()
                           : demand_epoch_ + kDemandWindowCapacity)
                    : 0;
            advance_revision(publication.revision);
            active.shared_sources.push_back(
                active_edge(shared_capability(record->publication_slot)));
        }
        observe_transfers(result);
        observe_operations(result);
        transaction_.template emplace<std::monostate>();
        program.finalize_context_transaction();
        return {.status = ContextTransactionStatus::Published};
    }

    void release_active_references(LaneId lane) {
        ActiveEntry& active = active_[lane.value];
        if (active.retained_private_source) {
            const ActiveOwnerEdge& edge = *active.retained_private_source;
            if (edge.owner.kind != LogicalOwnerKind::PrivateContinuation ||
                edge.slot >= catalog_count_) {
                throw std::logic_error("retained private source edge is malformed");
            }
            const CatalogEntry& source = catalog_[edge.slot];
            if (source.state != CatalogState::Catalogued || !source.handle ||
                source.id != edge.owner.id) {
                throw std::logic_error("retained private source edge is stale");
            }
        }
        for (std::size_t index = 0; index < active.shared_sources.size(); ++index) {
            const ActiveOwnerEdge& edge = active.shared_sources[index];
            if (edge.owner.kind != LogicalOwnerKind::SharedPrefix ||
                edge.slot >= shared_catalog_count_ ||
                std::find(active.shared_sources.begin(), active.shared_sources.begin() + index,
                          edge) != active.shared_sources.begin() + index) {
                throw std::logic_error("shared source edge is malformed");
            }
            const SharedCatalogEntry& source = shared_catalog_[edge.slot];
            if (source.state != SharedCatalogState::Catalogued || !source.handle ||
                source.id != edge.owner.id) {
                throw std::logic_error("shared source edge is stale");
            }
        }
        active.retained_private_source.reset();
        active.shared_sources.clear();
    }

    void release_cancelled_lane(LaneId lane) {
        if (lane.value >= lane_count_ || !active_[lane.value].occupied ||
            (lanes_[lane.value] != LogicalLaneState::Active &&
             lanes_[lane.value] != LogicalLaneState::TerminalPending)) {
            throw std::logic_error("cancelled lane has no logical active owner");
        }
        release_active_references(lane);
        clear_catalog_entry(catalog_.at(active_[lane.value].publication_slot));
        reset_active_entry(active_[lane.value]);
        lanes_[lane.value] = LogicalLaneState::Free;
    }

    [[nodiscard]] static std::uint64_t session_hash(const CacheSessionKey& key) noexcept {
        std::uint64_t hash = 1469598103934665603ULL;
        for (const unsigned char value : key.view()) {
            hash ^= value;
            hash *= 1099511628211ULL;
        }
        return hash;
    }

    [[nodiscard]] std::optional<std::size_t>
    find_session_cell(const CacheSessionKey& key) const noexcept {
        if (session_index_.empty()) { return std::nullopt; }
        const std::size_t begin = session_hash(key) % session_index_.size();
        for (std::size_t probe = 0; probe < session_index_.size(); ++probe) {
            const std::size_t cell         = (begin + probe) % session_index_.size();
            const SessionIndexEntry& entry = session_index_[cell];
            if (entry.state == SessionIndexState::Empty) { return std::nullopt; }
            if (entry.state == SessionIndexState::Occupied && entry.key == key) { return cell; }
        }
        return std::nullopt;
    }

    [[nodiscard]] std::size_t session_insert_cell(const CacheSessionKey& key) const {
        if (session_index_.empty()) {
            throw std::logic_error("session publication has no index capacity");
        }
        const std::size_t begin = session_hash(key) % session_index_.size();
        std::optional<std::size_t> deleted;
        for (std::size_t probe = 0; probe < session_index_.size(); ++probe) {
            const std::size_t cell         = (begin + probe) % session_index_.size();
            const SessionIndexEntry& entry = session_index_[cell];
            if (entry.state == SessionIndexState::Occupied && entry.key == key) { return cell; }
            if (entry.state == SessionIndexState::Deleted && !deleted) { deleted = cell; }
            if (entry.state == SessionIndexState::Empty) { return deleted.value_or(cell); }
        }
        if (deleted) { return *deleted; }
        throw std::logic_error("session index is full");
    }

    void erase_session_if_owner(std::uint64_t owner_id) noexcept {
        if (owner_id == 0) { return; }
        for (SessionIndexEntry& entry : session_index_) {
            if (entry.state == SessionIndexState::Occupied && entry.owner_id == owner_id) {
                entry.state             = SessionIndexState::Deleted;
                entry.slot              = kInvalidCatalogSlot;
                entry.owner_id          = 0;
                entry.revision          = 0;
                entry.publication_order = 0;
            }
        }
    }

    void refresh_session_owner_revision(std::uint64_t owner_id, std::uint32_t slot,
                                        std::uint64_t revision) noexcept {
        if (owner_id == 0 || slot >= catalog_count_) { return; }
        for (SessionIndexEntry& entry : session_index_) {
            if (entry.state == SessionIndexState::Occupied && entry.owner_id == owner_id &&
                entry.slot == slot) {
                entry.revision = revision;
                return;
            }
        }
    }

    [[nodiscard]] bool publish_session(const CacheSessionKey& key, std::uint32_t slot,
                                       std::uint64_t owner_id, std::uint64_t revision,
                                       std::uint64_t publication_order) {
        const std::size_t cell   = session_insert_cell(key);
        SessionIndexEntry& entry = session_index_[cell];
        std::optional<SessionIndexEntry> previous;
        if (entry.state == SessionIndexState::Occupied) {
            if (entry.publication_order > publication_order) { return false; }
            if (entry.publication_order == publication_order) {
                if (entry.slot != slot || entry.owner_id != owner_id) {
                    throw std::logic_error("equal publication order names two continuations");
                }
                entry.revision = revision;
                return true;
            }
            previous = entry;
        }
        entry = SessionIndexEntry{
            .state             = SessionIndexState::Occupied,
            .key               = key,
            .slot              = slot,
            .owner_id          = owner_id,
            .revision          = revision,
            .publication_order = publication_order,
        };
        if (previous && (previous->slot != slot || previous->owner_id != owner_id)) {
            demote_replaced_session(*previous, slot, owner_id);
        }
        return true;
    }

    void demote_replaced_session(const SessionIndexEntry& previous, std::uint32_t replacement_slot,
                                 std::uint64_t replacement_id) noexcept {
        if (previous.slot >= catalog_count_ ||
            (previous.slot == replacement_slot && previous.owner_id == replacement_id)) {
            return;
        }
        CatalogEntry& prior = catalog_[previous.slot];
        if (prior.state != CatalogState::Catalogued || !prior.handle ||
            prior.id != previous.owner_id || prior.revision != previous.revision) {
            return;
        }
        prior.session.reset();
        prior.retention = RetentionClass::RecentPrivate;
        for (CheckpointObservation& observation : prior.observations) {
            observation.observation.retention_class = RetentionClass::RecentPrivate;
        }
    }

    void observe_transfer(const ContextTransferObservation& observation) noexcept {
        const double seconds = static_cast<double>(observation.elapsed_ns) * 1.0e-9;
        context_stats_.actual_context_transfer_seconds += seconds;
        const std::uint64_t bytes = observation.units;
        switch (observation.resource) {
        case ContextResourceClass::State:
            switch (observation.direction) {
            case ContextTransferDirection::DeviceToHost:
                ++context_stats_.state_d2h_count;
                context_stats_.state_d2h_bytes += bytes;
                context_stats_.state_d2h_seconds += seconds;
                break;
            case ContextTransferDirection::HostToDevice:
                ++context_stats_.state_h2d_count;
                context_stats_.state_h2d_bytes += bytes;
                context_stats_.state_h2d_seconds += seconds;
                break;
            case ContextTransferDirection::DeviceToDevice:
                ++context_stats_.state_d2d_count;
                context_stats_.state_d2d_bytes += bytes;
                context_stats_.state_d2d_seconds += seconds;
                break;
            }
            break;
        case ContextResourceClass::MainKV:
            observe_kv_transfer(
                observation, context_stats_.main_kv_d2h_pages, context_stats_.main_kv_h2d_pages,
                context_stats_.main_kv_d2d_pages, context_stats_.main_kv_d2h_bytes,
                context_stats_.main_kv_h2d_bytes, context_stats_.main_kv_d2d_bytes,
                context_stats_.main_kv_d2h_seconds, context_stats_.main_kv_h2d_seconds,
                context_stats_.main_kv_d2d_seconds);
            break;
        case ContextResourceClass::BackendKV:
            observe_kv_transfer(
                observation, context_stats_.backend_kv_d2h_pages,
                context_stats_.backend_kv_h2d_pages, context_stats_.backend_kv_d2d_pages,
                context_stats_.backend_kv_d2h_bytes, context_stats_.backend_kv_h2d_bytes,
                context_stats_.backend_kv_d2d_bytes, context_stats_.backend_kv_d2h_seconds,
                context_stats_.backend_kv_h2d_seconds, context_stats_.backend_kv_d2d_seconds);
            break;
        }
    }

    static void observe_kv_transfer(const ContextTransferObservation& observation,
                                    std::uint64_t& d2h_pages, std::uint64_t& h2d_pages,
                                    std::uint64_t& d2d_pages, std::uint64_t& d2h_bytes,
                                    std::uint64_t& h2d_bytes, std::uint64_t& d2d_bytes,
                                    double& d2h_seconds, double& h2d_seconds,
                                    double& d2d_seconds) noexcept {
        const double seconds = static_cast<double>(observation.elapsed_ns) * 1.0e-9;
        switch (observation.direction) {
        case ContextTransferDirection::DeviceToHost:
            d2h_pages += observation.page_count;
            d2h_bytes += observation.units;
            d2h_seconds += seconds;
            break;
        case ContextTransferDirection::HostToDevice:
            h2d_pages += observation.page_count;
            h2d_bytes += observation.units;
            h2d_seconds += seconds;
            break;
        case ContextTransferDirection::DeviceToDevice:
            d2d_pages += observation.page_count;
            d2d_bytes += observation.units;
            d2d_seconds += seconds;
            break;
        }
    }

    template <class Result>
    void observe_transfers(const Result& result) noexcept {
        for (const ContextTransferObservation& observation : result.transfer_observations) {
            observe_transfer(observation);
        }
    }

    template <class Result>
    void observe_operations(const Result& result) noexcept {
        context_stats_.state_moves += result.operations.state_moves;
        context_stats_.state_forks += result.operations.state_forks;
        context_stats_.state_restores += result.operations.state_restores;
        context_stats_.pressure_spill_pages += result.operations.pressure_spill_pages;
        context_stats_.partial_tail_cow_pages += result.operations.partial_tail_cow_pages;
        context_stats_.historical_fork_hits += result.operations.historical_fork_hits;
    }

    std::uint32_t lane_count_           = 0;
    std::uint32_t catalog_count_        = 0;
    std::uint32_t shared_catalog_count_ = 0;
    bool cache_enabled_                 = true;
    std::array<LogicalLaneState, kMaximumConcurrency> lanes_{};
    std::vector<CatalogEntry> catalog_;
    std::vector<SharedCatalogEntry> shared_catalog_;
    std::vector<SessionIndexEntry> session_index_;
    std::vector<PrefixIndexEntry> prefix_index_;
    std::vector<CheckpointObservation> observation_scratch_;
    std::vector<PrefixDemandRecord> demand_window_;
    std::uint32_t max_long_anchors_ = 0;
    std::array<ActiveEntry, kMaximumConcurrency> active_{};
    using ContextTransaction =
        std::variant<std::monostate, MaterializationRecord, ActiveCaptureRecord>;
    ContextTransaction transaction_;
    ContextMachineCostModel cost_model_;
    Planner planner_;
    CapturePlanner capture_planner_;
    RuntimeStats context_stats_;
    std::uint64_t next_continuation_id_  = 1;
    std::uint64_t next_shared_prefix_id_ = 1;
    std::uint64_t retention_epoch_       = 0;
    std::uint64_t demand_epoch_          = 0;
};

} // namespace ninfer::runtime
