#pragma once

#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include "runtime/engine/context_cost.h"

#include <algorithm>
#include <array>
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

struct CostTieBreak {
    std::uint32_t dropped_shared_stable    = 0;
    std::uint32_t dropped_live_session     = 0;
    std::uint32_t dropped_recent_private   = 0;
    std::uint32_t evicted_continuations    = 0;
    std::uint32_t dropped_checkpoints      = 0;
    std::uint64_t remaining_text_prefill   = 0;
    std::uint64_t remaining_vision_prefill = 0;
    std::uint32_t transferred_state_images = 0;
    std::uint64_t transferred_bytes        = 0;
    std::uint32_t copy_operations          = 0;
    std::uint32_t reused_prompt_tokens     = 0;
};

struct CostEstimate {
    std::uint64_t nanoseconds = 0;
    CostTieBreak tie_break;
};

struct RetentionObservation {
    RetentionClass retention_class     = RetentionClass::RecentPrivate;
    std::uint64_t exact_eligible_count = 0;
    std::uint64_t selected_hit_count   = 0;
    std::uint64_t last_hit_epoch       = 0;
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
    using AdmissionPlan           = typename Package::AdmissionPlan;
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

        [[nodiscard]] std::uint64_t resource_revision() const noexcept {
            return plan_->resource_revision();
        }

    private:
        Choice(LaneId destination, ResourcePlan&& plan, std::uint32_t catalog_capacity,
               std::optional<CacheSessionKey> session, RetentionClass retention,
               bool update_session_index, std::uint64_t publication_order)
            : destination_(destination), plan_(std::move(plan)), session_(std::move(session)),
              retention_(retention), update_session_index_(update_session_index),
              publication_order_(publication_order) {
            evictions_.reserve(catalog_capacity);
            eviction_ids_.reserve(catalog_capacity);
            eviction_revisions_.reserve(catalog_capacity);
            pressure_options_.reserve(catalog_capacity);
            eligible_observations_.reserve(catalog_capacity);
        }

        LaneId destination_{};
        std::optional<ResourcePlan> plan_;
        ClaimDisposition source_disposition_  = ClaimDisposition::ConsumedToActive;
        std::uint32_t source_slot_            = kInvalidCatalogSlot;
        std::uint64_t source_id_              = 0;
        std::uint64_t source_revision_        = 0;
        std::uint32_t shared_source_slot_     = kInvalidCatalogSlot;
        std::uint64_t shared_source_id_       = 0;
        std::uint64_t shared_source_revision_ = 0;
        std::uint32_t publication_slot_       = kInvalidCatalogSlot;
        std::vector<std::uint32_t> evictions_;
        std::vector<std::uint64_t> eviction_ids_;
        std::vector<std::uint64_t> eviction_revisions_;
        std::vector<typename Package::PressureOption> pressure_options_;
        std::vector<std::uint32_t> shared_evictions_;
        std::vector<std::uint64_t> shared_eviction_ids_;
        std::vector<std::uint64_t> shared_eviction_revisions_;
        std::vector<typename Package::PressureOption> shared_pressure_options_;
        std::vector<PolicyObservationKey> eligible_observations_;
        std::optional<PolicyObservationKey> selected_observation_;
        std::optional<CacheSessionKey> session_;
        RetentionClass retention_                   = RetentionClass::RecentPrivate;
        bool update_session_index_                  = true;
        std::uint64_t publication_order_            = 0;
        std::uint64_t predicted_materialization_ns_ = 0;
        bool current_session_binding_               = false;

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
                    std::uint32_t max_long_anchors, ContextCostModel cost_model)
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
        for (CatalogEntry& entry : catalog_) {
            entry.summary.long_anchors.reserve(max_long_anchors_);
            entry.observations.reserve(observation_capacity);
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

        rebuild_prefix_index();
        std::optional<std::size_t> current_session_cell;
        if (cache_enabled_ && base.context_cache().session_key &&
            base.context_cache().update_session_index) {
            current_session_cell = find_session_cell(*base.context_cache().session_key);
        }
        std::vector<Candidate> candidates;
        candidates.reserve(1U + prefix_index_.size());
        std::optional<AdmissionPlan> root = program.inspect_admission(
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
                        entry.active_references != 0) {
                        continue;
                    }
                    const bool retain =
                        entry.session && (!base.context_cache().session_key ||
                                          *entry.session != *base.context_cache().session_key ||
                                          !base.context_cache().update_session_index);
                    std::optional<AdmissionPlan> plan =
                        program.inspect_admission(prompt, base, *destination, &*entry.handle,
                                                  nullptr, index.checkpoint, retain);
                    if (!plan) { continue; }
                    if (plan->summary().reusable_prompt_tokens == 0 ||
                        (retain && plan->source_disposition() != ClaimDisposition::Retained)) {
                        throw std::logic_error("Program returned an invalid private candidate");
                    }
                    const ClaimDisposition disposition = plan->source_disposition();
                    const bool current_session_binding =
                        current_session_cell &&
                        session_index_[*current_session_cell].slot == index.slot &&
                        session_index_[*current_session_cell].owner_id == entry.id &&
                        session_index_[*current_session_cell].revision == entry.revision;
                    candidates.push_back(Candidate{
                        .plan                    = std::move(*plan),
                        .current_session_binding = current_session_binding,
                        .source_slot             = index.slot,
                        .source_id               = entry.id,
                        .source_revision         = entry.revision,
                        .source_disposition      = disposition,
                        .selected_observation =
                            PolicyObservationKey{
                                .shared     = false,
                                .slot       = index.slot,
                                .owner_id   = entry.id,
                                .revision   = entry.revision,
                                .checkpoint = index.checkpoint,
                            },
                    });
                    continue;
                }

                const SharedCatalogEntry& entry = shared_catalog_[index.slot];
                if (entry.state != SharedCatalogState::Catalogued || !entry.handle) { continue; }
                std::optional<AdmissionPlan> plan = program.inspect_admission(
                    prompt, base, *destination, nullptr, &*entry.handle, index.checkpoint, false);
                if (!plan) { continue; }
                if (plan->summary().reusable_prompt_tokens == 0 ||
                    plan->source_disposition() != ClaimDisposition::Retained) {
                    throw std::logic_error("Program returned an invalid shared candidate");
                }
                candidates.push_back(Candidate{
                    .plan                   = std::move(*plan),
                    .source_disposition     = ClaimDisposition::Retained,
                    .shared_source_slot     = index.slot,
                    .shared_source_id       = entry.id,
                    .shared_source_revision = entry.revision,
                    .selected_observation =
                        PolicyObservationKey{
                            .shared     = true,
                            .slot       = index.slot,
                            .owner_id   = entry.id,
                            .revision   = entry.revision,
                            .checkpoint = index.checkpoint,
                        },
                });
            }
        }

        std::optional<Choice> selected;
        CostEstimate selected_cost;
        for (Candidate& candidate : candidates) {
            std::optional<Choice> closed =
                close_candidate(program, prompt, base, *destination, candidate, publication_order);
            if (!closed) { continue; }
            const CostEstimate cost = candidate_cost(*closed);
            if (!selected || compare_cost(cost, selected_cost) < 0 ||
                (compare_cost(cost, selected_cost) == 0 &&
                 closed->summary().reusable_prompt_tokens >
                     selected->summary().reusable_prompt_tokens) ||
                (compare_cost(cost, selected_cost) == 0 &&
                 closed->summary().reusable_prompt_tokens ==
                     selected->summary().reusable_prompt_tokens &&
                 closed->current_session_binding_ && !selected->current_session_binding_)) {
                selected.emplace(std::move(*closed));
                selected_cost = cost;
            }
        }

        if (!selected) {
            // Completeness is not delegated to the value policy.  Re-evaluate root while deleting
            // every unprotected inactive publication, even if each standalone release is zero.
            std::optional<Choice> maximal = close_maximal_root(
                program, prompt, base, *destination, candidates.front(), publication_order);
            if (maximal) { selected.emplace(std::move(*maximal)); }
        }
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
        const std::uint64_t resource_revision = program.resource_revision();
        if (!choice.plan_ || resource_revision == 0) {
            throw std::logic_error("resource choice is malformed");
        }
        if (choice.plan_->resource_revision() != resource_revision) {
            return MaterializationReserveResult::Stale;
        }
        validate_choice(choice, resource_revision);
        MaterializationRecord record = take_materialization_record(choice);
        reserve_logical_materialization(record);

        const ContextTransactionReserveStatus status = program.start_resource_transaction(
            std::move(*choice.plan_), std::move(prompt), cancellation);
        choice.plan_.reset();
        if (status == ContextTransactionReserveStatus::Aborted) {
            rollback_logical_materialization(record);
            return cancellation.requested() ? MaterializationReserveResult::Aborted
                                            : MaterializationReserveResult::Stale;
        }
        transaction_.template emplace<MaterializationRecord>(std::move(record));
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
                           bool permit_transfer, CancellationFlagView cancellation) {
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

        CaptureAssessment assessment =
            program.inspect_capture(offer, nullptr, nullptr, std::nullopt);
        const SharedPrefixHandle* exact_shared = nullptr;
        if (assessment.publishes_shared) {
            for (const PrefixIndexEntry& index : prefix_index_) {
                if (!index.shared || !valid_prefix_index_entry(index) ||
                    index.key != assessment.shortlist_key) {
                    continue;
                }
                SharedCatalogEntry& entry = shared_catalog_[index.slot];
                if (program.shared_capture_matches(offer, *entry.handle)) {
                    exact_shared = &*entry.handle;
                    assessment =
                        program.inspect_capture(offer, exact_shared, nullptr, std::nullopt);
                    break;
                }
            }
        }
        if (!assessment.publishes_private && !assessment.publishes_shared) {
            program.skip_capture(std::move(offer));
            return ActiveCaptureReserveResult::Skipped;
        }

        std::optional<CheckpointRef> private_replacement;
        if (!assessment.private_replacement_candidates.empty()) {
            private_replacement =
                *std::min_element(assessment.private_replacement_candidates.begin(),
                                  assessment.private_replacement_candidates.end(),
                                  [](CheckpointRef lhs, CheckpointRef rhs) {
                                      return std::tuple{lhs.kind, lhs.frontier, lhs.ordinal} <
                                             std::tuple{rhs.kind, rhs.frontier, rhs.ordinal};
                                  });
        }

        std::uint32_t publication_slot        = kInvalidCatalogSlot;
        const SharedPrefixHandle* replacement = nullptr;
        std::uint64_t replacement_id          = 0;
        std::uint64_t replacement_revision    = 0;
        if (assessment.publishes_shared && exact_shared == nullptr) {
            for (std::uint32_t slot = 0; slot < shared_catalog_count_; ++slot) {
                if (shared_catalog_[slot].state == SharedCatalogState::Vacant) {
                    publication_slot = slot;
                    break;
                }
            }
            if (publication_slot == kInvalidCatalogSlot) {
                std::optional<std::uint32_t> victim;
                for (std::uint32_t slot = 0; slot < shared_catalog_count_; ++slot) {
                    const SharedCatalogEntry& entry = shared_catalog_[slot];
                    if (entry.state != SharedCatalogState::Catalogued || !entry.handle ||
                        entry.transaction_pins != 0 || entry.summary.active_references != 0) {
                        continue;
                    }
                    if (!victim ||
                        shared_policy_key(entry) < shared_policy_key(shared_catalog_[*victim])) {
                        victim = slot;
                    }
                }
                if (!victim) {
                    program.skip_capture(std::move(offer));
                    return ActiveCaptureReserveResult::Skipped;
                }
                publication_slot          = *victim;
                SharedCatalogEntry& entry = shared_catalog_[*victim];
                replacement               = &*entry.handle;
                replacement_id            = entry.id;
                replacement_revision      = entry.revision;
            }
        }

        assessment = program.inspect_capture(offer, exact_shared, replacement, private_replacement);
        if ((!permit_transfer && assessment.needs_transfer) ||
            (!assessment.publishes_private && !assessment.publishes_shared)) {
            program.skip_capture(std::move(offer));
            return ActiveCaptureReserveResult::Skipped;
        }
        const ContextTransactionReserveStatus reserved = program.reserve_active_capture(
            std::move(offer), exact_shared, replacement, private_replacement, cancellation);
        if (reserved == ContextTransactionReserveStatus::Aborted) {
            return ActiveCaptureReserveResult::Skipped;
        }
        if (publication_slot != kInvalidCatalogSlot) {
            shared_catalog_[publication_slot].state = SharedCatalogState::ReservedCapture;
        }
        transaction_.template emplace<ActiveCaptureRecord>(ActiveCaptureRecord{
            .lane                 = lane,
            .publishes_private    = assessment.publishes_private,
            .publishes_shared     = assessment.publishes_shared,
            .publication_slot     = publication_slot,
            .replacement_id       = replacement_id,
            .replacement_revision = replacement_revision,
        });
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
            active             = {};
            lanes_[lane.value] = LogicalLaneState::Free;

            FinishResult released;
            released.status      = ConsumeStatus::Consumed;
            released.disposition = FinishDisposition::Released;
            released.timings     = discarded.timings;
            released.speculative = std::move(discarded.speculative);
            return released;
        }
        CatalogEntry& publication = catalog_.at(active.publication_slot);
        if (!cache_enabled_ || !active.publish_continuation ||
            result.disposition == FinishDisposition::Released) {
            if (result.disposition != FinishDisposition::Released || result.continuation) {
                throw std::logic_error("released finish returned a continuation");
            }
            release_active_references(lane);
            clear_catalog_entry(publication);
            active             = {};
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
        active             = {};
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
        active_[lane.value] = {};
        lanes_[lane.value]  = LogicalLaneState::Free;
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
        out.state_moves                       = context_stats_.state_moves;
        out.state_forks                       = context_stats_.state_forks;
        out.state_restores                    = context_stats_.state_restores;
        out.state_d2h_count                   = context_stats_.state_d2h_count;
        out.state_h2d_count                   = context_stats_.state_h2d_count;
        out.state_d2d_count                   = context_stats_.state_d2d_count;
        out.state_d2h_bytes                   = context_stats_.state_d2h_bytes;
        out.state_h2d_bytes                   = context_stats_.state_h2d_bytes;
        out.state_d2d_bytes                   = context_stats_.state_d2d_bytes;
        out.state_d2h_seconds                 = context_stats_.state_d2h_seconds;
        out.state_h2d_seconds                 = context_stats_.state_h2d_seconds;
        out.state_d2d_seconds                 = context_stats_.state_d2d_seconds;
        out.main_kv_d2h_pages                 = context_stats_.main_kv_d2h_pages;
        out.main_kv_h2d_pages                 = context_stats_.main_kv_h2d_pages;
        out.main_kv_d2d_pages                 = context_stats_.main_kv_d2d_pages;
        out.main_kv_d2h_bytes                 = context_stats_.main_kv_d2h_bytes;
        out.main_kv_h2d_bytes                 = context_stats_.main_kv_h2d_bytes;
        out.main_kv_d2d_bytes                 = context_stats_.main_kv_d2d_bytes;
        out.main_kv_d2h_seconds               = context_stats_.main_kv_d2h_seconds;
        out.main_kv_h2d_seconds               = context_stats_.main_kv_h2d_seconds;
        out.main_kv_d2d_seconds               = context_stats_.main_kv_d2d_seconds;
        out.backend_kv_d2h_pages              = context_stats_.backend_kv_d2h_pages;
        out.backend_kv_h2d_pages              = context_stats_.backend_kv_h2d_pages;
        out.backend_kv_d2d_pages              = context_stats_.backend_kv_d2d_pages;
        out.backend_kv_d2h_bytes              = context_stats_.backend_kv_d2h_bytes;
        out.backend_kv_h2d_bytes              = context_stats_.backend_kv_h2d_bytes;
        out.backend_kv_d2d_bytes              = context_stats_.backend_kv_d2d_bytes;
        out.backend_kv_d2h_seconds            = context_stats_.backend_kv_d2h_seconds;
        out.backend_kv_h2d_seconds            = context_stats_.backend_kv_h2d_seconds;
        out.backend_kv_d2d_seconds            = context_stats_.backend_kv_d2d_seconds;
        out.partial_spill_pages               = context_stats_.partial_spill_pages;
        out.partial_tail_cow_pages            = context_stats_.partial_tail_cow_pages;
        out.private_checkpoint_degradations   = context_stats_.private_checkpoint_degradations;
        out.private_checkpoint_evictions      = context_stats_.private_checkpoint_evictions;
        out.shared_checkpoint_degradations    = context_stats_.shared_checkpoint_degradations;
        out.shared_checkpoint_evictions       = context_stats_.shared_checkpoint_evictions;
        out.historical_fork_hits              = context_stats_.historical_fork_hits;
        out.last_predicted_materialization_ns = context_stats_.last_predicted_materialization_ns;
        out.actual_context_transfer_seconds   = context_stats_.actual_context_transfer_seconds;

        const auto usage                     = program.physical_usage();
        out.device_state_occupied_slots      = usage.device_state_slots;
        out.host_state_occupied_slots        = usage.host_state_slots;
        out.device_main_kv_occupied_pages    = usage.device_main_kv_pages;
        out.device_backend_kv_occupied_pages = usage.device_backend_kv_pages;
        out.host_kv_occupied_bytes           = usage.host_kv_bytes;
        std::uint64_t shared_references      = 0;
        for (const SharedCatalogEntry& entry : shared_catalog_) {
            if (entry.state == SharedCatalogState::Catalogued) {
                shared_references += entry.summary.active_references;
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
            lanes_[lane]  = LogicalLaneState::Free;
            active_[lane] = {};
        }
    }

private:
    struct Candidate {
        std::optional<AdmissionPlan> plan;
        bool current_session_binding         = false;
        std::uint32_t source_slot            = kInvalidCatalogSlot;
        std::uint64_t source_id              = 0;
        std::uint64_t source_revision        = 0;
        ClaimDisposition source_disposition  = ClaimDisposition::ConsumedToActive;
        std::uint32_t shared_source_slot     = kInvalidCatalogSlot;
        std::uint64_t shared_source_id       = 0;
        std::uint64_t shared_source_revision = 0;
        std::optional<PolicyObservationKey> selected_observation;
    };

    struct CatalogEntry {
        CatalogState state     = CatalogState::Vacant;
        std::uint64_t id       = 0;
        std::uint64_t revision = 1;
        ContinuationSummary summary;
        std::optional<ContinuationHandle> handle;
        std::optional<CacheSessionKey> session;
        std::vector<CheckpointObservation> observations;
        RetentionClass retention        = RetentionClass::RecentPrivate;
        std::uint32_t active_references = 0;
    };

    struct SharedCatalogEntry {
        SharedCatalogState state = SharedCatalogState::Vacant;
        std::uint64_t id         = 0;
        std::uint64_t revision   = 1;
        SharedPrefixSummary summary;
        std::optional<SharedPrefixHandle> handle;
        RetentionObservation observation{.retention_class = RetentionClass::SharedStable};
        std::uint32_t active_owner_mask = 0;
        std::uint32_t transaction_pins  = 0;
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
        RetentionClass retention           = RetentionClass::RecentPrivate;
        bool update_session_index          = true;
        bool publish_continuation          = true;
        std::uint64_t publication_order    = 0;
        std::uint32_t retained_source_slot = kInvalidCatalogSlot;
        std::uint64_t retained_source_id   = 0;
        std::uint32_t shared_source_slot   = kInvalidCatalogSlot;
        std::uint64_t shared_source_id     = 0;
    };

    struct MaterializationRecord {
        LaneId destination;
        std::uint32_t source_slot            = kInvalidCatalogSlot;
        std::uint64_t source_id              = 0;
        std::uint64_t source_revision        = 0;
        ClaimDisposition source_disposition  = ClaimDisposition::ConsumedToActive;
        std::uint32_t shared_source_slot     = kInvalidCatalogSlot;
        std::uint64_t shared_source_id       = 0;
        std::uint64_t shared_source_revision = 0;
        std::uint32_t publication_slot       = kInvalidCatalogSlot;
        std::vector<std::uint32_t> evictions;
        std::vector<std::uint64_t> eviction_ids;
        std::vector<std::uint64_t> eviction_revisions;
        std::vector<typename Package::PressureOption> pressure_options;
        std::vector<std::uint32_t> shared_evictions;
        std::vector<std::uint64_t> shared_eviction_ids;
        std::vector<std::uint64_t> shared_eviction_revisions;
        std::vector<typename Package::PressureOption> shared_pressure_options;
        std::vector<PolicyObservationKey> eligible_observations;
        std::optional<PolicyObservationKey> selected_observation;
        std::optional<CacheSessionKey> session;
        RetentionClass retention                   = RetentionClass::RecentPrivate;
        bool update_session_index                  = true;
        bool publish_continuation                  = true;
        std::uint64_t publication_order            = 0;
        std::uint64_t predicted_materialization_ns = 0;
    };

    struct ActiveCaptureRecord {
        LaneId lane;
        bool publishes_private             = false;
        bool publishes_shared              = false;
        std::uint32_t publication_slot     = kInvalidCatalogSlot;
        std::uint64_t replacement_id       = 0;
        std::uint64_t replacement_revision = 0;
    };

    struct PressureOwner {
        bool shared                  = false;
        std::uint32_t slot           = kInvalidCatalogSlot;
        std::uint64_t owner_id       = 0;
        std::uint64_t revision       = 0;
        RetentionClass retention     = RetentionClass::RecentPrivate;
        std::uint64_t last_hit_epoch = 0;
        std::vector<typename Package::PressureOption> alternatives;
        std::size_t next_alternative = 0;
        std::optional<typename Package::PressureOption> selected;
    };

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

    [[nodiscard]] static std::uint32_t retention_rank(RetentionClass value) noexcept {
        switch (value) {
        case RetentionClass::Disposable:
            return 0;
        case RetentionClass::RecentPrivate:
            return 1;
        case RetentionClass::LiveSession:
            return 2;
        case RetentionClass::SharedStable:
            return 3;
        }
        return 3;
    }

    [[nodiscard]] static auto shared_policy_key(const SharedCatalogEntry& entry) noexcept {
        return std::tuple{retention_rank(entry.observation.retention_class),
                          entry.observation.selected_hit_count, entry.observation.last_hit_epoch,
                          entry.id};
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
        for (const auto& anchor : summary.long_anchors) {
            if (anchor.ref.kind != CheckpointKind::LongAnchor || anchor.ref.ordinal == 0 ||
                anchor.ref.frontier == 0 || anchor.scope != CheckpointScope::Private ||
                anchor.shortlist_key.frontier != anchor.ref.frontier ||
                anchor.required_kv.main_pages == 0 || anchor.rebuild_work.tokens == 0) {
                return false;
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
                                            const ContinuationSummary& source) {
        if (source.long_anchors.size() > destination.long_anchors.capacity()) {
            throw std::logic_error("continuation summary exceeded preallocated storage");
        }
        destination.endpoint          = source.endpoint;
        destination.rewrite           = source.rewrite;
        destination.active_references = source.active_references;
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
                              RetentionClass retention) {
        observation_scratch_.clear();
        const auto append = [&](const auto& checkpoint) {
            if (observation_scratch_.size() == observation_scratch_.capacity()) {
                throw std::logic_error("checkpoint observations exceeded reserved storage");
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
        entry.retention         = RetentionClass::RecentPrivate;
        entry.active_references = 0;
        advance_revision(entry.revision);
    }

    void clear_shared_entry(SharedCatalogEntry& entry) noexcept {
        entry.state = SharedCatalogState::Vacant;
        entry.id    = 0;
        entry.handle.reset();
        entry.summary     = {};
        entry.observation = RetentionObservation{.retention_class = RetentionClass::SharedStable};
        entry.active_owner_mask = 0;
        entry.transaction_pins  = 0;
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

    static void saturating_add(std::uint64_t& destination, std::uint64_t value) noexcept {
        destination = value > std::numeric_limits<std::uint64_t>::max() - destination
                          ? std::numeric_limits<std::uint64_t>::max()
                          : destination + value;
    }

    [[nodiscard]] CostEstimate
    transfer_cost(std::span<const ContextTransferRequirement> requirements) const noexcept {
        CostEstimate out;
        for (const ContextTransferRequirement& requirement : requirements) {
            saturating_add(out.nanoseconds,
                           cost_model_.transfer_ns(requirement.direction, requirement.work));
            saturating_add(out.tie_break.transferred_bytes, requirement.work.payload_bytes);
            const std::uint64_t operations = requirement.work.copy_operations;
            out.tie_break.copy_operations =
                operations >
                        std::numeric_limits<std::uint32_t>::max() - out.tie_break.copy_operations
                    ? std::numeric_limits<std::uint32_t>::max()
                    : out.tie_break.copy_operations + static_cast<std::uint32_t>(operations);
            if (requirement.resource == ContextResourceClass::State) {
                ++out.tie_break.transferred_state_images;
            }
        }
        return out;
    }

    [[nodiscard]] CostEstimate pressure_cost(const typename Package::PressureOption& option,
                                             RetentionClass retention) const noexcept {
        CostEstimate out            = transfer_cost(option.transfer_requirements);
        std::uint64_t recovery_loss = 0;
        for (const auto& impact : option.checkpoint_impacts) {
            saturating_add(recovery_loss, cost_model_.prefill_ns(impact.fallback_rebuild_work));
            const CostEstimate restore = transfer_cost(impact.added_restore_requirements);
            saturating_add(recovery_loss, restore.nanoseconds);
            if (impact.drops_checkpoint) { ++out.tie_break.dropped_checkpoints; }
        }
        const std::uint32_t rank = retention_rank(retention);
        for (std::uint32_t scale = 0; scale < rank; ++scale) {
            recovery_loss = recovery_loss > std::numeric_limits<std::uint64_t>::max() / 4U
                                ? std::numeric_limits<std::uint64_t>::max()
                                : recovery_loss * 4U;
        }
        saturating_add(out.nanoseconds, recovery_loss);
        if (option.evicts_continuation) {
            ++out.tie_break.evicted_continuations;
            switch (retention) {
            case RetentionClass::Disposable:
                break;
            case RetentionClass::RecentPrivate:
                ++out.tie_break.dropped_recent_private;
                break;
            case RetentionClass::LiveSession:
                ++out.tie_break.dropped_live_session;
                break;
            case RetentionClass::SharedStable:
                ++out.tie_break.dropped_shared_stable;
                break;
            }
        }
        return out;
    }

    static void add_cost(CostEstimate& total, const CostEstimate& value) noexcept {
        saturating_add(total.nanoseconds, value.nanoseconds);
        const auto add_u32 = [](std::uint32_t& out, std::uint32_t add) {
            out = add > std::numeric_limits<std::uint32_t>::max() - out
                      ? std::numeric_limits<std::uint32_t>::max()
                      : out + add;
        };
        add_u32(total.tie_break.dropped_shared_stable, value.tie_break.dropped_shared_stable);
        add_u32(total.tie_break.dropped_live_session, value.tie_break.dropped_live_session);
        add_u32(total.tie_break.dropped_recent_private, value.tie_break.dropped_recent_private);
        add_u32(total.tie_break.evicted_continuations, value.tie_break.evicted_continuations);
        add_u32(total.tie_break.dropped_checkpoints, value.tie_break.dropped_checkpoints);
        saturating_add(total.tie_break.remaining_text_prefill,
                       value.tie_break.remaining_text_prefill);
        saturating_add(total.tie_break.remaining_vision_prefill,
                       value.tie_break.remaining_vision_prefill);
        add_u32(total.tie_break.transferred_state_images, value.tie_break.transferred_state_images);
        saturating_add(total.tie_break.transferred_bytes, value.tie_break.transferred_bytes);
        add_u32(total.tie_break.copy_operations, value.tie_break.copy_operations);
        add_u32(total.tie_break.reused_prompt_tokens, value.tie_break.reused_prompt_tokens);
    }

    [[nodiscard]] static auto cost_key(const CostEstimate& cost) noexcept {
        return std::tuple{
            cost.nanoseconds,
            cost.tie_break.dropped_shared_stable,
            cost.tie_break.dropped_live_session,
            cost.tie_break.dropped_recent_private,
            cost.tie_break.evicted_continuations,
            cost.tie_break.dropped_checkpoints,
            cost.tie_break.remaining_text_prefill,
            cost.tie_break.remaining_vision_prefill,
            cost.tie_break.transferred_state_images,
            cost.tie_break.transferred_bytes,
            cost.tie_break.copy_operations,
            std::numeric_limits<std::uint32_t>::max() - cost.tie_break.reused_prompt_tokens,
        };
    }

    [[nodiscard]] static int compare_cost(const CostEstimate& left,
                                          const CostEstimate& right) noexcept {
        if (cost_key(left) < cost_key(right)) { return -1; }
        if (cost_key(right) < cost_key(left)) { return 1; }
        return 0;
    }

    [[nodiscard]] CostEstimate candidate_base_cost(const AdmissionPlan& plan) const noexcept {
        CostEstimate out = transfer_cost(plan.transfer_requirements());
        saturating_add(out.nanoseconds, cost_model_.prefill_ns(plan.remaining_prefill_work()));
        out.tie_break.remaining_text_prefill   = plan.remaining_prefill_work().tokens;
        out.tie_break.remaining_vision_prefill = plan.remaining_prefill_work().vision_patches;
        out.tie_break.reused_prompt_tokens     = plan.summary().reusable_prompt_tokens;
        return out;
    }

    [[nodiscard]] CostEstimate candidate_cost(const Choice& choice) const noexcept {
        CostEstimate out;
        out.nanoseconds                    = choice.predicted_materialization_ns_;
        out.tie_break.reused_prompt_tokens = choice.summary().reusable_prompt_tokens;
        return out;
    }

    [[nodiscard]] auto pressure_key(const PressureOwner& owner,
                                    const typename Package::PressureOption& option) const noexcept {
        const CostEstimate loss           = pressure_cost(option, owner.retention);
        const std::uint32_t shared_fanout = owner.shared ? 1U : 0U;
        return std::tuple{
            retention_rank(owner.retention),
            loss.nanoseconds,
            shared_fanout,
            owner.last_hit_epoch,
            owner.owner_id,
            option.evicts_continuation ? 1U : 0U,
            option.id,
            owner.shared ? 1U : 0U,
            owner.slot,
        };
    }

    [[nodiscard]] std::vector<PressureOwner>
    build_pressure_owners(Program& program, const AdmissionPlan& admission,
                          const Candidate& candidate) const {
        std::vector<PressureOwner> owners;
        owners.reserve(catalog_count_ + shared_catalog_count_);
        for (std::uint32_t slot = 0; slot < catalog_count_; ++slot) {
            const CatalogEntry& entry = catalog_[slot];
            if (slot == candidate.source_slot || entry.state != CatalogState::Catalogued ||
                !entry.handle || entry.active_references != 0) {
                continue;
            }
            PressureOwner owner{
                .shared         = false,
                .slot           = slot,
                .owner_id       = entry.id,
                .revision       = entry.revision,
                .retention      = entry.retention,
                .last_hit_epoch = newest_hit_epoch(entry),
            };
            std::vector<typename Package::PressureOption> preserving =
                program.inspect_pressure_options(admission, *entry.handle);
            preserving.erase(
                std::remove_if(preserving.begin(), preserving.end(),
                               [](const auto& option) { return option.evicts_continuation; }),
                preserving.end());
            std::stable_sort(preserving.begin(), preserving.end(),
                             [&](const auto& left, const auto& right) {
                                 return pressure_key(owner, left) < pressure_key(owner, right);
                             });
            owner.alternatives = std::move(preserving);
            typename Package::PressureOption eviction =
                program.inspect_eviction_option(*entry.handle);
            if (!eviction.evicts_continuation || eviction.shared_owner) {
                throw std::logic_error("Program private eviction action is invalid");
            }
            owner.alternatives.push_back(std::move(eviction));
            owners.push_back(std::move(owner));
        }
        for (std::uint32_t slot = 0; slot < shared_catalog_count_; ++slot) {
            const SharedCatalogEntry& entry = shared_catalog_[slot];
            if (slot == candidate.shared_source_slot ||
                entry.state != SharedCatalogState::Catalogued || !entry.handle ||
                entry.transaction_pins != 0 || entry.summary.active_references != 0) {
                continue;
            }
            PressureOwner owner{
                .shared         = true,
                .slot           = slot,
                .owner_id       = entry.id,
                .revision       = entry.revision,
                .retention      = RetentionClass::SharedStable,
                .last_hit_epoch = entry.observation.last_hit_epoch,
            };
            std::vector<typename Package::PressureOption> preserving =
                program.inspect_shared_pressure_options(admission, *entry.handle);
            preserving.erase(
                std::remove_if(preserving.begin(), preserving.end(),
                               [](const auto& option) { return option.evicts_continuation; }),
                preserving.end());
            std::stable_sort(preserving.begin(), preserving.end(),
                             [&](const auto& left, const auto& right) {
                                 return pressure_key(owner, left) < pressure_key(owner, right);
                             });
            owner.alternatives = std::move(preserving);
            typename Package::PressureOption eviction =
                program.inspect_shared_eviction_option(*entry.handle);
            if (!eviction.evicts_continuation || !eviction.shared_owner) {
                throw std::logic_error("Program shared eviction action is invalid");
            }
            owner.alternatives.push_back(std::move(eviction));
            owners.push_back(std::move(owner));
        }
        return owners;
    }

    [[nodiscard]] std::uint64_t newest_hit_epoch(const CatalogEntry& entry) const noexcept {
        std::uint64_t epoch = 0;
        for (const CheckpointObservation& observation : entry.observations) {
            epoch = std::max(epoch, observation.observation.last_hit_epoch);
        }
        return epoch;
    }

    [[nodiscard]] std::optional<Choice>
    try_seal_choice(Program& program, const PreparedPrompt& prompt, const RequestBasePlan& base,
                    LaneId destination, const Candidate& candidate,
                    std::span<const PressureOwner> owners, std::uint64_t publication_order) {
        if (!candidate.plan) { return std::nullopt; }
        std::vector<const ContinuationHandle*> private_handles;
        std::vector<typename Package::PressureOption> private_options;
        std::vector<std::uint32_t> private_slots;
        std::vector<std::uint64_t> private_ids;
        std::vector<std::uint64_t> private_revisions;
        std::vector<const SharedPrefixHandle*> shared_handles;
        std::vector<typename Package::PressureOption> shared_options;
        std::vector<std::uint32_t> shared_slots;
        std::vector<std::uint64_t> shared_ids;
        std::vector<std::uint64_t> shared_revisions;
        private_handles.reserve(catalog_count_);
        private_options.reserve(catalog_count_);
        private_slots.reserve(catalog_count_);
        private_ids.reserve(catalog_count_);
        private_revisions.reserve(catalog_count_);
        shared_handles.reserve(shared_catalog_count_);
        shared_options.reserve(shared_catalog_count_);
        shared_slots.reserve(shared_catalog_count_);
        shared_ids.reserve(shared_catalog_count_);
        shared_revisions.reserve(shared_catalog_count_);

        std::uint32_t publication_slot = kInvalidCatalogSlot;
        for (std::uint32_t slot = 0; slot < catalog_count_; ++slot) {
            if (catalog_[slot].state == CatalogState::Vacant) {
                publication_slot = slot;
                break;
            }
        }
        if (candidate.source_slot != kInvalidCatalogSlot &&
            candidate.source_disposition == ClaimDisposition::ConsumedToActive) {
            publication_slot = candidate.source_slot;
        }

        CostEstimate total = candidate_base_cost(*candidate.plan);
        for (const PressureOwner& owner : owners) {
            if (!owner.selected) { continue; }
            add_cost(total, pressure_cost(*owner.selected, owner.retention));
            if (!owner.shared) {
                const CatalogEntry& entry = catalog_.at(owner.slot);
                if (entry.state != CatalogState::Catalogued || !entry.handle ||
                    entry.id != owner.owner_id || entry.revision != owner.revision) {
                    return std::nullopt;
                }
                private_handles.push_back(&*entry.handle);
                private_options.push_back(*owner.selected);
                private_slots.push_back(owner.slot);
                private_ids.push_back(owner.owner_id);
                private_revisions.push_back(owner.revision);
                if (owner.selected->evicts_continuation &&
                    publication_slot == kInvalidCatalogSlot) {
                    publication_slot = owner.slot;
                }
            } else {
                const SharedCatalogEntry& entry = shared_catalog_.at(owner.slot);
                if (entry.state != SharedCatalogState::Catalogued || !entry.handle ||
                    entry.id != owner.owner_id || entry.revision != owner.revision) {
                    return std::nullopt;
                }
                shared_handles.push_back(&*entry.handle);
                shared_options.push_back(*owner.selected);
                shared_slots.push_back(owner.slot);
                shared_ids.push_back(owner.owner_id);
                shared_revisions.push_back(owner.revision);
            }
        }
        if (publication_slot == kInvalidCatalogSlot) { return std::nullopt; }

        std::optional<ResourcePlan> sealed =
            program.seal_resource_plan(*candidate.plan, prompt, private_handles, private_options,
                                       shared_handles, shared_options);
        if (!sealed) { return std::nullopt; }

        Choice choice(destination, std::move(*sealed), catalog_count_,
                      base.context_cache().session_key, base.context_cache().retention,
                      base.context_cache().update_session_index, publication_order);
        choice.source_slot_               = candidate.source_slot;
        choice.source_id_                 = candidate.source_id;
        choice.source_revision_           = candidate.source_revision;
        choice.source_disposition_        = candidate.source_disposition;
        choice.shared_source_slot_        = candidate.shared_source_slot;
        choice.shared_source_id_          = candidate.shared_source_id;
        choice.shared_source_revision_    = candidate.shared_source_revision;
        choice.publication_slot_          = publication_slot;
        choice.evictions_                 = std::move(private_slots);
        choice.eviction_ids_              = std::move(private_ids);
        choice.eviction_revisions_        = std::move(private_revisions);
        choice.pressure_options_          = std::move(private_options);
        choice.shared_evictions_          = std::move(shared_slots);
        choice.shared_eviction_ids_       = std::move(shared_ids);
        choice.shared_eviction_revisions_ = std::move(shared_revisions);
        choice.shared_pressure_options_   = std::move(shared_options);
        choice.selected_observation_      = candidate.selected_observation;
        choice.current_session_binding_   = candidate.current_session_binding;
        if (candidate.selected_observation) {
            choice.eligible_observations_.push_back(*candidate.selected_observation);
        }
        choice.predicted_materialization_ns_ = total.nanoseconds;
        return choice;
    }

    [[nodiscard]] std::optional<Choice>
    close_candidate(Program& program, const PreparedPrompt& prompt, const RequestBasePlan& base,
                    LaneId destination, Candidate& candidate, std::uint64_t publication_order) {
        std::vector<PressureOwner> owners =
            build_pressure_owners(program, *candidate.plan, candidate);
        if (std::optional<Choice> ready = try_seal_choice(program, prompt, base, destination,
                                                          candidate, owners, publication_order)) {
            return ready;
        }

        for (;;) {
            std::optional<std::size_t> selected;
            for (std::size_t index = 0; index < owners.size(); ++index) {
                const PressureOwner& owner = owners[index];
                if (owner.next_alternative >= owner.alternatives.size()) { continue; }
                if (!selected ||
                    pressure_key(owner, owner.alternatives[owner.next_alternative]) <
                        pressure_key(
                            owners[*selected],
                            owners[*selected].alternatives[owners[*selected].next_alternative])) {
                    selected = index;
                }
            }
            if (!selected) { break; }
            PressureOwner& owner = owners[*selected];
            owner.selected       = owner.alternatives[owner.next_alternative++];
            if (std::optional<Choice> ready = try_seal_choice(
                    program, prompt, base, destination, candidate, owners, publication_order)) {
                return ready;
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] std::optional<Choice>
    close_maximal_root(Program& program, const PreparedPrompt& prompt, const RequestBasePlan& base,
                       LaneId destination, Candidate& root, std::uint64_t publication_order) {
        Candidate root_candidate;
        root_candidate.plan = program.inspect_admission(prompt, base, destination, nullptr, nullptr,
                                                        std::nullopt, false);
        if (!root_candidate.plan) {
            throw std::logic_error("Program lost root planning during completeness fallback");
        }
        std::vector<PressureOwner> owners =
            build_pressure_owners(program, *root_candidate.plan, root_candidate);
        for (PressureOwner& owner : owners) {
            if (owner.alternatives.empty() || !owner.alternatives.back().evicts_continuation) {
                throw std::logic_error("maximal release owner has no terminal action");
            }
            owner.selected         = owner.alternatives.back();
            owner.next_alternative = owner.alternatives.size();
        }
        (void)root;
        return try_seal_choice(program, prompt, base, destination, root_candidate, owners,
                               publication_order);
    }

    void validate_choice(const Choice& choice, std::uint64_t revision) const {
        if (!choice.plan_ || choice.destination_.value >= lane_count_ ||
            lanes_[choice.destination_.value] != LogicalLaneState::Free || revision == 0 ||
            choice.evictions_.size() != choice.eviction_ids_.size() ||
            choice.evictions_.size() != choice.eviction_revisions_.size() ||
            choice.evictions_.size() != choice.pressure_options_.size() ||
            choice.shared_evictions_.size() != choice.shared_eviction_ids_.size() ||
            choice.shared_evictions_.size() != choice.shared_eviction_revisions_.size() ||
            choice.shared_evictions_.size() != choice.shared_pressure_options_.size() ||
            choice.publication_slot_ >= catalog_count_ || choice.publication_order_ == 0) {
            throw std::logic_error("resource choice is stale or malformed");
        }
        if (choice.source_slot_ != kInvalidCatalogSlot) {
            if (choice.source_slot_ >= catalog_count_) {
                throw std::logic_error("private source slot is invalid");
            }
            const CatalogEntry& source = catalog_[choice.source_slot_];
            if (source.state != CatalogState::Catalogued || !source.handle ||
                source.id != choice.source_id_ || source.revision != choice.source_revision_ ||
                source.active_references != 0) {
                throw std::logic_error("private source changed after planning");
            }
        }
        if (choice.shared_source_slot_ != kInvalidCatalogSlot) {
            if (choice.shared_source_slot_ >= shared_catalog_count_) {
                throw std::logic_error("shared source slot is invalid");
            }
            const SharedCatalogEntry& source = shared_catalog_[choice.shared_source_slot_];
            if (source.state != SharedCatalogState::Catalogued || !source.handle ||
                source.id != choice.shared_source_id_ ||
                source.revision != choice.shared_source_revision_) {
                throw std::logic_error("shared source changed after planning");
            }
        }
        for (std::size_t row = 0; row < choice.evictions_.size(); ++row) {
            const std::uint32_t slot = choice.evictions_[row];
            if (slot >= catalog_count_ || slot == choice.source_slot_) {
                throw std::logic_error("private pressure owner is invalid");
            }
            const CatalogEntry& entry = catalog_[slot];
            if (entry.state != CatalogState::Catalogued || !entry.handle ||
                entry.id != choice.eviction_ids_[row] ||
                entry.revision != choice.eviction_revisions_[row] || entry.active_references != 0) {
                throw std::logic_error("private pressure owner changed after planning");
            }
        }
        for (std::size_t row = 0; row < choice.shared_evictions_.size(); ++row) {
            const std::uint32_t slot = choice.shared_evictions_[row];
            if (slot >= shared_catalog_count_ || slot == choice.shared_source_slot_) {
                throw std::logic_error("shared pressure owner is invalid");
            }
            const SharedCatalogEntry& entry = shared_catalog_[slot];
            if (entry.state != SharedCatalogState::Catalogued || !entry.handle ||
                entry.id != choice.shared_eviction_ids_[row] ||
                entry.revision != choice.shared_eviction_revisions_[row] ||
                entry.transaction_pins != 0 || entry.summary.active_references != 0) {
                throw std::logic_error("shared pressure owner changed after planning");
            }
        }
        const CatalogEntry& publication = catalog_[choice.publication_slot_];
        const bool source_cell          = choice.publication_slot_ == choice.source_slot_ &&
                                 choice.source_disposition_ == ClaimDisposition::ConsumedToActive;
        const auto victim =
            std::find(choice.evictions_.begin(), choice.evictions_.end(), choice.publication_slot_);
        const bool victim_cell =
            victim != choice.evictions_.end() &&
            choice.pressure_options_[static_cast<std::size_t>(victim - choice.evictions_.begin())]
                .evicts_continuation;
        if (publication.state != CatalogState::Vacant && !source_cell && !victim_cell) {
            throw std::logic_error("resource choice has no publication cell");
        }
    }

    [[nodiscard]] MaterializationRecord take_materialization_record(Choice& choice) {
        return MaterializationRecord{
            .destination                  = choice.destination_,
            .source_slot                  = choice.source_slot_,
            .source_id                    = choice.source_id_,
            .source_revision              = choice.source_revision_,
            .source_disposition           = choice.source_disposition_,
            .shared_source_slot           = choice.shared_source_slot_,
            .shared_source_id             = choice.shared_source_id_,
            .shared_source_revision       = choice.shared_source_revision_,
            .publication_slot             = choice.publication_slot_,
            .evictions                    = std::move(choice.evictions_),
            .eviction_ids                 = std::move(choice.eviction_ids_),
            .eviction_revisions           = std::move(choice.eviction_revisions_),
            .pressure_options             = std::move(choice.pressure_options_),
            .shared_evictions             = std::move(choice.shared_evictions_),
            .shared_eviction_ids          = std::move(choice.shared_eviction_ids_),
            .shared_eviction_revisions    = std::move(choice.shared_eviction_revisions_),
            .shared_pressure_options      = std::move(choice.shared_pressure_options_),
            .eligible_observations        = std::move(choice.eligible_observations_),
            .selected_observation         = choice.selected_observation_,
            .session                      = std::move(choice.session_),
            .retention                    = choice.retention_,
            .update_session_index         = choice.update_session_index_,
            .publish_continuation         = cache_enabled_,
            .publication_order            = choice.publication_order_,
            .predicted_materialization_ns = choice.predicted_materialization_ns_,
        };
    }

    void reserve_logical_materialization(const MaterializationRecord& record) {
        lanes_[record.destination.value] = LogicalLaneState::Materializing;
        if (record.source_slot != kInvalidCatalogSlot) {
            catalog_[record.source_slot].state = CatalogState::Claimed;
        }
        if (record.shared_source_slot != kInvalidCatalogSlot) {
            ++shared_catalog_[record.shared_source_slot].transaction_pins;
        }
        for (const std::uint32_t slot : record.evictions) {
            catalog_[slot].state = CatalogState::Claimed;
        }
        for (const std::uint32_t slot : record.shared_evictions) {
            shared_catalog_[slot].state = SharedCatalogState::Claimed;
        }
        CatalogEntry& publication = catalog_[record.publication_slot];
        if (publication.state == CatalogState::Vacant) {
            publication.state = CatalogState::Claimed;
        }
    }

    void rollback_logical_materialization(const MaterializationRecord& record) noexcept {
        lanes_[record.destination.value] = LogicalLaneState::Free;
        if (record.source_slot != kInvalidCatalogSlot) {
            catalog_[record.source_slot].state = CatalogState::Catalogued;
        }
        if (record.shared_source_slot != kInvalidCatalogSlot) {
            SharedCatalogEntry& source = shared_catalog_[record.shared_source_slot];
            if (source.transaction_pins != 0) { --source.transaction_pins; }
        }
        for (const std::uint32_t slot : record.evictions) {
            catalog_[slot].state = CatalogState::Catalogued;
        }
        for (const std::uint32_t slot : record.shared_evictions) {
            shared_catalog_[slot].state = SharedCatalogState::Catalogued;
        }
        CatalogEntry& publication = catalog_[record.publication_slot];
        if (publication.id == 0 && !publication.handle) {
            publication.state = CatalogState::Vacant;
        }
    }

    void observe_policy_selection(const MaterializationRecord& record) noexcept {
        for (const PolicyObservationKey& key : record.eligible_observations) {
            RetentionObservation* observation = resolve_observation(key);
            if (observation) { saturating_increment(observation->exact_eligible_count); }
        }
        if (record.selected_observation) {
            RetentionObservation* selected = resolve_observation(*record.selected_observation);
            if (selected) {
                saturating_increment(selected->selected_hit_count);
                selected->last_hit_epoch = ++retention_epoch_;
            }
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

    template <class Result>
    void apply_private_action(std::uint32_t slot, std::uint64_t owner_id, std::uint64_t revision,
                              const Result& result) {
        if (slot >= catalog_count_) {
            throw std::logic_error("private action result has an invalid slot");
        }
        CatalogEntry& entry = catalog_[slot];
        if (entry.state != CatalogState::Claimed || entry.id != owner_id ||
            entry.revision != revision || !entry.handle) {
            throw std::logic_error("private action owner changed before adoption");
        }
        if (result.disposition == ClaimDisposition::Evicted) {
            erase_session_if_owner(owner_id);
            clear_catalog_entry(entry);
            saturating_increment(context_stats_.private_checkpoint_evictions);
            return;
        }
        if (result.disposition != ClaimDisposition::Retained) {
            throw std::logic_error("private pressure action returned an invalid disposition");
        }
        if (result.final_summary) {
            if (!valid_continuation_summary(*result.final_summary)) {
                throw std::logic_error("private pressure action returned an invalid summary");
            }
            assign_continuation_summary(entry.summary, *result.final_summary);
            migrate_observations(entry, *result.final_summary, entry.retention);
            advance_revision(entry.revision);
            refresh_session_owner_revision(owner_id, slot, entry.revision);
            saturating_increment(context_stats_.private_checkpoint_degradations);
        }
        entry.state = CatalogState::Catalogued;
    }

    template <class Result>
    void apply_shared_action(std::uint32_t slot, std::uint64_t owner_id, std::uint64_t revision,
                             const Result& result) {
        if (slot >= shared_catalog_count_) {
            throw std::logic_error("shared action result has an invalid slot");
        }
        SharedCatalogEntry& entry = shared_catalog_[slot];
        if (entry.state != SharedCatalogState::Claimed || entry.id != owner_id ||
            entry.revision != revision || !entry.handle) {
            throw std::logic_error("shared action owner changed before adoption");
        }
        if (result.disposition == ClaimDisposition::Evicted) {
            clear_shared_entry(entry);
            saturating_increment(context_stats_.shared_checkpoint_evictions);
            return;
        }
        if (result.disposition != ClaimDisposition::Retained) {
            throw std::logic_error("shared pressure action returned an invalid disposition");
        }
        if (result.final_summary) {
            if (!valid_shared_prefix_summary(*result.final_summary)) {
                throw std::logic_error("shared pressure action returned an invalid summary");
            }
            entry.summary = *result.final_summary;
            advance_revision(entry.revision);
            saturating_increment(context_stats_.shared_checkpoint_degradations);
        }
        entry.state = SharedCatalogState::Catalogued;
    }

    void restore_unreported_materialization(const MaterializationRecord& record) noexcept {
        if (record.source_slot != kInvalidCatalogSlot &&
            catalog_[record.source_slot].state == CatalogState::Claimed) {
            catalog_[record.source_slot].state = CatalogState::Catalogued;
        }
        if (record.shared_source_slot != kInvalidCatalogSlot) {
            SharedCatalogEntry& source = shared_catalog_[record.shared_source_slot];
            if (source.transaction_pins != 0) { --source.transaction_pins; }
        }
        for (const std::uint32_t slot : record.evictions) {
            if (catalog_[slot].state == CatalogState::Claimed) {
                catalog_[slot].state = CatalogState::Catalogued;
            }
        }
        for (const std::uint32_t slot : record.shared_evictions) {
            if (shared_catalog_[slot].state == SharedCatalogState::Claimed) {
                shared_catalog_[slot].state = SharedCatalogState::Catalogued;
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
        if (result.victims.size() != record->evictions.size() ||
            result.shared_victims.size() != record->shared_evictions.size()) {
            throw std::logic_error("materialization result is not action aligned");
        }

        observe_policy_selection(*record);
        for (std::size_t row = 0; row < result.victims.size(); ++row) {
            apply_private_action(record->evictions[row], record->eviction_ids[row],
                                 record->eviction_revisions[row], result.victims[row]);
        }
        for (std::size_t row = 0; row < result.shared_victims.size(); ++row) {
            apply_shared_action(record->shared_evictions[row], record->shared_eviction_ids[row],
                                record->shared_eviction_revisions[row], result.shared_victims[row]);
        }

        bool retained_private_source = false;
        if (record->source_slot != kInvalidCatalogSlot) {
            CatalogEntry& source = catalog_[record->source_slot];
            if (!result.source || source.state != CatalogState::Claimed ||
                source.id != record->source_id || source.revision != record->source_revision) {
                throw std::logic_error("materialization private source result is missing");
            }
            if (result.source->disposition == ClaimDisposition::Retained) {
                if (result.source->final_summary) {
                    if (!valid_continuation_summary(*result.source->final_summary)) {
                        throw std::logic_error("materialization source summary is invalid");
                    }
                    assign_continuation_summary(source.summary, *result.source->final_summary);
                    migrate_observations(source, *result.source->final_summary, source.retention);
                    advance_revision(source.revision);
                    refresh_session_owner_revision(record->source_id, record->source_slot,
                                                   source.revision);
                }
                source.state            = CatalogState::Catalogued;
                retained_private_source = result.status == ContextTransactionStatus::Published;
                if (retained_private_source) { ++source.active_references; }
            } else if (result.source->disposition == ClaimDisposition::ConsumedToActive) {
                erase_session_if_owner(source.id);
                source.handle.reset();
                source.summary.endpoint.reset();
                source.summary.rewrite.reset();
                source.summary.long_anchors.clear();
                source.observations.clear();
                source.session.reset();
                source.active_references = 0;
            } else {
                throw std::logic_error("materialization source returned an invalid disposition");
            }
        } else if (result.source) {
            throw std::logic_error("root materialization returned a private source result");
        }

        if (record->shared_source_slot != kInvalidCatalogSlot) {
            SharedCatalogEntry& source = shared_catalog_[record->shared_source_slot];
            if (!result.shared_source || source.transaction_pins == 0 ||
                source.id != record->shared_source_id ||
                source.revision != record->shared_source_revision ||
                result.shared_source->disposition != ClaimDisposition::Retained) {
                throw std::logic_error("materialization shared source result is invalid");
            }
            --source.transaction_pins;
            if (result.shared_source->final_summary) {
                if (!valid_shared_prefix_summary(*result.shared_source->final_summary)) {
                    throw std::logic_error("materialization shared source summary is invalid");
                }
                source.summary = *result.shared_source->final_summary;
                advance_revision(source.revision);
            }
            if (result.status == ContextTransactionStatus::Published) {
                ++source.summary.active_references;
                source.active_owner_mask |= 1U << record->destination.value;
            }
        } else if (result.shared_source) {
            throw std::logic_error("materialization returned an unexpected shared source result");
        }

        observe_transfers(result);
        observe_operations(result);
        context_stats_.last_predicted_materialization_ns = record->predicted_materialization_ns;

        if (result.status == ContextTransactionStatus::Aborted) {
            if (result.published) {
                throw std::logic_error("aborted materialization published an active sequence");
            }
            restore_unreported_materialization(*record);
            lanes_[record->destination.value] = LogicalLaneState::Free;
            transaction_.template emplace<std::monostate>();
            program.finalize_context_transaction();
            return {.status = ContextTransactionStatus::Aborted};
        }
        if (result.status != ContextTransactionStatus::Published || !result.published) {
            throw std::logic_error("materialization terminal status is invalid");
        }

        CatalogEntry& publication = catalog_[record->publication_slot];
        if (publication.handle) {
            throw std::logic_error("active publication cell retained an inactive capability");
        }
        publication.state             = CatalogState::ReservedForActive;
        publication.id                = next_continuation_id_++;
        publication.session           = record->session;
        publication.retention         = record->retention;
        publication.active_references = 0;
        advance_revision(publication.revision);
        if (publication.id == 0) { publication.id = next_continuation_id_++; }

        active_[record->destination.value] = ActiveEntry{
            .occupied             = true,
            .publication_slot     = record->publication_slot,
            .continuation_id      = publication.id,
            .session              = record->session,
            .retention            = record->retention,
            .update_session_index = record->update_session_index,
            .publish_continuation = record->publish_continuation,
            .publication_order    = record->publication_order,
            .retained_source_slot =
                retained_private_source ? record->source_slot : kInvalidCatalogSlot,
            .retained_source_id = retained_private_source ? record->source_id : 0,
            .shared_source_slot = record->shared_source_slot,
            .shared_source_id   = record->shared_source_id,
        };
        StartResult start = std::move(*result.published);
        result.published.reset();
        return MaterializationOutcome{
            .status     = ContextTransactionStatus::Published,
            .activation = PublishedActivation(*this, std::move(start), record->destination),
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
        if (result.status == ContextTransactionStatus::Aborted) {
            if (result.shared) {
                throw std::logic_error("aborted active capture published a shared prefix");
            }
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
        if (result.status != ContextTransactionStatus::Published) {
            throw std::logic_error("active capture terminal status is invalid");
        }
        ActiveEntry& active = active_[record->lane.value];
        if (!active.occupied || lanes_[record->lane.value] != LogicalLaneState::Active) {
            throw std::logic_error("active capture owner left its lane");
        }
        if (record->publishes_private) {
            if (!valid_continuation_summary(result.active_summary)) {
                throw std::logic_error("active capture returned an invalid private summary");
            }
            CatalogEntry& publication = catalog_.at(active.publication_slot);
            if (publication.state != CatalogState::ReservedForActive ||
                publication.id != active.continuation_id) {
                throw std::logic_error("active private publication changed during capture");
            }
            assign_continuation_summary(publication.summary, result.active_summary);
            migrate_observations(publication, result.active_summary, active.retention);
            advance_revision(publication.revision);
        }
        if (record->publishes_shared) {
            if (!result.shared || record->publication_slot >= shared_catalog_count_ ||
                !valid_shared_prefix_summary(result.shared->summary)) {
                throw std::logic_error("active capture returned an invalid shared publication");
            }
            SharedCatalogEntry& publication = shared_catalog_[record->publication_slot];
            if (publication.state != SharedCatalogState::ReservedCapture ||
                (record->replacement_id != 0 &&
                 (publication.id != record->replacement_id ||
                  publication.revision != record->replacement_revision))) {
                throw std::logic_error("shared capture publication changed before adoption");
            }
            publication.handle.reset();
            publication.state = SharedCatalogState::Catalogued;
            publication.id    = next_shared_prefix_id_++;
            if (publication.id == 0) { publication.id = next_shared_prefix_id_++; }
            publication.summary = result.shared->summary;
            publication.handle.emplace(std::move(result.shared->handle));
            publication.observation =
                RetentionObservation{.retention_class = RetentionClass::SharedStable};
            publication.active_owner_mask = 1U << record->lane.value;
            publication.transaction_pins  = 0;
            advance_revision(publication.revision);
        } else if (result.shared) {
            throw std::logic_error("private capture returned an unexpected shared publication");
        }
        observe_transfers(result);
        observe_operations(result);
        transaction_.template emplace<std::monostate>();
        program.finalize_context_transaction();
        return {.status = ContextTransactionStatus::Published};
    }

    void release_active_references(LaneId lane) {
        ActiveEntry& active = active_[lane.value];
        if (active.retained_source_slot != kInvalidCatalogSlot) {
            CatalogEntry& source = catalog_.at(active.retained_source_slot);
            if (source.id != active.retained_source_id || source.active_references == 0) {
                throw std::logic_error("retained private source reference is stale");
            }
            --source.active_references;
            active.retained_source_slot = kInvalidCatalogSlot;
            active.retained_source_id   = 0;
        }
        const std::uint32_t owner_bit = 1U << lane.value;
        for (SharedCatalogEntry& entry : shared_catalog_) {
            if ((entry.active_owner_mask & owner_bit) == 0) { continue; }
            if (entry.summary.active_references == 0) {
                throw std::logic_error("shared active reference count underflowed");
            }
            entry.active_owner_mask &= ~owner_bit;
            --entry.summary.active_references;
        }
    }

    void release_cancelled_lane(LaneId lane) {
        if (lane.value >= lane_count_ || !active_[lane.value].occupied ||
            (lanes_[lane.value] != LogicalLaneState::Active &&
             lanes_[lane.value] != LogicalLaneState::TerminalPending)) {
            throw std::logic_error("cancelled lane has no logical active owner");
        }
        release_active_references(lane);
        clear_catalog_entry(catalog_.at(active_[lane.value].publication_slot));
        active_[lane.value] = {};
        lanes_[lane.value]  = LogicalLaneState::Free;
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
    std::uint32_t max_long_anchors_ = 0;
    std::array<ActiveEntry, kMaximumConcurrency> active_{};
    using ContextTransaction =
        std::variant<std::monostate, MaterializationRecord, ActiveCaptureRecord>;
    ContextTransaction transaction_;
    ContextCostModel cost_model_;
    RuntimeStats context_stats_;
    std::uint64_t next_continuation_id_  = 1;
    std::uint64_t next_shared_prefix_id_ = 1;
    std::uint64_t retention_epoch_       = 0;
};

} // namespace ninfer::runtime
