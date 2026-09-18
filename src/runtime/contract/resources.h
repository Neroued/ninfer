#pragma once

#include "core/math_util.h"
#include "runtime/contract/request.h"
#include "core/transfer_work.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>

namespace ninfer::runtime {

// Exact features for the startup-selected static prefill cost model. They describe only the
// suffix rebuilt after a selected prefix and remain separate from Scheduler service work.
struct PrefillWork {
    std::uint64_t chunks          = 0;
    std::uint64_t tokens          = 0;
    std::uint64_t attention_pairs = 0;
    std::uint64_t vision_items    = 0;
    std::uint64_t vision_patches  = 0;

    [[nodiscard]] friend constexpr bool operator==(PrefillWork, PrefillWork) noexcept = default;
};

// Exact prefill feature definition for a suffix beginning after prefix_tokens. Attention work is
// prefix*suffix + suffix*(suffix+1)/2 and all arithmetic saturates.
[[nodiscard]] inline PrefillWork make_prefill_work(std::uint64_t prefix_tokens,
                                                   std::uint64_t suffix_tokens,
                                                   std::uint64_t vision_items,
                                                   std::uint64_t vision_patches,
                                                   std::uint32_t prefill_chunk) noexcept {
    PrefillWork result;
    result.chunks =
        suffix_tokens == 0 || prefill_chunk == 0 ? 0 : 1U + (suffix_tokens - 1U) / prefill_chunk;
    result.tokens                       = suffix_tokens;
    result.vision_items                 = vision_items;
    result.vision_patches               = vision_patches;
    // attention_pairs = min(max64, prefix*suffix + suffix*(suffix+1)/2), computed with
    // u128 limbs so the 128-bit sum saturates at max64 exactly as the old __int128 code did.
    std::uint64_t linear_high = 0;
    const std::uint64_t linear_low = core::u128_mul(prefix_tokens, suffix_tokens, &linear_high);
    // One of suffix and suffix+1 is even, so halve before multiplying to stay in 64 bits.
    // suffix+1 wraps only for suffix == max64 (odd), whose half is 2^63 and must be special-cased.
    const bool suffix_even          = (suffix_tokens & 1U) == 0U;
    const std::uint64_t first       = suffix_even ? suffix_tokens / 2U : suffix_tokens;
    const std::uint64_t second      = suffix_even
                                          ? suffix_tokens + 1U
                                          : (suffix_tokens == std::numeric_limits<std::uint64_t>::max()
                                                 ? (std::numeric_limits<std::uint64_t>::max() >> 1U) + 1U
                                                 : (suffix_tokens + 1U) / 2U);
    std::uint64_t triangular_high = 0;
    const std::uint64_t triangular_low = core::u128_mul(first, second, &triangular_high);
    const std::uint64_t sum_low  = linear_low + triangular_low;
    const std::uint64_t sum_high = linear_high + triangular_high + (sum_low < linear_low ? 1U : 0U);
    result.attention_pairs = sum_high != 0 ? std::numeric_limits<std::uint64_t>::max() : sum_low;
    return result;
}

enum class ContextResourceClass : std::uint8_t {
    State,
    MainKV,
    BackendKV,
};

enum class ContextTransferDirection : std::uint8_t {
    DeviceToHost,
    HostToDevice,
    DeviceToDevice,
};

struct ContextTransferObservation {
    ContextResourceClass resource      = ContextResourceClass::State;
    ContextTransferDirection direction = ContextTransferDirection::DeviceToHost;
    std::uint64_t units                = 0; // State images for State; bytes for typed KV.
    std::uint32_t page_count           = 0;
    TransferWork work;
    std::uint64_t elapsed_ns = 0;
};

struct ContextTransferRequirement {
    ContextResourceClass resource      = ContextResourceClass::State;
    ContextTransferDirection direction = ContextTransferDirection::DeviceToHost;
    std::uint64_t units                = 0;
    std::uint32_t page_count           = 0;
    TransferWork work;

    [[nodiscard]] friend constexpr bool operator==(ContextTransferRequirement,
                                                   ContextTransferRequirement) noexcept = default;
};

struct ContextOperationCounts {
    std::uint64_t state_moves            = 0;
    std::uint64_t state_forks            = 0;
    std::uint64_t state_restores         = 0;
    std::uint64_t pressure_spill_pages   = 0;
    std::uint64_t partial_tail_cow_pages = 0;
    std::uint64_t historical_fork_hits   = 0;
};

enum class Readiness : std::uint8_t {
    Ready,
    NeedsTransfer,
    TemporarilyBlocked,
    PermanentlyInfeasible,
};

enum class ContextTransactionStatus : std::uint8_t {
    InProgress,
    Published,
    Aborted,
};

enum class ContextTransactionReserveStatus : std::uint8_t {
    Reserved,
    Aborted,
};

struct ContextTransactionInProgress {};

enum class ContextTransactionKind : std::uint8_t {
    Materialization,
    ActiveCapture,
};

enum class PreflightStatus : std::uint8_t {
    Ready,
    StalePolicyState,
    InvariantFailure,
};

enum class CheckpointKind : std::uint8_t {
    SessionEndpoint,
    TurnClosure,
    ResponseReplay,
    SharedStablePrefix,
    LongAnchor,
};

enum class CheckpointScope : std::uint8_t {
    Private,
    Shared,
};

enum class ReplicaResidency : std::uint8_t {
    DeviceOnly,
    HostOnly,
    Both,
};

enum class RetentionClass : std::uint8_t {
    SharedStable,
    LiveSession,
    RecentPrivate,
    Disposable,
};

enum class LogicalOwnerKind : std::uint8_t {
    PrivateContinuation,
    SharedPrefix,
};

struct LogicalOwnerKey {
    LogicalOwnerKind kind = LogicalOwnerKind::PrivateContinuation;
    std::uint64_t id      = 0;

    [[nodiscard]] friend constexpr bool operator==(LogicalOwnerKey,
                                                   LogicalOwnerKey) noexcept = default;
};

struct CatalogCapability {
    LogicalOwnerKey owner;
    std::uint32_t slot       = std::numeric_limits<std::uint32_t>::max();
    std::uint64_t generation = 0;

    [[nodiscard]] friend constexpr bool operator==(CatalogCapability,
                                                   CatalogCapability) noexcept = default;
};

struct PlanningOwnerId {
    std::uint32_t value = std::numeric_limits<std::uint32_t>::max();

    [[nodiscard]] friend constexpr bool operator==(PlanningOwnerId,
                                                   PlanningOwnerId) noexcept = default;
};

struct PlanningCandidateId {
    std::uint32_t value = std::numeric_limits<std::uint32_t>::max();

    [[nodiscard]] friend constexpr bool operator==(PlanningCandidateId,
                                                   PlanningCandidateId) noexcept = default;
};

struct ProgramResourceRevision {
    std::uint64_t value = 0;

    [[nodiscard]] friend constexpr bool operator==(ProgramResourceRevision,
                                                   ProgramResourceRevision) noexcept = default;
};

// ResourceManager-owned final schedule policy. Program validates and consumes this view while
// sealing the already assessed physical target; ResourcePlan is immutable after that boundary.
struct FinalScheduleIntent {
    std::span<const std::uint32_t> shared_capture_frontiers;
};

enum class PrivateSourceMode : std::uint8_t {
    Retain,
    ConsumeToActive,
};

enum class VictimDisposition : std::uint8_t {
    Retained,
    Evicted,
};

enum class FinishDisposition : std::uint8_t {
    Catalogued,
    Released,
};

struct CheckpointRef {
    CheckpointKind kind    = CheckpointKind::SessionEndpoint;
    std::uint32_t frontier = 0;
    // Singleton checkpoint kinds use zero. LongAnchor uses a nonzero, per-continuation slot.
    std::uint32_t ordinal = 0;

    [[nodiscard]] friend constexpr bool operator==(CheckpointRef, CheckpointRef) noexcept = default;
};

struct Revision {
    std::uint64_t value = 0;

    [[nodiscard]] friend constexpr bool operator==(Revision, Revision) noexcept = default;
};

enum class MaterializationPhysicalStatus : std::uint8_t {
    Feasible,
    Infeasible,
    StructuralInvalid,
};

inline constexpr std::size_t kContextTransferDirectionCount = 3;
using CoalescedTransferWork = std::array<TransferWork, kContextTransferDirectionCount>;

// Exact, unpriced machine work for one complete materialization projection. Program owns this
// physical fact; the common search runner applies the immutable planning cost model exactly once.
// `optimistic_candidate_transfers` is ordering evidence only and never proves feasibility.
struct MaterializationMachineWork {
    CoalescedTransferWork pressure_transfers;
    CoalescedTransferWork candidate_transfers;
    CoalescedTransferWork optimistic_candidate_transfers;
    PrefillWork remaining_prefill_work;
    std::uint32_t reused_prompt_tokens = 0;

    [[nodiscard]] friend constexpr bool
    operator==(const MaterializationMachineWork&,
               const MaterializationMachineWork&) noexcept = default;
};

// One exact recovery recipe. Program enumerates every supported alternative; pricing policy
// selects the cheapest alternative without changing physical legality or the target graph.
struct CheckpointRecoveryAlternativeWork {
    CoalescedTransferWork transfers;
    PrefillWork prefill;

    [[nodiscard]] friend constexpr bool
    operator==(const CheckpointRecoveryAlternativeWork&,
               const CheckpointRecoveryAlternativeWork&) noexcept = default;
};

struct IdentityMaterializationAssessment {
    MaterializationPhysicalStatus physical_status =
        MaterializationPhysicalStatus::StructuralInvalid;
    PrivateSourceMode source_mode = PrivateSourceMode::ConsumeToActive;
    MaterializationMachineWork machine_work;
    bool pressure_may_change_machine_work = false;
    bool expandable                       = false;
    std::uint64_t projection_work         = 0;
    std::uint64_t assessment_digest       = 0;

    [[nodiscard]] friend constexpr bool
    operator==(const IdentityMaterializationAssessment&,
               const IdentityMaterializationAssessment&) noexcept = default;
};

struct PressureCheckpointRecoveryImpact {
    PlanningOwnerId owner;
    CheckpointRef checkpoint;
    std::span<const CheckpointRecoveryAlternativeWork> target_recovery_work;
    bool survives = true;

    [[nodiscard]] friend constexpr bool
    operator==(const PressureCheckpointRecoveryImpact&,
               const PressureCheckpointRecoveryImpact&) noexcept = default;
};

struct PressureCheckpointOutcome {
    PlanningOwnerId owner;
    CheckpointRef checkpoint;
    bool survives = true;

    [[nodiscard]] friend constexpr bool operator==(PressureCheckpointOutcome,
                                                   PressureCheckpointOutcome) noexcept = default;
};

struct PressureOwnerOutcome {
    PlanningOwnerId owner;
    VictimDisposition disposition     = VictimDisposition::Retained;
    std::uint32_t degradation_units   = 0;
    std::uint32_t dropped_checkpoints = 0;

    [[nodiscard]] friend constexpr bool operator==(const PressureOwnerOutcome&,
                                                   const PressureOwnerOutcome&) noexcept = default;
};

// Cheap, target-neutral ordering evidence for an unassessed pressure target.  This is deliberately
// not a feasibility certificate: only PressureTargetAssessment may admit or seal a target.  The
// Program owns the physical projection and the common planner combines the owner outcomes with its
// retention policy.
struct PressurePhysicalGuidance {
    std::uint32_t unsatisfied_constraints   = 0;
    std::uint32_t estimated_remaining_steps = 0;
    std::uint64_t normalized_residual_q20   = 0;
    // Aggregate byte relief cannot resolve an extent/ordered-stage geometry question.
    bool requires_exact_feedback = false;
};

struct PressureOwnerRecoveryGuidance {
    PlanningOwnerId owner;
    CoalescedTransferWork additional_restore;
};

struct PressureConstructionOptionId {
    std::uint32_t cursor_generation = 0;
    std::uint32_t scan_generation   = 0;
    std::uint32_t index             = 0;
};

// The spans are borrowed from a PressurePlanningSession scratch generation and remain valid only
// until the next session operation.  The common planner folds them immediately into owning values.
struct PressureTargetGuidance {
    PressurePhysicalGuidance physical;
    MaterializationMachineWork estimated_machine_work;
    std::span<const PressureOwnerOutcome> owner_outcomes;
    PlanningCandidateId candidate;
    std::uint32_t stable_target_ordinal = 0;
    std::uint32_t degradation_units     = 0;
    std::uint32_t dropped_checkpoints   = 0;
    PrivateSourceMode source_mode       = PrivateSourceMode::ConsumeToActive;
    std::span<const PressureCheckpointOutcome> checkpoint_changes;
    std::span<const PressureOwnerRecoveryGuidance> recovery_estimates;
    bool recovery_estimate_complete = false;
};

// One resumable scan operation: one owner's successor generation or one option summary.
// Guidance spans expire at the next session call; the ID remains valid until choose/reset.
struct PressureConstructionStep {
    std::optional<PressureTargetGuidance> guidance;
    PressureConstructionOptionId option;
    bool exhausted = false;
};

// The spans are borrowed from a PressurePlanningSession scratch generation and remain valid only
// until the next session mutation. The common planner folds them immediately into owning values.
struct PressureTargetAssessment {
    MaterializationPhysicalStatus physical_status =
        MaterializationPhysicalStatus::StructuralInvalid;
    PrivateSourceMode source_mode = PrivateSourceMode::ConsumeToActive;
    MaterializationMachineWork machine_work;
    std::span<const PressureOwnerOutcome> owner_outcomes;
    std::span<const PressureCheckpointRecoveryImpact> checkpoint_impacts;
    PlanningCandidateId candidate;
    std::uint32_t stable_target_ordinal = 0;
    std::uint32_t degradation_units     = 0;
    std::uint32_t dropped_checkpoints   = 0;
    std::uint64_t projection_work       = 0;
    std::uint64_t assessment_digest     = 0;
    bool expandable                     = false;
    bool root_maximal                   = false;
};

// Target-produced affine reservation curve for one Main KV physical-capacity axis. The byte
// values come from complete target physical layout plans, not from a model geometry formula in
// the common runtime.
struct SequenceCapacityCurve {
    std::uint32_t main_page_tokens                   = 0;
    std::uint32_t minimum_main_page_groups           = 0;
    std::uint32_t maximum_main_page_groups           = 0;
    std::size_t minimum_device_reservation_bytes     = 0;
    std::size_t bytes_per_additional_main_page_group = 0;

    [[nodiscard]] std::size_t reservation_bytes(std::uint32_t main_page_groups) const;
    [[nodiscard]] std::uint32_t resolved_tokens(std::uint32_t main_page_groups) const;
};

struct KvCapacityResolution {
    KvCapacityMode mode                              = KvCapacityMode::Explicit;
    std::uint32_t main_page_groups                   = 0;
    std::uint32_t maximum_main_page_groups           = 0;
    std::uint32_t resolved_tokens                    = 0;
    std::size_t minimum_runtime_reservation_bytes    = 0;
    std::size_t bytes_per_additional_main_page_group = 0;
    std::size_t runtime_reservation_bytes            = 0;
    std::size_t available_after_weights_bytes        = 0;
    std::size_t available_after_startup_bytes        = 0;
    std::size_t automatic_headroom_bytes             = 0;
    std::size_t planned_slack_bytes                  = 0;
};

} // namespace ninfer::runtime
