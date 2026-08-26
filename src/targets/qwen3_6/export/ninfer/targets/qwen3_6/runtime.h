#pragma once

#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include <cstddef>
#include <cstdint>
#include <array>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer {
struct DeviceContext;
}

namespace ninfer::targets::qwen3_6 {

namespace detail {
struct PressureOptionImpl;
struct CaptureAssessmentImpl;

} // namespace detail

// Read-only diagnostics sampled from the real Program stores.  This is not an accounting input.
struct PhysicalUsageSnapshot {
    std::uint64_t resource_revision       = 0;
    std::uint32_t device_state_slots      = 0;
    std::uint32_t host_state_slots        = 0;
    std::uint32_t device_main_kv_pages    = 0;
    std::uint32_t device_backend_kv_pages = 0;
    std::size_t host_kv_bytes             = 0;

    [[nodiscard]] friend constexpr bool operator==(const PhysicalUsageSnapshot&,
                                                   const PhysicalUsageSnapshot&) noexcept = default;
};

enum class TextPhase {
    Prefill,
    Verify,
};

struct GraphExecutionProfile {
    std::uint32_t min            = 0;
    std::uint32_t max            = 0;
    std::uint32_t topology_class = 0;
};

// Program-minted shortlist metadata. It only narrows catalog inspection; Program still performs
// exact token, position, media and runtime-mode verification before a checkpoint can be selected.
struct PrefixShortlistKey {
    std::uint64_t digest       = 0;
    std::uint32_t frontier     = 0;
    std::uint32_t identity_tag = 0;

    [[nodiscard]] friend constexpr bool operator==(PrefixShortlistKey,
                                                   PrefixShortlistKey) noexcept = default;
};

struct TargetKVRequirement {
    std::uint32_t main_frontier    = 0;
    std::uint32_t backend_frontier = 0;
    std::uint32_t main_pages       = 0;
    std::uint32_t backend_pages    = 0;

    [[nodiscard]] friend constexpr bool operator==(TargetKVRequirement,
                                                   TargetKVRequirement) noexcept = default;
};

struct CheckpointSummary {
    runtime::CheckpointRef ref;
    runtime::CheckpointScope scope = runtime::CheckpointScope::Private;
    PrefixShortlistKey shortlist_key;
    runtime::ReplicaResidency state_residency = runtime::ReplicaResidency::DeviceOnly;
    TargetKVRequirement required_kv;
    runtime::PrefillWork rebuild_work;

    [[nodiscard]] friend bool operator==(const CheckpointSummary&,
                                         const CheckpointSummary&) noexcept = default;
};

struct ContinuationSummary {
    std::optional<CheckpointSummary> endpoint;
    std::optional<CheckpointSummary> rewrite;
    std::vector<CheckpointSummary> long_anchors;
    std::uint32_t active_references = 0;

    [[nodiscard]] friend bool operator==(const ContinuationSummary&,
                                         const ContinuationSummary&) noexcept = default;
};

struct SharedPrefixSummary {
    CheckpointSummary checkpoint;
    std::uint32_t active_references = 0;

    [[nodiscard]] friend bool operator==(const SharedPrefixSummary&,
                                         const SharedPrefixSummary&) noexcept = default;
};

enum class PressureStateAction : std::uint8_t {
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

enum class PressureKVActionKind : std::uint8_t {
    None,
    DropDeviceDuplicate,
    DemoteToHost,
    DropHostDuplicate,
};

struct PressureKVAction {
    std::uint32_t begin_page  = 0;
    std::uint32_t page_count  = 0;
    PressureKVActionKind kind = PressureKVActionKind::None;

    [[nodiscard]] friend constexpr bool operator==(PressureKVAction,
                                                   PressureKVAction) noexcept = default;
};

// Program-owned description of the policy loss caused by one pressure action. The common
// ResourceManager consumes static costs, but never reconstructs which target-private checkpoint
// depends on a StateImage or typed KV region.
struct PressureCheckpointImpact {
    runtime::CheckpointRef checkpoint;
    runtime::PrefillWork fallback_rebuild_work;
    std::vector<runtime::ContextTransferRequirement> current_restore_requirements;
    std::vector<runtime::ContextTransferRequirement> fallback_restore_requirements;
    std::vector<runtime::ContextTransferRequirement> added_restore_requirements;
    bool drops_checkpoint = false;

    [[nodiscard]] friend bool operator==(const PressureCheckpointImpact&,
                                         const PressureCheckpointImpact&) noexcept = default;
};

// The marginal recovery value contributed by one Host replica. Program supplies the exact
// affected checkpoint and its nearest surviving fallback; ResourceManager applies the
// checkpoint's retention observation and startup-selected transfer/prefill costs.
struct ReplicaValueImpact {
    runtime::CheckpointRef checkpoint;
    runtime::PrefillWork fallback_rebuild_work;
    std::vector<runtime::ContextTransferRequirement> fallback_restore_requirements;
    std::vector<runtime::ContextTransferRequirement> host_restore_requirements;

    [[nodiscard]] friend bool operator==(const ReplicaValueImpact&,
                                         const ReplicaValueImpact&) noexcept = default;
};

// A target-minted, side-effect-free pressure alternative for one private continuation. Common
// policy compares only its exact effect/cost fields and returns the value unchanged to Program;
// page positions and state disposition remain Qwen-family semantics.
struct PressureOption {
    PressureOption();

    std::uint64_t id          = 0;
    PressureStateAction state = PressureStateAction::None;
    PressureKVAction main_kv;
    PressureKVAction backend_kv;
    std::optional<runtime::CheckpointRef> dropped_checkpoint;
    // Opaque target-private planner payload. Its complete type exists only in Program code.
    std::shared_ptr<detail::PressureOptionImpl> implementation;
    std::uint64_t transfer_bytes = 0;
    std::vector<runtime::ContextTransferRequirement> transfer_requirements;
    std::vector<PressureCheckpointImpact> checkpoint_impacts;
    std::vector<ReplicaValueImpact> removed_host_replica_impacts;
    bool evicts_continuation = false;
    bool shared_owner        = false;

    friend bool operator==(const PressureOption&, const PressureOption&) noexcept;
};

namespace detail {
template <class Variant>
struct SequencePlanImpl;
template <class Variant>
struct SequencePlannerImpl;
template <class Variant>
struct AdmissionPlanImpl;
template <class Variant>
struct RequestBasePlanImpl;
template <class Variant>
class ProgramImpl;
template <class Variant>
struct RuntimeContractAccess;
} // namespace detail

template <class Variant>
class SequencePlanner;
template <class Variant>
class Program;

// These are the complete family execution types. Exact packages bind them to a private Variant;
// target selection remains outside this layer and happens once in the closed Engine registry.
template <class Variant>
class SequencePlan {
public:
    SequencePlan(SequencePlan&&) noexcept;
    SequencePlan& operator=(SequencePlan&&) noexcept;
    ~SequencePlan();

    SequencePlan(const SequencePlan&)            = delete;
    SequencePlan& operator=(const SequencePlan&) = delete;

    [[nodiscard]] std::uint32_t capacity() const noexcept;
    [[nodiscard]] std::uint32_t kv_capacity() const noexcept;
    [[nodiscard]] std::uint32_t max_concurrency() const noexcept;
    [[nodiscard]] std::size_t device_reservation_bytes() const noexcept;
    [[nodiscard]] std::size_t workspace_capacity_bytes() const noexcept;

public:
    // Family-private construction/storage seam; exact packages expose only the completed alias.
    explicit SequencePlan(std::unique_ptr<detail::SequencePlanImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::SequencePlanImpl<Variant>> impl_;

    template <class V>
    friend class SequencePlanner;
    template <class V>
    friend class detail::ProgramImpl;
};

template <class Variant>
class SequencePlanner {
public:
    SequencePlanner(SequencePlanner&&) noexcept;
    SequencePlanner& operator=(SequencePlanner&&) noexcept;
    ~SequencePlanner();

    SequencePlanner(const SequencePlanner&)            = delete;
    SequencePlanner& operator=(const SequencePlanner&) = delete;

    [[nodiscard]] const runtime::SequenceCapacityCurve& capacity_curve() const noexcept;
    [[nodiscard]] SequencePlan<Variant> finalize(std::uint32_t main_page_groups) &&;

public:
    explicit SequencePlanner(std::unique_ptr<detail::SequencePlannerImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::SequencePlannerImpl<Variant>> impl_;

    template <class V>
    friend SequencePlanner<V> make_sequence_planner(DeviceContext&, const EngineOptions&,
                                                    typename V::WeightsProfile);
};

template <class Variant>
class RequestBasePlan {
public:
    RequestBasePlan(RequestBasePlan&&) noexcept;
    RequestBasePlan& operator=(RequestBasePlan&&) noexcept;
    ~RequestBasePlan();

    RequestBasePlan(const RequestBasePlan&)            = delete;
    RequestBasePlan& operator=(const RequestBasePlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;
    [[nodiscard]] const PreparedContextCache& context_cache() const noexcept;
    [[nodiscard]] std::optional<PrefixShortlistKey>
    prefix_shortlist_key(std::uint32_t frontier) const noexcept;

public:
    explicit RequestBasePlan(std::unique_ptr<detail::RequestBasePlanImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::RequestBasePlanImpl<Variant>> impl_;
};

template <class Variant>
class AdmissionPlan {
public:
    AdmissionPlan(AdmissionPlan&&) noexcept;
    AdmissionPlan& operator=(AdmissionPlan&&) noexcept;
    ~AdmissionPlan();

    AdmissionPlan(const AdmissionPlan&)            = delete;
    AdmissionPlan& operator=(const AdmissionPlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;
    [[nodiscard]] runtime::ClaimDisposition source_disposition() const noexcept;
    [[nodiscard]] bool needs_transfer() const noexcept;
    [[nodiscard]] runtime::PrefillWork remaining_prefill_work() const noexcept;
    [[nodiscard]] std::span<const runtime::ContextTransferRequirement>
    transfer_requirements() const noexcept;

public:
    // Family-private construction/storage seam. Exact packages expose only the completed alias;
    // Engine code can inspect summary() but not target planning state.
    explicit AdmissionPlan(std::unique_ptr<detail::AdmissionPlanImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::AdmissionPlanImpl<Variant>> impl_;
};

// A sealed Program-owned physical decision.  ResourceManager may retain it and inspect the
// request-level summary, but cannot see allocator quantities, references, reservations, or stage
// deltas.  Start validates the bound Program revision before performing any mutation.
template <class Variant>
class ResourcePlan {
public:
    ResourcePlan(ResourcePlan&&) noexcept            = default;
    ResourcePlan& operator=(ResourcePlan&&) noexcept = default;
    ~ResourcePlan()                                  = default;

    ResourcePlan(const ResourcePlan&)            = delete;
    ResourcePlan& operator=(const ResourcePlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept {
        return admission_.summary();
    }

    [[nodiscard]] bool needs_transfer() const noexcept { return admission_.needs_transfer(); }

    [[nodiscard]] std::uint64_t resource_revision() const noexcept { return revision_; }

private:
    ResourcePlan(AdmissionPlan<Variant>&& admission, std::uint64_t revision) noexcept
        : admission_(std::move(admission)), revision_(revision) {}

    AdmissionPlan<Variant> admission_;
    std::uint64_t revision_ = 0;

    friend class Program<Variant>;
};

// A Program-minted proof that one FIFO borrower cannot consume the maximum physical entitlement
// reserved for the blocked head.  Common scheduling binds the opaque proof to logical identities
// and a revision; it cannot inspect or reproduce the resource arithmetic.
template <class Variant>
class PersistentBackfillProof {
public:
    PersistentBackfillProof(PersistentBackfillProof&&) noexcept            = default;
    PersistentBackfillProof& operator=(PersistentBackfillProof&&) noexcept = default;

    PersistentBackfillProof(const PersistentBackfillProof&)            = delete;
    PersistentBackfillProof& operator=(const PersistentBackfillProof&) = delete;

    [[nodiscard]] std::uint64_t resource_revision() const noexcept { return revision_; }

private:
    explicit PersistentBackfillProof(std::uint64_t revision) noexcept : revision_(revision) {}

    std::uint64_t revision_ = 0;

    friend class Program<Variant>;
};

template <class Variant>
class SequenceHandle {
public:
    SequenceHandle() noexcept                                 = default;
    SequenceHandle(const SequenceHandle&) noexcept            = default;
    SequenceHandle& operator=(const SequenceHandle&) noexcept = default;

private:
    const void* owner_ = nullptr;
    runtime::LaneId lane_{};
    std::uint64_t epoch_ = 0;

    friend struct detail::RuntimeContractAccess<Variant>;
};

template <class Variant>
class ContinuationHandle {
public:
    ContinuationHandle() noexcept = default;
    ~ContinuationHandle()         = default;

    ContinuationHandle(ContinuationHandle&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), index_(other.index_),
          generation_(std::exchange(other.generation_, 0)) {}

    ContinuationHandle& operator=(ContinuationHandle&&)      = delete;
    ContinuationHandle(const ContinuationHandle&)            = delete;
    ContinuationHandle& operator=(const ContinuationHandle&) = delete;

private:
    const void* owner_        = nullptr;
    std::uint32_t index_      = 0;
    std::uint64_t generation_ = 0;

    friend struct detail::RuntimeContractAccess<Variant>;
};

template <class Variant>
class SharedPrefixHandle {
public:
    SharedPrefixHandle() noexcept = default;
    ~SharedPrefixHandle()         = default;

    SharedPrefixHandle(SharedPrefixHandle&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), index_(other.index_),
          generation_(std::exchange(other.generation_, 0)) {}

    SharedPrefixHandle& operator=(SharedPrefixHandle&&)      = delete;
    SharedPrefixHandle(const SharedPrefixHandle&)            = delete;
    SharedPrefixHandle& operator=(const SharedPrefixHandle&) = delete;

private:
    const void* owner_        = nullptr;
    std::uint32_t index_      = 0;
    std::uint64_t generation_ = 0;

    friend struct detail::RuntimeContractAccess<Variant>;
};

template <class Variant>
class CaptureOffer {
public:
    CaptureOffer() noexcept = default;
    ~CaptureOffer()         = default;

    CaptureOffer(CaptureOffer&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), lane_(other.lane_), epoch_(other.epoch_),
          id_(std::exchange(other.id_, 0)) {}

    CaptureOffer& operator=(CaptureOffer&&)      = delete;
    CaptureOffer(const CaptureOffer&)            = delete;
    CaptureOffer& operator=(const CaptureOffer&) = delete;

private:
    const void* owner_ = nullptr;
    runtime::LaneId lane_{};
    std::uint64_t epoch_ = 0;
    std::uint64_t id_    = 0;

    friend struct detail::RuntimeContractAccess<Variant>;
};

template <class Variant>
class PendingBatch {
public:
    PendingBatch() noexcept = default;
    ~PendingBatch()         = default;

    PendingBatch(PendingBatch&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)),
          transaction_(std::exchange(other.transaction_, 0)), rows_(other.rows_),
          row_count_(std::exchange(other.row_count_, 0)), tokens_(other.tokens_),
          row_counts_(other.row_counts_), row_stride_(other.row_stride_), timing_(other.timing_) {
        other.tokens_     = {};
        other.row_counts_ = {};
        other.row_stride_ = 0;
        other.timing_     = {};
    }

    PendingBatch& operator=(PendingBatch&&)      = delete;
    PendingBatch(const PendingBatch&)            = delete;
    PendingBatch& operator=(const PendingBatch&) = delete;

    [[nodiscard]] std::size_t row_count() const noexcept { return row_count_; }

    [[nodiscard]] std::span<const TokenId> tokens() const noexcept { return tokens_; }

    [[nodiscard]] std::span<const std::int32_t> row_counts() const noexcept { return row_counts_; }

    [[nodiscard]] std::uint32_t row_stride() const noexcept { return row_stride_; }

    [[nodiscard]] runtime::ExecutionTiming execution_timing() const noexcept { return timing_; }

private:
    const void* owner_         = nullptr;
    std::uint64_t transaction_ = 0;
    std::array<SequenceHandle<Variant>, kMaximumConcurrency> rows_{};
    std::size_t row_count_ = 0;
    std::span<const TokenId> tokens_;
    std::span<const std::int32_t> row_counts_;
    std::uint32_t row_stride_ = 0;
    runtime::ExecutionTiming timing_;

    friend struct detail::RuntimeContractAccess<Variant>;
};

template <class Variant>
struct PrefillProgress {
    runtime::BeginSummary summary;
    std::uint32_t processed_prompt_tokens = 0;
    bool complete                         = false;
    runtime::ExecutionTiming timing;
    std::optional<PendingBatch<Variant>> pending;
    std::optional<CaptureOffer<Variant>> capture;
};

enum class CaptureStatePlacement : std::uint8_t {
    DeviceFork,
    HostSnapshot,
};

struct CaptureAssessment {
    CaptureAssessment();

    // Program retains the physical assessment in this opaque package-private payload.
    std::shared_ptr<detail::CaptureAssessmentImpl> implementation;
    PrefixShortlistKey shortlist_key;
    runtime::PrefillWork protected_rebuild_work;
    std::vector<runtime::ContextTransferRequirement> transfer_requirements;
    std::vector<PressureCheckpointImpact> replacement_impacts;
    std::vector<runtime::CheckpointRef> private_replacement_candidates;
    std::uint32_t frontier                = 0;
    bool publishes_private                = false;
    bool publishes_shared                 = false;
    bool needs_transfer                   = false;
    bool recycles_private_state           = false;
    CaptureStatePlacement state_placement = CaptureStatePlacement::DeviceFork;
};

template <class Variant>
struct SharedPrefixPublication {
    SharedPrefixHandle<Variant> handle;
    SharedPrefixSummary summary;
};

template <class Variant>
struct ActiveCaptureResult {
    runtime::ContextTransactionStatus status = runtime::ContextTransactionStatus::Aborted;
    bool capacity_preparation_committed      = false;
    ContinuationSummary active_summary;
    std::optional<SharedPrefixPublication<Variant>> shared;
    std::vector<runtime::ContextTransferObservation> transfer_observations;
    runtime::ContextOperationCounts operations;
};

template <class Variant>
struct StartResult {
    SequenceHandle<Variant> sequence;
};

struct MaterializationVictimResult {
    runtime::ClaimDisposition disposition = runtime::ClaimDisposition::Retained;
    std::optional<ContinuationSummary> final_summary;
};

struct MaterializationSharedVictimResult {
    runtime::ClaimDisposition disposition = runtime::ClaimDisposition::Retained;
    std::optional<SharedPrefixSummary> final_summary;
};

struct MaterializationSourceResult {
    runtime::ClaimDisposition disposition = runtime::ClaimDisposition::Retained;
    std::optional<ContinuationSummary> final_summary;
};

struct MaterializationSharedSourceResult {
    runtime::ClaimDisposition disposition = runtime::ClaimDisposition::Retained;
    std::optional<SharedPrefixSummary> final_summary;
};

template <class Variant>
struct MaterializationResult {
    runtime::ContextTransactionStatus status = runtime::ContextTransactionStatus::Aborted;
    std::optional<StartResult<Variant>> published;
    std::optional<MaterializationSourceResult> source;
    std::optional<MaterializationSharedSourceResult> shared_source;
    std::vector<MaterializationVictimResult> victims;
    std::vector<MaterializationSharedVictimResult> shared_victims;
    std::vector<runtime::ContextTransferObservation> transfer_observations;
    runtime::ContextOperationCounts operations;
};

template <class Variant>
using ContextTransactionProgress =
    std::variant<runtime::ContextTransactionInProgress, MaterializationResult<Variant>,
                 ActiveCaptureResult<Variant>>;

struct CommitRowResult {
    runtime::CommitDisposition disposition = runtime::CommitDisposition::Active;
    GenerationTimings timings;
    SpeculativeStats speculative;
};

template <class Variant>
struct CommitResult {
    std::array<CommitRowResult, kMaximumConcurrency> rows{};
    // A prompt-frontier capture becomes valid only after the generated Begin token is committed.
    // Keeping the move-only capability row-aligned avoids exposing provisional prompt state.
    std::array<std::optional<CaptureOffer<Variant>>, kMaximumConcurrency> captures{};
    std::size_t row_count = 0;
    runtime::ExecutionTiming timing;
};

template <class Variant>
struct DiscardResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    std::size_t row_count         = 0;
};

template <class Variant>
struct FinishResult {
    runtime::ConsumeStatus status          = runtime::ConsumeStatus::InvariantMismatch;
    runtime::FinishDisposition disposition = runtime::FinishDisposition::Released;
    GenerationTimings timings;
    SpeculativeStats speculative;
    ContinuationSummary summary;
    std::optional<ContinuationHandle<Variant>> continuation;
};

template <class Variant>
struct AbortResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    GenerationTimings timings;
    SpeculativeStats speculative;
};

template <class Variant>
struct ReleaseResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
};

template <class Variant>
class Program {
public:
    ~Program() noexcept;

    Program(const Program&)            = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&&)                 = delete;
    Program& operator=(Program&&)      = delete;

    // Engine owns scheduling and logical residency policy. Program owns physical lanes, opaque
    // capabilities, model state and one immutable pending transaction at a time.
    [[nodiscard]] RequestBasePlan<Variant>
    plan_request(const PreparedPrompt& prompt, const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] std::optional<AdmissionPlan<Variant>>
    inspect_admission(const PreparedPrompt& prompt, const RequestBasePlan<Variant>& base,
                      runtime::LaneId destination, const ContinuationHandle<Variant>* source,
                      const SharedPrefixHandle<Variant>* shared_source,
                      std::optional<runtime::CheckpointRef> checkpoint,
                      bool must_retain_private_source);
    [[nodiscard]] std::vector<PressureOption>
    inspect_pressure_options(const AdmissionPlan<Variant>& admission,
                             const ContinuationHandle<Variant>& continuation) const;
    [[nodiscard]] PressureOption
    inspect_eviction_option(const ContinuationHandle<Variant>& continuation) const;
    [[nodiscard]] std::vector<PressureOption>
    inspect_shared_pressure_options(const AdmissionPlan<Variant>& admission,
                                    const SharedPrefixHandle<Variant>& shared) const;
    [[nodiscard]] PressureOption
    inspect_shared_eviction_option(const SharedPrefixHandle<Variant>& shared) const;
    [[nodiscard]] std::optional<ResourcePlan<Variant>>
    seal_resource_plan(const AdmissionPlan<Variant>& admission, const PreparedPrompt& prompt,
                       std::span<const ContinuationHandle<Variant>* const> pressure_owners,
                       std::span<const PressureOption> pressure_options,
                       std::span<const SharedPrefixHandle<Variant>* const> shared_pressure_owners,
                       std::span<const PressureOption> shared_pressure_options);
    [[nodiscard]] runtime::ContextTransactionReserveStatus
    start_resource_transaction(ResourcePlan<Variant>&& plan, PreparedPrompt&& prompt,
                               runtime::CancellationFlagView cancellation);
    [[nodiscard]] std::optional<PersistentBackfillProof<Variant>>
    prove_persistent_backfill(const RequestBasePlan<Variant>& blocked_head,
                              const ResourcePlan<Variant>& candidate,
                              std::span<const SequenceHandle<Variant>> persistent_borrowers) const;
    [[nodiscard]] ContextTransactionProgress<Variant>
    progress_context_transaction(runtime::CancellationFlagView cancellation);
    void finalize_context_transaction() noexcept;
    [[nodiscard]] bool has_context_transaction() const noexcept;
    [[nodiscard]] PrefillProgress<Variant>
    advance_prefill(SequenceHandle<Variant> sequence,
                    runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] CaptureAssessment
    inspect_capture(const CaptureOffer<Variant>& offer,
                    const SharedPrefixHandle<Variant>* exact_shared,
                    const SharedPrefixHandle<Variant>* replacement,
                    std::optional<runtime::CheckpointRef> private_replacement) const;
    [[nodiscard]] bool shared_capture_matches(const CaptureOffer<Variant>& offer,
                                              const SharedPrefixHandle<Variant>& shared) const;
    void skip_capture(CaptureOffer<Variant>&& offer);
    [[nodiscard]] runtime::ContextTransactionReserveStatus
    reserve_active_capture(CaptureOffer<Variant>&& offer,
                           const SharedPrefixHandle<Variant>* exact_shared,
                           const SharedPrefixHandle<Variant>* replacement,
                           std::optional<runtime::CheckpointRef> private_replacement,
                           runtime::CancellationFlagView cancellation);
    [[nodiscard]] PendingBatch<Variant> decode(std::span<const SequenceHandle<Variant>> sequences,
                                               std::span<const runtime::RoundBudget> budgets,
                                               runtime::ExecutionTiming* failed_timing = nullptr);
    // Advance each live sequence with its exact target-owned token row. This does not sample or
    // advance sampler RNG/occurrence state; callers own output publication and budget accounting.
    [[nodiscard]] runtime::ExecutionTiming
    append_forced_tokens(std::span<const SequenceHandle<Variant>> sequences,
                         std::span<const TokenId> row_major_tokens, std::uint32_t row_stride,
                         runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] CommitResult<Variant>
    commit(PendingBatch<Variant>&& pending, std::span<const runtime::CommitDecision> decisions,
           runtime::CommitObservation observation  = runtime::CommitObservation::AllRows,
           runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] DiscardResult<Variant> abort_pending(PendingBatch<Variant>&& pending) noexcept;
    [[nodiscard]] FinishResult<Variant> finish(SequenceHandle<Variant> sequence) noexcept;
    [[nodiscard]] AbortResult<Variant> abort(SequenceHandle<Variant> sequence) noexcept;
    [[nodiscard]] ReleaseResult<Variant>
    release_continuation(ContinuationHandle<Variant>&& continuation) noexcept;
    [[nodiscard]] ReleaseResult<Variant>
    release_shared_prefix(SharedPrefixHandle<Variant>&& shared) noexcept;
    void fail_all_cleanup() noexcept;

    [[nodiscard]] bool
    isolated_request_feasible(const RequestBasePlan<Variant>& base) const noexcept;
    [[nodiscard]] std::uint64_t resource_revision() const noexcept;
    [[nodiscard]] PhysicalUsageSnapshot physical_usage() const noexcept;
    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    void reset_memory_peaks() noexcept;

private:
    explicit Program(std::unique_ptr<detail::ProgramImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::ProgramImpl<Variant>> impl_;

    template <class V>
    friend std::unique_ptr<Program<V>> create_program(const typename V::ModelView&,
                                                      typename V::WeightsProfile, SequencePlan<V>&&,
                                                      DeviceContext&);
};

namespace detail {

template <class Variant>
struct RuntimeContractAccess {
    [[nodiscard]] static SequenceHandle<Variant>
    make_sequence(const void* owner, runtime::LaneId lane, std::uint64_t epoch) noexcept {
        SequenceHandle<Variant> out;
        out.owner_ = owner;
        out.lane_  = lane;
        out.epoch_ = epoch;
        return out;
    }

    [[nodiscard]] static ContinuationHandle<Variant>
    make_continuation(const void* owner, std::uint32_t index, std::uint64_t generation) noexcept {
        ContinuationHandle<Variant> out;
        out.owner_      = owner;
        out.index_      = index;
        out.generation_ = generation;
        return out;
    }

    [[nodiscard]] static SharedPrefixHandle<Variant>
    make_shared_prefix(const void* owner, std::uint32_t index, std::uint64_t generation) noexcept {
        SharedPrefixHandle<Variant> out;
        out.owner_      = owner;
        out.index_      = index;
        out.generation_ = generation;
        return out;
    }

    [[nodiscard]] static CaptureOffer<Variant> make_capture_offer(const void* owner,
                                                                  runtime::LaneId lane,
                                                                  std::uint64_t epoch,
                                                                  std::uint64_t id) noexcept {
        CaptureOffer<Variant> out;
        out.owner_ = owner;
        out.lane_  = lane;
        out.epoch_ = epoch;
        out.id_    = id;
        return out;
    }

    [[nodiscard]] static const void* owner(const SequenceHandle<Variant>& handle) noexcept {
        return handle.owner_;
    }

    [[nodiscard]] static const void* owner(const CaptureOffer<Variant>& offer) noexcept {
        return offer.owner_;
    }

    [[nodiscard]] static runtime::LaneId lane(const CaptureOffer<Variant>& offer) noexcept {
        return offer.lane_;
    }

    [[nodiscard]] static std::uint64_t epoch(const CaptureOffer<Variant>& offer) noexcept {
        return offer.epoch_;
    }

    [[nodiscard]] static std::uint64_t id(const CaptureOffer<Variant>& offer) noexcept {
        return offer.id_;
    }

    static void consume(CaptureOffer<Variant>& offer) noexcept {
        offer.owner_ = nullptr;
        offer.id_    = 0;
    }

    [[nodiscard]] static runtime::LaneId lane(const SequenceHandle<Variant>& handle) noexcept {
        return handle.lane_;
    }

    [[nodiscard]] static std::uint64_t epoch(const SequenceHandle<Variant>& handle) noexcept {
        return handle.epoch_;
    }

    [[nodiscard]] static const void* owner(const ContinuationHandle<Variant>& handle) noexcept {
        return handle.owner_;
    }

    [[nodiscard]] static std::uint32_t index(const ContinuationHandle<Variant>& handle) noexcept {
        return handle.index_;
    }

    [[nodiscard]] static std::uint64_t epoch(const ContinuationHandle<Variant>& handle) noexcept {
        return handle.generation_;
    }

    static void consume(ContinuationHandle<Variant>& handle) noexcept {
        handle.owner_      = nullptr;
        handle.generation_ = 0;
    }

    [[nodiscard]] static const void* owner(const SharedPrefixHandle<Variant>& handle) noexcept {
        return handle.owner_;
    }

    [[nodiscard]] static std::uint32_t index(const SharedPrefixHandle<Variant>& handle) noexcept {
        return handle.index_;
    }

    [[nodiscard]] static std::uint64_t epoch(const SharedPrefixHandle<Variant>& handle) noexcept {
        return handle.generation_;
    }

    static void consume(SharedPrefixHandle<Variant>& handle) noexcept {
        handle.owner_      = nullptr;
        handle.generation_ = 0;
    }

    [[nodiscard]] static PendingBatch<Variant>
    make_pending(const void* owner, std::uint64_t transaction,
                 std::span<const SequenceHandle<Variant>> rows, std::span<const TokenId> tokens,
                 std::span<const std::int32_t> row_counts, std::uint32_t row_stride,
                 runtime::ExecutionTiming timing) {
        PendingBatch<Variant> out;
        out.owner_       = owner;
        out.transaction_ = transaction;
        out.row_count_   = rows.size();
        for (std::size_t i = 0; i < rows.size(); ++i) { out.rows_[i] = rows[i]; }
        out.tokens_     = tokens;
        out.row_counts_ = row_counts;
        out.row_stride_ = row_stride;
        out.timing_     = timing;
        return out;
    }

    [[nodiscard]] static const void* owner(const PendingBatch<Variant>& pending) noexcept {
        return pending.owner_;
    }

    [[nodiscard]] static std::uint64_t transaction(const PendingBatch<Variant>& pending) noexcept {
        return pending.transaction_;
    }

    [[nodiscard]] static std::span<const SequenceHandle<Variant>>
    rows(const PendingBatch<Variant>& pending) noexcept {
        return {pending.rows_.data(), pending.row_count_};
    }

    static void consume(PendingBatch<Variant>& pending) noexcept {
        pending.owner_       = nullptr;
        pending.transaction_ = 0;
        pending.row_count_   = 0;
        pending.tokens_      = {};
        pending.row_counts_  = {};
        pending.row_stride_  = 0;
        pending.timing_      = {};
    }
};

} // namespace detail

template <class Variant>
[[nodiscard]] SequencePlanner<Variant>
make_sequence_planner(DeviceContext& device, const EngineOptions& options,
                      typename Variant::WeightsProfile weights_profile);

template <class Variant>
[[nodiscard]] std::unique_ptr<Program<Variant>>
create_program(const typename Variant::ModelView& model,
               typename Variant::WeightsProfile weights_profile, SequencePlan<Variant>&& plan,
               DeviceContext& device);

} // namespace ninfer::targets::qwen3_6
