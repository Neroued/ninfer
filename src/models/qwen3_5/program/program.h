#pragma once
#include "ninfer/types.h"
#include "ninfer/branch_score.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/resources.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace ninfer {
struct DeviceContext;
}

namespace ninfer::models::qwen3_5 {
namespace execution {
class Parameters;
}

namespace detail {
struct SequencePlanImpl;
struct SequencePlannerImpl;
struct RequestBasePlanImpl;
struct ResumeStateImpl;
struct DemotionPlan;
struct ReleasePlan;
struct ReclaimPlan;
struct DemotionBatch;
class ProgramImpl;
struct ContractAccess;
} // namespace detail
class SequencePlanner;
class Program;

struct GraphExecutionProfile {
    std::uint32_t min            = 0;
    std::uint32_t max            = 0;
    std::uint32_t topology_class = 0;
};
enum class TextPhase { Prefill, Verify };

struct PrefixShortlistKey {
    std::array<std::uint64_t, 2> digests{};
    std::uint32_t frontier                                                  = 0;
    std::uint32_t identity_tag                                              = 0;
    friend bool operator==(PrefixShortlistKey, PrefixShortlistKey) noexcept = default;
};

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
    [[nodiscard]] std::size_t host_capacity_bytes() const noexcept;

private:
    explicit SequencePlan(std::unique_ptr<detail::SequencePlanImpl> impl) noexcept;
    std::unique_ptr<detail::SequencePlanImpl> impl_;

    friend class SequencePlanner;

    friend std::unique_ptr<Program> create_program(const execution::Parameters&, SequencePlan&&,
                                                   DeviceContext&, const StartupObserver&);
};

class SequencePlanner {
public:
    SequencePlanner(SequencePlanner&&) noexcept;
    SequencePlanner& operator=(SequencePlanner&&) noexcept;
    ~SequencePlanner();

    SequencePlanner(const SequencePlanner&)            = delete;
    SequencePlanner& operator=(const SequencePlanner&) = delete;

    [[nodiscard]] const runtime::SequenceCapacityCurve& capacity_curve() const noexcept;
    [[nodiscard]] SequencePlan finalize(std::uint32_t main_page_groups) &&;

private:
    explicit SequencePlanner(std::unique_ptr<detail::SequencePlannerImpl> impl) noexcept;
    std::unique_ptr<detail::SequencePlannerImpl> impl_;

    friend SequencePlanner make_sequence_planner(const execution::Parameters&, DeviceContext&,
                                                 const EngineOptions&);
};

class RequestBasePlan {
public:
    RequestBasePlan(RequestBasePlan&&) noexcept;
    RequestBasePlan& operator=(RequestBasePlan&&) noexcept;
    ~RequestBasePlan();

    RequestBasePlan(const RequestBasePlan&)            = delete;
    RequestBasePlan& operator=(const RequestBasePlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;
    [[nodiscard]] const PreparedContextCache& context_cache() const noexcept;
    [[nodiscard]] std::vector<std::uint32_t> capture_frontiers() const;
    [[nodiscard]] std::optional<PrefixShortlistKey>
    prefix_shortlist_key(std::uint32_t frontier) const noexcept;

private:
    explicit RequestBasePlan(std::shared_ptr<detail::RequestBasePlanImpl> impl) noexcept;
    std::shared_ptr<detail::RequestBasePlanImpl> impl_;

    friend class detail::ProgramImpl;
};

class SequenceHandle {
public:
    SequenceHandle() noexcept                                       = default;
    SequenceHandle(const SequenceHandle&) noexcept                  = default;
    SequenceHandle& operator=(const SequenceHandle&) noexcept       = default;
    friend bool operator==(SequenceHandle, SequenceHandle) noexcept = default;

private:
    const void* owner_ = nullptr;
    runtime::LaneId lane_{};
    std::uint64_t epoch_ = 0;

    friend struct detail::ContractAccess;
};

// One complete immutable state/KV coverage record. Generation protects reused descriptors.
struct CheckpointHandle {
    const void* owner                                                   = nullptr;
    std::uint32_t index                                                 = 0;
    std::uint64_t generation                                            = 0;
    friend bool operator==(CheckpointHandle, CheckpointHandle) noexcept = default;
};

struct CheckpointMetadata {
    std::uint32_t frontier       = 0;
    runtime::CheckpointRole role = runtime::CheckpointRole::Continuation;
    bool leased                  = false;
};

struct CheckpointSummary {
    PrefixShortlistKey key;
    std::uint32_t frontier       = 0;
    runtime::CheckpointRole role = runtime::CheckpointRole::Continuation;
    bool leased                  = false;
    // Resources relevant to eviction; shared optional aliases are counted per record.
    runtime::ContextResourceUsage evictable_resources;
};

struct SourceCandidate {
    std::optional<CheckpointHandle> checkpoint;
    std::uint32_t reused_tokens = 0;
    runtime::PrefillWork remaining_work;
    std::vector<runtime::ContextTransferRequirement> transfers;
    bool consume_source = false;
    bool take_private   = false;
    bool move_state     = false;
    bool move_history   = false;
    bool split_state    = false;
    bool backup_state   = false;
    // Compatible carry intent. Binding may consume the selected point if its State slot
    // is necessary for execution and no preservation destination can be obtained.
    std::vector<CheckpointHandle> private_points;
    // Authorized retirement, applied only after binding capacity has been checked.
    std::vector<CheckpointHandle> retired_points;
};

struct BindingReservation {
    bool reserved          = false;
    bool source_valid      = true;
    bool capacity_possible = true;
    runtime::ContextResourceUsage shortage;
    std::vector<CheckpointHandle> retired_points;
    std::optional<CheckpointHandle> consumed_source;

    explicit operator bool() const noexcept { return reserved; }
};

struct ContextDemotion {
    std::vector<CheckpointHandle> sources;
    std::size_t host_bytes = 0;
    runtime::ContextResourceUsage released;
    std::shared_ptr<const detail::DemotionPlan> impl;
};

struct ContextRelease {
    std::vector<CheckpointHandle> sources;
    // Guaranteed release in the queried shortage pool; other quantities are not inventoried.
    runtime::ContextResourceUsage released;
    std::shared_ptr<const detail::ReleasePlan> impl;
};

class ContextDemotionBatch {
public:
    ContextDemotionBatch(ContextDemotionBatch&&) noexcept;
    ContextDemotionBatch& operator=(ContextDemotionBatch&&) noexcept;
    ~ContextDemotionBatch();
    ContextDemotionBatch(const ContextDemotionBatch&)            = delete;
    ContextDemotionBatch& operator=(const ContextDemotionBatch&) = delete;

    [[nodiscard]] bool append(const ContextDemotion&);
    [[nodiscard]] bool covers(const ContextRelease&) const;
    [[nodiscard]] std::optional<ContextDemotion> finish() const;

private:
    explicit ContextDemotionBatch(std::unique_ptr<detail::DemotionBatch>);
    std::unique_ptr<detail::DemotionBatch> impl_;
    friend struct ContextReclaimPlan;
};

// Read-only facts for one synchronous evaluation. Discard before any Native mutation or yield.
// start_demote independently revalidates the selected quote before reserving destinations.
struct ContextReclaimPlan {
    std::vector<ContextDemotion> demotions;
    std::vector<ContextRelease> releases;

    [[nodiscard]] ContextDemotionBatch begin_kv_batch(runtime::ContextResourceUsage shortage) const;
    [[nodiscard]] std::uint64_t recovery_loss(std::span<const CheckpointHandle> removed,
                                              std::span<const CheckpointHandle> surviving) const;

private:
    std::shared_ptr<detail::ReclaimPlan> impl_;
    friend class detail::ProgramImpl;
};

class ResumeState {
public:
    ResumeState(ResumeState&&) noexcept;
    ResumeState& operator=(ResumeState&&) noexcept;
    ~ResumeState();
    ResumeState(const ResumeState&)            = delete;
    ResumeState& operator=(const ResumeState&) = delete;
    [[nodiscard]] bool has_snapshot() const noexcept;
    [[nodiscard]] std::optional<CheckpointHandle> snapshot_handle() const noexcept;
    [[nodiscard]] std::uint32_t frontier() const noexcept;

private:
    explicit ResumeState(std::unique_ptr<detail::ResumeStateImpl>) noexcept;
    std::unique_ptr<detail::ResumeStateImpl> impl_;

    friend class detail::ProgramImpl;
};
enum class ExecutionUnitKind : std::uint8_t { Prefill, Replay, Decode, Control, Normalize };

struct ExecutionUnit {
    SequenceHandle sequence;
    ExecutionUnitKind kind = ExecutionUnitKind::Decode;
    // Decode: remaining output budget. Control: exact forced span. Other kinds: zero.
    std::uint32_t tokens = 0;
};

class PendingBatch {
public:
    PendingBatch() noexcept = default;
    ~PendingBatch()         = default;

    PendingBatch(PendingBatch&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)),
          transaction_(std::exchange(other.transaction_, 0)), rows_(other.rows_),
          row_count_(std::exchange(other.row_count_, 0)), tokens_(other.tokens_),
          row_counts_(other.row_counts_), row_stride_(other.row_stride_), timing_(other.timing_),
          constraint_failed_(other.constraint_failed_) {
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

    [[nodiscard]] bool constraint_failed(std::size_t row) const {
        return constraint_failed_.at(row);
    }

    [[nodiscard]] runtime::ExecutionTiming execution_timing() const noexcept { return timing_; }

private:
    const void* owner_         = nullptr;
    std::uint64_t transaction_ = 0;
    std::array<SequenceHandle, kMaximumConcurrency> rows_{};
    std::size_t row_count_ = 0;
    std::span<const TokenId> tokens_;
    std::span<const std::int32_t> row_counts_;
    std::uint32_t row_stride_ = 0;
    runtime::ExecutionTiming timing_;
    std::array<bool, kMaximumConcurrency> constraint_failed_{};

    friend struct detail::ContractAccess;
};

struct PrefillProgress {
    runtime::BeginSummary summary;
    std::uint32_t processed_prompt_tokens = 0;
    bool complete                         = false;
    runtime::ExecutionTiming timing;
    std::optional<PendingBatch> pending;
    bool capture_ready = false;
};

struct CommitRowResult {
    runtime::CommitDisposition disposition = runtime::CommitDisposition::Active;
    GenerationTimings timings;
    SpeculativeStats speculative;

    // Fixed-size cumulative observation, including active rows without copying per-position data.
    struct SpeculativeCounters {
        std::uint64_t rounds          = 0;
        std::uint64_t drafted_tokens  = 0;
        std::uint64_t accepted_tokens = 0;
        std::uint64_t fallback_steps  = 0;
    } speculative_counters;
};

struct CommitResult {
    std::array<CommitRowResult, kMaximumConcurrency> rows{};
    std::array<bool, kMaximumConcurrency> capture_ready{};
    std::size_t row_count = 0;
    runtime::ExecutionTiming timing;
};

struct DiscardResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    std::size_t row_count         = 0;
};

struct FinishResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    GenerationTimings timings;
    SpeculativeStats speculative;
    std::optional<CheckpointHandle> checkpoint;
};

struct AbortResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    GenerationTimings timings;
    SpeculativeStats speculative;
};

struct ReplayProgress {
    bool capture_ready             = false;
    std::uint32_t processed_tokens = 0;
    bool complete                  = false;
    runtime::ExecutionTiming timing;
};

struct CapturePreparation {
    std::uint32_t frontier = 0;
    bool reserved          = false;
    runtime::ContextResourceUsage shortage;
    std::size_t host_bytes = 0;
};
enum class ContextOperationKind : std::uint8_t { Bind, Capture, Demote, Pause };

struct ContextProgress {
    ContextOperationKind kind = ContextOperationKind::Bind;
    bool advanced             = false;
    bool complete             = false;
    bool published            = false;
    std::optional<SequenceHandle> sequence;
    std::optional<ResumeState> paused;
    bool replaying = false;
    std::vector<CheckpointHandle> private_points;
    std::vector<CheckpointHandle> retired_checkpoints;
    std::vector<CheckpointHandle> captured_checkpoints;
    std::vector<runtime::ContextTransferObservation> transfers;
    runtime::ContextOperationCounts operations;
    std::optional<GenerationTimings> request_timings;
    SpeculativeStats request_speculative;
};

struct PhysicalUsageSnapshot {
    runtime::ContextResourceUsage occupied;
    runtime::ContextResourceUsage capacity;
    std::size_t host_reserved_bytes      = 0;
    std::size_t host_peak_occupied_bytes = 0;
    std::uint32_t host_state_slots       = 0;
    std::size_t host_kv_bytes            = 0;
};

class Program {
public:
    ~Program() noexcept;
    Program(const Program&)            = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&&)                 = delete;
    Program& operator=(Program&&)      = delete;
    [[nodiscard]] RequestBasePlan plan_request(PreparedPrompt&& prompt,
                                               const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] std::vector<float> causal_score(PreparedPrompt&& prompt,
                                                  std::uint32_t first_target);
    [[nodiscard]] std::optional<SourceCandidate>
    inspect_source(const RequestBasePlan& base, std::optional<CheckpointHandle> checkpoint,
                   bool consume_source                              = false,
                   std::span<const CheckpointHandle> private_points = {},
                   std::span<const CheckpointHandle> retired_points = {}) const;
    [[nodiscard]] PrefixShortlistKey checkpoint_key(CheckpointHandle, std::uint32_t frontier) const;
    [[nodiscard]] runtime::ContextResourceUsage
        checkpoint_footprint(std::span<const CheckpointHandle>) const;
    [[nodiscard]] CheckpointSummary checkpoint_summary(CheckpointHandle checkpoint) const;
    [[nodiscard]] CheckpointMetadata checkpoint_metadata(CheckpointHandle checkpoint) const;
    [[nodiscard]] bool checkpoint_matches(CheckpointHandle checkpoint,
                                          const RequestBasePlan& base) const;
    [[nodiscard]] std::uint32_t checkpoint_recovery_frontier(CheckpointHandle retained,
                                                             const RequestBasePlan& base,
                                                             std::uint32_t target) const;
    [[nodiscard]] std::uint64_t
    checkpoint_recovery_loss(std::span<const CheckpointHandle> removed,
                             std::span<const CheckpointHandle> surviving) const;
    [[nodiscard]] bool valid_checkpoint(CheckpointHandle checkpoint) const noexcept;
    [[nodiscard]] bool release_checkpoint(CheckpointHandle checkpoint) noexcept;
    [[nodiscard]] bool revoke_snapshot(ResumeState& paused) noexcept;
    [[nodiscard]] runtime::ContextResourceUsage snapshot_resources(const ResumeState& paused) const;
    // Exact Host bytes freed by deleting this fixed set, with physical sharing counted once.
    [[nodiscard]] std::size_t
    host_bytes_released(std::span<const CheckpointHandle> checkpoints) const;
    [[nodiscard]] std::optional<std::size_t> pause_host_bytes(SequenceHandle sequence) const;
    [[nodiscard]] std::size_t
    release_redundant_host(std::span<const CheckpointHandle> excluded,
                           std::optional<SequenceHandle> pending_backup = std::nullopt);
    // Reservations stay owned by their lane through commit. Failure leaves prior permits intact.
    [[nodiscard]] runtime::ResourceReservation reserve_units(std::span<const ExecutionUnit> units);
    [[nodiscard]] bool reclaim_capture_reservation(runtime::ContextResourceUsage shortage);
    void release_units(std::span<const SequenceHandle> sequences) noexcept;
    [[nodiscard]] BindingReservation
    start_binding(const RequestBasePlan& base, runtime::LaneId lane, const SourceCandidate& source,
                  ResumeState* resume           = nullptr,
                  ExecutionUnitKind resume_kind = ExecutionUnitKind::Decode,
                  std::uint32_t resume_tokens   = 1);
    [[nodiscard]] bool start_capture(SequenceHandle sequence);
    [[nodiscard]] bool capture_is_input(SequenceHandle sequence) const;
    [[nodiscard]] std::optional<CapturePreparation> prepare_capture(SequenceHandle sequence);
    void skip_capture(SequenceHandle sequence);
    // Finite physical actions, each including every optional holder needed for its release.
    [[nodiscard]] ContextReclaimPlan plan_reclaim(std::span<const CheckpointHandle> allowed,
                                                  std::span<const CheckpointHandle> excluded,
                                                  runtime::ContextResourceUsage shortage) const;
    [[nodiscard]] std::vector<ContextRelease>
    plan_releases(std::span<const CheckpointHandle> allowed,
                  std::span<const CheckpointHandle> excluded,
                  runtime::ContextResourceUsage shortage) const;
    [[nodiscard]] bool start_demote(const ContextDemotion& plan);
    [[nodiscard]] bool start_pause(SequenceHandle sequence, bool save_snapshot,
                                   runtime::ExecutionTiming* timing = nullptr);
    [[nodiscard]] ContextProgress poll_context(runtime::CancellationFlagView cancellation);
    [[nodiscard]] bool has_context_transaction() const noexcept;
    [[nodiscard]] bool context_blocks(SequenceHandle sequence) const noexcept;
    // A resumed binding retains its complete recovery capacity until committed new progress.
    [[nodiscard]] bool recovery_pending(SequenceHandle sequence) const noexcept;
    [[nodiscard]] PrefillProgress advance_prefill(SequenceHandle sequence,
                                                  runtime::ExecutionTiming* failed_timing = nullptr,
                                                  runtime::TokenMaskProvider* masks = nullptr);
    [[nodiscard]] BranchScoreResult score_branches(std::span<const BranchScoreRow>,
                                                   const PreparationControl&, bool reuse_prefix);
    // Decision cache is optional shared-pool occupancy; release before chat admission/work.
    bool release_branch_cache();
    [[nodiscard]] ReplayProgress advance_replay(SequenceHandle sequence,
                                                runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] PendingBatch decode(std::span<const SequenceHandle> sequences,
                                      std::span<const runtime::RoundBudget> budgets,
                                      runtime::ExecutionTiming* failed_timing = nullptr,
                                      runtime::TokenMaskProvider* masks       = nullptr);
    // Forced control contributes to counts once. Replay does not call this operation.
    [[nodiscard]] runtime::ExecutionTiming
    append_forced_tokens(std::span<const SequenceHandle> sequences,
                         std::span<const TokenId> row_major_tokens, std::uint32_t row_stride,
                         std::span<const std::optional<std::uint32_t>> prefix_execution_splits,
                         runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] CommitResult
    commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions,
           runtime::CommitObservation observation  = runtime::CommitObservation::AllRows,
           runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] DiscardResult abort_pending(PendingBatch&& pending) noexcept;
    [[nodiscard]] FinishResult finish(SequenceHandle sequence) noexcept;
    [[nodiscard]] AbortResult abort(SequenceHandle sequence) noexcept;
    void fail_all_cleanup() noexcept;
    [[nodiscard]] PhysicalUsageSnapshot physical_usage() const noexcept;
    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    void reset_memory_peaks() noexcept;
private:
    explicit Program(std::unique_ptr<detail::ProgramImpl> impl) noexcept;
    std::unique_ptr<detail::ProgramImpl> impl_;
    friend std::unique_ptr<Program> create_program(const execution::Parameters&, SequencePlan&&,
                                                   DeviceContext&, const StartupObserver&);
};

[[nodiscard]] SequencePlanner make_sequence_planner(const execution::Parameters&, DeviceContext&,
                                                    const EngineOptions&);
[[nodiscard]] std::unique_ptr<Program> create_program(const execution::Parameters&, SequencePlan&&,
                                                      DeviceContext&, const StartupObserver&);
} // namespace ninfer::models::qwen3_5
