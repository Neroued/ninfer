#pragma once

#include "ninfer/types.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/resources.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "runtime/prefix_cache/cost.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
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

namespace ninfer::models::qwen3_5 {

namespace execution {
class Parameters;
}

// Read-only diagnostics sampled from the real Program stores.  This is not an accounting input.
struct PhysicalUsageSnapshot {
    runtime::ProgramResourceRevision resource_revision;
    std::uint32_t device_state_slots      = 0;
    std::uint32_t device_main_kv_pages    = 0;
    std::uint32_t device_backend_kv_pages = 0;

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

namespace detail {

struct SequencePlanImpl;

struct SequencePlannerImpl;

struct RequestBasePlanImpl;

struct HybridQuoteImpl;

class ProgramImpl;

struct RuntimeContractAccess;
} // namespace detail

class SequencePlanner;

class Program;

// Concrete Qwen execution and resource contracts; model instances supply their own data.
// target selection remains outside this layer and happens once in the closed Engine registry.

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
    explicit SequencePlan(std::unique_ptr<detail::SequencePlanImpl> impl) noexcept;
    std::unique_ptr<detail::SequencePlanImpl> impl_;

    friend class SequencePlanner;

    friend class detail::ProgramImpl;
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

public:
    explicit SequencePlanner(std::unique_ptr<detail::SequencePlannerImpl> impl) noexcept;
    std::unique_ptr<detail::SequencePlannerImpl> impl_;

    friend SequencePlanner make_sequence_planner(const execution::Parameters&, DeviceContext&,
                                                 const EngineOptions&);
};

// Hybrid prefix cache admission quote (docs/maintainer/hybrid-prefix-cache.md §6). The
// summary already carries the selected reuse frontier; it is exact at admission.
struct HybridAdmissionQuote {
    runtime::Readiness readiness = runtime::Readiness::TemporarilyBlocked;
    runtime::LaneId destination{};
    runtime::RequestPlanSummary summary;
    std::shared_ptr<detail::HybridQuoteImpl> impl;
};

// Outcome of saving or restoring the hybrid prefix cache's Host tier.
struct HybridCachePersistence {
    bool ok = false;
    std::string message;
    std::uint64_t blocks    = 0;
    std::uint64_t snapshots = 0;
    std::uint64_t bytes     = 0;
    double seconds          = 0.0;
    // Load only: what the file holds and needs, against this Host tier.
    std::uint64_t saved_blocks        = 0;
    std::uint64_t saved_snapshots     = 0;
    std::uint64_t required_host_bytes = 0;
    std::uint64_t host_bytes          = 0;
};

struct HybridPrefixCacheStats {
    std::uint32_t nodes                      = 0;
    std::uint32_t snapshots                  = 0;
    std::uint32_t device_resident_blocks     = 0;
    std::uint32_t device_evictable_blocks    = 0;
    std::uint32_t host_slabs                 = 0;
    std::uint32_t host_free_slabs            = 0;
    std::uint64_t host_slab_bytes            = 0;
    std::uint32_t free_device_snapshot_slots = 0;
    std::uint64_t admissions                 = 0;
    std::uint64_t snapshot_hits              = 0;
    std::uint64_t reused_tokens              = 0;
    std::uint64_t blocks_inserted            = 0;
    std::uint64_t blocks_reattached          = 0;
    std::uint64_t blocks_duplicate           = 0;
    std::uint64_t taps_created               = 0;
    std::uint64_t taps_skipped               = 0;
    std::uint64_t endpoints_created          = 0;
    std::uint64_t host_image_writes          = 0;
    std::uint64_t host_block_writes          = 0;
    std::uint64_t host_image_restores        = 0;
    std::uint64_t host_block_restores        = 0;
    std::uint64_t host_tail_restores         = 0;
    std::uint64_t host_write_bytes           = 0;
    std::uint64_t host_restore_bytes         = 0;
    std::uint64_t evicted_blocks             = 0;
    std::uint64_t host_snapshot_evictions    = 0;
    std::uint64_t host_dead_reclaims         = 0;
    std::uint64_t unbacked_node_losses       = 0;
};

class RequestBasePlan {
public:
    RequestBasePlan(RequestBasePlan&&) noexcept;
    RequestBasePlan& operator=(RequestBasePlan&&) noexcept;
    ~RequestBasePlan();

    RequestBasePlan(const RequestBasePlan&)            = delete;
    RequestBasePlan& operator=(const RequestBasePlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;

public:
    explicit RequestBasePlan(std::unique_ptr<detail::RequestBasePlanImpl> impl) noexcept;
    std::unique_ptr<detail::RequestBasePlanImpl> impl_;
};

// A Program-minted proof that one FIFO borrower cannot consume the maximum physical entitlement
// reserved for the blocked head.  Common scheduling binds the opaque proof to logical identities
// and a revision; it cannot inspect or reproduce the resource arithmetic.

class PersistentBackfillProof {
public:
    PersistentBackfillProof(PersistentBackfillProof&&) noexcept            = default;
    PersistentBackfillProof& operator=(PersistentBackfillProof&&) noexcept = default;

    PersistentBackfillProof(const PersistentBackfillProof&)            = delete;
    PersistentBackfillProof& operator=(const PersistentBackfillProof&) = delete;

    [[nodiscard]] runtime::ProgramResourceRevision resource_revision() const noexcept {
        return revision_;
    }

private:
    explicit PersistentBackfillProof(runtime::ProgramResourceRevision revision) noexcept
        : revision_(revision) {}

    runtime::ProgramResourceRevision revision_;

    friend class Program;
};

class SequenceHandle {
public:
    SequenceHandle() noexcept                                 = default;
    SequenceHandle(const SequenceHandle&) noexcept            = default;
    SequenceHandle& operator=(const SequenceHandle&) noexcept = default;

private:
    const void* owner_ = nullptr;
    runtime::LaneId lane_{};
    std::uint64_t epoch_ = 0;

    friend struct detail::RuntimeContractAccess;
};

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
    std::array<SequenceHandle, kMaximumConcurrency> rows_{};
    std::size_t row_count_ = 0;
    std::span<const TokenId> tokens_;
    std::span<const std::int32_t> row_counts_;
    std::uint32_t row_stride_ = 0;
    runtime::ExecutionTiming timing_;

    friend struct detail::RuntimeContractAccess;
};

struct PrefillProgress {
    runtime::BeginSummary summary;
    std::uint32_t processed_prompt_tokens = 0;
    bool complete                         = false;
    runtime::ExecutionTiming timing;
    std::optional<PendingBatch> pending;
};

struct StartResult {
    SequenceHandle sequence;
};

struct MaterializationResult {
    runtime::ContextTransactionStatus status = runtime::ContextTransactionStatus::Aborted;
    std::optional<StartResult> published;
    MaterializationDiagnostics diagnostics;
};

using ContextTransactionProgress =
    std::variant<runtime::ContextTransactionInProgress, MaterializationResult>;

struct CommitRowResult {
    runtime::CommitDisposition disposition = runtime::CommitDisposition::Active;
    GenerationTimings timings;
    SpeculativeStats speculative;
};

struct CommitResult {
    std::array<CommitRowResult, kMaximumConcurrency> rows{};
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
};

struct AbortResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    GenerationTimings timings;
    SpeculativeStats speculative;
};


class Program {
public:
    ~Program() noexcept;

    Program(const Program&)            = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&&)                 = delete;
    Program& operator=(Program&&)      = delete;

    // Engine owns scheduling and logical residency policy. Program owns physical lanes, opaque
    // capabilities, model state and one immutable pending transaction at a time.
    [[nodiscard]] RequestBasePlan plan_request(const PreparedPrompt& prompt,
                                               const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] std::vector<float> causal_score(PreparedPrompt&& prompt,
                                                  std::uint32_t first_target);
    [[nodiscard]] ContextTransactionProgress
    progress_context_transaction(runtime::CancellationFlagView cancellation);
    void finalize_context_transaction() noexcept;
    [[nodiscard]] bool has_context_transaction() const noexcept;
    [[nodiscard]] PrefillProgress
    advance_prefill(SequenceHandle sequence, runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] PendingBatch decode(std::span<const SequenceHandle> sequences,
                                      std::span<const runtime::RoundBudget> budgets,
                                      runtime::ExecutionTiming* failed_timing = nullptr);
    // Advance each live sequence with its exact target-owned token row. This does not sample or
    // advance sampler RNG/occurrence state; callers own output publication and budget accounting.
    [[nodiscard]] runtime::ExecutionTiming
    append_forced_tokens(std::span<const SequenceHandle> sequences,
                         std::span<const TokenId> row_major_tokens, std::uint32_t row_stride,
                         runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] CommitResult
    commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions,
           runtime::CommitObservation observation  = runtime::CommitObservation::AllRows,
           runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] DiscardResult abort_pending(PendingBatch&& pending) noexcept;
    [[nodiscard]] FinishResult finish(SequenceHandle sequence) noexcept;
    [[nodiscard]] AbortResult abort(SequenceHandle sequence) noexcept;
    void fail_all_cleanup() noexcept;
    // fail_all_cleanup for the Engine's orderly stop. With a hybrid cache file attached, the Host
    // tier is saved once every lane has written its blocks through and before the cleanup drops
    // it; hybrid_shutdown_save() reports the result.
    void shutdown_cleanup() noexcept;

    [[nodiscard]] bool isolated_request_feasible(const RequestBasePlan& base) const noexcept;

    // Prefix-cache admission (docs/maintainer/hybrid-prefix-cache.md §6) runs as the
    // Engine's context transaction: quote, reserve, then progress_context_transaction publishes
    // the started sequence.
    [[nodiscard]] HybridAdmissionQuote hybrid_quote(const PreparedPrompt& prompt,
                                                    const RequestBasePlan& base,
                                                    runtime::LaneId destination);
    [[nodiscard]] runtime::ContextTransactionReserveStatus
    hybrid_reserve_materialization(HybridAdmissionQuote&& quote, PreparedPrompt&& prompt,
                                   runtime::CancellationFlagView cancellation);
    [[nodiscard]] std::optional<PersistentBackfillProof>
    prove_persistent_backfill(const RequestBasePlan& blocked_head,
                              const HybridAdmissionQuote& candidate,
                              std::span<const SequenceHandle> persistent_borrowers) const;
    // Copies Host-only blocks a waiting request resumes from into Device pages the pools can
    // spare as cache (hybrid-prefix-cache §6.6), so its admission restores less. Returns the
    // blocks whose copy started; absent while a prefetch or an admission is still in flight.
    [[nodiscard]] std::optional<std::uint32_t> hybrid_prefetch(const PreparedPrompt& prompt,
                                                               const RequestBasePlan& base);
    // Device pages a prefetch could fill now: free ones and host-backed cached ones.
    [[nodiscard]] std::uint32_t hybrid_prefetch_room() const noexcept;
    [[nodiscard]] HybridPrefixCacheStats hybrid_stats() const noexcept;
    // Installs the Engine's calibrated machine model for hybrid admission choice and eviction.
    void set_hybrid_cost(const runtime::prefix_cache::CacheCostModel& cost);
    // Restores a saved Host tier before the first request and attaches the file, so
    // shutdown_cleanup saves the tier back to it. `fingerprint` names everything the saved bytes
    // depend on; `observer` receives the file read as StartupPhase::PrefixCacheLoad.
    [[nodiscard]] HybridCachePersistence attach_hybrid_cache_file(const std::filesystem::path& path,
                                                                  std::string fingerprint,
                                                                  const StartupObserver& observer);
    [[nodiscard]] std::optional<HybridCachePersistence> hybrid_shutdown_save() const;

    [[nodiscard]] runtime::ProgramResourceRevision resource_revision() const noexcept;
    [[nodiscard]] PhysicalUsageSnapshot physical_usage() const noexcept;
    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    void reset_memory_peaks() noexcept;

private:
    explicit Program(std::unique_ptr<detail::ProgramImpl> impl) noexcept;
    std::unique_ptr<detail::ProgramImpl> impl_;

    friend std::unique_ptr<Program> create_program(const execution::Parameters&, SequencePlan&&,
                                                   DeviceContext&, const StartupObserver&);
};

namespace detail {

struct RuntimeContractAccess {
    [[nodiscard]] static SequenceHandle make_sequence(const void* owner, runtime::LaneId lane,
                                                      std::uint64_t epoch) noexcept {
        SequenceHandle out;
        out.owner_ = owner;
        out.lane_  = lane;
        out.epoch_ = epoch;
        return out;
    }

    [[nodiscard]] static const void* owner(const SequenceHandle& handle) noexcept {
        return handle.owner_;
    }

    [[nodiscard]] static runtime::LaneId lane(const SequenceHandle& handle) noexcept {
        return handle.lane_;
    }

    [[nodiscard]] static std::uint64_t epoch(const SequenceHandle& handle) noexcept {
        return handle.epoch_;
    }

    [[nodiscard]] static PendingBatch
    make_pending(const void* owner, std::uint64_t transaction, std::span<const SequenceHandle> rows,
                 std::span<const TokenId> tokens, std::span<const std::int32_t> row_counts,
                 std::uint32_t row_stride, runtime::ExecutionTiming timing) {
        PendingBatch out;
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

    [[nodiscard]] static const void* owner(const PendingBatch& pending) noexcept {
        return pending.owner_;
    }

    [[nodiscard]] static std::uint64_t transaction(const PendingBatch& pending) noexcept {
        return pending.transaction_;
    }

    [[nodiscard]] static std::span<const SequenceHandle>
    rows(const PendingBatch& pending) noexcept {
        return {pending.rows_.data(), pending.row_count_};
    }

    static void consume(PendingBatch& pending) noexcept {
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

[[nodiscard]] SequencePlanner make_sequence_planner(const execution::Parameters& parameters,
                                                    DeviceContext& device,
                                                    const EngineOptions& options);

[[nodiscard]] std::unique_ptr<Program> create_program(const execution::Parameters& parameters,
                                                      SequencePlan&& plan, DeviceContext& device,
                                                      const StartupObserver& startup_observer);

} // namespace ninfer::models::qwen3_5
