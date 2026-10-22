#pragma once

#include "models/common/context/checkpoint_store.h"
#include "models/common/qwen/frontend/prepared_prompt.h"
#include "models/qwen4_exp/program/planning.h"
#include "runtime/contract/execution.h"

namespace ninfer::models::qwen4_exp {
class Program;

namespace execution {
class Parameters;
}

namespace detail {
class ProgramImpl;
struct RequestPlan;
struct Suspended;
struct Demotion;
struct Reclaim;
} // namespace detail

using ContextResources    = runtime::ContextResources<2>;
using ResourceReservation = runtime::ResourceReservation<ContextResources>;
using CheckpointHandle    = context::CheckpointHandle;
inline constexpr runtime::KvPoolId kMainPool{0}, kMtpPool{1};

struct PrefixShortlistKey {
    std::array<std::uint64_t, 2> digests{};
    std::uint32_t frontier = 0, identity_tag = 0;
    friend bool operator==(PrefixShortlistKey, PrefixShortlistKey) noexcept = default;
};

class RequestBasePlan {
public:
    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;
    [[nodiscard]] const qwen::PreparedContextCache& context_cache() const noexcept;
    [[nodiscard]] std::vector<std::uint32_t> capture_frontiers() const;
    [[nodiscard]] std::optional<PrefixShortlistKey> prefix_shortlist_key(std::uint32_t) const;
private:
    std::shared_ptr<const detail::RequestPlan> plan_;

    explicit RequestBasePlan(std::shared_ptr<const detail::RequestPlan> p) : plan_(std::move(p)) {}
    friend class detail::ProgramImpl;
    friend class Program;
};

struct SequenceHandle {
    const void* owner                                               = nullptr;
    std::uint32_t lane                                              = 0;
    std::uint64_t epoch                                             = 0;
    friend bool operator==(SequenceHandle, SequenceHandle) noexcept = default;
};

struct CheckpointMetadata {
    std::uint32_t frontier       = 0;
    runtime::CheckpointRole role = runtime::CheckpointRole::Continuation;
    bool leased                  = false;
};

struct CheckpointSummary : CheckpointMetadata {
    PrefixShortlistKey key;
    ContextResources evictable_resources;
};

struct SourceCandidate {
    std::optional<CheckpointHandle> checkpoint;
    std::uint32_t reused_tokens = 0;
    runtime::PrefillWork remaining_work;
    std::vector<runtime::ContextTransferRequirement> transfers;
    bool consume_source = false, take_private = false;
    std::vector<CheckpointHandle> private_points, retired_points;
};

struct BindingReservation {
    bool reserved = false, source_valid = true, capacity_possible = true;
    ContextResources shortage;
    std::vector<CheckpointHandle> retired_points;
    std::optional<CheckpointHandle> consumed_source;

    explicit operator bool() const noexcept { return reserved; }
};

struct ContextDemotion {
    std::vector<CheckpointHandle> sources;
    std::size_t host_bytes = 0;
    ContextResources released;
    std::shared_ptr<const detail::Demotion> impl;
};

struct ContextRelease {
    std::vector<CheckpointHandle> sources;
    ContextResources released;
};

class ContextDemotionBatch {
public:
    [[nodiscard]] bool append(const ContextDemotion&);
    [[nodiscard]] bool covers(const ContextRelease&) const;
    [[nodiscard]] std::optional<ContextDemotion> finish() const;
private:
    std::shared_ptr<detail::Demotion> plan_;

    explicit ContextDemotionBatch(std::shared_ptr<detail::Demotion> p) : plan_(std::move(p)) {}
    friend struct ContextReclaimPlan;
};

struct ContextReclaimPlan {
    std::vector<ContextDemotion> demotions;
    std::vector<ContextRelease> releases;
    [[nodiscard]] ContextDemotionBatch begin_kv_batch(ContextResources) const;
    [[nodiscard]] std::uint64_t recovery_loss(std::span<const CheckpointHandle>,
                                              std::span<const CheckpointHandle>) const;
    std::shared_ptr<const detail::Reclaim> impl;
};

class ResumeState {
public:
    ResumeState(ResumeState&&) noexcept;
    ResumeState& operator=(ResumeState&&) noexcept;
    ~ResumeState();
    [[nodiscard]] bool has_snapshot() const noexcept;
    [[nodiscard]] std::optional<CheckpointHandle> snapshot_handle() const noexcept;
    [[nodiscard]] std::uint32_t frontier() const noexcept;
private:
    explicit ResumeState(std::unique_ptr<detail::Suspended>);
    std::unique_ptr<detail::Suspended> state_;
    friend class detail::ProgramImpl;
    friend class Program;
};

enum class ExecutionUnitKind : std::uint8_t { Prefill, Replay, Decode, Control, Normalize };

struct ExecutionUnit {
    SequenceHandle sequence;
    ExecutionUnitKind kind = ExecutionUnitKind::Decode;
    std::uint32_t tokens   = 0;
};

class PendingBatch {
public:
    PendingBatch(PendingBatch&&) noexcept = default;
    PendingBatch(const PendingBatch&)     = delete;

    [[nodiscard]] std::size_t row_count() const noexcept { return members_.size(); }

    [[nodiscard]] std::span<const TokenId> tokens() const noexcept { return tokens_; }

    [[nodiscard]] std::span<const std::int32_t> row_counts() const noexcept { return counts_; }

    [[nodiscard]] std::uint32_t row_stride() const noexcept { return stride_; }

    [[nodiscard]] bool constraint_failed(std::size_t row) const { return failed_.at(row); }

    [[nodiscard]] runtime::ExecutionTiming execution_timing() const noexcept { return timing_; }
private:
    PendingBatch()             = default;
    const void* owner_         = nullptr;
    std::uint64_t transaction_ = 0;
    std::vector<SequenceHandle> members_;
    std::vector<TokenId> tokens_;
    std::vector<std::int32_t> counts_;
    std::uint32_t stride_ = 1;
    bool decode_          = false;
    std::array<bool, kMaximumConcurrency> failed_{};
    runtime::ExecutionTiming timing_;
    friend class detail::ProgramImpl;
    friend class Program;
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

    struct Counters {
        std::uint64_t rounds = 0, drafted_tokens = 0, accepted_tokens = 0, fallback_steps = 0;
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
    ContextResources shortage;
    std::size_t host_bytes = 0;
};
enum class ContextOperationKind : std::uint8_t { Bind, Capture, Demote, Pause };

struct ContextProgress {
    ContextOperationKind kind = ContextOperationKind::Bind;
    bool complete = false, published = false;
    std::optional<SequenceHandle> sequence;
    std::optional<ResumeState> paused;
    bool replaying = false;
    std::vector<CheckpointHandle> private_points, retired_checkpoints, captured_checkpoints;
    std::vector<runtime::ContextTransferObservation> transfers;
    runtime::ContextOperationCounts operations;
    std::optional<GenerationTimings> request_timings;
    SpeculativeStats request_speculative;
};

struct PhysicalUsageSnapshot {
    ContextResources occupied, capacity;
    std::size_t host_reserved_bytes = 0, host_peak_occupied_bytes = 0;
    std::uint32_t host_state_slots = 0;
    std::size_t host_kv_bytes      = 0;
};

class Program {
public:
    using ContextResources = qwen4_exp::ContextResources;
    Program(const execution::Parameters&, ExecutionPlan, DeviceContext&);
    ~Program();
    Program(const Program&)            = delete;
    Program& operator=(const Program&) = delete;
    [[nodiscard]] std::span<const runtime::KvPoolDescriptor> kv_pools() const noexcept;
    [[nodiscard]] RequestBasePlan plan_request(qwen::PreparedPrompt&&,
                                               const runtime::ResolvedExecutionOptions&);
    [[nodiscard]] std::vector<float> causal_score(qwen::PreparedPrompt&&,
                                                  std::uint32_t first_target);
    [[nodiscard]] std::optional<SourceCandidate>
    inspect_source(const RequestBasePlan&, std::optional<CheckpointHandle>, bool consume = false,
                   std::span<const CheckpointHandle> carried = {},
                   std::span<const CheckpointHandle> retired = {}) const;
    [[nodiscard]] PrefixShortlistKey checkpoint_key(CheckpointHandle, std::uint32_t) const;
    [[nodiscard]] ContextResources checkpoint_footprint(std::span<const CheckpointHandle>) const;
    [[nodiscard]] CheckpointSummary checkpoint_summary(CheckpointHandle) const;
    [[nodiscard]] CheckpointMetadata checkpoint_metadata(CheckpointHandle) const;
    [[nodiscard]] bool checkpoint_matches(CheckpointHandle, const RequestBasePlan&) const;
    [[nodiscard]] std::uint32_t
    checkpoint_recovery_frontier(CheckpointHandle, const RequestBasePlan&, std::uint32_t) const;
    [[nodiscard]] std::uint64_t checkpoint_recovery_loss(std::span<const CheckpointHandle>,
                                                         std::span<const CheckpointHandle>) const;
    [[nodiscard]] bool valid_checkpoint(CheckpointHandle) const noexcept;
    [[nodiscard]] bool release_checkpoint(CheckpointHandle) noexcept;
    [[nodiscard]] bool revoke_snapshot(ResumeState&) noexcept;
    [[nodiscard]] ContextResources snapshot_resources(const ResumeState&) const;
    [[nodiscard]] std::size_t host_bytes_released(std::span<const CheckpointHandle>) const;
    [[nodiscard]] std::optional<std::size_t> pause_host_bytes(SequenceHandle) const;
    [[nodiscard]] std::size_t release_redundant_host(std::span<const CheckpointHandle>,
                                                     std::optional<SequenceHandle> = {});
    [[nodiscard]] ResourceReservation reserve_units(std::span<const ExecutionUnit>);
    [[nodiscard]] bool reclaim_capture_reservation(ContextResources);
    void release_units(std::span<const SequenceHandle>) noexcept;
    [[nodiscard]] BindingReservation start_binding(const RequestBasePlan&, runtime::LaneId,
                                                   const SourceCandidate&, ResumeState* = nullptr,
                                                   ExecutionUnitKind    = ExecutionUnitKind::Decode,
                                                   std::uint32_t tokens = 1);
    [[nodiscard]] bool start_capture(SequenceHandle);
    [[nodiscard]] bool capture_is_input(SequenceHandle) const;
    [[nodiscard]] std::optional<CapturePreparation> prepare_capture(SequenceHandle);
    void skip_capture(SequenceHandle);
    [[nodiscard]] ContextReclaimPlan plan_reclaim(std::span<const CheckpointHandle>,
                                                  std::span<const CheckpointHandle>,
                                                  ContextResources) const;
    [[nodiscard]] std::vector<ContextRelease> plan_releases(std::span<const CheckpointHandle>,
                                                            std::span<const CheckpointHandle>,
                                                            ContextResources) const;
    [[nodiscard]] bool start_demote(const ContextDemotion&);
    [[nodiscard]] bool start_pause(SequenceHandle, bool save, runtime::ExecutionTiming* = nullptr);
    [[nodiscard]] ContextProgress poll_context(runtime::CancellationFlagView);
    [[nodiscard]] bool has_context_operation() const noexcept;
    [[nodiscard]] bool context_blocks(SequenceHandle) const noexcept;
    [[nodiscard]] bool recovery_pending(SequenceHandle) const noexcept;
    [[nodiscard]] PrefillProgress advance_prefill(SequenceHandle,
                                                  runtime::ExecutionTiming*   = nullptr,
                                                  runtime::TokenMaskProvider* = nullptr);
    [[nodiscard]] ReplayProgress advance_replay(SequenceHandle,
                                                runtime::ExecutionTiming* = nullptr);
    [[nodiscard]] PendingBatch decode(std::span<const SequenceHandle>,
                                      std::span<const runtime::RoundBudget>,
                                      runtime::ExecutionTiming*   = nullptr,
                                      runtime::TokenMaskProvider* = nullptr);
    [[nodiscard]] runtime::ExecutionTiming
    append_forced_tokens(std::span<const SequenceHandle>, std::span<const TokenId>,
                         std::uint32_t stride, std::span<const std::optional<std::uint32_t>>,
                         runtime::ExecutionTiming* = nullptr);
    [[nodiscard]] CommitResult
    commit(PendingBatch&&, std::span<const runtime::CommitDecision>,
           runtime::CommitObservation = runtime::CommitObservation::AllRows,
           runtime::ExecutionTiming*  = nullptr);
    [[nodiscard]] DiscardResult abort_pending(PendingBatch&&) noexcept;
    [[nodiscard]] FinishResult finish(SequenceHandle) noexcept;
    [[nodiscard]] AbortResult abort(SequenceHandle) noexcept;
    void fail_all_cleanup() noexcept;
    [[nodiscard]] PhysicalUsageSnapshot physical_usage() const noexcept;
    // One StateImage stride in bytes; a checkpoint's state_slots x this is its state footprint.
    [[nodiscard]] std::size_t state_image_bytes() const noexcept;
    [[nodiscard]] MemorySummary memory_summary() const;
    void reset_memory_peaks() noexcept;
private:
    std::unique_ptr<detail::ProgramImpl> impl_;
};

} // namespace ninfer::models::qwen4_exp
