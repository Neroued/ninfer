#pragma once

#include "runtime/observability/gpu_timing.h"

#include "ninfer/ops/kv_cache_append.h"

#include "models/common/context/transfer.h"
#include "models/common/graph/suite.h"
#include "models/common/qwen/vision/vision.h"
#include "models/qwen3_5/program/program.h"
#include "models/qwen3_5/execution/attention.h"
#include "models/qwen3_5/execution/draft.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/resources.h"
#include "models/qwen3_5/program/planning/startup.h"

#include <array>
#include <chrono>
#include <deque>
#include <memory>
#include <optional>

namespace ninfer::models::qwen3_5::execution {
class TextRun;
}

namespace ninfer::models::qwen3_5::detail {

struct Residency {
    runtime::ResidencyId id;
    runtime::SequenceId sequence;
    bool installed = false;
    bool penalties = false;
};

enum class ComputeStage : std::uint8_t {
    Inputs,
    Proposal,
    Forward,
    History,
    AwaitDecision,
    Readback,
    AwaitCommit,
    Commit,
    Retiring,
};

struct PackedPrefill {
    ops::RaggedSequenceView ragged, mtp_ragged, dflash_local_ragged;
    std::int32_t sampled_count = 0;
};

void validate_prefill_work(const ForwardWork&, std::uint32_t prefill_chunk);
void validate_prefill_commit(const ForwardWork&, std::span<const CommitRow>);

struct ComputeOperation {
    runtime::ExecutionRequest<ExecutionTypes> request;
    runtime::CommandSequence last_command;
    std::vector<ExecutionRow> rows;
    std::vector<SamplingInput> sampling;
    std::uint32_t width          = 0;
    std::uint32_t batch          = 0;
    std::uint32_t forward_row    = 0;
    std::uint32_t forward_offset = 0;
    std::uint32_t score_cursor   = 0;
    std::uint32_t score_count    = 0;
    ComputeStage stage           = ComputeStage::Inputs;
    std::uint32_t decision_stage = 0;
    bool read_drafts_for_masks   = false;
    bool decision_requested      = false;
    bool decision_supplied       = false;
    bool result_published        = false;
    bool cancelled               = false;
    bool row_finished            = false;
    bool graph_started           = false;
    bool projected               = false;
    std::vector<std::uint32_t> dead_positions;
    PhysicalBatchShape shape;
    std::optional<runtime::DecisionInput> decision;
    std::optional<Commit> commit;
    std::optional<Result> result;
    std::vector<float> scores;
    GpuExecutionObservation gpu;
    AttentionWorkCounters attention;
    std::optional<Tensor> input_embeddings;
    std::optional<PackedPrefill> packed_prefill;
    std::unique_ptr<execution::TextRun> text;
    TargetGraphBody* graph_target = nullptr;
    std::optional<speculative::NarrowedTargetPlan> narrowed;

    [[nodiscard]] bool verifies_drafts() const noexcept {
        return width > 1 && std::holds_alternative<DecodeWork>(request.work.operation);
    }

    [[nodiscard]] const speculative::RoundRoute& decode_route() const {
        return narrowed ? narrowed->target_route : proposal_route();
    }

    [[nodiscard]] const speculative::RoundRoute& proposal_route() const {
        return std::get<DecodeWork>(request.work.operation).plan->route();
    }

    ComputeOperation(runtime::ExecutionRequest<ExecutionTypes>&&);
    ~ComputeOperation();
};

struct ContextOperation {
    runtime::ExecutionRequest<ExecutionTypes> request;
    runtime::CommandSequence last_command;
};

struct ProposalBuckets {
    std::array<ops::ContextBucket, kMtpDecodeMaximumDrafts> values{};
    std::size_t count = 0;

    [[nodiscard]] std::span<const ops::ContextBucket> span() const {
        return std::span(values).first(count);
    }
};

class RankProgramImpl {
public:
    RankProgramImpl(const execution::Parameters&, const SequencePlanImpl&, DeviceContext&,
                    const StartupObserver&);
    ~RankProgramImpl();

    void validate_work(const Work&, runtime::FrameKind) const;
    void validate_decision(runtime::ExecutionTicket, const runtime::DecisionInput&) const;
    void prepare_commit(runtime::ExecutionTicket, Commit&) const;
    void start(runtime::ExecutionRequest<ExecutionTypes>&&);
    void supply(runtime::ExecutionTicket, runtime::CommandSequence, runtime::DecisionInput&&);
    void commit(runtime::ExecutionTicket, runtime::CommandSequence, Commit&&);
    void discard(runtime::ExecutionTicket, runtime::CommandSequence);
    [[nodiscard]] std::optional<runtime::ExecutionEvent<ExecutionTypes>> poll();
    void drain();
    [[nodiscard]] RankMemoryUsage observe_resources() const;
    [[nodiscard]] RankExecutionObservation retirement_observation() const noexcept;
    void reset_measurement_peaks() noexcept;

    void validate_row(const ExecutionRow&) const;
    std::uint64_t next_validation_mark() const;
    void install_residencies(const ContextWork&);
    void initialize_inputs(ComputeOperation&);
    void initialize_decode(ComputeOperation&, const DecodeWork&);
    void initialize_forward(ComputeOperation&, const ForwardWork&);
    void initialize_packed_prefill(ComputeOperation&, const ForwardWork&);
    void initialize_score(ComputeOperation&, const ScoreWork&);
    void upload_input(std::uint32_t width, std::uint32_t batch);
    void upload_positions(std::uint32_t width, std::uint32_t batch);
    void upload_ingress();
    void begin_target(ComputeOperation&);
    void begin_packed_target(ComputeOperation&);
    void finish_packed_prefill(ComputeOperation&);
    bool advance_target(ComputeOperation&);
    void begin_draft(ComputeOperation&);
    void narrow_target(ComputeOperation&);
    void prepare_target_ingress(const ComputeOperation&, std::span<const std::uint32_t>);
    Tensor draft_tokens(const speculative::RoundRoute&) const;
    void readback_drafts(const speculative::RoundRoute&);
    void prepare_verification(const speculative::RoundRoute& target,
                              const speculative::RoundRoute& proposal);
    void prepare_mtp_draft(ComputeOperation&);
    [[nodiscard]] ProposalBuckets proposal_buckets(const speculative::RoundRoute&) const;
    [[nodiscard]] ops::ContextBucket target_bucket() const;
    void gather_mtp_conditioning(std::uint32_t batch);
    void bind_mtp_draft_step(std::uint32_t batch, std::uint32_t step,
                             const execution::CausalAttentionSubmit&);
    void run_mtp_draft_step(std::uint32_t batch, std::uint32_t step, ops::ContextBucket,
                            const execution::CausalAttentionSubmit& = {});
    void bind_dflash_draft(const speculative::RoundRoute&, const execution::DraftReadoutSubmit&);
    void run_dflash_draft(const speculative::RoundRoute&, std::span<const ops::ContextBucket>,
                          const execution::DraftReadoutSubmit& = {});
    void project_decode(std::uint32_t batch, std::uint32_t width);
    void decide_decode(const speculative::RoundRoute& target,
                       const speculative::RoundRoute& proposal);
    void readback_result(std::uint32_t batch, const speculative::RoundRoute* route = nullptr);
    GraphPreparationSample initialize_graphs(GraphPreparationMode = GraphPreparationMode::Complete);
    void release_graphs();
    void append_draft_history(ComputeOperation&);
    void append_draft_history(std::uint32_t batch, ops::KVCacheAppendPrefixExecutionEnvelope);
    void commit_draft_history(ComputeOperation&);
    void align_mtp_history(ComputeOperation&);
    void align_packed_mtp_history(const ComputeOperation&);
    void rebuild_mtp_history(std::uint32_t batch, std::uint32_t width);
    void execute_decode_commit(std::uint32_t batch, std::uint32_t width, bool repair, bool counts,
                               const SectionRecorder& repair_section = {},
                               const SectionRecorder& counts_section = {});
    void request_decision(ComputeOperation&);
    void launch_decision(ComputeOperation&);
    void launch_commit(ComputeOperation&);
    void launch_score(ComputeOperation&);
    void publish_result(ComputeOperation&);
    void retire_compute();
    void progress();
    [[nodiscard]] ComputeOperation& require_compute(runtime::ExecutionTicket);
    [[nodiscard]] const ComputeOperation& require_compute(runtime::ExecutionTicket) const;
    [[nodiscard]] ops::SamplingConfig sampling_config(const ExecutionRow&, const SamplingInput&,
                                                      std::size_t compact_row) const;

    const execution::Parameters& parameters;
    DeviceContext& device;
    const std::uint32_t capacity;
    const std::uint32_t max_concurrency;
    const std::shared_ptr<const BatchExecutionPlan> batch_execution;
    const std::shared_ptr<const speculative::DecodeGraphRoutes> decode_routes;
    const std::uint32_t prefill_chunk;
    const std::uint32_t draft_window;
    const SpeculativeBackend backend;
    const ProposalHead proposal_head;
    const bool use_cuda_graph;
    const KvCacheStorage kv_storage;
    const WorkspaceSpec workspace_plan;
    Resources resources;
    std::unique_ptr<context::ContextStorage> storage;
    std::unique_ptr<context::ContextMigration> compute_actions;
    std::unique_ptr<context::ContextMigration> context_actions;
    std::unique_ptr<ContextOperation> context_operation;
    PinnedHostBuffer context_counts;
    std::vector<Residency> residencies;
    mutable std::vector<std::uint64_t> validation_marks;
    mutable std::uint64_t validation_generation = 0;
    std::size_t workspace_peak                  = 0;
    std::unique_ptr<ComputeOperation> compute;
    CudaCompletionEvent result_ready;
    CudaCompletionEvent drafts_ready;
    CudaCompletionEvent compute_retired;
    CudaCompletionEvent context_ready;
    CudaEventTimer prefill_timer;
    runtime::ExecutionGpuTimer gpu_timer;
    std::deque<runtime::ExecutionEvent<ExecutionTypes>> events;
    std::unique_ptr<models::DecodeGraphSuite> graphs;
    std::optional<GraphMemorySummary> graph_memory;
    std::size_t workspace_lifetime_peak = 0;
};

} // namespace ninfer::models::qwen3_5::detail
