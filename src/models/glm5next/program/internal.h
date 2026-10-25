#pragma once
#include "models/common/vocabulary.h"
#include "models/glm5next/execution/mtp.h"
#include "models/glm5next/execution/text.h"
#include "models/glm5next/program/graphs.h"
#include "models/glm5next/program/program.h"
#include "ninfer/ops/sampling.h"
#include "runtime/observability/gpu_timing.h"

#include <deque>

namespace ninfer::models::glm5next::detail {
enum class ComputePhase {
    Forward,
    AwaitForward,
    Decision,
    AwaitResult,
    Score,
    AwaitScore,
    PendingCommit,
    Retiring
};

struct ComputeFrame {
    runtime::ExecutionRequest<ExecutionTypes> request;
    runtime::CommandSequence retirement;
    ComputePhase phase = ComputePhase::Forward;
    ComputeResult result;
    std::optional<runtime::DecisionRequest> requested;
    std::optional<runtime::DecisionInput> decision;
    std::optional<WorkRow> window;
    BatchPlan physical;
    std::vector<std::uint32_t> draft_visible;
    std::uint32_t control_row = 0, control_offset = 0, tokens = 0;
    std::uint32_t score_at = 0, score_count = 0;
    std::vector<float> logprobs;
    execution::BatchView batch;
    Tensor hidden;
    GpuExecutionObservation gpu;
    bool sequence = false, timed_prefill = false, counts_pending = false;
    bool head_ready = false, graph_round = false;
    bool release_vision = false;
};

struct ContextFrame {
    runtime::ExecutionRequest<ExecutionTypes> request;
    bool submitted = false;
};

class RankProgramImpl {
public:
    RankProgramImpl(const execution::Parameters&, std::unique_ptr<Resources>, DeviceContext&,
                    ParallelBinding);
    ~RankProgramImpl();
    void validate_work(const Work&, runtime::FrameKind) const;
    void start(runtime::ExecutionRequest<ExecutionTypes>&&);
    void validate_decision(runtime::ExecutionTicket, const runtime::DecisionInput&) const;
    void supply(runtime::ExecutionTicket, runtime::CommandSequence, runtime::DecisionInput&&);
    void prepare_commit(runtime::ExecutionTicket, const Commit&) const;
    void commit(runtime::ExecutionTicket, runtime::CommandSequence, Commit&&);
    void discard(runtime::ExecutionTicket, runtime::CommandSequence);
    bool can_discard(runtime::ExecutionTicket) const;
    std::optional<runtime::ExecutionEvent<ExecutionTypes>> poll();
    void drain();
    void advance_compute();
    void advance_context();
    [[nodiscard]] std::optional<models::vision::EmbeddingView>
    prepare_vision(const std::optional<vision::Work>&, std::uint32_t begin, std::uint32_t tokens);
    void collect_vision_timing();
    void retire_vision(runtime::ResidencyId) noexcept;
    void forward();
    void prepare_graphs();
    void release_graphs();
    void execute_round(const execution::BatchView&, std::span<const std::uint32_t> draft_bounds,
                       bool head, bool read_drafts, const execution::MlaSubmit& = {},
                       RoundSections* = nullptr);
    [[nodiscard]] execution::TextRun text_run(const execution::BatchView&, execution::StateUpdate);
    [[nodiscard]] execution::MtpRun draft_run(std::uint32_t step, const execution::BatchView&);
    [[nodiscard]] execution::MlaSelection draft_selection(std::uint32_t step, int local);
    void copy_result(int rows, bool verify);
    void prepare_mtp();
    void draft_forward(std::uint32_t step, const execution::BatchView&,
                       const execution::MlaSubmit& = {});
    void main_forward(const execution::BatchView&, execution::StateUpdate, Tensor hidden,
                      std::optional<models::vision::EmbeddingView> = {},
                      const execution::MlaSubmit&                  = {});
    void project_head(Tensor hidden);
    void accept_mtp(std::int32_t rows);
    [[nodiscard]] DistributedVocabulary vocabulary();
    void commit_mtp(const Commit&);
    void execute_mtp_commit(const execution::BatchView&, bool repair_history,
                            DecodeGraphSection* = nullptr);
    void stage_conditioning(std::int32_t rows);
    void seed_mtp_tail(const execution::BatchView&, bool canonical_destination);
    void localize(const execution::BatchView&);
    void align_history(const execution::BatchView&, Tensor hidden, std::int32_t width);
    void prepare_history_metadata(std::span<const WorkRow>,
                                  std::span<const std::uint32_t> skip_first = {});
    [[nodiscard]] execution::BatchView batch(const BatchLayout&, int tokens, int rows,
                                             int attention_rows, std::uint32_t visible,
                                             std::optional<std::uint32_t> sequence_owner = {});
    [[nodiscard]] Tensor row(const TensorRegion&, int rows);
    [[nodiscard]] MemorySpan output(const TensorRegion&);
    void finish_forward();
    void request_decision();
    void decide();
    void sample(std::int32_t rows, SampleKind);
    void score();
    void result();
    void upload(std::span<const WorkRow>, const BatchPlan&, std::uint32_t tokens,
                std::span<const std::uint32_t> valid_columns = {});
    void upload_sampling();
    void control_counts();
    [[nodiscard]] MemorySpan host_region(const LayoutRegion&);
    [[nodiscard]] Tensor host(const TensorRegion&);
    void upload_region(const LayoutRegion&, std::size_t bytes);
    [[nodiscard]] WorkContract contract() const;
    [[nodiscard]] RankExecutionObservation observation() const;
    [[nodiscard]] RankMemoryUsage memory() const;
    [[nodiscard]] RankMemoryBudget budget() const;
    void reset_peaks();

    const execution::Parameters& parameters;
    DeviceContext& device;
    CudaExecutionView cuda;
    std::unique_ptr<Resources> owner;
    Resources& resources;
    const ExecutionShape shape;
    std::optional<ParallelExecution> parallel;
    std::unique_ptr<GraphInputs> graph_inputs;
    std::unique_ptr<Graphs> graphs;
    GraphMemorySummary graph_memory;
    std::unique_ptr<ComputeFrame> compute;
    std::unique_ptr<ContextFrame> context;
    std::array<std::optional<runtime::ExecutionMember>, kMaximumConcurrency> bindings;
    std::deque<runtime::ExecutionEvent<ExecutionTypes>> events;
    runtime::ExecutionGpuTimer timer;
    std::optional<CudaEventTimer> prefill_timer;
    CudaCompletionEvent compute_done, context_done, drafts_done;
    std::size_t lifetime_peak = 0;
    bool drained              = false;
};
} // namespace ninfer::models::glm5next::detail
