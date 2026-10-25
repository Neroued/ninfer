#pragma once
#include "models/glm5next/program/program.h"
#include "models/glm5next/execution/text.h"
#include "runtime/observability/gpu_timing.h"
#include "ninfer/ops/sampling.h"
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
    std::uint32_t control_row = 0, control_offset = 0, tokens = 0;
    std::uint32_t score_at = 0, score_count = 0;
    std::vector<float> logprobs;
    execution::BatchView batch;
    Tensor hidden;
    GpuExecutionObservation gpu;
    bool sequence = false, timed_prefill = false, counts_pending = false;
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
    void forward();
    void finish_forward();
    void request_decision();
    void decide();
    void sample();
    void score();
    void result();
    void upload(std::span<const WorkRow>, const BatchPlan&, std::uint32_t tokens);
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
    std::unique_ptr<ComputeFrame> compute;
    std::unique_ptr<ContextFrame> context;
    std::array<std::optional<runtime::ExecutionMember>, kMaximumConcurrency> bindings;
    std::deque<runtime::ExecutionEvent<ExecutionTypes>> events;
    runtime::ExecutionGpuTimer timer;
    std::optional<CudaEventTimer> prefill_timer;
    CudaCompletionEvent compute_done, context_done;
    std::size_t lifetime_peak = 0;
    bool drained              = false;
};
} // namespace ninfer::models::glm5next::detail
