#pragma once

#include "models/qwen4_exp/program/program.h"
#include "models/qwen4_exp/program/resources.h"
#include "models/qwen4_exp/program/ple_pipeline.h"
#include "models/qwen4_exp/program/graphs.h"
#include "models/qwen4_exp/execution/text.h"
#include "models/common/qwen/vision/prefill.h"
#include "ninfer/ops/sampling.h"

#include <deque>

namespace ninfer::models::qwen4_exp::detail {

struct ResidentBinding {
    runtime::ExecutionMember member;
    std::shared_ptr<const qwen::VisionExecutionInput> vision_input;
    std::unique_ptr<qwen::VisionPrefillExecution> vision;
    double observed_vision_seconds = 0;
};

struct MtpInputRow {
    std::uint32_t base = 0;
    std::span<const TokenId> ids;
};

struct VerifyRound {
    std::uint32_t width = 0;
    std::vector<std::uint32_t> bases, extents;
    std::vector<TokenId> inputs;
    execution::BatchView batch;
};

enum class ComputePhase : std::uint8_t {
    Prepare,
    AwaitPle,
    Prefix,
    Suffix,
    AfterMain,
    Decision,
    AwaitResult,
    Score,
    AwaitScore,
    ControlCounts,
    AwaitControl,
    PendingCommit,
    Retiring,
};

struct ComputeFrame {
    runtime::ExecutionRequest<ExecutionTypes> request;
    ComputePhase phase = ComputePhase::Prepare;
    std::span<const WorkRow> rows;
    std::optional<ple::Ticket> ple;
    std::optional<VerifyRound> verify;
    std::unique_ptr<execution::TextRun> run;
    TextGraph* graph = nullptr;
    execution::BatchView batch;
    Tensor sample, conditioning, previous_conditioning;
    std::optional<execution::VisualEmbeddingView> visual;
    std::optional<runtime::DecisionInput> decision;
    bool decision_requested = false, sequence = false;
    bool first_sample = false, timer_started = false;
    std::uint32_t tokens = 0, visible = 0, layer = 0;
    std::uint32_t score_at = 0, score_count = 0;
    std::vector<float> logprobs;
    std::uint32_t control_row    = 0;
    std::uint32_t control_offset = 0;
    std::optional<WorkRow> control_window;
    std::vector<std::uint32_t> dead_positions;
    runtime::CommandSequence retirement;
    runtime::ExecutionTiming timing;
    ComputeResult result;
};

struct ContextFrame {
    runtime::ExecutionRequest<ExecutionTypes> request;
    bool submitted = false;
};

class RankProgramImpl {
public:
    RankProgramImpl(const execution::Parameters&, ExecutionPlan, DeviceContext&,
                    const StartupObserver& observer = {});
    ~RankProgramImpl();

    void initialize_graphs();
    void release_graphs();
    void validate_work(const Work&, runtime::FrameKind) const;
    void start(runtime::ExecutionRequest<ExecutionTypes>&&);
    void validate_decision(runtime::ExecutionTicket, const runtime::DecisionInput&) const;
    void supply(runtime::ExecutionTicket, runtime::CommandSequence, runtime::DecisionInput&&);
    void prepare_commit(runtime::ExecutionTicket, const Commit&) const;
    [[nodiscard]] bool can_discard(runtime::ExecutionTicket) const;
    void commit(runtime::ExecutionTicket, runtime::CommandSequence, Commit&&);
    void discard(runtime::ExecutionTicket, runtime::CommandSequence);
    [[nodiscard]] std::optional<runtime::ExecutionEvent<ExecutionTypes>> poll();
    void drain();
    void advance_compute();
    void advance_context();
    void prepare_forward();
    void prepare_mtp();
    void prepare_main();
    void after_main();
    void prepare_decision();
    void finish_result();
    void score_next();
    void prepare_control_window();
    void finish_control_window();
    void enqueue_control_counts();
    void retire_compute();
    [[nodiscard]] bool advance_layers(std::uint32_t end);
    void request_decision();
    void install_sampling(bool speculative);
    void upload_main(std::span<const WorkRow>, std::uint32_t tokens);
    void align_mtp(std::span<const WorkRow>, std::uint32_t tokens, const Tensor& previous,
                   const Tensor& conditioning, std::optional<execution::VisualEmbeddingView>);
    void commit_verify(const Commit&);
    void draft_forward(std::uint32_t step, std::int32_t batch, std::uint32_t visible);
    void save_draft(std::uint32_t step, std::int32_t batch);
    void sample_forward(std::int32_t batch, bool first_token);
    void accept_forward(std::int32_t batch);
    [[nodiscard]] execution::BatchView mtp_inputs(const MtpBatchLayout&, std::span<const WorkRow>,
                                                  std::uint32_t tokens,
                                                  std::span<const MtpInputRow>);
    [[nodiscard]] ops::QsaTailStateView mtp_working_tail(std::uint32_t slot, std::uint32_t batch);
    [[nodiscard]] Tensor device_tensor(const TensorRegion&);
    [[nodiscard]] Tensor host_tensor(const TensorRegion&);
    [[nodiscard]] Tensor packed_tensor(const TensorRegion&, std::initializer_list<std::int32_t>);
    [[nodiscard]] ResidentBinding& binding(runtime::ResidencyId);
    [[nodiscard]] std::optional<execution::VisualEmbeddingView>
    prepare_vision(ResidentBinding&, std::uint32_t begin, std::uint32_t tokens);
    double collect_vision_timing(ResidentBinding&, std::uint32_t committed_frontier);
    void retire_vision(ResidentBinding&) noexcept;
    [[nodiscard]] std::size_t active_vision_bytes() const noexcept;
    [[nodiscard]] MemorySummary memory_summary() const;
    void reset_memory_peaks() noexcept;

    const execution::Parameters& parameters;
    DeviceContext& device;
    CudaExecutionView cuda;
    Resources resources;
    PlePipeline ple;
    const ExecutionShape& shape;
    std::unique_ptr<Graphs> graphs;
    std::optional<GraphMemorySummary> graph_memory;
    CudaEventTimer prefill_timer;
    CudaCompletionEvent compute_done, context_done;
    qwen::VisionHandoffState vision_handoff;
    std::size_t vision_handoff_peak_bytes = 0, vision_workspace_peak_bytes = 0;
    std::array<std::optional<ResidentBinding>, kMaximumConcurrency> bindings;
    std::optional<ComputeFrame> compute;
    std::optional<ContextFrame> context;
    std::deque<runtime::ExecutionEvent<ExecutionTypes>> events;
    bool drained = false;
};

[[nodiscard]] ops::SamplingConfig sampling_config(const ResolvedSamplingParameters&);

} // namespace ninfer::models::qwen4_exp::detail
