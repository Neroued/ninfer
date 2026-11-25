#pragma once
#include "models/common/graph/suite.h"
#include "models/common/vocabulary.h"
#include "models/glm5next/execution/mtp.h"
#include "models/glm5next/execution/dflash.h"
#include "models/glm5next/execution/text.h"
#include "models/glm5next/program/program.h"
#include "ninfer/ops/sampling.h"
#include "runtime/observability/gpu_timing.h"

#include <deque>

namespace ninfer::models::glm5next::detail {
class GraphInputs;

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

struct PackedPrefill {
    execution::PackedPrefillUpdate update;
    ops::RaggedSequenceView mtp_ragged, dflash_ragged;
    std::vector<std::uint32_t> sampled_members;
};

struct ComputeFrame {
    runtime::ExecutionRequest<ExecutionTypes> request;
    runtime::CommandSequence retirement;
    ComputePhase phase = ComputePhase::Forward;
    ComputeResult result;
    std::optional<runtime::DecisionRequest> requested;
    std::optional<runtime::DecisionInput> decision;
    std::optional<WorkRow> window;
    std::optional<PackedPrefill> packed;
    BatchPlan physical;
    std::optional<speculative::RoundRoute> graph_route;
    std::vector<std::uint32_t> draft_visible;
    std::uint32_t control_row = 0, control_offset = 0, tokens = 0;
    std::uint32_t score_at = 0, score_count = 0;
    std::vector<float> logprobs;
    execution::BatchView batch;
    Tensor hidden;
    Tensor sampled_hidden;
    GpuExecutionObservation gpu;
    AttentionWorkCounters attention;
    bool sequence = false, timed_prefill = false, counts_pending = false;
    bool head_ready     = false;
    bool release_vision = false;

    [[nodiscard]] std::span<const std::uint32_t> sampling_members() const noexcept {
        return packed ? packed->sampled_members : physical.members;
    }
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
    void forward_packed_prefill(const ForwardWork&);
    GraphPreparationSample prepare_graphs(GraphPreparationMode);
    void release_graphs();
    void execute_proposal(const execution::BatchView&, std::span<const std::uint32_t> draft_bounds,
                          int head_columns, bool read_drafts);
    void begin_proposal(const execution::BatchView&, bool capture);
    void readback_drafts(const Tensor& drafts);
    void prepare_verify_inputs(const execution::BatchView&, const Tensor& drafts);
    void execute_target(const execution::BatchView&, bool head, bool forced_control = false);
    void begin_target(const execution::BatchView&, bool capture);
    [[nodiscard]] Tensor target_hidden(const execution::BatchView&);
    [[nodiscard]] execution::StateUpdate target_update(int width);
    void target_forward(const execution::BatchView&, Tensor hidden, bool forced_control,
                        const execution::MlaSubmit& = {});
    [[nodiscard]] execution::TextRun text_run(const execution::BatchView&, execution::StateUpdate);
    [[nodiscard]] execution::MtpRun mtp_run(std::uint32_t step, const execution::BatchView&);
    [[nodiscard]] execution::MlaSelection mtp_selection(std::uint32_t step, int local);
    void copy_result(int rows, bool verify, SampleKind kind = SampleKind::NextToken);
    void prepare_speculative();
    void mtp_forward(std::uint32_t step, const execution::BatchView&,
                     const execution::MlaSubmit& = {});
    void main_forward(const execution::BatchView&, execution::StateUpdate, Tensor hidden,
                      std::optional<models::vision::EmbeddingView> = {},
                      const execution::MlaSubmit&                  = {});
    void project_head(Tensor hidden);
    void accept_mtp(int rows, int width);
    void accept_dflash(int rows, int width, int head_columns);
    void accept_speculative(int rows, int width, int head_columns);
    [[nodiscard]] execution::DFlashRun dflash_run(int rows, ops::ContextBucket, int head_columns);
    void dflash_forward(int rows, ops::ContextBucket, int head_columns,
                        const execution::DraftSubmit& = {});
    [[nodiscard]] Tensor draft_features(const execution::BatchView&, bool packed = false);
    void append_dflash_context(const execution::BatchView&, const Tensor& counts,
                               ops::ContextKVMaterializeExecutionEnvelope);
    void append_packed_dflash_context(const execution::BatchView&, const ops::RaggedSequenceView&);
    void execute_speculative_commit(const execution::BatchView&, bool repair_history,
                                    bool count_tokens, const SectionRecorder& repair = {},
                                    const SectionRecorder& counts = {});
    [[nodiscard]] DistributedVocabulary vocabulary();
    void commit_speculative(const Commit&);
    void repair_mtp_history(const execution::BatchView&, bool repair,
                            const SectionRecorder& record_repair = {});
    void stage_conditioning(std::int32_t rows);
    void gather_decode_state(const execution::BatchView&);
    execution::BatchView decode_batch(int tokens, int rows, int local_rows, std::uint32_t visible);
    void fill_local_metadata(const BatchLayout&, int tokens, int rows);
    [[nodiscard]] Tensor draft_proposals(int batch, int columns);
    [[nodiscard]] Tensor mtp_output_ids(int step, int batch);
    void align_history(const execution::BatchView&, Tensor hidden, std::int32_t width);
    void align_packed_history(const execution::BatchView&, const PackedPrefill&, const Tensor&);
    void prepare_history_metadata(std::span<const WorkRow>);
    [[nodiscard]] execution::BatchView batch(const BatchLayout&, int tokens, int rows,
                                             int attention_rows, std::uint32_t visible,
                                             std::optional<std::uint32_t> sequence_owner = {});
    void select_frame(PhysicalBatchShape);
    [[nodiscard]] Tensor row(const TensorRegion&, int rows);
    [[nodiscard]] MemorySpan output(const TensorRegion&);
    void finish_forward();
    void request_decision();
    void decide();
    void sample(std::int32_t rows, SampleKind);
    void score();
    void result();
    void prepare_inputs(std::span<const WorkRow>, const BatchPlan&, std::uint32_t tokens,
                        std::span<const std::uint32_t> valid_columns = {});
    void prepare_packed_inputs(const ForwardWork&);
    [[nodiscard]] ops::RaggedSequenceView upload_ragged(const RaggedLayout&, int segments);
    void set_sampling_config(const WorkRow&, int slot);
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
    FrameLayout frame;
    std::optional<ParallelExecution> parallel;
    std::unique_ptr<GraphInputs> graph_inputs;
    std::unique_ptr<models::DecodeGraphSuite> graphs;
    GraphMemorySummary graph_memory;
    std::unique_ptr<ComputeFrame> compute;
    std::unique_ptr<ContextFrame> context;
    std::vector<std::optional<runtime::ExecutionMember>> bindings;
    std::deque<runtime::ExecutionEvent<ExecutionTypes>> events;
    runtime::ExecutionGpuTimer timer;
    std::optional<CudaEventTimer> prefill_timer;
    CudaCompletionEvent compute_done, context_done, drafts_done;
    std::size_t lifetime_peak = 0;
    bool drained              = false;
};
} // namespace ninfer::models::glm5next::detail
