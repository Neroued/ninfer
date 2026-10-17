#pragma once

#include "models/qwen4_exp/execution/batch.h"
#include "models/qwen4_exp/execution/workspace.h"
#include "models/qwen4_exp/execution/parameters.h"
#include "models/qwen4_exp/state/state_image.h"
#include "models/qwen4_exp/state/kv_cache.h"
#include "models/common/moe/schedule.h"

namespace ninfer::models::qwen4_exp::execution {

struct TextExecution {
    MemoryArena& workspace;
    const StateViews& states;
    const QsaCache& cache;
    moe::Schedule* experts = nullptr;
    Tensor qsa_raw_records, qsa_coordinate_records, ple_records;
};

// Activations stay live across the Host PLE handoff. All borrowed views follow the Program's
// compute stream; canonical publication belongs to Program.
class TextRun {
public:
    TextRun(const Parameters&, TextExecution, BatchView, StateUpdate, CudaExecutionView);
    void prefix(std::optional<VisualEmbeddingView> = {});
    void consume_ple(const Tensor& embedding);
    void suffix(TextOutputs, Tensor logits = {});

    // GPU bodies share the same activation frame in eager execution and graph capture.
    void stem(std::optional<VisualEmbeddingView> = {});
    void mixer(std::uint32_t layer);
    void resident_moe(std::uint32_t layer);
    void gpu_moe(const moe::GpuCall&);
    void combine(std::uint32_t layer);
    void finish(TextOutputs, Tensor logits = {});

    [[nodiscard]] Tensor& block_output() { return roots_.output; }

    void set_visible_tokens(std::uint32_t maximum) { batch_.max_visible_tokens = maximum; }

private:
    void layer(std::uint32_t);
    const Parameters& parameters_;
    TextExecution execution_;
    BatchView batch_;
    StateUpdate update_;
    CudaExecutionView cuda_;
    MemoryArena::Scope scope_;
    workspace::Roots roots_;
    std::optional<moe::Schedule::Activity> expert_activity_;
};

void project_head(const Tensor& sample, const LinearParameters&, Tensor& logits, MemoryArena&,
                  CudaExecutionView);

} // namespace ninfer::models::qwen4_exp::execution
