#pragma once

#include "models/common/parallel_execution.h"
#include "models/glm5next/execution/parameters.h"
#include "models/glm5next/execution/workspace.h"
#include "models/glm5next/state/context.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/sliding_window_attention.h"

#include <functional>

namespace ninfer::models::glm5next::execution {

struct DraftExecution {
    MemoryArena& workspace;
    const StateViews& states;
    ParallelExecution& parallel;
};

struct DraftInputs {
    Tensor anchors, frontiers, valid_columns, state_offsets;
    const ops::SamplingConfig* sampling;
    std::uint32_t context_bound;
};

struct DraftOutputs {
    Tensor ids, positions, drafts, candidate_ids, proposal_q;
};

struct DraftReadout {
    Tensor query, key, value, positions, valid_columns, output;
    ConstCyclicKVCacheLayerView context;
    ops::AttentionHeadGeometry geometry;
    MemorySpan scratch;

    [[nodiscard]] ops::SlidingWindowAttentionGraphProfile profile(std::uint32_t) const;
    void execute(std::uint32_t, CudaExecutionView) const;
};

using DraftSubmit = std::function<void(const DraftReadout&)>;

class DraftAttentionRun {
public:
    DraftAttentionRun(const Parameters&, DraftExecution, const DraftInputs&, const DraftOutputs&,
                      Tensor residual, std::uint32_t layer, CudaExecutionView);
    [[nodiscard]] DraftReadout readout();
    void execute(const DraftSubmit& = {});
private:
    const Parameters& parameters_;
    DraftExecution execution_;
    DraftInputs inputs_;
    Tensor residual_, positions_;
    std::uint32_t layer_;
    CudaExecutionView cuda_;
    MemoryArena::Scope scope_;
    workspace::DraftAttentionBuffers buffers_;
};

class DFlashRun {
public:
    DFlashRun(const Parameters&, DraftExecution, DraftInputs, DraftOutputs, CudaExecutionView);
    void stem();
    [[nodiscard]] DraftAttentionRun attention(std::uint32_t layer);
    void mlp(std::uint32_t layer);
    void finish();
private:
    const Parameters& parameters_;
    DraftExecution execution_;
    DraftInputs inputs_;
    DraftOutputs outputs_;
    CudaExecutionView cuda_;
    MemoryArena::Scope scope_;
    Tensor residual_;
};

void draft_append_context(const Parameters&, DraftExecution, const Tensor& features,
                          const Tensor& positions, const Tensor& counts, const Tensor& offsets,
                          ops::ContextKVMaterializeExecutionEnvelope, CudaExecutionView);

[[nodiscard]] std::size_t draft_workspace_capacity(const ModelProfile&, int width, int max_batch);
[[nodiscard]] std::size_t draft_context_workspace_capacity(const ModelProfile&,
                                                           const ExecutionShape&);

} // namespace ninfer::models::glm5next::execution
