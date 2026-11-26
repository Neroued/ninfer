#pragma once

#include "models/inkling2/execution/batch.h"
#include "models/inkling2/execution/parameters.h"
#include "models/inkling2/execution/workspace.h"
#include "models/inkling2/state/context.h"

namespace ninfer::models::inkling2::execution {

class LoopResidualRun {
public:
    LoopResidualRun(const Parameters&, workspace::MainRoots&, const BatchView&, const StateViews&,
                    const ops::RaggedSequenceView*, MemoryArena&, CudaExecutionView);

    void begin_visit(const LogicalLayerConfig& logical, std::size_t visit_index);
    void after_attention(const LogicalLayerConfig& logical, const Tensor& attn_proj,
                         Tensor& mlp_in_bf16);
    void after_mlp(const LogicalLayerConfig& logical, std::size_t visit_index,
                   const Tensor& moe_out, Tensor& final_hidden);

private:
    [[nodiscard]] bool at_core_start(const LogicalLayerConfig&) const;
    [[nodiscard]] bool starts_block(const LogicalLayerConfig&) const;
    [[nodiscard]] Tensor stack_slot(std::uint32_t slot) const;
    [[nodiscard]] Tensor view_tokens(const Tensor&) const;
    void restart(const PhysicalLayerParameters&);
    void blend_attention(const LogicalLayerConfig&, const PhysicalLayerParameters&);
    void initial_attention_norm(const LogicalLayerConfig&, const PhysicalLayerParameters&);

    const TextParameters& parameters_;
    const TextConfig& config_;
    const TextProfile& profile_;
    workspace::MainRoots& roots_;
    const BatchView& batch_;
    const StateViews& states_;
    const ops::RaggedSequenceView* ragged_;
    MemoryArena& arena_;
    CudaExecutionView cuda_;
    std::uint32_t num_blocks_     = 0;
    std::uint32_t prelude_blocks_ = 0;
    bool prelude_dests_ready_     = false;
};

} // namespace ninfer::models::inkling2::execution
