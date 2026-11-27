#include "models/common/moe/execution.h"

#if NINFER_ENABLE_DISTRIBUTED
#    include "models/common/moe/exchange.h"
#endif

namespace ninfer::models::moe {

void SigmoidPass::run(ExpertBankId bank, const ops::SigmoidMoeProfile& p,
                      const ops::SigmoidMoeWeights& weights, const LayerSource* source,
                      ParallelExecution* tensor, MemoryArena& arena, CudaExecutionView cuda) const {
#if NINFER_ENABLE_DISTRIBUTED
    auto step = exchange->stage(bank, columns);
    step.frame.sigmoid_layer(step.placement, p, weights, columns, source, representative, tensor,
                             arena, cuda);
#else
    throw std::logic_error("sigmoid expert exchange requires distributed execution");
#endif
}

void resident_layer(const Parameters& p, const LayerSource* src, const ExpertExchangeWork* step,
                    MemoryArena& workspace, CudaExecutionView cuda) {
    if (step) {
#if NINFER_ENABLE_DISTRIBUTED
        step->frame.softmax_layer(step->placement, p, step->physical_columns, src, workspace, cuda,
                                  step->observation);
        return;
#else
        throw std::invalid_argument("expert exchange requires distributed execution");
#endif
    }
    if (!src) throw std::invalid_argument("local resident MoE requires a source");
    auto output = src->output;
    execute(src->hidden, p, p.resident, src->mode, output, src->validity, workspace, cuda);
}


std::size_t workspace_bytes(const ops::MoeProfile& profile, ops::MoeEpilogue epilogue,
                            std::int32_t first, std::int32_t last, ops::ExpertSubsetKind subset) {
    return ops::topk_softmax_swiglu_moe_workspace_capacity_bytes(profile, subset, epilogue, first,
                                                                 last);
}

void execute(const Tensor& hidden, const Parameters& parameters,
             const ExpertBankDeviceView& experts, ops::MoeEpilogue epilogue, Tensor& output,
             ops::MoeTokenValidity validity, MemoryArena& workspace, CudaExecutionView execution,
             const ops::MoeExecutionHints& hints) {
    auto scope         = workspace.scope();
    const auto columns = hidden.ne[1];
    const auto storage = workspace.alloc_bytes(
        workspace_bytes(parameters.profile, epilogue, columns, columns, parameters.subset.kind));
    MemoryArena scratch(storage);
    const ops::MoeWeights weights{
        parameters.control, {experts, parameters.activation_calibration}, parameters.shared};
    ops::topk_softmax_swiglu_moe(hidden, parameters.profile, weights, parameters.subset, epilogue,
                                 output, validity, scratch, execution, hints);
}

} // namespace ninfer::models::moe
