#include "models/common/moe/execution.h"

namespace ninfer::models::moe {

std::size_t workspace_bytes(const ops::MoeProfile& profile, ops::MoeEpilogue epilogue,
                            std::int32_t first, std::int32_t last) {
    return ops::topk_softmax_swiglu_moe_workspace_capacity_bytes(profile, epilogue, first, last);
}

void execute(const Tensor& hidden, const Parameters& parameters,
             const ExpertBankDeviceView& experts, ops::MoeEpilogue epilogue, Tensor& output,
             MemoryArena& workspace, CudaExecutionView execution,
             const ops::MoeExecutionHints& hints) {
    auto scope         = workspace.scope();
    const auto columns = hidden.ne[1];
    const auto storage =
        workspace.alloc_bytes(workspace_bytes(parameters.profile, epilogue, columns, columns));
    MemoryArena scratch(storage);
    const ops::MoeWeights weights{
        parameters.control, {experts, parameters.activation_divisors}, parameters.shared};
    ops::topk_softmax_swiglu_moe(hidden, parameters.profile, weights, epilogue, output, scratch,
                                 execution, hints);
}

} // namespace ninfer::models::moe
