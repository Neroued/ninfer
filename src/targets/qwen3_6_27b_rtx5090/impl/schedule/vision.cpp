#include "targets/qwen3_6_27b_rtx5090/impl/schedule/schedule.h"

#include "targets/qwen3_6_27b_rtx5090/impl/schedule/vision_context.h"

#include <stdexcept>

namespace ninfer::targets::qwen3_6_27b_rtx5090::detail::schedule {

std::size_t vision_workspace_bytes(const PreparedPromptData& prompt) {
    return prompt.has_media() ? VisionContext::workspace_bytes(prompt) : 0;
}

Tensor encode_vision(State& state, const PreparedPromptData& prompt,
                     runtime::TransientRegion transient) {
    const std::size_t required = VisionContext::workspace_bytes(prompt);
    if (transient.data == nullptr || transient.size < required || transient.alignment < 256) {
        throw std::invalid_argument("Vision transient region is too small or misaligned");
    }
    WorkspaceArena workspace(DeviceSpan{transient.data, transient.size});
    VisionContext vision(state.device, state.model);
    void* tap_context = state.diagnostic_vision_tap != nullptr ? state.diagnostic_context : nullptr;
    return vision.encode(prompt, workspace, tap_context, state.diagnostic_vision_tap);
}

} // namespace ninfer::targets::qwen3_6_27b_rtx5090::detail::schedule
