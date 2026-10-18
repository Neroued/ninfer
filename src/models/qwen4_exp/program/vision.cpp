#include "models/qwen4_exp/program/internal.h"
#include "core/transfer.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp::detail {

std::optional<execution::VisualEmbeddingView>
RankProgramImpl::prepare_vision(ResidentBinding& binding, std::uint32_t begin,
                                std::uint32_t tokens) {
    const auto& input = binding.vision_input;
    if (!input || begin >= input->prompt_tokens) return {};
    const auto& plan = input->plan;
    const auto item  = std::lower_bound(
        plan.uses.begin(), plan.uses.end(), begin,
        [](const auto& use, std::uint32_t position) { return use.end <= position; });
    if (item == plan.uses.end() || item->begin >= std::uint64_t(begin) + tokens) return {};
    if (!binding.vision) {
        binding.vision = std::make_unique<qwen::VisionPrefillExecution>(
            device, *parameters.profile->config.vision, *parameters.vision,
            resources.workspace_storage(), *resources.plan.workspace.vision, input, vision_handoff,
            vision_handoff_peak_bytes);
    }
    const auto chunk = binding.vision->prepare_chunk(begin, tokens);
    if (chunk.length != static_cast<std::int32_t>(tokens) || !chunk.control)
        throw std::logic_error("Vision execution differs from the reserved input window");
    vision_workspace_peak_bytes =
        std::max(vision_workspace_peak_bytes, chunk.encode_workspace_bytes);
    const auto slice = qwen::visual_embedding_slice(chunk.control->scatter_indices,
                                                    input->prompt_tokens, begin, tokens);
    if (slice.empty()) return {};
    const auto count   = static_cast<std::int32_t>(slice.size());
    const auto& region = *resources.plan.persistent.buffers.batch.visual_indices;
    auto indices       = device_tensor(region).slice(0, 0, count);
    auto host          = host_tensor(region).slice(0, 0, count);
    std::copy(slice.destination_columns.begin(), slice.destination_columns.end(),
              static_cast<std::int32_t*>(host.data));
    copy_bytes_async({indices.data, indices.bytes(), cuda.device()},
                     {host.data, host.bytes(), Device::cpu(), true}, cuda);
    return execution::VisualEmbeddingView{
        chunk.embeddings.slice(1, static_cast<std::int32_t>(slice.source_begin), count), indices};
}

double RankProgramImpl::collect_vision_timing(ResidentBinding& binding,
                                              std::uint32_t committed_frontier) {
    if (!binding.vision) return 0;
    const auto elapsed              = binding.vision->elapsed_seconds();
    const auto increment            = elapsed - binding.observed_vision_seconds;
    binding.observed_vision_seconds = elapsed;
    if (committed_frontier >= binding.vision_input->plan.uses.back().end) retire_vision(binding);
    return increment;
}

void RankProgramImpl::retire_vision(ResidentBinding& binding) noexcept {
    binding.vision.reset();
    binding.observed_vision_seconds = 0;
}

std::size_t RankProgramImpl::active_vision_bytes() const noexcept {
    std::size_t bytes = 0;
    for (const auto& binding : bindings)
        if (binding && binding->vision)
            bytes = std::max(bytes, binding->vision->active_handoff_bytes());
    return bytes;
}

} // namespace ninfer::models::qwen4_exp::detail
