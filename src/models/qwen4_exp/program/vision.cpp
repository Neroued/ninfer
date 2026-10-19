#include "models/qwen4_exp/program/internal.h"
#include "core/transfer.h"
#include "models/qwen4_exp/parallel/vision.h"

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
    if (!parallel && !binding.vision) {
        binding.vision = std::make_unique<qwen::VisionPrefillExecution>(
            device, *parameters.profile->config.vision, *parameters.vision,
            resources.workspace_storage(), *resources.plan.workspace.vision, input, vision_handoff,
            vision_handoff_peak_bytes);
    }
    const auto chunk = parallel ? prepare_parallel_vision(binding, begin, tokens)
                                : binding.vision->prepare_chunk(begin, tokens);
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
    if (!binding.vision && !binding.vision_timer) return 0;
    const auto elapsed   = binding.vision_timer ? binding.vision_timer->elapsed_ms() * 0.001
                                                : binding.vision->elapsed_seconds();
    const auto increment = elapsed - binding.observed_vision_seconds;
    binding.observed_vision_seconds = elapsed;
    if (committed_frontier >= binding.vision_input->plan.uses.back().end) retire_vision(binding);
    return increment;
}

void RankProgramImpl::retire_vision(ResidentBinding& binding) noexcept {
    binding.vision.reset();
    binding.vision_timer.reset();
    binding.feature_offsets.clear();
    binding.distributed_vision_ready = false;
    binding.observed_vision_seconds  = 0;
}

qwen::VisionChunk RankProgramImpl::prepare_parallel_vision(ResidentBinding& binding,
                                                           std::uint32_t begin,
                                                           std::uint32_t tokens) {
    const auto& input = *binding.vision_input;
    const auto& plan  = input.plan;
    auto features     = device_tensor(*resources.plan.persistent.buffers.vision_features)
                        .slice(2, binding.member.residency.lane.value, 1);
    if (!binding.distributed_vision_ready) {
        const auto owners = vision_owners(input, parameters.profile->local.size);
        binding.feature_offsets.assign(input.vision_items.size(), UINT32_MAX);
        binding.vision_timer.emplace(device);
        binding.vision_timer->start();
        qwen::VisionContext encoder(cuda, *parameters.profile->config.vision, *parameters.vision);
        std::uint32_t offset = 0;
        for (const auto& use : plan.uses) {
            if (use.end <= begin) continue;
            const auto& control = plan.control->items.at(use.control_index);
            if (control.merged_count > std::size_t(features.ne[1]) - offset)
                throw std::logic_error("Vision features exceed the reserved resident capacity");
            binding.feature_offsets.at(use.prepared_item_index) = offset;
            auto output = features.slice(1, offset, control.merged_count);
            offset += control.merged_count;
            if (owners[use.prepared_item_index] != parameters.profile->local.rank) continue;
            const auto& payload = input.media_payloads.at(use.prepared_item_index);
            if (!payload) throw std::logic_error("Vision encoder owner has no input payload");
            const auto bytes =
                encoder.encode({payload->span(), &control}, output, resources.workspace_storage(),
                               *resources.plan.workspace.vision);
            vision_workspace_peak_bytes = std::max(vision_workspace_peak_bytes, bytes);
        }
        for (const auto& use : plan.uses) {
            const auto offset = binding.feature_offsets.at(use.prepared_item_index);
            if (offset == UINT32_MAX) continue;
            auto output =
                features.slice(1, offset, plan.control->items[use.control_index].merged_count);
            parallel->broadcast(output, owners[use.prepared_item_index], cuda);
        }
        binding.vision_timer->record_stop();
        binding.distributed_vision_ready = true;
    }
    const auto use = std::find_if(plan.uses.begin(), plan.uses.end(), [&](const auto& item) {
        return item.end > begin && item.begin < std::uint64_t(begin) + tokens;
    });
    if (use == plan.uses.end()) return {static_cast<int>(tokens), nullptr, {}};
    const auto& control = plan.control->items[use->control_index];
    const auto offset   = binding.feature_offsets.at(use->prepared_item_index);
    if (offset == UINT32_MAX) throw std::logic_error("Vision replay precedes retained features");
    return {static_cast<int>(tokens), &control, features.slice(1, offset, control.merged_count), 0};
}

std::size_t RankProgramImpl::active_vision_bytes() const noexcept {
    std::size_t bytes = 0;
    for (const auto& binding : bindings)
        if (binding && binding->vision)
            bytes = std::max(bytes, binding->vision->active_handoff_bytes());
    return bytes;
}

} // namespace ninfer::models::qwen4_exp::detail
