#include "models/qwen4_exp/program/internal.h"
#include "core/transfer.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp::detail {

std::optional<models::vision::EmbeddingView>
RankProgramImpl::visual_view(const vision::WindowPlan& plan, std::uint32_t begin,
                             std::uint32_t tokens, const TensorRegion& region, MemorySpan host) {
    return vision::prepare_embedding_view(*resources.vision, plan, begin, tokens,
                                          device_tensor(region), host, cuda);
}

std::optional<models::vision::EmbeddingView>
RankProgramImpl::prepare_vision(const std::optional<qwen::VisionWork>& work, std::uint32_t begin,
                                std::uint32_t tokens) {
    if (!work) return {};
    if (!resources.vision || !parameters.vision)
        throw std::logic_error("Vision work has no prepared resources");
    auto& window = *resources.vision;
    if (window.begin(work->window)) {
        if (!work->input) throw std::logic_error("Vision production has no input batch");
        const auto rank   = parameters.profile->local.rank;
        const auto& batch = work->input->batches.at(rank);
        std::size_t bytes = 0;
        if (batch.features) {
            const auto output =
                std::find_if(work->window.plan->outputs.begin(), work->window.plan->outputs.end(),
                             [&](const auto& b) { return b.rank == rank; });
            if (output == work->window.plan->outputs.end())
                throw std::logic_error("Vision rank batch has no output block");
            auto destination = window.output(output->feature_begin, batch.features);
            qwen::VisionContext encoder(cuda, *parameters.profile->config.vision,
                                        *parameters.vision);
            bytes =
                encoder.encode(batch, destination, resources.workspace_storage(), window.staging());
        }
        if (parallel)
            for (const auto& block : work->window.plan->outputs) {
                auto destination = window.output(block.feature_begin, block.token_positions.size());
                parallel->broadcast(destination, block.rank, cuda);
            }
        window.produced(bytes);
    }
    const auto& region = *resources.plan.persistent.buffers.batch.visual_indices;
    const auto host    = host_tensor(region);
    return visual_view(*work->window.plan, begin, tokens, region,
                       {host.data, host.bytes(), host.device, true});
}

std::optional<std::uint64_t>
RankProgramImpl::collect_vision_timing(const ResidentBinding& binding) {
    if (!resources.vision) return {};
    auto& window       = *resources.vision;
    const auto* active = window.current();
    if (!active || active->owner.sequence != binding.member.sequence ||
        active->owner.residency != binding.member.residency)
        return {};
    return window.take_timing();
}

void RankProgramImpl::retire_vision(ResidentBinding& binding) noexcept {
    if (!resources.vision) return;
    const auto* active = resources.vision->current();
    if (active && active->owner.residency == binding.member.residency)
        resources.vision->release(active->id);
}

} // namespace ninfer::models::qwen4_exp::detail
