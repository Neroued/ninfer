#include "models/glm5next/program/internal.h"

#include "models/glm5next/vision/execution.h"

#include <algorithm>

namespace ninfer::models::glm5next::detail {

std::optional<models::vision::EmbeddingView>
RankProgramImpl::prepare_vision(const std::optional<vision::Work>& work, std::uint32_t begin,
                                std::uint32_t tokens) {
    if (!work) return {};
    if (!resources.vision || !parameters.vision)
        throw std::logic_error("GLM Vision work has no prepared resources");
    auto& window = *resources.vision;
    if (window.begin(work->window)) {
        if (!work->input) throw std::logic_error("GLM Vision production has no input batch");
        const auto rank    = parameters.profile->local.rank;
        const auto& batch  = work->input->batches.at(rank);
        const auto& blocks = work->window.plan->outputs;
        const auto output  = std::find_if(blocks.begin(), blocks.end(),
                                          [&](const auto& b) { return b.rank == rank; });
        if (batch.features != (output == blocks.end() ? 0 : output->token_positions.size()))
            throw std::logic_error("GLM Vision rank batch differs from its output block");
        std::size_t bytes = 0;
        if (batch.features) {
            auto destination = window.output(output->feature_begin, batch.features);
            bytes =
                vision::encode(*parameters.profile->config.vision, *parameters.vision, batch,
                               destination, resources.workspace_storage(), window.staging(), cuda);
        }
        if (parallel)
            for (const auto& block : blocks) {
                auto destination = window.output(block.feature_begin, block.token_positions.size());
                parallel->broadcast(destination, block.rank, cuda);
            }
        window.produced(bytes);
    }
    const auto& region = *resources.plan.frame.visual_indices;
    auto indices       = host(region);
    return models::vision::prepare_embedding_view(
        window, *work->window.plan, begin, tokens, resources.tensor(region),
        {indices.data, indices.bytes(), indices.device, true}, cuda);
}

void RankProgramImpl::collect_vision_timing() {
    if (!resources.vision) return;
    const auto* forward = std::get_if<ForwardWork>(&compute->request.work.operation);
    if (!forward || !forward->visual) return;
    if (auto ns = resources.vision->take_timing(); ns && shape.gpu_timing)
        compute->gpu.add_vision(*ns);
}

void RankProgramImpl::retire_vision(runtime::ResidencyId residency) noexcept {
    if (!resources.vision) return;
    const auto* active = resources.vision->current();
    if (active && active->owner.residency == residency) resources.vision->release(active->id);
}

} // namespace ninfer::models::glm5next::detail
