#include "models/inkling2/program/internal.h"

#include "models/inkling2/vision/execution.h"

namespace ninfer::models::inkling2::detail {

std::optional<models::vision::EmbeddingView>
RankProgramImpl::prepare_vision(const std::optional<vision::Work>& work, std::uint32_t begin,
                                std::uint32_t tokens) {
    if (!work) return {};
    if (!resources.vision || !parameters.vision || !frame.visual_indices)
        throw std::logic_error("Inkling2 Vision work has no prepared resources");
    auto& window = *resources.vision;
    if (window.begin(work->window)) {
        if (!work->input || work->input->batch.features != work->window.plan->features)
            throw std::logic_error("Inkling2 Vision batch differs from its output window");
        const auto& batch = work->input->batch;
        auto destination  = window.output(0, batch.features);
        const auto bytes =
            vision::encode(*parameters.profile->config.vision, *parameters.vision, batch,
                           destination, resources.workspace_storage(), window.staging(), cuda);
        window.produced(bytes);
    }
    const auto& indices = *frame.visual_indices;
    return models::vision::prepare_embedding_view(window, *work->window.plan, begin, tokens,
                                                  resources.tensor(indices),
                                                  host_region(indices.region), cuda);
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

} // namespace ninfer::models::inkling2::detail
