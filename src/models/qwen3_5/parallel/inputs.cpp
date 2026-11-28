#include "models/qwen3_5/parallel/codec.h"

#include <stdexcept>

namespace ninfer::models::qwen3_5 {
namespace {
namespace wire = distributed::wire;

const qwen::VisionWork* find_produced_visual(const Work& work) {
    const auto* forward = std::get_if<ForwardWork>(&work.operation);
    if (!forward) return nullptr;
    for (const auto& window : forward->inputs) {
        if (window.visual && window.visual->window.produce) return &*window.visual;
    }
    return nullptr;
}

} // namespace

bool ExecutionCodec::has_inputs(const Work& work) const noexcept {
    return find_produced_visual(work) != nullptr;
}

std::vector<runtime::InputResource>
ExecutionCodec::prepare_inputs(const Work& work, const distributed::ReplicaTopology& replica) {
    const auto* visual = find_produced_visual(work);
    if (!visual) return {};
    auto resources = qwen::stream_vision_inputs(*visual, replica.ranks);
    inputs_.install(*visual);
    return resources;
}

void ExecutionCodec::retire_inputs(const Work& work) {
    if (const auto* visual = find_produced_visual(work)) inputs_.release_payload(visual->window.id);
}

std::optional<std::uint64_t> ExecutionCodec::receive_input(wire::Kind kind,
                                                           std::span<const std::byte> bytes) {
    return receiver_.receive(kind, bytes, contract_.vision_max_features, 0, 1, inputs_);
}

} // namespace ninfer::models::qwen3_5
