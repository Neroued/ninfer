#include "models/qwen4_exp/parallel/codec.h"

namespace ninfer::models::qwen4_exp {
namespace {
namespace wire = distributed::wire;

const qwen::VisionWork* find_produced_visual(const Work& work) {
    const auto* forward = std::get_if<ForwardWork>(&work.operation);
    if (!forward || !forward->visual || !forward->visual->window.produce) return nullptr;
    return &*forward->visual;
}

} // namespace

bool ExecutionCodec::has_inputs(const Work& work) const noexcept {
    return find_produced_visual(work) != nullptr;
}

std::vector<runtime::InputResource>
ExecutionCodec::prepare_inputs(const Work& work, const distributed::ReplicaTopology& replica) {
    const auto* visual = find_produced_visual(work);
    if (!visual) return {};
    auto resources = qwen::stream_vision_inputs(*visual, replica.stage_ranks.front());
    inputs_.install(*visual);
    return resources;
}

void ExecutionCodec::retire_inputs(const Work& work) {
    if (const auto* visual = find_produced_visual(work)) inputs_.release_payload(visual->window.id);
}

std::optional<std::uint64_t> ExecutionCodec::receive_input(wire::Kind kind,
                                                           std::span<const std::byte> bytes) {
    return receiver_.receive(kind, bytes, contract_.vision ? max_features_ : 0, rank_,
                             static_cast<std::uint32_t>(replica_.stage_ranks.front().size()),
                             inputs_);
}

} // namespace ninfer::models::qwen4_exp
