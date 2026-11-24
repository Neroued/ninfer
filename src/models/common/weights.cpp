#include "models/common/weights.h"

#include <array>
#include <stdexcept>

namespace ninfer::models {

ops::WeightInput weight_input(std::span<const BoundWeight> weights, WeightUseId id) {
    if (id.parameter.index >= weights.size()) throw std::out_of_range("weight parameter index");
    const auto& parameter = weights[id.parameter.index];
    const auto& use       = parameter.uses.at(id.use_index);
    return {parameter.view, use.policy, use.activation_input_divisor};
}

ops::WeightInput weight_input(std::span<const BoundWeight> weights, WeightId id) {
    if (id.index >= weights.size()) throw std::out_of_range("weight parameter index");
    if (weights[id.index].uses.size() != 1)
        throw std::invalid_argument("weight input requires an explicit mathematical use");
    return weight_input(weights, WeightUseId{id, 0});
}

Tensor parameter_tensor(const BoundWeight& parameter) {
    const auto& view = parameter.view;
    if (view.shape.size() > 4) throw std::invalid_argument("direct parameter exceeds Tensor rank");
    std::array<std::int32_t, 4> axes{1, 1, 1, 1};
    for (std::size_t i = 0; i < view.shape.size(); ++i) {
        const auto extent = view.shape[view.shape.size() - 1 - i];
        if (extent > std::uint64_t(std::numeric_limits<std::int32_t>::max()))
            throw std::invalid_argument("direct parameter exceeds Tensor extent");
        axes[i] = static_cast<std::int32_t>(extent);
    }
    return weight_tensor(view, {axes[0], axes[1], axes[2], axes[3]});
}

} // namespace ninfer::models
