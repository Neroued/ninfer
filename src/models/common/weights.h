#pragma once

#include "core/weight_view.h"
#include "ninfer/ops/weight_input.h"

#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninfer::models {

struct WeightId {
    std::size_t index                          = std::numeric_limits<std::size_t>::max();
    friend bool operator==(WeightId, WeightId) = default;
};

struct WeightUseId {
    WeightId parameter;
    std::size_t use_index = std::numeric_limits<std::size_t>::max();
};

struct WeightUse {
    std::string input;
    ops::LinearPolicy policy = ops::LinearPolicy::A16Only;
    std::optional<float> activation_input_divisor;
};

struct BoundWeight {
    std::string name;
    std::vector<std::string> source_objects;
    WeightView view;
    std::vector<WeightUse> uses;
};

[[nodiscard]] ops::WeightInput weight_input(std::span<const BoundWeight>, WeightUseId);
[[nodiscard]] ops::WeightInput weight_input(std::span<const BoundWeight>, WeightId);
[[nodiscard]] Tensor parameter_tensor(const BoundWeight&);

} // namespace ninfer::models
