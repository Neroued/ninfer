#pragma once

#include "core/weight_view.h"
#include "ninfer/ops/linear.h"

#include <optional>
#include <span>
#include <variant>

namespace ninfer::ops {

// One mathematical use of a logical matrix. The model owns the view and its backing.
struct WeightInput {
    const WeightView& weight;
    LinearPolicy policy = LinearPolicy::A16Only;
    std::optional<float> activation_input_divisor;
};

struct SingleProjectionWeight {
    Weight weight;
    LinearPolicy policy = LinearPolicy::A16Only;
};

struct PairedProjectionWeights {
    Weight first, second;
};

using ProjectionWeights = std::variant<SingleProjectionWeight, PairedProjectionWeights>;

// Prepare the existing native forms; no device allocation, upload, execution or graph rewrite.
// Runtime shape/phase choices and scratch remain with the actual calling Op.
[[nodiscard]] SingleProjectionWeight prepare_linear_weight(const WeightInput& input);
// Row order is supplied by the calling implementation, for example a Vision Q/K/V bank.
[[nodiscard]] SingleProjectionWeight prepare_linear_weight(std::span<const WeightInput> rows);
[[nodiscard]] SingleProjectionWeight prepare_attn_input_proj_weights(const WeightInput& query,
                                                                     const WeightInput& key,
                                                                     const WeightInput& value);
[[nodiscard]] ProjectionWeights prepare_attn_input_proj_weights(const WeightInput& query,
                                                                const WeightInput& key,
                                                                const WeightInput& gate,
                                                                const WeightInput& value);
[[nodiscard]] ProjectionWeights prepare_gdn_input_proj_weights(const WeightInput& query,
                                                               const WeightInput& key,
                                                               const WeightInput& value,
                                                               const WeightInput& z);
[[nodiscard]] ProjectionWeights prepare_gdn_gating_proj_weights(const WeightInput& a,
                                                                const WeightInput& b);
[[nodiscard]] SingleProjectionWeight prepare_linear_swiglu_weight(const WeightInput& gate,
                                                                  const WeightInput& up);

} // namespace ninfer::ops
