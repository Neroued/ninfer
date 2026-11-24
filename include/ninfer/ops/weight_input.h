#pragma once

#include "core/weight_view.h"
#include "ninfer/ops/projection_weights.h"

#include <optional>
#include <span>

namespace ninfer::ops {

// One mathematical use of a logical matrix. The model owns the view and its backing.
struct WeightInput {
    const WeightView& weight;
    LinearPolicy policy = LinearPolicy::A16Only;
    std::optional<float> activation_input_divisor;
};

[[nodiscard]] LinearProfile describe_linear_weight(std::span<const WeightInput> rows);
[[nodiscard]] ProjectionProfile describe_projection_weights(std::span<const WeightInput> rows);

// Prepare the existing native forms; no device allocation, upload, execution or graph rewrite.
// Runtime shape/phase choices and scratch remain with the actual calling Op.
[[nodiscard]] SingleProjectionWeight prepare_linear_weight(const WeightInput& input);
// Row order is supplied by the calling implementation, for example a Vision Q/K/V bank.
[[nodiscard]] SingleProjectionWeight prepare_linear_weight(std::span<const WeightInput> rows);
// Logical row concatenation backed by one or two consecutive native parent regions.
// Paired regions use A16; the consuming Op owns format/shape/segmentation admission.
[[nodiscard]] ProjectionWeights prepare_projection_weights(std::span<const WeightInput> rows);
[[nodiscard]] ProjectionWeights prepare_gdn_control_proj_weights(const WeightInput& a,
                                                                 const WeightInput& b);
[[nodiscard]] SingleProjectionWeight prepare_linear_swiglu_weight(const WeightInput& gate,
                                                                  const WeightInput& up);

} // namespace ninfer::ops
