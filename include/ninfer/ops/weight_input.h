#pragma once

#include "core/weight_view.h"
#include "ninfer/ops/projection_weights.h"

#include <cmath>
#include <span>
#include <stdexcept>
#include <variant>

namespace ninfer::ops {

enum class ActivationScaleOperation { Divide, Multiply };

// A scalar broadcasts to all logical matrices. A vector is indexed by logical matrix ID.
// Values are positive and finite. Vector storage and its matching optional CUDA copy are
// immutable and owned by the caller; execution does not rescan vector contents.
class ActivationCalibrationView {
public:
    ActivationScaleOperation operation = ActivationScaleOperation::Divide;
    ConstMemorySpan device_values;

    [[nodiscard]] static ActivationCalibrationView broadcast(ActivationScaleOperation operation,
                                                             float value) {
        ActivationCalibrationView result;
        result.operation = operation;
        result.values_   = value;
        return result;
    }

    [[nodiscard]] static ActivationCalibrationView per_matrix(ActivationScaleOperation operation,
                                                              std::span<const float> values,
                                                              ConstMemorySpan device_values = {}) {
        ActivationCalibrationView result;
        result.operation     = operation;
        result.values_       = values;
        result.device_values = device_values;
        return result;
    }

    [[nodiscard]] std::span<const float> values() const noexcept {
        if (const auto* value = std::get_if<float>(&values_)) return {value, 1};
        if (const auto* values = std::get_if<std::span<const float>>(&values_)) return *values;
        return {};
    }

    [[nodiscard]] bool empty() const noexcept { return values().empty(); }

    [[nodiscard]] float scalar_divisor() const {
        const auto data = values();
        if (operation != ActivationScaleOperation::Divide || data.size() != 1 ||
            !std::isfinite(data.front()) || data.front() <= 0)
            throw std::invalid_argument(
                "activation calibration requires one positive finite divisor");
        return data.front();
    }

private:
    std::variant<std::monostate, float, std::span<const float>> values_;
};

// One mathematical use of a logical matrix. The model owns the view and its backing.
struct WeightInput {
    const WeightView& weight;
    LinearPolicy policy = LinearPolicy::A16Only;
    ActivationCalibrationView calibration;
};

[[nodiscard]] LinearProfile describe_linear_weight(std::span<const WeightInput> rows);
[[nodiscard]] ProjectionProfile describe_projection_weights(std::span<const WeightInput> rows);

// Prepare the existing native forms; no device allocation, upload, execution or graph rewrite.
// Runtime shape/phase choices and scratch remain with the actual calling Op.
[[nodiscard]] SingleProjectionWeight prepare_linear_weight(const WeightInput& input);
// Row order is supplied by the calling implementation, for example a Vision Q/K/V bank.
[[nodiscard]] SingleProjectionWeight prepare_linear_weight(std::span<const WeightInput> rows);
// Logical row concatenation backed by one or two consecutive native parent regions.
// Each region preserves its own activation permission; the Op admits format/shape/segmentation.
[[nodiscard]] ProjectionWeights prepare_projection_weights(std::span<const WeightInput> rows);
[[nodiscard]] ProjectionWeights prepare_gdn_control_proj_weights(const WeightInput& a,
                                                                 const WeightInput& b);
[[nodiscard]] SingleProjectionWeight prepare_linear_swiglu_weight(const WeightInput& gate,
                                                                  const WeightInput& up);

} // namespace ninfer::ops
