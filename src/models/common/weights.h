#pragma once

#include "core/weight_view.h"
#include "ninfer/ops/weight_input.h"

#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ninfer::artifact {
struct Binding;
}

namespace ninfer::models {

struct WeightId {
    std::size_t index                          = std::numeric_limits<std::size_t>::max();
    friend bool operator==(WeightId, WeightId) = default;
};

struct DenseWeights {
    WeightId gate, up, down;
};

struct WeightUseId {
    WeightId parameter;
    std::size_t use_index = std::numeric_limits<std::size_t>::max();
};

struct ActivationCalibrationData {
    ops::ActivationScaleOperation operation = ops::ActivationScaleOperation::Divide;
    std::vector<float> values;
};

struct WeightUse {
    std::string input;
    ops::LinearPolicy policy = ops::LinearPolicy::A16Only;
    std::shared_ptr<const ActivationCalibrationData> calibration;
    std::shared_ptr<const artifact::Binding> calibration_source;
    ConstMemorySpan calibration_device;

    [[nodiscard]] ops::ActivationCalibrationView calibration_view() const {
        if (!calibration) return {};
        if (calibration->values.size() == 1)
            return ops::ActivationCalibrationView::broadcast(calibration->operation,
                                                             calibration->values.front());
        return ops::ActivationCalibrationView::per_matrix(calibration->operation,
                                                          calibration->values, calibration_device);
    }
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
