#include "models/common/bindings.h"

#include "artifact/views.h"

#include <bit>
#include <cmath>

namespace ninfer::models {

WeightUseId Bindings::use(WeightId id, std::string_view input) const {
    const auto& parameter = at(id);
    for (std::size_t i = 0; i < parameter.uses.size(); ++i) {
        if (parameter.uses[i].input == input) { return {id, i}; }
    }
    throw artifact::ArtifactError(parameter.reference.name + ": unresolved Use " +
                                  std::string(input));
}

WeightId Bindings::parameter(std::string name, artifact::Shape shape,
                             std::vector<std::string> inputs, std::optional<QType> exact_format) {
    if (parameters_.contains(name)) {
        throw artifact::ArtifactError(name + ": duplicate model parameter declaration");
    }
    PendingWeight pending;
    pending.reference = binder.parameter(name, std::move(shape), exact_format);
    for (const auto& input : inputs) {
        const auto& use = binder.use(name, input);
        if (!use.activation_policy) {
            throw artifact::ArtifactError(name + "@" + input + ": missing activation policy");
        }
        WeightUse result;
        result.input = input;
        switch (*use.activation_policy) {
        case artifact::ActivationPolicy::A16Only:
            result.policy = ops::LinearPolicy::A16Only;
            break;
        case artifact::ActivationPolicy::AllowA8:
            result.policy = ops::LinearPolicy::AllowA8;
            break;
        case artifact::ActivationPolicy::AllowA4:
            result.policy = ops::LinearPolicy::AllowA4;
            break;
        }
        for (const auto& [role, binding] : use.auxiliaries) {
            if (role != "activation_input_divisor" && role != "activation_input_multiplier") {
                throw artifact::ArtifactError(name + "@" + input + ": unknown auxiliary " + role);
            }
            if (result.calibration)
                throw artifact::ArtifactError(name + "@" + input +
                                              ": activation calibration is specified twice");
            const auto words    = binder.values(binding, QType::FP32);
            const auto matrices = pending.reference.shape.size() == 3
                                      ? pending.reference.shape.front()
                                      : std::uint64_t{1};
            if (words.elements != 1 && words.elements != matrices)
                throw artifact::ArtifactError(
                    name + "@" + input +
                    ": activation calibration must broadcast or cover every matrix");
            auto calibration       = std::make_shared<ActivationCalibrationData>();
            calibration->operation = role == "activation_input_multiplier"
                                         ? ops::ActivationScaleOperation::Multiply
                                         : ops::ActivationScaleOperation::Divide;
            calibration->values.reserve(words.elements);
            for (std::size_t i = 0; i < words.elements; ++i) {
                const auto value = std::bit_cast<float>(
                    artifact::read_u32_le(words.data.data() + i * sizeof(float)));
                if (!std::isfinite(value) || value <= 0)
                    throw artifact::ArtifactError(
                        name + "@" + input + ": activation scale must be positive finite FP32");
                calibration->values.push_back(value);
            }
            result.calibration        = std::move(calibration);
            result.calibration_source = std::make_shared<const artifact::Binding>(binding);
        }
        pending.uses.push_back(std::move(result));
    }
    for (const auto& part : pending.reference.binding.parts) {
        pending.source_objects.push_back(
            artifact::object_id(binder.reader().directory().object(part.object)));
    }
    const WeightId id{weights.size()};
    parameters_.emplace(std::move(name), id);
    weights.push_back(std::move(pending));
    return id;
}

WeightId Bindings::direct(std::string name, artifact::Shape shape, std::optional<QType> format) {
    if (!format) {
        const auto reference = binder.parameter(name, shape);
        for (const auto& part : reference.binding.parts) {
            const auto actual = binder.reader().geometry(part.object).format;
            if (actual != QType::BF16 && actual != QType::FP32)
                throw artifact::ArtifactError(name + ": direct parameter requires BF16 or FP32");
        }
    }
    return parameter(std::move(name), std::move(shape), {}, format);
}

std::vector<BoundWeight> resolve_weights(std::vector<PendingWeight>&& pending,
                                         const artifact::MaterializedArtifact& materialized,
                                         std::span<const Device> devices) {
    std::vector<BoundWeight> out;
    out.reserve(pending.size());
    for (std::size_t i = 0; i < pending.size(); ++i) {
        auto& item = pending[i];
        auto view  = artifact::bind_view(item.reference, materialized, devices[i]);
        out.push_back({std::move(item.reference.name), std::move(item.source_objects),
                       std::move(view), std::move(item.uses)});
    }
    return out;
}

} // namespace ninfer::models
