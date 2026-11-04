#include "ninfer/ops/weight_input.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(message); }
}

const std::vector<std::uint64_t>& matrix(const WeightInput& input) {
    require(input.weight.shape.size() == 2, "projection weight must be a matrix");
    const auto elements = weight_element_count(input.weight.shape);
    require(!input.weight.parts.empty(), "projection weight has no parent regions");
    std::uint64_t covered = 0;
    for (const auto& part : input.weight.parts) {
        require(part.parent && part.begin < part.end &&
                    part.end <= part.parent->geometry.elements &&
                    part.end - part.begin <= elements - covered,
                "projection region exceeds its parent or logical matrix");
        covered += part.end - part.begin;
    }
    require(covered == elements, "projection regions do not cover its logical matrix");
    return input.weight.shape;
}

WeightView concatenate_rows(std::span<const WeightInput> inputs) {
    require(!inputs.empty(), "projection bank is empty");
    const auto columns = matrix(inputs.front())[1];
    WeightView out{{0, columns}, {}};
    for (const auto& input : inputs) {
        const auto& shape = matrix(input);
        require(shape[1] == columns &&
                    shape[0] <= std::numeric_limits<std::uint64_t>::max() - out.shape[0],
                "projection bank has inconsistent columns or overflowing rows");
        out.shape[0] += shape[0];
        out.parts.insert(out.parts.end(), input.weight.parts.begin(), input.weight.parts.end());
    }
    return out;
}

bool contiguous(const WeightView& view) {
    if (view.parts.empty()) { return false; }
    const auto* parent = view.parts.front().parent;
    auto end           = view.parts.front().begin;
    for (const auto& part : view.parts) {
        if (!parent || part.parent != parent || part.begin != end || part.end <= part.begin) {
            return false;
        }
        end = part.end;
    }
    return true;
}

LinearPolicy common_policy(std::span<const WeightInput> inputs) {
    auto result = LinearPolicy::AllowA4;
    for (const auto& input : inputs) {
        switch (input.policy) {
        case LinearPolicy::A16Only:
            result = LinearPolicy::A16Only;
            break;
        case LinearPolicy::AllowA8:
            if (result == LinearPolicy::AllowA4) { result = LinearPolicy::AllowA8; }
            break;
        case LinearPolicy::AllowA4:
            break;
        default:
            throw std::invalid_argument("invalid projection activation policy");
        }
    }
    return result;
}

SingleProjectionWeight single(const WeightView& view, std::span<const WeightInput> inputs) {
    const auto region = contiguous_weight_region(view);
    const auto policy = common_policy(inputs);
    float input_scale = 0;
    const auto format = region.parent->geometry.format;
    if (format == QType::NVFP4 || format == QType::NVFP4_MUL) {
        const auto operation = format == QType::NVFP4 ? ActivationScaleOperation::Divide
                                                      : ActivationScaleOperation::Multiply;
        for (const auto& input : inputs) {
            const auto values = input.calibration.values();
            if (values.empty()) {
                require(!allows_a4(policy), "NVFP4 A4 input requires activation calibration");
                continue;
            }
            require(
                input.calibration.operation == operation && values.size() == 1 &&
                    std::isfinite(values.front()) && values.front() > 0,
                "NVFP4 matrix input requires a positive scalar calibration in its scale domain");
            if (input_scale == 0) {
                input_scale = values.front();
            } else if (allows_a4(policy)) {
                require(std::bit_cast<std::uint32_t>(input_scale) ==
                            std::bit_cast<std::uint32_t>(values.front()),
                        "shared NVFP4 A4 input requires identical calibration");
            }
        }
        if (input_scale == 0) { input_scale = 1.0F; }
    }
    return {native_weight(view, input_scale), policy};
}

SingleProjectionWeight single(std::span<const WeightInput> inputs) {
    return single(concatenate_rows(inputs), inputs);
}

LinearProfile describe_single(const WeightView& view, std::span<const WeightInput> inputs) {
    const auto region = contiguous_weight_region(view);
    require(view.shape.size() == 2 && view.shape[0] <= INT32_MAX && view.shape[1] <= INT32_MAX,
            "projection dimensions exceed int32");
    const auto& geometry = region.parent->geometry;
    const auto layout    = geometry.layout == QuantLayout::BatchedBlockScaleK16M128x4
                               ? QuantLayout::BlockScaleK16M128x4
                               : geometry.layout;
    return {geometry.format, static_cast<std::int32_t>(view.shape[0]),
            static_cast<std::int32_t>(view.shape[1]), common_policy(inputs), layout};
}

template <class Make>
auto projection_regions(std::span<const WeightInput> rows, Make make) {
    const auto view = concatenate_rows(rows);
    if (contiguous(view)) return std::vector{make(view, rows)};

    const auto columns = view.shape[1];
    WeightView bank{{0, columns}, {}};
    std::uint64_t elements = 0;
    std::vector<WeightInput> bank_rows;
    using Result = decltype(make(view, rows));
    std::vector<Result> banks;
    const auto finish_bank = [&] {
        require(elements % columns == 0, "projection parent boundary must align to a matrix row");
        bank.shape[0] = elements / columns;
        banks.push_back(make(bank, bank_rows));
    };
    for (const auto& input : rows) {
        bool included = false;
        for (const auto& part : input.weight.parts) {
            if (!bank.parts.empty() &&
                (part.parent != bank.parts.back().parent || part.begin != bank.parts.back().end)) {
                finish_bank();
                require(banks.size() < 2, "projection bank requires more than two native regions");
                bank.parts.clear();
                bank_rows.clear();
                elements = 0;
                included = false;
            }
            bank.parts.push_back(part);
            elements += part.end - part.begin;
            if (!included) {
                bank_rows.push_back(input);
                included = true;
            }
        }
    }
    finish_bank();
    return banks;
}

} // namespace

SingleProjectionWeight prepare_linear_weight(const WeightInput& input) {
    return single({&input, 1});
}

SingleProjectionWeight prepare_linear_weight(std::span<const WeightInput> rows) {
    return single(rows);
}

LinearProfile describe_linear_weight(std::span<const WeightInput> rows) {
    return describe_single(concatenate_rows(rows), rows);
}

ProjectionProfile describe_projection_weights(std::span<const WeightInput> rows) {
    auto banks = projection_regions(rows, describe_single);
    if (banks.size() == 1) return {banks[0], {}};
    return {banks[0], banks[1]};
}

ProjectionWeights prepare_projection_weights(std::span<const WeightInput> rows) {
    const auto banks =
        projection_regions(rows, [](const WeightView& view, std::span<const WeightInput> inputs) {
            return single(view, inputs);
        });
    if (banks.size() == 1) return banks[0];
    return PairedProjectionWeights{banks[0], banks[1]};
}

ProjectionWeights prepare_gdn_control_proj_weights(const WeightInput& a, const WeightInput& b) {
    const auto& shape = matrix(a);
    require(shape == matrix(b) && (shape == std::vector<std::uint64_t>{48, 5120} ||
                                   shape == std::vector<std::uint64_t>{32, 2048} ||
                                   shape == std::vector<std::uint64_t>{48, 2560} ||
                                   shape == std::vector<std::uint64_t>{24, 2560}),
            "GDN control: unsupported A/B geometry");
    const std::array inputs{a, b};
    if (contiguous(concatenate_rows(inputs))) {
        auto result = single(inputs);
        require(result.weight.qtype == QType::BF16, "GDN control requires BF16 weights");
        return result;
    }
    require(shape == std::vector<std::uint64_t>{48, 5120},
            "GDN control: this geometry requires a combined parent");
    const auto first  = prepare_linear_weight(a);
    const auto second = prepare_linear_weight(b);
    require(first.weight.qtype == QType::BF16 && second.weight.qtype == QType::BF16,
            "GDN control requires BF16 weights");
    return PairedProjectionWeights{first, second};
}

SingleProjectionWeight prepare_linear_swiglu_weight(const WeightInput& gate,
                                                    const WeightInput& up) {
    require(matrix(gate) == matrix(up), "SwiGLU gate and up geometry differs");
    const std::array inputs{gate, up};
    return single(inputs);
}

} // namespace ninfer::ops
