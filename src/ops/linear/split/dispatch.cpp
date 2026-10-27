#include "ops/linear/split/dispatch.h"
#include "core/platform.h"

#include <algorithm>
#include <ranges>
#include <stdexcept>

namespace ninfer::ops::detail::linear_split {
namespace {
const SplitShape& match(std::span<const SplitShape* const> shapes, ProjectionProfile weights,
                        const auto& rows) {
    for (const auto* shape : shapes)
        if (shape->first_rows == weights.first.n && shape->input_rows == weights.first.k &&
            shape->second_rows == (weights.second ? weights.second->n : 0) &&
            (!weights.second || weights.second->k == shape->input_rows) &&
            std::ranges::equal(shape->segments, rows))
            return *shape;
    throw std::invalid_argument("linear_split: unsupported parent geometry or output segmentation");
}

const SplitShape& select(ProjectionProfile weights, const auto& rows) {
    if (!valid_linear_policy(weights.first.policy))
        throw std::invalid_argument("linear_split: invalid activation permission");
    if (weights.second) {
        if (weights.first.qtype != QType::Q4_G64_FP16 ||
            weights.second->qtype != QType::Q5_G64_FP16 ||
            weights.first.policy != LinearPolicy::A16Only)
            throw std::invalid_argument("linear_split: paired weights require Q4/Q5 A16");
        const std::array shapes{&kQ4Q5SplitN7168N7168K5120, &kQ4Q5SplitN4096N12288K5120};
        return match(shapes, weights, rows);
    }
    switch (weights.first.qtype) {
    case QType::BF16: {
        const std::array shapes{
#if NINFER_TARGET_SPARK
            &kBf16SplitN20480K1536,
            &kBf16SplitN288K4096,
            &kBf16SplitN1152K2560,
            &kBf16SplitN7296K2560,
            &kBf16SplitN640K2560,
#endif
            &kBf16SplitN1152K2560ThreeOutput,
            &kBf16SplitN1664K2560,
            &kBf16SplitN13952K2560,
            &kBf16SplitN14336K5120};
        return match(shapes, weights, rows);
    }
    case QType::Q8_G32_FP16: {
        const std::array shapes{
#if NINFER_TARGET_SPARK
            &kQ8SplitN8192K2560,  &kQ8SplitN6144K2560,  &kQ8SplitN12288K4096,
#endif
            &kQ8SplitN12288K2560, &kQ8SplitN16384K2560, &kQ8SplitN9216K2048,
            &kQ8SplitN6144K2048,  &kQ8SplitN6144K5120,  &kQ8SplitN12288K2048};
        return match(shapes, weights, rows);
    }
    case QType::FP8_E4M3FN_ROW_BF16: {
        const std::array shapes{
#if NINFER_TARGET_RTX5090
            &kFp8SplitN14976K5120, &kFp8SplitN2176K5120,
#endif
            &kFp8SplitN14336K5120, &kFp8SplitN16384K5120};
        return match(shapes, weights, rows);
    }
    case QType::NVFP4: {
        const std::array shapes{&kNvfp4SplitN14336K5120, &kNvfp4SplitN16384K5120};
        return match(shapes, weights, rows);
    }
    default:
        throw std::invalid_argument("linear_split: unsupported weight format");
    }
}
} // namespace

const SplitShape& select_split_shape(ProjectionProfile weights,
                                     std::span<const std::int32_t> output_rows) {
    return select(weights, output_rows);
}

const SplitShape& select_split_shape(ProjectionProfile weights, std::span<const Tensor> outputs) {
    return select(weights,
                  outputs | std::views::transform([](const Tensor& t) { return t.ne[0]; }));
}
} // namespace ninfer::ops::detail::linear_split
