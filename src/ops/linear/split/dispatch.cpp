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
        if (!valid_linear_policy(weights.second->policy))
            throw std::invalid_argument(
                "linear_split: invalid second-parent activation permission");
        if (weights.first.policy != LinearPolicy::A16Only ||
            weights.second->policy != LinearPolicy::A16Only)
            throw std::invalid_argument("linear_split: registered paired weights require A16");
        if (weights.first.qtype == QType::Q4_G64_FP16 &&
            weights.second->qtype == QType::Q5_G64_FP16) {
            const std::array shapes{&kQ4Q5SplitN7168N7168K5120, &kQ4Q5SplitN4096N12288K5120};
            return match(shapes, weights, rows);
        }
#if NINFER_TARGET_SPARK
        if (weights.first.qtype == QType::FP8_E4M3FN_BLOCK128X128_FP32 &&
            weights.second->qtype == QType::BF16) {
            const std::array shapes{&kFp8BlockBf16SplitN16384N4096K1536};
            return match(shapes, weights, rows);
        }
#endif
        throw std::invalid_argument("linear_split: unsupported paired weight formats");
    }
    switch (weights.first.qtype) {
    case QType::BF16: {
        const std::array shapes{
#if NINFER_TARGET_SPARK
            &kBf16SplitN3072K4096,
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
        const std::array shapes{&kFp8SplitN14336K5120, &kFp8SplitN16384K5120};
        return match(shapes, weights, rows);
    }
#if NINFER_TARGET_SPARK
    case QType::MXFP8: {
        const std::array shapes{&kMxfp8SplitN20480K1536};
        return match(shapes, weights, rows);
    }
#endif
    case QType::NVFP4: {
        const std::array shapes{&kNvfp4SplitN14336K5120, &kNvfp4SplitN16384K5120};
        return match(shapes, weights, rows);
    }
    default:
        throw std::invalid_argument("linear_split: unsupported weight format");
    }
}

const SplitBiasShape& select_bias(QType type, int n, int k, LinearPolicy policy, const auto& rows) {
    if (!valid_linear_policy(policy))
        throw std::invalid_argument("linear_split bias: invalid activation permission");
#if NINFER_TARGET_SPARK
    const auto* shape = type == QType::BF16    ? &kBf16SplitBiasN3072K1024
                        : type == QType::MXFP8 ? &kMxfp8SplitBiasN3072K1024
                                               : nullptr;
    if (shape && shape->rows == n && shape->input_rows == k &&
        std::ranges::equal(shape->segments, rows))
        return *shape;
#endif
    throw std::invalid_argument("linear_split bias: unsupported format, geometry or segmentation");
}
} // namespace

const SplitBiasShape& select_split_bias_shape(QType type, int n, int k, LinearPolicy policy,
                                              std::span<const std::int32_t> rows) {
    return select_bias(type, n, k, policy, rows);
}

const SplitBiasShape& select_split_bias_shape(QType type, int n, int k, LinearPolicy policy,
                                              std::span<const Tensor> outputs) {
    return select_bias(type, n, k, policy,
                       outputs | std::views::transform([](const Tensor& t) { return t.ne[0]; }));
}

const SplitShape& select_split_shape(ProjectionProfile weights,
                                     std::span<const std::int32_t> output_rows) {
    return select(weights, output_rows);
}

const SplitShape& select_split_shape(ProjectionProfile weights, std::span<const Tensor> outputs) {
    return select(weights,
                  outputs | std::views::transform([](const Tensor& t) { return t.ne[0]; }));
}
} // namespace ninfer::ops::detail::linear_split
