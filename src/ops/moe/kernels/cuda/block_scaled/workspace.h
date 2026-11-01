#pragma once

#include "ops/moe/kernels/cuda/block_scaled/launch.h"

namespace ninfer::ops::detail::moe_cuda::block_scaled {

template <class Arena>
Storage allocate(Arena& arena, const Plan& plan, int tokens, bool routed, int shared_intermediate) {
    Storage out;
    if (routed && plan.vector) {
        out.vector_activation =
            static_cast<float*>(arena.alloc(DType::FP32, {plan.intermediate, kTopK * tokens}).data);
        if (shared_intermediate) {
            out.routed_sum = static_cast<float*>(arena.alloc(DType::FP32, {kHidden, tokens}).data);
        }
    } else if (routed) {
        const int assignments = tokens * kTopK;
        const auto bytes      = [&](std::size_t size) {
            return static_cast<std::uint8_t*>(arena.alloc_bytes(size).data);
        };
        const auto integers = [&](int count) {
            return static_cast<int*>(arena.alloc(DType::I32, {count}).data);
        };
        const bool fp8 = plan.format == QType::FP8_E4M3FN_BLOCK128X128_BF16 ||
                         plan.format == QType::FP8_E4M3FN_BLOCK64X64_BF16;
        const int packing = fp8 ? 1 : 2, scale_group = fp8 ? 128 : 16;
        const auto scale_bytes = fp8 ? sizeof(float) : sizeof(std::uint8_t);
        out.input_codes        = bytes(std::size_t(tokens) * kHidden / packing);
        out.input_scales       = bytes(std::size_t(tokens) * (kHidden / scale_group) * scale_bytes);
        out.product_codes      = bytes(std::size_t(assignments) * plan.intermediate / packing);
        const int product_scale_stride = fp8 ? (plan.intermediate + 127) / 128
                                             : nvfp4_activation_scale_stride(plan.intermediate);
        out.product_scales = bytes(std::size_t(assignments) * product_scale_stride * scale_bytes);
        if (plan.atomic_output) {
            out.routed_sum = static_cast<float*>(arena.alloc(DType::FP32, {kHidden, tokens}).data);
        } else {
            out.contributions =
                static_cast<float*>(arena.alloc(DType::FP32, {kHidden, assignments}).data);
        }
        auto& g = out.groups;
        if (!plan.single_block_grouping) { g.counts = integers(kExperts); }
        g.offsets = integers(kExperts + 1);
        if (!plan.single_block_grouping) { g.cursors = integers(kExperts); }
        const int max_jobs = assignments / plan.job_columns() + kExperts;
        g.job_experts      = integers(max_jobs);
        g.job_columns      = integers(max_jobs);
        g.jobs             = integers(1);
        if (plan.dense.gate_columns || plan.fp8.maximum_small_load || plan.pool_expert_limit)
            g.load_range = integers(3);
        g.assignments = integers(assignments);
        if (!plan.atomic_output || plan.pool_expert_limit) { g.inverse = integers(assignments); }
    }
    if (shared_intermediate) {
        const bool vector     = plan.shared_algorithm == SharedAlgorithm::Vector;
        const auto type       = vector ? DType::FP32 : DType::BF16;
        const int planes      = vector || !plan.shared_split_activation ? 1 : 2;
        out.shared_activation = arena.alloc(type, {shared_intermediate, tokens, planes}).data;
    }
    return out;
}

} // namespace ninfer::ops::detail::moe_cuda::block_scaled
