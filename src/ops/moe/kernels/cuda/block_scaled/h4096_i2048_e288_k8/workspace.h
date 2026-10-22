#pragma once

#include "ops/moe/kernels/cuda/block_scaled/h4096_i2048_e288_k8/launch.h"

namespace ninfer::ops::detail::moe_cuda::block_scaled::h4096_i2048_e288_k8 {

template <class Arena>
Storage allocate(Arena& arena, const Plan& plan, int tokens, bool routed) {
    Storage out;
    if (routed && plan.vector) {
        out.vector_activation =
            static_cast<float*>(arena.alloc(DType::FP32, {kIntermediate, kTopK * tokens}).data);
    } else if (routed) {
        const int assignments = tokens * kTopK;
        const auto bytes      = [&](std::size_t size) {
            return static_cast<std::uint8_t*>(arena.alloc_bytes(size).data);
        };
        const auto integers = [&](int count) {
            return static_cast<int*>(arena.alloc(DType::I32, {count}).data);
        };
        const bool fp8    = plan.format == QType::FP8_E4M3FN_BLOCK128X128_BF16;
        const int packing = fp8 ? 1 : 2, scale_group = fp8 ? 128 : 16;
        const auto scale_bytes = fp8 ? sizeof(float) : sizeof(std::uint8_t);
        out.input_codes        = bytes(std::size_t(tokens) * kHidden / packing);
        out.input_scales       = bytes(std::size_t(tokens) * (kHidden / scale_group) * scale_bytes);
        out.product_codes      = bytes(std::size_t(assignments) * kIntermediate / packing);
        out.product_scales =
            bytes(std::size_t(assignments) * (kIntermediate / scale_group) * scale_bytes);
        if (plan.atomic_output) {
            out.routed_sum = static_cast<float*>(arena.alloc(DType::FP32, {kHidden, tokens}).data);
        } else {
            out.contributions =
                static_cast<float*>(arena.alloc(DType::FP32, {kHidden, assignments}).data);
        }
        out.active_tokens = integers(tokens);
        auto& g           = out.groups;
        if (!plan.single_block_grouping) { g.counts = integers(kExperts); }
        g.offsets = integers(kExperts + 1);
        if (!plan.single_block_grouping) { g.cursors = integers(kExperts); }
        const int max_jobs = assignments / plan.job_columns() + kExperts;
        g.job_experts      = integers(max_jobs);
        g.job_columns      = integers(max_jobs);
        g.jobs             = integers(2);
        if (plan.dense.gate_columns) g.load_range = integers(2);
        g.assignments = integers(assignments);
        if (!plan.atomic_output) { g.inverse = integers(assignments); }
    }
    return out;
}

} // namespace ninfer::ops::detail::moe_cuda::block_scaled::h4096_i2048_e288_k8
