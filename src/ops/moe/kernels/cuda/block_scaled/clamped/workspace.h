#pragma once

#include "ops/moe/kernels/cuda/block_scaled/clamped/launch.h"

namespace ninfer::ops::detail::moe_cuda::block_scaled::clamped {

template <class Arena>
Storage allocate(Arena& arena, const Plan& plan, int tokens, bool routed, bool partial_output) {
    Storage out;
    out.partial_output = partial_output && routed;
    if (routed && plan.direct_assignments) {
        out.direct_product =
            static_cast<float*>(arena.alloc(DType::FP32, {plan.intermediate, kTopK * tokens}).data);
    } else if (routed) {
        const int assignments = tokens * kTopK;
        const auto bytes      = [&](std::size_t size) {
            return static_cast<std::uint8_t*>(arena.alloc_bytes(size).data);
        };
        const auto integers = [&](int count) {
            return static_cast<int*>(arena.alloc(DType::I32, {count}).data);
        };
        const bool fp8    = plan.format == QType::FP8_E4M3FN_BLOCK128X128_BF16;
        const bool a16    = plan.arithmetic == Nvfp4Arithmetic::MultiplyA16;
        const int inputs  = plan.arithmetic == Nvfp4Arithmetic::MultiplyA4 ? assignments : tokens;
        const int packing = fp8 ? 1 : 2, scale_group = fp8 ? 128 : 16;
        const auto scale_bytes = fp8 ? sizeof(float) : sizeof(std::uint8_t);
        if (a16) {
            out.a16_product.values = static_cast<std::uint16_t*>(
                arena
                    .alloc(plan.a16.fp16_product ? DType::FP32 : DType::BF16,
                           {plan.intermediate, assignments})
                    .data);
            if (!plan.a16.fp16_product)
                out.a16_product.residual = static_cast<std::uint16_t*>(
                    arena.alloc(DType::BF16, {plan.intermediate, assignments}).data);
        } else {
            out.input_codes   = bytes(std::size_t(inputs) * kHidden / packing);
            out.input_scales  = bytes(std::size_t(inputs) * (kHidden / scale_group) * scale_bytes);
            out.product_codes = bytes(std::size_t(assignments) * plan.intermediate / packing);
            out.product_scales =
                bytes(std::size_t(assignments) * (plan.intermediate / scale_group) * scale_bytes);
        }
        if (plan.atomic_output && !partial_output) {
            out.routed_sum = static_cast<float*>(arena.alloc(DType::FP32, {kHidden, tokens}).data);
        } else if (!plan.atomic_output) {
            out.contributions =
                static_cast<float*>(arena.alloc(DType::FP32, {kHidden, assignments}).data);
        }
        if (!partial_output) out.active_tokens = integers(tokens);
        auto& g = out.groups;
        if (!plan.single_block_grouping) { g.counts = integers(kExperts); }
        g.offsets = integers(kExperts + 1);
        if (!plan.single_block_grouping) { g.cursors = integers(kExperts); }
        const int max_jobs = assignments / plan.job_columns() + kExperts;
        g.job_experts      = integers(max_jobs);
        g.job_columns      = integers(max_jobs);
        g.jobs             = integers(fp8 ? 4 : 2);
        if (plan.format == QType::NVFP4_MUL || plan.dense.gate_columns || plan.dense_down_columns ||
            (fp8 && plan.fp8.k_step == 0) || plan.pool_expert_limit)
            g.load_range = integers(3);
        g.assignments = integers(assignments);
        if (!plan.atomic_output || (fp8 && plan.fp8.dense_down) || plan.dense_down_columns ||
            plan.pool_expert_limit) {
            g.inverse = integers(assignments);
        }
    }
    return out;
}

} // namespace ninfer::ops::detail::moe_cuda::block_scaled::clamped
