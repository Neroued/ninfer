#pragma once

#include "ops/moe/kernels/cuda/block_scaled/clamped/launch.h"

#include <algorithm>

namespace ninfer::ops::detail::moe_routed::softcap {

inline constexpr int hidden = 2304, intermediate = 1152, experts = 128, top_k = 8;
inline constexpr int minimum_a4_reuse_twice = 3;
inline constexpr int down_k_partitions      = 3;
inline constexpr int product_scale_stride   = 80;
inline constexpr int slice_tokens = 4096, single_block_tokens = 64;

enum class Schedule {
    VectorComplete,
    VectorDecodeSplit,
    DecodeA4Reuse,
    Bf16Bucketed,
    Bf16Short16,
    Bf16Mma,
    Fp4Short,
    Fp4Mma,
    Sm120Tma64,
    Sm120TmaDense,
    A16ScaledProduct
};

constexpr bool native_tma(Schedule schedule) {
    return schedule == Schedule::Sm120TmaDense || schedule == Schedule::Sm120Tma64;
}

constexpr bool direct_vector(Schedule schedule) {
    return schedule == Schedule::VectorComplete || schedule == Schedule::VectorDecodeSplit;
}

constexpr bool quantized(Schedule schedule) {
    return schedule == Schedule::Fp4Short || schedule == Schedule::Fp4Mma || native_tma(schedule);
}

struct Plan {
    Schedule schedule          = Schedule::VectorComplete;
    int columns                = 0;
    bool single_block_grouping = false;
};

struct Storage {
    moe_cuda::block_scaled::clamped::Storage grouped;
    float* activation = nullptr;
    int* use_grouped  = nullptr;
};

template <class Arena>
Storage allocate(Arena& arena, Plan plan, int tokens, bool routed) {
    Storage storage;
    if (!routed) return storage;
    const int assignments = top_k * tokens;
    if (direct_vector(plan.schedule)) {
        storage.activation =
            static_cast<float*>(arena.alloc(DType::FP32, {intermediate, assignments}).data);
        if (plan.schedule == Schedule::VectorDecodeSplit)
            storage.grouped.contributions = static_cast<float*>(
                arena.alloc(DType::FP32, {hidden, tokens, down_k_partitions}).data);
        return storage;
    }
    const auto bytes = [&](std::size_t size) {
        return static_cast<std::uint8_t*>(arena.alloc_bytes(size).data);
    };
    const auto integers = [&](int count) {
        return static_cast<int*>(arena.alloc(DType::I32, {count}).data);
    };
    auto& s   = storage.grouped;
    auto& g   = s.groups;
    g.counts  = integers(experts);
    g.offsets = integers(experts + 1);
    if (!plan.single_block_grouping) g.cursors = integers(experts);
    g.assignments  = integers(assignments);
    const int jobs = assignments / plan.columns + std::min(experts, assignments);
    g.job_experts  = integers(jobs);
    g.job_columns  = integers(jobs);
    g.jobs         = integers(1);
    g.load_range   = integers(3);
    if (native_tma(plan.schedule)) {
        const auto code_bytes  = std::size_t(assignments) * (hidden / 2);
        const auto input_bytes = code_bytes + std::size_t(assignments) * (hidden / 16);
        // Input dies at the gate/down boundary. Down clears and reuses the same allocation.
        auto* buffer   = bytes(std::max(input_bytes, std::size_t(hidden) * tokens * sizeof(float)));
        s.input_codes  = buffer;
        s.input_scales = buffer + code_bytes;
        s.routed_sum   = reinterpret_cast<float*>(buffer);
        s.product_codes  = bytes(std::size_t(assignments) * (intermediate / 2));
        s.product_scales = bytes(std::size_t(assignments) * product_scale_stride);
        return storage;
    }
    if (plan.schedule == Schedule::DecodeA4Reuse) {
        storage.use_grouped       = integers(1);
        const auto input_codes    = std::size_t(assignments) * (hidden / 2);
        const auto input_scales   = std::size_t(assignments) * (hidden / 16);
        const auto product_codes  = std::size_t(assignments) * (intermediate / 2);
        const auto product_scales = std::size_t(assignments) * (intermediate / 16);
        // Both schedules can replay from one captured graph. The device chooses
        // one before either schedule writes these buffers.
        auto* input        = bytes(std::max(std::size_t(assignments) * intermediate * sizeof(float),
                                            input_codes + input_scales + product_codes + product_scales));
        storage.activation = reinterpret_cast<float*>(input);
        s.input_codes      = input;
        s.input_scales     = input + input_codes;
        s.product_codes    = input + input_codes + input_scales;
        s.product_scales   = input + input_codes + input_scales + product_codes;
        s.routed_sum       = s.contributions =
            static_cast<float*>(arena.alloc(DType::FP32, {hidden, tokens, down_k_partitions}).data);
        return storage;
    }
    s.routed_sum = static_cast<float*>(arena.alloc(DType::FP32, {hidden, tokens}).data);
    if (plan.schedule == Schedule::A16ScaledProduct) {
        s.a16_product.values =
            static_cast<std::uint16_t*>(arena.alloc(DType::FP32, {intermediate, assignments}).data);
    } else if (!quantized(plan.schedule)) {
        s.a16_product.values =
            static_cast<std::uint16_t*>(arena.alloc(DType::BF16, {intermediate, assignments}).data);
        s.a16_product.residual =
            static_cast<std::uint16_t*>(arena.alloc(DType::BF16, {intermediate, assignments}).data);
    }
    if (quantized(plan.schedule)) {
        s.input_codes    = bytes(std::size_t(assignments) * hidden / 2);
        s.input_scales   = bytes(std::size_t(assignments) * hidden / 16);
        s.product_codes  = bytes(std::size_t(assignments) * intermediate / 2);
        s.product_scales = bytes(std::size_t(assignments) * intermediate / 16);
    }
    return storage;
}

void launch(const Tensor&, const SwiGLUProfile&, const CudaSwiGLUExperts&, const ExpertRoutes&,
            ExpertSubset, MoeEpilogue, Tensor&, Plan, const Storage&, CudaExecutionView);

} // namespace ninfer::ops::detail::moe_routed::softcap
