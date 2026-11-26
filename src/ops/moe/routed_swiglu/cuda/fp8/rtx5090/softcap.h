#pragma once

#include "ops/moe/kernels/cuda/block_scaled/clamped/launch.h"
#include "ops/linear/kernels/mxfp8/a8_plan.h"

#include <algorithm>

namespace ninfer::ops::detail::moe_routed::mxfp8_softcap {

inline constexpr int hidden = 2304, intermediate = 1152, experts = 128, top_k = 8;
inline constexpr int input_scale_stride = 80, product_scale_stride = 48;
inline constexpr int slice_tokens = 4096, single_block_tokens = 64;

enum class Schedule {
    Vector,
    Grouped16,
    Grouped32,
    Grouped64,
    Grouped128,
    Wide128,
    Wide128Transposed
};

struct Plan {
    Schedule schedule = Schedule::Vector;
    bool gate_a8 = false, down_a8 = false;
    bool single_block = false;

    bool grouped() const { return schedule != Schedule::Vector; }

    int columns() const {
        switch (schedule) {
        case Schedule::Grouped128:
        case Schedule::Wide128:
        case Schedule::Wide128Transposed:
            return 128;
        case Schedule::Grouped64:
            return 64;
        case Schedule::Grouped32:
            return 32;
        default:
            return 16;
        }
    }
};

struct Storage {
    moe_cuda::block_scaled::clamped::Groups groups{};
    Mxfp8A8Workspace input{}, product{};
    std::uint16_t* bf16_product = nullptr;
    float* vector_product       = nullptr;
    float* sum                  = nullptr;
};

inline int job_capacity(Plan plan, int assignments) {
    const int active = std::min(experts, assignments);
    // The first job costs one assignment per expert. Each additional job costs a full tile.
    return active + (assignments - active) / plan.columns();
}

template <class Arena>
Storage allocate(Arena& arena, Plan plan, int tokens, bool routed) {
    Storage out{};
    if (!routed) return out;
    const int assignments = tokens * top_k;
    const auto sizes      = [&](std::size_t bytes) {
        return static_cast<std::uint8_t*>(arena.alloc_bytes(bytes).data);
    };
    const auto integers = [&](int count) {
        return static_cast<int*>(arena.alloc(DType::I32, {count}).data);
    };
    const auto sum_bytes = std::size_t(tokens) * hidden * sizeof(float);
    if (!plan.grouped()) {
        out.vector_product = reinterpret_cast<float*>(
            sizes(std::size_t(assignments) * intermediate * sizeof(float)));
        return out;
    }
    auto& g   = out.groups;
    g.counts  = integers(experts);
    g.offsets = integers(experts + 1);
    if (!plan.single_block) g.cursors = integers(experts);
    g.assignments = integers(assignments);
    g.job_experts = integers(job_capacity(plan, assignments));
    g.job_columns = integers(job_capacity(plan, assignments));
    g.jobs        = integers(1);
    if (plan.gate_a8 && plan.columns() == 128) g.load_range = integers(3);
    // The input scratch becomes the routed FP32 sum after Gate/Up.
    const int input_rows     = plan.gate_a8 ? tokens : 0;
    const auto input_codes   = std::size_t(input_rows) * hidden;
    const auto input_bytes   = input_codes + std::size_t(input_rows) * input_scale_stride;
    const auto product_codes = std::size_t(assignments) * intermediate;
    const auto product_bytes = product_codes * (plan.down_a8 ? 1 : sizeof(std::uint16_t)) +
                               (plan.down_a8 ? std::size_t(assignments) * product_scale_stride : 0);
    auto* product = sizes(product_bytes);
    auto* scratch = sizes(std::max(input_bytes, sum_bytes));
    out.sum       = reinterpret_cast<float*>(scratch);
    if (plan.gate_a8 && scratch) out.input = {scratch, scratch + input_codes};
    if (plan.down_a8) {
        if (product) out.product = {product, product + product_codes};
    } else {
        out.bf16_product = reinterpret_cast<std::uint16_t*>(product);
    }
    return out;
}

void launch(const Tensor&, const SwiGLUProfile&, const CudaSwiGLUExperts&, const ExpertRoutes&,
            ExpertSubset, MoeEpilogue, Tensor&, Plan, const Storage&, CudaExecutionView);

} // namespace ninfer::ops::detail::moe_routed::mxfp8_softcap
