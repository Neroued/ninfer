#pragma once

#include "ops/moe/kernels/cuda/groupwise_int/launch.h"

#include <algorithm>

namespace ninfer::ops::detail::moe_cuda::groupwise_int {

inline constexpr int kAssignmentTile = 64;

template <class Arena>
GroupedStorage allocate_grouping(Arena& arena, int tokens, int columns) {
    using S = SwiGLUH2048I512E256K8;
    GroupedStorage out;
    const int assignments = S::top_k * tokens;
    const int tiles       = (assignments + kAssignmentTile - 1) / kAssignmentTile;
    const auto integers   = [&](int count) {
        return static_cast<int*>(arena.alloc(DType::I32, {count}).data);
    };
    out.local_ranks = integers(assignments);
    out.tile_counts = integers(S::experts * tiles);
    out.tile_bases  = integers(S::experts * tiles);
    out.offsets     = integers(S::experts + 1);
    const int jobs  = assignments / columns + S::experts;
    out.job_experts = integers(jobs);
    out.job_columns = integers(jobs);
    // Count/scan has finished before index construction. Its dead counters hold both maps.
    static_assert(S::experts >= 2 * kAssignmentTile);
    out.inverse = out.tile_counts;
    out.tokens  = out.tile_counts ? out.tile_counts + assignments : nullptr;
    out.info    = static_cast<GroupedInfo*>(arena.alloc_bytes(sizeof(GroupedInfo)).data);
    return out;
}

template <class Arena>
ExpertStorage allocate_token(Arena& arena, int tokens, bool shared, bool routed = true) {
    using S = SwiGLUH2048I512E256K8;
    ExpertStorage out;
    out.token_activation = static_cast<float*>(
        arena.alloc(DType::FP32, {S::intermediate, (int(routed) * S::top_k + int(shared)) * tokens})
            .data);
    return out;
}

template <class Arena>
ExpertStorage allocate_grouped(Arena& arena, int tokens, const GroupedPlan& plan, bool shared,
                               bool split_activation) {
    using S = SwiGLUH2048I512E256K8;
    ExpertStorage out;
    out.groups     = allocate_grouping(arena, tokens, plan.columns);
    out.grouped_io = arena.alloc(DType::BF16, {S::hidden, S::top_k * tokens}).data;
    // Down consumes both activation planes before reduction overwrites them with the FP32 sum.
    const auto activation_bytes =
        std::size_t(S::intermediate) * S::top_k * tokens * (split_activation ? 4 : 2);
    const auto sum_bytes   = shared ? std::size_t(S::hidden) * tokens * 4 : 0;
    out.grouped_activation = arena.alloc_bytes(std::max(activation_bytes, sum_bytes)).data;
    out.routed_sum         = shared ? static_cast<float*>(out.grouped_activation) : nullptr;
    out.token_activation   = static_cast<float*>(out.grouped_io);
    if (shared) {
        out.shared_activation = arena.alloc(DType::BF16, {S::intermediate, tokens, 2}).data;
    }
    return out;
}

template <class Arena>
ExpertStorage allocate_shared(Arena& arena, int tokens) {
    using S = SwiGLUH2048I512E256K8;
    ExpertStorage out;
    // SIMT keeps FP32 values; MMA keeps BF16 high/low planes in the same-sized storage.
    const auto buffer     = arena.alloc(DType::FP32, {S::intermediate, tokens});
    out.token_activation  = static_cast<float*>(buffer.data);
    out.shared_activation = buffer.data;
    return out;
}

} // namespace ninfer::ops::detail::moe_cuda::groupwise_int
