#pragma once

#include "ninfer/ops/sliding_window_attention.h"

namespace ninfer::ops::detail {

inline constexpr std::int32_t kSlidingWindowMaxSplits = 32;

enum class SlidingWindowAttentionRoute {
    Direct,
    SplitKv,
};

struct SlidingWindowAttentionPlan {
    SlidingWindowAttentionRoute route;
    std::int32_t tokens;
    std::int32_t warps;
    std::int32_t key_block;
    std::int32_t reduce_warps;
    std::int32_t split_capacity;
    std::int32_t max_context;
    std::int32_t window;
};

[[nodiscard]] SlidingWindowAttentionPlan
sliding_window_attention_resolve_plan(std::uint32_t window, std::int32_t tokens, std::int32_t batch,
                                      SlidingWindowAttentionExecutionEnvelope envelope);
[[nodiscard]] const char* sliding_window_attention_route_name(SlidingWindowAttentionRoute route);

void sliding_window_attention_launch(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                                     const Tensor& positions, const Tensor& valid_columns,
                                     const Tensor& lanes, float scale,
                                     const CyclicKVCacheLayerView& context,
                                     const SlidingWindowAttentionPlan& plan, Tensor& partial_acc,
                                     Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                     cudaStream_t stream);

} // namespace ninfer::ops::detail
