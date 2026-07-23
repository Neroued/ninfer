#pragma once

#include "ninfer/ops/swa.h"

namespace ninfer::ops::detail {

inline constexpr std::int32_t kSwaMaxCandidateSplit = 128;

enum class SwaRoute {
    Direct,
    SplitKv,
};

struct SwaPlan {
    SwaRoute route;
    std::int32_t tokens;
    std::int32_t warps;
    std::int32_t key_block;
    std::int32_t split_capacity;
    std::int32_t max_context;
};

[[nodiscard]] SwaPlan swa_resolve_plan(std::int32_t tokens, SwaContextExecutionEnvelope envelope);
[[nodiscard]] const char* swa_route_name(SwaRoute route);

void swa_launch(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                const Tensor& positions, float scale, const CyclicKVCacheLayerView& context,
                const SwaPlan& plan, Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
