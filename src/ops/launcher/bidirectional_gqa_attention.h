#pragma once

#include "ninfer/ops/bidirectional_gqa_attention.h"

namespace ninfer::ops::detail {

enum class BidirectionalGqaRoute {
    Direct,
    SplitKv,
};

struct BidirectionalGqaPlan {
    BidirectionalGqaRoute route = BidirectionalGqaRoute::SplitKv;
    std::int32_t tokens         = 0;
    std::int32_t warps          = 0;
    std::int32_t key_block      = 0;
    std::int32_t split_capacity = 0;
};

[[nodiscard]] BidirectionalGqaPlan
bidirectional_gqa_resolve_plan(std::int32_t tokens, GqaContextExecutionEnvelope envelope);

[[nodiscard]] const char* bidirectional_gqa_route_name(BidirectionalGqaRoute route);

void bidirectional_gqa_attention_launch(const Tensor& q, const Tensor& query_k,
                                        const Tensor& query_v, const Tensor& context_length,
                                        float scale, const KVCacheLayerView& context,
                                        const BidirectionalGqaPlan& plan, Tensor& partial_acc,
                                        Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                        cudaStream_t stream);

} // namespace ninfer::ops::detail
