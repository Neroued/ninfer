#pragma once

#include "ninfer/ops/kv_cache_append_prefix.h"

namespace ninfer::ops::detail {

enum class KVCacheAppendPrefixRoute {
    Flat16,
    Flat32,
    Persistent32,
    Token,
};

struct KVCacheAppendPrefixPlan {
    KVCacheAppendPrefixRoute route;
    std::int32_t tokens;
    std::int32_t min_count;
    std::int32_t max_count;
};

[[nodiscard]] KVCacheAppendPrefixPlan
kv_cache_append_prefix_resolve_plan(std::int32_t tokens,
                                    KVCacheAppendPrefixExecutionEnvelope envelope);
[[nodiscard]] const char* kv_cache_append_prefix_route_name(KVCacheAppendPrefixRoute route);

void kv_cache_append_prefix_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                   const Tensor& commit_count, KVCacheLayerView cache,
                                   const KVCacheAppendPrefixPlan& plan, cudaStream_t stream);
void kv_cache_append_prefix_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                   const Tensor& commit_count, CyclicKVCacheLayerView cache,
                                   const KVCacheAppendPrefixPlan& plan, cudaStream_t stream);

} // namespace ninfer::ops::detail
