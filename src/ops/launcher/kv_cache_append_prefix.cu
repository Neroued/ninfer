#include "ops/launcher/kv_cache_append_prefix.h"

#include "core/device.h"
#include "ops/kernel/kv_cache_append_prefix.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock = 256;

template <bool Cyclic, class Cache>
void launch_for_layout(const Tensor& k, const Tensor& v, const Tensor& positions,
                       const Tensor& commit_count, Cache cache, const KVCacheAppendPrefixPlan& plan,
                       cudaStream_t stream) {
    if (plan.tokens != k.ne[2] || plan.min_count < 0 || plan.max_count < plan.min_count ||
        plan.max_count > plan.tokens) {
        throw std::invalid_argument("kv_cache_append_prefix: inconsistent plan");
    }
    if (plan.max_count == 0) return;
    auto* cache_k       = static_cast<__nv_bfloat16*>(cache.k.data);
    auto* cache_v       = static_cast<__nv_bfloat16*>(cache.v.data);
    const auto* input_k = static_cast<const __nv_bfloat16*>(k.data);
    const auto* input_v = static_cast<const __nv_bfloat16*>(v.data);
    const auto* pos     = static_cast<const std::int32_t*>(positions.data);
    const auto* count   = static_cast<const std::int32_t*>(commit_count.data);
    int padded          = 0;
    if constexpr (Cyclic) {
        padded = static_cast<int>(cache.padded_capacity);
    } else {
        padded = static_cast<int>(cache.padded_context);
    }

    switch (plan.route) {
    case KVCacheAppendPrefixRoute::Flat16: {
        const int grid = 1 + (plan.max_count - 1) / 2;
        kv_cache_append_prefix_flat_kernel<Cyclic, 16, false><<<grid, kBlock, 0, stream>>>(
            input_k, input_v, pos, count, cache_k, cache_v, plan.min_count, plan.max_count, padded);
        break;
    }
    case KVCacheAppendPrefixRoute::Flat32: {
        const int grid = 1 + (plan.max_count - 1) / 4;
        kv_cache_append_prefix_flat_kernel<Cyclic, 32, false><<<grid, kBlock, 0, stream>>>(
            input_k, input_v, pos, count, cache_k, cache_v, plan.min_count, plan.max_count, padded);
        break;
    }
    case KVCacheAppendPrefixRoute::Persistent32:
        kv_cache_append_prefix_flat_kernel<Cyclic, 32, true><<<1, kBlock, 0, stream>>>(
            input_k, input_v, pos, count, cache_k, cache_v, plan.min_count, plan.max_count, padded);
        break;
    case KVCacheAppendPrefixRoute::Token:
        kv_cache_append_prefix_token_kernel<Cyclic><<<plan.max_count, 128, 0, stream>>>(
            input_k, input_v, pos, count, cache_k, cache_v, plan.min_count, plan.max_count, padded);
        break;
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

KVCacheAppendPrefixPlan
kv_cache_append_prefix_resolve_plan(std::int32_t tokens,
                                    KVCacheAppendPrefixExecutionEnvelope envelope) {
    if (tokens < 1) {
        throw std::invalid_argument("kv_cache_append_prefix plan: T must be positive");
    }
    if (envelope.min_count > envelope.max_count ||
        envelope.max_count > static_cast<std::uint32_t>(tokens)) {
        throw std::invalid_argument("kv_cache_append_prefix plan: invalid execution envelope");
    }
    return {
        .route     = KVCacheAppendPrefixRoute::Flat32,
        .tokens    = tokens,
        .min_count = static_cast<std::int32_t>(envelope.min_count),
        .max_count = static_cast<std::int32_t>(envelope.max_count),
    };
}

const char* kv_cache_append_prefix_route_name(KVCacheAppendPrefixRoute route) {
    switch (route) {
    case KVCacheAppendPrefixRoute::Flat16:
        return "flat16";
    case KVCacheAppendPrefixRoute::Flat32:
        return "flat32";
    case KVCacheAppendPrefixRoute::Persistent32:
        return "persistent32";
    case KVCacheAppendPrefixRoute::Token:
        return "token";
    }
    return "unknown";
}

void kv_cache_append_prefix_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                   const Tensor& commit_count, KVCacheLayerView cache,
                                   const KVCacheAppendPrefixPlan& plan, cudaStream_t stream) {
    launch_for_layout<false>(k, v, positions, commit_count, cache, plan, stream);
}

void kv_cache_append_prefix_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                   const Tensor& commit_count, CyclicKVCacheLayerView cache,
                                   const KVCacheAppendPrefixPlan& plan, cudaStream_t stream) {
    launch_for_layout<true>(k, v, positions, commit_count, cache, plan, stream);
}

} // namespace ninfer::ops::detail
