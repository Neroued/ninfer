#pragma once

#include "core/kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Host execution-resource promise for kv_cache_append_prefix.
 *
 * The device scalar commit_count must remain in this inclusive interval on every replay. The
 * envelope fixes launch resources; it does not select or publish the committed frontier.
 */
struct KVCacheAppendPrefixExecutionEnvelope {
    std::uint32_t min_count = 0;
    std::uint32_t max_count = 0;
};

/**
 * Op: append a device-selected exact K/V prefix to linear cache storage.
 *
 * k/v are contiguous BF16 [128,8,T], positions is contiguous sequential device I32 [T], and
 * commit_count is a contiguous device I32 scalar. For every i in [0,commit_count), the Op copies
 * k/v[:, :, i] bit-for-bit into the linear cache row positions[i]. No cache byte for any rejected
 * i is written. Inputs are unchanged, and the Op neither decides nor publishes a frontier.
 *
 * The caller guarantees positive T, 0 <= commit_count <= T within the declared envelope, valid
 * sequential nonnegative positions, pairwise non-aliasing, and sufficient cache capacity.
 */
void kv_cache_append_prefix(const Tensor& k, const Tensor& v, const Tensor& positions,
                            const Tensor& commit_count,
                            KVCacheAppendPrefixExecutionEnvelope envelope, KVCacheLayerView cache,
                            cudaStream_t stream);

/**
 * Op: append the same device-selected exact K/V prefix to cyclic cache storage.
 *
 * Absolute position p maps to physical slot p mod 4096. The caller guarantees that the live
 * interval ends immediately before positions[0] and that advancing it by commit_count makes every
 * overwritten old slot dead. One invocation may commit at most the ring capacity, so no two live
 * writes race for one physical slot.
 */
void kv_cache_append_prefix(const Tensor& k, const Tensor& v, const Tensor& positions,
                            const Tensor& commit_count,
                            KVCacheAppendPrefixExecutionEnvelope envelope,
                            CyclicKVCacheLayerView cache, cudaStream_t stream);

} // namespace ninfer::ops
