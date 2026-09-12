#pragma once

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

inline constexpr std::size_t kContextKVMaterializeLayers = 5;

/** One layer's non-owning projection, norm, and destination-state views. */
struct ContextKVMaterializeLayerView {
    Weight key_weight;
    Weight value_weight;
    Tensor key_norm_weight;
    CyclicKVCacheLayerView cache;
};

/**
 * Host launch-resource promise for device-selected context prefixes. The device counts determine
 * the state transition; this interval only bounds them for capture-safe execution.
 */
struct ContextKVMaterializeExecutionEnvelope {
    std::uint32_t min_count = 0;
    std::uint32_t max_count = 0;
};

/**
 * Materialize one normalized context block into five layer-local cyclic K/V caches.
 *
 * context is contiguous BF16 [5120,W,B], positions is contiguous device I32 [W,B], and counts
 * and state_slots are contiguous device I32 [B]. The registered domains are context blocks
 * W=1..16,B=1..8 and single-request prefill B=1,W=1..2048. W is the physical context width,
 * independent of the draft width and the device-selected committed prefix. Each of the five layer
 * views contains independent RowSplit W8G32_F16S key/value weights [1024,5120], a BF16 key norm
 * weight [128], and a capacity-2048 D128/H8 cyclic cache with BF16 K and FP16 V. All five caches
 * share padded capacity and lane capacity.
 *
 * For every layer l, row b, and i in [0,counts[b]), the complete operation is
 *
 *   k_raw = linear(context[:,i,b], key_weight[l])
 *   v_raw = linear(context[:,i,b], value_weight[l])     // represented BF16
 *   inv   = 1 / sqrt(sum_d k_raw[h,d]^2 / 128 + 1e-6)
 *   n[d]  = k_raw[h,d] * inv * key_norm_weight[l][d]
 *   angle(j) = positions[i,b] * (1e7)^(-2*j/128), 0<=j<64
 *   k[j]     = n[j]    * cos(angle(j)) - n[j+64] * sin(angle(j))
 *   k[j+64]  = n[j+64] * cos(angle(j)) + n[j]    * sin(angle(j))
 *
 * K is stored as BF16 and V as FP16_RNE from represented BF16 v_raw at absolute position
 * positions[i,b] modulo 2048 in lane state_slots[b]. Columns i>=counts[b] have no state effect.
 * Existing cache values outside the addressed rows and all read-only inputs are unchanged. The Op
 * emits no raw K/V and owns no logical frontier, commit, rollback, request identity, or persistent
 * allocation.
 *
 * The caller guarantees min_count <= counts[b] <= max_count <= W, sequential nonnegative live
 * positions, in-range state slots for live rows, and disjoint live destinations. Each selected
 * lane's old live interval ends immediately before positions[0,b], and advancing it by counts[b]
 * makes every overwritten ring slot dead. Inputs, weights, live workspace, and all cache planes are
 * mutually non-overlapping. Raw K has no semantic storage cast. Reduction association, projection
 * staging precision, and intermediate arithmetic are implementation details. The Op is safe for
 * CUDA Graph capture and replay over the complete envelope.
 */
void context_kv_materialize(
    const Tensor& context, const Tensor& positions, const Tensor& counts, const Tensor& state_slots,
    const std::array<ContextKVMaterializeLayerView, kContextKVMaterializeLayers>& layers,
    ContextKVMaterializeExecutionEnvelope envelope, WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Return transient capacity for every registered W in the inclusive interval at the exact batch
 * size, covering every valid count envelope and its selected route.
 */
[[nodiscard]] std::size_t context_kv_materialize_workspace_capacity_bytes(std::int32_t batch_size,
                                                                          std::int32_t min_width,
                                                                          std::int32_t max_width);

} // namespace ninfer::ops
