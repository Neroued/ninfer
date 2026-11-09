#pragma once

#include "core/device.h"
#include "core/weight.h"
#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/tensor.h"
#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

inline constexpr std::size_t kContextKVMaterializeLayers = 5;

/** Address-independent mathematical/representation profile, shared by planning and execution. */
struct ContextKVMaterializeProfile {
    std::int32_t hidden_size;
    std::int32_t head_dim;
    std::int32_t kv_heads;
    std::uint32_t capacity;
    QType weight_type;
    float epsilon;
    float theta;
};

/** One layer's borrowed projection, norm and destination views. */
struct ContextKVMaterializeLayerView {
    Weight key_weight;
    Weight value_weight;
    Tensor key_norm_weight;
    CyclicKVCacheLayerView cache;
};

/** Host resource promise; Device counts determine the exact committed prefix. */
struct ContextKVMaterializeExecutionEnvelope {
    std::uint32_t min_count = 0;
    std::uint32_t max_count = 0;
};

/**
 * Project normalized context into five layer-local cyclic K/V caches.
 *
 * context is contiguous BF16 [H,C,B], positions is Device I32 [C,B], counts is Device I32 [B].
 * C=1..16/B=1..8 or C=1..2048/B=1. Profile epsilon/theta are positive and finite.
 * Registered (H,D,Hkv,capacity,weight_type) are (5120,128,8,2048,Q8_G32_FP16) and
 * (4096,128,4,2048,BF16). Per-layer key/value weights are [D*Hkv,H]. Q8 uses RowSplit
 * FP16 scales; BF16 uses Contiguous consecutive K/V row views of one [2*D*Hkv,H]
 * parent prefix. Thus the BF16 form consumes [K512;V512] without computing parent Q rows.
 * Each key norm is contiguous BF16 [D], a direct multiplicative gain.
 *
 * For each layer, row b and i<count[b], compute complete projections k_raw=Wk*context
 * and v_raw=Wv*context, then per-head n[d]=k_raw[d]*gamma[d]/sqrt(mean(k_raw^2)+epsilon).
 * Full-head split-half RoPE uses angle(j)=positions[i,b]*theta^(-2*j/D):
 *   K[j] = BF16(n[j]*cos(angle)-n[j+D/2]*sin(angle))
 *   K[j+D/2] = BF16(n[j+D/2]*cos(angle)+n[j]*sin(angle)).
 * There is no represented BF16 boundary between K projection and norm/RoPE.
 * V stores FP16_RNE(BF16_RNE(v_raw)); that BF16 boundary is semantic.
 *
 * Each layer independently supplies dense [D,padded_capacity,Hkv] BF16 K / FP16 V
 * fields with padded_capacity>=capacity and 16-byte field alignment. K/V within a layer
 * share a State arena and Device I64 [B] record byte-offset table; layers may use separate
 * tables/backing. Absolute position p addresses slot p mod capacity. Live columns have
 * sequential nonnegative positions. Each old live interval ends before positions[0,b];
 * advancing by counts[b] makes overwritten slots dead. Live destinations are disjoint.
 *
 * The caller guarantees min_count<=counts[b]<=max_count<=C. Inactive rows/columns never
 * read positions, projected K/V or record-offset values and never modify state. Live
 * caches, workspace and read-only operands do not overlap. Inputs and unaddressed state
 * are preserved. Weights may borrow their original parent backing; no repacking occurs.
 * Context/weight bases are 16-byte aligned, norms are 4-byte aligned; metadata is naturally
 * aligned.
 *
 * The Op owns no frontier, rollback, persistent state, allocation or synchronization.
 * It borrows scratch in an arena scope and supports capture/replay with current Device
 * metadata. All operands and workspace belong to execution.device().
 */
void context_kv_materialize(
    const Tensor& context, const Tensor& positions, const Tensor& counts,
    const std::array<ContextKVMaterializeLayerView, kContextKVMaterializeLayers>& layers,
    const ContextKVMaterializeProfile& profile, ContextKVMaterializeExecutionEnvelope envelope,
    MemoryArena& workspace, CudaExecutionView execution);

/** Capacity over every C and legal count envelope in the inclusive interval at exact B. */
[[nodiscard]] std::size_t
context_kv_materialize_workspace_capacity_bytes(const ContextKVMaterializeProfile& profile,
                                                std::int32_t batch_size, std::int32_t min_width,
                                                std::int32_t max_width);

} // namespace ninfer::ops
