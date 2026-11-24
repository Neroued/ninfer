#pragma once

#include "core/device.h"
#include "core/cyclic_kv_cache.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"


#include <cstdint>

namespace ninfer::ops {

/**
 * Host launch-resource promise for device-selected prefix append. It bounds every device count
 * during capture/replay but neither selects nor publishes a committed frontier.
 */
struct KVCacheAppendPrefixExecutionEnvelope {
    std::uint32_t min_count = 0;
    std::uint32_t max_count = 0;
};

/**
 * Append every K/V row to single-sequence paged growing-cache storage.
 *
 * k/v are contiguous BF16 [256,4|2,T] and positions is contiguous sequential device I32 [T].
 * Vector codecs and K/V profile composition are defined in ninfer/kv_cache_format.h.
 * BF16 cache mode copies K bit-for-bit and stores V as FP16_RNE(BF16 input). Quantized
 * code and scale groups run along contiguous features: INT8 G64/FP16, FP8 G128/UE8M0,
 * NVFP4 G16/UE4M3. One FP8 D256 vector occupies 256 code bytes and two scale bytes;
 * one NVFP4 vector occupies 128 code bytes and sixteen scale bytes. K8V4 composes
 * the same FP8 K codec and NVFP4 V codec.
 *
 * INT8 and FP8 apply the fixed normalized D256 Hadamard preparation to K; NVFP4 and K8V4 apply it
 * to both K and V. Every transform is evaluated in FP32 from represented BF16 source values. The
 * paired Q and output interpretation belongs to the causal softmax_attention contract. Transform
 * results and raw code/scale bytes are not standalone mathematical outputs. Standalone and fused
 * append produce byte-identical cache representations. Every addressed code/value and scale is
 * overwritten, and no unrelated cache row is read or written. Page tables contain I64 record byte
 * offsets into the shared arena. Inputs, tables and addressed record-local planes are pairwise
 * non-overlapping. The Op owns no persistent allocation, frontier, request identity,
 * or commit authority.
 */
void kv_cache_append(const Tensor& k, const Tensor& v, const Tensor& positions,
                     PagedKVLayerView cache, CudaExecutionView execution);

/**
 * Append device-selected BF16 prefixes to batched paged growing-cache storage.
 *
 * k/v are contiguous BF16 [D,Hkv,T,B], positions is contiguous device I32 [T,B], and counts and
 * table_rows are contiguous device I32 [B]. For row b and i in [0,counts[b]), k/v[:, :, i, b]
 * are encoded at logical position positions[i,b] through table row table_rows[b]. D128/Hkv8
 * uses BF16 K and FP16 V record-local planes [128,64,8]. D256/Hkv2|4 uses
 * record-local planes and any storage format defined by kv_cache_append, with the same exact codec.
 * No byte belonging only to the rejected physical tail [counts[b],T) is written. A zero count
 * reads no positions or K/V for that batch row. Inputs are unchanged, and the Op
 * neither decides nor publishes a committed frontier.
 *
 * The caller guarantees T>0, B=1..8, envelope.min_count <= counts[b] <= envelope.max_count <= T,
 * sequential nonnegative live positions, materialized table entries for every position allowed by
 * the envelope, and pairwise non-aliasing of inputs and cache storage.
 */
void kv_cache_append_prefix(const Tensor& k, const Tensor& v, const Tensor& positions,
                            const Tensor& counts, const Tensor& table_rows,
                            KVCacheAppendPrefixExecutionEnvelope envelope,
                            PagedKVBatchLayerView cache, CudaExecutionView execution);

/**
 * Append device-selected BF16 prefixes to StateImage cyclic storage.
 *
 * k/v, positions, counts, and their storage-conversion and mutation contracts match the paged
 * overload. The cache State fields select destination records through device I64 byte offsets
 * [B]. Each field is dense [128,padded_capacity,8] within its record and 16-byte aligned. D=128,
 * Hkv=8 and capacity 2048 or 4096. Absolute position p maps to slot p mod capacity. For a nonempty
 * prefix, the caller guarantees that the row's existing live interval ends immediately before
 * positions[0,b]. Advancing it by counts[b] makes every overwritten old slot dead, and one row
 * commits at most the ring capacity. A zero count reads no positions or K/V and changes no cache
 * bytes. Physical T is independent of the committed prefix length. Consequently, no two live writes
 * race for one physical slot. The Op does not own or publish the state frontier.
 */
void kv_cache_append_prefix(const Tensor& k, const Tensor& v, const Tensor& positions,
                            const Tensor& counts, KVCacheAppendPrefixExecutionEnvelope envelope,
                            CyclicKVCacheLayerView cache, CudaExecutionView execution);

} // namespace ninfer::ops
