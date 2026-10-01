#pragma once
#include "core/device.h"
#include "ops/softmax_attention/common/mxfp8_tiled_plan.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/tiled_mma.cuh"
#include <stdexcept>

namespace ninfer::ops::detail {
template <class G, class S>
void launch_nvfp4_kv_tiled_mma(const CausalAttentionOperands& p, Nvfp4KvReadView cache,
                               CausalKvPartition partition, CausalPartialView partial,
                               cudaStream_t stream) {
    validate_quantized_causal_operands<G>(p, cache);
    if (p.batch != 1 || partition.target < 1 || partition.target > kMxfp8TiledMaxSplits ||
        partition.capacity != partition.active(p.visible_capacity) || !partial.acc ||
        !partial.maximum || !partial.sum)
        throw std::invalid_argument("NVFP4 tiled attention: invalid batch or partial storage");
    const auto invoke = [&]<class Metadata>(Metadata metadata) {
        constexpr auto kernel    = nvfp4_kv_tiled_mma_kernel<G, S, Metadata>;
        static const auto status = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, S::kSharedBytes);
        CUDA_CHECK(status);
        const dim3 grid(div_up(p.width, S::kQueryRows), G::QHeads, partition.capacity);
        kernel<<<grid, S::kThreads, S::kSharedBytes, stream>>>(
            p.q, cache.keys, cache.values, cache.key_scales, cache.value_scales, metadata,
            p.positions, p.scale, p.width, partition, partial);
        CUDA_CHECK(cudaGetLastError());
    };
    if (!cache.table_rows)
        invoke(PagedKVDirectMetadata{cache.tables});
    else if (cache.valid_columns)
        invoke(PagedKVBatchMetadata<true>{cache.tables, cache.valid_columns, cache.table_rows,
                                          cache.table_stride});
    else
        invoke(PagedKVBatchMetadata<false>{cache.tables, nullptr, cache.table_rows,
                                           cache.table_stride});
}

} // namespace ninfer::ops::detail
