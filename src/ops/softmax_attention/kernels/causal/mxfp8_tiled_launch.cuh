#pragma once

#include "ops/kv_cache/kernels/layout.cuh"
#include "core/device.h"
#include "ops/common/host/launch.h"
#include "ops/softmax_attention/kernels/causal/mxfp8_tiled_mma.cuh"
#include <stdexcept>

namespace ninfer::ops::detail {

template <class G, class S, class Values, class View>
void launch_mxfp8_kv_tiled_mma(const CausalAttentionOperands& p, View cache,
                               CausalKvPartition partition, CausalPartialView partial,
                               cudaStream_t stream) {
    validate_quantized_causal_operands<G>(p, cache);
    if (p.batch != 1 || partition.target < 1 || partition.target > CausalKvPartition::kMaxSplits ||
        partition.capacity != partition.active(p.visible_capacity) || !partial.acc ||
        !partial.maximum || !partial.sum)
        throw std::invalid_argument("MXFP8 tiled attention: invalid batch or partial storage");
    const auto invoke = [&]<class Metadata>(Metadata metadata) {
        constexpr auto kernel = mxfp8_kv_tiled_mma_kernel<G, S, Values, Metadata>;
        const int shared      = common::prepare_shared<S::kSharedBytes, kernel, true>();
        const dim3 grid(common::div_up(p.width, S::kQueryRows), G::QHeads, partition.capacity);
#if defined(_MSC_VER)
        // MSVC's cudafe cannot launch through the constexpr function-pointer variable
        // captured by this lambda (C2326), so reference the kernel template directly.
        mxfp8_kv_tiled_mma_kernel<G, S, Values, Metadata>
            <<<grid, S::kThreads, shared, stream>>>(
                p.q, cache.keys, cache.values, cache.key_scales, cache.value_scales, metadata,
                p.positions, p.scale, p.width, partition, partial);
#else
        kernel<<<grid, S::kThreads, shared, stream>>>(
            p.q, cache.keys, cache.values, cache.key_scales, cache.value_scales, metadata,
            p.positions, p.scale, p.width, partition, partial);
#endif
        CUDA_CHECK(cudaGetLastError());
    };
    if (!cache.table_rows)
        invoke(kv::DirectMetadata{cache.tables});
    else if (cache.valid_columns)
        invoke(kv::BatchMetadata<true>{cache.tables, cache.valid_columns, cache.table_rows,
                                       cache.table_stride});
    else
        invoke(
            kv::BatchMetadata<false>{cache.tables, nullptr, cache.table_rows, cache.table_stride});
}

} // namespace ninfer::ops::detail
