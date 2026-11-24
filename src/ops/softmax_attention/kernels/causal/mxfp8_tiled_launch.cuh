#pragma once

#include "ops/softmax_attention/kernels/causal/ragged_launch.cuh"

#include "ops/kv_cache/kernels/layout.cuh"
#include "core/device.h"
#include "ops/common/host/launch.h"
#include "ops/softmax_attention/kernels/causal/mxfp8_tiled_mma.cuh"
#include <stdexcept>

namespace ninfer::ops::detail {

template <class G, class S, class Values, class View, class... Ragged>
void launch_mxfp8_kv_tiled_mma(const CausalAttentionOperands& p, View cache,
                               CausalKvPartition partition, CausalPartialView partial,
                               cudaStream_t stream, Ragged... ragged) {
    validate_quantized_causal_operands<G>(p, cache);
    constexpr bool Packed = sizeof...(Ragged) != 0;
    const bool direct     = Packed && partition.capacity == 1;
    if (p.batch != 1 || partition.target < 1 || partition.target > CausalKvPartition::kMaxSplits ||
        partition.capacity != partition.active(p.visible_capacity) ||
        (!direct && (!partial.acc || !partial.maximum || !partial.sum)))
        throw std::invalid_argument("MXFP8 tiled attention: invalid batch or partial storage");
    const int tiles = [&] {
        if constexpr (sizeof...(Ragged) != 0)
            return (ragged.tile_count, ...);
        else
            return common::div_up(p.width, S::kQueryRows);
    }();
    const auto invoke = [&]<bool DirectOut, class Metadata>(Metadata metadata) {
        constexpr auto kernel = [] {
            if constexpr (sizeof...(Ragged) != 0)
                return mxfp8_kv_ragged_tiled_mma_kernel<G, S, Values, DirectOut>;
            else
                return mxfp8_kv_tiled_mma_kernel<G, S, Values, Metadata>;
        }();
        const int shared = prepare_causal_shared<S::kSharedBytes, kernel, true>(ragged...);
        const dim3 grid(tiles, G::QHeads, partition.capacity);
        if constexpr (sizeof...(Ragged) != 0)
            kernel<<<grid, S::kThreads, shared, stream>>>(
                p.q, cache.keys, cache.values, cache.key_scales, cache.value_scales, metadata,
                p.positions, p.scale, p.width, partition, partial, p.out);
        else
            kernel<<<grid, S::kThreads, shared, stream>>>(
                p.q, cache.keys, cache.values, cache.key_scales, cache.value_scales, metadata,
                p.positions, p.scale, p.width, partition, partial);
        CUDA_CHECK(cudaGetLastError());
    };
    if constexpr (sizeof...(Ragged) != 0) {
        if (direct)
            invoke.template operator()<true>(ragged...);
        else
            invoke.template operator()<false>(ragged...);
    } else if (!cache.table_rows)
        invoke.template operator()<false>(kv::DirectMetadata{cache.tables});
    else if (cache.valid_columns)
        invoke.template operator()<false>(kv::BatchMetadata<true>{
            cache.tables, cache.valid_columns, cache.table_rows, cache.table_stride});
    else
        invoke.template operator()<false>(
            kv::BatchMetadata<false>{cache.tables, nullptr, cache.table_rows, cache.table_stride});
}

} // namespace ninfer::ops::detail
