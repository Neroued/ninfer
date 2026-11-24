#pragma once

#include "ops/softmax_attention/kernels/causal/ragged_launch.cuh"
#include "ops/kv_cache/kernels/layout.cuh"
#include "core/device.h"
#include "ops/common/host/launch.h"
#include "ops/softmax_attention/kernels/causal/nvfp4/tiled_mma.cuh"
#include <stdexcept>

namespace ninfer::ops::detail {
template <class G, class S, class... Ragged>
void launch_nvfp4_kv_tiled_mma(const CausalAttentionOperands& p, Nvfp4KvReadView cache,
                               cudaStream_t stream, Ragged... ragged) {
    validate_quantized_causal_operands<G>(p, cache);
    if (p.batch != 1)
        throw std::invalid_argument("NVFP4 tiled attention requires a complete single query row");
    constexpr bool Packed = sizeof...(Ragged) != 0;
    const int tiles       = [&] {
        if constexpr (Packed)
            return (ragged.tile_count, ...);
        else
            return common::div_up(p.width, S::kQueryRows);
    }();
    const auto invoke = [&]<class Metadata>(Metadata metadata) {
        constexpr auto kernel = [] {
            if constexpr (Packed)
                return nvfp4_kv_ragged_tiled_mma_kernel<G, S>;
            else
                return nvfp4_kv_tiled_mma_kernel<G, S, Metadata>;
        }();
        const int shared = prepare_causal_shared<S::kSharedBytes, kernel, true>(ragged...);
        const dim3 grid(tiles, G::QHeads);
        kernel<<<grid, S::kThreads, shared, stream>>>(
            p.q, cache.keys, cache.values, cache.key_scales, cache.value_scales, metadata,
            p.positions, p.scale, p.out, p.width);
        CUDA_CHECK(cudaGetLastError());
    };
    if constexpr (Packed) {
        invoke(ragged...);
    } else if (!cache.table_rows)
        invoke(kv::DirectMetadata{cache.tables});
    else if (cache.valid_columns)
        invoke(kv::BatchMetadata<true>{cache.tables, cache.valid_columns, cache.table_rows,
                                       cache.table_stride});
    else
        invoke(
            kv::BatchMetadata<false>{cache.tables, nullptr, cache.table_rows, cache.table_stride});
}

} // namespace ninfer::ops::detail
