#pragma once

#include "ops/softmax_attention/kernels/causal/ragged_launch.cuh"

#include "core/device.h"
#include "ops/common/host/launch.h"
#include "ops/softmax_attention/kernels/causal/fp8/grouped_mma.cuh"
#include "ops/softmax_attention/kernels/causal/merge.cuh"
#include <stdexcept>

namespace ninfer::ops::detail {

template <class G, class S, bool MultiBatch, bool Masked, bool Writable, class Input,
          bool ParallelQueries = false, class... Metadata>
void launch_fp8_kv_grouped_mma(const CausalAttentionOperands& p, Fp8KvCacheView<Writable> cache,
                               Input input, CausalKvPartition partition, CausalPartialView partial,
                               cudaStream_t stream, Metadata... metadata) {
    static_assert(Writable == Input::writes_cache);
    validate_quantized_causal_operands<G>(p, cache);
    if ((!ParallelQueries && p.width != S::kTokenTile) ||
        (sizeof...(Metadata) == 0 && MultiBatch != (p.batch > 1)) ||
        Masked != (cache.valid_columns != nullptr) || partition.capacity < 1 ||
        partition.target > CausalKvPartition::kMaxSplits || partition.target < 1 ||
        partition.key_shift < 6 || partition.key_shift > 12 ||
        partition.capacity != partition.active(p.visible_capacity) ||
        (partition.capacity != 1 && (!partial.acc || !partial.maximum || !partial.sum)))
        throw std::invalid_argument("FP8 grouped attention: invalid schedule/partials");
    if constexpr (Input::writes_cache)
        if (!input.k || !input.v) throw std::invalid_argument("FP8 append requires K/V");
    const auto launch = [&]<bool Direct>() {
        constexpr auto kernel = [] {
            if constexpr (sizeof...(Metadata) == 0)
                return fp8_kv_grouped_mma_kernel<G, S, MultiBatch, Masked, Input, ParallelQueries,
                                                 Direct>;
            else
                return fp8_kv_ragged_grouped_mma_kernel<G, S, MultiBatch, Masked, Input,
                                                        ParallelQueries, Direct>;
        }();
        constexpr int bytes = S::kDynamicArena ? S::kArenaBytes : 0;
        const int shared    = prepare_causal_shared<bytes, kernel, true>(metadata...);
        const dim3 grid(G::KVHeads * (ParallelQueries ? common::div_up(p.width, S::kTokenTile) : 1),
                        partition.capacity, p.batch);
        kernel<<<grid, S::kThreads, shared, stream>>>(
            p.q, input, p.positions, cache.keys, cache.values, cache.key_scales, cache.value_scales,
            cache.tables, cache.valid_columns, cache.table_rows, cache.table_stride, p.width,
            p.visible_capacity, partition, p.scale, partial.acc, partial.maximum, partial.sum,
            p.out, metadata...);
    };
    if (partition.capacity == 1)
        launch.template operator()<true>();
    else
        launch.template operator()<false>();
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
