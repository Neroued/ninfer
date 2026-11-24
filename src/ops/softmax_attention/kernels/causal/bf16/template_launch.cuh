#pragma once

#include "ops/softmax_attention/kernels/causal/ragged_launch.cuh"

#include "ops/kv_cache/kernels/layout.cuh"
#include "core/device.h"
#include "ops/common/host/launch.h"
#include "ops/common/host/operands.h"
#include "ops/softmax_attention/kernels/causal/bf16/grouped_mma.cuh"
#include "ops/softmax_attention/kernels/causal/bf16/tiled_mma.cuh"
#include "ops/softmax_attention/kernels/causal/bf16/merge.cuh"
#include "ops/softmax_attention/kernels/causal/merge.cuh"
#include <stdexcept>

namespace ninfer::ops::detail {

template <class Geometry, bool Writable>
void validate_bf16_kv_operands(const CausalAttentionOperands& p,
                               const Bf16KvCacheView<Writable>& cache) {
    if (p.head_dim != Geometry::kHeadDim || cache.head_dim != Geometry::kHeadDim ||
        p.query_heads != Geometry::QHeads || cache.kv_heads != Geometry::KVHeads ||
        !common::operand::aligned(p.q, 16) || !common::operand::aligned(p.out, 16) ||
        !common::operand::aligned(cache.keys, 16) || !common::operand::aligned(cache.values, 16) ||
        !p.positions || !cache.tables || p.width <= 0 || p.batch < 1 || p.batch > 65535 ||
        cache.table_stride <= 0 || p.visible_capacity <= 0 ||
        (cache.valid_columns && !cache.table_rows))
        throw std::invalid_argument("BF16 KV template operands do not match its geometry/layout");
}

template <class G, class S, bool MultiBatch, bool Masked, bool Writable, class Input,
          class... Metadata>
void launch_bf16_kv_grouped_mma(const CausalAttentionOperands& p, Bf16KvCacheView<Writable> cache,
                                Input input, Bf16KvPartition partition, CausalPartialView partials,
                                cudaStream_t stream, Metadata... metadata) {
    static_assert(Writable == Input::writes_cache);
    validate_bf16_kv_operands<G>(p, cache);
    if ((S::kFixedWidth && p.width != S::kFixedWidth) || partition.capacity < 1 ||
        partition.capacity > 256 || partition.normal_target < 1 || partition.long_target < 1 ||
        partition.key_rows != S::kKeyRows ||
        partition.live(p.visible_capacity).splits > partition.capacity ||
        (partition.capacity != 1 && (!partials.acc || !partials.maximum || !partials.sum)) ||
        (sizeof...(Metadata) == 0 && MultiBatch != (p.batch > 1)) ||
        Masked != (cache.valid_columns != nullptr))
        throw std::invalid_argument("BF16 grouped attention: invalid partition or metadata");
    if constexpr (Input::writes_cache) {
        if (!input.k || !input.v) throw std::invalid_argument("BF16 grouped append requires K/V");
    }
    const auto launch = [&]<bool Direct>() {
        constexpr auto kernel = [] {
            if constexpr (sizeof...(Metadata) == 0)
                return bf16_kv_grouped_mma_kernel<G, S, MultiBatch, Masked, Input, Direct>;
            else
                return bf16_kv_ragged_grouped_mma_kernel<G, S, MultiBatch, Masked, Input, Direct>;
        }();
        constexpr int bytes = sizeof(Bf16KvGroupedStorage<G, S>);
        const int shared    = prepare_causal_shared<bytes, kernel>(metadata...);
        const dim3 grid(G::KVHeads * common::div_up(p.width * G::GroupSize, S::kQueryRows),
                        partition.capacity, p.batch);
        kernel<<<grid, S::kThreads, shared, stream>>>(
            p.q, input, p.positions, cache.keys, cache.values, cache.tables, cache.valid_columns,
            cache.table_rows, cache.table_stride, p.width, p.scale, partition, partials, p.out,
            metadata...);
    };
    if (partition.capacity == 1)
        launch.template operator()<true>();
    else
        launch.template operator()<false>();
    CUDA_CHECK(cudaGetLastError());
}

template <class G, class S, bool MultiBatch, bool Masked, bool Writable, class... Metadata>
void launch_bf16_kv_merge(const CausalAttentionOperands& p, Bf16KvCacheView<Writable> cache,
                          Bf16KvPartition partition, CausalPartialView partials,
                          cudaStream_t stream, Metadata... metadata) {
    if (partition.capacity == 1) {
        if constexpr (sizeof...(Metadata) == 0) {
            causal_merge_idle_kernel<<<1, 1, 0, stream>>>();
            CUDA_CHECK(cudaGetLastError());
        }
        return;
    }
    if (partition.capacity > S::kThreads)
        throw std::invalid_argument("BF16 merge capacity exceeds the reduction block");
    const dim3 grid(G::QHeads, common::div_up(G::kHeadDim, S::kDChunk), p.width * p.batch);
    constexpr auto kernel = [] {
        if constexpr (sizeof...(Metadata) == 0)
            return bf16_kv_merge_kernel<G, S, MultiBatch, Masked>;
        else
            return bf16_kv_merge_ragged_kernel<G, S, MultiBatch, Masked>;
    }();
    kernel<<<grid, S::kThreads, 0, stream>>>(partials.acc, partials.maximum, partials.sum,
                                             p.positions, cache.valid_columns, p.width, p.batch,
                                             partition, p.out, metadata...);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, class Schedule, class... Ragged>
void launch_bf16_kv_tiled_mma(const CausalAttentionOperands& p, Bf16KvReadView cache,
                              cudaStream_t stream, Ragged... ragged) {
    validate_bf16_kv_operands<Geometry>(p, cache);
    if (p.batch != 1)
        throw std::invalid_argument("BF16 tiled attention requires a complete single-row query");
    constexpr bool Packed = sizeof...(Ragged) != 0;
    const int tiles       = [&] {
        if constexpr (Packed)
            return (ragged.tile_count, ...);
        else
            return common::div_up(p.width, Schedule::kQueryRows);
    }();
    const auto launch = [&]<class Metadata>(Metadata metadata) {
        constexpr auto kernel = [] {
            if constexpr (Packed)
                return bf16_kv_ragged_tiled_mma_kernel<Geometry, Schedule>;
            else
                return bf16_kv_tiled_mma_kernel<Geometry, Schedule, Metadata>;
        }();
        const int bytes =
            prepare_causal_shared<bf16_kv_tiled_shared_bytes<Geometry, Schedule>, kernel, true>(
                ragged...);
        const dim3 grid(tiles, Geometry::QHeads);
        kernel<<<grid, Schedule::kThreads, bytes, stream>>>(p.q, cache.keys, cache.values, metadata,
                                                            p.positions, p.scale, p.out, p.width);
        CUDA_CHECK(cudaGetLastError());
    };
    if constexpr (Packed) {
        launch(ragged...);
    } else if (!cache.table_rows) {
        launch(kv::DirectMetadata{cache.tables});
    } else if (cache.valid_columns) {
        launch(kv::BatchMetadata<true>{cache.tables, cache.valid_columns, cache.table_rows,
                                       cache.table_stride});
    } else {
        launch(
            kv::BatchMetadata<false>{cache.tables, nullptr, cache.table_rows, cache.table_stride});
    }
}

} // namespace ninfer::ops::detail
