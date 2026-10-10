#pragma once

#include "core/device.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/grouped_mma.cuh"
#include "ops/softmax_attention/dense/causal_cache/bf16/tiled_mma.cuh"
#include "ops/softmax_attention/dense/causal_cache/bf16/merge.cuh"
#include <stdexcept>

namespace ninfer::ops::detail {

template <class Geometry, bool Writable>
void validate_bf16_kv_operands(const CausalAttentionOperands& p,
                               const Bf16KvCacheView<Writable>& cache) {
    const auto aligned = [](const void* pointer) {
        return pointer && reinterpret_cast<std::uintptr_t>(pointer) % 16 == 0;
    };
    if (p.head_dim != Geometry::kHeadDim || cache.head_dim != Geometry::kHeadDim ||
        p.query_heads != Geometry::QHeads || cache.kv_heads != Geometry::KVHeads || !aligned(p.q) ||
        !aligned(p.out) || !aligned(cache.keys) || !aligned(cache.values) || !p.positions ||
        !cache.tables || p.width <= 0 || p.batch < 1 || p.batch > 65535 ||
        cache.table_stride <= 0 || p.visible_capacity <= 0 ||
        static_cast<std::int64_t>(p.visible_capacity) >
            static_cast<std::int64_t>(cache.table_stride) * kPagedKVPageSize ||
        (cache.valid_columns && !cache.table_rows))
        throw std::invalid_argument("BF16 KV template operands do not match its geometry/layout");
}

template <int Bytes, auto Kernel>
int bf16_kv_dynamic_shared() {
    static_assert(Bytes <= 99 * 1024);
    if constexpr (Bytes > 48 * 1024) {
        static const auto status =
            cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Bytes);
        CUDA_CHECK(status);
    }
    return Bytes;
}

template <class G, class S, bool MultiBatch, bool Masked, bool Writable, class Input>
void launch_bf16_kv_grouped_mma(const CausalAttentionOperands& p, Bf16KvCacheView<Writable> cache,
                                Input input, Bf16KvPartition partition, CausalPartialView partials,
                                cudaStream_t stream) {
    static_assert(Writable == Input::writes_cache);
    validate_bf16_kv_operands<G>(p, cache);
    if ((S::kFixedWidth && p.width != S::kFixedWidth) || partition.capacity < 1 ||
        partition.capacity > 256 || partition.normal_target < 1 || partition.long_target < 1 ||
        partition.key_rows != S::kKeyRows ||
        partition.live(p.visible_capacity).splits > partition.capacity || !partials.acc ||
        !partials.maximum || !partials.sum || MultiBatch != (p.batch > 1) ||
        Masked != (cache.valid_columns != nullptr))
        throw std::invalid_argument("BF16 grouped attention: invalid partition or metadata");
    if constexpr (Input::writes_cache) {
        if (!input.k || !input.v) throw std::invalid_argument("BF16 grouped append requires K/V");
    }
    {
        constexpr auto kernel = bf16_kv_grouped_mma_kernel<G, S, MultiBatch, Masked, Input>;
        constexpr int bytes   = sizeof(Bf16KvGroupedStorage<G, S>);
        int dynamic           = 0;
        if constexpr (bytes > 48 * 1024) dynamic = bf16_kv_dynamic_shared<bytes, kernel>();
        const dim3 grid(G::KVHeads * div_up(p.width * G::GroupSize, S::kQueryRows),
                        partition.capacity, p.batch);
        kernel<<<grid, S::kThreads, dynamic, stream>>>(
            p.q, input, p.positions, cache.keys, cache.values, cache.tables, cache.valid_columns,
            cache.table_rows, cache.table_stride, p.width, p.scale, partition, partials);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <class G, class S, bool MultiBatch, bool Masked, bool Writable>
void launch_bf16_kv_merge(const CausalAttentionOperands& p, Bf16KvCacheView<Writable> cache,
                          Bf16KvPartition partition, CausalPartialView partials,
                          cudaStream_t stream) {
    if (partition.capacity > S::kThreads)
        throw std::invalid_argument("BF16 merge capacity exceeds the reduction block");
    const dim3 grid(G::QHeads, div_up(G::kHeadDim, S::kDChunk), p.width * p.batch);
    bf16_kv_merge_kernel<G, S, MultiBatch, Masked><<<grid, S::kThreads, 0, stream>>>(
        partials.acc, partials.maximum, partials.sum, p.positions, cache.valid_columns, p.width,
        p.batch, partition, p.out);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, class Schedule>
void launch_bf16_kv_tiled_mma(const CausalAttentionOperands& p, Bf16KvReadView cache,
                              cudaStream_t stream) {
    validate_bf16_kv_operands<Geometry>(p, cache);
    if (p.batch != 1)
        throw std::invalid_argument("BF16 tiled attention requires a complete single-row query");
    const auto launch = [&]<class Metadata>(Metadata metadata) {
        constexpr auto kernel = bf16_kv_tiled_mma_kernel<Geometry, Schedule, Metadata>;
        const int bytes =
            bf16_kv_dynamic_shared<bf16_kv_tiled_shared_bytes<Geometry, Schedule>, kernel>();
        const dim3 grid(div_up(p.width, Schedule::kQueryRows), Geometry::QHeads);
#if defined(_MSC_VER)
        bf16_kv_tiled_mma_kernel<Geometry, Schedule, Metadata>
            <<<grid, Schedule::kThreads, bytes, stream>>>(p.q, cache.keys, cache.values, metadata,
                                                          p.positions, p.scale, p.out, p.width);
#else
        kernel<<<grid, Schedule::kThreads, bytes, stream>>>(p.q, cache.keys, cache.values, metadata,
                                                            p.positions, p.scale, p.out, p.width);
#endif
        CUDA_CHECK(cudaGetLastError());
    };
    if (!cache.table_rows) {
        launch(PagedKVDirectMetadata{cache.tables});
    } else if (cache.valid_columns) {
        launch(PagedKVBatchMetadata<true>{cache.tables, cache.valid_columns, cache.table_rows,
                                          cache.table_stride});
    } else {
        launch(PagedKVBatchMetadata<false>{cache.tables, nullptr, cache.table_rows,
                                           cache.table_stride});
    }
}

} // namespace ninfer::ops::detail
