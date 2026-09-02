// ninfer::ops::detail - group-16 NVFP4 split-KV small-T launch ownership.
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/softmax_attention/dense/causal_cache/small_t_nvfp4.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <typename Geometry, int TokenTile, bool MultiBatch, bool Masked, typename CacheInput>
void launch_nvfp4_partial(const Tensor& q, CacheInput input, const Tensor& positions, float scale,
                          PagedKVBatchLayerView cache, const CausalSmallTInvocation& invocation,
                          std::int32_t logical_capacity, std::int32_t splits, Tensor& partial_acc,
                          Tensor& partial_m, Tensor& partial_l, cudaStream_t stream) {
    constexpr int RowCount             = TokenTile * Geometry::GroupSize;
    constexpr int RowTiles             = (RowCount + 15) / 16;
    constexpr int Warps                = RowTiles == 3 ? 12 : 8;
    constexpr int KeyBlock             = 32;
    constexpr int MinBlocks            = RowTiles <= 2 ? 2 : 1;
    constexpr std::size_t DynamicBytes = (RowTiles <= 2 ? 3u : 5u) * KeyBlock * kCausalHeadDim;
    using KernelInput                  = CacheInput;
    const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
    const auto launch = [&]() {
        auto kernel = causal_attention_small_t_nvfp4_tiled_kernel<
            Geometry, TokenTile, Warps, MinBlocks, KeyBlock, true, MultiBatch, Masked, KernelInput>;
        static const cudaError_t attr = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(DynamicBytes));
        CUDA_CHECK(attr);

        const auto q_ptr         = static_cast<const __nv_bfloat16*>(q.data);
        const auto positions_ptr = static_cast<const std::int32_t*>(positions.data);
        const auto cache_k_ptr   = static_cast<std::uint8_t*>(cache.k_pages.data);
        const auto cache_v_ptr   = static_cast<std::uint8_t*>(cache.v_pages.data);
        const auto k_scale_ptr   = static_cast<std::uint8_t*>(cache.k_scale_pages.data);
        const auto v_scale_ptr   = static_cast<std::uint8_t*>(cache.v_scale_pages.data);
        const auto tables_ptr    = static_cast<const std::int32_t*>(cache.block_tables.data);
        const auto valid_ptr =
            invocation.valid_columns == nullptr
                ? nullptr
                : static_cast<const std::int32_t*>(invocation.valid_columns->data);
        const auto rows_ptr   = invocation.table_rows == nullptr
                                    ? nullptr
                                    : static_cast<const std::int32_t*>(invocation.table_rows->data);
        auto* partial_acc_ptr = static_cast<float*>(partial_acc.data);
        auto* partial_m_ptr   = static_cast<float*>(partial_m.data);
        auto* partial_l_ptr   = static_cast<float*>(partial_l.data);
        kernel<<<grid, Warps * 32, DynamicBytes, stream>>>(
            q_ptr, input, positions_ptr, cache_k_ptr, cache_v_ptr, k_scale_ptr, v_scale_ptr,
            tables_ptr, valid_ptr, rows_ptr, cache.block_tables.ne[0], invocation.full_width,
            invocation.column_begin, logical_capacity, scale, partial_acc_ptr, partial_m_ptr,
            partial_l_ptr);
        CUDA_CHECK(cudaGetLastError());
    };

    launch();
}

template <typename Geometry, bool MultiBatch, bool Masked>
void launch_nvfp4_reduce(const Tensor& positions, const CausalSmallTInvocation& invocation,
                         std::int32_t splits, const Tensor& partial_acc, const Tensor& partial_m,
                         const Tensor& partial_l, Tensor& out, cudaStream_t stream) {
    constexpr int Block = 256;
    const dim3 grid(Geometry::QHeads, invocation.width * invocation.batch_size);
    const auto launch = [&]<bool Offset>() {
        causal_attention_small_t_nvfp4_reduce_output_kernel<Geometry, MultiBatch, Masked, Offset>
            <<<grid, Block, 0, stream>>>(
                static_cast<const float*>(partial_acc.data),
                static_cast<const float*>(partial_m.data),
                static_cast<const float*>(partial_l.data),
                static_cast<const std::int32_t*>(positions.data),
                invocation.valid_columns == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                invocation.width, invocation.full_width, invocation.column_begin,
                invocation.batch_size, splits, static_cast<__nv_bfloat16*>(out.data));
    };
    if (invocation.column_begin == 0) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, typename CacheInput>
void causal_attention_small_t_nvfp4_launch_for(
    const Tensor& q, CacheInput input, const Tensor& positions, float scale,
    PagedKVBatchLayerView cache, const CausalSmallTInvocation& invocation,
    CausalAttentionExecutionEnvelope envelope, Tensor& partial_acc, Tensor& partial_m,
    Tensor& partial_l, Tensor& out, cudaStream_t stream) {
    const auto logical_capacity = static_cast<std::int32_t>(envelope.max_visible_keys);
    const auto splits = causal_attention_split_capacity(Geometry::QHeads, invocation.width,
                                                        cache.storage, envelope);

    const auto launch_partial = [&]<int Tokens, bool MultiBatch, bool Masked>() {
        launch_nvfp4_partial<Geometry, Tokens, MultiBatch, Masked>(
            q, input, positions, scale, cache, invocation, logical_capacity, splits, partial_acc,
            partial_m, partial_l, stream);
    };
    const auto dispatch_metadata = [&]<int Tokens>() {
        const bool masked = invocation.valid_columns != nullptr;
        if (invocation.batch_size == 1) {
            if (masked) {
                launch_partial.template operator()<Tokens, false, true>();
            } else {
                launch_partial.template operator()<Tokens, false, false>();
            }
        } else if (masked) {
            launch_partial.template operator()<Tokens, true, true>();
        } else {
            launch_partial.template operator()<Tokens, true, false>();
        }
    };

    switch (invocation.width) {
    case 1:
        dispatch_metadata.template operator()<1>();
        break;
    case 2:
        dispatch_metadata.template operator()<2>();
        break;
    case 3:
        dispatch_metadata.template operator()<3>();
        break;
    case 4:
        dispatch_metadata.template operator()<4>();
        break;
    case 5:
        dispatch_metadata.template operator()<5>();
        break;
    case 6:
        dispatch_metadata.template operator()<6>();
        break;
    default:
        throw std::invalid_argument("causal_attention_small_t_nvfp4_launch: unsupported T");
    }

    const bool masked = invocation.valid_columns != nullptr;
    if (invocation.batch_size == 1) {
        if (masked) {
            launch_nvfp4_reduce<Geometry, false, true>(positions, invocation, splits, partial_acc,
                                                       partial_m, partial_l, out, stream);
        } else {
            launch_nvfp4_reduce<Geometry, false, false>(positions, invocation, splits, partial_acc,
                                                        partial_m, partial_l, out, stream);
        }
    } else if (masked) {
        launch_nvfp4_reduce<Geometry, true, true>(positions, invocation, splits, partial_acc,
                                                  partial_m, partial_l, out, stream);
    } else {
        launch_nvfp4_reduce<Geometry, true, false>(positions, invocation, splits, partial_acc,
                                                   partial_m, partial_l, out, stream);
    }
}

} // namespace

void causal_attention_small_t_nvfp4_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream) {
    const CausalAppendInput input{static_cast<const __nv_bfloat16*>(k.data),
                                  static_cast<const __nv_bfloat16*>(v.data)};
    const CausalSmallTInvocation invocation{
        .valid_columns = valid_columns.data == nullptr ? nullptr : &valid_columns,
        .table_rows    = &table_rows,
        .full_width    = q.ne[2],
        .column_begin  = column_begin,
        .width         = width,
        .batch_size    = q.ne[3],
    };
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        causal_attention_small_t_nvfp4_launch_for<CausalD256H24Kv4>(
            q, input, positions, scale, cache, invocation, envelope, partial_acc, partial_m,
            partial_l, out, stream);
        return;
    }
    causal_attention_small_t_nvfp4_launch_for<CausalD256H16Kv2>(q, input, positions, scale, cache,
                                                                invocation, envelope, partial_acc,
                                                                partial_m, partial_l, out, stream);
}

void causal_attention_cached_small_t_nvfp4_launch(const Tensor& q, const Tensor& positions,
                                                  float scale, const PagedKVLayerView& cache,
                                                  CausalAttentionExecutionEnvelope envelope,
                                                  Tensor& partial_acc, Tensor& partial_m,
                                                  Tensor& partial_l, Tensor& out,
                                                  cudaStream_t stream) {
    const CausalCachedInput input{};
    const CausalSmallTInvocation invocation{
        .valid_columns = nullptr,
        .table_rows    = nullptr,
        .full_width    = q.ne[2],
        .column_begin  = 0,
        .width         = q.ne[2],
        .batch_size    = 1,
    };
    PagedKVBatchLayerView batch_cache = single_row_paged_kv_batch_view(cache);
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        causal_attention_small_t_nvfp4_launch_for<CausalD256H24Kv4>(
            q, input, positions, scale, batch_cache, invocation, envelope, partial_acc, partial_m,
            partial_l, out, stream);
        return;
    }
    causal_attention_small_t_nvfp4_launch_for<CausalD256H16Kv2>(
        q, input, positions, scale, batch_cache, invocation, envelope, partial_acc, partial_m,
        partial_l, out, stream);
}

} // namespace ninfer::ops::detail
