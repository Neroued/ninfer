// ninfer::ops::detail - row-scaled E4M3FN-cache causal prompt launch ownership.
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/kv_cache/append/launch.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_fp8.cuh"
#ifdef NINFER_VOLTA_BUILD
#include "ops/softmax_attention/dense/causal_cache/prompt_volta.cuh"
#endif

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

template <typename Geometry, typename CacheView, typename Metadata>
void causal_attention_prompt_fp8_attention_launch_for(const Tensor& q, const Tensor& positions,
                                                      float scale, const CacheView& cache,
                                                      Metadata metadata, Tensor& out,
                                                      cudaStream_t stream) {
#ifdef NINFER_VOLTA_BUILD
    const auto tokens = static_cast<std::int32_t>(q.ne[2]);
    const dim3 grid(static_cast<unsigned>(tokens), static_cast<unsigned>(Geometry::QHeads), 1u);
    causal_attention_prompt_fp8_volta_kernel<Geometry, Metadata>
        <<<grid, kCausalPromptVoltaThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const std::uint8_t*>(cache.k_pages.data),
            static_cast<const std::uint8_t*>(cache.v_pages.data),
            static_cast<const __half*>(cache.k_scale_pages.data),
            static_cast<const __half*>(cache.v_scale_pages.data), metadata,
            static_cast<const std::int32_t*>(positions.data), scale,
            static_cast<__nv_bfloat16*>(out.data), tokens);
    CUDA_CHECK(cudaGetLastError());
    return;
#else
    static const cudaError_t attr = cudaFuncSetAttribute(
        causal_attention_prompt_fp8_kernel<Geometry, Metadata>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kCausalPromptFp8SmemBytes);
    CUDA_CHECK(attr);

    const auto tokens = static_cast<std::int32_t>(q.ne[2]);
    const dim3 grid(static_cast<unsigned>(div_up(tokens, kCausalPromptFp8Br)),
                    static_cast<unsigned>(Geometry::QHeads), 1u);
    causal_attention_prompt_fp8_kernel<Geometry, Metadata>
        <<<grid, kCausalPromptFp8Threads, kCausalPromptFp8SmemBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const std::uint8_t*>(cache.k_pages.data),
            static_cast<const std::uint8_t*>(cache.v_pages.data),
            static_cast<const __half*>(cache.k_scale_pages.data),
            static_cast<const __half*>(cache.v_scale_pages.data), metadata,
            static_cast<const std::int32_t*>(positions.data), scale,
            static_cast<__nv_bfloat16*>(out.data), tokens);
    CUDA_CHECK(cudaGetLastError());
#endif
}

template <typename CacheView, typename Metadata>
void causal_attention_prompt_fp8_attention_dispatch(const Tensor& q, const Tensor& positions,
                                                    float scale, const CacheView& cache,
                                                    Metadata metadata, Tensor& out,
                                                    cudaStream_t stream) {
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        causal_attention_prompt_fp8_attention_launch_for<CausalD256H24Kv4>(
            q, positions, scale, cache, metadata, out, stream);
        return;
    }
    causal_attention_prompt_fp8_attention_launch_for<CausalD256H16Kv2>(q, positions, scale, cache,
                                                                       metadata, out, stream);
}

} // namespace

void causal_attention_prompt_fp8_attention_launch(const Tensor& q, const Tensor& positions,
                                                  float scale, const PagedKVLayerView& cache,
                                                  Tensor& out, cudaStream_t stream) {
    const CausalPromptDirectMetadata metadata{
        static_cast<const std::int32_t*>(cache.block_table.data)};
    causal_attention_prompt_fp8_attention_dispatch(q, positions, scale, cache, metadata, out,
                                                   stream);
}

void causal_attention_prompt_fp8_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                        const Tensor& positions, const Tensor& valid_columns,
                                        const Tensor& table_rows, float scale,
                                        PagedKVBatchLayerView cache, Tensor& out,
                                        cudaStream_t stream) {
    const auto launch_row = [&]<bool Masked>(const Tensor& q_row, const Tensor& k_row,
                                            const Tensor& v_row, const Tensor& positions_row,
                                            const Tensor& valid_row, const Tensor& table_row,
                                            Tensor& out_row) {
        kv_cache_append_batch_launch(k_row, v_row, positions_row, valid_row, table_row, cache,
                                     stream);
        const CausalPromptBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_row.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_row.data),
            .table_stride = cache.block_tables.ne[0],
        };
        causal_attention_prompt_fp8_attention_dispatch(q_row, positions_row, scale, cache, metadata,
                                                       out_row, stream);
    };
    for (std::int32_t batch = 0; batch < q.ne[3]; ++batch) {
        Tensor q_row         = q.slice(3, batch, 1);
        Tensor k_row         = k.slice(3, batch, 1);
        Tensor v_row         = v.slice(3, batch, 1);
        Tensor positions_row = positions.slice(1, batch, 1);
        Tensor table_row     = table_rows.slice(0, batch, 1);
        Tensor out_row       = out.slice(3, batch, 1);
        if (valid_columns.data == nullptr) {
            launch_row.template operator()<false>(q_row, k_row, v_row, positions_row, Tensor{},
                                                  table_row, out_row);
        } else {
            Tensor valid_row = valid_columns.slice(0, batch, 1);
            launch_row.template operator()<true>(q_row, k_row, v_row, positions_row, valid_row,
                                                 table_row, out_row);
        }
    }
}

} // namespace ninfer::ops::detail
