// ninfer::ops - gqa_attention prompt-scale launcher: fill k/v at device
// positions then launch causal attention over absolute cached history.
#include "ops/launcher/gqa_attention.h"

#include "ops/common/math.h"
#include "ops/kernel/gqa_attention_prefill_bf16.cuh"
#include "ops/kernel/gqa_attention_prefill_i8.cuh"
#include "ops/kernel/gqa_attention_prefill_mixed.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

template <typename Geometry, bool KInt8>
void gqa_attention_prompt_attention_launch_mixed(const Tensor& q, const Tensor& positions,
                                                 float scale, const KVCacheLayerView& cache,
                                                 Tensor& out, cudaStream_t stream) {
    static const cudaError_t attr =
        cudaFuncSetAttribute(gqa_attention_prefill_mixed_kernel<Geometry, KInt8, !KInt8>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             kGqaPrefillMixedSmemBytes);
    CUDA_CHECK(attr);

    const auto tokens         = static_cast<std::int32_t>(q.ne[2]);
    const auto padded_context = static_cast<std::int32_t>(cache.padded_context);
    const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kGqaPrefillBr)),
                              static_cast<unsigned>(Geometry::QHeads), 1u);
    gqa_attention_prefill_mixed_kernel<Geometry, KInt8, !KInt8>
        <<<attention_grid, kGqaPrefillThreads, kGqaPrefillMixedSmemBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data), cache.k.data, cache.v.data,
            static_cast<const __half*>(cache.k_scale.data),
            static_cast<const __half*>(cache.v_scale.data),
            static_cast<const std::int32_t*>(positions.data), scale,
            static_cast<__nv_bfloat16*>(out.data), tokens, padded_context);
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry>
void gqa_attention_prompt_attention_launch_for(const Tensor& q, const Tensor& positions,
                                               float scale, const KVCacheLayerView& cache,
                                               Tensor& out, cudaStream_t stream) {
    const Tensor& cache_k = cache.k;
    const Tensor& cache_v = cache.v;
    // All format-specialized kernels exceed the default 48 KiB dynamic-smem ceiling.
    static const cudaError_t attr_bf16 =
        cudaFuncSetAttribute(gqa_attention_prefill_bf16_kernel<Geometry>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kGqaPrefillSmemBytes);
    CUDA_CHECK(attr_bf16);
    static const cudaError_t attr_i8 =
        cudaFuncSetAttribute(gqa_attention_prefill_i8_kernel<Geometry>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kGqaPrefillI8SmemBytes);
    CUDA_CHECK(attr_i8);

    const auto tokens         = static_cast<std::int32_t>(q.ne[2]);
    const auto padded_context = static_cast<std::int32_t>(cache.padded_context);
    const bool k_i8           = cache.k_dtype == DType::I8;
    const bool v_i8           = cache.v_dtype == DType::I8;
    if (k_i8 && v_i8) {
        const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kGqaPrefillI8Br)),
                                  static_cast<unsigned>(Geometry::QHeads), 1u);
        const Tensor& cache_k_scale = cache.k_scale;
        const Tensor& cache_v_scale = cache.v_scale;
        gqa_attention_prefill_i8_kernel<Geometry>
            <<<attention_grid, kGqaPrefillI8Threads, kGqaPrefillI8SmemBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const std::int8_t*>(cache_k.data),
                static_cast<const std::int8_t*>(cache_v.data),
                static_cast<const __half*>(cache_k_scale.data),
                static_cast<const __half*>(cache_v_scale.data),
                static_cast<const std::int32_t*>(positions.data), scale,
                static_cast<__nv_bfloat16*>(out.data), tokens, padded_context);
    } else if (k_i8) {
        gqa_attention_prompt_attention_launch_mixed<Geometry, true>(q, positions, scale, cache,
                                                                    out, stream);
    } else if (v_i8) {
        gqa_attention_prompt_attention_launch_mixed<Geometry, false>(q, positions, scale, cache,
                                                                     out, stream);
    } else {
        const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kGqaPrefillBr)),
                                  static_cast<unsigned>(Geometry::QHeads), 1u);
        gqa_attention_prefill_bf16_kernel<Geometry>
            <<<attention_grid, kGqaPrefillThreads, kGqaPrefillSmemBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const __nv_bfloat16*>(cache_k.data),
                static_cast<const __nv_bfloat16*>(cache_v.data),
                static_cast<const std::int32_t*>(positions.data), scale,
                static_cast<__nv_bfloat16*>(out.data), tokens, padded_context);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, bool KInt8>
void gqa_kv_append_launch_mixed(const Tensor& k, const Tensor& v, const Tensor& positions,
                                KVCacheLayerView cache, cudaStream_t stream) {
    const auto tokens         = static_cast<std::int32_t>(k.ne[2]);
    const auto padded_context = static_cast<std::int32_t>(cache.padded_context);
    constexpr int kFillBlock  = 256;
    constexpr int kFillWarps  = kFillBlock / 32;
    const std::int64_t fill_units =
        static_cast<std::int64_t>(tokens) * Geometry::KVHeads * kGqaKvQuantGroups;
    const int fill_grid =
        static_cast<int>(div_up(fill_units, static_cast<std::int64_t>(kFillWarps)));
    gqa_attention_prefill_fill_mixed_kernel<Geometry, KInt8, !KInt8>
        <<<fill_grid, kFillBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
            static_cast<const std::int32_t*>(positions.data), cache.k.data, cache.v.data,
            static_cast<__half*>(cache.k_scale.data), static_cast<__half*>(cache.v_scale.data),
            tokens, padded_context);
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry>
void gqa_kv_append_launch_for(const Tensor& k, const Tensor& v, const Tensor& positions,
                              KVCacheLayerView cache, cudaStream_t stream) {
    const auto tokens         = static_cast<std::int32_t>(k.ne[2]);
    const auto padded_context = static_cast<std::int32_t>(cache.padded_context);
    Tensor& cache_k           = cache.k;
    Tensor& cache_v           = cache.v;
    const bool k_i8           = cache.k_dtype == DType::I8;
    const bool v_i8           = cache.v_dtype == DType::I8;
    if (k_i8 && v_i8) {
        Tensor& cache_k_scale    = cache.k_scale;
        Tensor& cache_v_scale    = cache.v_scale;
        constexpr int kFillBlock = 256;
        constexpr int kFillWarps = kFillBlock / 32;
        const std::int64_t fill_units =
            static_cast<std::int64_t>(tokens) * Geometry::KVHeads * kGqaKvQuantGroups;
        const int fill_grid =
            static_cast<int>(div_up(fill_units, static_cast<std::int64_t>(kFillWarps)));
        gqa_attention_prefill_fill_i8_kernel<Geometry><<<fill_grid, kFillBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
            static_cast<const std::int32_t*>(positions.data),
            static_cast<std::int8_t*>(cache_k.data), static_cast<std::int8_t*>(cache_v.data),
            static_cast<__half*>(cache_k_scale.data), static_cast<__half*>(cache_v_scale.data),
            tokens, padded_context);
        CUDA_CHECK(cudaGetLastError());
    } else if (k_i8) {
        gqa_kv_append_launch_mixed<Geometry, true>(k, v, positions, cache, stream);
    } else if (v_i8) {
        gqa_kv_append_launch_mixed<Geometry, false>(k, v, positions, cache, stream);
    } else {
        constexpr int kBlock           = 256;
        constexpr int kFillVecElems    = 8;
        const std::int64_t kv_elements = static_cast<std::int64_t>(tokens) * Geometry::KVHeads *
                                         (kGqaPrefillHeadDim / kFillVecElems);
        const int fill_grid =
            static_cast<int>(div_up(kv_elements, static_cast<std::int64_t>(kBlock)));
        gqa_attention_prefill_fill_bf16_kernel<Geometry><<<fill_grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
            static_cast<const std::int32_t*>(positions.data),
            static_cast<__nv_bfloat16*>(cache_k.data), static_cast<__nv_bfloat16*>(cache_v.data),
            tokens, padded_context);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace

void gqa_attention_prompt_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                           const KVCacheLayerView& cache, Tensor& out,
                                           cudaStream_t stream) {
    if (q.ne[1] == Gqa27Geometry::QHeads) {
        gqa_attention_prompt_attention_launch_for<Gqa27Geometry>(q, positions, scale, cache, out,
                                                                 stream);
        return;
    }
    gqa_attention_prompt_attention_launch_for<Gqa35Geometry>(q, positions, scale, cache, out,
                                                             stream);
}

void gqa_kv_append_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                          KVCacheLayerView cache, cudaStream_t stream) {
    if (k.ne[1] == Gqa27Geometry::KVHeads) {
        gqa_kv_append_launch_for<Gqa27Geometry>(k, v, positions, cache, stream);
        return;
    }
    gqa_kv_append_launch_for<Gqa35Geometry>(k, v, positions, cache, stream);
}

void gqa_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                 const Tensor& positions, float scale, KVCacheLayerView cache,
                                 Tensor& out, cudaStream_t stream) {
    gqa_kv_append_launch(k, v, positions, cache, stream);
    gqa_attention_prompt_attention_launch(q, positions, scale, cache, out, stream);
}

} // namespace ninfer::ops::detail
