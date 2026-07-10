// qus::kernels - gqa_attention prompt-scale launcher: fill k/v at device
// positions then launch causal attention over absolute cached history.
#include "kernels/launcher/gqa_attention.h"

#include "kernels/kernel/gqa_attention_prefill_bf16.cuh"
#include "kernels/kernel/gqa_attention_prefill_i8.cuh"
#include "qus/core/device.h" // CUDA_CHECK

#include <cstdint>

namespace qus::kernels::detail {

void gqa_attention_prompt_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                           KVCache& kv, int layer, Tensor& out,
                                           cudaStream_t stream) {
    Tensor& cache_k = kv.k[static_cast<std::uint32_t>(layer)];
    Tensor& cache_v = kv.v[static_cast<std::uint32_t>(layer)];
    // Both dtype-specialized kernels exceed the default 48 KiB dynamic-smem ceiling.
    static const cudaError_t attr_bf16 =
        cudaFuncSetAttribute(gqa_attention_prefill_bf16_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kGqaPrefillSmemBytes);
    CUDA_CHECK(attr_bf16);
    static const cudaError_t attr_i8 =
        cudaFuncSetAttribute(gqa_attention_prefill_i8_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kGqaPrefillI8SmemBytes);
    CUDA_CHECK(attr_i8);

    const auto tokens         = static_cast<std::int32_t>(q.ne[2]);
    const auto padded_context = static_cast<std::int32_t>(kv.padded_context);
    if (kv.dtype == DType::I8) {
        const dim3 attention_grid(
            static_cast<unsigned>((tokens + kGqaPrefillI8Br - 1) / kGqaPrefillI8Br),
            static_cast<unsigned>(kGqaPrefillQHeads), 1u);
        Tensor& cache_k_scale = kv.k_scale[static_cast<std::uint32_t>(layer)];
        Tensor& cache_v_scale = kv.v_scale[static_cast<std::uint32_t>(layer)];
        gqa_attention_prefill_i8_kernel<<<attention_grid, kGqaPrefillI8Threads,
                                          kGqaPrefillI8SmemBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const std::int8_t*>(cache_k.data),
            static_cast<const std::int8_t*>(cache_v.data),
            static_cast<const __half*>(cache_k_scale.data),
            static_cast<const __half*>(cache_v_scale.data),
            static_cast<const std::int32_t*>(positions.data), scale,
            static_cast<__nv_bfloat16*>(out.data), tokens, padded_context);
    } else {
        const dim3 attention_grid(
            static_cast<unsigned>((tokens + kGqaPrefillBr - 1) / kGqaPrefillBr),
            static_cast<unsigned>(kGqaPrefillQHeads), 1u);
        gqa_attention_prefill_bf16_kernel<<<attention_grid, kGqaPrefillThreads,
                                            kGqaPrefillSmemBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const __nv_bfloat16*>(cache_k.data),
            static_cast<const __nv_bfloat16*>(cache_v.data),
            static_cast<const std::int32_t*>(positions.data), scale,
            static_cast<__nv_bfloat16*>(out.data), tokens, padded_context);
    }
    CUDA_CHECK(cudaGetLastError());
}

void gqa_kv_append_launch(const Tensor& k, const Tensor& v, const Tensor& positions, KVCache& kv,
                          int layer, cudaStream_t stream) {
    const auto tokens         = static_cast<std::int32_t>(k.ne[2]);
    const auto padded_context = static_cast<std::int32_t>(kv.padded_context);
    Tensor& cache_k           = kv.k[static_cast<std::uint32_t>(layer)];
    Tensor& cache_v           = kv.v[static_cast<std::uint32_t>(layer)];
    if (kv.dtype == DType::I8) {
        Tensor& cache_k_scale    = kv.k_scale[static_cast<std::uint32_t>(layer)];
        Tensor& cache_v_scale    = kv.v_scale[static_cast<std::uint32_t>(layer)];
        constexpr int kFillBlock = 256;
        constexpr int kFillWarps = kFillBlock / 32;
        const std::int64_t fill_units =
            static_cast<std::int64_t>(tokens) * kGqaPrefillKVHeads * kGqaKvQuantGroups;
        const int fill_grid = static_cast<int>((fill_units + kFillWarps - 1) / kFillWarps);
        gqa_attention_prefill_fill_i8_kernel<<<fill_grid, kFillBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
            static_cast<const std::int32_t*>(positions.data),
            static_cast<std::int8_t*>(cache_k.data), static_cast<std::int8_t*>(cache_v.data),
            static_cast<__half*>(cache_k_scale.data), static_cast<__half*>(cache_v_scale.data),
            tokens, padded_context);
        CUDA_CHECK(cudaGetLastError());
    } else {
        constexpr int kBlock           = 256;
        constexpr int kFillVecElems    = 8;
        const std::int64_t kv_elements = static_cast<std::int64_t>(tokens) * kGqaPrefillKVHeads *
                                         (kGqaPrefillHeadDim / kFillVecElems);
        const int fill_grid = static_cast<int>((kv_elements + kBlock - 1) / kBlock);
        gqa_attention_prefill_fill_bf16_kernel<<<fill_grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
            static_cast<const std::int32_t*>(positions.data),
            static_cast<__nv_bfloat16*>(cache_k.data), static_cast<__nv_bfloat16*>(cache_v.data),
            tokens, padded_context);
        CUDA_CHECK(cudaGetLastError());
    }
}

void gqa_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                 const Tensor& positions, float scale, KVCache& kv, int layer,
                                 Tensor& out, cudaStream_t stream) {
    gqa_kv_append_launch(k, v, positions, kv, layer, stream);
    gqa_attention_prompt_attention_launch(q, positions, scale, kv, layer, out, stream);
}

} // namespace qus::kernels::detail
