// qus::kernels - gqa_attention prompt-scale launcher: fill k/v at device
// positions then launch causal attention over absolute cached history.
#include "kernels/launcher/gqa_attention.h"

#include "kernels/kernel/gqa_attention_prefill.cuh"
#include "qus/core/device.h" // CUDA_CHECK

#include <cstdint>

namespace qus::kernels::detail {

void gqa_attention_prompt_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                           KVCache& kv, int layer, Tensor& out,
                                           cudaStream_t stream) {
    Tensor& cache_k = kv.k[static_cast<std::uint32_t>(layer)];
    Tensor& cache_v = kv.v[static_cast<std::uint32_t>(layer)];

    constexpr int kQBlock         = kGqaPrefillBr;
    constexpr int kAttentionBlock = kGqaPrefillThreads;
    constexpr int kSmemBytes      = kGqaPrefillSmemBytes;  // 96 KiB dynamic smem (Q + K + V)

    // The FA-style tile needs > 48 KiB of shared memory, so opt in once.
    static const cudaError_t attr_rc = cudaFuncSetAttribute(
        gqa_attention_prefill_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
    CUDA_CHECK(attr_rc);

    const auto tokens         = static_cast<std::int32_t>(q.ne[2]);
    const auto padded_context = static_cast<std::int32_t>(kv.padded_context);
    const dim3 attention_grid(static_cast<unsigned>((tokens + kQBlock - 1) / kQBlock),
                              static_cast<unsigned>(kGqaPrefillQHeads), 1u);
    gqa_attention_prefill_kernel<<<attention_grid, kAttentionBlock, kSmemBytes, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(cache_k.data),
        static_cast<const __nv_bfloat16*>(cache_v.data),
        static_cast<const std::int32_t*>(positions.data), scale,
        static_cast<__nv_bfloat16*>(out.data), tokens, padded_context);
    CUDA_CHECK(cudaGetLastError());
}

void gqa_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                 const Tensor& positions, float scale, KVCache& kv, int layer,
                                 Tensor& out, cudaStream_t stream) {
    Tensor& cache_k = kv.k[static_cast<std::uint32_t>(layer)];
    Tensor& cache_v = kv.v[static_cast<std::uint32_t>(layer)];

    constexpr int kBlock      = 256;
    const auto tokens         = static_cast<std::int32_t>(q.ne[2]);
    const auto padded_context = static_cast<std::int32_t>(kv.padded_context);

    constexpr int kFillVecElems = 8;
    const std::int64_t kv_elements =
        static_cast<std::int64_t>(tokens) * kGqaPrefillKVHeads * (kGqaPrefillHeadDim / kFillVecElems);
    const int fill_grid = static_cast<int>((kv_elements + kBlock - 1) / kBlock);
    gqa_attention_prefill_fill_kernel<<<fill_grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const std::int32_t*>(positions.data), static_cast<__nv_bfloat16*>(cache_k.data),
        static_cast<__nv_bfloat16*>(cache_v.data), tokens, padded_context);
    CUDA_CHECK(cudaGetLastError());

    gqa_attention_prompt_attention_launch(q, positions, scale, kv, layer, out, stream);
}

} // namespace qus::kernels::detail
