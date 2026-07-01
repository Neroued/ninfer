// qus::kernels - gqa_attention prefill launcher: fill prompt k/v then launch
// causal full-prompt attention.
#include "kernels/launcher/gqa_attention.h"

#include "kernels/kernel/gqa_attention_prefill.cuh"
#include "qus/core/device.h" // CUDA_CHECK

#include <cstdint>

namespace qus::kernels::detail {

void gqa_attention_prefill_launch(const Tensor& q, const Tensor& k, const Tensor& v, float scale,
                                  KVCache& kv, int layer, Tensor& out, cudaStream_t stream) {
    Tensor& cache_k = kv.k[static_cast<std::uint32_t>(layer)];
    Tensor& cache_v = kv.v[static_cast<std::uint32_t>(layer)];

    constexpr int kBlock = 256;
    const auto tokens    = static_cast<std::int32_t>(q.ne[2]);
    const auto padded_context = static_cast<std::int32_t>(kv.padded_context);

    const std::int64_t kv_elements =
        static_cast<std::int64_t>(tokens) * kGqaPrefillKVHeads * kGqaPrefillHeadDim;
    const int fill_grid = static_cast<int>((kv_elements + kBlock - 1) / kBlock);
    gqa_attention_prefill_fill_kernel<<<fill_grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
        static_cast<__nv_bfloat16*>(cache_k.data), static_cast<__nv_bfloat16*>(cache_v.data),
        tokens, padded_context);
    CUDA_CHECK(cudaGetLastError());

    constexpr int kQBlock = 32;
    constexpr int kAttentionBlock = 512;
    const dim3 attention_grid(static_cast<unsigned>((tokens + kQBlock - 1) / kQBlock),
                              static_cast<unsigned>(kGqaPrefillQHeads), 1u);
    gqa_attention_prefill_kernel<<<attention_grid, kAttentionBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(cache_k.data),
        static_cast<const __nv_bfloat16*>(cache_v.data), scale,
        static_cast<__nv_bfloat16*>(out.data), tokens, padded_context);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
