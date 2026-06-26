// qus::kernels - gqa_attention decode launcher: append current k/v then launch
// single-query attention.
#include "kernels/launcher/gqa_attention.h"

#include "kernels/kernel/gqa_attention_decode.cuh"
#include "qus/core/device.h" // CUDA_CHECK

#include <cstdint>

namespace qus::kernels::detail {

void gqa_attention_decode_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                 const Tensor& pos, float scale, KVCache& kv, int layer,
                                 Tensor& out, cudaStream_t stream) {
    Tensor& cache_k = kv.k[static_cast<std::uint32_t>(layer)];
    Tensor& cache_v = kv.v[static_cast<std::uint32_t>(layer)];

    constexpr int kBlock      = 256;
    constexpr int kKVElements = kGqaHeadDim * kGqaKVHeads;
    constexpr int kAppendGrid = (kKVElements + kBlock - 1) / kBlock;

    gqa_attention_decode_append_kernel<<<kAppendGrid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const std::int32_t*>(pos.data), static_cast<__nv_bfloat16*>(cache_k.data),
        static_cast<__nv_bfloat16*>(cache_v.data), static_cast<std::int32_t>(kv.max_context));
    CUDA_CHECK(cudaGetLastError());

    gqa_attention_decode_kernel<<<kGqaQHeads, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(cache_k.data),
        static_cast<const __nv_bfloat16*>(cache_v.data), static_cast<const std::int32_t*>(pos.data),
        scale, static_cast<__nv_bfloat16*>(out.data), static_cast<std::int32_t>(kv.max_context));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
