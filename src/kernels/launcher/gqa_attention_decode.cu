// qus::kernels - split-KV GQA decode launcher.
#include "kernels/launcher/gqa_attention.h"

#include "kernels/kernel/gqa_attention_decode.cuh"
#include "qus/core/device.h" // CUDA_CHECK
#include "qus/kernels/gqa_attention.h"

#include <cstdint>

namespace qus::kernels::detail {
namespace {

template <int QHeadsPerCta, bool WarpPerQueryHead>
void launch_partial(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& pos,
                    float scale, Tensor& cache_k, Tensor& cache_v, std::int32_t padded_context,
                    std::int32_t max_context, Tensor& partial_acc, Tensor& partial_m,
                    Tensor& partial_l, cudaStream_t stream) {
    constexpr int kBlock      = WarpPerQueryHead ? 32 * QHeadsPerCta : 256;
    constexpr int q_subgroups = (kGqaGroupSize + QHeadsPerCta - 1) / QHeadsPerCta;
    const dim3 grid(kGqaKVHeads * q_subgroups, kGqaDecodeSplits);
    gqa_attention_decode_partial_kernel<QHeadsPerCta, WarpPerQueryHead>
        <<<grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
            static_cast<const __nv_bfloat16*>(v.data), static_cast<const std::int32_t*>(pos.data),
            static_cast<__nv_bfloat16*>(cache_k.data), static_cast<__nv_bfloat16*>(cache_v.data),
            padded_context, max_context, scale, static_cast<__nv_bfloat16*>(partial_acc.data),
            static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void gqa_attention_decode_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                 const Tensor& pos, float scale, KVCache& kv, int layer,
                                 Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                 Tensor& out, cudaStream_t stream) {
    Tensor& cache_k = kv.k[static_cast<std::uint32_t>(layer)];
    Tensor& cache_v = kv.v[static_cast<std::uint32_t>(layer)];

    const auto padded_context = static_cast<std::int32_t>(kv.padded_context);
    const auto max_context    = static_cast<std::int32_t>(kv.max_context);

    launch_partial<6, true>(q, k, v, pos, scale, cache_k, cache_v, padded_context, max_context,
                            partial_acc, partial_m, partial_l, stream);

    constexpr int kReduceBlock = 256;
    constexpr int kDChunk      = 64;
    const dim3 reduce_grid(kGqaQHeads, (kGqaHeadDim + kDChunk - 1) / kDChunk);
    gqa_attention_decode_reduce_output_kernel<kDChunk><<<reduce_grid, kReduceBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(partial_acc.data),
        static_cast<const float*>(partial_m.data), static_cast<const float*>(partial_l.data),
        kGqaDecodeSplits, static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
