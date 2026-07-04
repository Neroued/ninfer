// qus::kernels - split-KV GQA small-T launcher and unified route dispatcher.
#include "kernels/launcher/gqa_attention.h"

#include "kernels/kernel/gqa_attention_decode.cuh"
#include "qus/core/device.h" // CUDA_CHECK
#include "qus/kernels/gqa_attention.h"

#include <cstdint>
#include <stdexcept>

namespace qus::kernels::detail {
namespace {

template <int QHeadsPerCta, int TokenTile>
void launch_partial(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& pos,
                    float scale, Tensor& cache_k, Tensor& cache_v, std::int32_t padded_context,
                    std::int32_t max_context, Tensor& partial_acc, Tensor& partial_m,
                    Tensor& partial_l, cudaStream_t stream) {
    constexpr int kBlock      = 32 * QHeadsPerCta;
    constexpr int q_subgroups = (kGqaGroupSize + QHeadsPerCta - 1) / QHeadsPerCta;
    const int tokens          = q.ne[2];
    const dim3 grid(kGqaKVHeads * q_subgroups, kGqaDecodeSplits,
                    (tokens + TokenTile - 1) / TokenTile);
    gqa_attention_small_t_partial_kernel<QHeadsPerCta, TokenTile><<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data), static_cast<const std::int32_t*>(pos.data),
        static_cast<__nv_bfloat16*>(cache_k.data), static_cast<__nv_bfloat16*>(cache_v.data),
        tokens, padded_context, max_context, scale, static_cast<__nv_bfloat16*>(partial_acc.data),
        static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
    CUDA_CHECK(cudaGetLastError());
}

template <int TokenTile, int WarpsPerCta>
void launch_tc_partial(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& pos,
                       float scale, Tensor& cache_k, Tensor& cache_v, std::int32_t padded_context,
                       std::int32_t max_context, Tensor& partial_acc, Tensor& partial_m,
                       Tensor& partial_l, cudaStream_t stream) {
    constexpr int kBlock = 32 * WarpsPerCta;
    const int tokens     = q.ne[2];
    const dim3 grid(kGqaKVHeads, kGqaDecodeSplits, 1);
    gqa_attention_small_t_tc_partial_kernel<TokenTile, WarpsPerCta><<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data), static_cast<const std::int32_t*>(pos.data),
        static_cast<__nv_bfloat16*>(cache_k.data), static_cast<__nv_bfloat16*>(cache_v.data),
        tokens, padded_context, max_context, scale, static_cast<__nv_bfloat16*>(partial_acc.data),
        static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

bool gqa_attention_uses_small_t(std::int32_t tokens) { return tokens >= 1 && tokens <= 6; }

void gqa_attention_small_t_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                  const Tensor& pos, float scale, KVCache& kv, int layer,
                                  Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                  Tensor& out, cudaStream_t stream) {
    Tensor& cache_k = kv.k[static_cast<std::uint32_t>(layer)];
    Tensor& cache_v = kv.v[static_cast<std::uint32_t>(layer)];

    const auto padded_context = static_cast<std::int32_t>(kv.padded_context);
    const auto max_context    = static_cast<std::int32_t>(kv.max_context);

    switch (q.ne[2]) {
    case 1:
        launch_partial<6, 1>(q, k, v, pos, scale, cache_k, cache_v, padded_context, max_context,
                             partial_acc, partial_m, partial_l, stream);
        break;
    case 2:
        launch_tc_partial<2, 4>(q, k, v, pos, scale, cache_k, cache_v, padded_context, max_context,
                                partial_acc, partial_m, partial_l, stream);
        break;
    case 3:
        launch_tc_partial<3, 4>(q, k, v, pos, scale, cache_k, cache_v, padded_context, max_context,
                                partial_acc, partial_m, partial_l, stream);
        break;
    case 4:
        launch_tc_partial<4, 4>(q, k, v, pos, scale, cache_k, cache_v, padded_context, max_context,
                                partial_acc, partial_m, partial_l, stream);
        break;
    case 5:
        launch_tc_partial<5, 4>(q, k, v, pos, scale, cache_k, cache_v, padded_context, max_context,
                                partial_acc, partial_m, partial_l, stream);
        break;
    case 6:
        launch_tc_partial<6, 4>(q, k, v, pos, scale, cache_k, cache_v, padded_context, max_context,
                                partial_acc, partial_m, partial_l, stream);
        break;
    default:
        throw std::invalid_argument("gqa_attention_small_t_launch: unsupported T");
    }

    constexpr int kReduceBlock = 256;
    constexpr int kDChunk      = 64;
    const dim3 reduce_grid(kGqaQHeads, (kGqaHeadDim + kDChunk - 1) / kDChunk, q.ne[2]);
    gqa_attention_small_t_reduce_output_kernel<kDChunk><<<reduce_grid, kReduceBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(partial_acc.data),
        static_cast<const float*>(partial_m.data), static_cast<const float*>(partial_l.data),
        static_cast<const std::int32_t*>(pos.data), q.ne[2], kGqaDecodeSplits,
        static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void gqa_attention_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                          const Tensor& positions, float scale, KVCache& kv, int layer,
                          Tensor* partial_acc, Tensor* partial_m, Tensor* partial_l, Tensor& out,
                          cudaStream_t stream) {
    if (gqa_attention_uses_small_t(q.ne[2])) {
        if (partial_acc == nullptr || partial_m == nullptr || partial_l == nullptr) {
            throw std::invalid_argument("gqa_attention: small-T route requires workspace");
        }
        gqa_attention_small_t_launch(q, k, v, positions, scale, kv, layer, *partial_acc, *partial_m,
                                     *partial_l, out, stream);
        return;
    }
    gqa_attention_prompt_launch(q, k, v, positions, scale, kv, layer, out, stream);
}

} // namespace qus::kernels::detail
