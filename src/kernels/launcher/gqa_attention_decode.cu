// qus::kernels - split-KV GQA decode launcher.
#include "kernels/launcher/gqa_attention.h"

#include "kernels/kernel/gqa_attention_decode.cuh"
#include "qus/core/device.h" // CUDA_CHECK

#include <cstdint>
#include <stdexcept>

namespace qus::kernels::detail {
namespace {

template <int TileN, int QHeadsPerCta, bool WarpPerQueryHead>
void launch_partial(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& pos,
                    float scale, Tensor& cache_k, Tensor& cache_v, std::int32_t padded_context,
                    std::int32_t max_context, std::int32_t tile_count, Tensor& partial_acc,
                    Tensor& partial_m, Tensor& partial_l, cudaStream_t stream) {
    constexpr int kBlock      = WarpPerQueryHead ? 32 * QHeadsPerCta : 256;
    constexpr int q_subgroups = (kGqaGroupSize + QHeadsPerCta - 1) / QHeadsPerCta;
    const dim3 grid(kGqaKVHeads * q_subgroups, tile_count);
    gqa_attention_decode_partial_kernel<TileN, QHeadsPerCta, WarpPerQueryHead>
        <<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data), static_cast<const std::int32_t*>(pos.data),
        static_cast<__nv_bfloat16*>(cache_k.data), static_cast<__nv_bfloat16*>(cache_v.data),
        padded_context, max_context, scale, static_cast<__nv_bfloat16*>(partial_acc.data),
        static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
    CUDA_CHECK(cudaGetLastError());
}

template <int QHeadsPerCta>
void launch_tile(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& pos, float scale,
                 Tensor& cache_k, Tensor& cache_v, std::int32_t padded_context,
                 std::int32_t max_context, std::int32_t tile_n, std::int32_t tile_count,
                 Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                 cudaStream_t stream) {
    constexpr bool kWarpPerQueryHead = QHeadsPerCta == 6;
    switch (tile_n) {
    case 32:
        launch_partial<32, QHeadsPerCta, kWarpPerQueryHead>(
            q, k, v, pos, scale, cache_k, cache_v, padded_context, max_context, tile_count,
            partial_acc, partial_m, partial_l, stream);
        return;
    case 64:
        launch_partial<64, QHeadsPerCta, kWarpPerQueryHead>(
            q, k, v, pos, scale, cache_k, cache_v, padded_context, max_context, tile_count,
            partial_acc, partial_m, partial_l, stream);
        return;
    case 128:
        launch_partial<128, QHeadsPerCta, kWarpPerQueryHead>(
            q, k, v, pos, scale, cache_k, cache_v, padded_context, max_context, tile_count,
            partial_acc, partial_m, partial_l, stream);
        return;
    default:
        throw std::invalid_argument("gqa_attention_decode_launch: unsupported tile_n");
    }
}

} // namespace

void gqa_attention_decode_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                 const Tensor& pos, float scale, KVCache& kv, int layer,
                                 std::int32_t tile_n, std::int32_t tile_count,
                                 std::int32_t q_heads_per_cta, Tensor& partial_acc,
                                 Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                 cudaStream_t stream) {
    Tensor& cache_k = kv.k[static_cast<std::uint32_t>(layer)];
    Tensor& cache_v = kv.v[static_cast<std::uint32_t>(layer)];

    const auto padded_context = static_cast<std::int32_t>(kv.padded_context);
    const auto max_context    = static_cast<std::int32_t>(kv.max_context);

    switch (q_heads_per_cta) {
    case 6:
        launch_tile<6>(q, k, v, pos, scale, cache_k, cache_v, padded_context, max_context, tile_n,
                       tile_count, partial_acc, partial_m, partial_l, stream);
        break;
    case 3:
        launch_tile<3>(q, k, v, pos, scale, cache_k, cache_v, padded_context, max_context, tile_n,
                       tile_count, partial_acc, partial_m, partial_l, stream);
        break;
    case 2:
        launch_tile<2>(q, k, v, pos, scale, cache_k, cache_v, padded_context, max_context, tile_n,
                       tile_count, partial_acc, partial_m, partial_l, stream);
        break;
    default:
        throw std::invalid_argument("gqa_attention_decode_launch: unsupported q_heads_per_cta");
    }

    constexpr int kReduceBlock = 256;
    constexpr int kDChunk      = 32;
    const dim3 reduce_grid(kGqaQHeads, (kGqaHeadDim + kDChunk - 1) / kDChunk);
    gqa_attention_decode_reduce_output_kernel<kDChunk><<<reduce_grid, kReduceBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(partial_acc.data),
        static_cast<const float*>(partial_m.data), static_cast<const float*>(partial_l.data),
        tile_count, static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
