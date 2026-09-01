#include "ops/softmax_attention/dense/packed/launch.h"

#include "ops/softmax_attention/dense/packed/kernel.cuh"
#include "core/device.h"
#ifdef NINFER_VOLTA_BUILD
#include "ops/softmax_attention/dense/packed/volta.cuh"
#endif

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

std::int64_t stride_elements(const Tensor& tensor, int dim) {
    return tensor.nb[dim] / static_cast<std::int64_t>(sizeof(__nv_bfloat16));
}

template <int Br, int Bc>
void launch_flash(const Tensor& q, const Tensor& k, const Tensor& v,
                  const PackedAttentionTile* tiles, std::int32_t uniform_segment_length,
                  std::int32_t query_tiles, Tensor& out, cudaStream_t stream) {
    constexpr int kThreads = Br * 2;
    constexpr int kSmemBytes =
        (Br + 2 * Bc) * kPackedAttentionPaddedD * static_cast<int>(sizeof(__nv_bfloat16));
    const dim3 grid(static_cast<unsigned>(query_tiles),
                    static_cast<unsigned>(kPackedAttentionHeads), 1u);
    packed_attention_flash_kernel<Br, Bc><<<grid, kThreads, kSmemBytes, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data), tiles, q.ne[2], uniform_segment_length,
        static_cast<__nv_bfloat16*>(out.data), stride_elements(q, 0), stride_elements(q, 1),
        stride_elements(q, 2), stride_elements(k, 0), stride_elements(k, 1), stride_elements(k, 2),
        stride_elements(v, 0), stride_elements(v, 1), stride_elements(v, 2));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void packed_attention_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                             const Tensor& cu_seqlens, Tensor* tiles, Tensor& out,
                             cudaStream_t stream) {
#ifdef NINFER_VOLTA_BUILD
    const dim3 grid(static_cast<unsigned>(q.ne[2]), kPackedAttentionHeads, 1u);
    packed_attention_volta_kernel<<<grid, kPackedAttentionVoltaThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const std::int32_t*>(cu_seqlens.data), cu_seqlens.ne[0] - 1, 0, q.ne[2],
        static_cast<__nv_bfloat16*>(out.data), stride_elements(q, 0), stride_elements(q, 1),
        stride_elements(q, 2), stride_elements(k, 0), stride_elements(k, 1),
        stride_elements(k, 2), stride_elements(v, 0), stride_elements(v, 1),
        stride_elements(v, 2));
    CUDA_CHECK(cudaGetLastError());
    return;
#else
    const bool packed_segments = tiles != nullptr;
    const int max_tiles =
        packed_segments ? tiles->ne[1] : (q.ne[2] + kPackedAttentionBr - 1) / kPackedAttentionBr;
    if (packed_segments) {
        packed_attention_prepare_tiles_kernel<<<1, 256, 0, stream>>>(
            static_cast<const std::int32_t*>(cu_seqlens.data), cu_seqlens.ne[0] - 1,
            static_cast<PackedAttentionTile*>(tiles->data), max_tiles, q.ne[2]);
        CUDA_CHECK(cudaGetLastError());
    }

    launch_flash<kPackedAttentionBr, kPackedAttentionBc>(
        q, k, v, packed_segments ? static_cast<const PackedAttentionTile*>(tiles->data) : nullptr,
        0, max_tiles, out, stream);
#endif
}

std::int32_t packed_attention_uniform_tile(std::int32_t segment_length) {
    // Measured issued-MMA rates are approximately 3:6:8 for the 16/32/64
    // tiles. Minimize padded tile area divided by that rate.
    constexpr std::int32_t tiles[] = {16, 32, 64};
    constexpr std::int32_t rates[] = {3, 6, 8};
    std::int32_t best_tile         = tiles[0];
    std::int64_t best_padded       = ((segment_length + best_tile - 1) / best_tile) * best_tile;
    std::int64_t best_square       = best_padded * best_padded;
    std::int32_t best_rate         = rates[0];
    for (int i = 1; i < 3; ++i) {
        const std::int64_t padded = ((segment_length + tiles[i] - 1) / tiles[i]) * tiles[i];
        const std::int64_t square = padded * padded;
        if (square * best_rate < best_square * rates[i]) {
            best_tile   = tiles[i];
            best_square = square;
            best_rate   = rates[i];
        }
    }
    return best_tile;
}

void packed_attention_uniform_launch_with_tile(const Tensor& q, const Tensor& k, const Tensor& v,
                                               std::int32_t segment_length, std::int32_t tile_size,
                                               Tensor& out, cudaStream_t stream) {
#ifdef NINFER_VOLTA_BUILD
    const dim3 grid(static_cast<unsigned>(q.ne[2]), kPackedAttentionHeads, 1u);
    packed_attention_volta_kernel<<<grid, kPackedAttentionVoltaThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data), nullptr, 0, segment_length, q.ne[2],
        static_cast<__nv_bfloat16*>(out.data), stride_elements(q, 0), stride_elements(q, 1),
        stride_elements(q, 2), stride_elements(k, 0), stride_elements(k, 1),
        stride_elements(k, 2), stride_elements(v, 0), stride_elements(v, 1),
        stride_elements(v, 2));
    CUDA_CHECK(cudaGetLastError());
    return;
#else
    const std::int32_t segments    = q.ne[2] / segment_length;
    const std::int32_t query_tiles = segments * ((segment_length + tile_size - 1) / tile_size);
    switch (tile_size) {
    case 16:
        launch_flash<16, 16>(q, k, v, nullptr, segment_length, query_tiles, out, stream);
        return;
    case 32:
        launch_flash<32, 32>(q, k, v, nullptr, segment_length, query_tiles, out, stream);
        return;
    case 64:
        launch_flash<64, 64>(q, k, v, nullptr, segment_length, query_tiles, out, stream);
        return;
    default:
        throw std::invalid_argument("packed_softmax_attention: invalid uniform tile size");
    }
#endif
}

void packed_attention_uniform_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                     std::int32_t segment_length, Tensor& out,
                                     cudaStream_t stream) {
    packed_attention_uniform_launch_with_tile(
        q, k, v, segment_length, packed_attention_uniform_tile(segment_length), out, stream);
}

} // namespace ninfer::ops::detail
