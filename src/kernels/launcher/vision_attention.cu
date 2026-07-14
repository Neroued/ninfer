#include "kernels/launcher/vision_attention.h"

#include "kernels/kernel/vision_attention.cuh"
#include "ninfer/core/device.h"

#include <cstdint>

namespace ninfer::kernels::detail {
namespace {

std::int64_t stride_elements(const Tensor& tensor, int dim) {
    return tensor.nb[dim] / static_cast<std::int64_t>(sizeof(__nv_bfloat16));
}

} // namespace

void vision_attention_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                             const Tensor& cu_seqlens, Tensor* tiles, Tensor& out,
                             cudaStream_t stream) {
    const bool packed_segments = tiles != nullptr;
    const int max_tiles =
        packed_segments ? tiles->ne[1] : (q.ne[2] + kVisionAttentionBr - 1) / kVisionAttentionBr;
    if (packed_segments) {
        vision_attention_prepare_tiles_kernel<<<1, 256, 0, stream>>>(
            static_cast<const std::int32_t*>(cu_seqlens.data), cu_seqlens.ne[0] - 1,
            static_cast<VisionAttentionTile*>(tiles->data), max_tiles, q.ne[2]);
        CUDA_CHECK(cudaGetLastError());
    }

    const dim3 grid(static_cast<unsigned>(max_tiles), static_cast<unsigned>(kVisionAttentionHeads),
                    1u);
    vision_attention_flash_kernel<<<grid, kVisionAttentionThreads, kVisionAttentionSmemBytes,
                                    stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data),
        packed_segments ? static_cast<const VisionAttentionTile*>(tiles->data) : nullptr, q.ne[2],
        static_cast<__nv_bfloat16*>(out.data), stride_elements(q, 0), stride_elements(q, 1),
        stride_elements(q, 2), stride_elements(k, 0), stride_elements(k, 1), stride_elements(k, 2),
        stride_elements(v, 0), stride_elements(v, 1), stride_elements(v, 2));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::kernels::detail
