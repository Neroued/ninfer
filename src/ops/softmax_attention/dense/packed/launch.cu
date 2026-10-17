#include "ops/softmax_attention/dense/packed/launch.h"
#include "ops/softmax_attention/kernels/noncausal/kernel.cuh"

namespace ninfer::ops::detail {
void launch_packed_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                             const Tensor& cu_seqlens, Tensor* tiles, Tensor& out,
                             cudaStream_t stream) {
    const bool packed_segments = tiles != nullptr;
    const int max_tiles        = packed_segments
                                     ? tiles->ne[1]
                                     : (q.ne[2] + kPackedDescriptorRows - 1) / kPackedDescriptorRows;
    if (packed_segments) {
        packed_attention_prepare_tiles_kernel<kPackedDescriptorRows><<<1, 256, 0, stream>>>(
            static_cast<const std::int32_t*>(cu_seqlens.data), cu_seqlens.ne[0] - 1,
            static_cast<NoncausalAttentionTile*>(tiles->data), max_tiles, q.ne[2]);
        CUDA_CHECK(cudaGetLastError());
    }

    launch_flash<kPackedDescriptorRows, kPackedDescriptorRows>(
        q, k, v,
        packed_segments ? static_cast<const NoncausalAttentionTile*>(tiles->data) : nullptr, 0,
        max_tiles, out, stream);
}

} // namespace ninfer::ops::detail
