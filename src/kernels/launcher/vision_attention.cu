#include "kernels/launcher/vision_attention.h"

#include "kernels/kernel/vision_attention.cuh"
#include "qus/core/device.h"

#include <cstdint>

namespace qus::kernels::detail {
namespace {

std::int64_t stride_elements(const Tensor& tensor, int dim) {
    return tensor.nb[dim] / static_cast<std::int64_t>(sizeof(__nv_bfloat16));
}

} // namespace

void vision_attention_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                             const Tensor& cu_seqlens, Tensor& out, cudaStream_t stream) {
    constexpr int block = 128;
    const dim3 grid(static_cast<unsigned>(q.ne[2]), static_cast<unsigned>(kVisionAttentionHeads),
                    1u);
    vision_attention_reference_kernel<block><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const std::int32_t*>(cu_seqlens.data), cu_seqlens.ne[0] - 1,
        static_cast<__nv_bfloat16*>(out.data), q.ne[2], stride_elements(q, 0),
        stride_elements(q, 1), stride_elements(q, 2), stride_elements(k, 0), stride_elements(k, 1),
        stride_elements(k, 2), stride_elements(v, 0), stride_elements(v, 1), stride_elements(v, 2));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
