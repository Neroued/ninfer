#include "ops/softmax_attention/dense/noncausal_gqa/launch.h"

#include "ops/softmax_attention/dense/noncausal_gqa/kernel.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

std::int64_t stride_elements(const Tensor& tensor, int dim) {
    return tensor.nb[dim] / static_cast<std::int64_t>(sizeof(__nv_bfloat16));
}

void noncausal_gqa_attention_launch_impl(const Tensor& q, const Tensor& k, const Tensor& v,
                                         const NoncausalGQATile* tiles, std::int32_t max_tiles,
                                         Tensor& out, cudaStream_t stream) {
    constexpr int kThreads   = kNoncausalGQABr * 2;
    constexpr int kSmemBytes = (kNoncausalGQABr + 2 * kNoncausalGQABc) * kNoncausalGQAPaddedD *
                               static_cast<int>(sizeof(__nv_bfloat16));
    // D=256 single-buffered smem (Br + 2*Bc) * Dp * 2 = 96 KiB exceeds the 48 KiB default.
    static const cudaError_t attr = cudaFuncSetAttribute(
        noncausal_gqa_flash_kernel<kNoncausalGQABr, kNoncausalGQABc>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
    CUDA_CHECK(attr);
    const dim3 grid(static_cast<unsigned>(max_tiles),
                    static_cast<unsigned>(kNoncausalGQAQueryHeads), 1u);
    noncausal_gqa_flash_kernel<kNoncausalGQABr, kNoncausalGQABc>
        <<<grid, kThreads, kSmemBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
            static_cast<const __nv_bfloat16*>(v.data), tiles, q.ne[2], 0,
            static_cast<__nv_bfloat16*>(out.data), stride_elements(q, 0), stride_elements(q, 1),
            stride_elements(q, 2), stride_elements(k, 0), stride_elements(k, 1),
            stride_elements(k, 2), stride_elements(v, 0), stride_elements(v, 1),
            stride_elements(v, 2));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void noncausal_gqa_attention_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                    const Tensor& cu_seqlens, Tensor* tiles, Tensor& out,
                                    cudaStream_t stream) {
    const bool segmented   = tiles != nullptr;
    const std::int32_t max_tiles =
        segmented ? tiles->ne[1] : (q.ne[2] + kNoncausalGQABr - 1) / kNoncausalGQABr;
    if (segmented) {
        noncausal_gqa_prepare_tiles_kernel<<<1, 256, 0, stream>>>(
            static_cast<const std::int32_t*>(cu_seqlens.data), cu_seqlens.ne[0] - 1,
            static_cast<NoncausalGQATile*>(tiles->data), max_tiles, q.ne[2]);
        CUDA_CHECK(cudaGetLastError());
        noncausal_gqa_attention_launch_impl(
            q, k, v, static_cast<const NoncausalGQATile*>(tiles->data), max_tiles, out, stream);
    } else {
        noncausal_gqa_attention_launch_impl(q, k, v, nullptr, max_tiles, out, stream);
    }
}

} // namespace ninfer::ops::detail
