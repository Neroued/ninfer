#pragma once

#include "core/device.h"
#include "core/tensor.h"
#include "ops/common/numeric/convert.cuh"

namespace ninfer::ops::detail {

// A one-key segment has probability one, independently of Q and K.
template <int Vector>
__global__ void noncausal_singleton64_kernel(const __nv_bfloat16* v, std::int64_t stride,
                                             int tokens, __nv_bfloat16* out) {
    constexpr int Width = 64 * 16;
    for (std::int64_t token = blockIdx.x; token < tokens; token += gridDim.x) {
        for (int feature = threadIdx.x * Vector; feature < Width; feature += blockDim.x * Vector) {
            common::Bf16Vector<Vector>::load(v + token * stride + feature)
                .store(out + token * Width + feature);
        }
    }
}

inline void launch_noncausal_singleton64(const Tensor& v, Tensor& out, cudaStream_t stream) {
    const unsigned blocks = v.ne[2] < 65535 ? unsigned(v.ne[2]) : 65535;
    const auto alignment  = reinterpret_cast<std::uintptr_t>(v.data) |
                           reinterpret_cast<std::uintptr_t>(out.data) | v.nb[2];
    if ((alignment & 15) == 0)
        noncausal_singleton64_kernel<8>
            <<<blocks, 128, 0, stream>>>(static_cast<const __nv_bfloat16*>(v.data), v.nb[2] / 2,
                                         v.ne[2], static_cast<__nv_bfloat16*>(out.data));
    else
        noncausal_singleton64_kernel<1>
            <<<blocks, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(v.data), v.nb[2] / 2,
                                         v.ne[2], static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
