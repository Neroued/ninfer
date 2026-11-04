#pragma once
#include "ops/linear/projection/mxfp8/dispatch.h"
#include "ops/linear/kernels/fp8/native_a16.cuh"
#include "ops/linear/kernels/mxfp8/a8.cuh"
#include "ops/linear/kernels/mxfp8/a8_tma.cuh"
#include "ops/linear/kernels/fp8/native_a16_sliced_k.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail {
template <class S>
void mxfp8_linear_simt(const Tensor& x, const Weight& w, Tensor& out, Mxfp8A8Workspace,
                       cudaStream_t stream) {
    launch_native_fp8_a16_simt<S>(mxfp8_a16_operands(x, w),
                                  LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                                  LinearIdentityEpilogue{}, stream);
}

template <class S>
void mxfp8_linear_mma(const Tensor& x, const Weight& w, Tensor& out, Mxfp8A8Workspace,
                      cudaStream_t stream) {
    launch_native_fp8_a16_mma<S>(mxfp8_a16_operands(x, w),
                                 LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                                 LinearIdentityEpilogue{}, stream);
}

template <class S>
void mxfp8_linear_a8(const Tensor& x, const Weight& w, Tensor& out, Mxfp8A8Workspace workspace,
                     cudaStream_t stream) {
    launch_mxfp8_a8_quantize(static_cast<const __nv_bfloat16*>(x.data), w.k, x.ne[1], workspace,
                             stream);
    launch_mxfp8_a8_mma<S>(mxfp8_a8_operands(w, workspace, x.ne[1]),
                           LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                           LinearIdentityEpilogue{}, stream);
}

template <class S>
void mxfp8_linear_sliced(const Tensor& x, const Weight& w, Tensor& out, Mxfp8A8Workspace,
                         cudaStream_t stream) {
    launch_native_fp8_a16_sliced_k<S>(mxfp8_a16_operands(x, w),
                                      LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                                      LinearIdentityEpilogue{}, stream);
}

template <class S, bool Streaming = false>
void mxfp8_linear_tma(const Tensor& x, const Weight& w, Tensor& out, Mxfp8A8Workspace workspace,
                      cudaStream_t stream) {
    launch_mxfp8_a8_quantize(static_cast<const __nv_bfloat16*>(x.data), w.k, x.ne[1], workspace,
                             stream);
    using Epilogue =
        std::conditional_t<Streaming, Bf16MmaStreamingWarpStoreEpilogue, LinearIdentityEpilogue>;
    launch_mxfp8_a8_tma<S>(mxfp8_a8_operands(w, workspace, x.ne[1]),
                           LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n}, Epilogue{},
                           stream);
}
} // namespace ninfer::ops::detail
