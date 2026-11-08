#pragma once
#include "ops/linear/projection/fp8_block/dispatch.h"
#include "ops/linear/kernels/fp8/native_a16.cuh"
#include "ops/linear/kernels/fp8/native_a16_sliced_k.cuh"
#include "ops/linear/kernels/fp8_block/operands.h"
#include "ops/linear/kernels/fp8_block/a16_tma.cuh"
#include "ops/linear/kernels/fp8_block/a16_split_tail.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail {
template <class S>
void fp8_block_linear_split_tail(const Tensor& x, const Weight& w, Tensor& out, MemorySpan backing,
                                 cudaStream_t stream) {
    launch_fp8_block_a16_split_tail<S>(
        fp8_block_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
        Bf16MmaStreamingWarpStoreEpilogue{},
        bind_fp8_block_a16_tail_workspace<typename S::TailPlan>(backing), stream);
}

template <class S>
void fp8_block_linear_tma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_fp8_block_a16_tma<S>(fp8_block_a16_operands(x, w),
                                LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                                Bf16MmaStreamingWarpStoreEpilogue{}, stream);
}

template <class S>
void fp8_block_linear_simt(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_native_fp8_a16_simt<S>(fp8_block_a16_operands(x, w),
                                  LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                                  LinearIdentityEpilogue{}, stream);
}

template <class S, bool Streaming = false>
void fp8_block_linear_mma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    using Epilogue =
        std::conditional_t<Streaming, Bf16MmaStreamingWarpStoreEpilogue, LinearIdentityEpilogue>;
    launch_native_fp8_a16_mma<S>(fp8_block_a16_operands(x, w),
                                 LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                                 Epilogue{}, stream);
}

template <class S>
void fp8_block_linear_sliced(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_native_fp8_a16_sliced_k<S>(fp8_block_a16_operands(x, w),
                                      LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                                      LinearIdentityEpilogue{}, stream);
}
} // namespace ninfer::ops::detail
