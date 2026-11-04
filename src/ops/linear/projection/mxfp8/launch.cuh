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

template <class S, bool Streaming = false, bool PaddedCodes = false>
void mxfp8_linear_tma(const Tensor& x, const Weight& w, Tensor& out, Mxfp8A8Workspace workspace,
                      cudaStream_t stream) {
    const int code_tokens = PaddedCodes ? mxfp8_a8_padded_tokens(x.ne[1], S::kBlockTokens) : x.ne[1];
    launch_mxfp8_a8_quantize<__nv_bfloat16, PaddedCodes>(
        static_cast<const __nv_bfloat16*>(x.data), w.k, x.ne[1], workspace, stream, code_tokens);
    using Epilogue =
        std::conditional_t<Streaming, Bf16MmaStreamingWarpStoreEpilogue, LinearIdentityEpilogue>;
    launch_mxfp8_a8_tma<S, PaddedCodes>(mxfp8_a8_operands(w, workspace, x.ne[1]),
                           LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n}, Epilogue{},
                           stream);
}

struct Mxfp8LinearStreamingVectorOutput : LinearBf16Output {
    __device__ __forceinline__ void store_vector(int row, int token, uint4 values) const {
        asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(at(row, token)),
                     "r"(values.x), "r"(values.y), "r"(values.z), "r"(values.w)
                     : "memory");
    }
};

struct Mxfp8LinearVectorEpilogue : Bf16TmaVectorEpilogue {
    template <class S, bool Full, class Output>
    __device__ __forceinline__ void finish_tile(const Output& output, unsigned char* scratch,
                                                float (&accum)[S::kMmaRows][S::kMmaTokens][4],
                                                int row_begin, int token_begin, int rows,
                                                int tokens) const {
        // MXFP8's Full specialization covers both axes of the output tile.
        Bf16TmaVectorEpilogue::template finish_tile<S, Full, !Full>(
            output, scratch, accum, row_begin, token_begin, rows, tokens);
    }
};

template <class S, bool PaddedCodes = false>
void mxfp8_linear_tma_vector(const Tensor& x, const Weight& w, Tensor& out,
                             Mxfp8A8Workspace workspace, cudaStream_t stream) {
    const int code_tokens = PaddedCodes ? mxfp8_a8_padded_tokens(x.ne[1], S::kBlockTokens) : x.ne[1];
    launch_mxfp8_a8_quantize<__nv_bfloat16, PaddedCodes>(
        static_cast<const __nv_bfloat16*>(x.data), w.k, x.ne[1], workspace, stream, code_tokens);
    launch_mxfp8_a8_tma<S, PaddedCodes>(
        mxfp8_a8_operands(w, workspace, x.ne[1]),
        Mxfp8LinearStreamingVectorOutput{{static_cast<__nv_bfloat16*>(out.data), w.n}},
        Mxfp8LinearVectorEpilogue{}, stream);
}

// The same schedule supplies the physical code extent to planning and launch.
template <class S, bool Vector = false>
constexpr Mxfp8LinearRoute mxfp8_linear_padded_tma_route(int last_tokens) {
    if constexpr (Vector)
        return {last_tokens, mxfp8_linear_tma_vector<S, true>, Mxfp8LinearCompute::A8,
                S::kBlockTokens};
    else
        return {last_tokens, mxfp8_linear_tma<S, true, true>, Mxfp8LinearCompute::A8,
                S::kBlockTokens};
}
} // namespace ninfer::ops::detail
