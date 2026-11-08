#pragma once

#include "ops/linear/kernels/fp8/native_a16.cuh"
#include "ops/linear/kernels/mxfp8/a8.cuh"
#include "ops/linear/kernels/mxfp8/a8_tma.cuh"
#include "ops/linear/kernels/fp8/native_a16_sliced_k.cuh"
#include "ops/linear/swiglu/product_epilogue.cuh"
#include "ops/linear/swiglu/row_major_bf16_epilogue.cuh"

namespace ninfer::ops::detail {

template <class S>
void mxfp8_swiglu_simt(const Tensor& x, const Weight& weight, Tensor& out, float limit,
                       bool clamped, Mxfp8A8Workspace, cudaStream_t stream) {
    const auto launch = [&]<class Product>(Product product) {
        launch_native_fp8_a16_simt<S>(
            mxfp8_a16_operands(x, weight),
            LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n / 2},
            BasicSwiGluPairEpilogue<Product>{product}, stream,
            SwiGluBranchRows<native_fp8_pair_across_warps<S> ? S::kRowsPerWarp
                                                             : S::kRowsPerWarp / 2>{});
    };
    if (clamped)
        launch(ClampedSwiGluProduct{limit});
    else
        launch(SwiGluProduct{});
}

template <class S>
void mxfp8_swiglu_a16(const Tensor& x, const Weight& weight, Tensor& out, float limit, bool clamped,
                      Mxfp8A8Workspace, cudaStream_t stream) {
    const auto launch = [&]<class Product>(Product product) {
        launch_native_fp8_a16_mma<S>(
            mxfp8_a16_operands(x, weight),
            LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n / 2},
            BasicSwiGluInterleavedMmaEpilogue<Product>{product}, stream,
            SwiGluInterleavedRows<8>{});
    };
    if (clamped)
        launch(ClampedSwiGluProduct{limit});
    else
        launch(SwiGluProduct{});
}

template <class S>
void mxfp8_swiglu_a8(const Tensor& x, const Weight& weight, Tensor& out, float limit, bool clamped,
                     Mxfp8A8Workspace scratch, cudaStream_t stream) {
    launch_mxfp8_a8_quantize(static_cast<const __nv_bfloat16*>(x.data), weight.k, x.ne[1], scratch,
                             stream);
    const auto launch = [&]<class Product>(Product product) {
        launch_mxfp8_a8_mma<S>(
            mxfp8_a8_operands(weight, scratch, x.ne[1]),
            LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n / 2},
            BasicSwiGluInterleavedMmaEpilogue<Product>{product}, stream,
            SwiGluInterleavedRows<8>{});
    };
    if (clamped)
        launch(ClampedSwiGluProduct{limit});
    else
        launch(SwiGluProduct{});
}

template <class S>
void mxfp8_swiglu_sliced(const Tensor& x, const Weight& weight, Tensor& out, float limit,
                         bool clamped, Mxfp8A8Workspace, cudaStream_t stream) {
    const auto launch = [&]<class Product>(Product product) {
        launch_native_fp8_a16_sliced_k<S>(
            mxfp8_a16_operands(x, weight),
            LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n / 2},
            BasicSwiGluPairEpilogue<Product>{product}, stream,
            SwiGluBranchRows<S::kBlockRows / 2>{});
    };
    if (clamped)
        launch(ClampedSwiGluProduct{limit});
    else
        launch(SwiGluProduct{});
}

template <class S, bool PaddedCodes = false,
          template <class> class Epilogue = BasicSwiGluInterleavedMmaEpilogue,
          class Rows                      = SwiGluInterleavedRows<8>>
void mxfp8_swiglu_tma(const Tensor& x, const Weight& weight, Tensor& out, float limit, bool clamped,
                      Mxfp8A8Workspace scratch, cudaStream_t stream) {
    const int code_tokens =
        PaddedCodes ? mxfp8_a8_padded_tokens(x.ne[1], S::kBlockTokens) : x.ne[1];
    launch_mxfp8_a8_quantize<__nv_bfloat16, PaddedCodes>(
        static_cast<const __nv_bfloat16*>(x.data), weight.k, x.ne[1], scratch, stream, code_tokens);
    const auto launch = [&]<class Product>(Product product) {
        launch_mxfp8_a8_tma<S, PaddedCodes>(
            mxfp8_a8_operands(weight, scratch, x.ne[1]),
            LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n / 2},
            Epilogue<Product>{product}, stream, Rows{});
    };
    if (clamped)
        launch(ClampedSwiGluProduct{limit});
    else
        launch(SwiGluProduct{});
}

template <class S, int OutputRows, template <class> class Epilogue = BasicSwiGluRowMajorMmaEpilogue,
          class Rows = SwiGluTmaBranchRows<S::kBlockRows / 2>>
void mxfp8_swiglu_padded_tma(const Tensor& x, const Weight& weight, Tensor& out, float limit,
                             bool clamped, Mxfp8A8Workspace scratch, cudaStream_t stream) {
    static_assert(2 * OutputRows % S::kBlockRows == 0);
    const int padded = mxfp8_a8_padded_tokens(x.ne[1], S::kBlockTokens);
    launch_mxfp8_a8_quantize_padded(static_cast<const __nv_bfloat16*>(x.data), weight.k, x.ne[1],
                                    scratch, stream, padded);
    const auto launch = [&]<class Product>(Product product) {
        launch_mxfp8_a8_tma<S>(
            mxfp8_a8_operands(weight, scratch, padded),
            LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), OutputRows},
            BasicSwiGluPaddedBf16Epilogue<Product, OutputRows, Epilogue>{product, int(x.ne[1])},
            stream, Rows{});
    };
    if (clamped)
        launch(ClampedSwiGluProduct{limit});
    else
        launch(SwiGluProduct{});
}

template <class S, int OutputRows>
void mxfp8_swiglu_padded_mma(const Tensor& x, const Weight& weight, Tensor& out, float limit,
                             bool clamped, Mxfp8A8Workspace scratch, cudaStream_t stream) {
    static_assert(2 * OutputRows % S::kBlockRows == 0);
    const int padded = mxfp8_a8_padded_tokens(x.ne[1], S::kBlockTokens);
    launch_mxfp8_a8_quantize_padded(static_cast<const __nv_bfloat16*>(x.data), weight.k, x.ne[1],
                                    scratch, stream, padded);
    const auto launch = [&]<class Product>(Product product) {
        launch_mxfp8_a8_mma<S>(
            mxfp8_a8_operands(weight, scratch, padded),
            LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), OutputRows},
            BasicSwiGluPaddedBf16Epilogue<Product, OutputRows, BasicSwiGluInterleavedMmaEpilogue>{
                product, int(x.ne[1])},
            stream, SwiGluInterleavedRows<8>{});
    };
    if (clamped)
        launch(ClampedSwiGluProduct{limit});
    else
        launch(SwiGluProduct{});
}

} // namespace ninfer::ops::detail
