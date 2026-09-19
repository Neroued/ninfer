#include "core/weight.h"
#include "ops/linear_add/q5/q5_linear_add_kernels.h"

#include "core/device.h"
#include "ops/linear/q5/q5_rowsplit_gemm_simt.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <int Cols, int FullSlabs, int Stride>
void launch_split2(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    constexpr int kThreads = 2 * 32;
    const dim3 grid(static_cast<unsigned>(residual_out.ne[0]), 1u, 1u);
    q5_rowsplit_gemm_simt_split2_kernel<Q5RowSplitSimtSchedule, Cols, FullSlabs, Stride, false, 0,
                                        true><<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
        static_cast<__nv_bfloat16*>(residual_out.data), residual_out.ne[0], x.ne[0], x.ne[1],
        w.padded_shape[1], FullSlabs);
}

template <int Cols>
void dispatch_shape(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    if (w.k == 6144) {
        launch_split2<Cols, 6, 6144>(x, w, residual_out, stream);
    } else if (w.k == 17408) {
        launch_split2<Cols, 17, 17408>(x, w, residual_out, stream);
    } else {
        throw std::invalid_argument("q5 linear_add split2: unsupported exact K");
    }
}

template <class Launch>
void dispatch_cols(std::int32_t cols, Launch&& launch) {
    switch (cols) {
#define NINFER_Q5_LINEAR_ADD_EXACT(COLS)                                                           \
    case COLS:                                                                                     \
        launch.template operator()<COLS>();                                                        \
        return
        NINFER_Q5_LINEAR_ADD_EXACT(1);
        NINFER_Q5_LINEAR_ADD_EXACT(2);
        NINFER_Q5_LINEAR_ADD_EXACT(3);
        NINFER_Q5_LINEAR_ADD_EXACT(4);
        NINFER_Q5_LINEAR_ADD_EXACT(5);
        NINFER_Q5_LINEAR_ADD_EXACT(6);
        NINFER_Q5_LINEAR_ADD_EXACT(7);
        NINFER_Q5_LINEAR_ADD_EXACT(8);
        NINFER_Q5_LINEAR_ADD_EXACT(9);
        NINFER_Q5_LINEAR_ADD_EXACT(10);
        NINFER_Q5_LINEAR_ADD_EXACT(11);
        NINFER_Q5_LINEAR_ADD_EXACT(12);
        NINFER_Q5_LINEAR_ADD_EXACT(13);
        NINFER_Q5_LINEAR_ADD_EXACT(14);
        NINFER_Q5_LINEAR_ADD_EXACT(15);
        NINFER_Q5_LINEAR_ADD_EXACT(16);
#undef NINFER_Q5_LINEAR_ADD_EXACT
    default:
        throw std::invalid_argument("q5 linear_add split2: T must be in [1,16]");
    }
}

} // namespace

void q5_linear_add_split2_exact_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream) {
    dispatch_cols(x.ne[1], [&]<int Cols>() { dispatch_shape<Cols>(x, w, residual_out, stream); });
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
