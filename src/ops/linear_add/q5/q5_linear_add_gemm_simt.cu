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
        throw std::invalid_argument("q5 linear_add split2: T must be in [2,16]");
    }
}

} // namespace

void q5_linear_add_split2_exact_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream) {
    dispatch_cols(x.ne[1], [&]<int Cols>() { dispatch_shape<Cols>(x, w, residual_out, stream); });
    CUDA_CHECK(cudaGetLastError());
}

// Volta fallback for wide T (the MmaResidualR64C* schedules need Ampere+ mma/ldmatrix, trap-
// stubbed on sm_70): q5_rowsplit_gemm_simt_kernel already takes cols as a runtime grid
// parameter (unlike split2's compile-time Cols switch above), so a single instantiation covers
// any T. AddResidual reuses the exact accumulate-before-store pattern split2 already validated.
// See docs/v100.md.
void q5_linear_add_simt_wide_t_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    constexpr int kColsPerTile  = 8;
    constexpr int kRowsPerBlock = 8;
    constexpr int kStages       = 2;
    constexpr int kThreads      = kRowsPerBlock * 32;
    const std::int32_t rows     = residual_out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t cols     = x.ne[1];
    const std::int32_t out_ld   = static_cast<std::int32_t>(residual_out.nb[1] / sizeof(__nv_bfloat16));
    const dim3 grid(static_cast<unsigned>(div_up(rows, kRowsPerBlock)),
                    static_cast<unsigned>(div_up(cols, kColsPerTile)), 1u);
    // full_slabs>0 enables the staged/vectorized prefetch path instead of routing every group
    // through the scalar tail (direct global reads). Every offset q5_simt_consume_slab derives
    // from x0 (xslab = slab*1024 elems, c*256 elems, lane*8 elems) is a multiple of 16 bytes in
    // bf16 units, and k (6144 or 17408, this op's only two supported shapes) is itself a
    // multiple of 1024, so a single runtime check on x.data's own alignment is sufficient to
    // guarantee every subsequent load_vec<uint4> stays 16-byte aligned -- see
    // q5_rowsplit_gemm_simt.cuh's q5_simt_consume_slab. Falls back to the always-correct
    // full_slabs=0 scalar tail if that check fails.
    const bool staged_safe = (k % 1024) == 0 &&
                             (reinterpret_cast<std::uintptr_t>(x.data) % 16) == 0;
    const std::int32_t full_slabs = staged_safe ? (k / 1024) : 0;
    q5_rowsplit_gemm_simt_kernel<Q5RowSplitSimtSchedule, kColsPerTile, kRowsPerBlock, kStages,
                                 false, 0, true><<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
        static_cast<__nv_bfloat16*>(residual_out.data), nullptr, rows, out_ld, k, cols,
        w.padded_shape[1], full_slabs);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
