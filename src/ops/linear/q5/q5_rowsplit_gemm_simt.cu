#include "ops/linear/q5/q5_rowsplit_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q5/q5_rowsplit_gemm_simt.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kRowsPerBlock = 8;
constexpr int kStages       = 2;

template <int ColsPerTile>
void launch_simt(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t rows     = out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t cols     = x.ne[1];
    const std::int32_t padded_k = w.padded_shape[1];
    const auto* xp              = static_cast<const __nv_bfloat16*>(x.data);
    const bool aligned_x = (k % 8) == 0 && (reinterpret_cast<std::uintptr_t>(xp) & 0xfu) == 0;
    const std::int32_t full_slabs = aligned_x ? k / 1024 : 0;
    constexpr int kThreads        = kRowsPerBlock * 32;
    const dim3 grid(static_cast<unsigned>(div_up(rows, kRowsPerBlock)),
                    static_cast<unsigned>(div_up(cols, ColsPerTile)), 1u);
    q5_rowsplit_gemm_simt_kernel<Q5RowSplitSimtSchedule, ColsPerTile, kRowsPerBlock, kStages>
        <<<grid, kThreads, 0, stream>>>(
            xp, static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
            static_cast<__nv_bfloat16*>(out.data), rows, k, cols, padded_k, full_slabs);
    CUDA_CHECK(cudaGetLastError());
}

template <int Cols, int FullSlabs, int Stride>
void launch_split2(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    constexpr int kThreads = 2 * 32;
    const dim3 grid(static_cast<unsigned>(out.ne[0]), 1u, 1u);
    q5_rowsplit_gemm_simt_split2_kernel<Q5RowSplitSimtSchedule, Cols, FullSlabs, Stride>
        <<<grid, kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
            static_cast<__nv_bfloat16*>(out.data), out.ne[0], x.ne[0], x.ne[1], w.padded_shape[1],
            FullSlabs);
}

template <int Cols>
void dispatch_split2_cols(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (w.k == 6144) {
        launch_split2<Cols, 6, 6144>(x, w, out, stream);
    } else if (w.k == 17408) {
        launch_split2<Cols, 17, 17408>(x, w, out, stream);
    } else {
        throw std::invalid_argument("q5 split2 SIMT: unsupported exact K");
    }
}

template <int Cols>
void launch_split4(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    constexpr int kThreads = 4 * 32;
    const dim3 grid(static_cast<unsigned>(out.ne[0]), 1u, 1u);
    q5_rowsplit_gemm_simt_split4_kernel<Q5RowSplitSimtSchedule, Cols, 5, 5120>
        <<<grid, kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
            static_cast<__nv_bfloat16*>(out.data), out.ne[0], x.ne[0], x.ne[1], w.padded_shape[1],
            5);
}

template <class Launch>
void dispatch_exact_cols(std::int32_t cols, Launch&& launch) {
    switch (cols) {
    case 2:
        launch.template operator()<2>();
        return;
    case 3:
        launch.template operator()<3>();
        return;
    case 4:
        launch.template operator()<4>();
        return;
    case 5:
        launch.template operator()<5>();
        return;
    case 6:
        launch.template operator()<6>();
        return;
    default:
        throw std::invalid_argument("q5 exact SIMT: column count must be in [2,6]");
    }
}

} // namespace

void q5_rowsplit_simt_r8_c4_launch(const Tensor& x, const Weight& w, Tensor& out,
                                   cudaStream_t stream) {
    launch_simt<4>(x, w, out, stream);
}

void q5_rowsplit_simt_r8_c8_launch(const Tensor& x, const Weight& w, Tensor& out,
                                   cudaStream_t stream) {
    launch_simt<8>(x, w, out, stream);
}

void q5_rowsplit_simt_split2_exact_launch(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream) {
    dispatch_exact_cols(x.ne[1],
                        [&]<int Cols>() { dispatch_split2_cols<Cols>(x, w, out, stream); });
    CUDA_CHECK(cudaGetLastError());
}

void q5_rowsplit_simt_split4_exact_launch(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream) {
    dispatch_exact_cols(x.ne[1], [&]<int Cols>() { launch_split4<Cols>(x, w, out, stream); });
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
