#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/q5/q5_launch.h"
#include "ops/linear/q5/q5_rowsplit_gemm_simt.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kRowsPerBlock = 8;
constexpr int kStages       = 2;

template <int ColsPerTile, int ColWarpsPerRow = 1>
void launch_simt(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t rows     = out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t cols     = x.ne[1];
    const std::int32_t out_ld   = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    const std::int32_t padded_k = w.padded_shape[1];
    const auto* xp              = static_cast<const __nv_bfloat16*>(x.data);
    const bool aligned_x = (k % 8) == 0 && (reinterpret_cast<std::uintptr_t>(xp) & 0xfu) == 0;
    const std::int32_t full_slabs = aligned_x ? k / 1024 : 0;
    constexpr int kThreads        = kRowsPerBlock * 32;
    constexpr int kLogicalRowsPerBlock = kRowsPerBlock / ColWarpsPerRow;
    constexpr int kColsPerBlock        = ColsPerTile * ColWarpsPerRow;
    const dim3 grid(static_cast<unsigned>(div_up(rows, kLogicalRowsPerBlock)),
                    static_cast<unsigned>(div_up(cols, kColsPerBlock)), 1u);
    q5_rowsplit_gemm_simt_kernel<Q5RowSplitSimtSchedule, ColsPerTile, kRowsPerBlock, kStages, false,
                                 0, false, ColWarpsPerRow>
        <<<grid, kThreads, 0, stream>>>(xp, static_cast<const std::uint8_t*>(w.qdata),
                                        static_cast<const std::uint8_t*>(w.qhigh),
                                        static_cast<const std::uint8_t*>(w.scales),
                                        static_cast<__nv_bfloat16*>(out.data), nullptr, rows,
                                        out_ld, k, cols, padded_k, full_slabs);
    CUDA_CHECK(cudaGetLastError());
}

template <int ColsPerTile, int ColWarpsPerRow = 1>
void launch_simt_route(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    constexpr int kColsPerBlock = ColsPerTile * ColWarpsPerRow;
    for_each_token_slice(x.ne[1], kColsPerBlock, [&](std::int32_t offset, std::int32_t count) {
        const Tensor x_slice = x.slice(1, offset, count);
        Tensor out_slice     = out.slice(1, offset, count);
        launch_simt<ColsPerTile, ColWarpsPerRow>(x_slice, w, out_slice, stream);
    });
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
    const std::int32_t out_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    q5_rowsplit_gemm_simt_split4_kernel<Q5RowSplitSimtSchedule, Cols, 5, 5120>
        <<<grid, kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
            static_cast<__nv_bfloat16*>(out.data), nullptr, out.ne[0], out_ld, x.ne[0], x.ne[1],
            w.padded_shape[1], 5);
}

// Reachable range is [2,15]: q5_dispatch carries split2 to T=15 and split4 only to T=10, both
// measured crossovers. Cols 11..15 are therefore instantiated for split4 without being
// dispatchable -- the two share this switch. Splitting the dispatcher per family would drop those
// five kernels; not done because the shared form is what keeps the two routes' addressing
// provably identical.
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
    case 7:
        launch.template operator()<7>();
        return;
    case 8:
        launch.template operator()<8>();
        return;
    case 9:
        launch.template operator()<9>();
        return;
    case 10:
        launch.template operator()<10>();
        return;
    case 11:
        launch.template operator()<11>();
        return;
    case 12:
        launch.template operator()<12>();
        return;
    case 13:
        launch.template operator()<13>();
        return;
    case 14:
        launch.template operator()<14>();
        return;
    case 15:
        launch.template operator()<15>();
        return;
    default:
        throw std::invalid_argument("q5 exact SIMT: column count must be in [2,15]");
    }
}

} // namespace

void launch_q5_simt_r8_c4(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_simt_route<4>(x, w, out, stream);
}

void launch_q5_simt_r8_c8(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_simt_route<8>(x, w, out, stream);
}

void launch_q5_simt_r4_c16(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_simt_route<8, 2>(x, w, out, stream);
}

void launch_q5_simt_split2_exact(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream) {
    dispatch_exact_cols(x.ne[1],
                        [&]<int Cols>() { dispatch_split2_cols<Cols>(x, w, out, stream); });
    CUDA_CHECK(cudaGetLastError());
}

void launch_q5_simt_split4_exact(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream) {
    dispatch_exact_cols(x.ne[1], [&]<int Cols>() { launch_split4<Cols>(x, w, out, stream); });
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
