#include "core/weight.h"
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

template <int ColsPerTile>
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
    const dim3 grid(static_cast<unsigned>(div_up(rows, kRowsPerBlock)),
                    static_cast<unsigned>(div_up(cols, ColsPerTile)), 1u);
    q5_rowsplit_gemm_simt_kernel<Q5RowSplitSimtSchedule, ColsPerTile, kRowsPerBlock, kStages>
        <<<grid, kThreads, 0, stream>>>(xp, static_cast<const std::uint8_t*>(w.qdata),
                                        static_cast<const std::uint8_t*>(w.qhigh),
                                        static_cast<const std::uint8_t*>(w.scales),
                                        static_cast<__nv_bfloat16*>(out.data), nullptr, rows,
                                        out_ld, k, cols, padded_k, full_slabs);
    CUDA_CHECK(cudaGetLastError());
}

template <int ColsPerTile>
void launch_simt_route(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    for_each_token_slice(x.ne[1], ColsPerTile, [&](std::int32_t offset, std::int32_t count) {
        const Tensor x_slice = x.slice(1, offset, count);
        Tensor out_slice     = out.slice(1, offset, count);
        launch_simt<ColsPerTile>(x_slice, w, out_slice, stream);
    });
}

// Exact one-column launch of the 4-warp split kernel. kStride names K itself, so the activation
// block of a row is contiguous and 16-byte aligned; the four warps split that block four ways
// instead of the single warp a tile kernel would give one column.
template <int K>
void launch_split4_c1(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    static_assert(K % 1024 == 0, "direct split4 needs whole 1024-wide K slabs");
    if (x.ne[1] != 1) { throw std::invalid_argument("q5 split4 c1: exact one-column launch"); }
    constexpr int kThreads = 4 * 32;
    const dim3 grid(static_cast<unsigned>(out.ne[0]), 1u, 1u);
    q5_rowsplit_gemm_simt_split4_kernel<Q5RowSplitSimtSchedule, 1, K / 1024, K>
        <<<grid, kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
            static_cast<__nv_bfloat16*>(out.data), nullptr, out.ne[0], out.ne[0], K, 1,
            w.padded_shape[1], K / 1024);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_q5_simt_r8_c4(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_simt_route<4>(x, w, out, stream);
}

void launch_q5_simt_r8_c8(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_simt_route<8>(x, w, out, stream);
}

void launch_q5_split4_c1_k5120(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_split4_c1<5120>(x, w, out, stream);
}

void launch_q5_split4_c1_k6144(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_split4_c1<6144>(x, w, out, stream);
}

void launch_q5_split4_c1_k17408(const Tensor& x, const Weight& w, Tensor& out,
                                cudaStream_t stream) {
    launch_split4_c1<17408>(x, w, out, stream);
}

} // namespace ninfer::ops::detail
