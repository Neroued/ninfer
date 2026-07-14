#include "kernels/linear/gemv/linear_rowsplit_gemv_mlp_down.cuh"

#include "kernels/linear/gemv/linear_rowsplit_gemv_q5_core.cuh"
#include "ninfer/core/device.h" // CUDA_CHECK

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::kernels::detail {
namespace {

constexpr int kN            = 5120;
constexpr int kK            = 17408; // kK/1024 = 17 tiles per row.
constexpr int kRowsPerBlock = 16;
constexpr int kStages       = 2;

} // namespace

// x (K=17408 -> 34 KiB) is too large to stage into shared alongside the weight
// buffers, so this shape reads x from global (kStageX=false) and relies on the deep
// 17-tile, 3-stage cp.async pipeline + L2-resident x to hide latency.
void linear_rowsplit_gemv_mlp_down_q5_launch(const Tensor& x, const Weight& w, Tensor& out,
                                             WorkspaceArena& ws, cudaStream_t stream) {
    if (w.n != kN || w.k != kK || w.padded_shape[1] != kK) {
        throw std::invalid_argument("linear: MLP down Q5 tuned GEMV requires 5120x17408");
    }
    (void)ws;

    launch_q5_rowsplit_gemv<kN, kK, kRowsPerBlock, kStages, /*kStageX=*/false>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), stream);
    CUDA_CHECK(cudaGetLastError());
}

void linear_rowsplit_gemv_mlp_down_residual_q5_launch(const Tensor& x, const Weight& w,
                                                      Tensor& residual_out, WorkspaceArena& ws,
                                                      cudaStream_t stream) {
    if (w.n != kN || w.k != kK || w.padded_shape[1] != kK) {
        throw std::invalid_argument("linear_add: MLP down Q5 requires 5120x17408");
    }
    (void)ws;

    launch_q5_rowsplit_gemv_residual<kN, kK, kRowsPerBlock, kStages, /*kStageX=*/false>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
        static_cast<const __nv_bfloat16*>(residual_out.data),
        static_cast<__nv_bfloat16*>(residual_out.data), stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::kernels::detail
