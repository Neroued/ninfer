#include "kernels/linear/gemv/linear_rowsplit_gemv_proj_6144.cuh"

#include "kernels/linear/gemv/linear_rowsplit_gemv_q5_core.cuh"
#include "qus/core/device.h" // CUDA_CHECK

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace qus::kernels::detail {
namespace {

constexpr int kN            = 6144;
constexpr int kK            = 5120; // kK/1024 = 5 tiles per row.
constexpr int kRowsPerBlock = 16;
constexpr int kStages       = 2;

} // namespace

void linear_rowsplit_gemv_proj_6144_q5_launch(const Tensor& x, const Weight& w, Tensor& out,
                                              WorkspaceArena& ws, cudaStream_t stream) {
    if (w.n != kN || w.k != kK || w.padded_shape[1] != kK) {
        throw std::invalid_argument("linear: Proj6144 Q5 tuned GEMV requires 6144x5120");
    }
    (void)ws;

    launch_q5_rowsplit_gemv<kN, kK, kRowsPerBlock, kStages, /*kStageX=*/true>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
