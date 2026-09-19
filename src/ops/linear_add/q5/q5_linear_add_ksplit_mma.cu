#include "core/weight.h"
#include "ops/linear_add/q5/q5_linear_add_kernels.h"

#include "core/device.h"
#include "ops/linear/q5/q5_ksplit_mma.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kRows = 5120;

// Capacity tiles 8/16/24/32 serve T <= 32 with one column tile; T in (32, 64] runs two
// 32-column tiles along blockIdx.y. The residual form reads and rewrites residual_out.
template <int InputRows>
void launch_shape(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    const std::int32_t cols = x.ne[1];
    if (cols <= 8) {
        launch_q5_ksplit_mma<kRows, InputRows, 8, 8, true>(x, w, residual_out, stream);
    } else if (cols <= 16) {
        launch_q5_ksplit_mma<kRows, InputRows, 16, 16, true>(x, w, residual_out, stream);
    } else if (cols <= 24) {
        launch_q5_ksplit_mma<kRows, InputRows, 24, 24, true>(x, w, residual_out, stream);
    } else if (cols <= 32) {
        launch_q5_ksplit_mma<kRows, InputRows, 32, 32, true>(x, w, residual_out, stream);
    } else if (cols <= 64) {
        launch_q5_ksplit_mma<kRows, InputRows, 32, 64, true>(x, w, residual_out, stream);
    } else {
        throw std::invalid_argument("q5 linear_add K-split MMA: T must be in [1,64]");
    }
}

} // namespace

void q5_linear_add_ksplit_mma_residual_launch(const Tensor& x, const Weight& w,
                                              Tensor& residual_out, cudaStream_t stream) {
    if (residual_out.ne[0] != kRows) {
        throw std::invalid_argument("q5 linear_add K-split MMA requires 5120 output rows");
    }
    if (w.k == 6144) {
        launch_shape<6144>(x, w, residual_out, stream);
    } else if (w.k == 17408) {
        launch_shape<17408>(x, w, residual_out, stream);
    } else {
        throw std::invalid_argument("q5 linear_add K-split MMA: unsupported exact K");
    }
}

} // namespace ninfer::ops::detail
