#include "ops/linear_add/q8/q8_linear_add_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q8/q8_grouped_sliced_k_launch.cuh"

namespace ninfer::ops::detail {
namespace {

template <int K>
void launch(const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream) {
    constexpr int kColumns     = 128;
    constexpr int kSplits      = 2;
    constexpr int kTokenGroups = 4;
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(residual.data), 5120};
    launch_q8_a16_grouped_sliced_k_mma<Q8A16GroupedSlicedKMmaSchedule<
        kColumns, kSplits, kTokenGroups, 1, 1, K, Cache::cg, Cache::cg, true>>(
        q8_linear_operands(x, w), output, LinearResidualAddEpilogue{{output.data, output.rows, 0}},
        stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void q8_linear_add_grouped_launch(const Tensor& x, const Weight& w, Tensor& residual,
                                  cudaStream_t stream) {
    if (w.k == 6144) {
        launch<6144>(x, w, residual, stream);
    } else {
        launch<17408>(x, w, residual, stream);
    }
}

} // namespace ninfer::ops::detail
