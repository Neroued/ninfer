#include "ops/linear_add/q5/q5_linear_add_kernels.h"
#include "ops/linear/q5/q5_instances.cuh"
#include "ops/linear/q5/q5_sliced_k_launch.cuh"

namespace ninfer::ops::detail {
namespace {
template <class Schedule>
void launch_sliced(const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream) {
    auto* data        = static_cast<__nv_bfloat16*>(residual.data);
    const auto stride = std::int64_t(residual.nb[1] / sizeof(__nv_bfloat16));
    launch_q5_a16_sliced_k_mma<Schedule>(q5_linear_operands(x, w),
                                         LinearBf16StridedOutput{data, stride, 0},
                                         LinearResidualAddEpilogue{{data, stride, 0}}, stream);
}
} // namespace

void q5_linear_add_sliced_r16_t8_launch(const Tensor& x, const Weight& w, Tensor& residual,
                                        cudaStream_t stream) {
    launch_sliced<q5_instances::SlicedR16T8W4S2>(x, w, residual, stream);
}

void q5_linear_add_sliced_r16_t16_launch(const Tensor& x, const Weight& w, Tensor& residual,
                                         cudaStream_t stream) {
    launch_sliced<q5_instances::SlicedR16T16W4S2>(x, w, residual, stream);
}

void q5_linear_add_sliced_r16_t24_launch(const Tensor& x, const Weight& w, Tensor& residual,
                                         cudaStream_t stream) {
    launch_sliced<q5_instances::SlicedR16T24W4S2>(x, w, residual, stream);
}

void q5_linear_add_sliced_r32_t32_w4_s2_launch(const Tensor& x, const Weight& w, Tensor& residual,
                                               cudaStream_t stream) {
    launch_sliced<q5_instances::SlicedR32T32W4S2>(x, w, residual, stream);
}

void q5_linear_add_sliced_r32_t24_pairwise_launch(const Tensor& x, const Weight& w,
                                                  Tensor& residual, cudaStream_t stream) {
    launch_sliced<q5_instances::SlicedR32T24W4S2Pairwise>(x, w, residual, stream);
}

void q5_linear_add_sliced_r32_t32_w4_s1_launch(const Tensor& x, const Weight& w, Tensor& residual,
                                               cudaStream_t stream) {
    launch_sliced<q5_instances::SlicedR32T32W4S1>(x, w, residual, stream);
}

void q5_linear_add_sliced_r32_t32_w2_s2_launch(const Tensor& x, const Weight& w, Tensor& residual,
                                               cudaStream_t stream) {
    launch_sliced<q5_instances::SlicedR32T32W2S2>(x, w, residual, stream);
}

void q5_linear_add_sliced_r32_t64_w2_s1_launch(const Tensor& x, const Weight& w, Tensor& residual,
                                               cudaStream_t stream) {
    launch_sliced<q5_instances::SlicedR32T64W2S1>(x, w, residual, stream);
}
} // namespace ninfer::ops::detail
