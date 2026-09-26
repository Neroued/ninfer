#include "ops/linear/q4/q4_instance_launch.cuh"

namespace ninfer::ops::detail {

void launch_q4_a16_sliced_r16_t8_w4_s2(const Tensor& x, const Weight& w, Tensor& out,
                                       cudaStream_t stream) {
    launch_q4_a16_sliced_instance<q4_instances::SlicedR16T8W4S2>(x, w, out, stream);
}

void launch_q4_a16_sliced_r16_t8_capacity4(const Tensor& x, const Weight& w, Tensor& out,
                                           cudaStream_t stream) {
    launch_q4_a16_sliced_instance<q4_instances::SlicedR16T8Capacity4>(x, w, out, stream);
}

void launch_q4_a16_sliced_r16_t16_w4_s2(const Tensor& x, const Weight& w, Tensor& out,
                                        cudaStream_t stream) {
    launch_q4_a16_sliced_instance<q4_instances::SlicedR16T16W4S2>(x, w, out, stream);
}

void launch_q4_a16_sliced_r16_t16_w2_s2(const Tensor& x, const Weight& w, Tensor& out,
                                        cudaStream_t stream) {
    launch_q4_a16_sliced_instance<q4_instances::SlicedR16T16W2S2>(x, w, out, stream);
}

void launch_q4_a16_sliced_r16_t32_w4_s2(const Tensor& x, const Weight& w, Tensor& out,
                                        cudaStream_t stream) {
    launch_q4_a16_sliced_instance<q4_instances::SlicedR16T32W4S2>(x, w, out, stream);
}

void launch_q4_a16_sliced_r32_t8_w4_s2(const Tensor& x, const Weight& w, Tensor& out,
                                       cudaStream_t stream) {
    launch_q4_a16_sliced_instance<q4_instances::SlicedR32T8W4S2>(x, w, out, stream);
}

void launch_q4_a16_sliced_r32_t16_w4_s2(const Tensor& x, const Weight& w, Tensor& out,
                                        cudaStream_t stream) {
    launch_q4_a16_sliced_instance<q4_instances::SlicedR32T16W4S2>(x, w, out, stream);
}

void launch_q4_a16_sliced_r32_t16_w4_s1(const Tensor& x, const Weight& w, Tensor& out,
                                        cudaStream_t stream) {
    launch_q4_a16_sliced_instance<q4_instances::SlicedR32T16W4S1>(x, w, out, stream);
}

void launch_q4_a16_sliced_r32_t32_w4_s1(const Tensor& x, const Weight& w, Tensor& out,
                                        cudaStream_t stream) {
    launch_q4_a16_sliced_instance<q4_instances::SlicedR32T32W4S1>(x, w, out, stream);
}

void launch_q4_a16_sliced_r32_t32_w4_s2(const Tensor& x, const Weight& w, Tensor& out,
                                        cudaStream_t stream) {
    launch_q4_a16_sliced_instance<q4_instances::SlicedR32T32W4S2>(x, w, out, stream);
}

void launch_q4_a16_sliced_r32_t32_w2_s2(const Tensor& x, const Weight& w, Tensor& out,
                                        cudaStream_t stream) {
    launch_q4_a16_sliced_instance<q4_instances::SlicedR32T32W2S2>(x, w, out, stream);
}

void launch_q4_a16_sliced_r32_t64_w2_s1(const Tensor& x, const Weight& w, Tensor& out,
                                        cudaStream_t stream) {
    launch_q4_a16_sliced_instance<q4_instances::SlicedR32T64W2S1>(x, w, out, stream);
}

void launch_q4_a16_sliced_k5120_t16(const Tensor& x, const Weight& w, Tensor& out,
                                    cudaStream_t stream) {
    launch_q4_a16_sliced_instance<q4_instances::SlicedK5120T16>(x, w, out, stream);
}

void launch_q4_a16_sliced_k5120_t24(const Tensor& x, const Weight& w, Tensor& out,
                                    cudaStream_t stream) {
    launch_q4_a16_sliced_instance<q4_instances::SlicedK5120T24>(x, w, out, stream);
}

void launch_q4_a16_sliced_k5120_t4(const Tensor& x, const Weight& w, Tensor& out,
                                   cudaStream_t stream) {
    launch_q4_a16_sliced_instance<q4_instances::SlicedK5120T4>(x, w, out, stream);
}

void launch_q4_a16_sliced_k2048_t4(const Tensor& x, const Weight& w, Tensor& out,
                                   cudaStream_t stream) {
    launch_q4_a16_sliced_instance<q4_instances::SlicedK2048T4>(x, w, out, stream);
}

} // namespace ninfer::ops::detail
