#include "ops/linear/q5/q5_instance_launch.cuh"

namespace ninfer::ops::detail {
void launch_q5_a16_sliced_r16_t16_w2_s2(const Tensor& x, const Weight& w, Tensor& out,
                                        cudaStream_t stream) {
    launch_q5_a16_sliced_k_mma_instance<q5_instances::SlicedR16T16W2S2>(x, w, out, stream);
}

void launch_q5_a16_sliced_r16_t16_w4_s2(const Tensor& x, const Weight& w, Tensor& out,
                                        cudaStream_t stream) {
    launch_q5_a16_sliced_k_mma_instance<q5_instances::SlicedR16T16W4S2>(x, w, out, stream);
}

void launch_q5_a16_sliced_r16_t24_w4_s2(const Tensor& x, const Weight& w, Tensor& out,
                                        cudaStream_t stream) {
    launch_q5_a16_sliced_k_mma_instance<q5_instances::SlicedR16T24W4S2>(x, w, out, stream);
}

void launch_q5_a16_sliced_r16_t32_w4_s2(const Tensor& x, const Weight& w, Tensor& out,
                                        cudaStream_t stream) {
    launch_q5_a16_sliced_k_mma_instance<q5_instances::SlicedR16T32W4S2>(x, w, out, stream);
}

void launch_q5_a16_sliced_r16_t8_capacity4(const Tensor& x, const Weight& w, Tensor& out,
                                           cudaStream_t stream) {
    launch_q5_a16_sliced_k_mma_instance<q5_instances::SlicedR16T8Capacity4>(x, w, out, stream);
}

void launch_q5_a16_sliced_r16_t8_w4_s2(const Tensor& x, const Weight& w, Tensor& out,
                                       cudaStream_t stream) {
    launch_q5_a16_sliced_k_mma_instance<q5_instances::SlicedR16T8W4S2>(x, w, out, stream);
}

void launch_q5_a16_sliced_r32_t16_w4_s2(const Tensor& x, const Weight& w, Tensor& out,
                                        cudaStream_t stream) {
    launch_q5_a16_sliced_k_mma_instance<q5_instances::SlicedR32T16W4S2>(x, w, out, stream);
}

void launch_q5_a16_sliced_r32_t24_w4_s2_pairwise(const Tensor& x, const Weight& w, Tensor& out,
                                                 cudaStream_t stream) {
    launch_q5_a16_sliced_k_mma_instance<q5_instances::SlicedR32T24W4S2Pairwise>(x, w, out, stream);
}

void launch_q5_a16_sliced_r32_t32_w2_s2(const Tensor& x, const Weight& w, Tensor& out,
                                        cudaStream_t stream) {
    launch_q5_a16_sliced_k_mma_instance<q5_instances::SlicedR32T32W2S2>(x, w, out, stream);
}

void launch_q5_a16_sliced_r32_t32_w4_s1(const Tensor& x, const Weight& w, Tensor& out,
                                        cudaStream_t stream) {
    launch_q5_a16_sliced_k_mma_instance<q5_instances::SlicedR32T32W4S1>(x, w, out, stream);
}

void launch_q5_a16_sliced_r32_t32_w4_s2(const Tensor& x, const Weight& w, Tensor& out,
                                        cudaStream_t stream) {
    launch_q5_a16_sliced_k_mma_instance<q5_instances::SlicedR32T32W4S2>(x, w, out, stream);
}

void launch_q5_a16_sliced_r32_t64_w2_s1(const Tensor& x, const Weight& w, Tensor& out,
                                        cudaStream_t stream) {
    launch_q5_a16_sliced_k_mma_instance<q5_instances::SlicedR32T64W2S1>(x, w, out, stream);
}

void launch_q5_a16_sliced_r32_t64_w2_s2(const Tensor& x, const Weight& w, Tensor& out,
                                        cudaStream_t stream) {
    launch_q5_a16_sliced_k_mma_instance<q5_instances::SlicedR32T64W2S2>(x, w, out, stream);
}
} // namespace ninfer::ops::detail
