#include "ops/linear/q4/q4_instance_launch.cuh"

namespace ninfer::ops::detail {

void launch_q4_a16_gemv_r4_w1_direct(const Tensor& x, const Weight& w, Tensor& out,
                                     cudaStream_t stream) {
    launch_q4_a16_gemv_instance<q4_instances::GemvR4W1>(x, w, out, stream);
}

void launch_q4_a16_gemv_r1_w8_direct(const Tensor& x, const Weight& w, Tensor& out,
                                     cudaStream_t stream) {
    launch_q4_a16_gemv_instance<q4_instances::GemvR1W8K5120>(x, w, out, stream);
}

void launch_q4_a16_gemv_r1_w8_k6144(const Tensor& x, const Weight& w, Tensor& out,
                                    cudaStream_t stream) {
    launch_q4_a16_gemv_instance<q4_instances::GemvR1W8K6144>(x, w, out, stream);
}

} // namespace ninfer::ops::detail
