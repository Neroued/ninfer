#include "ops/linear/q4/q4_instance_launch.cuh"

namespace ninfer::ops::detail {

void launch_q4_a16_simt_r4_t4_w2_g8_s2(const Tensor& x, const Weight& w, Tensor& out,
                                       cudaStream_t stream) {
    launch_q4_a16_simt_instance<q4_instances::SimtR4T4W2G8S2>(x, w, out, stream);
}

void launch_q4_a16_simt_r4_t1_w2_g8_s2(const Tensor& x, const Weight& w, Tensor& out,
                                       cudaStream_t stream) {
    launch_q4_a16_simt_instance<q4_instances::SimtR4T1W2G8S2>(x, w, out, stream);
}

} // namespace ninfer::ops::detail
