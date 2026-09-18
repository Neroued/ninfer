#include "ops/linear/q4/q4_gemv_launch.cuh"

namespace ninfer::ops::detail {

void launch_q4_gemv_r4_w1_direct(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream) {
    launch_q4_gemv<Q4GemvR4W1DirectSchedule>(x, w, out, stream);
}

void launch_q4_gemv_r1_q8_direct(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream) {
    launch_q4_gemv<Q4GemvR1Q8DirectSchedule>(x, w, out, stream);
}

} // namespace ninfer::ops::detail
