#include "ops/linear/q5/q5_instance_launch.cuh"

namespace ninfer::ops::detail {
void launch_q5_a16_mma_r32_t128(const Tensor& x, const Weight& w, Tensor& out,
                                cudaStream_t stream) {
    launch_q5_a16_mma_instance<q5_instances::MmaR32T128>(x, w, out, stream);
}

void launch_q5_a16_mma_r64_t128(const Tensor& x, const Weight& w, Tensor& out,
                                cudaStream_t stream) {
    launch_q5_a16_mma_instance<q5_instances::MmaR64T128>(x, w, out, stream);
}

void launch_q5_a16_mma_r64_t96_k128_s1_a1(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream) {
    launch_q5_a16_mma_instance<q5_instances::MmaR64T96K128S1A1>(x, w, out, stream);
}
} // namespace ninfer::ops::detail
