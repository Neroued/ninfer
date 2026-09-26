#include "ops/linear/q5/q5_instance_launch.cuh"

namespace ninfer::ops::detail {
void launch_q5_a16_direct_r1_t1_w4_k17408(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream) {
    launch_q5_a16_direct_simt_instance<q5_instances::DirectR1T1W4K17408>(x, w, out, stream);
}

void launch_q5_a16_direct_r1_t1_w4_k5120(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream) {
    launch_q5_a16_direct_simt_instance<q5_instances::DirectR1T1W4K5120>(x, w, out, stream);
}

void launch_q5_a16_direct_r1_t1_w4_k6144(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream) {
    launch_q5_a16_direct_simt_instance<q5_instances::DirectR1T1W4K6144>(x, w, out, stream);
}

void launch_q5_a16_direct_r1_t2_w2_k17408(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream) {
    launch_q5_a16_direct_simt_instance<q5_instances::DirectR1T2W2K17408>(x, w, out, stream);
}

void launch_q5_a16_direct_r1_t2_w2_k6144(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream) {
    launch_q5_a16_direct_simt_instance<q5_instances::DirectR1T2W2K6144>(x, w, out, stream);
}

void launch_q5_a16_direct_r1_t2_w4_k5120(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream) {
    launch_q5_a16_direct_simt_instance<q5_instances::DirectR1T2W4K5120>(x, w, out, stream);
}

void launch_q5_a16_direct_r1_t3_w2_k17408(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream) {
    launch_q5_a16_direct_simt_instance<q5_instances::DirectR1T3W2K17408>(x, w, out, stream);
}

void launch_q5_a16_direct_r1_t3_w2_k6144(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream) {
    launch_q5_a16_direct_simt_instance<q5_instances::DirectR1T3W2K6144>(x, w, out, stream);
}

void launch_q5_a16_direct_r1_t3_w4_k5120(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream) {
    launch_q5_a16_direct_simt_instance<q5_instances::DirectR1T3W4K5120>(x, w, out, stream);
}

void launch_q5_a16_direct_r1_t4_w2_k17408(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream) {
    launch_q5_a16_direct_simt_instance<q5_instances::DirectR1T4W2K17408>(x, w, out, stream);
}

void launch_q5_a16_direct_r2_t4_w2_g8_b4(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream) {
    launch_q5_a16_direct_simt_instance<q5_instances::DirectR2T4W2G8B4>(x, w, out, stream);
}

void launch_q5_a16_direct_r2_t4_w4_g4_b4(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream) {
    launch_q5_a16_direct_simt_instance<q5_instances::DirectR2T4W4G4B4>(x, w, out, stream);
}
} // namespace ninfer::ops::detail
