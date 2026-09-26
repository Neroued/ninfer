#include "core/weight.h"
#include "ops/attn_input_proj/q8/q8_attn_input_kernels.h"

#include "core/device.h"
#include "ops/linear/q8/q8_gemv_launch.cuh"

namespace ninfer::ops::detail {

namespace {

template <int RowsPerCta>
void launch_companion_decode(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                             cudaStream_t stream) {

    static_assert((4096 % RowsPerCta) == 0 && (1024 % RowsPerCta) == 0);
    using Output = LinearBf16SegmentedOutput<4096, 1024, 1024>;
    const Output output{static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
                        static_cast<__nv_bfloat16*>(v.data)};
    launch_q8_a16_gemv<Q8A16GemvSchedule<RowsPerCta, 1, 2, 2048>>(
        q8_linear_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void q8_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                 Tensor& k, Tensor& v, cudaStream_t stream) {

    constexpr int kRowsPerCta = 8;
    static_assert((4096 % kRowsPerCta) == 0 && (512 % kRowsPerCta) == 0);
    using Output = LinearBf16SegmentedOutput<4096, 512, 4096, 512>;
    const Output output{static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
                        static_cast<__nv_bfloat16*>(gate.data),
                        static_cast<__nv_bfloat16*>(v.data)};
    launch_q8_a16_gemv<Q8A16GemvSchedule<kRowsPerCta, 1, 2, 2048>>(
        q8_linear_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
    CUDA_CHECK(cudaGetLastError());
}

void q8_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                                 Tensor& v, cudaStream_t stream) {
    launch_companion_decode<8>(x, weight, q, k, v, stream);
}

void q8_companion_attn_input_decode_r4_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                              Tensor& k, Tensor& v, cudaStream_t stream) {
    launch_companion_decode<4>(x, weight, q, k, v, stream);
}

void q8_companion_attn_input_decode_r16_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                               Tensor& k, Tensor& v, cudaStream_t stream) {
    launch_companion_decode<16>(x, weight, q, k, v, stream);
}

} // namespace ninfer::ops::detail
