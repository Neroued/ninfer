#include "ops/attn_input_proj/w8/w8_attn_input_kernels.h"

#include "core/device.h"
#include "ops/linear/w8/w8_k2048_decode.cuh"

namespace ninfer::ops::detail {

namespace {

template <int RowsPerCta>
void launch_companion_decode(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                             cudaStream_t stream) {
    constexpr int kRows = 6144;
    static_assert((4096 % RowsPerCta) == 0 && (1024 % RowsPerCta) == 0);
    using Output = W8SplitOutput3<4096, 1024, 1024>;
    const Output output{static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
                        static_cast<__nv_bfloat16*>(v.data)};
    w8_k2048_decode_kernel<kRows, RowsPerCta><<<kRows / RowsPerCta, RowsPerCta * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), output);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void w8_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                 Tensor& k, Tensor& v, cudaStream_t stream) {
    constexpr int kRows       = 9216;
    constexpr int kRowsPerCta = 8;
    static_assert((4096 % kRowsPerCta) == 0 && (512 % kRowsPerCta) == 0);
    using Output = W8SplitOutput4<4096, 512, 4096, 512>;
    const Output output{static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
                        static_cast<__nv_bfloat16*>(gate.data),
                        static_cast<__nv_bfloat16*>(v.data)};
    w8_k2048_decode_kernel<kRows, kRowsPerCta>
        <<<kRows / kRowsPerCta, kRowsPerCta * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output);
    CUDA_CHECK(cudaGetLastError());
}

void w8_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                                 Tensor& v, cudaStream_t stream) {
    launch_companion_decode<8>(x, weight, q, k, v, stream);
}

void w8_companion_attn_input_decode_r4_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                              Tensor& k, Tensor& v, cudaStream_t stream) {
    launch_companion_decode<4>(x, weight, q, k, v, stream);
}

void w8_companion_attn_input_decode_r16_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                               Tensor& k, Tensor& v, cudaStream_t stream) {
    launch_companion_decode<16>(x, weight, q, k, v, stream);
}

} // namespace ninfer::ops::detail
