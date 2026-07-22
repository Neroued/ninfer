#include "ops/gdn_input_proj/w8/w8_gdn_input_kernels.h"

#include "core/device.h"
#include "ops/gdn_input_proj/gdn_conv_snapshot.cuh"
#include "ops/linear/w8/w8_k2048_decode.cuh"

namespace ninfer::ops::detail {
namespace {

using Output = W8SplitOutput2<8192, 4096>;

struct W8GdnDecodeConvEpilogue {
    GdnConvSnapshotEpilogue conv;
    __nv_bfloat16* z;

    template <class IgnoredOutput>
    __device__ __forceinline__ void operator()(const IgnoredOutput&, std::int32_t, std::int32_t row,
                                               float accumulator) const {
        if (row < 8192) {
            const float projected[1]{accumulator};
            conv.store(row, projected);
        } else {
            z[row - 8192] = __float2bfloat16_rn(accumulator);
        }
    }
};

GdnConvSnapshotEpilogue make_conv_epilogue(const Tensor& conv_weight, Tensor& conv_states,
                                           const Tensor& initial_slot, Tensor& query, Tensor& key,
                                           Tensor& value) {
    return {
        static_cast<const __nv_bfloat16*>(conv_weight.data),
        static_cast<__nv_bfloat16*>(conv_states.data),
        static_cast<const std::int32_t*>(initial_slot.data),
        static_cast<__nv_bfloat16*>(query.data),
        static_cast<__nv_bfloat16*>(key.data),
        static_cast<__nv_bfloat16*>(value.data),
        8192,
        2048,
        2048,
        4096,
        0,
    };
}

} // namespace

void w8_gdn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                cudaStream_t stream) {
    constexpr int kRows       = 12288;
    constexpr int kRowsPerCta = 8;
    static_assert((8192 % kRowsPerCta) == 0 && (4096 % kRowsPerCta) == 0);
    const Output output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)};
    w8_k2048_decode_kernel<kRows, kRowsPerCta>
        <<<kRows / kRowsPerCta, kRowsPerCta * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output);
    CUDA_CHECK(cudaGetLastError());
}

void w8_gdn_input_decode_conv_snapshot_launch(const Tensor& x, const Weight& weight,
                                              const Tensor& conv_weight, Tensor& conv_states,
                                              const Tensor& initial_slot, Tensor& query,
                                              Tensor& key, Tensor& value, Tensor& z,
                                              cudaStream_t stream) {
    constexpr int kRows       = 12288;
    constexpr int kRowsPerCta = 8;
    const Output ignored_output{static_cast<__nv_bfloat16*>(query.data),
                                static_cast<__nv_bfloat16*>(z.data)};
    const W8GdnDecodeConvEpilogue epilogue{
        make_conv_epilogue(conv_weight, conv_states, initial_slot, query, key, value),
        static_cast<__nv_bfloat16*>(z.data),
    };
    w8_k2048_decode_kernel<kRows, kRowsPerCta, Output, W8GdnDecodeConvEpilogue>
        <<<kRows / kRowsPerCta, kRowsPerCta * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), ignored_output, epilogue);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
