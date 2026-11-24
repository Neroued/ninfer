#include "core/weight.h"
#include "ops/linear_attention/gated_delta_net/input_conv/q8/launch.h"
#include "core/device.h"
#include "ops/linear_attention/gated_delta_net/input_conv/kernels/output.cuh"
#include "ops/linear/kernels/q8/q8_gemv_launch.cuh"

namespace ninfer::ops::detail {
namespace {

using Output = LinearBf16SegmentedOutput<8192, 4096>;

} // namespace

void launch_q8_gdn_input_decode(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                cudaStream_t stream) {

    constexpr int kRowsPerCta = 8;
    static_assert((8192 % kRowsPerCta) == 0 && (4096 % kRowsPerCta) == 0);
    const Output output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)};
    launch_q8_a16_gemv<Q8A16GemvSchedule<kRowsPerCta, 1, 2, 2048>>(
        q8_linear_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
    CUDA_CHECK(cudaGetLastError());
}

void launch_q8_gdn_input_decode_conv_snapshot(const Tensor& x, const Weight& weight,
                                              const Tensor& conv_weight,
                                              const ConstStateFieldView& conv_states,
                                              const Tensor& valid_columns,
                                              const gdn_input::HistoryDestination& destination,
                                              Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                              cudaStream_t stream) {

    constexpr int kRowsPerCta = 8;
    const auto output         = make_gdn_conv_output<GdnConvGeometry<4096>, 1>(
        conv_weight, conv_states, valid_columns, query, key, value, z,
        make_snapshot_history_publish(destination, 8192));
    launch_q8_a16_gemv<Q8A16GemvSchedule<kRowsPerCta, 1, 2, 2048>>(
        q8_linear_operands(x, weight), output, GdnConvRowEpilogue<1>{}, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
