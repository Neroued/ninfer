#include "ops/linear_attention/gated_delta_net/input_conv/q8/convolution_window.cuh"
#include "ops/linear_attention/gated_delta_net/input_conv/q8/dgx_spark/launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry = GdnConvGeometry<3072, 1024>;
using Gemv     = Q8A16GemvSchedule<4, 2, 2, 2560, 4>;
} // namespace

void launch_q8_gdn_conv_n8192_k2560_snapshot(const Tensor& x, const Weight& weight,
                                             const Tensor& conv_weight,
                                             const ConstStateFieldView& states, const Tensor& valid,
                                             const gdn_input::HistoryDestination& destination,
                                             Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                             gdn_input::ConvKernel kernel, cudaStream_t stream) {
    gdn_q8_window::dispatch<Geometry, Gemv, 8, gdn_input::kQ8AggregateColumns>(
        x, weight, conv_weight, states, valid, query, key, value, z,
        make_snapshot_history_publish(destination, Geometry::channels),
        kernel == gdn_input::ConvKernel::Q8Aggregate, stream);
}

void launch_q8_gdn_conv_n8192_k2560_record(const Tensor& x, const Weight& weight,
                                           const Tensor& conv_weight,
                                           const ConstStateFieldView& states, const Tensor& valid,
                                           Tensor& records, Tensor& query, Tensor& key,
                                           Tensor& value, Tensor& z, gdn_input::ConvKernel kernel,
                                           cudaStream_t stream) {
    gdn_q8_window::dispatch<Geometry, Gemv, 8, gdn_input::kQ8AggregateColumns>(
        x, weight, conv_weight, states, valid, query, key, value, z,
        RecordColumnPublish{static_cast<__nv_bfloat16*>(records.data), Geometry::channels, x.ne[1]},
        kernel == gdn_input::ConvKernel::Q8Aggregate, stream);
}

} // namespace ninfer::ops::detail
