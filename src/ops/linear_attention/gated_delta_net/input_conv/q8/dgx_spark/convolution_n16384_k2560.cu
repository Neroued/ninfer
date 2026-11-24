#include "ops/linear_attention/gated_delta_net/input_conv/q8/convolution_window.cuh"
#include "ops/linear_attention/gated_delta_net/input_conv/q8/dgx_spark/launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Gemv = Q8A16GemvSchedule<8, 1, 2, 2560, 8>;
} // namespace

void launch_q8_gdn_conv_n16384_k2560_snapshot(const Tensor& x, const Weight& weight,
                                              const Tensor& conv_weight,
                                              const ConstStateFieldView& states,
                                              const Tensor& valid,
                                              const gdn_input::HistoryDestination& destination,
                                              Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                              gdn_input::ConvKernel kernel, cudaStream_t stream) {
    if (kernel == gdn_input::ConvKernel::Q8TmaBatch) {
        launch_q8_gdn_tma_n16384_k2560_snapshot(x, weight, conv_weight, states, valid, destination,
                                                query, key, value, z, stream);
        return;
    }
    gdn_q8_window::dispatch<ConvGeometry, Gemv>(
        x, weight, conv_weight, states, valid, query, key, value, z,
        make_snapshot_history_publish(destination, ConvGeometry::channels),
        kernel == gdn_input::ConvKernel::Q8Aggregate, stream);
}

void launch_q8_gdn_conv_n16384_k2560_record(const Tensor& x, const Weight& weight,
                                            const Tensor& conv_weight,
                                            const ConstStateFieldView& states, const Tensor& valid,
                                            Tensor& records, Tensor& query, Tensor& key,
                                            Tensor& value, Tensor& z, gdn_input::ConvKernel kernel,
                                            cudaStream_t stream) {
    if (kernel == gdn_input::ConvKernel::Q8TmaBatch) {
        launch_q8_gdn_tma_n16384_k2560_record(x, weight, conv_weight, states, valid, records, query,
                                              key, value, z, stream);
        return;
    }
    gdn_q8_window::dispatch<ConvGeometry, Gemv>(
        x, weight, conv_weight, states, valid, query, key, value, z,
        RecordColumnPublish{static_cast<__nv_bfloat16*>(records.data), ConvGeometry::channels,
                            x.ne[1]},
        kernel == gdn_input::ConvKernel::Q8Aggregate, stream);
}

} // namespace ninfer::ops::detail
