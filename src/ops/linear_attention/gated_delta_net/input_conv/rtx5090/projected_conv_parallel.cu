#include "ops/linear_attention/gated_delta_net/input_conv/kernels/projected_conv_parallel.cuh"

namespace ninfer::ops::detail {
void launch_gdn_projected_conv_parallel_snapshot(
    const Tensor& projected, const Tensor& conv_weight, const ConstStateFieldView& conv_states,
    const Tensor& valid_columns, const gdn_input::HistoryDestination& destination, Tensor& query,
    Tensor& key, Tensor& value, gdn_input::ProjectedConvKernel, cudaStream_t stream) {
    gdn_projected_parallel::launch<32>(
        projected, conv_weight, conv_states, valid_columns, query, key, value,
        make_snapshot_history_publish(destination, ConvGeometry::channels), stream);
}

void launch_gdn_projected_conv_parallel_record(const Tensor& conv_record, const Tensor& conv_weight,
                                               const ConstStateFieldView& conv_states,
                                               const Tensor& valid_columns, Tensor& query,
                                               Tensor& key, Tensor& value,
                                               gdn_input::ProjectedConvKernel,
                                               cudaStream_t stream) {
    gdn_projected_parallel::launch<32>(conv_record, conv_weight, conv_states, valid_columns, query,
                                       key, value, NoHistoryPublish{}, stream);
}

} // namespace ninfer::ops::detail
