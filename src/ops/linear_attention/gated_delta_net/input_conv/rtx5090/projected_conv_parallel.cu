#include "ops/linear_attention/gated_delta_net/input_conv/kernels/projected_conv_parallel.cuh"

namespace ninfer::ops::detail {
void launch_gdn_projected_conv_parallel_snapshot(const Tensor& projected, const Tensor& conv_weight,
                                                 Tensor& conv_states, const Tensor& valid_columns,
                                                 const Tensor& initial_slot,
                                                 const Tensor& snapshot_base_slot, Tensor& query,
                                                 Tensor& key, Tensor& value,
                                                 gdn_input::ProjectedConvKernel,
                                                 cudaStream_t stream) {
    gdn_projected_parallel::launch<32>(
        projected, conv_weight, conv_states, valid_columns, initial_slot, query, key, value,
        SnapshotHistoryPublish{static_cast<__nv_bfloat16*>(conv_states.data), conv_states.nb[2] / 2,
                               static_cast<const std::int32_t*>(snapshot_base_slot.data),
                               ConvGeometry::channels},
        stream);
}

void launch_gdn_projected_conv_parallel_record(
    const Tensor& conv_record, const Tensor& conv_weight, const Tensor& conv_states,
    const Tensor& valid_columns, const Tensor& initial_slot, Tensor& query, Tensor& key,
    Tensor& value, gdn_input::ProjectedConvKernel, cudaStream_t stream) {
    gdn_projected_parallel::launch<32>(conv_record, conv_weight, conv_states, valid_columns,
                                       initial_slot, query, key, value, NoHistoryPublish{}, stream);
}

} // namespace ninfer::ops::detail
