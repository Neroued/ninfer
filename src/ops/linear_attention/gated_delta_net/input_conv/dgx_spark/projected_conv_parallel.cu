#include "ops/linear_attention/gated_delta_net/input_conv/kernels/projected_conv_parallel.cuh"

namespace ninfer::ops::detail {
void launch_gdn_projected_conv_parallel_snapshot(
    const Tensor& projected, const Tensor& conv_weight, const ConstStateFieldView& conv_states,
    const Tensor& valid_columns, const gdn_input::HistoryDestination& destination, Tensor& query,
    Tensor& key, Tensor& value, gdn_input::ProjectedConvKernel kernel, cudaStream_t stream) {
    const auto invoke = [&]<int Warps, class Geometry>() {
        gdn_projected_parallel::launch<Warps, Geometry>(
            projected, conv_weight, conv_states, valid_columns, query, key, value,
            make_snapshot_history_publish(destination, Geometry::channels), stream);
    };
    if (query.ne[0] == 1024) {
        invoke.template operator()<8, GdnConvGeometry<3072, 1024>>();
    } else if (kernel == gdn_input::ProjectedConvKernel::TokenParallel8)
        invoke.template operator()<8, ConvGeometry>();
    else
        invoke.template operator()<32, ConvGeometry>();
}

void launch_gdn_projected_conv_parallel_record(const Tensor& conv_record, const Tensor& conv_weight,
                                               const ConstStateFieldView& conv_states,
                                               const Tensor& valid_columns, Tensor& query,
                                               Tensor& key, Tensor& value,
                                               gdn_input::ProjectedConvKernel kernel,
                                               cudaStream_t stream) {
    const auto invoke = [&]<int Warps, class Geometry>() {
        gdn_projected_parallel::launch<Warps, Geometry>(conv_record, conv_weight, conv_states,
                                                        valid_columns, query, key, value,
                                                        NoHistoryPublish{}, stream);
    };
    if (query.ne[0] == 1024) {
        invoke.template operator()<8, GdnConvGeometry<3072, 1024>>();
    } else if (kernel == gdn_input::ProjectedConvKernel::TokenParallel8)
        invoke.template operator()<8, ConvGeometry>();
    else
        invoke.template operator()<32, ConvGeometry>();
}

} // namespace ninfer::ops::detail
