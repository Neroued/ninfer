#pragma once

#include "ops/linear_attention/gated_delta_net/input_conv/plan.h"

#include "core/tensor.h"
#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void launch_gdn_projected_conv_snapshot(const Tensor& projected, const Tensor& conv_weight,
                                        Tensor& conv_states, const Tensor& valid_columns,
                                        const Tensor& initial_state_slots,
                                        const Tensor& snapshot_base_slots, Tensor& query,
                                        Tensor& key, Tensor& value, cudaStream_t stream);

void launch_gdn_projected_conv_record(const Tensor& conv_record, const Tensor& conv_weight,
                                      const Tensor& conv_states, const Tensor& valid_columns,
                                      const Tensor& initial_state_slots, Tensor& query, Tensor& key,
                                      Tensor& value, cudaStream_t stream);

void launch_gdn_projected_conv_parallel_snapshot(const Tensor& projected, const Tensor& conv_weight,
                                                 Tensor& conv_states, const Tensor& valid_columns,
                                                 const Tensor& initial_state_slots,
                                                 const Tensor& snapshot_base_slots, Tensor& query,
                                                 Tensor& key, Tensor& value,
                                                 gdn_input::ProjectedConvKernel kernel,
                                                 cudaStream_t stream);

void launch_gdn_projected_conv_parallel_record(
    const Tensor& conv_record, const Tensor& conv_weight, const Tensor& conv_states,
    const Tensor& valid_columns, const Tensor& initial_state_slots, Tensor& query, Tensor& key,
    Tensor& value, gdn_input::ProjectedConvKernel kernel, cudaStream_t stream);

} // namespace ninfer::ops::detail
