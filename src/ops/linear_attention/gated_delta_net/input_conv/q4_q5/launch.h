#pragma once

#include "ops/linear_attention/gated_delta_net/input_conv/operands.h"

#include "core/weight.h"
#include "core/tensor.h"
#include <cuda_runtime.h>

namespace ninfer::ops::detail {

enum class Q4Q5GdnInputScheduleId { Mma32x32, Mma32x64, Mma64x128 };

void q4_q5_gdn_input_dispatch(const Tensor& x, const Weight& qk_weight,
                              const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                              cudaStream_t stream);

void launch_q4_q5_gdn_input_independent(const Tensor& x, const Weight& qk_weight,
                                        const Weight& value_z_weight, Tensor& qk, Tensor& value,
                                        Tensor& z, cudaStream_t stream);

void launch_q4_q5_gdn_input_grouped_mma(const Tensor& x, const Weight& qk_weight,
                                        const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                                        Q4Q5GdnInputScheduleId schedule, cudaStream_t stream);

void launch_q4_q5_gdn_input_conv_snapshot(const Tensor& x, const Weight& qk_weight,
                                          const Weight& value_z_weight, const Tensor& conv_weight,
                                          const ConstStateFieldView& conv_states,
                                          const Tensor& valid_columns,
                                          const gdn_input::HistoryDestination& destination,
                                          Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                          cudaStream_t stream);

void launch_q4_q5_gdn_input_conv_record(const Tensor& x, const Weight& qk_weight,
                                        const Weight& value_z_weight, const Tensor& conv_weight,
                                        const ConstStateFieldView& conv_states,
                                        const Tensor& valid_columns, Tensor& conv_record,
                                        Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                        cudaStream_t stream);

} // namespace ninfer::ops::detail
