#pragma once

#include "ops/linear_attention/gated_delta_net/input_conv/operands.h"

#include "core/weight.h"
#include "core/tensor.h"
#include "ops/linear/kernels/nvfp4/nvfp4_a4_plan.h"
#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void launch_nvfp4_gdn_input_a16(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                cudaStream_t stream);

void launch_nvfp4_gdn_input_decode(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                   cudaStream_t stream);

void launch_nvfp4_gdn_input_small_t(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                    cudaStream_t stream);

void launch_nvfp4_gdn_input_a4(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                               Nvfp4A4Workspace workspace, cudaStream_t stream);

void launch_nvfp4_gdn_snapshot_decode(const Tensor& x, const Weight& weight,
                                      const Tensor& conv_weight,
                                      const ConstStateFieldView& conv_states,
                                      const Tensor& valid_columns,
                                      const gdn_input::HistoryDestination& destination,
                                      Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                      cudaStream_t stream);

void launch_nvfp4_gdn_snapshot_small_t(const Tensor& x, const Weight& weight,
                                       const Tensor& conv_weight,
                                       const ConstStateFieldView& conv_states,
                                       const Tensor& valid_columns,
                                       const gdn_input::HistoryDestination& destination,
                                       Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                       cudaStream_t stream);

void launch_nvfp4_gdn_record_small_t(const Tensor& x, const Weight& weight,
                                     const Tensor& conv_weight,
                                     const ConstStateFieldView& conv_states,
                                     const Tensor& valid_columns, Tensor& conv_record,
                                     Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                     cudaStream_t stream);

} // namespace ninfer::ops::detail
