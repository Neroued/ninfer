#pragma once

#include "ops/linear_attention/gated_delta_net/input_conv/operands.h"

#include "core/weight.h"
#include "core/tensor.h"
#include "ops/linear/kernels/fp8/fp8_a8_plan.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t fp8_gdn_input_partial_capacity_bytes(std::int32_t max_tokens);

void launch_fp8_gdn_input_decode(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                 cudaStream_t stream);

void launch_fp8_gdn_input_matrix(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                 cudaStream_t stream);

void launch_fp8_gdn_input_a8(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                             Fp8A8Workspace workspace, cudaStream_t stream);

void launch_fp8_gdn_snapshot_fused(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                   const ConstStateFieldView& conv_states,
                                   const Tensor& valid_columns,
                                   const gdn_input::HistoryDestination& destination, Tensor& query,
                                   Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream);

void launch_fp8_gdn_record_fused(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                 const ConstStateFieldView& conv_states,
                                 const Tensor& valid_columns, Tensor& conv_record, Tensor& query,
                                 Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream);

} // namespace ninfer::ops::detail
