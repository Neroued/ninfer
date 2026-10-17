#pragma once

#include "ops/linear_attention/gated_delta_net/input_conv/plan.h"

#include "core/weight.h"
#include "core/tensor.h"
#include <cuda_runtime.h>

namespace ninfer::ops::detail {
void q8_gdn_projection_n16384_k2560(const Tensor&, const Weight&, Tensor&, Tensor&, cudaStream_t);

void q8_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                           cudaStream_t stream);

void launch_q8_gdn_conv_n16384_k2560_snapshot(const Tensor& x, const Weight& weight,
                                              const Tensor& conv_weight, Tensor& states,
                                              const Tensor& valid, const Tensor& initial,
                                              const Tensor& bases, Tensor& query, Tensor& key,
                                              Tensor& value, Tensor& z,
                                              gdn_input::ConvKernel kernel, cudaStream_t stream);
void launch_q8_gdn_conv_n16384_k2560_record(const Tensor& x, const Weight& weight,
                                            const Tensor& conv_weight, const Tensor& states,
                                            const Tensor& valid, const Tensor& initial,
                                            Tensor& records, Tensor& query, Tensor& key,
                                            Tensor& value, Tensor& z, gdn_input::ConvKernel kernel,
                                            cudaStream_t stream);

void launch_q8_gdn_input_decode(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                cudaStream_t stream);
void launch_q8_gdn_input_decode_conv_snapshot(
    const Tensor& x, const Weight& weight, const Tensor& conv_weight, Tensor& conv_states,
    const Tensor& valid_columns, const Tensor& initial_slot, const Tensor& snapshot_base_slot,
    Tensor& query, Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream);
void launch_q8_gdn_input_splitk_mma(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                    cudaStream_t stream);
void launch_q8_gdn_input_splitk_conv_snapshot(
    const Tensor& x, const Weight& weight, const Tensor& conv_weight, Tensor& conv_states,
    const Tensor& valid_columns, const Tensor& initial_slot, const Tensor& snapshot_base_slot,
    Tensor& query, Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream);
void launch_q8_gdn_input_splitk_conv_record(const Tensor& x, const Weight& weight,
                                            const Tensor& conv_weight, const Tensor& conv_states,
                                            const Tensor& valid_columns, const Tensor& initial_slot,
                                            Tensor& conv_record, Tensor& query, Tensor& key,
                                            Tensor& value, Tensor& z, cudaStream_t stream);
void launch_q8_gdn_input_mma_r64_c128(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                      cudaStream_t stream);

} // namespace ninfer::ops::detail
