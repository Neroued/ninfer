#pragma once

#include "ninfer/core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::kernels::detail {

void gated_delta_rule_recurrent_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                       const Tensor& g, const Tensor& beta, float scale,
                                       Tensor& ssm_state, Tensor& out, cudaStream_t stream);

void gated_delta_rule_recurrent_bf16_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                            const Tensor& g, const Tensor& beta, float scale,
                                            Tensor& ssm_state, Tensor& out, cudaStream_t stream);

void gated_delta_rule_recurrent_inout_bf16_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                                  const Tensor& g, const Tensor& beta, float scale,
                                                  const Tensor& ssm_state_in, Tensor& ssm_state_out,
                                                  Tensor& out, cudaStream_t stream);

void gated_delta_rule_recurrent_snapshot_bf16_launch(const Tensor& q, const Tensor& k,
                                                     const Tensor& v, const Tensor& g,
                                                     const Tensor& beta, float scale,
                                                     Tensor& ssm_states,
                                                     const Tensor& initial_slot, Tensor& out,
                                                     cudaStream_t stream);

std::size_t gdn_chunked_workspace_bytes(std::int64_t T);

void gated_delta_rule_chunked_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                     const Tensor& g, const Tensor& beta, float scale,
                                     const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                                     void* workspace, std::size_t workspace_bytes,
                                     cudaStream_t stream);

} // namespace ninfer::kernels::detail
