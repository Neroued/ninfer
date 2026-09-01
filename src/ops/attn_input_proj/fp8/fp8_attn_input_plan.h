#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/fp8/fp8_a8_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t fp8_attn_input_workspace_capacity_bytes(LinearPolicy policy,
                                                                  std::int32_t min_tokens,
                                                                  std::int32_t max_tokens);

void fp8_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                  Tensor& k, Tensor& v, cudaStream_t stream);

void fp8_attn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                   Tensor& k, Tensor& v, cudaStream_t stream);

void fp8_attn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                              Tensor& k, Tensor& v, Fp8A8Workspace workspace, cudaStream_t stream);

void fp8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                             Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                             cudaStream_t stream);

#ifdef NINFER_VOLTA_BUILD
void launch_fp8_attn_input_volta_qpn(const Tensor& x, const Weight& weight, Tensor& query,
                                     Tensor& gate, Tensor& key, Tensor& value,
                                     cudaStream_t stream);
#endif

} // namespace ninfer::ops::detail
