#pragma once

#include "ops/linear_attention/causal_conv/plan.h"
#include <cuda_runtime.h>

namespace ninfer::ops::detail {
void launch_causal_conv(const Tensor& x, const Tensor& weight, const Tensor& state_in,
                        Tensor& state_out, Tensor& out, CausalConvRoute route, cudaStream_t stream);
void launch_causal_conv_split(const Tensor& x, const Tensor& weight, const Tensor& state_in,
                              Tensor& state_out, Tensor& out0, Tensor& out1, Tensor& out2,
                              CausalConvSplitGeometry geometry, cudaStream_t stream);
void launch_causal_conv_state(const Tensor& x, const Tensor& state_in, Tensor& state_out,
                              cudaStream_t stream);
void launch_causal_conv_split_10240(const Tensor& x, const Tensor& weight, const Tensor& state_in,
                                    Tensor& state_out, Tensor& out0, Tensor& out1, Tensor& out2,
                                    cudaStream_t stream);
void launch_causal_conv_split_5120(const Tensor& x, const Tensor& weight, const Tensor& state_in,
                                   Tensor& state_out, Tensor& out0, Tensor& out1, Tensor& out2,
                                   cudaStream_t stream);
void launch_causal_conv_snapshot(const Tensor& x, const Tensor& weight, Tensor& states,
                                 const Tensor& valid, const Tensor& initial_slots,
                                 const Tensor& snapshot_slots, Tensor& out,
                                 CausalConvSnapshotPlan plan, cudaStream_t stream);
} // namespace ninfer::ops::detail
