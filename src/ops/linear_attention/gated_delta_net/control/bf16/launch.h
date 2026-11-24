#pragma once

#include "ops/linear_attention/gated_delta_net/control/workspace.h"

namespace ninfer::ops::detail::gdn_control {

struct Inputs {
    const Tensor& x;
    const Tensor& a_log;
    const Tensor& dt_bias;
    const void* a;
    const void* b;
};

struct Norm {
    const Tensor& weight;
    float eps;
    Tensor normalized;
};

void launch_n96_k5120(const Inputs& in, GdnControlOutputs out, Plan plan, Scratch scratch,
                      CudaExecutionView execution);
void launch_n64_k2048(const Inputs& in, GdnControlOutputs out, Plan plan, Scratch scratch,
                      CudaExecutionView execution);
void launch_n96_k2560(const Inputs& in, GdnControlOutputs out, Plan plan, Scratch scratch,
                      CudaExecutionView execution);
void launch_n48_k2560(const Inputs& in, GdnControlOutputs out, Plan plan, Scratch scratch,
                      CudaExecutionView execution);
void launch_norm_n96_k5120(const Inputs& in, const Norm& norm, GdnControlOutputs out, NormPlan plan,
                           cudaStream_t stream);
bool launch_norm_n64_k2048(const Inputs& in, const Norm& norm, GdnControlOutputs out, NormPlan plan,
                           Scratch scratch, CudaExecutionView execution);

} // namespace ninfer::ops::detail::gdn_control
