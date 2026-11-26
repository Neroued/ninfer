#pragma once

#include "ops/moe/kernels/cuda/block_scaled/clamped/nvfp4/compute.cuh"
#include "ops/moe/kernels/cuda/block_scaled/swiglu_activation.cuh"
#include "ops/moe/routed_swiglu/cuda/nvfp4/rtx5090/softcap.h"

namespace ninfer::ops::detail::moe_routed::softcap::tma {

namespace clamped = moe_cuda::block_scaled::clamped;

bool supports(const ExpertBankDeviceView&);
void contract(bool gate, Schedule schedule, const ExpertMatrixDeviceView&,
              clamped::nvfp4::ActivationScale input_scale,
              clamped::nvfp4::ActivationScale product_scale,
              moe_cuda::block_scaled::SoftcappedActivation, const clamped::Storage&,
              const float* coefficients, int jobs, int tokens, bool broadcast, CudaExecutionView);

} // namespace ninfer::ops::detail::moe_routed::softcap::tma
