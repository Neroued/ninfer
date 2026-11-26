#pragma once

#include "ops/moe/routed_swiglu/cuda/fp8/rtx5090/softcap.h"
#include "ops/moe/kernels/cuda/block_scaled/swiglu_activation.cuh"

namespace ninfer::ops::detail::moe_routed::mxfp8_softcap {

using moe_cuda::block_scaled::SoftcappedActivation;

struct Matrix {
    const std::uint8_t* codes;
    const std::uint8_t* scales;
    unsigned physical;
};

__device__ __forceinline__ Matrix resolve(const ExpertMatrixDeviceView& bank, int expert) {
    const auto* codes  = reinterpret_cast<const std::uint8_t*>(bank.codes);
    const auto* scales = reinterpret_cast<const std::uint8_t*>(bank.scales);
    if (bank.offsets) {
        const auto offset = bank.offsets[expert];
        if (offset == kExpertAbsentOffset) return {};
        return {codes + offset, scales + offset, 0};
    }
    const unsigned slot = bank.expert_indices ? bank.expert_indices[expert] : unsigned(expert);
    if (slot >= bank.physical_count) return {};
    return {codes + slot * bank.code_stride, scales + slot * bank.scale_stride, slot};
}

void launch_mma(const __nv_bfloat16* x, const ExpertBankDeviceView&, const float* coefficients,
                SoftcappedActivation, Plan, const Storage&, int tokens, CudaExecutionView);

} // namespace ninfer::ops::detail::moe_routed::mxfp8_softcap
