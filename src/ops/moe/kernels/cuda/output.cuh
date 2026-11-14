#pragma once

#include "ops/moe/kernels/cuda/launch.h"
#include "ops/common/numeric/convert.cuh"

namespace ninfer::ops::detail::moe_cuda {

struct FloatContribution {
    float* data;
    TokenValidity validity;

    __device__ __forceinline__ bool is_live(int token) const { return validity.contains(token); }

    __device__ __forceinline__ float initial(std::int64_t, bool) const { return 0; }

    __device__ __forceinline__ void store(std::int64_t index, float value, bool live) const {
        data[index] = live ? value : 0;
    }
};

struct BfloatResult {
    __nv_bfloat16* data;
    bool add_residual;
    TokenValidity validity;

    __device__ __forceinline__ bool is_live(int token) const { return validity.contains(token); }

    __device__ __forceinline__ float initial(std::int64_t index, bool live) const {
        if (!live) return 0;
        return add_residual ? __bfloat162float(data[index]) : 0.0F;
    }

    __device__ __forceinline__ void store(std::int64_t index, float value, bool live) const {
        if (!live) {
            if (!add_residual) data[index] = __float2bfloat16_rn(0);
            return;
        }
        data[index] = __float2bfloat16_rn(value);
    }
};

template <class Function>
void with_output(Tensor& output, MoeEpilogue epilogue, Function&& function,
                 TokenValidity validity = {}) {
    if (output.dtype == DType::FP32) {
        function(FloatContribution{static_cast<float*>(output.data), validity});
    } else {
        function(BfloatResult{static_cast<__nv_bfloat16*>(output.data),
                              epilogue == MoeEpilogue::AddResidual, validity});
    }
}

} // namespace ninfer::ops::detail::moe_cuda
