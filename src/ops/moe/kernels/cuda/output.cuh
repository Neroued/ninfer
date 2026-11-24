#pragma once

#include "ops/moe/kernels/cuda/launch.h"
#include "ops/common/numeric/convert.cuh"

namespace ninfer::ops::detail::moe_cuda {

struct FloatContribution {
    float* data;

    __device__ __forceinline__ float initial(std::int64_t) const { return 0; }

    __device__ __forceinline__ void store(std::int64_t index, float value) const {
        data[index] = value;
    }
};

struct BfloatResult {
    __nv_bfloat16* data;
    bool add_residual;

    __device__ __forceinline__ float initial(std::int64_t index) const {
        return add_residual ? __bfloat162float(data[index]) : 0.0F;
    }

    __device__ __forceinline__ void store(std::int64_t index, float value) const {
        data[index] = __float2bfloat16_rn(value);
    }
};

template <class Function>
void with_output(Tensor& output, MoeEpilogue epilogue, Function&& function) {
    if (output.dtype == DType::FP32) {
        function(FloatContribution{static_cast<float*>(output.data)});
    } else {
        function(BfloatResult{static_cast<__nv_bfloat16*>(output.data),
                              epilogue == MoeEpilogue::AddResidual});
    }
}

} // namespace ninfer::ops::detail::moe_cuda
