#pragma once

#include "ops/common/memory.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct Nvfp4IdentityEpilogue {
    __device__ __forceinline__ float apply(std::int32_t, std::int32_t, float value) const {
        return value;
    }
};

struct Nvfp4ContiguousOutput {
    __nv_bfloat16* data;
    std::int32_t rows;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        data[static_cast<std::int64_t>(token) * rows + parent_row] = __float2bfloat16_rn(value);
    }

    __device__ __forceinline__ void store_vector(std::int32_t parent_row, std::int32_t token,
                                                 uint4 values) const {
        auto* destination = data + static_cast<std::int64_t>(token) * rows + parent_row;
        store_vec(destination, values);
    }
};

// Writes the mma accumulator straight through, no BF16 round. Exists for split-projection SwiGLU
// (see nvfp4_linear_swiglu_qpn_split.cuh): two independent QPN2 launches -- one per weight half,
// unmodified, at whatever schedule QPN2 already measured fastest for this shape -- write gate and
// up into two fp32 scratch planes, and a small combine kernel applies silu(gate)*up in fp32 before
// the single BF16 round. Splitting keeps each launch at QPN2's own tuned register/occupancy
// profile instead of the fused kernel's doubled one; see the plan.cpp route for the measured
// comparison.
struct Nvfp4Fp32ContiguousOutput {
    float* data;
    std::int32_t rows;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        data[static_cast<std::int64_t>(token) * rows + parent_row] = value;
    }
};

} // namespace ninfer::ops::detail
