#pragma once

#include "ops/common/memory.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct Fp8IdentityEpilogue {
    __device__ __forceinline__ float apply(std::int32_t, std::int32_t, float value) const {
        return value;
    }
};

struct Fp8ContiguousOutput {
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

// Writes the epilogue's already-row-scaled value straight through, no BF16 round. Exists for
// split-projection SwiGLU (fp8_linear_swiglu_qpn_split.cuh), the FP8 sibling of
// Nvfp4Fp32ContiguousOutput: two independent QPN8 launches -- one per weight half, unmodified --
// write gate and up into fp32 scratch, and a small combine kernel applies silu(gate) * up in fp32
// before the single BF16 round. See nvfp4_output.cuh for why fp32 scratch rather than composing
// linear() + silu_mul(), and nvfp4_linear_swiglu_qpn_split.cuh for why splitting rather than
// fusing -- both apply unchanged to the FP8 case.
struct Fp8Fp32ContiguousOutput {
    float* data;
    std::int32_t rows;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        data[static_cast<std::int64_t>(token) * rows + parent_row] = value;
    }
};

} // namespace ninfer::ops::detail
