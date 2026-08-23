#pragma once

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <int RowsPerBranch, int IntermediateRows>
struct Fp8SwiGluRows {
    static_assert(RowsPerBranch > 0 && (RowsPerBranch & (RowsPerBranch - 1)) == 0);

    __device__ __forceinline__ int weight_row(int row_begin, int local_row) const {
        return row_begin + (local_row & (RowsPerBranch - 1)) +
               (local_row >= RowsPerBranch ? IntermediateRows : 0);
    }
};

union Fp8SwiGluBf16Pair {
    unsigned bits;
    __nv_bfloat162 values;
};

struct Fp8SwiGluOutput {
    __nv_bfloat16* data;
    std::int32_t rows;

    __device__ __forceinline__ unsigned combine(unsigned gate_bits, unsigned up_bits) const {
        Fp8SwiGluBf16Pair gate{gate_bits};
        Fp8SwiGluBf16Pair up{up_bits};
        const float2 gate_values = __bfloat1622float2(gate.values);
        const float2 up_values   = __bfloat1622float2(up.values);
        Fp8SwiGluBf16Pair result;
        result.values = __floats2bfloat162_rn(silu(gate_values.x) * up_values.x,
                                              silu(gate_values.y) * up_values.y);
        return result.bits;
    }

    __device__ __forceinline__ void store_pair_vector(std::int32_t row, std::int32_t token,
                                                      uint4 gate, uint4 up) const {
        const uint4 values = make_uint4(combine(gate.x, up.x), combine(gate.y, up.y),
                                        combine(gate.z, up.z), combine(gate.w, up.w));
        store_vec(data + static_cast<std::int64_t>(token) * rows + row, values);
    }

    __device__ __forceinline__ void store_pair(std::int32_t row, std::int32_t token, float gate,
                                               float up) const {
        data[static_cast<std::int64_t>(token) * rows + row] = __float2bfloat16_rn(silu(gate) * up);
    }
};

} // namespace ninfer::ops::detail
