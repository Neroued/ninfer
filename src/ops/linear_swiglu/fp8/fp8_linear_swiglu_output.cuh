#pragma once
#include "ops/common/math.cuh"
#include "ops/linear/common/output.cuh"
#include "ops/linear_swiglu/token_major_mma_epilogue.cuh"

namespace ninfer::ops::detail {
template <int RowsPerBranch, int IntermediateRows>
struct Fp8SwiGluRows {
    static_assert(RowsPerBranch > 0 && (RowsPerBranch & (RowsPerBranch - 1)) == 0);
    static constexpr bool kPaired = true;

    __device__ __forceinline__ int weight_row(int begin, int row, int) const {
        return begin + row % RowsPerBranch + (row >= RowsPerBranch ? IntermediateRows : 0);
    }
};

struct Fp8SwiGluEpilogue {
    template <class Output>
    __device__ __forceinline__ void apply_pair(Output output, int row, int token, float gate,
                                               float up) const {
        output.store(row, token, silu(gate) * up);
    }
};
} // namespace ninfer::ops::detail
