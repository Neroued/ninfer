#pragma once
#include "ops/common/numeric/math.cuh"
#include "ops/linear/swiglu/token_major_mma_epilogue.cuh"

namespace ninfer::ops::detail {
struct SwiGluProduct {
    __device__ __forceinline__ float operator()(float gate, float up) const {
        return common::silu(gate) * up;
    }
};

struct ClampedSwiGluProduct {
    float limit;

    __device__ __forceinline__ float operator()(float gate, float up) const {
        return common::silu(fminf(gate, limit)) * fmaxf(-limit, fminf(up, limit));
    }
};


template <int RowsPerBranch, int IntermediateRows = 0>
struct SwiGluBranchRows {
    static_assert(RowsPerBranch > 0 && (RowsPerBranch & (RowsPerBranch - 1)) == 0);
    static constexpr bool kPaired = true;

    __device__ __forceinline__ int weight_row(int begin, int row, int rows) const {
        const int intermediate = IntermediateRows ? IntermediateRows : rows / 2;
        return begin + row % RowsPerBranch + (row >= RowsPerBranch ? intermediate : 0);
    }
};

struct SwiGluPairEpilogue {
    template <class Output>
    __device__ __forceinline__ void apply_pair(Output output, int row, int token, float gate,
                                               float up) const {
        output.store(row, token, common::silu(gate) * up);
    }
};
} // namespace ninfer::ops::detail
