#pragma once

#include "ops/linear/common/output.cuh"

namespace ninfer::ops::detail {

// Thread-local operations over a fully reduced accumulator. The contraction
// owns predicates and synchronization; epilogues only receive valid coordinates.
struct LinearIdentityEpilogue {
    __device__ __forceinline__ float apply(int, int, float value) const { return value; }
};

struct LinearResidualAddEpilogue {
    LinearBf16InputView residual;

    __device__ __forceinline__ float apply(int row, int token, float value) const {
        return value + residual.load(row, token);
    }
};

} // namespace ninfer::ops::detail
