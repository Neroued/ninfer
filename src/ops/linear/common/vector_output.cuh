#pragma once
#include "ops/linear/common/output.cuh"

namespace ninfer::ops::detail {
// A vector is already the final BF16 result. Output views without a vector store
// retain their scalar addressing, including strided or split destinations.
template <class Output>
__device__ __forceinline__ void linear_store_bf16_vector(Output output, int row, int token,
                                                         uint4 values) {
    if constexpr (requires { output.store_vector(row, token, values); }) {
        output.store_vector(row, token, values);
    } else {
        const unsigned words[]{values.x, values.y, values.z, values.w};
#pragma unroll
        for (int i = 0; i < 8; ++i)
            output.store(row + i, token,
                         __bfloat162float(__ushort_as_bfloat16(
                             static_cast<unsigned short>(words[i / 2] >> ((i & 1) * 16)))));
    }
}
} // namespace ninfer::ops::detail
