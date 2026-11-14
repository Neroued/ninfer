#pragma once

#include "ninfer/ops/moe.h"
#include "ops/common/numeric/math.h"

namespace ninfer::ops::detail::moe_cuda {

struct TokenValidity {
    const std::int32_t* valid_columns = nullptr;
    common::FastDivisorU31 columns{1, 0, 1};
    std::int32_t token_begin = 0;
#ifdef __CUDACC__
    __device__ __forceinline__ bool contains(int token) const {
        if (!valid_columns) return true;
        const auto absolute = std::uint32_t(token_begin + token);
        const auto row      = columns.divide(absolute);
        return absolute - row * columns.divisor < std::uint32_t(valid_columns[row]);
    }
#endif
};

inline TokenValidity token_validity(const MoeTokenValidity& validity, int begin = 0) {
    return {static_cast<const std::int32_t*>(validity.valid_columns.data),
            common::FastDivisorU31::make(validity.valid_columns.data ? validity.tokens_per_row : 1),
            begin};
}

#ifdef __CUDACC__
__device__ __forceinline__ bool expert_in_subset(ExpertSubsetKind kind, const int* mask,
                                                 int expert) {
    return expert >= 0 && (kind == ExpertSubsetKind::All ||
                           (kind != ExpertSubsetKind::None &&
                            mask[expert] == (kind == ExpertSubsetKind::MaskOnes)));
}
#endif

template <int H, int I, int E, int K>
struct SwiGLUShape {
    static constexpr int hidden = H, intermediate = I, experts = E, top_k = K;
    static constexpr int control_rows = E + 1;
    static constexpr int score_rows   = (E + 4) / 4 * 4;
};

using SwiGLUH2048I512E256K8  = SwiGLUShape<2048, 512, 256, 8>;
using SwiGLUH2560I640E512K10 = SwiGLUShape<2560, 640, 512, 10>;

enum class ExpertImplementation { GroupwiseInt, BlockScaled };

inline void launch_zero(Tensor& destination, CudaExecutionView execution) {
    CUDA_CHECK(cudaMemsetAsync(destination.data, 0, destination.bytes(), execution.stream()));
}

} // namespace ninfer::ops::detail::moe_cuda
