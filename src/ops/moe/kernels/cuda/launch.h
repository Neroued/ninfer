#pragma once

#include "ninfer/ops/moe.h"

namespace ninfer::ops::detail::moe_cuda {

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
