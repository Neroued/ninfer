#include "ops/dynamic_grouped_conv/prepare/launch.h"
#include "ops/dynamic_grouped_conv/prepare/tma.cuh"

namespace ninfer::ops::detail::dynamic_grouped_conv::prepare {
namespace {
template <int Hidden>
void launch_profile(const NormSource& source, const Tensor& normalized, const Tensor& base,
                    const Weight& projection, Tensor& prepared, Tensor& finish,
                    cudaStream_t stream) {
    const int tokens = normalized.ne[1] * normalized.ne[2];
    if (tokens <= (Hidden == 4096 ? 512 : 256))
        return launch_tma_fused<Hidden, 32, 64, 64, 16, 32, 3>(source, normalized, projection, base,
                                                               prepared, finish, stream);
    if (tokens <= 512)
        return launch_tma_fused<Hidden, 64, 64, 64, 32, 32, 3>(source, normalized, projection, base,
                                                               prepared, finish, stream);
    if (tokens <= 1024)
        return launch_tma_fused<Hidden, 64, 128, 64, 32, 32, 2>(source, normalized, projection,
                                                                base, prepared, finish, stream);
    launch_tma_fused<Hidden, 64, 128, 64, 32, 32, 2, true>(source, normalized, projection, base,
                                                           prepared, finish, stream);
}
} // namespace

void launch_fused(const NormSource& source, const Tensor& normalized, const Tensor& base,
                  const Weight& projection, Tensor& prepared, Tensor& finish, cudaStream_t stream) {
    if (normalized.ne[0] == 4096)
        launch_profile<4096>(source, normalized, base, projection, prepared, finish, stream);
    else
        launch_profile<5120>(source, normalized, base, projection, prepared, finish, stream);
}
} // namespace ninfer::ops::detail::dynamic_grouped_conv::prepare
