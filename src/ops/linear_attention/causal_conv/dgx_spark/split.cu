#include "ops/linear_attention/causal_conv/launch.h"
#include "ops/linear_attention/causal_conv/temporal.cuh"
#include "core/device.h"

#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
template <bool Publish>
void launch_temporal_split(const Tensor& x, const Tensor& weight, const Tensor& state_in,
                           Tensor& state_out, Tensor& query, Tensor& key, Tensor& value,
                           cudaStream_t stream) {
    constexpr int kChannels = 10240, kTokens = 2, kBlock = 128;
    const auto tiles  = (std::int64_t(x.ne[1]) + kTokens - 1) / kTokens;
    const auto blocks = tiles * (kChannels / 2 / kBlock);
    if (blocks > std::numeric_limits<int>::max())
        throw std::overflow_error("causal_conv1d: temporal split grid exceeds CUDA limit");
    const CausalConvSplitOutput3<__nv_bfloat162, 1024, 1024, 3072> output{
        static_cast<__nv_bfloat162*>(query.data), static_cast<__nv_bfloat162*>(key.data),
        static_cast<__nv_bfloat162*>(value.data)};
    causal_conv_temporal_kernel<kChannels, kTokens, kBlock, Publish>
        <<<static_cast<unsigned>(blocks), kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat162*>(x.data),
            static_cast<const __nv_bfloat162*>(weight.data),
            static_cast<const __nv_bfloat162*>(state_in.data),
            static_cast<__nv_bfloat162*>(state_out.data), output, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace

void launch_causal_conv_split_10240(const Tensor& x, const Tensor& weight, const Tensor& state_in,
                                    Tensor& state_out, Tensor& out0, Tensor& out1, Tensor& out2,
                                    cudaStream_t stream) {
    if (state_in.data != state_out.data) {
        launch_temporal_split<true>(x, weight, state_in, state_out, out0, out1, out2, stream);
    } else {
        launch_temporal_split<false>(x, weight, state_in, state_out, out0, out1, out2, stream);
        // Other token tiles can still be reading the old history in the first kernel.
        launch_causal_conv_state(x, state_in, state_out, stream);
    }
}
} // namespace ninfer::ops::detail
