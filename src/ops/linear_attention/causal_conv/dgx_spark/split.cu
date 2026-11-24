#include "ops/linear_attention/causal_conv/launch.h"
#include "ops/linear_attention/causal_conv/temporal.cuh"
#include "core/device.h"

#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
template <int KeyRows, int ValueRows, int Block, bool Publish, bool Indexed>
void launch_temporal_split(const Tensor& x, const Tensor& weight, const Tensor& state_in,
                           Tensor& state_out, Tensor& query, Tensor& key, Tensor& value,
                           cudaStream_t stream, CausalConvOffsets offsets) {
    constexpr int kChannels = 2 * KeyRows + ValueRows, kTokens = 2;
    constexpr int kChannelBlocks = (kChannels / 2 + Block - 1) / Block;
    const auto blocks            = (std::int64_t(x.ne[1]) + kTokens - 1) / kTokens * kChannelBlocks;
    if (blocks > std::numeric_limits<int>::max())
        throw std::overflow_error("causal_conv1d: temporal split grid exceeds CUDA limit");
    const CausalConvSplitOutput3<__nv_bfloat162, KeyRows / 2, KeyRows / 2, ValueRows / 2> output{
        static_cast<__nv_bfloat162*>(query.data), static_cast<__nv_bfloat162*>(key.data),
        static_cast<__nv_bfloat162*>(value.data)};
    causal_conv_temporal_kernel<kChannels, kTokens, Block, Publish, Indexed>
        <<<static_cast<unsigned>(blocks), Block, 0, stream>>>(
            static_cast<const __nv_bfloat162*>(x.data),
            static_cast<const __nv_bfloat162*>(weight.data),
            static_cast<const __nv_bfloat162*>(state_in.data),
            static_cast<__nv_bfloat162*>(state_out.data), output, x.ne[1], offsets.source);
    CUDA_CHECK(cudaGetLastError());
}

template <int KeyRows, int ValueRows, int Block>
void launch_split(const Tensor& x, const Tensor& weight, const Tensor& state_in, Tensor& state_out,
                  Tensor& out0, Tensor& out1, Tensor& out2, cudaStream_t stream,
                  CausalConvOffsets offsets) {
    if (offsets.source) {
        launch_temporal_split<KeyRows, ValueRows, Block, false, true>(
            x, weight, state_in, state_out, out0, out1, out2, stream, offsets);
        launch_causal_conv_state(x, state_in, state_out, stream, offsets);
    } else if (state_in.data != state_out.data) {
        launch_temporal_split<KeyRows, ValueRows, Block, true, false>(
            x, weight, state_in, state_out, out0, out1, out2, stream, offsets);
    } else {
        launch_temporal_split<KeyRows, ValueRows, Block, false, false>(
            x, weight, state_in, state_out, out0, out1, out2, stream, offsets);
        // Other token tiles can still be reading the old history in the first kernel.
        launch_causal_conv_state(x, state_in, state_out, stream);
    }
}
} // namespace

void launch_causal_conv_split_5120(const Tensor& x, const Tensor& weight, const Tensor& state_in,
                                   Tensor& state_out, Tensor& out0, Tensor& out1, Tensor& out2,
                                   cudaStream_t stream, CausalConvOffsets offsets) {
    if (x.ne[1] <= 1024)
        launch_split<1024, 3072, 64>(x, weight, state_in, state_out, out0, out1, out2, stream,
                                     offsets);
    else
        launch_split<1024, 3072, 128>(x, weight, state_in, state_out, out0, out1, out2, stream,
                                      offsets);
}

void launch_causal_conv_split_10240(const Tensor& x, const Tensor& weight, const Tensor& state_in,
                                    Tensor& state_out, Tensor& out0, Tensor& out1, Tensor& out2,
                                    cudaStream_t stream, CausalConvOffsets offsets) {
    launch_split<2048, 6144, 128>(x, weight, state_in, state_out, out0, out1, out2, stream,
                                  offsets);
}
} // namespace ninfer::ops::detail
