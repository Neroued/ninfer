// ninfer::ops - causal_conv1d launcher: grid/block/stream configuration + kernel launch.
#include "ops/launcher/causal_conv1d.h"

#include "ops/common/math.h"
#include "ops/kernel/causal_conv1d.cuh"
#include "core/device.h" // CUDA_CHECK

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

int grid_for(std::int64_t n, int block, const char* label) {
    const std::int64_t grid = div_up(n, static_cast<std::int64_t>(block));
    if (grid > std::numeric_limits<int>::max()) {
        throw std::overflow_error(std::string("causal_conv1d: ") + label +
                                  " grid exceeds CUDA limit");
    }
    return static_cast<int>(std::max<std::int64_t>(1, grid));
}

int prefill_output_grid_for(std::int32_t C, std::int32_t T, int block) {
    const std::int64_t c_blocks =
        div_up(static_cast<std::int64_t>(C), static_cast<std::int64_t>(block));
    const std::int64_t grid = c_blocks * static_cast<std::int64_t>(T);
    if (grid > std::numeric_limits<int>::max()) {
        throw std::overflow_error("causal_conv1d: prefill output grid exceeds CUDA limit");
    }
    return static_cast<int>(std::max<std::int64_t>(1, grid));
}

} // namespace

void causal_conv1d_prefill_launch(const Tensor& x, const Tensor& weight,
                                  const Tensor& conv_state_in, Tensor& conv_state_out, Tensor& out,
                                  cudaStream_t stream) {
    constexpr int kOutputBlock  = 256;
    constexpr int kChannelBlock = 256;
    constexpr int kPairBlock    = 256;
    const std::int32_t C        = x.ne[0];
    const std::int32_t T        = x.ne[1];
    const auto x_addr           = reinterpret_cast<std::uintptr_t>(x.data);
    const auto w_addr           = reinterpret_cast<std::uintptr_t>(weight.data);
    const auto in_addr          = reinterpret_cast<std::uintptr_t>(conv_state_in.data);
    const auto out_state_addr   = reinterpret_cast<std::uintptr_t>(conv_state_out.data);
    const auto out_addr         = reinterpret_cast<std::uintptr_t>(out.data);

    if (((x_addr | w_addr | in_addr | out_state_addr | out_addr) & (alignof(__nv_bfloat162) - 1)) ==
            0 &&
        (C & 1) == 0) {
        causal_conv1d_prefill_pairs_kernel<<<prefill_output_grid_for(C / 2, T, kPairBlock),
                                             kPairBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.data),
            static_cast<const __nv_bfloat16*>(conv_state_in.data),
            static_cast<__nv_bfloat16*>(out.data), C, T);
        CUDA_CHECK(cudaGetLastError());
    } else {
        causal_conv1d_prefill_kernel<<<prefill_output_grid_for(C, T, kOutputBlock), kOutputBlock, 0,
                                       stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.data),
            static_cast<const __nv_bfloat16*>(conv_state_in.data),
            static_cast<__nv_bfloat16*>(out.data), C, T);
        CUDA_CHECK(cudaGetLastError());
    }

    causal_conv1d_prefill_state_kernel<<<grid_for(C, kChannelBlock, "prefill state"), kChannelBlock,
                                         0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(conv_state_in.data),
        static_cast<__nv_bfloat16*>(conv_state_out.data), C, T);
    CUDA_CHECK(cudaGetLastError());
}

void causal_conv1d_smallt_launch(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                                 Tensor& conv_state_out, Tensor& out, cudaStream_t stream) {
    const std::int32_t C = x.ne[0];
    const std::int32_t T = x.ne[1];
    const dim3 block(kCausalConvChannelTile, static_cast<unsigned int>(T));
    const int grid = grid_for(C, kCausalConvChannelTile, "small-T");

    causal_conv1d_smallt_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.data),
        static_cast<const __nv_bfloat16*>(conv_state_in.data),
        static_cast<__nv_bfloat16*>(conv_state_out.data), static_cast<__nv_bfloat16*>(out.data), C,
        T);
    CUDA_CHECK(cudaGetLastError());
}

void causal_conv1d_sequence_launch(const Tensor& x, const Tensor& weight,
                                   const Tensor& conv_state_in, Tensor& conv_state_out, Tensor& out,
                                   cudaStream_t stream) {
    const std::int32_t C = x.ne[0];
    const std::int32_t T = x.ne[1];
    const int block      = T == 1 ? 256 : 32;

    causal_conv1d_sequence_kernel<<<grid_for(C, block, "sequence"), block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.data),
        static_cast<const __nv_bfloat16*>(conv_state_in.data),
        static_cast<__nv_bfloat16*>(conv_state_out.data), static_cast<__nv_bfloat16*>(out.data), C,
        T);
    CUDA_CHECK(cudaGetLastError());
}

void causal_conv1d_decode_launch(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                                 Tensor& conv_state_out, Tensor& out, cudaStream_t stream) {
    constexpr int kBlock = 256;
    const std::int32_t C = x.ne[0];

    if (conv_state_in.data == conv_state_out.data) {
        causal_conv1d_decode_kernel<<<grid_for(C, kBlock, "decode"), kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.data),
            static_cast<__nv_bfloat16*>(conv_state_out.data), static_cast<__nv_bfloat16*>(out.data),
            C);
    } else {
        causal_conv1d_decode_distinct_kernel<<<grid_for(C, kBlock, "distinct decode"), kBlock, 0,
                                               stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.data),
            static_cast<const __nv_bfloat16*>(conv_state_in.data),
            static_cast<__nv_bfloat16*>(conv_state_out.data), static_cast<__nv_bfloat16*>(out.data),
            C);
    }
    CUDA_CHECK(cudaGetLastError());
}

void causal_conv1d_sequence_snapshot_launch(const Tensor& x, const Tensor& weight,
                                            Tensor& conv_states, const Tensor& initial_slot,
                                            Tensor& out, cudaStream_t stream) {
    const std::int32_t C = x.ne[0];
    const std::int32_t T = x.ne[1];
    const std::int64_t slot_stride =
        static_cast<std::int64_t>(conv_states.ne[0]) * static_cast<std::int64_t>(conv_states.ne[1]);

    if (T == 1) {
        constexpr int kBlock = 256;
        causal_conv1d_snapshot_decode_kernel<<<grid_for(C, kBlock, "snapshot decode"), kBlock, 0,
                                               stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.data),
            static_cast<__nv_bfloat16*>(conv_states.data),
            static_cast<const std::int32_t*>(initial_slot.data),
            static_cast<__nv_bfloat16*>(out.data), C, slot_stride);
    } else if (T <= kCausalConvParallelMaxTokens) {
        const dim3 block(kCausalConvChannelTile, static_cast<unsigned int>(T));
        const int grid = grid_for(C, kCausalConvChannelTile, "small-T snapshot");
        causal_conv1d_snapshot_smallt_kernel<<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.data),
            static_cast<__nv_bfloat16*>(conv_states.data),
            static_cast<const std::int32_t*>(initial_slot.data),
            static_cast<__nv_bfloat16*>(out.data), C, T, slot_stride);
    } else {
        constexpr int kBlock = 32;
        causal_conv1d_sequence_snapshot_kernel<<<grid_for(C, kBlock, "sequence snapshot"), kBlock,
                                                 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.data),
            static_cast<__nv_bfloat16*>(conv_states.data),
            static_cast<const std::int32_t*>(initial_slot.data),
            static_cast<__nv_bfloat16*>(out.data), C, T, slot_stride);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
