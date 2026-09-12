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

// The small-T routes launch one thread per channel and token in one block, so the bound and the
// channel tile together may not exceed the hardware block ceiling.
static_assert(kCausalConvChannelTile * kCausalConvParallelMaxTokens <= 1024,
              "small-T block would exceed 1024 threads");
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
        const CausalConvContiguousOutput<__nv_bfloat162> out_map{
            reinterpret_cast<__nv_bfloat162*>(out.data), C / 2};
        causal_conv1d_prefill_pairs_kernel<<<prefill_output_grid_for(C / 2, T, kPairBlock),
                                             kPairBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.data),
            static_cast<const __nv_bfloat16*>(conv_state_in.data), out_map, C, T);
        CUDA_CHECK(cudaGetLastError());
    } else {
        const CausalConvContiguousOutput<__nv_bfloat16> out_map{
            static_cast<__nv_bfloat16*>(out.data), C};
        causal_conv1d_prefill_kernel<<<prefill_output_grid_for(C, T, kOutputBlock), kOutputBlock, 0,
                                       stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.data),
            static_cast<const __nv_bfloat16*>(conv_state_in.data), out_map, C, T);
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

    const CausalConvContiguousOutput<__nv_bfloat16> out_map{static_cast<__nv_bfloat16*>(out.data),
                                                            C};
    causal_conv1d_smallt_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.data),
        static_cast<const __nv_bfloat16*>(conv_state_in.data),
        static_cast<__nv_bfloat16*>(conv_state_out.data), out_map, C, T);
    CUDA_CHECK(cudaGetLastError());
}

template <std::int32_t Rows0, std::int32_t Rows1, std::int32_t Rows2>
void smallt_split_launch(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                         Tensor& conv_state_out, Tensor& out0, Tensor& out1, Tensor& out2,
                         cudaStream_t stream) {
    const std::int32_t C = x.ne[0];
    const std::int32_t T = x.ne[1];
    if (Rows0 + Rows1 + Rows2 != C) {
        throw std::invalid_argument("causal_conv1d: split geometry does not cover the channels");
    }
    const dim3 block(kCausalConvChannelTile, static_cast<unsigned int>(T));
    const int grid = grid_for(C, kCausalConvChannelTile, "small-T");

    const CausalConvSplitOutput3<__nv_bfloat16, Rows0, Rows1, Rows2> out_map{
        static_cast<__nv_bfloat16*>(out0.data), static_cast<__nv_bfloat16*>(out1.data),
        static_cast<__nv_bfloat16*>(out2.data)};
    causal_conv1d_smallt_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.data),
        static_cast<const __nv_bfloat16*>(conv_state_in.data),
        static_cast<__nv_bfloat16*>(conv_state_out.data), out_map, C, T);
    CUDA_CHECK(cudaGetLastError());
}

void causal_conv1d_smallt_split_launch(const Tensor& x, const Tensor& weight,
                                       const Tensor& conv_state_in, Tensor& conv_state_out,
                                       Tensor& out0, Tensor& out1, Tensor& out2,
                                       CausalConvSplitGeometry geometry, cudaStream_t stream) {
    switch (geometry) {
    case CausalConvSplitGeometry::Rows2048x2048x4096:
        smallt_split_launch<2048, 2048, 4096>(x, weight, conv_state_in, conv_state_out, out0, out1,
                                              out2, stream);
        return;
    case CausalConvSplitGeometry::Rows2048x2048x6144:
        smallt_split_launch<2048, 2048, 6144>(x, weight, conv_state_in, conv_state_out, out0, out1,
                                              out2, stream);
        return;
    }
    throw std::logic_error("causal_conv1d: unhandled split geometry");
}

template <std::int32_t Rows0, std::int32_t Rows1, std::int32_t Rows2>
void prefill_split_launch(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                          Tensor& conv_state_out, Tensor& out0, Tensor& out1, Tensor& out2,
                          cudaStream_t stream) {
    static_assert((Rows0 % 2) == 0 && (Rows1 % 2) == 0 && (Rows2 % 2) == 0,
                  "a pair must not straddle two destinations");
    constexpr int kChannelBlock = 256;
    constexpr int kPairBlock    = 256;
    const std::int32_t C        = x.ne[0];
    const std::int32_t T        = x.ne[1];
    if (Rows0 + Rows1 + Rows2 != C) {
        throw std::invalid_argument("causal_conv1d: split geometry does not cover the channels");
    }

    const CausalConvSplitOutput3<__nv_bfloat162, Rows0 / 2, Rows1 / 2, Rows2 / 2> out_map{
        reinterpret_cast<__nv_bfloat162*>(out0.data), reinterpret_cast<__nv_bfloat162*>(out1.data),
        reinterpret_cast<__nv_bfloat162*>(out2.data)};
    causal_conv1d_prefill_pairs_kernel<<<prefill_output_grid_for(C / 2, T, kPairBlock), kPairBlock,
                                         0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.data),
        static_cast<const __nv_bfloat16*>(conv_state_in.data), out_map, C, T);
    CUDA_CHECK(cudaGetLastError());

    causal_conv1d_prefill_state_kernel<<<grid_for(C, kChannelBlock, "prefill state"), kChannelBlock,
                                         0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(conv_state_in.data),
        static_cast<__nv_bfloat16*>(conv_state_out.data), C, T);
    CUDA_CHECK(cudaGetLastError());
}

void causal_conv1d_prefill_split_launch(const Tensor& x, const Tensor& weight,
                                        const Tensor& conv_state_in, Tensor& conv_state_out,
                                        Tensor& out0, Tensor& out1, Tensor& out2,
                                        CausalConvSplitGeometry geometry, cudaStream_t stream) {
    switch (geometry) {
    case CausalConvSplitGeometry::Rows2048x2048x4096:
        prefill_split_launch<2048, 2048, 4096>(x, weight, conv_state_in, conv_state_out, out0, out1,
                                               out2, stream);
        return;
    case CausalConvSplitGeometry::Rows2048x2048x6144:
        prefill_split_launch<2048, 2048, 6144>(x, weight, conv_state_in, conv_state_out, out0, out1,
                                               out2, stream);
        return;
    }
    throw std::logic_error("causal_conv1d: unhandled split geometry");
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

template <bool Masked>
void launch_batched_sequence_snapshot(const Tensor& x, const Tensor& weight, Tensor& conv_states,
                                      const Tensor& valid_columns,
                                      const Tensor& initial_state_slots,
                                      const Tensor& snapshot_base_slots, Tensor& out,
                                      std::int64_t slot_stride, cudaStream_t stream) {
    const std::int32_t C     = x.ne[0];
    const std::int32_t width = x.ne[1];
    const std::int32_t batch = x.ne[2];
    const int block          = width == 1 ? 256 : 32;
    const dim3 grid(grid_for(C, block, "batched sequence snapshot"),
                    static_cast<unsigned int>(batch));
    causal_conv1d_batched_sequence_snapshot_kernel<Masked><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.data),
        static_cast<__nv_bfloat16*>(conv_states.data),
        Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
        static_cast<const std::int32_t*>(initial_state_slots.data),
        static_cast<const std::int32_t*>(snapshot_base_slots.data),
        static_cast<__nv_bfloat16*>(out.data), C, width, slot_stride);
}

template <bool Masked>
void launch_batched_smallt_snapshot(const Tensor& x, const Tensor& weight, Tensor& conv_states,
                                    const Tensor& valid_columns, const Tensor& initial_state_slots,
                                    const Tensor& snapshot_base_slots, Tensor& out,
                                    std::int64_t slot_stride, cudaStream_t stream) {
    const std::int32_t C     = x.ne[0];
    const std::int32_t width = x.ne[1];
    const std::int32_t batch = x.ne[2];
    const dim3 block(kCausalConvChannelTile, static_cast<unsigned int>(width));
    const dim3 grid(grid_for(C, kCausalConvChannelTile, "batched small-T snapshot"),
                    static_cast<unsigned int>(batch));
    causal_conv1d_batched_snapshot_smallt_kernel<Masked><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.data),
        static_cast<__nv_bfloat16*>(conv_states.data),
        Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
        static_cast<const std::int32_t*>(initial_state_slots.data),
        static_cast<const std::int32_t*>(snapshot_base_slots.data),
        static_cast<__nv_bfloat16*>(out.data), C, width, slot_stride);
}

void causal_conv1d_snapshot_launch(const Tensor& x, const Tensor& weight, Tensor& conv_states,
                                   const Tensor& valid_columns, const Tensor& initial_state_slots,
                                   const Tensor& snapshot_base_slots, Tensor& out,
                                   cudaStream_t stream) {
    const std::int32_t C = x.ne[0];
    const std::int32_t T = x.ne[1];
    const std::int32_t B = x.ne[2];
    const std::int64_t slot_stride =
        static_cast<std::int64_t>(conv_states.ne[0]) * static_cast<std::int64_t>(conv_states.ne[1]);

    if (B == 1 && valid_columns.data == nullptr && T == 1) {
        constexpr int kBlock = 256;
        causal_conv1d_snapshot_decode_kernel<<<grid_for(C, kBlock, "snapshot decode"), kBlock, 0,
                                               stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.data),
            static_cast<__nv_bfloat16*>(conv_states.data),
            static_cast<const std::int32_t*>(initial_state_slots.data),
            static_cast<const std::int32_t*>(snapshot_base_slots.data),
            static_cast<__nv_bfloat16*>(out.data), C, slot_stride);
    } else if (B == 1 && valid_columns.data == nullptr && T <= kCausalConvParallelMaxTokens) {
        const dim3 block(kCausalConvChannelTile, static_cast<unsigned int>(T));
        const int grid = grid_for(C, kCausalConvChannelTile, "small-T snapshot");
        causal_conv1d_snapshot_smallt_kernel<<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.data),
            static_cast<__nv_bfloat16*>(conv_states.data),
            static_cast<const std::int32_t*>(initial_state_slots.data),
            static_cast<const std::int32_t*>(snapshot_base_slots.data),
            static_cast<__nv_bfloat16*>(out.data), C, T, slot_stride);
    } else if (B == 1 && valid_columns.data == nullptr) {
        constexpr int kBlock = 32;
        causal_conv1d_sequence_snapshot_kernel<<<grid_for(C, kBlock, "sequence snapshot"), kBlock,
                                                 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.data),
            static_cast<__nv_bfloat16*>(conv_states.data),
            static_cast<const std::int32_t*>(initial_state_slots.data),
            static_cast<const std::int32_t*>(snapshot_base_slots.data),
            static_cast<__nv_bfloat16*>(out.data), C, T, slot_stride);
    } else if (T == 1) {
        if (valid_columns.data == nullptr) {
            launch_batched_sequence_snapshot<false>(x, weight, conv_states, valid_columns,
                                                    initial_state_slots, snapshot_base_slots, out,
                                                    slot_stride, stream);
        } else {
            launch_batched_sequence_snapshot<true>(x, weight, conv_states, valid_columns,
                                                   initial_state_slots, snapshot_base_slots, out,
                                                   slot_stride, stream);
        }
    } else if (T <= kCausalConvParallelMaxTokens) {
        if (valid_columns.data == nullptr) {
            launch_batched_smallt_snapshot<false>(x, weight, conv_states, valid_columns,
                                                  initial_state_slots, snapshot_base_slots, out,
                                                  slot_stride, stream);
        } else {
            launch_batched_smallt_snapshot<true>(x, weight, conv_states, valid_columns,
                                                 initial_state_slots, snapshot_base_slots, out,
                                                 slot_stride, stream);
        }
    } else if (valid_columns.data == nullptr) {
        launch_batched_sequence_snapshot<false>(x, weight, conv_states, valid_columns,
                                                initial_state_slots, snapshot_base_slots, out,
                                                slot_stride, stream);
    } else {
        launch_batched_sequence_snapshot<true>(x, weight, conv_states, valid_columns,
                                               initial_state_slots, snapshot_base_slots, out,
                                               slot_stride, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
