// ninfer::kernels - causal_conv1d launcher: grid/block/stream configuration + kernel launch.
#include "kernels/causal_conv1d/launch.h"

#include "kernels/common/math.h"
#include "kernels/causal_conv1d/kernel.cuh"
#include "core/device.h" // CUDA_CHECK

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::kernels::detail {
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
    constexpr int kBlock     = 256;
    constexpr int kPairBlock = 256;
    const std::int32_t C     = x.ne[0];
    const std::int32_t T     = x.ne[1];
    const auto x_addr        = reinterpret_cast<std::uintptr_t>(x.data);
    const auto w_addr        = reinterpret_cast<std::uintptr_t>(weight.data);
    const auto in_addr       = reinterpret_cast<std::uintptr_t>(conv_state_in.data);
    const auto out_state_addr = reinterpret_cast<std::uintptr_t>(conv_state_out.data);
    const auto out_addr      = reinterpret_cast<std::uintptr_t>(out.data);

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
        causal_conv1d_prefill_kernel<<<prefill_output_grid_for(C, T, kBlock), kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.data),
            static_cast<const __nv_bfloat16*>(conv_state_in.data),
            static_cast<__nv_bfloat16*>(out.data), C, T);
        CUDA_CHECK(cudaGetLastError());
    }

    causal_conv1d_prefill_state_kernel<<<grid_for(C, kBlock, "prefill state"), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(conv_state_in.data),
        static_cast<__nv_bfloat16*>(conv_state_out.data), C, T);
    CUDA_CHECK(cudaGetLastError());
}

void causal_conv1d_decode_launch(const Tensor& x, const Tensor& weight, Tensor& conv_state,
                                 Tensor& out, cudaStream_t stream) {
    constexpr int kBlock = 256;
    const std::int32_t C = x.ne[0];

    causal_conv1d_decode_kernel<<<grid_for(C, kBlock, "decode"), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.data),
        static_cast<__nv_bfloat16*>(conv_state.data), static_cast<__nv_bfloat16*>(out.data), C);
    CUDA_CHECK(cudaGetLastError());
}

void causal_conv1d_sequence_snapshot_launch(const Tensor& x, const Tensor& weight,
                                            Tensor& conv_states, const Tensor& initial_slot,
                                            Tensor& out, cudaStream_t stream) {
    constexpr int kBlock = 256;
    const std::int32_t C = x.ne[0];
    const std::int32_t T = x.ne[1];
    const std::int32_t slots = conv_states.ne[2];
    const std::int64_t slot_stride =
        static_cast<std::int64_t>(conv_states.ne[0]) * static_cast<std::int64_t>(conv_states.ne[1]);

    causal_conv1d_sequence_snapshot_kernel<<<grid_for(C, kBlock, "sequence snapshot"), kBlock, 0,
                                             stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.data),
        static_cast<__nv_bfloat16*>(conv_states.data),
        static_cast<const std::int32_t*>(initial_slot.data), static_cast<__nv_bfloat16*>(out.data),
        C, T, slots, slot_stride);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::kernels::detail
