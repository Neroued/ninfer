#include "ops/linear_attention/gated_delta_net/recurrence/launch.h"
#include "ops/linear_attention/gated_delta_net/recurrence/recurrent_io.cuh"
#include "ops/linear_attention/gated_delta_net/kernels/recurrent_subgroup.cuh"
#include "core/device.h"
#include "ops/common/host/operands.h"

#include <algorithm>

namespace ninfer::ops::detail::gated_delta_net {
namespace {

template <bool Normalize, bool Prefetch>
__global__ __launch_bounds__(128, 2) void direct_kernel(DirectAccess access) {
    extern __shared__ __align__(16) char backing[];
    subgroup::run<16, Normalize, subgroup::Mode::Update, Prefetch>(
        access, access.coordinates(), access.width,
        reinterpret_cast<subgroup::Window<subgroup::kWindowSize>*>(backing));
}

template <bool Normalize, bool GroupStateTiles, bool Prefetch>
__global__ __launch_bounds__(128, 2) void batch_kernel(BatchUpdateAccess access) {
    extern __shared__ __align__(16) char backing[];
    const auto coord = subgroup::coordinates<GroupStateTiles>(
        access.heads, access.batch_begin + static_cast<int>(blockIdx.y));
    if (access.valid_columns && !access.valid_columns[coord.batch]) {
        zero_output_suffix(access, coord, 0, 1);
        return;
    }
    subgroup::run<16, Normalize, subgroup::Mode::Update, Prefetch>(
        access, coord, 1, reinterpret_cast<subgroup::Window<1>*>(backing));
}

template <bool Normalize>
void launch_direct(const DirectAccess& access, cudaStream_t stream) {
    const dim3 grid(access.heads.H_v, 1, kStateDim / kBlockDv), block(32, kNumWarps);
    const auto alignment = reinterpret_cast<std::uintptr_t>(access.q) |
                           reinterpret_cast<std::uintptr_t>(access.k) |
                           reinterpret_cast<std::uintptr_t>(access.v);
    if (access.width <= subgroup::kWindowSize && (alignment & 15) == 0) {
        direct_kernel<Normalize, true>
            <<<grid, block, sizeof(subgroup::Window<subgroup::kWindowSize>), stream>>>(access);
    } else {
        direct_kernel<Normalize, false><<<grid, block, 0, stream>>>(access);
    }
}

} // namespace

void launch_recurrent_inout(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                            const Tensor& beta, float scale, bool normalize_qk,
                            const Tensor& state_in, Tensor& state_out, Tensor& out,
                            cudaStream_t stream) {
    const DirectAccess access{static_cast<const __nv_bfloat16*>(q.data),
                              static_cast<const __nv_bfloat16*>(k.data),
                              static_cast<const __nv_bfloat16*>(v.data),
                              static_cast<const float*>(g.data),
                              static_cast<const float*>(beta.data),
                              static_cast<const float*>(state_in.data),
                              static_cast<float*>(state_out.data),
                              static_cast<__nv_bfloat16*>(out.data),
                              HeadMap::of(q.ne[1], v.ne[1]),
                              q.ne[2],
                              scale};
    if (normalize_qk)
        launch_direct<true>(access, stream);
    else
        launch_direct<false>(access, stream);
    CUDA_CHECK(cudaGetLastError());
}

void launch_recurrent_batch_update(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   bool normalize_qk, ConstStateFieldView source_state,
                                   StateFieldView destination_state, const Tensor& valid_columns,
                                   Tensor& out, cudaStream_t stream) {
    BatchUpdateAccess access{
        static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data),
        common::operand::state_field_base<const float>(source_state),
        common::operand::state_field_base<float>(destination_state),
        static_cast<const std::int64_t*>(source_state.records.byte_offsets.data),
        static_cast<const std::int64_t*>(destination_state.records.byte_offsets.data),
        static_cast<const std::int32_t*>(valid_columns.data),
        static_cast<__nv_bfloat16*>(out.data),
        HeadMap::of(q.ne[1], v.ne[1]),
        scale};
    const bool group_tiles = q.ne[3] >= 16;
    const auto alignment   = reinterpret_cast<std::uintptr_t>(q.data) |
                           reinterpret_cast<std::uintptr_t>(k.data) |
                           reinterpret_cast<std::uintptr_t>(v.data);
    dim3 grid(v.ne[1] * (group_tiles ? kStateDim / kBlockDv : 1), 1,
              group_tiles ? 1 : kStateDim / kBlockDv);
    const dim3 block(32, kNumWarps);
    for (std::int64_t begin = 0; begin < q.ne[3]; begin += 65535) {
        access.batch_begin = static_cast<int>(begin);
        grid.y             = static_cast<unsigned>(std::min<std::int64_t>(65535, q.ne[3] - begin));
        const auto submit  = [&]<bool Normalize, bool Prefetch>() {
            constexpr auto bytes = Prefetch ? sizeof(subgroup::Window<1>) : 0;
            if (group_tiles)
                batch_kernel<Normalize, true, Prefetch><<<grid, block, bytes, stream>>>(access);
            else
                batch_kernel<Normalize, false, Prefetch><<<grid, block, bytes, stream>>>(access);
        };
        if ((alignment & 15) == 0) {
            if (normalize_qk)
                submit.template operator()<true, true>();
            else
                submit.template operator()<false, true>();
        } else {
            if (normalize_qk)
                submit.template operator()<true, false>();
            else
                submit.template operator()<false, false>();
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail::gated_delta_net
