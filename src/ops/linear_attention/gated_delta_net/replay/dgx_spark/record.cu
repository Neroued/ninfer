#include "ops/linear_attention/gated_delta_net/replay/launch.h"
#include "ops/common/cuda/reduce.cuh"
#include "ops/linear_attention/gated_delta_net/replay/record_io.cuh"
#include "core/device.h"
#include "ops/common/host/operands.h"
#include "ops/linear_attention/gated_delta_net/kernels/recurrent_subgroup.cuh"
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net {
namespace {
template <bool Masked, bool GroupStateTiles, bool Prefetch, class StateT>
__global__ __launch_bounds__(128, 2) void record_kernel(RecordAccess<Masked, StateT> access) {
    extern __shared__ __align__(16) char backing[];
    const auto coord = subgroup::coordinates<GroupStateTiles>(access.heads, blockIdx.y);
    const int valid  = access.active_columns(coord);
    if constexpr (Masked) {
        if (!valid) {
            zero_output_suffix(access, coord, 0, access.width);
            return;
        }
    }
    subgroup::run<16, true, subgroup::Mode::Record, Prefetch>(
        access, coord, valid, reinterpret_cast<subgroup::Window<subgroup::kWindowSize>*>(backing));
}

template <bool Masked, class StateT>
void launch_recurrent_record_fixed(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   ConstStateFieldView initial_state, const Tensor& valid_columns,
                                   const Tensor& key_record, const Tensor& delta_record,
                                   const Tensor& alpha_record, Tensor& out, cudaStream_t stream) {
    const auto heads       = HeadMap::of(q.ne[1], v.ne[1]);
    const bool group_tiles = q.ne[3] >= 16;
    dim3 grid(v.ne[1] * (group_tiles ? kStateDim / kBlockDv : 1), 1,
              group_tiles ? 1 : kStateDim / kBlockDv);
    const dim3 block(common::kWarpSize, kNumWarps, 1);
    RecordAccess<Masked, StateT> access{
        static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data),
        common::operand::state_field_base<const StateT>(initial_state),
        Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
        static_cast<const std::int64_t*>(initial_state.records.byte_offsets.data),
        static_cast<__nv_bfloat16*>(key_record.data),
        static_cast<float*>(delta_record.data),
        static_cast<float*>(alpha_record.data),
        static_cast<__nv_bfloat16*>(out.data),
        heads,
        q.ne[2],
        scale,
    };
    const auto alignment = reinterpret_cast<std::uintptr_t>(q.data) |
                           reinterpret_cast<std::uintptr_t>(k.data) |
                           reinterpret_cast<std::uintptr_t>(v.data);
    for (std::int64_t begin = 0; begin < q.ne[3]; begin += 65535) {
        grid.y            = static_cast<unsigned>(std::min<std::int64_t>(65535, q.ne[3] - begin));
        const auto submit = [&]<bool Prefetch>() {
            constexpr auto bytes = Prefetch ? sizeof(subgroup::Window<subgroup::kWindowSize>) : 0;
            if (group_tiles)
                record_kernel<Masked, true, Prefetch, StateT>
                    <<<grid, block, bytes, stream>>>(access);
            else
                record_kernel<Masked, false, Prefetch, StateT>
                    <<<grid, block, bytes, stream>>>(access);
        };
        if ((alignment & 15) == 0)
            submit.template operator()<true>();
        else
            submit.template operator()<false>();
        if (begin + grid.y < q.ne[3]) access.advance_rows(grid.y);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_recurrent_record(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                             const Tensor& beta, float scale, ConstStateFieldView initial_state,
                             const Tensor& valid_columns, const Tensor& key_record,
                             const Tensor& delta_record, const Tensor& alpha_record, Tensor& out,
                             cudaStream_t stream) {
    const auto dispatch = [&]<class StateT>() {
        if (valid_columns.data == nullptr) {
            launch_recurrent_record_fixed<false, StateT>(q, k, v, g, beta, scale, initial_state,
                                                         valid_columns, key_record, delta_record,
                                                         alpha_record, out, stream);
        } else {
            launch_recurrent_record_fixed<true, StateT>(q, k, v, g, beta, scale, initial_state,
                                                        valid_columns, key_record, delta_record,
                                                        alpha_record, out, stream);
        }
    };
    if (initial_state.field.tensor.dtype == DType::BF16)
        dispatch.template operator()<__nv_bfloat16>();
    else
        dispatch.template operator()<float>();
}

} // namespace ninfer::ops::detail::gated_delta_net
