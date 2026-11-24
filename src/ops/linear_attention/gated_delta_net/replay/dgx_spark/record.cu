#include "ops/linear_attention/gated_delta_net/replay/launch.h"
#include "ops/common/cuda/reduce.cuh"
#include "ops/linear_attention/gated_delta_net/replay/record_io.cuh"
#include "core/device.h"
#include "ops/common/host/operands.h"
#include "ops/linear_attention/gated_delta_net/kernels/recurrent_subgroup.cuh"
#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net {
namespace {
template <bool Masked>
__global__ __launch_bounds__(128, 2) void record_kernel(RecordAccess<Masked> access) {
    extern __shared__ __align__(16) char backing[];
    const auto coord = access.coordinates();
    subgroup::run<16, true, subgroup::Mode::Record, true>(
        access, coord, access.active_columns(coord),
        reinterpret_cast<subgroup::Window<subgroup::kWindowSize>*>(backing));
}

template <bool Masked>
void launch_recurrent_record_fixed(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   ConstStateFieldView initial_state, const Tensor& valid_columns,
                                   Tensor& key_record, Tensor& value_record, Tensor& gate_record,
                                   Tensor& out, cudaStream_t stream) {
    const auto heads = HeadMap::of(q.ne[1], v.ne[1]);
    const dim3 grid(static_cast<unsigned>(v.ne[1]), static_cast<unsigned>(q.ne[3]),
                    static_cast<unsigned>(kStateDim / kBlockDv));
    const dim3 block(common::kWarpSize, kNumWarps, 1);
    const RecordAccess<Masked> access{
        static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data),
        common::operand::state_field_base<const float>(initial_state),
        Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
        static_cast<const std::int64_t*>(initial_state.records.byte_offsets.data),
        static_cast<__nv_bfloat16*>(key_record.data),
        static_cast<__nv_bfloat16*>(value_record.data),
        reinterpret_cast<uint2*>(gate_record.data),
        static_cast<__nv_bfloat16*>(out.data),
        heads,
        q.ne[2],
        scale,
    };
    record_kernel<Masked>
        <<<grid, block, sizeof(subgroup::Window<subgroup::kWindowSize>), stream>>>(access);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_recurrent_record(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                             const Tensor& beta, float scale, ConstStateFieldView initial_state,
                             const Tensor& valid_columns, Tensor& key_record, Tensor& value_record,
                             Tensor& gate_record, Tensor& out, cudaStream_t stream) {
    if (valid_columns.data == nullptr) {
        launch_recurrent_record_fixed<false>(q, k, v, g, beta, scale, initial_state, valid_columns,
                                             key_record, value_record, gate_record, out, stream);
    } else {
        launch_recurrent_record_fixed<true>(q, k, v, g, beta, scale, initial_state, valid_columns,
                                            key_record, value_record, gate_record, out, stream);
    }
}

} // namespace ninfer::ops::detail::gated_delta_net
