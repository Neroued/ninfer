#include "ops/linear_attention/gated_delta_net/replay/launch.h"
#include "ops/common/cuda/reduce.cuh"
#include "ops/linear_attention/gated_delta_net/replay/record_io.cuh"
#include "core/device.h"
#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net {
namespace {
template <bool Masked>
void launch_recurrent_record_fixed(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   const Tensor& ssm_states, const Tensor& valid_columns,
                                   const Tensor& initial_state_slots, Tensor& key_record,
                                   Tensor& value_record, Tensor& gate_record, Tensor& out,
                                   cudaStream_t stream) {
    const auto heads = HeadMap::of(q.ne[1], v.ne[1]);
    const dim3 grid(static_cast<unsigned>(v.ne[1]), static_cast<unsigned>(q.ne[3]),
                    static_cast<unsigned>(kStateDim / kBlockDv));
    const dim3 block(common::kWarpSize, kNumWarps, 1);
    const std::int64_t state_slot_stride =
        ssm_states.nb[3] / static_cast<std::int64_t>(sizeof(float));
    const RecordAccess<Masked> access{
        static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data),
        static_cast<const float*>(ssm_states.data),
        Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
        static_cast<const std::int32_t*>(initial_state_slots.data),
        static_cast<__nv_bfloat16*>(key_record.data),
        static_cast<__nv_bfloat16*>(value_record.data),
        reinterpret_cast<uint2*>(gate_record.data),
        static_cast<__nv_bfloat16*>(out.data),
        heads,
        q.ne[2],
        state_slot_stride,
        scale,
    };
    recurrent_record_kernel<Masked><<<grid, block, 0, stream>>>(access);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_recurrent_record(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                             const Tensor& beta, float scale, const Tensor& ssm_states,
                             const Tensor& valid_columns, const Tensor& initial_state_slots,
                             Tensor& key_record, Tensor& value_record, Tensor& gate_record,
                             Tensor& out, cudaStream_t stream) {
    if (valid_columns.data == nullptr) {
        launch_recurrent_record_fixed<false>(q, k, v, g, beta, scale, ssm_states, valid_columns,
                                             initial_state_slots, key_record, value_record,
                                             gate_record, out, stream);
    } else {
        launch_recurrent_record_fixed<true>(q, k, v, g, beta, scale, ssm_states, valid_columns,
                                            initial_state_slots, key_record, value_record,
                                            gate_record, out, stream);
    }
}

} // namespace ninfer::ops::detail::gated_delta_net
