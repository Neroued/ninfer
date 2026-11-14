#include "ops/linear_attention/gated_delta_net/replay/launch.h"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/linear_attention/gated_delta_net/replay/record_io.cuh"
#include "core/device.h"
#include "ops/common/host/operands.h"
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net {
namespace {
template <int Capacity>
struct RecordWindow {
    __nv_bfloat16 key[Capacity][kStateDim];
    __nv_bfloat16 query[Capacity][kStateDim];
    __nv_bfloat16 value[Capacity][kStateDim];
    float gate[Capacity], beta[Capacity];
};

template <bool Masked, int Capacity>
struct WindowedRecordAccess : RecordAccess<Masked> {
    const RecordWindow<Capacity>* window;

    __device__ const __nv_bfloat16* key_ptr(const RecurrentCoordinates&, int token) const {
        return window->key[token];
    }

    __device__ const __nv_bfloat16* query_ptr(const RecurrentCoordinates&, int token) const {
        return window->query[token];
    }

    __device__ const __nv_bfloat16* value_ptr(const RecurrentCoordinates&, int token) const {
        return window->value[token];
    }

    __device__ RawGatePair load_gate(const RecurrentCoordinates&, int token) const {
        const float g = window->gate[token], beta = window->beta[token];
        return {make_uint2(__float_as_uint(g), __float_as_uint(beta)), g, beta};
    }
};

// Stage the represented inputs while fetching state; the transition keeps the
// same arithmetic and record effects as normalized one-token execution.
template <bool Masked, int Capacity>
__global__ __launch_bounds__(128, 2) void record_window_kernel(RecordAccess<Masked> access) {
    __shared__ __align__(16) RecordWindow<Capacity> window;
    const auto coord = access.coordinates();
    const int valid = access.active_columns(coord), tid = coord.warp * 32 + coord.lane;
    if constexpr (Masked) {
        if (!valid) {
            zero_output_suffix(access, coord, 0, access.width);
            return;
        }
    }
    for (int i = tid; i < valid * (kStateDim / 8); i += 128) {
        const int token = i / (kStateDim / 8), column = i % (kStateDim / 8) * 8;
        common::cp_async<16, Cache::ca>(&window.key[token][column],
                                        access.key_ptr(coord, token) + column);
        if (token < valid) {
            common::cp_async<16, Cache::ca>(&window.query[token][column],
                                            access.query_ptr(coord, token) + column);
            common::cp_async<16, Cache::ca>(&window.value[token][column],
                                            access.value_ptr(coord, token) + column);
        }
    }
    if (tid < valid) {
        const auto gate  = access.load_gate(coord, tid);
        window.gate[tid] = gate.g;
        window.beta[tid] = gate.beta;
    }
    common::cp_commit();
    __align__(16) float state[kDvPerWarp][kQkPerLane];
    load_state_tile(state, access.state_read_base(coord), coord);
    common::cp_wait<0>();
    __syncthreads();
    const WindowedRecordAccess<Masked, Capacity> staged{access, &window};
    run_recurrent_sequence<true, RecordEffects, false>(state, staged, coord, valid);
    zero_output_suffix(access, coord, valid, access.width);
}

template <bool Masked>
void launch_recurrent_record_fixed(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   ConstStateFieldView initial_state, const Tensor& valid_columns,
                                   Tensor& key_record, Tensor& value_record, Tensor& gate_record,
                                   Tensor& out, cudaStream_t stream) {
    const auto heads = HeadMap::of(q.ne[1], v.ne[1]);
    dim3 grid(static_cast<unsigned>(v.ne[1]), 1, static_cast<unsigned>(kStateDim / kBlockDv));
    const dim3 block(common::kWarpSize, kNumWarps, 1);
    RecordAccess<Masked> access{
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
    const auto alignment = reinterpret_cast<std::uintptr_t>(q.data) |
                           reinterpret_cast<std::uintptr_t>(k.data) |
                           reinterpret_cast<std::uintptr_t>(v.data);
    for (std::int64_t begin = 0; begin < q.ne[3]; begin += 65535) {
        grid.y = static_cast<unsigned>(std::min<std::int64_t>(65535, q.ne[3] - begin));
        if (alignment % 16 != 0)
            recurrent_record_kernel<Masked, false><<<grid, block, 0, stream>>>(access);
        else if (q.ne[2] <= 4)
            record_window_kernel<Masked, 4><<<grid, block, 0, stream>>>(access);
        else if (q.ne[2] <= 8)
            record_window_kernel<Masked, 8><<<grid, block, 0, stream>>>(access);
        else
            record_window_kernel<Masked, 16><<<grid, block, 0, stream>>>(access);
        if (begin + grid.y < q.ne[3]) access.advance_rows(grid.y);
    }
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
