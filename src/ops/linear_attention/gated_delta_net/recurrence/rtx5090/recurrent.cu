#include "ops/linear_attention/gated_delta_net/recurrence/launch.h"
#include "ops/common/cuda/reduce.cuh"
#include "ops/linear_attention/gated_delta_net/recurrence/recurrent_io.cuh"
#include "core/device.h"
#include "ops/common/host/operands.h"
#include "ops/common/host/launch.h"
#include "ops/linear_attention/gated_delta_net/recurrence/chunked/packed.cuh"
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net {
namespace {
template <bool NormalizeQK, class StateT>
void launch_recurrent_direct_fixed(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   const Tensor& state_read, Tensor& state_write, Tensor& out,
                                   cudaStream_t stream, const std::int64_t* source_offset,
                                   const std::int64_t* destination_offset) {
    const auto heads = HeadMap::of(q.ne[1], v.ne[1]);
    const dim3 grid(static_cast<unsigned>(v.ne[1]), 1, static_cast<unsigned>(kStateDim / kBlockDv));
    const dim3 block(common::kWarpSize, kNumWarps, 1);
    recurrent_bf16_direct_kernel<NormalizeQK, StateT><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data), static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data), static_cast<const StateT*>(state_read.data),
        static_cast<StateT*>(state_write.data), static_cast<__nv_bfloat16*>(out.data), q.ne[2],
        heads, scale, source_offset, destination_offset);
    CUDA_CHECK(cudaGetLastError());
}

template <bool NormalizeInputs, class StateT>
void launch_recurrent_batch_update_fixed(const Tensor& q, const Tensor& k, const Tensor& v,
                                         const Tensor& g, const Tensor& beta, float scale,
                                         ConstStateFieldView source_state,
                                         StateFieldView destination_state,
                                         const Tensor& valid_columns, Tensor& out,
                                         cudaStream_t stream) {
    const auto heads = HeadMap::of(q.ne[1], v.ne[1]);
    dim3 grid(static_cast<unsigned>(v.ne[1]), 1, static_cast<unsigned>(kStateDim / kBlockDv));
    const dim3 block(common::kWarpSize, kNumWarps, 1);
    BatchUpdateAccess<StateT> access{
        static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data),
        common::operand::state_field_base<const StateT>(source_state),
        common::operand::state_field_base<StateT>(destination_state),
        static_cast<const std::int64_t*>(source_state.records.byte_offsets.data),
        static_cast<const std::int64_t*>(destination_state.records.byte_offsets.data),
        static_cast<const std::int32_t*>(valid_columns.data),
        static_cast<__nv_bfloat16*>(out.data),
        heads,
        scale,
    };
    for (std::int64_t begin = 0; begin < q.ne[3]; begin += 65535) {
        access.batch_begin = static_cast<int>(begin);
        grid.y             = static_cast<unsigned>(std::min<std::int64_t>(65535, q.ne[3] - begin));
        recurrent_batch_update_kernel<NormalizeInputs, StateT><<<grid, block, 0, stream>>>(access);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <bool Normalize, class StateT>
__global__ __launch_bounds__(common::kWarpSize* kNumWarps, 2) void recurrent_ragged_kernel(
    RecurrenceOperands args, RaggedSegments segments, HeadMap heads, int recurrent_before,
    chunked::ChunkIndexEntry* build_index) {
    if (build_index && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        chunked::build_chunk_index<common::kWarpSize, kNumWarps>(segments, recurrent_before,
                                                                 build_index);
    }
    for (std::int64_t s = blockIdx.y; s < segments.count; s += gridDim.y) {
        const int length = segments.lengths[s];
        if (recurrent_before && length >= recurrent_before) continue;
        const auto access =
            ragged_access<StateT>(args, heads, int(s), segments.cu_seqlens[s], length);
        const auto coord = access.coordinates();
        __align__(16) float state[kDvPerWarp][kQkPerLane];
        load_state_tile(state, access.state_read_base(coord), coord);
        run_recurrent_sequence<Normalize, OutputEffects>(state, access, coord, length);
        store_state_tile(state, access.state_write_base(coord), coord);
    }
}

} // namespace

void launch_recurrent_ragged(const RecurrenceOperands& args, RaggedSegments segments,
                             bool normalize_qk, int recurrent_before,
                             chunked::ChunkIndexEntry* build_index, cudaStream_t stream) {
    const auto heads = HeadMap::of(args.qk_heads, args.value_heads);
    const dim3 grid(args.value_heads, std::min(segments.count, common::kCudaGridYLimit),
                    kStateDim / kBlockDv);
    const dim3 block(common::kWarpSize, kNumWarps);
    const auto dispatch = [&]<class StateT>() {
        if (normalize_qk)
            recurrent_ragged_kernel<true, StateT>
                <<<grid, block, 0, stream>>>(args, segments, heads, recurrent_before, build_index);
        else
            recurrent_ragged_kernel<false, StateT>
                <<<grid, block, 0, stream>>>(args, segments, heads, recurrent_before, build_index);
    };
    if (args.state_dtype == DType::BF16)
        dispatch.template operator()<__nv_bfloat16>();
    else
        dispatch.template operator()<float>();
    CUDA_CHECK(cudaGetLastError());
}

void launch_recurrent_inout(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                            const Tensor& beta, float scale, bool normalize_qk,
                            const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                            cudaStream_t stream, const std::int64_t* source_offset,
                            const std::int64_t* destination_offset) {
    const auto dispatch = [&]<class StateT>() {
        if (normalize_qk) {
            launch_recurrent_direct_fixed<true, StateT>(q, k, v, g, beta, scale, ssm_state_in,
                                                        ssm_state_out, out, stream, source_offset,
                                                        destination_offset);
        } else {
            launch_recurrent_direct_fixed<false, StateT>(q, k, v, g, beta, scale, ssm_state_in,
                                                         ssm_state_out, out, stream, source_offset,
                                                         destination_offset);
        }
    };
    if (ssm_state_out.dtype == DType::BF16)
        dispatch.template operator()<__nv_bfloat16>();
    else
        dispatch.template operator()<float>();
}

void launch_recurrent_batch_update(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   bool normalize_qk, ConstStateFieldView source_state,
                                   StateFieldView destination_state, const Tensor& valid_columns,
                                   Tensor& out, cudaStream_t stream) {
    const auto dispatch = [&]<class StateT>() {
        if (normalize_qk) {
            launch_recurrent_batch_update_fixed<true, StateT>(q, k, v, g, beta, scale, source_state,
                                                              destination_state, valid_columns, out,
                                                              stream);
        } else {
            launch_recurrent_batch_update_fixed<false, StateT>(q, k, v, g, beta, scale,
                                                               source_state, destination_state,
                                                               valid_columns, out, stream);
        }
    };
    if (destination_state.field.tensor.dtype == DType::BF16)
        dispatch.template operator()<__nv_bfloat16>();
    else
        dispatch.template operator()<float>();
}

} // namespace ninfer::ops::detail::gated_delta_net
