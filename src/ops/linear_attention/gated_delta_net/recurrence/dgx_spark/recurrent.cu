#include "ops/linear_attention/gated_delta_net/recurrence/launch.h"
#include "ops/linear_attention/gated_delta_net/recurrence/recurrent_io.cuh"
#include "ops/linear_attention/gated_delta_net/kernels/recurrent_subgroup.cuh"
#include "core/device.h"
#include "ops/common/host/operands.h"
#include "ops/common/host/launch.h"
#include "ops/linear_attention/gated_delta_net/recurrence/chunked/packed.cuh"

#include <algorithm>

namespace ninfer::ops::detail::gated_delta_net {
namespace {

template <bool Normalize, bool Prefetch, class StateT>
__global__ __launch_bounds__(128, 2) void direct_kernel(DirectAccess<StateT> access) {
    extern __shared__ __align__(16) char backing[];
    subgroup::run<16, Normalize, subgroup::Mode::Update, Prefetch>(
        access, access.coordinates(), access.width,
        reinterpret_cast<subgroup::Window<subgroup::kWindowSize>*>(backing));
}

template <bool Normalize, bool GroupStateTiles, bool Prefetch, class StateT>
__global__ __launch_bounds__(128, 2) void batch_kernel(BatchUpdateAccess<StateT> access) {
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

template <bool Normalize, class StateT>
void launch_direct(const DirectAccess<StateT>& access, cudaStream_t stream) {
    const dim3 grid(access.heads.H_v, 1, kStateDim / kBlockDv), block(32, kNumWarps);
    const auto alignment = reinterpret_cast<std::uintptr_t>(access.q) |
                           reinterpret_cast<std::uintptr_t>(access.k) |
                           reinterpret_cast<std::uintptr_t>(access.v);
    if (access.width <= subgroup::kWindowSize && (alignment & 15) == 0) {
        direct_kernel<Normalize, true, StateT>
            <<<grid, block, sizeof(subgroup::Window<subgroup::kWindowSize>), stream>>>(access);
    } else {
        direct_kernel<Normalize, false, StateT><<<grid, block, 0, stream>>>(access);
    }
}

template <bool Normalize, bool Prefetch, class StateT>
__global__ __launch_bounds__(common::kWarpSize* kNumWarps, 2) void recurrent_ragged_kernel(
    RecurrenceOperands args, RaggedSegments segments, HeadMap heads, int recurrent_before,
    chunked::ChunkIndexEntry* build_index) {
    extern __shared__ __align__(16) char backing[];
    if (build_index && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        chunked::build_chunk_index<common::kWarpSize, kNumWarps>(segments, recurrent_before,
                                                                 build_index);
    }
    for (std::int64_t s = blockIdx.y; s < segments.count; s += gridDim.y) {
        const int length = segments.lengths[s];
        if (recurrent_before && length >= recurrent_before) continue;
        const auto access =
            ragged_access<StateT>(args, heads, int(s), segments.cu_seqlens[s], length);
        subgroup::run<16, Normalize, subgroup::Mode::Update, Prefetch>(
            access, access.coordinates(), length,
            reinterpret_cast<subgroup::Window<subgroup::kWindowSize>*>(backing));
        if constexpr (Prefetch) {
            if (s + gridDim.y < segments.count) __syncthreads();
        }
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
    const auto alignment = reinterpret_cast<std::uintptr_t>(args.q) |
                           reinterpret_cast<std::uintptr_t>(args.k) |
                           reinterpret_cast<std::uintptr_t>(args.v);
    const auto launch = [&]<bool Normalize, class StateT> {
        if ((alignment & 15) == 0) {
            recurrent_ragged_kernel<Normalize, true, StateT>
                <<<grid, block, sizeof(subgroup::Window<subgroup::kWindowSize>), stream>>>(
                    args, segments, heads, recurrent_before, build_index);
        } else {
            recurrent_ragged_kernel<Normalize, false, StateT>
                <<<grid, block, 0, stream>>>(args, segments, heads, recurrent_before, build_index);
        }
    };
    const auto dispatch = [&]<class StateT>() {
        if (normalize_qk)
            launch.template operator()<true, StateT>();
        else
            launch.template operator()<false, StateT>();
    };
    if (args.state_dtype == DType::BF16)
        dispatch.template operator()<__nv_bfloat16>();
    else
        dispatch.template operator()<float>();
    CUDA_CHECK(cudaGetLastError());
}

void launch_recurrent_inout(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                            const Tensor& beta, float scale, bool normalize_qk,
                            const Tensor& state_in, Tensor& state_out, Tensor& out,
                            cudaStream_t stream, const std::int64_t* source_offset,
                            const std::int64_t* destination_offset) {
    const auto dispatch = [&]<class StateT>() {
        const DirectAccess<StateT> access{static_cast<const __nv_bfloat16*>(q.data),
                                          static_cast<const __nv_bfloat16*>(k.data),
                                          static_cast<const __nv_bfloat16*>(v.data),
                                          static_cast<const float*>(g.data),
                                          static_cast<const float*>(beta.data),
                                          static_cast<const StateT*>(state_in.data),
                                          static_cast<StateT*>(state_out.data),
                                          static_cast<__nv_bfloat16*>(out.data),
                                          HeadMap::of(q.ne[1], v.ne[1]),
                                          q.ne[2],
                                          scale,
                                          source_offset,
                                          destination_offset};
        if (normalize_qk)
            launch_direct<true, StateT>(access, stream);
        else
            launch_direct<false, StateT>(access, stream);
    };
    if (state_out.dtype == DType::BF16)
        dispatch.template operator()<__nv_bfloat16>();
    else
        dispatch.template operator()<float>();
    CUDA_CHECK(cudaGetLastError());
}

void launch_recurrent_batch_update(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   bool normalize_qk, ConstStateFieldView source_state,
                                   StateFieldView destination_state, const Tensor& valid_columns,
                                   Tensor& out, cudaStream_t stream) {
    const bool group_tiles = q.ne[3] >= 16;
    const auto alignment   = reinterpret_cast<std::uintptr_t>(q.data) |
                           reinterpret_cast<std::uintptr_t>(k.data) |
                           reinterpret_cast<std::uintptr_t>(v.data);
    dim3 grid(v.ne[1] * (group_tiles ? kStateDim / kBlockDv : 1), 1,
              group_tiles ? 1 : kStateDim / kBlockDv);
    const dim3 block(32, kNumWarps);
    const auto dispatch = [&]<class StateT>() {
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
            HeadMap::of(q.ne[1], v.ne[1]),
            scale};
        for (std::int64_t begin = 0; begin < q.ne[3]; begin += 65535) {
            access.batch_begin = static_cast<int>(begin);
            grid.y = static_cast<unsigned>(std::min<std::int64_t>(65535, q.ne[3] - begin));
            const auto submit = [&]<bool Normalize, bool Prefetch>() {
                constexpr auto bytes = Prefetch ? sizeof(subgroup::Window<1>) : 0;
                if (group_tiles)
                    batch_kernel<Normalize, true, Prefetch, StateT>
                        <<<grid, block, bytes, stream>>>(access);
                else
                    batch_kernel<Normalize, false, Prefetch, StateT>
                        <<<grid, block, bytes, stream>>>(access);
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
    };
    if (destination_state.field.tensor.dtype == DType::BF16)
        dispatch.template operator()<__nv_bfloat16>();
    else
        dispatch.template operator()<float>();
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail::gated_delta_net
