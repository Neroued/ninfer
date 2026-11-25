#include "ops/common/host/launch.h"
#include "ops/linear_attention/gated_delta_net/recurrence/chunked/launch.h"
#include "ops/common/numeric/convert.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/cuda/mma.cuh"
#include "ops/common/cuda/memory.cuh"

#include <atomic>
#include <climits>

namespace ninfer::ops::detail::gated_delta_net::chunked {
namespace {

template <int DV>
inline constexpr int kThreads = DV >= 32 ? 256 : 128;

template <int Width>
__device__ __forceinline__ int value_index(int row, int col) {
    if constexpr (Width == 8) return row * Width + col;
    if constexpr (Width == 16) return row * Width + (col ^ (((row >> 1) & 1) * 8));
    return row * Width + (col ^ ((row & 3) * 8));
}

__device__ __forceinline__ void load_vector_tf32_a(const __nv_bfloat16* p, int k, int lane,
                                                   unsigned (&a)[4]) {
    unsigned low, high;
    common::ldmatrix_x2(low, high,
                        common::smem_addr(p + vector_index((lane & 7) + ((lane >> 3) & 1) * 8, k)));
    common::unpack_bf16x2_to_fp32_bits(low, a[0], a[2]);
    common::unpack_bf16x2_to_fp32_bits(high, a[1], a[3]);
}

// Q/K retain their native BF16 coordinates. TF32 A expands adjacent pairs;
// its FP32 B loads follow the corresponding even/odd inner-coordinate order.
__device__ __forceinline__ void load_vector_bf16_a(const __nv_bfloat16* p, int k, int lane,
                                                   unsigned (&a)[4]) {
    common::ldmatrix_x4(a[0], a[1], a[2], a[3],
                        common::smem_addr(p + vector_index((lane & 7) + ((lane >> 3) & 1) * 8,
                                                           k + (lane >> 4) * 8)));
}

__device__ __forceinline__ void load_transposed_bf16_a(const __nv_bfloat16* p, int key, int lane,
                                                       unsigned (&a)[4]) {
    common::ldmatrix_x4_t(a[0], a[1], a[2], a[3],
                          common::smem_addr(p + vector_index((lane & 7) + (lane >> 4) * 8,
                                                             key + ((lane >> 3) & 1) * 8)));
}

template <int DV>
__device__ __forceinline__ void load_bf16_b(const float* p, int k, int n, int lane, unsigned& b0,
                                            unsigned& b1) {
    const int t = lane & 3, col = n * 8 + lane / 4;
    b0 = common::pack_bf16x2(p[value_index<DV>(k + 2 * t, col)],
                             p[value_index<DV>(k + 2 * t + 1, col)]);
    b1 = common::pack_bf16x2(p[value_index<DV>(k + 2 * t + 8, col)],
                             p[value_index<DV>(k + 2 * t + 9, col)]);
}

template <int DV>
struct RecurrenceShared {
    struct alignas(256) Stage {
        QkChunk qk;
        ControlChunk control;
        __nv_bfloat16 value[kChunkSize * DV];
    } stage[2];

    float snapshot[kStateDim * DV];
    float delta[kChunkSize * DV];
};

struct SegmentPackets {
    const QkChunk* qk;
    const ControlChunk* control;
};

template <int DV, bool Packed>
__device__ __forceinline__ void
load_stage(typename RecurrenceShared<DV>::Stage& stage, const RecurrenceOperands& args,
           const QkChunk* qk_workspace, const ControlChunk* control_workspace, int head,
           int value_start, int chunk, int tid, SegmentPackets packets) {
    const int qh       = head / (args.value_heads / args.qk_heads);
    const int chunks   = chunk_count(args.tokens);
    const auto* qk_src = reinterpret_cast<const uint4*>(
        Packed ? packets.qk + chunk
               : qk_workspace + static_cast<std::int64_t>(qh) * chunks + chunk);
    auto* qk_dst = reinterpret_cast<uint4*>(&stage.qk);
#pragma unroll
    for (int i = tid; i < sizeof(QkChunk) / 16; i += kThreads<DV>)
        common::cp_async<16, Cache::cg>(qk_dst + i, qk_src + i);
    const auto* ctrl_src = reinterpret_cast<const uint4*>(
        Packed ? packets.control + chunk
               : control_workspace + static_cast<std::int64_t>(head) * chunks + chunk);
    auto* ctrl_dst = reinterpret_cast<uint4*>(&stage.control);
#pragma unroll
    for (int i = tid; i < sizeof(ControlChunk) / 16; i += kThreads<DV>)
        common::cp_async<16, Cache::cg>(ctrl_dst + i, ctrl_src + i);
    constexpr int vectors = kChunkSize * DV / 8;
#pragma unroll
    for (int i = tid; i < vectors; i += kThreads<DV>) {
        const int row = i / (DV / 8), col = i % (DV / 8) * 8;
        const int token       = chunk * kChunkSize + row;
        const int valid_token = token < args.tokens ? token : 0;
        const auto off =
            (static_cast<std::int64_t>(valid_token) * args.value_heads + head) * kStateDim +
            value_start + col;
        common::cp_async_zfill<16, Cache::cg>(stage.value + row * DV + col, args.v + off,
                                              token < args.tokens ? 16 : 0);
    }
    common::cp_commit();
}

template <int DV>
__device__ __forceinline__ void store_residual(float* delta, const __nv_bfloat16* value,
                                               const float (&prediction)[4], const float* prefix,
                                               int n, int lane) {
    const int row = lane / 4, col = n * 8 + (lane & 3) * 2;
    const float2 v0 = __bfloat1622float2(common::load_vec<__nv_bfloat162>(value + row * DV + col));
    const float2 v1 =
        __bfloat1622float2(common::load_vec<__nv_bfloat162>(value + (row + 8) * DV + col));
    common::store_vec(
        delta + value_index<DV>(row, col),
        make_float2(v0.x - prefix[row] * prediction[0], v0.y - prefix[row] * prediction[1]));
    common::store_vec(delta + value_index<DV>(row + 8, col),
                      make_float2(v1.x - prefix[row + 8] * prediction[2],
                                  v1.y - prefix[row + 8] * prediction[3]));
}

template <int DV>
__device__ __forceinline__ void matrix_delta(float (&acc)[4], const float* matrix,
                                             const float* delta, int n, int lane) {
    const int g = lane / 4, t = lane & 3;
#pragma unroll
    for (int k = 0; k < kChunkSize; k += 8) {
        unsigned af[4];
        common::ldmatrix_x4(
            af[0], af[1], af[2], af[3],
            common::smem_addr(
                matrix + square_index((lane & 7) + ((lane >> 3) & 1) * 8, k + (lane >> 4) * 4)));
        common::mma_tf32_bits(acc[0], acc[1], acc[2], acc[3], af[0], af[1], af[2], af[3],
                              __float_as_uint(delta[value_index<DV>(k + t, n * 8 + g)]),
                              __float_as_uint(delta[value_index<DV>(k + t + 4, n * 8 + g)]));
    }
}

__device__ __forceinline__ void store_output(const RecurrenceOperands& args, float (&acc)[4],
                                             int chunk, int head, int value_start, int n,
                                             int lane) {
    const int row = lane / 4, col = value_start + n * 8 + (lane & 3) * 2;
    const int t0  = chunk * kChunkSize + row;
    const auto o0 = (static_cast<std::int64_t>(t0) * args.value_heads + head) * kStateDim + col;
    if (t0 < args.tokens) {
        common::store_vec(args.out + o0,
                          __floats2bfloat162_rn(acc[0] * args.scale, acc[1] * args.scale));
    }
    if (t0 + 8 < args.tokens) {
        common::store_vec(args.out + o0 +
                              static_cast<std::int64_t>(8) * args.value_heads * kStateDim,
                          __floats2bfloat162_rn(acc[2] * args.scale, acc[3] * args.scale));
    }
}

template <int DV, class StateT, class... Metadata>
__global__ __launch_bounds__(kThreads<DV>, (DV >= 32 ? 2 : 4)) void chunk_recurrence_kernel(
    RecurrenceOperands base_args, const QkChunk* __restrict__ qk_workspace,
    const ControlChunk* __restrict__ control_workspace,
    const __grid_constant__ Metadata... metadata) {
    constexpr bool Packed = sizeof...(Metadata) != 0;
    constexpr int kWarps = kThreads<DV> / 32;
    constexpr int NT     = DV / 8;
    constexpr int MW     = NT < kWarps ? kWarps / NT : 1;
    constexpr int MPW    = (kStateDim / 16) / MW;
    constexpr int NPW    = NT > kWarps ? NT / kWarps : 1;
    extern __shared__ __align__(256) unsigned char backing[];
    auto& sm      = *reinterpret_cast<RecurrenceShared<DV>*>(backing);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    const int g = lane / 4, t = lane & 3;
    const PackedChunks* packed = nullptr;
    if constexpr (Packed) packed = &(..., metadata);
    unsigned i = Packed ? blockIdx.y : 0;
    do {
        auto args = base_args;
        int head, value_start, chunks;
        SegmentPackets packets{};
        if constexpr (Packed) {
            constexpr int stripes = kStateDim / DV;
            head                  = gridDim.z == 1 ? blockIdx.x / stripes : blockIdx.x;
            value_start           = (gridDim.z == 1 ? blockIdx.x % stripes : blockIdx.z) * DV;
            const auto* index     = packed->index();
            const auto [s, first] = index[i];
            const auto start      = static_cast<std::int64_t>(packed->segments.cu_seqlens[s]) *
                               args.value_heads * kStateDim;
            // Use segment-local tokens in load_stage and store_output, including their padding
            // predicates. Rebase the shared packets independently of the physical token rows.
            args.v += start;
            args.out += start;
            args.tokens = packed->segments.lengths[s];
            args.source_offset += s;
            args.destination_offset += s;
            const int end =
                i + 1 < packed->chunked_segments ? index[i + 1].first_chunk : packed->total_chunks;
            chunks       = end - first;
            const int qh = head / (args.value_heads / args.qk_heads);
            packets = {qk_workspace + static_cast<std::int64_t>(qh) * packed->total_chunks + first,
                       control_workspace + static_cast<std::int64_t>(head) * packed->total_chunks +
                           first};
        } else {
            head        = blockIdx.x / (kStateDim / DV);
            value_start = blockIdx.x % (kStateDim / DV) * DV;
            chunks      = chunk_count(args.tokens);
        }
        const int n_start     = warp % NT;
        const int m_start     = NT < kWarps ? warp / NT : 0;
        const auto state_base = static_cast<std::int64_t>(head) * kStateDim * kStateDim;
        const auto source_base =
            state_base +
            (args.source_offset ? *args.source_offset / std::int64_t(sizeof(StateT)) : 0);
        const auto destination_base =
            state_base +
            (args.destination_offset ? *args.destination_offset / std::int64_t(sizeof(StateT)) : 0);
        const auto* state_in = static_cast<const StateT*>(args.state_in);
        auto* state_out      = static_cast<StateT*>(args.state_out);

        // MMA accumulators describe S^T[key,value]. External state remains [value,key].
        float state[NPW][MPW][4];
#pragma unroll
        for (int ni = 0; ni < NPW; ++ni) {
            const int n = n_start + ni * kWarps;
#pragma unroll
            for (int mi = 0; mi < MPW; ++mi) {
                const int key   = (m_start + mi * MW) * 16 + g;
                const int value = value_start + n * 8 + t * 2;
                state[ni][mi][0] =
                    common::load_state_scalar(state_in + source_base + value * kStateDim + key);
                state[ni][mi][1] = common::load_state_scalar(
                    state_in + source_base + (value + 1) * kStateDim + key);
                state[ni][mi][2] = common::load_state_scalar(
                    state_in + source_base + value * kStateDim + key + 8);
                state[ni][mi][3] = common::load_state_scalar(
                    state_in + source_base + (value + 1) * kStateDim + key + 8);
            }
        }
        load_stage<DV, Packed>(sm.stage[0], args, qk_workspace, control_workspace, head,
                               value_start, 0, tid, packets);
        if (chunks > 1)
            load_stage<DV, Packed>(sm.stage[1], args, qk_workspace, control_workspace, head,
                                   value_start, 1, tid, packets);

        for (int chunk = 0; chunk < chunks; ++chunk) {
            if (chunk + 1 < chunks)
                common::cp_wait<1>();
            else
                common::cp_wait<0>();
#pragma unroll
            for (int ni = 0; ni < NPW; ++ni) {
                const int n = n_start + ni * kWarps;
#pragma unroll
                for (int mi = 0; mi < MPW; ++mi) {
                    const int key = (m_start + mi * MW) * 16 + g;
                    const int col = n * 8 + t * 2;
                    common::store_vec(sm.snapshot + value_index<DV>(key, col),
                                      make_float2(state[ni][mi][0], state[ni][mi][1]));
                    common::store_vec(sm.snapshot + value_index<DV>(key + 8, col),
                                      make_float2(state[ni][mi][2], state[ni][mi][3]));
                }
            }
            __syncthreads();
            const auto& stage     = sm.stage[chunk & 1];
            const ControlChunk& p = stage.control;
            const QkChunk& qk     = stage.qk;
            float history[NPW][4]{};

            if constexpr (NT >= kWarps) {
                float prediction[NPW][4]{};
#pragma unroll
                for (int k = 0; k < kStateDim; k += 16) {
                    unsigned ka0[4], ka1[4], qa[4];
                    load_vector_tf32_a(qk.k, k, lane, ka0);
                    load_vector_tf32_a(qk.k, k + 8, lane, ka1);
                    load_vector_bf16_a(qk.q, k, lane, qa);
#pragma unroll
                    for (int ni = 0; ni < NPW; ++ni) {
                        const int n    = warp + ni * kWarps;
                        const float b0 = sm.snapshot[value_index<DV>(k + 2 * t, n * 8 + g)];
                        const float b1 = sm.snapshot[value_index<DV>(k + 2 * t + 1, n * 8 + g)];
                        const float b2 = sm.snapshot[value_index<DV>(k + 2 * t + 8, n * 8 + g)];
                        const float b3 = sm.snapshot[value_index<DV>(k + 2 * t + 9, n * 8 + g)];
                        common::mma_tf32_bits(prediction[ni][0], prediction[ni][1],
                                              prediction[ni][2], prediction[ni][3], ka0[0], ka0[1],
                                              ka0[2], ka0[3], __float_as_uint(b0),
                                              __float_as_uint(b1));
                        common::mma_tf32_bits(prediction[ni][0], prediction[ni][1],
                                              prediction[ni][2], prediction[ni][3], ka1[0], ka1[1],
                                              ka1[2], ka1[3], __float_as_uint(b2),
                                              __float_as_uint(b3));
                        common::mma_bf16(history[ni][0], history[ni][1], history[ni][2],
                                         history[ni][3], qa[0], qa[1], qa[2], qa[3],
                                         common::pack_bf16x2(b0, b1), common::pack_bf16x2(b2, b3));
                    }
                }
#pragma unroll
                for (int ni = 0; ni < NPW; ++ni) {
                    store_residual<DV>(sm.delta, stage.value, prediction[ni], p.prefix,
                                       warp + ni * kWarps, lane);
                }
            } else {
                if (warp < 2 * NT) {
                    const bool query = warp >= NT;
                    const int n      = warp % NT;
                    if (query) {
#pragma unroll
                        for (int k = 0; k < kStateDim; k += 16) {
                            unsigned af[4], b0, b1;
                            load_vector_bf16_a(qk.q, k, lane, af);
                            load_bf16_b<DV>(sm.snapshot, k, n, lane, b0, b1);
                            common::mma_bf16(history[0][0], history[0][1], history[0][2],
                                             history[0][3], af[0], af[1], af[2], af[3], b0, b1);
                        }
                    } else {
#pragma unroll
                        for (int k = 0; k < kStateDim; k += 8) {
                            unsigned af[4];
                            load_vector_tf32_a(qk.k, k, lane, af);
                            common::mma_tf32_bits(
                                history[0][0], history[0][1], history[0][2], history[0][3], af[0],
                                af[1], af[2], af[3],
                                __float_as_uint(sm.snapshot[value_index<DV>(k + 2 * t, n * 8 + g)]),
                                __float_as_uint(
                                    sm.snapshot[value_index<DV>(k + 2 * t + 1, n * 8 + g)]));
                        }
                        store_residual<DV>(sm.delta, stage.value, history[0], p.prefix, n, lane);
                    }
                }
            }
            __syncthreads();

            // Each value tile reads only its own R columns, so R can become Delta in place.
#pragma unroll
            for (int n = warp; n < NT; n += kWarps) {
                float delta[4]{};
                matrix_delta<DV>(delta, p.solve, sm.delta, n, lane);
                const int col = n * 8 + t * 2;
                __syncwarp();
                common::store_vec(sm.delta + value_index<DV>(g, col),
                                  make_float2(delta[0], delta[1]));
                common::store_vec(sm.delta + value_index<DV>(g + 8, col),
                                  make_float2(delta[2], delta[3]));
            }
            __syncthreads();

            if constexpr (NT >= kWarps) {
#pragma unroll
                for (int ni = 0; ni < NPW; ++ni) {
                    const int n = warp + ni * kWarps;
                    history[ni][0] *= p.prefix[g];
                    history[ni][1] *= p.prefix[g];
                    history[ni][2] *= p.prefix[g + 8];
                    history[ni][3] *= p.prefix[g + 8];
                    matrix_delta<DV>(history[ni], p.mqk, sm.delta, n, lane);
                    store_output(args, history[ni], chunk, head, value_start, n, lane);
                }
            } else if (warp >= NT && warp < 2 * NT) {
                const int n = warp - NT;
                history[0][0] *= p.prefix[g];
                history[0][1] *= p.prefix[g];
                history[0][2] *= p.prefix[g + 8];
                history[0][3] *= p.prefix[g + 8];
                matrix_delta<DV>(history[0], p.mqk, sm.delta, n, lane);
                store_output(args, history[0], chunk, head, value_start, n, lane);
            }

            // S^T = gamma S^T + K^T diag(suffix) Delta. The master state stays FP32.
            // Convert Delta only for this MMA and reuse its fragment across all key tiles.
            // The output branch above consumes the original FP32 Delta through TF32 MMA.
            unsigned delta_b[NPW][2];
#pragma unroll
            for (int ni = 0; ni < NPW; ++ni) {
                const int col  = (n_start + ni * kWarps) * 8 + g;
                delta_b[ni][0] = common::pack_bf16x2(
                    p.suffix[2 * t] * sm.delta[value_index<DV>(2 * t, col)],
                    p.suffix[2 * t + 1] * sm.delta[value_index<DV>(2 * t + 1, col)]);
                delta_b[ni][1] = common::pack_bf16x2(
                    p.suffix[2 * t + 8] * sm.delta[value_index<DV>(2 * t + 8, col)],
                    p.suffix[2 * t + 9] * sm.delta[value_index<DV>(2 * t + 9, col)]);
            }
#pragma unroll
            for (int mi = 0; mi < MPW; ++mi) {
                const int key     = (m_start + mi * MW) * 16 + g;
                const float decay = p.prefix[kChunkSize - 1];
#pragma unroll
                for (int ni = 0; ni < NPW; ++ni) {
                    state[ni][mi][0] *= decay;
                    state[ni][mi][1] *= decay;
                    state[ni][mi][2] *= decay;
                    state[ni][mi][3] *= decay;
                }
                unsigned af[4];
                load_transposed_bf16_a(qk.k, key - g, lane, af);
#pragma unroll
                for (int ni = 0; ni < NPW; ++ni) {
                    common::mma_bf16(state[ni][mi][0], state[ni][mi][1], state[ni][mi][2],
                                     state[ni][mi][3], af[0], af[1], af[2], af[3], delta_b[ni][0],
                                     delta_b[ni][1]);
                }
            }
            __syncthreads();
            if (chunk + 2 < chunks) {
                load_stage<DV, Packed>(sm.stage[chunk & 1], args, qk_workspace, control_workspace,
                                       head, value_start, chunk + 2, tid, packets);
            }
        }
#pragma unroll
        for (int ni = 0; ni < NPW; ++ni) {
            const int n = n_start + ni * kWarps;
#pragma unroll
            for (int mi = 0; mi < MPW; ++mi) {
                const int key   = (m_start + mi * MW) * 16 + g;
                const int value = value_start + n * 8 + t * 2;
                common::store_state_scalar(state_out + destination_base + value * kStateDim + key,
                                           state[ni][mi][0]);
                common::store_state_scalar(
                    state_out + destination_base + (value + 1) * kStateDim + key, state[ni][mi][1]);
                common::store_state_scalar(
                    state_out + destination_base + value * kStateDim + key + 8, state[ni][mi][2]);
                common::store_state_scalar(
                    state_out + destination_base + (value + 1) * kStateDim + key + 8,
                    state[ni][mi][3]);
            }
        }
        if constexpr (Packed) i += gridDim.y;
    } while (Packed && i < packed->chunked_segments);
}

template <int DV, class StateT, class... Metadata>
void opt_in_shared_memory() {
    constexpr int shared_bytes = sizeof(RecurrenceShared<DV>);
    if constexpr (shared_bytes > 48 * 1024) {
        static std::atomic<std::uint64_t> configured{0};
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        // CUDA's device ordinal is nonnegative. Fall back to opting in at launch
        // for ordinals beyond the mask.
        const auto bit = device < 64 ? std::uint64_t{1} << device : 0;
        if ((configured.load(std::memory_order_acquire) & bit) == 0) {
            // NVCC's host stub retains the const qualification on the argument pack.
            CUDA_CHECK(cudaFuncSetAttribute(
                reinterpret_cast<const void*>(chunk_recurrence_kernel<DV, StateT, Metadata...>),
                cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
            if (bit) configured.fetch_or(bit, std::memory_order_release);
        }
    }
}

template <int DV, class StateT>
void launch_ragged(const RecurrenceOperands& args, const PackedChunks& packed, const QkChunk* qk,
                   const ControlChunk* control, cudaStream_t stream) {
    constexpr int stripes      = kStateDim / DV;
    constexpr int shared_bytes = sizeof(RecurrenceShared<DV>);
    opt_in_shared_memory<DV, StateT, PackedChunks>();
    const int segment_grid = std::min(packed.chunked_segments, common::kCudaGridYLimit);
    const dim3 grid        = args.value_heads <= INT_MAX / stripes
                                 ? dim3(args.value_heads * stripes, segment_grid)
                                 : dim3(args.value_heads, segment_grid, stripes);
    chunk_recurrence_kernel<DV, StateT, PackedChunks>
        <<<grid, kThreads<DV>, shared_bytes, stream>>>(args, qk, control, packed);
    CUDA_CHECK(cudaGetLastError());
}

template <int DV, class StateT>
void launch(const RecurrenceOperands& args, const QkChunk* qk, const ControlChunk* control,
            cudaStream_t stream) {
    constexpr int shared_bytes = sizeof(RecurrenceShared<DV>);
    opt_in_shared_memory<DV, StateT>();
    chunk_recurrence_kernel<DV, StateT>
        <<<args.value_heads*(kStateDim / DV), kThreads<DV>, shared_bytes, stream>>>(args, qk,
                                                                                    control);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_recurrence_ragged(const RecurrenceOperands& args, const PackedChunks& packed,
                              const QkChunk* qk, const ControlChunk* control, int value_tile,
                              cudaStream_t stream) {
    const auto dispatch = [&]<class StateT>() {
        switch (value_tile) {
        case 64:
            return launch_ragged<64, StateT>(args, packed, qk, control, stream);
        case 32:
            return launch_ragged<32, StateT>(args, packed, qk, control, stream);
        case 16:
            return launch_ragged<16, StateT>(args, packed, qk, control, stream);
        case 8:
            return launch_ragged<8, StateT>(args, packed, qk, control, stream);
        }
        throw std::logic_error("linear attention: invalid recurrence value tile");
    };
    if (args.state_dtype == DType::BF16)
        dispatch.template operator()<__nv_bfloat16>();
    else
        dispatch.template operator()<float>();
}

void launch_recurrence(const RecurrenceOperands& args, const QkChunk* qk,
                       const ControlChunk* control, int value_tile, cudaStream_t stream) {
    const auto dispatch = [&]<class StateT>() {
        switch (value_tile) {
        case 64:
            return launch<64, StateT>(args, qk, control, stream);
        case 32:
            return launch<32, StateT>(args, qk, control, stream);
        case 16:
            return launch<16, StateT>(args, qk, control, stream);
        case 8:
            return launch<8, StateT>(args, qk, control, stream);
        }
        throw std::logic_error("linear attention: invalid recurrence value tile");
    };
    if (args.state_dtype == DType::BF16)
        dispatch.template operator()<__nv_bfloat16>();
    else
        dispatch.template operator()<float>();
}

} // namespace ninfer::ops::detail::gated_delta_net::chunked
