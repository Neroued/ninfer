#include "ops/common/host/launch.h"
#include "ops/linear_attention/kimi_delta_attention/chunked/workspace.h"
#include "ops/linear_attention/kimi_delta_attention/launch.h"
#include "ops/common/numeric/convert.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/cuda/mma.cuh"
#include "ops/common/cuda/memory.cuh"

#include <atomic>

namespace ninfer::ops::detail::kimi_delta_attention {
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

// The producer interleaves each eight inner coordinates as {0,4,1,5,2,6,3,7}.
// BF16 A consumes those bits directly; B packing applies the matching permutation.
__device__ __forceinline__ void load_vector_bf16_a(const __nv_bfloat16* p, int k, int lane,
                                                   unsigned (&a)[4]) {
    common::ldmatrix_x4(a[0], a[1], a[2], a[3],
                        common::smem_addr(p + vector_index((lane & 7) + ((lane >> 3) & 1) * 8,
                                                           k + (lane >> 4) * 8)));
}

__device__ __forceinline__ void load_restored_bf16_a(const __nv_bfloat16* p, int key, int lane,
                                                     unsigned (&a)[4]) {
    common::ldmatrix_x4_t(a[0], a[1], a[2], a[3],
                          common::smem_addr(p + vector_index((lane & 7) + (lane >> 4) * 8,
                                                             key + ((lane >> 3) & 1) * 8)));
}

template <int DV>
__device__ __forceinline__ void load_bf16_b(const float* p, int k, int n, int lane, unsigned& b0,
                                            unsigned& b1) {
    const int t = lane & 3, col = n * 8 + lane / 4;
    b0 = common::pack_bf16x2(p[value_index<DV>(k + t, col)], p[value_index<DV>(k + t + 4, col)]);
    b1 = common::pack_bf16x2(p[value_index<DV>(k + t + 8, col)],
                             p[value_index<DV>(k + t + 12, col)]);
}

template <int DV>
struct RecurrenceShared {
    struct alignas(256) Stage {
        Chunk packet;
        __nv_bfloat16 value[kChunkSize * DV];
    } stage[2];

    float snapshot[kStateDim * DV];
    float delta[kChunkSize * DV];
};

struct PacketRange {
    int width, packet, seg_chunk;
};

template <int DV, class... Range>
__device__ __forceinline__ void load_stage(typename RecurrenceShared<DV>::Stage& stage,
                                           const Arguments& args, const Chunk* workspace, int head,
                                           int value_start, int chunk, int tid,
                                           const Range&... range) {
    constexpr bool Packed = sizeof...(Range) != 0;
    const uint4* src;
    if constexpr (Packed) {
        const auto& r = (..., range);
        src = reinterpret_cast<const uint4*>(workspace + static_cast<std::int64_t>(head) * r.width +
                                             r.packet + chunk);
        chunk += r.seg_chunk;
    } else {
        src = reinterpret_cast<const uint4*>(
            workspace + static_cast<std::int64_t>(head) * chunk_count(args.tokens) + chunk);
    }
    auto* dst = reinterpret_cast<uint4*>(&stage.packet);
#pragma unroll
    for (int i = tid; i < static_cast<int>(sizeof(Chunk) / 16); i += kThreads<DV>) {
        common::cp_async<16, Cache::cg>(dst + i, src + i);
    }
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
                                               const float (&prediction)[4], int n, int lane) {
    const int row = lane / 4, col = n * 8 + (lane & 3) * 2;
    const float2 v0 = __bfloat1622float2(common::load_vec<__nv_bfloat162>(value + row * DV + col));
    const float2 v1 =
        __bfloat1622float2(common::load_vec<__nv_bfloat162>(value + (row + 8) * DV + col));
    common::store_vec(delta + value_index<DV>(row, col),
                      make_float2(v0.x - prediction[0], v0.y - prediction[1]));
    common::store_vec(delta + value_index<DV>(row + 8, col),
                      make_float2(v1.x - prediction[2], v1.y - prediction[3]));
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

__device__ __forceinline__ void store_output(const Arguments& args, float (&acc)[4], int chunk,
                                             int head, int value_start, int n, int lane) {
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
    Arguments args, const Chunk* __restrict__ workspace,
    const __grid_constant__ Metadata... metadata) {
    constexpr bool Packed = sizeof...(Metadata) != 0;
    constexpr int kWarps  = kThreads<DV> / 32;
    constexpr int NT      = DV / 8;
    constexpr int MW      = NT < kWarps ? kWarps / NT : 1;
    constexpr int MPW     = (kStateDim / 16) / MW;
    constexpr int NPW     = NT > kWarps ? NT / kWarps : 1;
    extern __shared__ __align__(256) unsigned char backing[];
    auto& sm      = *reinterpret_cast<RecurrenceShared<DV>*>(backing);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    const int g = lane / 4, t = lane & 3;
    const int head        = blockIdx.x / (kStateDim / DV);
    const int value_start = blockIdx.x % (kStateDim / DV) * DV;
    const int n_start     = warp % NT;
    const int m_start     = NT < kWarps ? warp / NT : 0;
    int chunks            = chunk_count(args.tokens);
    PacketRange range{};
    if constexpr (Packed) {
        const auto& packed = (..., metadata);
        const auto window  = packed.window;
        const unsigned row = blockIdx.y + blockIdx.z * gridDim.y;
        if (row >= static_cast<unsigned>(window.segments)) return;
        int index = window.first_segment + row;
        // Give the longest-running CTAs the first wave on skewed windows.
        if (index == window.first_segment)
            index = window.longest_segment;
        else if (index == window.longest_segment)
            index = window.first_segment;
        const auto [s, first] = packed.index()[index];
        args.tokens           = packed.segments.lengths[s];
        const int begin       = max(first, window.begin);
        const int end         = min(first + chunk_count(args.tokens), window.end);
        range                 = {window.end - window.begin, begin - window.begin, begin - first};
        chunks                = end - begin;

        const auto offset =
            static_cast<std::int64_t>(packed.segments.cu_seqlens[s]) * args.value_heads * kStateDim;
        args.v += offset;
        args.out += offset;
        args.destination_offset += s;
        if (begin != first) {
            args.state_in      = args.state_out;
            args.source_offset = args.destination_offset;
        } else {
            args.source_offset += s;
        }
    }
    const auto* state_in  = static_cast<const StateT*>(args.state_in);
    auto* state_out       = static_cast<StateT*>(args.state_out);
    const auto state_base = static_cast<std::int64_t>(head) * kStateDim * kStateDim;
    const auto src_base =
        state_base +
        (args.source_offset ? *args.source_offset / static_cast<std::int64_t>(sizeof(StateT)) : 0);
    const auto dst_base =
        state_base + (args.destination_offset
                          ? *args.destination_offset / static_cast<std::int64_t>(sizeof(StateT))
                          : 0);

    // MMA accumulators describe S^T[key,value]. External state remains [value,key].
    float state[NPW][MPW][4];
#pragma unroll
    for (int ni = 0; ni < NPW; ++ni) {
        const int n = n_start + ni * kWarps;
#pragma unroll
        for (int mi = 0; mi < MPW; ++mi) {
            const int key    = (m_start + mi * MW) * 16 + g;
            const int value  = value_start + n * 8 + t * 2;
            state[ni][mi][0] = common::load_state_scalar(state_in + src_base + value * kStateDim + key);
            state[ni][mi][1] =
                common::load_state_scalar(state_in + src_base + (value + 1) * kStateDim + key);
            state[ni][mi][2] =
                common::load_state_scalar(state_in + src_base + value * kStateDim + key + 8);
            state[ni][mi][3] =
                common::load_state_scalar(state_in + src_base + (value + 1) * kStateDim + key + 8);
        }
    }
    if constexpr (Packed) {
        load_stage<DV>(sm.stage[0], args, workspace, head, value_start, 0, tid, range);
        if (chunks > 1)
            load_stage<DV>(sm.stage[1], args, workspace, head, value_start, 1, tid, range);
    } else {
        load_stage<DV>(sm.stage[0], args, workspace, head, value_start, 0, tid);
        if (chunks > 1) load_stage<DV>(sm.stage[1], args, workspace, head, value_start, 1, tid);
    }

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
        const auto& stage = sm.stage[chunk & 1];
        const Chunk& p    = stage.packet;
        float history[NPW][4]{};

        if constexpr (NT >= kWarps) {
            float prediction[NPW][4]{};
#pragma unroll
            for (int k = 0; k < kStateDim; k += 16) {
                unsigned ka0[4], ka1[4], qa[4];
                load_vector_tf32_a(p.kd, k, lane, ka0);
                load_vector_tf32_a(p.kd, k + 8, lane, ka1);
                load_vector_bf16_a(p.qd, k, lane, qa);
#pragma unroll
                for (int ni = 0; ni < NPW; ++ni) {
                    const int n    = warp + ni * kWarps;
                    const float b0 = sm.snapshot[value_index<DV>(k + t, n * 8 + g)];
                    const float b1 = sm.snapshot[value_index<DV>(k + t + 4, n * 8 + g)];
                    const float b2 = sm.snapshot[value_index<DV>(k + t + 8, n * 8 + g)];
                    const float b3 = sm.snapshot[value_index<DV>(k + t + 12, n * 8 + g)];
                    common::mma_tf32_bits(prediction[ni][0], prediction[ni][1], prediction[ni][2],
                                          prediction[ni][3], ka0[0], ka0[1], ka0[2], ka0[3],
                                          __float_as_uint(b0), __float_as_uint(b1));
                    common::mma_tf32_bits(prediction[ni][0], prediction[ni][1], prediction[ni][2],
                                          prediction[ni][3], ka1[0], ka1[1], ka1[2], ka1[3],
                                          __float_as_uint(b2), __float_as_uint(b3));
                    common::mma_bf16(history[ni][0], history[ni][1], history[ni][2], history[ni][3],
                                     qa[0], qa[1], qa[2], qa[3], common::pack_bf16x2(b0, b1),
                                     common::pack_bf16x2(b2, b3));
                }
            }
#pragma unroll
            for (int ni = 0; ni < NPW; ++ni) {
                store_residual<DV>(sm.delta, stage.value, prediction[ni], warp + ni * kWarps, lane);
            }
        } else {
            if (warp < 2 * NT) {
                const bool query = warp >= NT;
                const int n      = warp % NT;
                if (query) {
#pragma unroll
                    for (int k = 0; k < kStateDim; k += 16) {
                        unsigned af[4], b0, b1;
                        load_vector_bf16_a(p.qd, k, lane, af);
                        load_bf16_b<DV>(sm.snapshot, k, n, lane, b0, b1);
                        common::mma_bf16(history[0][0], history[0][1], history[0][2], history[0][3],
                                         af[0], af[1], af[2], af[3], b0, b1);
                    }
                } else {
#pragma unroll
                    for (int k = 0; k < kStateDim; k += 8) {
                        unsigned af[4];
                        load_vector_tf32_a(p.kd, k, lane, af);
                        common::mma_tf32_bits(
                            history[0][0], history[0][1], history[0][2], history[0][3], af[0],
                            af[1], af[2], af[3],
                            __float_as_uint(sm.snapshot[value_index<DV>(k + t, n * 8 + g)]),
                            __float_as_uint(sm.snapshot[value_index<DV>(k + t + 4, n * 8 + g)]));
                    }
                    store_residual<DV>(sm.delta, stage.value, history[0], n, lane);
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
            common::store_vec(sm.delta + value_index<DV>(g, col), make_float2(delta[0], delta[1]));
            common::store_vec(sm.delta + value_index<DV>(g + 8, col),
                              make_float2(delta[2], delta[3]));
        }
        __syncthreads();

        if constexpr (NT >= kWarps) {
#pragma unroll
            for (int ni = 0; ni < NPW; ++ni) {
                const int n = warp + ni * kWarps;
                matrix_delta<DV>(history[ni], p.mqk, sm.delta, n, lane);
                store_output(args, history[ni], Packed ? chunk + range.seg_chunk : chunk, head,
                             value_start, n, lane);
            }
        } else if (warp >= NT && warp < 2 * NT) {
            const int n = warp - NT;
            matrix_delta<DV>(history[0], p.mqk, sm.delta, n, lane);
            store_output(args, history[0], Packed ? chunk + range.seg_chunk : chunk, head,
                         value_start, n, lane);
        }

        // S^T = diag(gamma) S^T + Kr^T Delta. State remains an FP32 accumulator.
        // Convert Delta only for this MMA and reuse its fragment across all key tiles.
        // The output branch above consumes the original FP32 Delta through TF32 MMA.
        unsigned delta_b[NPW][2];
#pragma unroll
        for (int ni = 0; ni < NPW; ++ni) {
            load_bf16_b<DV>(sm.delta, 0, n_start + ni * kWarps, lane, delta_b[ni][0],
                            delta_b[ni][1]);
        }
#pragma unroll
        for (int mi = 0; mi < MPW; ++mi) {
            const int key  = (m_start + mi * MW) * 16 + g;
            const float g0 = p.gamma[key], g1 = p.gamma[key + 8];
#pragma unroll
            for (int ni = 0; ni < NPW; ++ni) {
                state[ni][mi][0] *= g0;
                state[ni][mi][1] *= g0;
                state[ni][mi][2] *= g1;
                state[ni][mi][3] *= g1;
            }
            unsigned af[4];
            load_restored_bf16_a(p.kr, key - g, lane, af);
#pragma unroll
            for (int ni = 0; ni < NPW; ++ni) {
                common::mma_bf16(state[ni][mi][0], state[ni][mi][1], state[ni][mi][2],
                                 state[ni][mi][3], af[0], af[1], af[2], af[3], delta_b[ni][0],
                                 delta_b[ni][1]);
            }
        }
        __syncthreads();
        if (chunk + 2 < chunks) {
            if constexpr (Packed)
                load_stage<DV>(sm.stage[chunk & 1], args, workspace, head, value_start, chunk + 2,
                               tid, range);
            else
                load_stage<DV>(sm.stage[chunk & 1], args, workspace, head, value_start, chunk + 2,
                               tid);
        }
    }
#pragma unroll
    for (int ni = 0; ni < NPW; ++ni) {
        const int n = n_start + ni * kWarps;
#pragma unroll
        for (int mi = 0; mi < MPW; ++mi) {
            const int key   = (m_start + mi * MW) * 16 + g;
            const int value = value_start + n * 8 + t * 2;
            common::store_state_scalar(state_out + dst_base + value * kStateDim + key,
                                       state[ni][mi][0]);
            common::store_state_scalar(state_out + dst_base + (value + 1) * kStateDim + key,
                                       state[ni][mi][1]);
            common::store_state_scalar(state_out + dst_base + value * kStateDim + key + 8,
                                       state[ni][mi][2]);
            common::store_state_scalar(state_out + dst_base + (value + 1) * kStateDim + key + 8,
                                       state[ni][mi][3]);
        }
    }
}

template <int DV, class StateT, class... Metadata>
void launch(const Arguments& args, const Chunk* workspace, CudaExecutionView execution,
            const Metadata&... metadata) {
    constexpr int shared_bytes = sizeof(RecurrenceShared<DV>);
    if constexpr (shared_bytes > 48 * 1024) {
        static std::atomic<std::uint64_t> configured{0};
        const int device = execution.device().index();
        const auto bit   = device < 64 ? std::uint64_t{1} << device : 0;
        if ((configured.load(std::memory_order_acquire) & bit) == 0) {
            CUDA_CHECK(cudaFuncSetAttribute(
                reinterpret_cast<const void*>(chunk_recurrence_kernel<DV, StateT, Metadata...>),
                cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
            if (bit) configured.fetch_or(bit, std::memory_order_release);
        }
    }
    dim3 grid(args.value_heads * (kStateDim / DV));
    if constexpr (sizeof...(Metadata) != 0) {
        const auto window = (..., metadata).window;
        grid.y            = std::min(window.segments, common::kCudaGridYLimit);
        grid.z            = common::div_up(static_cast<unsigned>(window.segments), grid.y);
    }
    chunk_recurrence_kernel<DV, StateT, Metadata...>
        <<<grid, kThreads<DV>, shared_bytes, execution.stream()>>>(args, workspace, metadata...);
    CUDA_CHECK(cudaGetLastError());
}

template <class StateT, class... Metadata>
void dispatch_tile(const Arguments& args, const Chunk* workspace, int value_tile,
                   CudaExecutionView execution, const Metadata&... metadata) {
    switch (value_tile) {
    case 64:
        return launch<64, StateT>(args, workspace, execution, metadata...);
    case 32:
        return launch<32, StateT>(args, workspace, execution, metadata...);
    case 16:
        return launch<16, StateT>(args, workspace, execution, metadata...);
    case 8:
        return launch<8, StateT>(args, workspace, execution, metadata...);
    }
    throw std::logic_error("linear attention: invalid recurrence value tile");
}

template <class... Metadata>
void dispatch(const Arguments& args, const Chunk* workspace, int value_tile,
              CudaExecutionView execution, const Metadata&... metadata) {
    switch (args.state_dtype) {
    case DType::FP32:
        return dispatch_tile<float>(args, workspace, value_tile, execution, metadata...);
    case DType::BF16:
        return dispatch_tile<__nv_bfloat16>(args, workspace, value_tile, execution, metadata...);
    default:
        throw std::logic_error("linear attention: unsupported recurrent state dtype");
    }
}

} // namespace

void launch_chunk_recurrence(const Arguments& args, const Chunk* workspace, int value_tile,
                             CudaExecutionView execution) {
    dispatch(args, workspace, value_tile, execution);
}

void launch_chunk_recurrence(const Arguments& args, const Chunk* workspace, int value_tile,
                             const PackedChunks& packed, CudaExecutionView execution) {
    dispatch(args, workspace, value_tile, execution, packed);
}

} // namespace ninfer::ops::detail::kimi_delta_attention
