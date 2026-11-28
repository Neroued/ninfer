#include "ninfer/ops/moe_exchange.h"
#include "core/layout.h"
#include "ops/moe/operands.h"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/quant/nvfp4.cuh"
#include "ops/common/numeric/math.cuh"

#include <algorithm>
#include <climits>

namespace ninfer::ops {
namespace {
constexpr int Tile = 4;

struct Job {
    int expert, begin, end;
};

void validate(const RoutedSwiGLUProfile& p, int capacity) {
    const auto& f = p.ffn;
    const bool nv =
        f.gate_up_type == QType::NVFP4 && f.gate_up_layout == QuantLayout::BlockScaleK16M128x4;
    const bool fp128 = f.gate_up_type == QType::FP8_E4M3FN_BLOCK128X128_BF16 &&
                       f.gate_up_layout == QuantLayout::BlockScaleM128K128;
    const bool fp64 = f.gate_up_type == QType::FP8_E4M3FN_BLOCK64X64_BF16 &&
                      f.gate_up_layout == QuantLayout::BlockScaleM64K64;
    if (!(nv || fp128 || fp64) || f.down_type != f.gate_up_type ||
        f.down_layout != f.gate_up_layout || !valid_linear_policy(f.gate_up_policy) ||
        !valid_linear_policy(f.down_policy))
        throw std::invalid_argument("compact routed SwiGLU: unsupported expert representation");
    const int alignment = nv ? 16 : fp128 ? 128 : 64;
    if (f.hidden <= 0 || f.intermediate <= 0 || f.intermediate > INT_MAX / 2 ||
        f.hidden % alignment || f.intermediate % alignment || capacity <= 0 || p.experts <= 0 ||
        p.top_k <= 0 || p.top_k > p.experts ||
        (f.clamp_limit && (!std::isfinite(*f.clamp_limit) || *f.clamp_limit <= 0)))
        throw std::invalid_argument("compact routed SwiGLU: invalid contraction geometry");
}

struct Scratch {
    int* count;
    Job* jobs;
    float* product;
};

template <class Arena>
Scratch allocate(Arena& a, int m, int intermediate) {
    return {static_cast<int*>(a.alloc_bytes(sizeof(int)).data),
            static_cast<Job*>(a.alloc_bytes(std::size_t(m) * sizeof(Job)).data),
            static_cast<float*>(a.alloc(DType::FP32, {intermediate, m}).data)};
}

__global__ void make_jobs(const int* offsets, const int* logical, int slots, int* count,
                          Job* jobs) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= slots) return;
    const int begin = offsets[s], end = offsets[s + 1];
    if (begin == end) return;
    const int n    = (end - begin - 1) / Tile + 1;
    const int base = atomicAdd(count, n);
    for (int j = 0; j < n; ++j)
        jobs[base + j] = {logical[s], begin + j * Tile, min(end, begin + j * Tile + Tile)};
}

__global__ void clear_partial(float* out, const int* counts, int hidden) {
    const auto n = static_cast<std::int64_t>(counts[0]) * hidden;
    for (std::int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        out[i] = 0;
}

struct ContiguousRows {
    const __nv_bfloat16* hidden;
    float* partial;

    __device__ const __nv_bfloat16* input(int row, int h) const {
        return hidden + std::int64_t(row) * h;
    }

    __device__ float* output(int row, int h) const { return partial + std::int64_t(row) * h; }
};

struct PacketRows {
    ExpertPacketBatch batch;

    __device__ const __nv_bfloat16* input(int row, int h) const {
        return reinterpret_cast<const __nv_bfloat16*>(
                   batch.peers[row / batch.columns].receive.hidden) +
               std::int64_t(row % batch.columns) * h;
    }

    __device__ float* output(int row, int h) const {
        return batch.peers[row / batch.columns].partial + std::int64_t(row % batch.columns) * h;
    }
};

__global__ void clear_packet_partials(ExpertPacketBatch batch, int h) {
    const auto peer = batch.peers[blockIdx.y];
    const auto n    = std::int64_t(peer.receive.counts[0]) * h;
    for (std::int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        peer.partial[i] = 0;
}

struct Matrix {
    const unsigned char* codes;
    const unsigned char* scales;
    float factor;
};

template <int Block>
__device__ Matrix resolve(const ExpertMatrixDeviceView& b, int e) {
    const auto slot   = b.expert_indices ? b.expert_indices[e] : static_cast<unsigned>(e);
    const auto offset = b.offsets ? b.offsets[e] : 0;
    float factor      = 1;
    if constexpr (Block == 16) factor = 1.F / b.matrix_scales[slot * b.matrix_scale_stride];
    return {reinterpret_cast<const unsigned char*>(b.codes) +
                (b.offsets ? offset : slot * b.code_stride),
            reinterpret_cast<const unsigned char*>(b.scales) +
                (b.offsets ? offset : slot * b.scale_stride),
            factor};
}

template <int Block>
__device__ __forceinline__ float4 weights(Matrix w, int r, int c, int k) {
    float2 lo, hi;
    float scale;
    if constexpr (Block == 16) {
        const auto code =
            *reinterpret_cast<const unsigned short*>(w.codes + (std::int64_t(r) * k + c) / 2);
        lo = common::decode_nvfp4_e2m1x2(code);
        hi = common::decode_nvfp4_e2m1x2(code >> 8);
        scale =
            common::fp8_code2_to_float2(w.scales[common::nvfp4_scale_byte_offset(r, c / 16, k)]).x;
    } else {
        const auto code = *reinterpret_cast<const unsigned*>(w.codes + std::int64_t(r) * k + c);
        lo              = common::fp8_code2_to_float2(code);
        hi              = common::fp8_code2_to_float2(code >> 16);
        scale           = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(
            w.scales)[(r / Block) * (k / Block) + c / Block]);
    }
    return make_float4(lo.x * scale, lo.y * scale, hi.x * scale, hi.y * scale);
}

__device__ __forceinline__ float dot(float4 a, float4 b, float acc) {
    return fmaf(a.w, b.w, fmaf(a.z, b.z, fmaf(a.y, b.y, fmaf(a.x, b.x, acc))));
}

template <bool Gate, int Block, class RowStorage>
__global__ void contract(const int* job_count, const Job* jobs, const int* rows, const float* coeff,
                         ExpertMatrixDeviceView bank, RowStorage storage, float* product, int h,
                         int intermediate, float clamp) {
    const int r = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    const int k = Gate ? h : intermediate;
    for (int j = blockIdx.y; j < *job_count; j += gridDim.y) {
        const auto job = jobs[j];
        const auto w   = resolve<Block>(bank, job.expert);
        float sums[Tile]{}, ups[Tile]{};
        const __nv_bfloat16* inputs[Tile];
        if constexpr (Gate)
            for (int t = 0; t < job.end - job.begin; ++t)
                inputs[t] = storage.input(rows[job.begin + t], h);
        for (int c = lane * 4; c < k; c += 128) {
            const float4 a = weights<Block>(w, r, c, k);
            float4 b;
            if constexpr (Gate) b = weights<Block>(w, r + intermediate, c, k);
            for (int t = 0; t < job.end - job.begin; ++t) {
                const int m = job.begin + t;
                float4 x;
                if constexpr (Gate) {
                    const auto* p = inputs[t] + c;
                    const auto v  = *reinterpret_cast<const uint2*>(p);
                    const auto lo = common::bf16x2_bits_to_float2(v.x),
                               hi = common::bf16x2_bits_to_float2(v.y);
                    x             = make_float4(lo.x, lo.y, hi.x, hi.y);
                    ups[t]        = dot(b, x, ups[t]);
                } else {
                    x = *reinterpret_cast<const float4*>(product + std::int64_t(m) * k + c);
                }
                sums[t] = dot(a, x, sums[t]);
            }
        }
        for (int t = 0; t < job.end - job.begin; ++t) {
            const int m = job.begin + t;
            float value = common::warp_reduce_sum(sums[t]) * w.factor;
            if constexpr (Gate) {
                float up = common::warp_reduce_sum(ups[t]) * w.factor;
                if (clamp > 0) {
                    value = fminf(value, clamp);
                    up    = fmaxf(-clamp, fminf(clamp, up));
                }
                if (!lane) product[std::int64_t(m) * intermediate + r] = common::silu(value) * up;
            } else if (!lane) {
                atomicAdd(storage.output(rows[m], h) + r, value * coeff[m]);
            }
        }
    }
}
} // namespace

std::size_t compact_routed_swiglu_workspace_bytes(const RoutedSwiGLUProfile& p, int capacity) {
    validate(p, capacity);
    WorkspaceLayoutBuilder layout;
    allocate(layout, capacity, p.ffn.intermediate);
    return layout.peak_bytes();
}

namespace {
template <class RowStorage>
void execute_compact(RowStorage rows, int row_capacity, const RoutedSwiGLUProfile& p,
                     const CudaSwiGLUExperts& experts, const CompactExpertAssignments& a,
                     detail::MoeOperands& op, MemoryArena& workspace, CudaExecutionView execution) {
    const int slots = a.logical_ids.ne[0], capacity = a.rows.ne[0], h = p.ffn.hidden,
              i = p.ffn.intermediate;
    validate(p, capacity);
    if (slots <= 0 || slots > p.experts || slots == INT_MAX)
        throw std::invalid_argument("compact routed SwiGLU: invalid slots");
    op.vector_input(a.counts, DType::I32, 2, "compact counts");
    op.vector_input(a.offsets, DType::I32, slots + 1, "expert offsets");
    op.vector_input(a.logical_ids, DType::I32, slots, "logical expert IDs");
    op.vector_input(a.rows, DType::I32, capacity, "compact row IDs");
    op.vector_input(a.coefficients, DType::FP32, capacity, "compact coefficients");
    op.experts(experts, p);
    auto scope         = workspace.scope();
    const auto storage = workspace.alloc_bytes(compact_routed_swiglu_workspace_bytes(p, capacity));
    op.workspace(storage);
    op.finish();
    MemoryArena arena(storage);
    const auto s      = allocate(arena, capacity, i);
    const auto stream = execution.stream();
    CUDA_CHECK(cudaMemsetAsync(s.count, 0, sizeof(int), stream));
    make_jobs<<<(slots + 127) / 128, 128, 0, stream>>>(static_cast<const int*>(a.offsets.data),
                                                       static_cast<const int*>(a.logical_ids.data),
                                                       slots, s.count, s.jobs);
    const auto grid = std::min<std::int64_t>(execution.multiprocessor_count() * 4,
                                             (std::int64_t(row_capacity) * h + 255) / 256);
    if constexpr (std::is_same_v<RowStorage, PacketRows>)
        clear_packet_partials<<<dim3(grid, rows.batch.segments), 256, 0, stream>>>(rows.batch, h);
    else
        clear_partial<<<grid, 256, 0, stream>>>(rows.partial,
                                                static_cast<const int*>(a.counts.data), h);
    const auto run = [&]<int Block, bool Gate>() {
        const int n = Gate ? i : h;
        contract<Gate, Block><<<dim3(n / 4, std::min(capacity, 32)), 128, 0, stream>>>(
            s.count, s.jobs, static_cast<const int*>(a.rows.data),
            static_cast<const float*>(a.coefficients.data),
            Gate ? experts.bank.gate_up : experts.bank.down, rows, s.product, h, i,
            p.ffn.clamp_limit.value_or(0));
    };
    const auto both = [&]<int Block>() {
        run.template operator()<Block, true>();
        run.template operator()<Block, false>();
    };
    if (p.ffn.gate_up_type == QType::NVFP4)
        both.template operator()<16>();
    else if (p.ffn.gate_up_type == QType::FP8_E4M3FN_BLOCK64X64_BF16)
        both.template operator()<64>();
    else
        both.template operator()<128>();
    CUDA_CHECK(cudaGetLastError());
}
} // namespace

void compact_routed_swiglu(const Tensor& hidden, const RoutedSwiGLUProfile& p,
                           const CudaSwiGLUExperts& experts, const CompactExpertAssignments& a,
                           Tensor& partial, MemoryArena& arena, CudaExecutionView exec) {
    if (partial.dtype != DType::FP32)
        throw std::invalid_argument("compact routed SwiGLU: expected FP32 partial output");
    detail::MoeOperands op("compact routed SwiGLU", exec);
    const int u = op.input(hidden, p.ffn.hidden);
    op.output(partial, p.ffn.hidden, u, MoeEpilogue::Write);
    execute_compact(ContiguousRows{static_cast<const __nv_bfloat16*>(hidden.data),
                                   static_cast<float*>(partial.data)},
                    u, p, experts, a, op, arena, exec);
}

void compact_routed_swiglu(ExpertPacketBatch batch, const RoutedSwiGLUProfile& p,
                           const CudaSwiGLUExperts& experts, const CompactExpertAssignments& a,
                           MemoryArena& arena, CudaExecutionView exec) {
    if (!batch.peers || batch.device != exec.device() || batch.columns <= 0 ||
        batch.segments <= 0 || batch.segments > 65535 ||
        std::int64_t(batch.segments) * batch.columns > INT_MAX)
        throw std::invalid_argument("compact routed SwiGLU: invalid receive segments");
    detail::MoeOperands op("segmented compact routed SwiGLU", exec);
    execute_compact(PacketRows{batch}, batch.columns, p, experts, a, op, arena, exec);
}
} // namespace ninfer::ops
