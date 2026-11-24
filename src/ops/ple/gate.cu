#include "ops/ple/operands.h"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/common/numeric/math.cuh"

namespace ninfer::ops::detail::ple {
namespace {

struct GateArguments {
    const __nv_bfloat16 *key, *value, *residual, *norm_key, *norm_query, *norm_conv;
    __nv_bfloat16 *gated, *records;
    const std::int32_t *valid, *offsets;
    std::int64_t key_token, key_batch, value_token, value_batch, record_token, record_batch;
    int tokens, batch;
    float epsilon;
};

template <int Block, int Width>
__global__ void gate_kernel(const __grid_constant__ GateArguments a) {
    constexpr int items = (kFeatures + Block * Width - 1) / (Block * Width);
    constexpr int warps = Block / common::kWarpSize;
    __shared__ float sums[3][warps];
    __shared__ float scalar;
    const int lane = threadIdx.x % common::kWarpSize, warp = threadIdx.x / common::kWarpSize;
    const std::int64_t work = std::int64_t(a.tokens) * kStreams;
    for (std::int64_t job = blockIdx.x; job < work; job += gridDim.x) {
        const int s = job % kStreams, t = job / kStreams, b = blockIdx.y;
        const auto column = std::int64_t(b) * a.tokens + t;
        const auto base   = column * kChannels + s * kFeatures;
        if (t >= a.valid[b]) {
#pragma unroll
            for (int i = 0; i < items; ++i) {
                const int d = (threadIdx.x + i * Block) * Width;
                if (d < kFeatures) common::Bf16Vector<Width>{}.store(a.gated + base + d);
            }
            continue;
        }
        const auto* key = a.key + b * a.key_batch + t * a.key_token + s * kFeatures;
        float kk = 0, qq = 0, dot = 0;
#pragma unroll
        for (int i = 0; i < items; ++i) {
            const int d = (threadIdx.x + i * Block) * Width;
            if (d < kFeatures) {
                const auto k  = common::Bf16Vector<Width>::load(key + d);
                const auto q  = common::Bf16Vector<Width>::load(a.residual + base + d);
                const auto wk = common::Bf16Vector<Width>::load(a.norm_key + s * kFeatures + d);
                const auto wq = common::Bf16Vector<Width>::load(a.norm_query + s * kFeatures + d);
#pragma unroll
                for (int v = 0; v < Width; ++v) {
                    const float x = __bfloat162float(k.word[v]), y = __bfloat162float(q.word[v]);
                    kk  = fmaf(x, x, kk);
                    qq  = fmaf(y, y, qq);
                    dot = fmaf(x * (1.f + __bfloat162float(wk.word[v])),
                               y * (1.f + __bfloat162float(wq.word[v])), dot);
                }
            }
        }
        kk  = common::warp_reduce_sum(kk);
        qq  = common::warp_reduce_sum(qq);
        dot = common::warp_reduce_sum(dot);
        if (lane == 0) {
            sums[0][warp] = kk;
            sums[1][warp] = qq;
            sums[2][warp] = dot;
        }
        __syncthreads();
        if (warp == 0) {
            kk  = common::warp_reduce_sum(lane < warps ? sums[0][lane] : 0.f);
            qq  = common::warp_reduce_sum(lane < warps ? sums[1][lane] : 0.f);
            dot = common::warp_reduce_sum(lane < warps ? sums[2][lane] : 0.f);
            if (lane == 0) {
                const float score = dot * rsqrtf(kk / kFeatures + a.epsilon) *
                                    rsqrtf(qq / kFeatures + a.epsilon) *
                                    (1.f / sqrtf(float(kFeatures)));
                const float sign = float(score > 0.f) - float(score < 0.f);
                scalar           = common::sigmoid(sign * sqrtf(fmaxf(fabsf(score), 1e-6f)));
            }
        }
        __syncthreads();
        const float gate  = scalar;
        const auto* value = a.value + b * a.value_batch + t * a.value_token;
        common::Bf16Vector<Width> values[items];
        float uu = 0;
#pragma unroll
        for (int i = 0; i < items; ++i) {
            const int d = (threadIdx.x + i * Block) * Width;
            if (d < kFeatures) {
                const auto v = common::Bf16Vector<Width>::load(value + d);
#pragma unroll
                for (int j = 0; j < Width; ++j) {
                    values[i].word[j] = __float2bfloat16_rn(gate * __bfloat162float(v.word[j]));
                    const float u     = __bfloat162float(values[i].word[j]);
                    uu                = fmaf(u, u, uu);
                }
                values[i].store(a.gated + base + d);
            }
        }
        // The reduction consumes the same BF16 words just published in U.
        if constexpr ((Block & (Block - 1)) == 0) {
            uu = common::block_reduce_sum<Block>(uu, sums[0]);
        } else {
            uu = common::warp_reduce_sum(uu);
            if (lane == 0) sums[0][warp] = uu;
            __syncthreads();
            if (warp == 0) uu = common::warp_reduce_sum(lane < warps ? sums[0][lane] : 0.f);
        }
        if (threadIdx.x == 0) scalar = rsqrtf(uu / kFeatures + a.epsilon);
        __syncthreads();
        const float inverse = scalar;
        auto* records       = a.records + b * a.record_batch +
                        (std::int64_t(a.offsets[b]) + t) * a.record_token + s * kFeatures;
#pragma unroll
        for (int i = 0; i < items; ++i) {
            const int d = (threadIdx.x + i * Block) * Width;
            if (d < kFeatures) {
                const auto weight =
                    common::Bf16Vector<Width>::load(a.norm_conv + s * kFeatures + d);
                common::Bf16Vector<Width> out;
#pragma unroll
                for (int v = 0; v < Width; ++v)
                    out.word[v] =
                        __float2bfloat16_rn(__bfloat162float(values[i].word[v]) * inverse *
                                            (1.f + __bfloat162float(weight.word[v])));
                out.store(records + d);
            }
        }
        __syncthreads();
    }
}

template <int Block, int Width>
void launch_gate(const GateArguments& args, cudaStream_t stream) {
    const auto groups = std::int64_t(args.tokens) * kStreams;
    const auto grid   = static_cast<unsigned>(groups < 65535 ? groups : 65535);
    gate_kernel<Block, Width><<<dim3(grid, args.batch), Block, 0, stream>>>(args);
    CUDA_CHECK(cudaGetLastError());
}

using GateLaunch = void (*)(const GateArguments&, cudaStream_t);

GateLaunch select_gate(const GateArguments& a, bool paired) {
    if (!paired) return launch_gate<256, 1>;
    bool wide = true;
    for (const auto* p : {a.key, a.value, a.norm_key, a.norm_query, a.norm_conv,
                          static_cast<const __nv_bfloat16*>(a.records)})
        wide &= reinterpret_cast<std::uintptr_t>(p) % 16 == 0;
    for (auto stride :
         {a.key_token, a.key_batch, a.value_token, a.value_batch, a.record_token, a.record_batch})
        wide &= stride % 8 == 0;
    // A stream fits exactly one 16-byte vector per thread in the aligned route.
    return wide ? launch_gate<320, 8> : launch_gate<256, 2>;
}

} // namespace
} // namespace ninfer::ops::detail::ple

namespace ninfer::ops {

void ple_gate(const PleProjectionView& projected, const Tensor& residual,
              const PleGateParameters& parameters, const PleBatchView& batch,
              PleConvRecordWindowView records, Tensor& gated_value, CudaExecutionView execution) {
    using namespace detail::ple;
    validate_gate(projected, residual, parameters, batch, records, gated_value, execution);
    const GateArguments args{static_cast<const __nv_bfloat16*>(projected.key.data),
                             static_cast<const __nv_bfloat16*>(projected.value.data),
                             static_cast<const __nv_bfloat16*>(residual.data),
                             static_cast<const __nv_bfloat16*>(parameters.norm_key.data),
                             static_cast<const __nv_bfloat16*>(parameters.norm_query.data),
                             static_cast<const __nv_bfloat16*>(parameters.norm_conv.data),
                             static_cast<__nv_bfloat16*>(gated_value.data),
                             static_cast<__nv_bfloat16*>(records.values.data),
                             static_cast<const std::int32_t*>(batch.valid_columns.data),
                             static_cast<const std::int32_t*>(records.offsets.data),
                             projected.key.nb[2] / 2,
                             projected.key.nb[3] / 2,
                             projected.value.nb[1] / 2,
                             projected.value.nb[2] / 2,
                             records.values.nb[1] / 2,
                             records.values.nb[2] / 2,
                             residual.ne[2],
                             residual.ne[3],
                             parameters.rms_epsilon};
    const bool paired = pair_aligned(projected.key) && pair_aligned(projected.value) &&
                        pair_aligned(parameters.norm_key) && pair_aligned(parameters.norm_query) &&
                        pair_aligned(parameters.norm_conv) && pair_aligned(records.values);
    select_gate(args, paired)(args, execution.stream());
}

} // namespace ninfer::ops
