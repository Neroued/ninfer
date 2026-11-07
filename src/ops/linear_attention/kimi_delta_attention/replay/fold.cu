#include "ops/linear_attention/kimi_delta_attention/replay/launch.h"
#include "ops/linear_attention/kimi_delta_attention/plan.h"
#include "core/platform.h"
#include "ops/linear_attention/kimi_delta_attention/kernels/transition.cuh"
#include "ops/common/host/operands.h"

namespace ninfer::ops::detail::kimi_delta_attention {
namespace {
__device__ KdaReplayFoldRow select_row(const KdaReplayFoldKernelRows& rows, int b) {
    return rows.row[b];
}

__device__ KdaReplayFoldRow select_row(const KdaReplayFoldDeviceRows& rows, int b) {
    const int count = rows.columns[b];
    if (!count) return {-1, -1, 0};
    return {rows.sources[b], rows.destinations[b], count};
}

struct FoldArguments {
    const __nv_bfloat16 *key, *value, *conv;
    const float *log_decay, *beta;
    float* recurrent;
    __nv_bfloat16* history;
    std::int64_t recurrent_stride, history_stride;
    int qk_heads, value_heads, conv_channels, width, capacity;
};

template <class Rows, int Warps, int Values>
__global__ __launch_bounds__(32 * Warps, 1) void fold_kernel(FoldArguments a, Rows rows) {
    constexpr int kBlockDv = Warps * Values;
    const auto row         = select_row(rows, blockIdx.y);
    if (!row.commit_columns) return;
    const int head = blockIdx.x / (kStateDim / kBlockDv);
    const int tile = blockIdx.x % (kStateDim / kBlockDv), layer = blockIdx.z;
    const int lane = threadIdx.x, warp = threadIdx.y, tid = warp * 32 + lane;
    const int dv = tile * kBlockDv + warp * Values, dk = lane * kQkPerLane;
    const auto base = static_cast<std::int64_t>(layer) * a.recurrent_stride +
                      static_cast<std::int64_t>(head) * kStateDim * kStateDim;
    const auto record_begin =
        (static_cast<std::int64_t>(layer) * a.capacity + blockIdx.y) * a.width;
    __align__(16) float state[Values][kQkPerLane];
    load_state(state, a.recurrent + base + row.source_offset / sizeof(float), dv, dk);
    for (int t = 0; t < row.commit_columns; ++t) {
        const auto column = record_begin + t;
        const auto qo     = (column * a.qk_heads + head / (a.value_heads / a.qk_heads)) * kStateDim;
        const auto vo     = (column * a.value_heads + head) * kStateDim;
        float key[kQkPerLane], alpha[kQkPerLane], decay[kQkPerLane];
        common::load_bf16x4(key, a.key + qo + dk);
        normalize_qk(key, lane);
        common::store_vec(decay, common::load_vec<float4>(a.log_decay + vo + dk));
#pragma unroll
        for (int c = 0; c < kQkPerLane; ++c) alpha[c] = common::exp_approx_ftz(decay[c]);
        const float value = lane < Values ? __bfloat162float(a.value[vo + dv + lane]) : 0.0F;
        apply_transition(state, key, alpha, value, a.beta[column * a.value_heads + head]);
    }
    store_state(state, a.recurrent + base + row.destination_offset / sizeof(float), dv, dk);
    // Each CTA owns one history stripe, including all of its old/new columns.
    const int channel = blockIdx.x * (32 * Warps) + tid;
    if (channel < a.conv_channels) {
        const auto history_base = static_cast<std::int64_t>(layer) * a.history_stride + channel;
        __nv_bfloat16 tail[3];
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            const int t = row.commit_columns - 3 + j;
            tail[j] =
                t < 0
                    ? a.history[row.source_offset / 2 + history_base + (t + 3LL) * a.conv_channels]
                    : a.conv[(record_begin + t) * a.conv_channels + channel];
        }
#pragma unroll
        for (int j = 0; j < 3; ++j)
            a.history[row.destination_offset / 2 + history_base +
                      static_cast<std::int64_t>(j) * a.conv_channels] = tail[j];
    }
}

template <class RecordRows, int Steps, int Warps, int Values>
__global__ __launch_bounds__(32 * Warps, 1) void staged_fold_kernel(FoldArguments a,
                                                                    RecordRows rows) {
    const auto row = select_row(rows, blockIdx.y);
    if (!row.commit_columns) return;
    constexpr int Tile = Warps * Values;
    const int head = blockIdx.x / (128 / Tile), tile = blockIdx.x % (128 / Tile),
              layer = blockIdx.z;
    const int lane = threadIdx.x, warp = threadIdx.y, tid = warp * 32 + lane, dk = lane * 4,
              dv     = tile * Tile + warp * Values;
    const auto base  = std::int64_t(layer) * a.recurrent_stride + std::int64_t(head) * 128 * 128;
    const auto start = (std::int64_t(layer) * a.capacity + blockIdx.y) * a.width;
    __align__(16) float state[Values][4];
    load_state(state, a.recurrent + base + row.source_offset / 4, dv, dk);

    __shared__ struct {
        float key[Steps][128], alpha[Steps][128], value[Steps][128], beta[Steps];
    } stage;

    for (int begin = 0; begin < row.commit_columns; begin += Steps) {
#pragma unroll
        for (int j = warp; j < Steps; j += Warps)
            if (begin + j < row.commit_columns) {
                auto col = start + begin + j;
                auto qo  = (col * a.qk_heads + head / (a.value_heads / a.qk_heads)) * 128,
                     vo  = (col * a.value_heads + head) * 128;
                float k[4], alpha[4], decay[4], v[4];
                common::load_bf16x4(k, a.key + qo + dk);
                normalize_qk(k, lane);
                common::store_vec(decay, common::load_vec<float4>(a.log_decay + vo + dk));
                common::load_bf16x4(v, a.value + vo + dk);
#pragma unroll
                for (int z = 0; z < 4; ++z) alpha[z] = common::exp_approx_ftz(decay[z]);
                common::store_vec(stage.key[j] + dk, common::load_vec<float4>(k));
                common::store_vec(stage.alpha[j] + dk, common::load_vec<float4>(alpha));
                common::store_vec(stage.value[j] + dk, common::load_vec<float4>(v));
                if (!lane) stage.beta[j] = a.beta[col * a.value_heads + head];
            }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < Steps; ++j)
            if (begin + j < row.commit_columns) {
                float k[4], alpha[4];
                common::store_vec(k, common::load_vec<float4>(stage.key[j] + dk));
                common::store_vec(alpha, common::load_vec<float4>(stage.alpha[j] + dk));
                float v = lane < Values ? stage.value[j][dv + lane] : 0;
                apply_transition(state, k, alpha, v, stage.beta[j]);
            }
        __syncthreads();
    }
    store_state(state, a.recurrent + base + row.destination_offset / 4, dv, dk);
    for (int c = blockIdx.x * (32 * Warps) + tid; c < a.conv_channels;
         c += gridDim.x * (32 * Warps)) {
        auto hb = std::int64_t(layer) * a.history_stride + c;
        __nv_bfloat16 tail[3];
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            int t   = row.commit_columns - 3 + j;
            tail[j] = t < 0 ? a.history[row.source_offset / 2 + hb + (t + 3LL) * a.conv_channels]
                            : a.conv[(start + t) * a.conv_channels + c];
        }
#pragma unroll
        for (int j = 0; j < 3; ++j)
            a.history[row.destination_offset / 2 + hb + std::int64_t(j) * a.conv_channels] =
                tail[j];
    }
}

template <class Rows>
void launch(const KdaReplayRecords& r, LinearAttentionStateAllLayersView s, const Rows& rows,
            int batch, int multiprocessors, cudaStream_t stream) {
    const FoldArguments a{static_cast<const __nv_bfloat16*>(r.key.data),
                          static_cast<const __nv_bfloat16*>(r.value.data),
                          static_cast<const __nv_bfloat16*>(r.conv.data),
                          static_cast<const float*>(r.log_decay.data),
                          static_cast<const float*>(r.beta.data),
                          common::operand::state_field_base<float>(s.recurrent),
                          common::operand::state_field_base<__nv_bfloat16>(s.conv),
                          static_cast<std::int64_t>(s.recurrent.field.layer_stride / sizeof(float)),
                          static_cast<std::int64_t>(s.conv.field.layer_stride / 2),
                          r.spec.qk_heads,
                          r.spec.value_heads,
                          r.spec.conv_channels,
                          r.spec.width,
                          r.spec.record_capacity};
    const auto submit = [&]<int Warps, int Values>() {
        fold_kernel<Rows, Warps, Values>
            <<<dim3(r.spec.value_heads * (kStateDim / (Warps * Values)), batch, r.spec.layers),
               dim3(32, Warps), 0, stream>>>(a, rows);
    };
    const auto schedule = select_fold_schedule(r.spec.layers, r.spec.value_heads, r.spec.width,
                                               batch, multiprocessors);
#if NINFER_TARGET_SPARK
    const auto staged = [&]<int Steps, int Values>() {
        staged_fold_kernel<Rows, Steps, 4, Values>
            <<<dim3(r.spec.value_heads * (kStateDim / (4 * Values)), batch, r.spec.layers),
               dim3(32, 4), 0, stream>>>(a, rows);
    };
    if (schedule.steps == 8)
        staged.template operator()<8, 8>();
    else if (schedule.steps == 4 && schedule.values == 16)
        staged.template operator()<4, 16>();
    else if (schedule.steps == 4 && schedule.values == 8)
        staged.template operator()<4, 8>();
    else if (schedule.steps == 4)
        staged.template operator()<4, 4>();
    else if (schedule.warps == 4 && schedule.values == 2)
        submit.template operator()<4, 2>();
    else if (schedule.values == 8)
        submit.template operator()<4, 8>();
    else if (schedule.values == 2)
        submit.template operator()<16, 2>();
    else
        submit.template operator()<4, 4>();
#else
    if (schedule.warps == 16)
        submit.template operator()<16, 4>();
    else if (schedule.warps == 8)
        submit.template operator()<8, 4>();
    else
        submit.template operator()<4, 4>();
#endif
    CUDA_CHECK(cudaGetLastError());
}
} // namespace

void launch_replay_fold(const KdaReplayRecords& r, LinearAttentionStateAllLayersView s,
                        const KdaReplayFoldKernelRows& rows, int batch, int multiprocessors,
                        cudaStream_t stream) {
    launch(r, s, rows, batch, multiprocessors, stream);
}

void launch_replay_fold(const KdaReplayRecords& r, LinearAttentionStateAllLayersView s,
                        const KdaReplayFoldDeviceRows& rows, int batch, int multiprocessors,
                        cudaStream_t stream) {
    launch(r, s, rows, batch, multiprocessors, stream);
}
} // namespace ninfer::ops::detail::kimi_delta_attention
