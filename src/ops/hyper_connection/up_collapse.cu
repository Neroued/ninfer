// Implements: ops/hyper_connection/launch.h (hc_up_collapse_*)
//
// The general route's Up projection fused with the collapse. Unfused, the Up Linear
// writes m = up . low as BF16 [S*H, T] and hyper_connection_collapse reads it back with the
// normalized residual: at prefill widths about 1 GB of traffic per mixer and chunk. Here the Up
// GEMM runs the same tile schedule as the unfused Linear at T > 128 (MmaR64T128), each CTA taking
// 16 features of every stream, and its epilogue does the collapse: the projection is rounded to
// BF16 as the Linear stores it, then mixed[h,t] = BF16((sum_s sigmoid(m[s*H+h,t]) * n[s*H+h,t]) *
// (1/S)) in stream order, the vector collapse kernel's arithmetic. The result is bit for bit that
// of the unfused route.
#include "ops/hyper_connection/launch.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/linear/common/output.cuh"
#include "ops/linear/q8/q8_instances.cuh"
#include "ops/linear/q8/q8_mma_launch.cuh"
#include "ops/linear/q8/q8_operands.h"

#include <cuda_bf16.h>

#include <atomic>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

using UpSchedule        = q8_instances::MmaR64T128;
constexpr int kFeatures = 16; // features per CTA, one 16-row block per stream
static_assert(UpSchedule::kBlockRows == kHcStreams * kFeatures);

std::atomic<int> g_force{-1};

// Local row s * 16 + i of a CTA whose output rows start at h0 is weight row s * hidden + h0 + i.
struct HcStreamRows {
    static constexpr int kOutputRowsPerCta = kFeatures;
    std::int32_t hidden;

    __device__ __forceinline__ int weight_row(int row0, int local_row, int) const {
        return (local_row / kFeatures) * hidden + row0 + local_row % kFeatures;
    }
};

struct HcCollapseEpilogue {
    const __nv_bfloat16* normalized; // [S*H, T]
    __nv_bfloat16* mixed;            // [H, T]
    std::int32_t hidden;

    // Scratch rows of BM + 4 floats: the fragment stores of a warp (8 rows x 4 column pairs) hit
    // distinct banks.
    template <class Schedule>
    static constexpr int kSharedBytes =
        (Schedule::kBlockRows + 4) * Schedule::kBlockTokens * sizeof(float);

    template <class Cfg, bool Full, class Output>
    __device__ __forceinline__ void finish_tile(Output, unsigned char* scratch,
                                                float (&acc)[Cfg::kMmaRows][Cfg::kMmaTokens][4],
                                                int m0, int n0, int, int n) const {
        constexpr int BM = Cfg::kBlockRows, BN = Cfg::kBlockTokens;
        constexpr int WM = Cfg::kWarpRows, WN = Cfg::kWarpTokens;
        constexpr int MT = Cfg::kMmaRows, NT = Cfg::kMmaTokens;
        const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
        const int wm = warp / Cfg::kWarpGridTokens, wn = warp % Cfg::kWarpGridTokens;
        const int gid = lane >> 2, lid = lane & 3;
        // The operand storage is free once every warp's MMAs are done.
        __syncthreads();
        constexpr int SP = BM + 4;
        float* up =
            reinterpret_cast<float*>(scratch); // [BN][SP], BF16-rounded as the Linear stores m
        const auto rounded = [](float v) { return __bfloat162float(__float2bfloat16_rn(v)); };
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
            const int r0 = wm * WM + mi * 16 + gid;
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int c0               = wn * WN + ni * 8 + 2 * lid;
                const float* a             = acc[mi][ni];
                up[c0 * SP + r0]           = rounded(a[0]);
                up[(c0 + 1) * SP + r0]     = rounded(a[1]);
                up[c0 * SP + r0 + 8]       = rounded(a[2]);
                up[(c0 + 1) * SP + r0 + 8] = rounded(a[3]);
            }
        }
        __syncthreads();
        const float inv_streams = 1.0F / static_cast<float>(kHcStreams);
        const std::int64_t wide = static_cast<std::int64_t>(kHcStreams) * hidden;
        // One thread per 8 consecutive features of one token: 16-byte loads of n and a 16-byte
        // store of mixed, the vector collapse kernel's arithmetic per element.
        for (int item = tid; item < BN * (kFeatures / 8); item += Cfg::kThreads) {
            const int c     = item / (kFeatures / 8);
            const int i     = (item % (kFeatures / 8)) * 8;
            const int token = n0 + c;
            if (!Full && token >= n) continue;
            const __nv_bfloat16* nrow = normalized + token * wide + m0 + i;
            float sum[8]              = {};
#pragma unroll
            for (int s = 0; s < kHcStreams; ++s) {
                const uint4 raw =
                    *reinterpret_cast<const uint4*>(nrow + static_cast<std::int64_t>(s) * hidden);
                const auto* pairs = reinterpret_cast<const __nv_bfloat162*>(&raw);
                const float* u    = up + c * SP + s * kFeatures + i;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 nv = __bfloat1622float2(pairs[j]);
                    sum[2 * j] += sigmoid(u[2 * j]) * nv.x;
                    sum[2 * j + 1] += sigmoid(u[2 * j + 1]) * nv.y;
                }
            }
            uint4 out;
            auto* packed = reinterpret_cast<__nv_bfloat162*>(&out);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                packed[j] =
                    __floats2bfloat162_rn(sum[2 * j] * inv_streams, sum[2 * j + 1] * inv_streams);
            }
            *reinterpret_cast<uint4*>(mixed + static_cast<std::int64_t>(token) * hidden + m0 + i) =
                out;
        }
    }
};

} // namespace

bool hc_up_collapse_supported(const Weight& up, std::int32_t hidden, std::int32_t tokens) {
    const int force = g_force.load(std::memory_order_relaxed);
    if (force == 0) return false;
    // T > 128: the unfused Up Linear takes MmaR64T128 there (q8 select_q8_generic), so the bits
    // agree.
    return up.qtype == QType::Q8_G32_FP16 && up.n == kHcStreams * hidden &&
           hidden % kFeatures == 0 && hidden % 8 == 0 && tokens > 128;
}

void hc_up_collapse_force_for_test(int mode) { g_force.store(mode, std::memory_order_relaxed); }

void hc_up_collapse_launch(const Tensor& lowrank, const Weight& up, const Tensor& normalized,
                           Tensor& mixed, cudaStream_t stream) {
    const std::int32_t hidden = mixed.ne[0];
    auto* out                 = static_cast<__nv_bfloat16*>(mixed.data);
    launch_q8_a16_mma<UpSchedule, HcStreamRows>(
        q8_linear_operands(lowrank, up), LinearBf16Output{out, hidden},
        HcCollapseEpilogue{static_cast<const __nv_bfloat16*>(normalized.data), out, hidden}, stream,
        HcStreamRows{hidden});
}

} // namespace ninfer::ops::detail
