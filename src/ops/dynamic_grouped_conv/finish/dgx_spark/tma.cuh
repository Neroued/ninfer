#pragma once

#include "ops/linear/kernels/bf16/bf16_a16_tma_mma.cuh"

namespace ninfer::ops::detail::dynamic_grouped_conv::finish {
inline constexpr int kOutputRows = 4096, kOutputGroups = kOutputRows / 16;

struct TmaOutput {
    const __nv_bfloat16 *base, *delta;
    __nv_bfloat16* result;
    int width;
};

struct TmaConvolution {
    static constexpr bool kTmaCollective = true;
    template <class S>
    static constexpr int kSharedBytes = S::kBlockRows * S::kBlockTokens * 4;

    template <class S, bool Full, class O>
    __device__ void finish_tile(const O& out, unsigned char* scratch,
                                float (&acc)[S::kMmaRows][S::kMmaTokens][4], int row0, int token0,
                                int, int tokens) const {
        constexpr int R = S::kBlockRows, T = S::kBlockTokens;
        auto* tile = reinterpret_cast<float*>(scratch);
        auto at    = [&](int row, int col) {
            return tile + col * R + (row ^ (((col / 2) & (R / 8 - 1)) * 8));
        };
        if (threadIdx.x >= S::kProducerThreads) {
            int tid = threadIdx.x - S::kProducerThreads, warp = tid / 32, lane = tid % 32;
            int wm = warp / S::kWarpsTokens, wn = warp % S::kWarpsTokens;
#pragma unroll
            for (int mi = 0; mi < S::kMmaRows; ++mi) {
                int row = wm * S::kWarpRows + mi * 16 + lane / 4;
#pragma unroll
                for (int ni = 0; ni < S::kMmaTokens; ++ni) {
                    int col               = wn * S::kWarpTokens + ni * 8 + 2 * (lane % 4);
                    *at(row, col)         = acc[mi][ni][0];
                    *at(row, col + 1)     = acc[mi][ni][1];
                    *at(row + 8, col)     = acc[mi][ni][2];
                    *at(row + 8, col + 1) = acc[mi][ni][3];
                }
            }
        }
        __syncthreads();
        const int columns = Full ? T : min(T, tokens - token0);
        for (int i = threadIdx.x; i < columns * R; i += S::kThreads) {
            int row = i % R, col = i / R, hidden = row0 + row, token = token0 + col;
            auto di     = std::int64_t(token) * 2 * kOutputGroups + hidden / 16;
            float value = (__bfloat162float(out.base[2 * kOutputRows + hidden]) +
                           __bfloat162float(out.delta[di])) *
                          *at(row, col);
            if (token % out.width)
                value = fmaf(__bfloat162float(out.base[3 * kOutputRows + hidden]) +
                                 __bfloat162float(out.delta[di + kOutputGroups]),
                             *at(row, col - 1), value);
            out.result[std::int64_t(token) * kOutputRows + hidden] = __float2bfloat16_rn(value);
        }
    }
};

template <class S>
__global__ __launch_bounds__(S::kThreads, S::kMinBlocksPerSm) void request_tile_kernel(
    const __grid_constant__ Bf16TmaDescriptors descriptors, TmaOutput output, int token_offset,
    int count, int step) {
    const int first = int(blockIdx.y) * step;
    bf16_a16_tma_mma_body<S, false, TmaOutput, TmaConvolution>(descriptors, output, {}, kOutputRows,
                                                               S::kStaticK, token_offset + first,
                                                               min(step, count - first));
}

template <int K, int Rows, int Tokens, int WarpRows, int WarpTokens>
void run_tma(const Tensor& input, const Weight& weight, const Tensor& base, const Tensor& delta,
             Tensor& result, cudaStream_t stream) {
    using S =
        Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<Rows, Tokens, 64, WarpRows, WarpTokens, 2>, K>;
    const Bf16A16Operands p{static_cast<const __nv_bfloat16*>(input.data),
                            static_cast<const __nv_bfloat16*>(weight.qdata), kOutputRows, K,
                            input.ne[1] * input.ne[2]};
    const TmaOutput output{static_cast<const __nv_bfloat16*>(base.data),
                           static_cast<const __nv_bfloat16*>(delta.data),
                           static_cast<__nv_bfloat16*>(result.data), input.ne[1]};
    if (Tokens % input.ne[1] == 0) {
        launch_bf16_a16_tma_mma<S, 1, TmaOutput, TmaConvolution>(p, output, {}, stream);
        return;
    }
    // Keep complete requests in a tile; the extra TMA columns are never published.
    const int step         = Tokens / input.ne[1] * input.ne[1];
    const auto descriptors = make_bf16_tma_descriptors<S>(p);
    constexpr auto entry   = request_tile_kernel<S>;
    constexpr int bytes    = bf16_tma_scratch_bytes<S, TmaConvolution> + S::kBarrierBytes;
    common::prepare_shared<bytes, entry, true>();
    common::for_each_token_slice(p.tokens, step, [&](int offset, int count) {
        entry<<<dim3(kOutputRows / Rows, (count + step - 1) / step), S::kThreads, bytes, stream>>>(
            descriptors, output, offset, count, step);
        CUDA_CHECK(cudaGetLastError());
    });
}
} // namespace ninfer::ops::detail::dynamic_grouped_conv::finish
