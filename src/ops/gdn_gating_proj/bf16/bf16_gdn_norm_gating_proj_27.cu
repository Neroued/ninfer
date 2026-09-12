#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.h"
#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {
namespace {
// Each head computes both control dots and the complete norm. RMS scaling can be
// applied after the dots; h is independently rounded from the full normalized input.
template <int Tile, int Threads>
__global__ __launch_bounds__(Threads) void gdn_norm_gating_27_simt(
    const __nv_bfloat16* x, const __nv_bfloat16* nw, const __nv_bfloat16* aw,
    const __nv_bfloat16* bw, const float* alog, const float* bias, __nv_bfloat16* h, float* g,
    float* beta, int tokens, float eps) {
    constexpr int D = 5120, H = 48, Warps = Threads / 32;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, head = blockIdx.x,
              first = blockIdx.y * Tile;
    float aa[Tile]{}, bb[Tile]{}, ss[Tile]{};
    for (int base = tid * 8; base < D; base += Threads * 8) {
        const int4 av = load_vec<int4>(aw + head * D + base),
                   bv = load_vec<int4>(bw + head * D + base), nv = load_vec<int4>(nw + base);
        const auto* ap = reinterpret_cast<const unsigned*>(&av);
        const auto* bp = reinterpret_cast<const unsigned*>(&bv);
        const auto* np = reinterpret_cast<const unsigned*>(&nv);
#pragma unroll
        for (int t = 0; t < Tile; ++t) {
            if (first + t >= tokens) continue;
            const int4 xv  = load_vec<int4>(x + std::int64_t(first + t) * D + base);
            const auto* xp = reinterpret_cast<const unsigned*>(&xv);
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const float2 a = bf16x2_bits_to_float2(ap[p]), b = bf16x2_bits_to_float2(bp[p]);
                const float2 n = bf16x2_bits_to_float2(np[p]), v = bf16x2_bits_to_float2(xp[p]);
                const float z0 = v.x * (1.0f + n.x), z1 = v.y * (1.0f + n.y);
                aa[t] = fmaf(a.x, z0, aa[t]);
                aa[t] = fmaf(a.y, z1, aa[t]);
                bb[t] = fmaf(b.x, z0, bb[t]);
                bb[t] = fmaf(b.y, z1, bb[t]);
                ss[t] = fmaf(v.x, v.x, ss[t]);
                ss[t] = fmaf(v.y, v.y, ss[t]);
            }
        }
    }
    __shared__ float partial[3][Tile][Warps], inverse[Tile];
#pragma unroll
    for (int t = 0; t < Tile; ++t) {
        aa[t] = warp_reduce_sum(aa[t]);
        bb[t] = warp_reduce_sum(bb[t]);
        ss[t] = warp_reduce_sum(ss[t]);
        if (lane == 0) {
            partial[0][t][warp] = aa[t];
            partial[1][t][warp] = bb[t];
            partial[2][t][warp] = ss[t];
        }
    }
    __syncthreads();
    if (tid < Tile) {
        float a = 0, b = 0, s = 0;
#pragma unroll
        for (int w = 0; w < Warps; ++w) {
            a += partial[0][tid][w];
            b += partial[1][tid][w];
            s += partial[2][tid][w];
        }
        const float inv = rsqrtf(s / D + eps);
        inverse[tid]    = inv;
        if (first + tid < tokens) {
            const auto i = std::int64_t(first + tid) * H + head;
            g[i]         = -expf(alog[head]) * softplus(a * inv + bias[head]);
            beta[i]      = sigmoid(b * inv);
        }
    }
    __syncthreads();
    // Disjoint, pair-aligned h slices across the 48 heads avoid a separate norm kernel.
    constexpr int PairsPerHead = (D / 2 + H - 1) / H;
    const int pair             = head * PairsPerHead + tid;
    if (tid < PairsPerHead && pair < D / 2) {
        const float2 n = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(nw)[pair]);
#pragma unroll
        for (int t = 0; t < Tile; ++t)
            if (first + t < tokens) {
                const auto i   = std::int64_t(first + t) * (D / 2) + pair;
                const float2 v = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(x)[i]);
                reinterpret_cast<__nv_bfloat162*>(h)[i] = __floats2bfloat162_rn(
                    v.x * inverse[t] * (1 + n.x), v.y * inverse[t] * (1 + n.y));
            }
    }
}
} // namespace

void bf16_gdn_norm_gating_proj_27_launch(const Tensor& x, const Tensor& norm_weight, float eps,
                                         Tensor& h, const Weight& a_weight, const Weight& b_weight,
                                         const Tensor& alog, const Tensor& bias, Tensor& g,
                                         Tensor& beta, cudaStream_t stream) {
    const auto launch = [&]<int T, int Threads>() {
        gdn_norm_gating_27_simt<T, Threads>
            <<<dim3(48, (x.ne[1] + T - 1) / T), Threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const __nv_bfloat16*>(norm_weight.data),
                static_cast<const __nv_bfloat16*>(a_weight.qdata),
                static_cast<const __nv_bfloat16*>(b_weight.qdata),
                static_cast<const float*>(alog.data), static_cast<const float*>(bias.data),
                static_cast<__nv_bfloat16*>(h.data), static_cast<float*>(g.data),
                static_cast<float*>(beta.data), x.ne[1], eps);
    };
    const int tokens = x.ne[1];
    if (tokens <= 2)
        launch.template operator()<1, 512>();
    else if (tokens <= 14)
        launch.template operator()<2, 512>();
    else if (tokens <= 28)
        launch.template operator()<2, 256>();
    else
        launch.template operator()<4, 256>();
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail
