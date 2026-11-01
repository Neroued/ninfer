#pragma once
#include <cooperative_groups.h>
#include "ops/mhc/kernel/cluster_sync.cuh"
#include "ops/mhc/kernel/coefficients.cuh"
#include "ops/mhc/kernel/projection_math.cuh"
#include "ops/common/numeric/convert.cuh"

namespace ninfer::ops::detail::mhc {
template <int Threads, int BK>
struct ExpandClusterTile {
    static constexpr int KParts     = (Threads / 64 < BK / 16 ? Threads / 64 : BK / 16);
    static constexpr int StageBytes = 24 * BK * 4, ReductionBytes = (KParts * 8 * 32 + 2) * 4;
    static constexpr int kThreads = Threads;
    static constexpr int kSharedBytes =
        8 * 512 * 2 + (StageBytes > ReductionBytes ? StageBytes : ReductionBytes);
};

// Four identical residual streams give F*X = (F0+F1+F2+F3)*embedding.
// Fold native weights on chip into a high/low BF16 pair, accumulate both products
// in FP32, and retain the embedding until the expanded residual and branch output
// are published. Projection, coefficient reduction and output share one cluster allocation.
template <int Threads, int BK>
__global__ __launch_bounds__(Threads, 3) void __cluster_dims__(8, 1, 1)
    expand_cluster_kernel(Operands o, std::int64_t tile_offset) {
    constexpr int H = 512, Tokens = 8, KParts = ExpandClusterTile<Threads, BK>::KParts;
    static_assert(kHidden == 8 * H && H % BK == 0 && BK % (16 * KParts) == 0);
    static_assert(Threads % 32 == 0 && KParts >= 2);
    extern __shared__ __align__(16) __nv_bfloat16 expand_memory[];
    auto* x      = expand_memory;
    auto* a      = x + Tokens * H;
    auto* low    = a + 24 * BK;
    auto* data   = reinterpret_cast<float*>(a);
    auto cluster = cooperative_groups::this_cluster();
    int rank = cluster.block_rank(), tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    const std::int64_t m0 = (tile_offset + std::int64_t(blockIdx.x / 8)) * Tokens;
    for (int i = tid * 8; i < Tokens * H; i += Threads * 8) {
        int t = i / H, d = i % H;
        common::Bf16Vector<8> v{};
        if (m0 + t < o.columns)
            v = common::Bf16Vector<8>::load(static_cast<const __nv_bfloat16*>(o.residual) +
                                            (m0 + t) * kHidden + rank * H + d);
        v.store(x + t * H + projection_shared_column(t, d));
    }
    float accum[4] = {};
    for (int h = 0; h < H; h += BK) {
        for (int i = tid * 8; i < 24 * BK; i += Threads * 8) {
            int r = i / BK, c = i % BK;
            common::Bf16Vector<8> f[4];
#pragma unroll
            for (int s = 0; s < 4; ++s)
                f[s] = common::Bf16Vector<8>::load(static_cast<const __nv_bfloat16*>(o.projection) +
                                                   r * kWidth + s * kHidden + rank * H + h + c);
            common::Bf16Vector<8> hi, lo;
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                float sum = (__bfloat162float(f[0].word[v]) + __bfloat162float(f[1].word[v])) +
                            (__bfloat162float(f[2].word[v]) + __bfloat162float(f[3].word[v]));
                hi.word[v] = __float2bfloat16_rn(sum);
                lo.word[v] = __float2bfloat16_rn(sum - __bfloat162float(hi.word[v]));
            }
            hi.store(a + r * BK + projection_shared_column(r, c));
            lo.store(low + r * BK + projection_shared_column(r, c));
        }
        __syncthreads();
        if (warp < KParts * 2) {
#pragma unroll
            for (int sub = 0; sub < BK / 16 / KParts; ++sub) {
                int k  = warp / 2 * (BK / 16 / KParts) + sub;
                int ar = warp % 2 * 16 + (lane & 7) + (((lane >> 3) & 1) * 8),
                    ac = k * 16 + (lane >> 4) * 8;
                int bt = lane & 7, bc = h + k * 16 + (((lane >> 3) & 1) * 8), xt = ar - 24;
                unsigned af[4], bf[2];
                auto* ap = ar < 24 ? a + ar * BK + projection_shared_column(ar, ac)
                                   : x + xt * H + projection_shared_column(xt, h + ac);
                common::ldmatrix_x4(af[0], af[1], af[2], af[3], common::smem_addr(ap));
                common::ldmatrix_x2(
                    bf[0], bf[1], common::smem_addr(x + bt * H + projection_shared_column(bt, bc)));
                common::mma_bf16(accum[0], accum[1], accum[2], accum[3], af[0], af[1], af[2], af[3],
                                 bf[0], bf[1]);
                ap = ar < 24 ? low + ar * BK + projection_shared_column(ar, ac)
                             : x + xt * H + projection_shared_column(xt, h + ac);
                common::ldmatrix_x4(af[0], af[1], af[2], af[3], common::smem_addr(ap));
                if (warp % 2) af[1] = af[3] = 0;
                common::mma_bf16(accum[0], accum[1], accum[2], accum[3], af[0], af[1], af[2], af[3],
                                 bf[0], bf[1]);
            }
        }
        __syncthreads();
    }
    // Weight staging is retired. Reuse it for K partials and cluster publication.
    if (warp < KParts * 2) {
        int r = warp % 2 * 16 + (lane >> 2), t = 2 * (lane & 3);
        auto store = [&](int row, int token, float value) {
            if (row < 24 || row == 24 + token)
                data[((warp / 2) * Tokens + token) * 32 + (row < 24 ? row : 24)] =
                    row < 24 ? value : 4 * value;
        };
        store(r, t, accum[0]);
        store(r, t + 1, accum[1]);
        store(r + 8, t, accum[2]);
        store(r + 8, t + 1, accum[3]);
    }
    __syncthreads();
    for (int i = tid; i < Tokens * 32; i += Threads) {
        int token = i / 32, c = i % 32;
        if (c <= 24) {
            float value = 0;
#pragma unroll
            for (int k = 0; k < KParts; ++k) value += data[(k * Tokens + token) * 32 + c];
            data[i] = value;
        }
    }
    auto* ready = reinterpret_cast<unsigned*>(data + KParts * Tokens * 32);
    if (!tid) *ready = 0;
    cluster.sync();
    auto* pre  = data + Tokens * 32;
    auto* gain = pre + 4;
    // Rank r owns token r. Its branch gain follows from the embedding square sum,
    // so output workers can run alongside this owner's Sinkhorn iterations.
    if (!warp) {
        float value = 0;
        if (lane <= 24) {
#pragma unroll
            for (int c = 0; c < 8; ++c) value += cluster.map_shared_rank(data, c)[rank * 32 + lane];
        }
        float square = __shfl_sync(0xffffffffu, value, 24),
              logit  = activate_coefficients(o, value, m0 + rank, pre, m0 + rank < o.columns);
        __syncwarp();
        if (!lane) {
            float gate = (pre[0] + pre[1]) + (pre[2] + pre[3]);
            *gain      = gate * rsqrtf(gate * gate * square / kWidth + o.parameters.branch_epsilon);
            publish_cluster_value(ready);
        }
        if (m0 + rank < o.columns) sinkhorn(o, logit, m0 + rank);
    } else {
        for (int i = (tid - 32) * 8; i < Tokens * H; i += (Threads - 32) * 8) {
            int t = i / H, d = i % H;
            common::Bf16Vector<8> v, gamma;
            if (m0 + t < o.columns) {
                // Expansion is independent of the coefficients. Stream these large
                // residual writes while the owner finishes the branch gain.
                v     = common::Bf16Vector<8>::load(x + t * H + projection_shared_column(t, d));
                gamma = common::Bf16Vector<8>::load(static_cast<const __nv_bfloat16*>(o.gamma) +
                                                    rank * H + d);
                const auto bits = common::load_vec<uint4>(&v);
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    auto* target = static_cast<__nv_bfloat16*>(o.updated) + (m0 + t) * kWidth +
                                   s * kHidden + rank * H + d;
                    asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(target),
                                 "r"(bits.x), "r"(bits.y), "r"(bits.z), "r"(bits.w)
                                 : "memory");
                }
            }
            wait_cluster_value(cluster.map_shared_rank(ready, t));
            float factor = *cluster.map_shared_rank(gain, t);
            if (m0 + t < o.columns) {
                common::Bf16Vector<8> out;
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    out.word[j] = __float2bfloat16_rn(__bfloat162float(v.word[j]) *
                                                      __bfloat162float(gamma.word[j]) * factor);
                out.store(static_cast<__nv_bfloat16*>(o.output) + (m0 + t) * kHidden + rank * H +
                          d);
            }
        }
    }
    // Remote gain readers must finish before their owner's shared memory retires.
    cluster.sync();
}
} // namespace ninfer::ops::detail::mhc
