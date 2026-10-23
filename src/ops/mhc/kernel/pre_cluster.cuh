#pragma once
#include <cooperative_groups.h>
#include "ops/mhc/kernel/cluster_sync.cuh"
#include "ops/mhc/kernel/coefficients.cuh"
#include "ops/mhc/kernel/residual.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/mhc/kernel/projection_math.cuh"

namespace ninfer::ops::detail::mhc {
template <int Tokens, int Cluster, int BK, int Stages>
struct ClusterPreTile {
    static constexpr int kThreads = Tokens == 8 ? 512 : 256;
    static constexpr int kSharedBytes =
        (4 * Tokens * (kHidden / Cluster) + Stages * kRows * BK) * 2;
};

// A cluster owns complete token columns and partitions their hidden dimension.
// Each CTA retains its rounded residual words until collapse, so the complete
// pre/post-pre needs neither a residual reread nor global projection/norm scratch.
template <Form F, int Tokens, int Cluster, int BK, int Stages>
__global__ __launch_bounds__(Tokens == 8 ? 512 : 256, 2) void __cluster_dims__(Cluster, 1, 1)
    pre_cluster_kernel(Operands o, std::int64_t tile_offset) {
    constexpr int H       = kHidden / Cluster;
    constexpr int Threads = ClusterPreTile<Tokens, Cluster, BK, Stages>::kThreads;
    constexpr int Warps   = Threads / 32;
    constexpr int KParts  = Warps / 2 < BK / 16 ? Warps / 2 : BK / 16;
    constexpr int Workers = Threads - Tokens * 32, Wpt = Workers / 32 / Tokens, V = 4;
    constexpr int Groups = (H + Wpt * 32 * V - 1) / (Wpt * 32 * V);
    static_assert((Tokens == 4 || Tokens == 8) && Tokens == Cluster);
    static_assert(H % BK == 0 && Stages > 0 && Stages <= 4 * H / BK);
    static_assert(F == Form::Pre || F == Form::PostPre);
    static_assert(Workers == Tokens * 32 && Wpt == 1);
    static_assert(Stages * kRows * BK * 2 >= (KParts * Tokens * 32 + Tokens + 1) * sizeof(float));
    extern __shared__ __align__(16) __nv_bfloat16 cluster_memory[];
    auto* x        = cluster_memory;
    auto* a        = reinterpret_cast<__nv_bfloat16(*)[24 * BK]>(cluster_memory + 4 * Tokens * H);
    auto* data     = reinterpret_cast<float*>(a);
    auto cluster   = cooperative_groups::this_cluster();
    const int rank = cluster.block_rank(), tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    const auto first_tile = tile_offset + blockIdx.x / Cluster;
    auto tile             = first_tile;
    do {
        const std::int64_t m0 = tile * Tokens;

        if constexpr (F == Form::PostPre) {
            for (int i = tid; i < Tokens * 20; i += Threads) {
                auto m  = m0 + i / 20;
                int c   = i % 20;
                data[i] = m < o.columns ? (c < 4 ? o.post[m * 4 + c] : o.mix[m * 16 + c - 4]) : 0.f;
            }
            __syncthreads();
        }
        if constexpr (F == Form::Pre) {
            if (tile == first_tile) {
                // Neighboring requests cover all streams of one token. The retained
                // shared copy makes the source disposable while F remains reusable in L2.
                const auto streaming = common::l2_streaming_policy();
                for (int i = tid; i < 4 * Tokens * (H / 8); i += Threads) {
                    const int t = i / (4 * (H / 8)), s = i / (H / 8) % 4, d = i % (H / 8) * 8;
                    const auto m = m0 + t;
                    common::cp_async_cache_hint<Cache::cg>(
                        x + (s * Tokens + t) * H + projection_shared_column(t, d),
                        static_cast<const __nv_bfloat16*>(o.residual) +
                            (m < o.columns ? m : 0) * kWidth + s * kHidden + rank * H + d,
                        m < o.columns ? 16 : 0, streaming);
                }
                common::cp_commit();
                common::cp_wait<0>();
                __syncthreads();
            }
        } else {
            for (int i = tid * 8; i < Tokens * H; i += Threads * 8) {
                int t = i / H, d = i % H;
                common::Bf16Vector<8> values[4] = {};
                if (m0 + t < o.columns) {
                    mixed_vectors(o, m0 + t, rank * H + d, data + t * 20, data + t * 20 + 4,
                                  values);
#pragma unroll
                    for (int s = 0; s < 4; ++s)
                        values[s].store(static_cast<__nv_bfloat16*>(o.updated) + (m0 + t) * kWidth +
                                        s * kHidden + rank * H + d);
                }
#pragma unroll
                for (int s = 0; s < 4; ++s)
                    values[s].store(x + (s * Tokens + t) * H + projection_shared_column(t, d));
            }
            __syncthreads();
        }
        // Pipeline native F tiles. K partitions distribute the MMA work across warps;
        // the otherwise padded rows compute X*X and supply the coefficient square sum.
        constexpr int Steps = H / BK, Count = 4 * Steps;
        auto load = [&](int step, int slot) {
            int stream = step / Steps, h = (step % Steps) * BK;
            for (int i = tid; i < 24 * (BK / 8); i += Threads) {
                int r = i / (BK / 8), c = i % (BK / 8) * 8;
                auto* destination  = a[slot] + r * BK + projection_shared_column(r, c);
                const auto* source = static_cast<const __nv_bfloat16*>(o.projection) + r * kWidth +
                                     stream * kHidden + rank * H + h + c;
                if constexpr (F == Form::Pre)
                    common::cp_async_cache_hint<Cache::ca>(destination, source, 16,
                                                           common::l2_retained_policy());
                else
                    common::cp_async_zfill<16, Cache::ca>(destination, source, 16);
            }
            common::cp_commit();
        };
        float accum[4] = {};
#pragma unroll
        for (int s = 0; s < Stages; ++s) load(s, s);
        for (int step = 0; step < Count; ++step) {
            int slot = step % Stages, s = step / Steps, h = step % Steps * BK;
            if (step + Stages < Count)
                common::cp_wait<Stages - 1>();
            else
                common::cp_wait<0>();
            __syncthreads();
            if (warp < KParts * 2) {
#pragma unroll
                for (int sub = 0; sub < BK / 16 / KParts; ++sub) {
                    int k  = (warp / 2) * (BK / 16 / KParts) + sub;
                    int ar = (warp % 2) * 16 + (lane & 7) + (((lane >> 3) & 1) * 8),
                        ac = k * 16 + (lane >> 4) * 8;
                    int bt = (lane & 7) % Tokens, bc = h + k * 16 + (((lane >> 3) & 1) * 8),
                        xt = (ar - 24) % Tokens;
                    const auto* ap =
                        ar < 24 ? a[slot] + ar * BK + projection_shared_column(ar, ac)
                                : x + (s * Tokens + xt) * H + projection_shared_column(xt, h + ac);
                    const auto* bp = x + (s * Tokens + bt) * H + projection_shared_column(bt, bc);
                    unsigned af[4], bf[2];
                    common::ldmatrix_x4(af[0], af[1], af[2], af[3], common::smem_addr(ap));
                    common::ldmatrix_x2(bf[0], bf[1], common::smem_addr(bp));
                    common::mma_bf16(accum[0], accum[1], accum[2], accum[3], af[0], af[1], af[2],
                                     af[3], bf[0], bf[1]);
                }
            }
            __syncthreads();
            if (step + Stages < Count) load(step + Stages, slot);
        }
        if (warp < KParts * 2) {
            int r = (warp % 2) * 16 + (lane >> 2), t = 2 * (lane & 3);
            auto store = [&](int row, int token, float value) {
                if (token < Tokens && (row < 24 || row == 24 + token))
                    data[((warp / 2) * Tokens + token) * 32 + (row < 24 ? row : 24)] = value;
            };
            store(r, t, accum[0]);
            store(r, t + 1, accum[1]);
            store(r + 8, t, accum[2]);
            store(r + 8, t + 1, accum[3]);
        }
        __syncthreads();
        // Reuse the retired weight stages for local K reduction, then publish only
        // the first record per token to the other CTAs in this cluster.
        if (warp < Tokens && lane <= 24) {
            float value = 0;
#pragma unroll
            for (int k = 0; k < KParts; ++k) value += data[(k * Tokens + warp) * 32 + lane];
            data[warp * 32 + lane] = value;
        }
        auto* ready      = reinterpret_cast<unsigned*>(data + KParts * Tokens * 32);
        auto* gate_ready = ready + Tokens;
        if (tid < Tokens) ready[tid] = 0;
        if (tid == 0) *gate_ready = 0;
        cluster.sync();
        auto* pre  = data + Tokens * 32;
        auto* norm = pre + Tokens * 4;

        // Rank r owns token r: reduce its projection once and publish its four gates.
        float logit = 0;
        if (warp == 0) {
            float value = 0;
            if (lane <= 24) {
#pragma unroll
                for (int c = 0; c < Cluster; ++c)
                    value += cluster.map_shared_rank(data, c)[rank * 32 + lane];
            }
            logit = activate_coefficients(o, value, m0 + rank, pre, m0 + rank < o.columns);
            __syncwarp();
            if (lane == 0) publish_cluster_value(gate_ready);
        }
        // The owner finishes Sinkhorn while workers acquire the gates, collapse the
        // retained residual and reduce the branch RMS without waiting for Sinkhorn.
        float values[Groups][V], square = 0;
        const int token = (warp - Tokens) / Wpt, local = (warp - Tokens) % Wpt;
        if (warp < Tokens) {
            if (warp == 0 && m0 + rank < o.columns) sinkhorn(o, logit, m0 + rank);
        } else {
            wait_cluster_value(cluster.map_shared_rank(gate_ready, token));
            auto* remote_pre = cluster.map_shared_rank(pre, token);
            float gates[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) gates[i] = remote_pre[i];
#pragma unroll
            for (int g = 0; g < Groups; ++g) {
                int d = (g * Wpt * 32 + local * 32 + lane) * V;
                common::Bf16Vector<V> gamma;
                if constexpr (F == Form::Pre)
                    gamma = common::Bf16Vector<V>::load(static_cast<const __nv_bfloat16*>(o.gamma) +
                                                        rank * H + d);
#pragma unroll
                for (int v = 0; v < V; ++v) {
                    float value = 0;
                    if (d + v < H) {
#pragma unroll
                        for (int s = 0; s < 4; ++s)
                            value =
                                fmaf(gates[s],
                                     __bfloat162float(x[(s * Tokens + token) * H +
                                                        projection_shared_column(token, d + v)]),
                                     value);
                        square = fmaf(value, value, square);
                    }
                    if constexpr (F == Form::Pre)
                        values[g][v] = value * __bfloat162float(gamma.word[v]);
                    else
                        values[g][v] = value;
                }
            }
            square = common::warp_reduce_sum(square);
            // One warp owns this token's complete local collapse. Publish it without
            // waiting for the other tokens; each consumer needs only its own norm.
            if (lane == 0) {
                norm[token] = square;
                publish_cluster_value(ready + token);
            }
            if constexpr (F == Form::Pre) {
                const auto next_m0 = (tile + gridDim.x / Cluster) * Tokens;
                if (next_m0 < o.columns) {
                    // This warp has finished every read of its token's old residual.
                    // Fill the same shared slice while the current norm/output retires.
                    // The next iteration's first F wait also waits for this copy group.
                    const auto streaming = common::l2_streaming_policy();
#pragma unroll
                    for (int s = 0; s < 4; ++s) {
#pragma unroll
                        for (int g = 0; g < H / (32 * 8); ++g) {
                            const int d  = (g * 32 + lane) * 8;
                            const auto m = next_m0 + token;
                            common::cp_async_cache_hint<Cache::cg>(
                                x + (s * Tokens + token) * H + projection_shared_column(token, d),
                                static_cast<const __nv_bfloat16*>(o.residual) +
                                    (m < o.columns ? m : 0) * kWidth + s * kHidden + rank * H + d,
                                m < o.columns ? 16 : 0, streaming);
                        }
                    }
                    common::cp_commit();
                }
            }
            if (lane < Cluster) wait_cluster_value(cluster.map_shared_rank(ready, lane) + token);
            __syncwarp();
            float ss      = lane < Cluster ? cluster.map_shared_rank(norm, lane)[token] : 0.f;
            ss            = common::warp_reduce_sum(ss);
            float inverse = 0;
            if (lane == 0) inverse = rsqrtf(ss / kHidden + o.parameters.branch_epsilon);
            inverse = __shfl_sync(0xffffffffu, inverse, 0);
            if (m0 + token < o.columns) {
#pragma unroll
                for (int g = 0; g < Groups; ++g) {
                    int d = (g * Wpt * 32 + local * 32 + lane) * V;
                    if (d < H) {
                        common::Bf16Vector<V> gamma;
                        if constexpr (F != Form::Pre)
                            gamma = common::Bf16Vector<V>::load(
                                static_cast<const __nv_bfloat16*>(o.gamma) + rank * H + d);
                        common::Bf16Vector<V> output;
#pragma unroll
                        for (int v = 0; v < V; ++v) {
                            float value = values[g][v];
                            if constexpr (F != Form::Pre) value *= __bfloat162float(gamma.word[v]);
                            output.word[v] = __float2bfloat16_rn(value * inverse);
                        }
                        output.store(static_cast<__nv_bfloat16*>(o.output) +
                                     (m0 + token) * kHidden + rank * H + d);
                    }
                }
            }
        }
        // Every remote gate/norm read must complete before any CTA retires shared memory.
        cluster.sync();
        if constexpr (F != Form::Pre) break;
        tile += gridDim.x / Cluster;
    } while (tile * Tokens < o.columns);
}
} // namespace ninfer::ops::detail::mhc
