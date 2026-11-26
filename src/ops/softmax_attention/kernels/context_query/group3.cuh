#pragma once

#include "ops/softmax_attention/kernels/context_query/kernel.cuh"

namespace ninfer::ops::detail {

struct Group3Arguments {
    const __nv_bfloat16* q;
    const __nv_bfloat16* query_k;
    const __nv_bfloat16* query_v;
    const std::int32_t* lengths;
    const std::int32_t* valid;
    const std::int32_t* table_rows;
    const std::int64_t* key_offsets;
    const std::int64_t* value_offsets;
    const __nv_bfloat16* context_k;
    const __half* context_v;
    int record_extent;
    int window_mask;
    int max_context;
    int splits;
    int tokens;
    float scale;
    // Split slots are contiguous for each output row. Each slot holds one D-element mean.
    __half* partial;
    float* maxima;
    float* normalizers;
    __nv_bfloat16* output;
};

inline constexpr int kGroup3Heads     = 24;
inline constexpr int kGroup3KVHeads   = 8;
inline constexpr int kGroup3D         = kContextQueryHeadDim;
inline constexpr float kGroup3Log2E   = 1.4426950408889634074f;
inline constexpr unsigned kGroup3Mask = 0xffffffffu;

__device__ __forceinline__ float group3_weight(float m, float maximum) {
    return m == -CUDART_INF_F ? 0.F : common::exp2_approx(m - maximum);
}

template <bool Paged>
__device__ __forceinline__ int group3_length(const Group3Arguments& a, int batch) {
    return a.lengths[std::int64_t(batch) * (Paged ? 1 : a.tokens)];
}

template <bool Paged, int BlockKeys>
__device__ __forceinline__ int group3_work_tiles(const Group3Arguments& a, int length) {
    const int history = Paged ? length : min(length, a.window_mask);
    return (history + BlockKeys - 1) / BlockKeys + 1;
}

// Partition the history and query tiles together so the query tile can occupy its own split.
template <bool Paged, int Tokens, int MWarps, int KeyBlock, int WarpsK, bool StageQ, bool Aligned,
          bool Direct>
struct Group3Pipeline {
    static constexpr int D            = kGroup3D;
    static constexpr int Stages       = 2;
    static constexpr int Warps        = MWarps * WarpsK;
    static constexpr int Threads      = Warps * 32;
    static constexpr int BlockKeys    = KeyBlock * WarpsK;
    static constexpr int TileElements = 2 * BlockKeys * D;
    static constexpr int QElements    = Tokens * 3 * D;
    static constexpr int StagingBytes = (Stages * TileElements + (StageQ ? QElements : 0)) * 2;
    static constexpr int MergeBytes   = WarpsK == 2 ? Tokens * 3 * (D + 2) * sizeof(float) : 0;
    static constexpr int ScratchBytes = StagingBytes > MergeBytes ? StagingBytes : MergeBytes;
    static_assert(WarpsK == 1 || WarpsK == 2);
    static_assert(KeyBlock == 16 || KeyBlock == 32);
    static_assert(ScratchBytes <= 48 * 1024);
    static_assert((!StageQ && WarpsK == 1) || Tokens > 0);
    static_assert(!StageQ || Aligned);

    __device__ __forceinline__ static void run(const Group3Arguments& a, int batch, int kv_head,
                                               int split) {
        const int width      = Tokens ? Tokens : a.tokens;
        const int rows       = width * 3;
        const int batch_rows = width * kGroup3Heads;
        const int valid      = a.valid[batch];
        const int length     = valid > 0 ? group3_length<Paged>(a, batch) : 0;
        if (valid <= 0 || valid > width || length < 0 || length > a.max_context) {
            if constexpr (Direct) {
                auto* out = a.output + std::int64_t(batch_rows) * D * batch;
                for (int unit = threadIdx.x; unit < rows * D / 8; unit += Threads) {
                    const int r = unit / (D / 8);
                    const int d = (unit % (D / 8)) * 8;
                    auto* dst   = out + ((r / 3) * kGroup3Heads + kv_head * 3 + r % 3) * D + d;
                    if constexpr (Aligned) {
                        common::store_vec(dst, make_uint4(0, 0, 0, 0));
                    } else {
#pragma unroll
                        for (int j = 0; j < 8; ++j) dst[j] = __float2bfloat16(0.F);
                    }
                }
            }
            return;
        }

        const int history       = Paged ? length : min(length, a.window_mask);
        const int context_start = length - history;
        const int total_tiles   = group3_work_tiles<Paged, BlockKeys>(a, length);
        const int active        = min(total_tiles, a.splits);
        if (split >= active) return;
        const int begin = total_tiles * split / active;
        const int end   = total_tiles * (split + 1) / active;

        const int tid       = threadIdx.x;
        const int lane      = tid & 31;
        const int warp      = tid / 32;
        const int warp_m    = warp % MWarps;
        const int warp_k    = warp / MWarps;
        const int gid       = lane >> 2;
        const int lid       = lane & 3;
        const int row0      = warp_m * 16 + gid;
        const int row1      = row0 + 8;
        const int live_rows = valid * 3;
        const int pos0 =
            !Paged && row0 < live_rows ? a.lengths[std::int64_t(batch) * width + row0 / 3] : 0;
        const int pos1 =
            !Paged && row1 < live_rows ? a.lengths[std::int64_t(batch) * width + row1 / 3] : 0;
        const auto* q             = a.q + std::int64_t(batch_rows) * D * batch;
        const auto* qk            = a.query_k + std::int64_t(width * kGroup3KVHeads) * D * batch;
        const auto* qv            = a.query_v + std::int64_t(width * kGroup3KVHeads) * D * batch;
        const auto* ck            = a.context_k;
        const auto* cv            = a.context_v;
        const std::int64_t* table = nullptr;
        if (begin < total_tiles - 1) {
            if constexpr (Paged) {
                table = a.key_offsets + std::int64_t(a.record_extent) * a.table_rows[batch];
            } else {
                ck += a.key_offsets[batch] / 2 + std::int64_t(kv_head) * D * a.record_extent;
                cv += a.value_offsets[batch] / 2 + std::int64_t(kv_head) * D * a.record_extent;
            }
        }

        extern __shared__ __align__(16) __nv_bfloat16 scratch3[];
        const auto stage = [&](int tile) {
            auto* k         = scratch3 + (tile % Stages) * TileElements;
            auto* v         = reinterpret_cast<__half*>(k + BlockKeys * D);
            const int start = context_start + tile * BlockKeys;
            const int count = min(BlockKeys, length - start);
            if (tile == total_tiles - 1) {
#pragma unroll
                for (int chunk = tid; chunk < BlockKeys * (D / 8); chunk += Threads) {
                    const int r      = chunk / (D / 8);
                    const int d      = (chunk % (D / 8)) * 8;
                    const int dest   = r * D + common::swizzle_xor_64(r, d);
                    const bool live  = r < valid;
                    const int source = ((live ? r : 0) * kGroup3KVHeads + kv_head) * D + d;
                    if constexpr (Aligned) {
                        common::cp_async_zfill<16, Cache::cg>(k + dest, qk + source, live ? 16 : 0);
                        const int4 value = live ? common::bf16x8_bits_to_f16x8_bits(
                                                      common::load_vec<int4>(qv + source))
                                                : make_int4(0, 0, 0, 0);
                        common::store_vec(v + dest, value);
                    } else {
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            k[dest + j] = live ? qk[source + j] : __float2bfloat16(0.F);
                            v[dest + j] = live ? __half(qv[source + j]) : __float2half(0.F);
                        }
                    }
                }
            } else {
                std::int64_t page_base = 0;
                if constexpr (Paged) {
                    page_base = (lane == 0 ? table[start >> 6] : 0);
                    page_base = __shfl_sync(kGroup3Mask, page_base, 0) / 2 +
                                (kv_head * 64 + (start & 63)) * D;
                }
#pragma unroll
                for (int chunk = tid; chunk < BlockKeys * (D / 8); chunk += Threads) {
                    const int r            = chunk / (D / 8);
                    const int d            = (chunk % (D / 8)) * 8;
                    const int dest         = r * D + common::swizzle_xor_64(r, d);
                    const int key          = start + (r < count ? r : 0);
                    const std::int64_t src = Paged ? page_base + (r < count ? r : 0) * D + d
                                                   : (key & a.window_mask) * D + d;
                    const int bytes        = r < count ? 16 : 0;
                    common::cp_async_zfill<16, Cache::cg>(k + dest, ck + src, bytes);
                    common::cp_async_zfill<16, Cache::cg>(v + dest, cv + src, bytes);
                }
            }
            common::cp_commit();
        };

        auto* sq = scratch3 + Stages * TileElements;
        if constexpr (StageQ) {
#pragma unroll
            for (int chunk = tid; chunk < QElements / 8; chunk += Threads) {
                const int r    = chunk / (D / 8);
                const int d    = (chunk % (D / 8)) * 8;
                const int safe = r < live_rows ? r : 0;
                const int src  = ((safe / 3) * kGroup3Heads + kv_head * 3 + safe % 3) * D + d;
                common::cp_async_zfill<16, Cache::ca>(sq + r * D + common::swizzle_xor_64(r, d),
                                                      q + src, r < live_rows ? 16 : 0);
            }
        }
        stage(begin);

        unsigned qf[8][4];
        const auto q_row = [&](int row) {
            const int safe = row < live_rows ? row : 0;
            return q + ((safe / 3) * kGroup3Heads + kv_head * 3 + safe % 3) * D;
        };
        if constexpr (!StageQ) {
            const auto load_pair = [](const __nv_bfloat16* ptr) {
                if constexpr (Aligned) return common::load_vec<unsigned>(ptr);
                return unsigned(__bfloat16_as_ushort(ptr[0])) |
                       (unsigned(__bfloat16_as_ushort(ptr[1])) << 16);
            };
            const auto* q0 = q_row(row0);
            const auto* q1 = q_row(row1);
#pragma unroll
            for (int ks = 0; ks < D / 16; ++ks) {
                const int d = ks * 16 + 2 * lid;
                qf[ks][0]   = row0 < live_rows ? load_pair(q0 + d) : 0;
                qf[ks][1]   = row1 < live_rows ? load_pair(q1 + d) : 0;
                qf[ks][2]   = row0 < live_rows ? load_pair(q0 + d + 8) : 0;
                qf[ks][3]   = row1 < live_rows ? load_pair(q1 + d + 8) : 0;
            }
        }

        float acc[D / 8][4]{};
        float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F;
        float l0 = 0.F, l1 = 0.F;
        const float log_scale = a.scale * kGroup3Log2E;
        for (int tile = begin; tile < end; ++tile) {
            common::cp_wait<0>();
            __syncthreads();
            if constexpr (StageQ) {
                if (tile == begin) {
                    const int qr = warp_m * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
#pragma unroll
                    for (int ks = 0; ks < D / 16; ++ks) {
                        const int d       = ks * 16 + (lane >> 4) * 8;
                        const int safe_qr = qr < rows ? qr : qr - 8;
                        common::ldmatrix_x4(qf[ks][0], qf[ks][1], qf[ks][2], qf[ks][3],
                                            common::smem_addr(sq + safe_qr * D +
                                                              common::swizzle_xor_64(safe_qr, d)));
                    }
                }
            }
            if (tile + 1 < end) stage(tile + 1);
            const auto* k       = scratch3 + (tile % Stages) * TileElements + warp_k * KeyBlock * D;
            const auto* v       = k + BlockKeys * D;
            const bool is_query = tile == total_tiles - 1;
            const int key0      = context_start + tile * BlockKeys + warp_k * KeyBlock;
            const int key_count = is_query ? valid - warp_k * KeyBlock : length - key0;
            const unsigned kbase       = common::smem_addr(k + (lane & 7) * D);
            const unsigned vbase       = common::smem_addr(v + (lane & 15) * D);
            const unsigned swizzle_row = (lane & 7) * 16;
            float s[KeyBlock / 8][4]{};
#pragma unroll
            for (int n = 0; n < KeyBlock / 8; ++n) {
#pragma unroll
                for (int ks = 0; ks < D / 16; ++ks) {
                    unsigned b0, b1;
                    const unsigned d = ks * 32 + ((lane >> 3) & 1) * 16;
                    common::ldmatrix_x2(b0, b1, kbase + n * 8 * D * 2 + (d ^ swizzle_row));
                    common::mma_bf16(s[n][0], s[n][1], s[n][2], s[n][3], qf[ks][0], qf[ks][1],
                                     qf[ks][2], qf[ks][3], b0, b1);
                }
            }
            float next0 = m0, next1 = m1;
#pragma unroll
            for (int n = 0; n < KeyBlock / 8; ++n) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int c         = n * 8 + 2 * lid + j % 2;
                    const bool row_live = (j < 2 ? row0 : row1) < live_rows;
                    const bool visible =
                        Paged || is_query || key0 + c >= (j < 2 ? pos0 : pos1) - a.window_mask;
                    s[n][j] =
                        row_live && c < key_count && visible ? s[n][j] * log_scale : -CUDART_INF_F;
                }
                next0 = fmaxf(next0, fmaxf(s[n][0], s[n][1]));
                next1 = fmaxf(next1, fmaxf(s[n][2], s[n][3]));
            }
            next0              = common::warp_max<4>(next0, kGroup3Mask);
            next1              = common::warp_max<4>(next1, kGroup3Mask);
            const float alpha0 = group3_weight(m0, next0);
            const float alpha1 = group3_weight(m1, next1);
            m0                 = next0;
            m1                 = next1;
            float sum0 = 0.F, sum1 = 0.F;
            unsigned p[KeyBlock / 16][4];
#pragma unroll
            for (int n = 0; n < KeyBlock / 8; ++n) {
                const float p00 = group3_weight(s[n][0], m0);
                const float p01 = group3_weight(s[n][1], m0);
                const float p10 = group3_weight(s[n][2], m1);
                const float p11 = group3_weight(s[n][3], m1);
                sum0 += p00 + p01;
                sum1 += p10 + p11;
                p[n / 2][(n & 1) * 2]     = common::pack_f16x2(p00, p01);
                p[n / 2][(n & 1) * 2 + 1] = common::pack_f16x2(p10, p11);
            }
            l0 = l0 * alpha0 + common::warp_sum<4>(sum0, kGroup3Mask);
            l1 = l1 * alpha1 + common::warp_sum<4>(sum1, kGroup3Mask);
#pragma unroll
            for (int n = 0; n < D / 8; ++n) {
                acc[n][0] *= alpha0;
                acc[n][1] *= alpha0;
                acc[n][2] *= alpha1;
                acc[n][3] *= alpha1;
            }
#pragma unroll
            for (int pk = 0; pk < KeyBlock / 16; ++pk) {
#pragma unroll
                for (int n = 0; n < D / 8; n += 2) {
                    unsigned b[4];
                    const unsigned d = n * 16 + (lane >> 4) * 16;
                    common::ldmatrix_x4_t(b[0], b[1], b[2], b[3],
                                          vbase + pk * 16 * D * 2 + (d ^ swizzle_row));
                    common::mma_f16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], p[pk][0], p[pk][1],
                                    p[pk][2], p[pk][3], b[0], b[1]);
                    common::mma_f16(acc[n + 1][0], acc[n + 1][1], acc[n + 1][2], acc[n + 1][3],
                                    p[pk][0], p[pk][1], p[pk][2], p[pk][3], b[2], b[3]);
                }
            }
        }

        if constexpr (WarpsK == 2) {
            __syncthreads();
            auto* other   = reinterpret_cast<float*>(scratch3);
            auto* other_m = other + rows * D;
            auto* other_l = other_m + rows;
            if (warp_k != 0) {
                if (lid == 0) {
                    if (row0 < live_rows) {
                        other_m[row0] = m0;
                        other_l[row0] = l0;
                    }
                    if (row1 < live_rows) {
                        other_m[row1] = m1;
                        other_l[row1] = l1;
                    }
                }
#pragma unroll
                for (int n = 0; n < D / 8; ++n) {
                    const int d = n * 8 + 2 * lid;
                    if (row0 < live_rows)
                        common::store_vec(other + row0 * D + d, make_float2(acc[n][0], acc[n][1]));
                    if (row1 < live_rows)
                        common::store_vec(other + row1 * D + d, make_float2(acc[n][2], acc[n][3]));
                }
            }
            __syncthreads();
            if (warp_k != 0) return;
            const float om0 = row0 < live_rows ? other_m[row0] : -CUDART_INF_F;
            const float om1 = row1 < live_rows ? other_m[row1] : -CUDART_INF_F;
            const float mx0 = fmaxf(m0, om0), mx1 = fmaxf(m1, om1);
            const float a0 = group3_weight(m0, mx0), a1 = group3_weight(m1, mx1);
            const float b0 = group3_weight(om0, mx0), b1 = group3_weight(om1, mx1);
            l0 = l0 * a0 + (row0 < live_rows ? other_l[row0] * b0 : 0.F);
            l1 = l1 * a1 + (row1 < live_rows ? other_l[row1] * b1 : 0.F);
            m0 = mx0;
            m1 = mx1;
#pragma unroll
            for (int n = 0; n < D / 8; ++n) {
                const int d = n * 8 + 2 * lid;
                if (row0 < live_rows) {
                    const auto y = common::load_vec<float2>(other + row0 * D + d);
                    acc[n][0]    = acc[n][0] * a0 + y.x * b0;
                    acc[n][1]    = acc[n][1] * a0 + y.y * b0;
                }
                if (row1 < live_rows) {
                    const auto y = common::load_vec<float2>(other + row1 * D + d);
                    acc[n][2]    = acc[n][2] * a1 + y.x * b1;
                    acc[n][3]    = acc[n][3] * a1 + y.y * b1;
                }
            }
        }

        const auto output_row = [&](int row) {
            return std::int64_t(batch) * batch_rows + (row / 3) * kGroup3Heads + kv_head * 3 +
                   row % 3;
        };
        const auto slot0 = output_row(row0) * a.splits + split;
        const auto slot1 = output_row(row1) * a.splits + split;
        if constexpr (!Direct) {
            if (lid == 0) {
                if (row0 < live_rows) {
                    a.maxima[slot0]      = m0;
                    a.normalizers[slot0] = l0;
                }
                if (row1 < live_rows) {
                    a.maxima[slot1]      = m1;
                    a.normalizers[slot1] = l1;
                }
            }
        }
        // Half the split mean leaves headroom for rounding the MMA probability operands.
        constexpr float PartialScale = Direct ? 1.F : 0.5F;
        const float inv0             = l0 > 0.F ? PartialScale / l0 : 0.F;
        const float inv1             = l1 > 0.F ? PartialScale / l1 : 0.F;
        const auto store_pair        = [&](int row, std::int64_t slot, int d, float x, float y) {
            if constexpr (Direct) {
                if (row < rows) {
                    auto* out = a.output + slot * D + d;
                    if constexpr (Aligned) {
                        common::store_vec(out, common::pack_bf16x2(x, y));
                    } else {
                        out[0] = __float2bfloat16(x);
                        out[1] = __float2bfloat16(y);
                    }
                }
            } else if (row < live_rows) {
                common::store_vec(a.partial + slot * D + d, common::pack_f16x2(x, y));
            }
        };
#pragma unroll
        for (int n = 0; n < D / 8; ++n) {
            const int d = n * 8 + 2 * lid;
            store_pair(row0, slot0, d, acc[n][0] * inv0, acc[n][1] * inv0);
            store_pair(row1, slot1, d, acc[n][2] * inv1, acc[n][3] * inv1);
        }
    }
};

template <bool Paged, int BlockKeys, bool Aligned>
__device__ __forceinline__ void group3_reduce_row(const Group3Arguments& a, int batch, int row) {
    const int lane        = threadIdx.x & 31;
    const int token       = row / kGroup3Heads;
    const int valid       = a.valid[batch];
    const auto output_row = std::int64_t(batch) * (a.tokens * kGroup3Heads) + row;
    auto* output          = a.output + output_row * kGroup3D;
    const int length      = token < valid ? group3_length<Paged>(a, batch) : 0;
    if (valid > a.tokens || token >= valid || length < 0 || length > a.max_context) {
        if (lane < 16) {
            if constexpr (Aligned) {
                common::store_vec(output + lane * 8, make_uint4(0, 0, 0, 0));
            } else {
#pragma unroll
                for (int j = 0; j < 8; ++j) output[lane * 8 + j] = __float2bfloat16(0.F);
            }
        }
        return;
    }
    const int active    = min(a.splits, group3_work_tiles<Paged, BlockKeys>(a, length));
    const auto base     = output_row * a.splits;
    const float m0      = lane < active ? a.maxima[base + lane] : -CUDART_INF_F;
    const float m1      = lane + 32 < active ? a.maxima[base + lane + 32] : -CUDART_INF_F;
    const float maximum = common::warp_max<32>(fmaxf(m0, m1), kGroup3Mask);
    const float w0 = lane < active ? group3_weight(m0, maximum) * a.normalizers[base + lane] : 0.F;
    const float w1 =
        lane + 32 < active ? group3_weight(m1, maximum) * a.normalizers[base + lane + 32] : 0.F;
    const float denominator = common::warp_sum<32>(w0 + w1, kGroup3Mask);
    const int d             = (lane & 15) * 8;
    float out[8]{};
    for (int pair = 0; pair < active; pair += 2) {
        const int split    = pair + (lane >> 4);
        const float weight = __shfl_sync(kGroup3Mask, pair < 32 ? w0 : w1, split & 31);
        if (split < active) {
            const uint4 packed = common::load_vec<uint4>(a.partial + (base + split) * kGroup3D + d);
            const unsigned bits[4]{packed.x, packed.y, packed.z, packed.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 x = __half22float2(common::load_vec<__half2>(bits + j));
                out[j * 2] += weight * x.x;
                out[j * 2 + 1] += weight * x.y;
            }
        }
    }
    const float inv = denominator > 0.F ? 2.F / denominator : 0.F;
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] += __shfl_xor_sync(kGroup3Mask, out[j], 16);
    if (lane < 16) {
        if constexpr (Aligned) {
            common::store_vec(output + d,
                              make_uint4(common::pack_bf16x2(out[0] * inv, out[1] * inv),
                                         common::pack_bf16x2(out[2] * inv, out[3] * inv),
                                         common::pack_bf16x2(out[4] * inv, out[5] * inv),
                                         common::pack_bf16x2(out[6] * inv, out[7] * inv)));
        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j) output[d + j] = __float2bfloat16(out[j] * inv);
        }
    }
}

template <bool Paged, int BlockKeys, int Warps, bool Aligned>
__launch_bounds__(Warps * 32) __global__
    void group3_reduce(const __grid_constant__ Group3Arguments a) {
    const int row = blockIdx.x * Warps + threadIdx.x / 32;
    if (row < a.tokens * kGroup3Heads)
        group3_reduce_row<Paged, BlockKeys, Aligned>(a, blockIdx.z, row);
}

template <bool Paged, int Tokens, int MWarps, int KeyBlock, int WarpsK, bool StageQ, bool Aligned,
          bool Direct>
__launch_bounds__(MWarps* WarpsK * 32, 2) __global__
    void group3_partial(const __grid_constant__ Group3Arguments a) {
    using P = Group3Pipeline<Paged, Tokens, MWarps, KeyBlock, WarpsK, StageQ, Aligned, Direct>;
    P::run(a, blockIdx.z, blockIdx.x, blockIdx.y);
}

static __global__ void group3_idle() {}

} // namespace ninfer::ops::detail
