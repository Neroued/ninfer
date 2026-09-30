#pragma once

#include "ops/softmax_attention/dense/causal_cache/bf16/epilogue.cuh"
#include "ops/softmax_attention/dense/causal_cache/bf16/softmax.cuh"
#include "ops/softmax_attention/dense/causal_cache/bf16/split_policy.h"

namespace ninfer::ops::detail {

template <class G, class S, class Input>
__device__ __forceinline__ void
bf16_kv_load_grouped_tile(__nv_bfloat16* key_tile, __half* value_tile, const __nv_bfloat16* keys,
                          const __half* values, Input input, int page, int head, int begin, int end,
                          int first_position, int tid) {
    constexpr int D       = G::kHeadDim;
    const auto cache_base = bf16_kv_cache_index<G>(page, head, 0, begin & kPagedKVPageMask);
    // Small query tiles need enough outstanding loads to hide memory latency.
#pragma unroll 1
    for (int i = tid; i < S::kKeyRows * (D / 8); i += S::kThreads) {
        const int row = i / (D / 8), d = i % (D / 8) * 8, key = begin + row;
        const auto offset = row * D + causal_swizzle(row, d);
        auto* kd          = key_tile + offset;
        auto* vd          = value_tile + offset;
        if (key >= end) {
            store_vec(kd, make_int4(0, 0, 0, 0));
            store_vec(vd, make_int4(0, 0, 0, 0));
        } else {
            if constexpr (Input::writes_cache) {
                if (key >= first_position) {
                    const auto index = causal_new_index<G>(head, d, key - first_position);
                    cp_async<16>(kd, input.k + index);
                    store_vec(vd, bf16x8_bits_to_f16x8_bits(load_vec<int4>(input.v + index)));
                    continue;
                }
            }
            const auto index = cache_base + row * D + d;
            cp_async<16>(kd, keys + index);
            cp_async<16>(vd, values + index);
        }
    }
}

// Packed-query tiles and KV partitions are independent grid axes. Each KV row
// has one append owner; all query CTAs read new rows from the immutable inputs.
template <class G, class S, bool MultiBatch, bool Masked, class Input>
__launch_bounds__(S::kLaunchBoundThreads, S::kMinBlocks) __global__
    void bf16_kv_grouped_mma_kernel(const __nv_bfloat16* q, Input input, const int* positions,
                                    typename Bf16KvCacheView<Input::writes_cache>::Key* cache_k,
                                    typename Bf16KvCacheView<Input::writes_cache>::Value* cache_v,
                                    const int* tables, const int* validity, const int* table_rows,
                                    int table_stride, int runtime_width, float scale,
                                    Bf16KvPartition partition, CausalPartialView partial) {
    const int width = S::kFixedWidth ? S::kFixedWidth : runtime_width;
    constexpr int D = G::kHeadDim, M = S::kQueryRows, N = S::kKeyRows;
    constexpr int NK = N / S::kWarpsKV, QKNt = NK / 8, QKKs = D / 16;
    constexpr int PVNt = D / 8, PVKs = NK / 16;
    using Storage = Bf16KvGroupedStorage<G, S>;
    static_assert(sizeof(Storage) <= 99 * 1024);
    Storage* storage;
    if constexpr (sizeof(Storage) <= 48 * 1024) {
        __shared__ Storage fixed;
        storage = &fixed;
    } else {
        extern __shared__ __align__(16) unsigned char dynamic[];
        storage = reinterpret_cast<Storage*>(dynamic);
    }
    auto* k_s     = reinterpret_cast<__nv_bfloat16*>(storage->qkv);
    auto* v_s     = reinterpret_cast<__half*>(storage->qkv + N * D);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int warp_q = warp % S::kWarpsQ, warp_kv = warp / S::kWarpsQ;
    const int head  = S::kFixedWidth ? blockIdx.x : blockIdx.x % G::KVHeads;
    const int tile  = S::kFixedWidth ? 0 : blockIdx.x / G::KVHeads;
    const int split = blockIdx.y, batch = MultiBatch ? blockIdx.z : 0;
    const int row_begin = tile * M, packed_rows = width * G::GroupSize;
    const int live = Masked ? validity[batch] : width;
    q += static_cast<std::int64_t>(batch) * width * D * G::QHeads;
    positions += batch * width;
    if constexpr (Input::writes_cache) {
        input.k += static_cast<std::int64_t>(batch) * width * D * G::KVHeads;
        input.v += static_cast<std::int64_t>(batch) * width * D * G::KVHeads;
    }
    {
        partial.acc +=
            static_cast<std::int64_t>(batch) * width * G::QHeads * D * partition.capacity;
        partial.maximum +=
            static_cast<std::int64_t>(batch) * width * G::QHeads * partition.capacity;
        partial.sum += static_cast<std::int64_t>(batch) * width * G::QHeads * partition.capacity;
    }
    const auto neutral = [&]() {
        for (int i = tid; i < M * (D / 2); i += S::kThreads) {
            const int row = row_begin + i / (D / 2), d = i % (D / 2) * 2;
            if (row < packed_rows) {
                const int token = row / G::GroupSize;
                const int h     = head * G::GroupSize + row % G::GroupSize;
                bf16_kv_store_pair<G, true>(partial, nullptr, h, token, d, width, split, 0.0f, 0.0f,
                                            -CUDART_INF_F, 0.0f);
            }
        }
    };
    if (row_begin >= live * G::GroupSize) return;
    const int first = positions[0], window = positions[live - 1] + 1;
    const auto work = partition.live(window);
    if (split >= work.splits) return;
    const int start     = split * work.keys_per_split;
    const int stop      = min(window, start + work.keys_per_split);
    const int table_row = table_rows ? table_rows[batch] : 0;
    const int* table    = tables + static_cast<std::int64_t>(table_row) * table_stride;
    if constexpr (Input::writes_cache) {
        if (tile == 0) {
            for (int i = tid; i < live * (D / 8); i += S::kThreads) {
                const int token = i / (D / 8), d = i % (D / 8) * 8, key = positions[token];
                if (key >= start && key < stop) {
                    const auto src = causal_new_index<G>(head, d, token);
                    const auto dst = bf16_kv_cache_index<G>(table[key >> kPagedKVPageShift], head,
                                                            d, key & kPagedKVPageMask);
                    store_vec(cache_k + dst, load_vec<int4>(input.k + src));
                    store_vec(cache_v + dst,
                              bf16x8_bits_to_f16x8_bits(load_vec<int4>(input.v + src)));
                }
            }
        }
    }
    const int last_token = min(live, div_up(row_begin + M, G::GroupSize)) - 1;
    const int end        = min(stop, positions[last_token] + 1);
    if (start >= end) {
        neutral();
        return;
    }
    for (int i = tid; i < M * (D / 8); i += S::kThreads) {
        const int row = i / (D / 8), d = i % (D / 8) * 8, packed = row_begin + row;
        const int token = packed / G::GroupSize, h = head * G::GroupSize + packed % G::GroupSize;
        auto* dst = k_s + row * D + causal_swizzle(row, d);
        if (token < live)
            cp_async<16>(dst, q + causal_q_index<G>(h, d, token));
        else
            store_vec(dst, make_int4(0, 0, 0, 0));
    }
    cp_commit();
    cp_wait<0>();
    __syncthreads();
    unsigned q_frag[QKKs][4];
    const int a_row = warp_q * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
#pragma unroll
    for (int k = 0; k < QKKs; ++k) {
        const int d = k * 16 + (lane >> 4) * 8;
        ldmatrix_x4(q_frag[k][0], q_frag[k][1], q_frag[k][2], q_frag[k][3],
                    smem_addr(k_s + a_row * D + causal_swizzle(a_row, d)));
    }
    __syncthreads();
    float acc[PVNt][4] = {};
    Bf16KvSoftmaxRow state[2];
    const float scale_log2 = scale * kLog2E;
    const int gid = lane >> 2, lid = lane & 3;
    const int rows[2]   = {row_begin + warp_q * 16 + gid, row_begin + warp_q * 16 + gid + 8};
    const int tokens[2] = {rows[0] / G::GroupSize, rows[1] / G::GroupSize};
    const int qabs[2]   = {tokens[0] < live ? positions[tokens[0]] : -1,
                         tokens[1] < live ? positions[tokens[1]] : -1};
    int page_window = -Storage::kPageWindow;
    for (int k0 = start; k0 < end; k0 += N) {
        const int logical_page = k0 >> kPagedKVPageShift;
        if (logical_page >= page_window + Storage::kPageWindow) {
            page_window = logical_page;
            const int page_count =
                min(Storage::kPageWindow, div_up(end, kPagedKVPageSize) - page_window);
            for (int i = tid; i < page_count; i += S::kThreads)
                storage->pages[i] = table[page_window + i];
            __syncthreads();
        }
        if (paged_kv_page_is_hole(storage->pages[logical_page - page_window])) continue;
        bf16_kv_load_grouped_tile<G, S>(k_s, v_s, cache_k, cache_v, input,
                                        storage->pages[logical_page - page_window], head, k0, end,
                                        first, tid);
        cp_commit();
        cp_wait<0>();
        __syncthreads();
        float score[QKNt][4] = {};
        // Q and O already occupy 192 registers per lane at D=256. Keep one
        // K fragment live at a time; double buffering all N fragments spills.
#pragma unroll
        for (int n = 0; n < QKNt; ++n) {
#pragma unroll
            for (int k = 0; k < QKKs; ++k) {
                unsigned bf[2];
                const int row = warp_kv * NK + n * 8 + (lane & 7);
                const int d   = k * 16 + ((lane >> 3) & 1) * 8;
                ldmatrix_x2(bf[0], bf[1], smem_addr(k_s + row * D + causal_swizzle(row, d)));
                mma_bf16(score[n][0], score[n][1], score[n][2], score[n][3], q_frag[k][0],
                         q_frag[k][1], q_frag[k][2], q_frag[k][3], bf[0], bf[1]);
            }
        }
        float maximum[2] = {-CUDART_INF_F, -CUDART_INF_F};
#pragma unroll
        for (int n = 0; n < QKNt; ++n) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int key = k0 + warp_kv * NK + n * 8 + 2 * lid + (j & 1);
                if (key >= end || key > qabs[j / 2]) score[n][j] = -CUDART_INF_F;
                maximum[j / 2] = fmaxf(maximum[j / 2], score[n][j]);
            }
        }
        const float alpha[2] = {state[0].update(warp_max<4>(maximum[0], 0xffffffffu), scale_log2),
                                state[1].update(warp_max<4>(maximum[1], 0xffffffffu), scale_log2)};
        unsigned pf[PVKs][4];
        float tile_sum[2] = {};
#pragma unroll
        for (int n = 0; n < QKNt; ++n) {
            const float p0 = state[0].probability(score[n][0], scale_log2);
            const float p1 = state[0].probability(score[n][1], scale_log2);
            const float p2 = state[1].probability(score[n][2], scale_log2);
            const float p3 = state[1].probability(score[n][3], scale_log2);
            tile_sum[0] += p0 + p1;
            tile_sum[1] += p2 + p3;
            pf[n / 2][(n % 2) * 2]     = pack_f16x2(p0, p1);
            pf[n / 2][(n % 2) * 2 + 1] = pack_f16x2(p2, p3);
        }
        state[0].accumulate(alpha[0], tile_sum[0]);
        state[1].accumulate(alpha[1], tile_sum[1]);
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            acc[n][0] *= alpha[0];
            acc[n][1] *= alpha[0];
            acc[n][2] *= alpha[1];
            acc[n][3] *= alpha[1];
        }
        // Batched and read-only wide tiles benefit from paired V loads. Single-row
        // fused append uses scalar fragments to leave room for its input state.
        if constexpr (S::kWarpsQ >= 4 && (MultiBatch || !Input::writes_cache)) {
            unsigned vf[2][4];
            const auto load_v = [&](int i, int slot) {
                const int k = i / (PVNt / 2), n = i % (PVNt / 2) * 2;
                const int row = warp_kv * NK + k * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
                const int d   = n * 8 + (lane >> 4) * 8;
                ldmatrix_x4_t(vf[slot][0], vf[slot][1], vf[slot][2], vf[slot][3],
                              smem_addr(v_s + row * D + causal_swizzle(row, d)));
            };
            load_v(0, 0);
#pragma unroll
            for (int i = 0; i < PVKs * (PVNt / 2); ++i) {
                const int k = i / (PVNt / 2), n = i % (PVNt / 2) * 2, slot = i & 1;
                if (i + 1 < PVKs * (PVNt / 2)) load_v(i + 1, slot ^ 1);
                mma_f16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[k][0], pf[k][1], pf[k][2],
                        pf[k][3], vf[slot][0], vf[slot][1]);
                mma_f16(acc[n + 1][0], acc[n + 1][1], acc[n + 1][2], acc[n + 1][3], pf[k][0],
                        pf[k][1], pf[k][2], pf[k][3], vf[slot][2], vf[slot][3]);
            }
        } else {
            // Limit live V fragments too; the grouped path is bandwidth-bound.
#pragma unroll
            for (int n = 0; n < PVNt; ++n) {
#pragma unroll
                for (int k = 0; k < PVKs; ++k) {
                    unsigned vf[2];
                    const int row = warp_kv * NK + k * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
                    const int d   = n * 8;
                    ldmatrix_x2_t(vf[0], vf[1], smem_addr(v_s + row * D + causal_swizzle(row, d)));
                    mma_f16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[k][0], pf[k][1],
                            pf[k][2], pf[k][3], vf[0], vf[1]);
                }
            }
        }
        __syncthreads();
    }
    state[0].finish();
    state[1].finish();
    float m[2]    = {state[0].maximum * scale_log2, state[1].maximum * scale_log2};
    float sums[2] = {state[0].sum, state[1].sum};
    if constexpr (S::kWarpsKV > 1) {
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int row = warp_kv * M + warp_q * 16 + gid + j * 8;
            if (lid == 0) {
                storage->maximum[row] = m[j];
                storage->sum[row]     = sums[j];
            }
#pragma unroll
            for (int n = 0; n < PVNt; ++n) {
                const int d = n * 8 + 2 * lid;
                *reinterpret_cast<float2*>(storage->reduction + row * D + d) =
                    make_float2(acc[n][j * 2], acc[n][j * 2 + 1]);
            }
        }
        __syncthreads();
        if (warp_kv != 0) return;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int row = warp_q * 16 + gid + j * 8;
            m[j]          = -CUDART_INF_F;
            sums[j]       = 0;
#pragma unroll
            for (int w = 0; w < S::kWarpsKV; ++w) m[j] = fmaxf(m[j], storage->maximum[w * M + row]);
            float weights[S::kWarpsKV];
#pragma unroll
            for (int w = 0; w < S::kWarpsKV; ++w) {
                weights[w] = bf16_kv_state_weight(storage->maximum[w * M + row],
                                                  storage->sum[w * M + row], m[j]);
                sums[j] += weights[w] * storage->sum[w * M + row];
            }
#pragma unroll
            for (int n = 0; n < PVNt; ++n) {
                float a = 0, b = 0;
#pragma unroll
                for (int w = 0; w < S::kWarpsKV; ++w) {
                    const auto x = *reinterpret_cast<const float2*>(
                        storage->reduction + (w * M + row) * D + n * 8 + 2 * lid);
                    a += weights[w] * x.x;
                    b += weights[w] * x.y;
                }
                acc[n][2 * j]     = a;
                acc[n][2 * j + 1] = b;
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        if (tokens[j] < width) {
            const int h = head * G::GroupSize + rows[j] % G::GroupSize;
#pragma unroll
            for (int n = 0; n < PVNt; ++n)
                bf16_kv_store_pair<G, true>(partial, nullptr, h, tokens[j], n * 8 + 2 * lid, width,
                                            split, acc[n][2 * j], acc[n][2 * j + 1], m[j], sums[j]);
        }
    }
}

} // namespace ninfer::ops::detail
