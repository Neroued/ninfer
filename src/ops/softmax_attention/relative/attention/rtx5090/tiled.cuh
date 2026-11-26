#pragma once

#include "ops/softmax_attention/relative/attention/rtx5090/device.cuh"

#include <type_traits>

namespace ninfer::ops::relative {
namespace mma = common;

template <int Warps, int K, bool RegisterQ, int QRows, int Stages>
struct QueryStorage {
    std::uint16_t q[Warps * 16 * kDim];
    float bias[Warps][16][K + 16];
};

template <int Warps, int K, int QRows, int Stages>
struct QueryStorage<Warps, K, true, QRows, Stages> {
    float bias[Warps][16][K + 16];
    alignas(16) std::uint16_t projection[Stages][K + QRows][kRelDim];
};

template <int K, bool Fp8>
struct EncodedKV {};

template <int K>
struct alignas(16) EncodedKV<K, true> {
    unsigned char codes[2][K * kDim];
    unsigned char scales[2][K];
};

template <int K>
struct ExpandedKV {
    std::uint16_t k[K * kDim];
    std::uint16_t v[K * kDim];
};

template <int K>
struct RawKV {
    // Consumers widen directly from the current compressed tile while the next tile arrives.
    EncodedKV<K, true> raw[2];
};

template <int Warps, bool RegisterQ, bool Fp8, int K = kKeyTile, int QRows = 16,
          bool UseRawFp8 = false>
struct alignas(16) TileStorage : EncodedKV<K, Fp8 && !UseRawFp8>,
                                 std::conditional_t<Fp8 && UseRawFp8, RawKV<K>, ExpandedKV<K>> {
    QueryStorage<Warps, K, RegisterQ, QRows, Fp8 ? 2 : 1> input;
};

template <bool Swa, bool Value, int K, int Threads>
__device__ __forceinline__ void stage_16bit(std::uint16_t* dest, const CacheRows<Swa, false>& cache,
                                            int key, int lo, int hi, std::int64_t record) {
    const int count     = min(K - 1, hi - key) + 1;
    const bool chunk    = cache.is_chunk(key);
    const bool boundary = !chunk && cache.is_chunk(key + (count - 1));
    const int stride    = chunk ? kKvHeads * (2 * kDim) : 2 * kDim;
    const auto* base    = cache.codes(Value, key + (boundary ? count - 1 : 0), record);
    const int lower     = Swa ? max(0, lo - key) : 0;
#pragma unroll
    for (int i = threadIdx.x; i < K * (kDim / 8); i += Threads) {
        const int row = i / (kDim / 8), d = i % (kDim / 8) * 8;
        const bool live = unsigned(row - lower) < unsigned(count - lower);
        const auto* src = base + row * stride;
        if constexpr (Swa)
            if (boundary) src = live ? cache.codes(Value, key + row, record) : base;
        mma::cp_async_zfill<16, Cache::cg>(dest + row * kDim + mma::swizzle_xor_64(row, d),
                                           src + d * 2, live ? 16 : 0);
    }
}

template <bool Swa, int K, int Threads, bool Swizzle = false>
__device__ __forceinline__ void
stage_fp8(EncodedKV<K, true>& dest, const CacheRows<Swa, true>& cache, int key, int lo, int hi) {
    const int count        = min(K - 1, hi - key) + 1;
    const auto record      = cache.record(key);
    const bool chunk       = cache.is_chunk(key);
    const bool boundary    = !chunk && cache.is_chunk(key + (count - 1));
    const int stride       = chunk ? kKvHeads * kDim : kDim;
    const int lower        = Swa ? max(0, lo - key) : 0;
    const int scale_index  = int(threadIdx.x) - (Threads - 2 * K);
    unsigned char exponent = 0;
    const bool swa_vec_scales =
        Swa && !chunk && !boundary && lower == 0 &&
        ((reinterpret_cast<std::uintptr_t>(cache.scale(0, key, record)) |
          reinterpret_cast<std::uintptr_t>(cache.scale(1, key, record))) &
         15u) == 0;
    if constexpr (Swa) {
        static_assert(2 * K <= Threads);
        // Chunk scales are head-interleaved and boundary tiles cross storage regions;
        // pure cyclic tiles have contiguous aligned scale bytes that stream via cp.async.
        if (!swa_vec_scales && unsigned(scale_index) < 2 * K) {
            const int plane = scale_index / K, row = scale_index % K;
            if (unsigned(row - lower) < unsigned(count - lower)) {
                const auto* source =
                    boundary ? cache.scale(plane, key + row, record)
                             : cache.scale(plane, key, record) + row * (chunk ? kKvHeads : 1);
                exponent = *source;
            }
        }
    }
#pragma unroll
    for (int i = threadIdx.x; i < 2 * K * (kDim / 16); i += Threads) {
        constexpr int Vectors = K * (kDim / 16);
        const int plane = i / Vectors, index = i % Vectors;
        const int row = index / (kDim / 16), col = index % (kDim / 16) * 16;
        const bool live  = unsigned(row - lower) < unsigned(count - lower);
        const auto* base = cache.codes(plane, key + (boundary ? count - 1 : 0), record);
        const auto* src  = base + row * stride;
        if constexpr (Swa)
            if (boundary) src = live ? cache.codes(plane, key + row, record) : base;
        const int target = Swizzle ? mma::swizzle_xor_128(row, col) : col;
        mma::cp_async_zfill<16, Cache::cg>(dest.codes[plane] + row * kDim + target, src + col,
                                           live ? 16 : 0);
    }
    if constexpr (Swa) {
        if (swa_vec_scales) {
            if (threadIdx.x < 2 * K / 16) {
                const int byte     = threadIdx.x * 16;
                const int plane    = byte / K, index = byte % K;
                const auto* scales = cache.scale(plane, key, record);
                mma::cp_async_zfill<16, Cache::cg>(dest.scales[plane] + index, scales + index,
                                                   min(16, max(0, count - index)));
            }
        } else if (unsigned(scale_index) < 2 * K) {
            dest.scales[scale_index / K][scale_index % K] = exponent;
        }
    } else if (threadIdx.x < 2 * K / 16) {
        // Paged scales have aligned bases and a 64-byte head stride.
        const int byte  = threadIdx.x * 16;
        const int plane = byte / K, index = byte % K;
        const auto* scales = cache.scale(plane, key, record);
        mma::cp_async_zfill<16, Cache::cg>(dest.scales[plane] + index, scales + index,
                                           min(16, max(0, count - index)));
    }
}

template <int K, int Threads>
__device__ __forceinline__ void unpack_fp8(std::uint16_t* keys, std::uint16_t* values,
                                           const EncodedKV<K, true>& src) {
    constexpr int Vectors = K * (kDim / 16);
#pragma unroll
    for (int i = threadIdx.x; i < 2 * Vectors; i += Threads) {
        const int plane = i / Vectors, index = i % Vectors;
        const int row = index / (kDim / 16), col = index % (kDim / 16) * 16;
        const auto codes     = mma::load_vec<uint4>(src.codes[plane] + index * 16);
        const unsigned scale = src.scales[plane][row];
        const auto widen     = [scale](unsigned lo, unsigned hi) {
            return make_uint4(
                mma::fp8_dequant_bf16x2(lo, scale), mma::fp8_dequant_bf16x2(lo >> 16, scale),
                mma::fp8_dequant_bf16x2(hi, scale), mma::fp8_dequant_bf16x2(hi >> 16, scale));
        };
        auto* dest = (plane ? values : keys) + row * kDim;
        mma::store_vec(dest + mma::swizzle_xor_64(row, col), widen(codes.x, codes.y));
        mma::store_vec(dest + mma::swizzle_xor_64(row, col + 8), widen(codes.z, codes.w));
    }
}

template <int K, int Threads, int QRows>
__device__ __forceinline__ void stage_projection(std::uint16_t (&dst)[K + QRows][kRelDim],
                                                 const std::uint16_t* projection, int low, int key,
                                                 bool shear) {
    if (!shear || low - (key + (K - 1)) >= kExtent) return;
    const int origin = low - key + QRows;
#pragma unroll
    for (int i = threadIdx.x; i < 2 * (K + QRows); i += Threads) {
        const int d        = origin - i / 2;
        const auto* source = unsigned(d) < kExtent ? projection + d * kRelDim : projection;
        const int half     = (i & 1) * 8;
        mma::cp_async_zfill<16, Cache::ca>(dst[i / 2] + half, source + half,
                                           unsigned(d) < kExtent ? 16 : 0);
    }
}

__device__ __forceinline__ unsigned widen_value_pair(unsigned raw, unsigned s0, unsigned s1,
                                                     bool fold) {
    const unsigned codes = common::fp8_e4m3x2_to_bf16x2_bits(raw);
    if (fold) return codes;
    const auto s      = [](unsigned e) { return e ? e << 7 : 0x40u; };
    const auto scales = s(s0) | (s(s1) << 16);
    const auto value  = __hmul2(common::load_vec<__nv_bfloat162>(&codes),
                                common::load_vec<__nv_bfloat162>(&scales));
    return common::load_vec<unsigned>(&value);
}

template <int K, bool Swizzle>
__device__ __forceinline__ int bias_col(int row, int column) {
    if constexpr (!Swizzle) return column;
    if constexpr (K == 16) return column ^ ((row & 1) * 8);
    if (column >= K) return column ^ ((row & 1) * 8);
    // Permute the four adjacent 8-float rows written in each store transaction.
    return (column & ~31) + ((column & 31) ^ ((row & 3) * 8));
}

struct QueryRow {
    std::int64_t index;
    int position;
};

template <bool Ragged, int QRows, int HeadGroup>
__device__ __forceinline__ QueryRow query_row(Query q, std::int64_t first, int count,
                                              int first_head, int row, int limit) {
    const int token = Ragged ? row % QRows : row / kGroup;
    const int head  = first_head + (Ragged ? row / QRows : row % kGroup);
    int p = token < count && head < first_head + HeadGroup ? q.positions[first + token] : -1;
    return {(first + token) * kHeads + head, p >= 0 && p <= limit ? p : -1};
}

__device__ __forceinline__ unsigned load_pair(const unsigned short* data, int d) {
    return mma::load_vec<unsigned>(data + d);
}

template <bool Fp8, bool Swa, bool Ragged, int Warps, int Threads, bool Direct, int K = kKeyTile,
          int QRows = 16, int HeadGroup = kGroup>
__device__ __forceinline__ void
tiled_attention(Query q, CacheView cache, int sequence, std::int64_t sequence_first,
                std::int64_t first, int live, int width, int head_index, int split, int splits,
                Partials partials, std::size_t total_rows,
                TileStorage<Warps, Ragged || Swa, Fp8, K, QRows, Swa && Ragged>& smem) {
    constexpr bool RegisterQ  = Ragged || Swa;
    // Use cooperative CTA-wide FP8 unpacking + ldmatrix_x4/x4_t for Full Ragged, Full Verify,
    // and SWA Verify; SWA Ragged reads interleaved chunk scales and keeps RawFp8.
    constexpr bool RawFp8     = Fp8 && Swa && Ragged;
    constexpr bool PackedFp8  = Fp8 && !RawFp8;
    constexpr int ScoreTiles  = K / 8;
    constexpr int OutputTiles = kDim / 8;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    const int quad = lane >> 2, pair = (lane & 3) * 2;
    const int kvhead     = HeadGroup == 1 ? head_index / kGroup : head_index;
    const int first_head = head_index * HeadGroup;
    const int limit      = position_limit<Swa>(cache);
    const auto range = key_range<Swa, K, QRows>(q.positions + first, live, split, splits, limit);
    const int row0 = warp * 16 + quad, row1 = row0 + 8;
    const auto qr0   = query_row<Ragged, QRows, HeadGroup>(q, first, live, first_head, row0, limit);
    const auto qr1   = query_row<Ragged, QRows, HeadGroup>(q, first, live, first_head, row1, limit);
    const int local0 = Ragged ? row0 % QRows : row0 / kGroup;
    const int local1 = Ragged ? row1 % QRows : row1 / kGroup;
    const bool owns0 = local0 < width && (Ragged ? warp < Warps : row0 < kGroup * width);
    const bool owns1 = local1 < width && (Ragged ? warp < Warps : row1 < kGroup * width);
    // Only a live interval may read table rows or per-record offsets.
    const int high = range.high, low = range.low;
    const int begin         = range.first;
    const int first_visible = Swa ? max(0, low - (kWindow - 1)) : 0;
    const int table         = begin < range.last ? cache_row<Swa>(cache, sequence) : -1;
    float sum0 = 0, sum1 = 0;
    float ref0 = -CUDART_INF_F, ref1 = -CUDART_INF_F;
    float out[OutputTiles][4]{};
    if (table >= 0 && table < cache.table_count) {
        const bool staged_projection = high - low < QRows;
        int warp_low = low, warp_high = high;
        if constexpr (Ragged && QRows > 16) {
            const int t = (warp * 16) % QRows + lane;
            const int p = lane < 16 && t < live ? q.positions[first + t] : -1;
            warp_low    = __reduce_min_sync(kWarp, p >= 0 && p <= limit ? p : INT_MAX);
            warp_high   = __reduce_max_sync(kWarp, p <= limit ? p : -1);
        }
        const int projection_shift = QRows - 16 - (warp_low - low);
        const bool can_shear       = staged_projection && warp_high >= 0 && projection_shift >= 0 &&
                               warp_high - warp_low < 16;
        const float scale2 = q.scale * kLog2E;
        int chunk_start    = 0;
        if constexpr (Swa)
            if (cache.chunk.k) chunk_start = q.positions[sequence_first];
        const CacheRows<Swa, Fp8> rows(cache, table, kvhead, sequence_first, chunk_start,
                                       !cache.chunk.k ||
                                           max(begin * K, first_visible) < chunk_start);

        if constexpr (!RegisterQ) {
#pragma unroll
            for (int i = tid; i < Warps * 16 * (kDim / 8); i += Threads) {
                const int row = i / (kDim / 8), col = i % (kDim / 8) * 8;
                const auto input =
                    query_row<Ragged, QRows, HeadGroup>(q, first, live, first_head, row, limit);
                mma::cp_async_zfill<16, Cache::cg>(
                    smem.input.q + row * kDim + mma::swizzle_xor_64(row, col),
                    input.position >= 0 ? q.q + input.index * kDim + col : q.q,
                    input.position >= 0 ? 16 : 0);
            }
        }
        // BF16 K and FP16 V occupy separate buffers. Prefetch the first K
        // with Q, then overlap each V with QK and the next K with PV.
        auto record = rows.record(begin * K);
        if constexpr (PackedFp8)
            stage_fp8<Swa, K, Threads>(smem, rows, begin * K, first_visible, high);
        else if constexpr (RawFp8)
            stage_fp8<Swa, K, Threads, true>(smem.raw[0], rows, begin * K, first_visible, high);
        else
            stage_16bit<Swa, false, K, Threads>(smem.k, rows, begin * K, first_visible, high,
                                                record);
        if constexpr (RegisterQ)
            stage_projection<K, Threads, QRows>(smem.input.projection[0], q.projection, low,
                                                begin * K, staged_projection);
        mma::cp_commit();

        const int matrix_row   = (lane & 7) + (lane & 8);
        const int matrix_col   = (lane >> 4) * 8;
        const int first_row    = min(warp, Warps - 1) * 16;
        const bool has_queries = warp < Warps && (Ragged || warp * 16 < live * kGroup);
        const auto load_q      = [&](int d, unsigned* a) {
            if constexpr (!RegisterQ) {
                const auto* qrow = smem.input.q + (first_row + matrix_row) * kDim;
                mma::ldmatrix_x4(
                    a[0], a[1], a[2], a[3],
                    mma::smem_addr(qrow + mma::swizzle_xor_64(matrix_row, d + matrix_col)));
            }
        };
        unsigned q_frag[kDim / 16][4];
        if constexpr (RegisterQ) {
#pragma unroll
            for (int d = 0; d < kDim / 16; ++d) {
                const int column = d * 16 + pair;
                const auto* q0   = q.q + qr0.index * kDim;
                const auto* q1   = q.q + qr1.index * kDim;
                q_frag[d][0]     = qr0.position >= 0 ? load_pair(q0, column) : 0;
                q_frag[d][1]     = qr1.position >= 0 ? load_pair(q1, column) : 0;
                q_frag[d][2]     = qr0.position >= 0 ? load_pair(q0, column + 8) : 0;
                q_frag[d][3]     = qr1.position >= 0 ? load_pair(q1, column + 8) : 0;
            }
        }
        mma::cp_wait<0>();
        __syncthreads();

        unsigned rel0 = 0, rel1 = 0, rel2 = 0, rel3 = 0;
        if (can_shear && range.last * std::int64_t(K) > std::int64_t(warp_low) - kExtent + 1) {
            if (qr0.position >= 0) {
                rel0 = load_pair(q.r + qr0.index * kRelDim, pair);
                rel2 = load_pair(q.r + qr0.index * kRelDim, pair + 8);
            }
            if (qr1.position >= 0) {
                rel1 = load_pair(q.r + qr1.index * kRelDim, pair);
                rel3 = load_pair(q.r + qr1.index * kRelDim, pair + 8);
            }
        }

        int stage = 0;
        for (int tile = begin; tile < range.last;) {
            const int key       = tile * K;
            const int next_tile = tile + 1;
            if constexpr (PackedFp8) {
                unpack_fp8<K, Threads>(smem.k, smem.v, smem);
                __syncthreads();
                if (next_tile < range.last) {
                    const int next = next_tile * K;
                    stage_fp8<Swa, K, Threads>(smem, rows, next, first_visible, high);
                    if constexpr (RegisterQ)
                        stage_projection<K, Threads, QRows>(smem.input.projection[stage ^ 1],
                                                            q.projection, low, next,
                                                            staged_projection);
                    mma::cp_commit();
                }
            } else if constexpr (RawFp8) {
                if (next_tile < range.last) {
                    const int next = next_tile * K;
                    stage_fp8<Swa, K, Threads, true>(smem.raw[stage ^ 1], rows, next, first_visible,
                                                     high);
                    stage_projection<K, Threads, QRows>(smem.input.projection[stage ^ 1],
                                                        q.projection, low, next, staged_projection);
                    mma::cp_commit();
                }
            } else if constexpr (!Fp8) {
                stage_16bit<Swa, true, K, Threads>(smem.v, rows, key, first_visible, high, record);
                mma::cp_commit();
            }
            const bool computes = has_queries && key <= warp_high &&
                                  (!Swa || key + (K - 1) >= warp_low - kWindow + 1);
            unsigned p[K / 16][4];
            bool fold_values = false;
            if (computes) {
                if constexpr (RawFp8) {
                    bool safe_scale = true;
#pragma unroll
                    for (int i = lane; i < K; i += 32)
                        safe_scale &= i < first_visible - key || i > high - key ||
                                      smem.raw[stage].scales[1][i] >= 32;
                    fold_values = __all_sync(kWarp, safe_scale);
                }
                float scores[ScoreTiles][4]{};
                unsigned key_frag[2][ScoreTiles][2];
                const int kr      = (lane & 7) + (lane >> 4) * 8;
                const auto load_k = [&](int d, auto& fragments) {
                    if constexpr (RawFp8) {
                        const auto& raw = smem.raw[stage];
#pragma unroll
                        for (int n = 0; n < ScoreTiles; ++n) {
                            const int row = n * 8 + quad;
                            const auto* source =
                                raw.codes[0] + row * kDim + mma::swizzle_xor_128(row, d + pair);
                            const auto scale = raw.scales[0][row];
                            fragments[n][0]  = mma::fp8_dequant_bf16x2(
                                mma::load_vec<unsigned short>(source), scale);
                            fragments[n][1] = mma::fp8_dequant_bf16x2(
                                mma::load_vec<unsigned short>(source + 8), scale);
                        }
                    } else {
#pragma unroll
                        for (int n = 0; n < ScoreTiles; n += 2)
                            mma::ldmatrix_x4(
                                fragments[n][0], fragments[n][1], fragments[n + 1][0],
                                fragments[n + 1][1],
                                mma::smem_addr(smem.k + (n * 8 + kr) * kDim +
                                               mma::swizzle_xor_64(kr, d + (lane & 8))));
                    }
                };
                load_k(0, key_frag[0]);
#pragma unroll
                for (int d = 0; d < kDim / 16; ++d) {
                    const int current = d & 1;
                    unsigned a[4];
                    if constexpr (RegisterQ) {
#pragma unroll
                        for (int i = 0; i < 4; ++i) a[i] = q_frag[d][i];
                    } else {
                        load_q(d * 16, a);
                    }
                    if (d + 1 < kDim / 16) load_k((d + 1) * 16, key_frag[current ^ 1]);
#pragma unroll
                    for (int n = 0; n < ScoreTiles; ++n) {
                        const auto& b = key_frag[current][n];
                        mma::mma_bf16(scores[n][0], scores[n][1], scores[n][2], scores[n][3], a[0],
                                      a[1], a[2], a[3], b[0], b[1]);
                    }
                }

                const bool near = warp_low - (key + (K - 1)) < kExtent;
                if (near && can_shear) {
                    // Columns of this MMA are distances descending from low - key + 16.
                    // Gathering column k + 16 - (p - low) shears each query row in registers.
                    auto& bias       = smem.input.bias[warp];
                    const int origin = warp_low - key + 16;
#pragma unroll
                    for (int n = 0; n < ScoreTiles + 2; ++n) {
                        const int distance = origin - n * 8 - quad;
                        unsigned b0 = 0, b1 = 0;
                        if constexpr (RegisterQ) {
                            const auto* pr =
                                smem.input.projection[stage][projection_shift + n * 8 + quad];
                            b0 = load_pair(pr, pair);
                            b1 = load_pair(pr, pair + 8);
                        } else if (unsigned(distance) < kExtent) {
                            const auto* pr = q.projection + distance * kRelDim;
                            b0             = load_pair(pr, pair);
                            b1             = load_pair(pr, pair + 8);
                        }
                        float b00 = 0, b01 = 0, b10 = 0, b11 = 0;
                        mma::mma_bf16(b00, b01, b10, b11, rel0, rel1, rel2, rel3, b0, b1);
                        const int col = bias_col<K, RegisterQ>(quad, n * 8 + pair);
                        mma::store_vec(bias[quad] + col, make_float2(b00, b01));
                        mma::store_vec(bias[quad + 8] + col, make_float2(b10, b11));
                    }
                    __syncwarp();
                }

                if (!Swa && !near) {
                    // On Full layers every query sees these keys, and the relative
                    // projection has zero support here.
#pragma unroll
                    for (int n = 0; n < ScoreTiles; ++n)
#pragma unroll
                        for (int elt = 0; elt < 4; ++elt) scores[n][elt] *= scale2;
                } else {
                    const int shift0    = 16 - max(0, qr0.position - warp_low);
                    const int shift1    = 16 - max(0, qr1.position - warp_low);
                    const bool interior = key + (K - 1) <= low && (!Swa || high - key < kWindow);
#pragma unroll
                    for (int n = 0; n < ScoreTiles; ++n) {
#pragma unroll
                        for (int elt = 0; elt < 4; ++elt) {
                            const auto row = elt < 2 ? qr0 : qr1;
                            const int col  = n * 8 + pair + (elt & 1);
                            const int dist = row.position - (key + col);
                            if ((can_shear && interior) ||
                                (dist >= 0 && (!Swa || dist < kWindow))) {
                                const float bias =
                                    can_shear
                                        ? smem.input
                                              .bias[warp][quad + (elt < 2 ? 0 : 8)]
                                                   [bias_col<K, RegisterQ>(
                                                       quad, col + (elt < 2 ? shift0 : shift1))]
                                        : relative_dot(q.r + row.index * kRelDim, q.projection,
                                                       dist);
                                scores[n][elt] = fmaf(scores[n][elt], scale2, bias * kLog2E);
                            } else {
                                scores[n][elt] = -CUDART_INF_F;
                            }
                        }
                    }
                }

                float tile0 = -CUDART_INF_F, tile1 = -CUDART_INF_F;
#pragma unroll
                for (int n = 0; n < ScoreTiles; ++n) {
                    tile0 = fmaxf(tile0, fmaxf(scores[n][0], scores[n][1]));
                    tile1 = fmaxf(tile1, fmaxf(scores[n][2], scores[n][3]));
                }
                tile0          = quad_max(tile0);
                tile1          = quad_max(tile1);
                float new_ref0 = fmaxf(ref0, tile0), new_ref1 = fmaxf(ref1, tile1);
                if constexpr (!Fp8) {
                    // Use the same exponent reference for the numerator and denominator.
                    // A weak tile may boost P by lowering the reference. Limit an
                    // upward rescale to mass below 2^81, leaving room for finite
                    // FP16 values (< 2^16) in the FP32 numerator.
                    const auto reference = [](float maximum, float tile_max, float mass) {
                        if (tile_max < maximum - 10.0f && tile_max != -CUDART_INF_F) {
                            const int exponent = int(__float_as_uint(mass) >> 23) - 127;
                            const float shift  = float(min(40, max(0, 80 - exponent)));
                            return fmaxf(tile_max, maximum - shift);
                        }
                        return maximum;
                    };
                    new_ref0 = reference(new_ref0, tile0, sum0);
                    new_ref1 = reference(new_ref1, tile1, sum1);
                }
                const float safe0  = new_ref0 == -CUDART_INF_F ? 0.0f : new_ref0;
                const float safe1  = new_ref1 == -CUDART_INF_F ? 0.0f : new_ref1;
                const float alpha0 = common::exp2_approx(ref0 - safe0);
                const float alpha1 = common::exp2_approx(ref1 - safe1);
                ref0               = new_ref0;
                ref1               = new_ref1;

                if constexpr (RawFp8) {
                    if (fold_values) {
                        // Scales >= 2^-95 and nonzero weights >= 2^-31 keep
                        // the folded operand in the normal BF16 range.
                        bool safe = true;
#pragma unroll
                        for (int n = 0; n < ScoreTiles; ++n) {
#pragma unroll
                            for (int elt = 0; elt < 4; ++elt) {
                                const float score   = scores[n][elt];
                                const float maximum = elt < 2 ? safe0 : safe1;
                                safe &= score == -CUDART_INF_F || score >= maximum - 31.0f;
                            }
                        }
                        fold_values = __all_sync(kWarp, safe);
                    }
                }

                float norm0 = 0, norm1 = 0;
#pragma unroll
                for (int n = 0; n < ScoreTiles; ++n) {
                    const float a = common::exp2_approx(scores[n][0] - safe0);
                    const float b = common::exp2_approx(scores[n][1] - safe0);
                    const float c = common::exp2_approx(scores[n][2] - safe1);
                    const float d = common::exp2_approx(scores[n][3] - safe1);
                    norm0 += a + b;
                    norm1 += c + d;
                    if constexpr (Fp8) {
                        float scale0 = 1.0f, scale1 = 1.0f;
                        if constexpr (RawFp8) {
                            if (fold_values) {
                                const auto* scales = smem.raw[stage].scales[1] + n * 8 + pair;
                                scale0             = common::ue8m0_to_float(scales[0]);
                                scale1             = common::ue8m0_to_float(scales[1]);
                            }
                        }
                        p[n / 2][(n % 2) * 2]     = mma::pack_bf16x2(a * scale0, b * scale1);
                        p[n / 2][(n % 2) * 2 + 1] = mma::pack_bf16x2(c * scale0, d * scale1);
                    } else {
                        p[n / 2][(n % 2) * 2]     = mma::pack_f16x2(a, b);
                        p[n / 2][(n % 2) * 2 + 1] = mma::pack_f16x2(c, d);
                    }
                }
                sum0 = fmaf(sum0, alpha0, quad_sum(norm0));
                sum1 = fmaf(sum1, alpha1, quad_sum(norm1));
#pragma unroll
                for (int n = 0; n < OutputTiles; ++n) {
                    out[n][0] *= alpha0;
                    out[n][1] *= alpha0;
                    out[n][2] *= alpha1;
                    out[n][3] *= alpha1;
                }
            }

            if constexpr (!Fp8) {
                mma::cp_wait<0>();
                __syncthreads();
                if (next_tile < range.last) {
                    const int next = next_tile * K;
                    record         = rows.record(next);
                    stage_16bit<Swa, false, K, Threads>(smem.k, rows, next, first_visible, high,
                                                        record);
                    if constexpr (RegisterQ)
                        stage_projection<K, Threads, QRows>(smem.input.projection[0], q.projection,
                                                            low, next, staged_projection);
                    mma::cp_commit();
                }
            }
            if (computes) {
                constexpr int Pairs = OutputTiles / 2;
                constexpr int Loads = (K / 16) * Pairs;
                const int vr        = lane & 15;
                const auto load_v   = [&](int step, unsigned* frag) {
                    const int k = step / Pairs, n = step % Pairs * 2;
                    if constexpr (RawFp8) {
                        // Transpose packed V pairs and contract the even and odd
                        // feature columns independently. The epilogue interleaves them.
                        const auto& raw = smem.raw[stage];
                        unsigned lo, hi;
                        const int col = mma::swizzle_xor_128(vr, n * 8);
                        mma::ldmatrix_x2_t(
                            lo, hi, mma::smem_addr(raw.codes[1] + (k * 16 + vr) * kDim + col));
                        const auto* scales = raw.scales[1] + k * 16 + pair;
                        const auto even    = [](unsigned x) { return __byte_perm(x, 0, 0x4420); };
                        const auto odd     = [](unsigned x) { return __byte_perm(x, 0, 0x4431); };
                        frag[0] = widen_value_pair(even(lo), scales[0], scales[1], fold_values);
                        frag[1] = widen_value_pair(even(hi), scales[8], scales[9], fold_values);
                        frag[2] = widen_value_pair(odd(lo), scales[0], scales[1], fold_values);
                        frag[3] = widen_value_pair(odd(hi), scales[8], scales[9], fold_values);
                    } else {
                        const int vc = n * 8 + (lane >> 4) * 8;
                        mma::ldmatrix_x4_t(frag[0], frag[1], frag[2], frag[3],
                                             mma::smem_addr(smem.v + (k * 16 + vr) * kDim +
                                                            mma::swizzle_xor_64(vr, vc)));
                    }
                };
                unsigned value_frag[2][4];
                load_v(0, value_frag[0]);
#pragma unroll
                for (int step = 0; step < Loads; ++step) {
                    const int k = step / Pairs, n = step % Pairs * 2;
                    const int current = step & 1;
                    if (step + 1 < Loads) load_v(step + 1, value_frag[current ^ 1]);
                    const auto multiply = [&](int frag, unsigned x, unsigned y) {
                        auto& o = out[frag];
                        if constexpr (Fp8)
                            mma::mma_bf16(o[0], o[1], o[2], o[3], p[k][0], p[k][1], p[k][2],
                                          p[k][3], x, y);
                        else
                            mma::mma_f16(o[0], o[1], o[2], o[3], p[k][0], p[k][1], p[k][2], p[k][3],
                                         x, y);
                    };
                    const auto& v = value_frag[current];
                    multiply(n, v[0], v[1]);
                    multiply(n + 1, v[2], v[3]);
                }
            }

            mma::cp_wait<0>();
            __syncthreads();
            if constexpr (Fp8 && RegisterQ) stage ^= 1;
            tile = next_tile;
        }
    }

    auto write = [&](bool owns, QueryRow row, float sum, float reference, int half) {
        if (!owns) return;
        const bool visible       = row.position >= 0 && sum > 0;
        const float lse          = visible ? reference + __log2f(sum) : -CUDART_INF_F;
        const float inverse      = visible ? __frcp_rn(sum) : 0.0f;
        const std::size_t record = Direct ? 0 : split * total_rows + row.index;
        if constexpr (RawFp8) {
            const float factor = Direct && visible ? sink_factor(q, row.index, lse) : 0.0f;
            if constexpr (!Direct) {
                if (pair == 0) partials.lse[record] = lse;
                if (!visible) return;
            }
#pragma unroll
            for (int n = 0; n < OutputTiles; n += 2) {
                const int dim = n * 8 + 2 * pair;
                if constexpr (Direct) {
                    ushort4 val{};
                    if (visible)
                        val = {
                            gated(q, row.index, dim, out[n][half] * inverse, factor),
                            gated(q, row.index, dim + 1, out[n + 1][half] * inverse, factor),
                            gated(q, row.index, dim + 2, out[n][half + 1] * inverse, factor),
                            gated(q, row.index, dim + 3, out[n + 1][half + 1] * inverse, factor)};
                    mma::store_vec(q.out + row.index * kDim + dim, val);
                } else {
                    mma::store_vec(partials.values + record * kDim + dim,
                                   make_float4(out[n][half] * inverse, out[n + 1][half] * inverse,
                                               out[n][half + 1] * inverse,
                                               out[n + 1][half + 1] * inverse));
                }
            }
        } else if constexpr (Direct) {
            const float factor = visible ? sink_factor(q, row.index, lse) : 0.0f;
#pragma unroll
            for (int n = 0; n < OutputTiles; ++n) {
                const int col = n * 8 + pair;
                const unsigned short lo =
                    visible ? gated(q, row.index, col, out[n][half] * inverse, factor) : 0;
                const unsigned short hi =
                    visible ? gated(q, row.index, col + 1, out[n][half + 1] * inverse, factor) : 0;
                mma::store_vec(q.out + row.index * kDim + col, unsigned(lo) | unsigned(hi) << 16);
            }
        } else {
            if (pair == 0) partials.lse[record] = lse;
            if (!visible) return;
#pragma unroll
            for (int n = 0; n < OutputTiles; ++n)
                mma::store_vec(partials.values + record * kDim + n * 8 + pair,
                               make_float2(out[n][half] * inverse, out[n][half + 1] * inverse));
        }
    };
    write(owns0, qr0, sum0, ref0, 0);
    write(owns1, qr1, sum1, ref1, 2);
}

} // namespace ninfer::ops::relative
