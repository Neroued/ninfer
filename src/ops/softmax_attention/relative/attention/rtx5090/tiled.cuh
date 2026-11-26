#pragma once

#include "ops/softmax_attention/relative/attention/rtx5090/device.cuh"

#include <type_traits>

namespace ninfer::ops::relative {
namespace mma = common;

template <int Warps, int K, bool RegisterQ, int QRows, int Stages, bool RegisterBias>
struct QueryStorage {
    std::uint16_t q[Warps * 16 * kDim];
    float bias[Warps][16][K + 16];
};

template <int Warps, int K, int QRows, int Stages>
struct QueryStorage<Warps, K, true, QRows, Stages, false> {
    float bias[Warps][16][K + 16];
    alignas(16) std::uint16_t projection[Stages][K + QRows][kRelDim];
};

template <int Warps, int K, int QRows, int Stages>
struct QueryStorage<Warps, K, true, QRows, Stages, true> {
    alignas(16) std::uint16_t projection[Stages][K + QRows][kRelDim];
};

template <int K>
struct alignas(16) EncodedKV {
    unsigned char codes[2][K * kDim];
    unsigned char scales[2][K];
};

template <int K>
struct ExpandedKV {
    alignas(16) std::uint16_t k[K * kDim];
    alignas(16) std::uint16_t v[K * kDim];
};

template <int K>
struct RawKV {
    EncodedKV<K> raw[2];
};

template <int K>
struct PipelinedKV {
    ExpandedKV<K> kv[2];
};

template <int K, int QRows>
struct ProjectionStorage {
    alignas(16) std::uint16_t projection[2][K + QRows][kRelDim];
};

struct EmptyStorage {};

template <int Warps, bool RegisterQ, bool Fp8, int K = kKeyTile, int QRows = 16,
          bool SharedHeads = false, bool RegisterBias = false>
struct alignas(16) TileStorage
    : std::conditional_t<Fp8, RawKV<K>,
                         std::conditional_t<SharedHeads, PipelinedKV<K>, ExpandedKV<K>>> {
    std::conditional_t<
        Fp8, std::conditional_t<RegisterQ, ProjectionStorage<K, QRows>, EmptyStorage>,
        QueryStorage<Warps, K, RegisterQ, QRows, SharedHeads ? 2 : 1, SharedHeads || RegisterBias>>
        input;
};

template <bool Swa, bool Value, int K, int Threads, bool Contiguous = false>
__device__ __forceinline__ void stage_16bit(std::uint16_t* dest, const CacheRows<Swa, false>& cache,
                                            int key, int lo, int hi, std::int64_t record) {
    const int count     = min(K - 1, hi - key) + 1;
    const bool chunk    = cache.is_chunk(key);
    const bool boundary = !chunk && cache.is_chunk(key + (count - 1));
    const int stride    = chunk ? kKvHeads * (2 * kDim) : 2 * kDim;
    const auto* base    = cache.codes(Value, key + (boundary ? count - 1 : 0), record);
    const int lower     = Swa ? max(0, lo - key) : 0;
    if constexpr (Contiguous) {
        // Adjacent query tiles can reuse complete cache tiles in L1.
        if (count == K && lower == 0 && !boundary) {
#pragma unroll
            for (int i = threadIdx.x; i < K * (kDim / 8); i += Threads) {
                const int row = i / (kDim / 8), d = i % (kDim / 8) * 8;
                mma::cp_async<16, Cache::ca>(dest + row * kDim + mma::swizzle_xor_64(row, d),
                                             base + row * stride + d * 2);
            }
            return;
        }
    }
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

template <bool Swa, int K, int Threads>
__device__ __forceinline__ void stage_fp8(EncodedKV<K>& dest, const CacheRows<Swa, true>& cache,
                                          int key, int lo, int hi) {
    const int count     = min(K - 1, hi - key) + 1;
    const auto record   = cache.record(key);
    const bool chunk    = cache.is_chunk(key);
    const bool boundary = !chunk && cache.is_chunk(key + (count - 1));
    const int stride    = chunk ? kKvHeads * kDim : kDim;
    const int lower     = Swa ? max(0, lo - key) : 0;
#pragma unroll 1
    for (int i = threadIdx.x; i < 2 * K * (kDim / 16); i += Threads) {
        constexpr int Vectors = K * (kDim / 16);
        const int plane = i / Vectors, index = i % Vectors;
        const int row = index / (kDim / 16), col = index % (kDim / 16) * 16;
        const bool live  = unsigned(row - lower) < unsigned(count - lower);
        const auto* base = cache.codes(plane, key + (boundary ? count - 1 : 0), record);
        const auto* src  = base + row * stride;
        if constexpr (Swa)
            if (boundary) src = live ? cache.codes(plane, key + row, record) : base;
        const int target = mma::swizzle_xor_128(row, col);
        mma::cp_async_zfill<16, Cache::cg>(dest.codes[plane] + row * kDim + target, src + col,
                                           live ? 16 : 0);
    }
    if constexpr (Swa) {
        const bool vector_scales =
            !chunk && !boundary && lower == 0 &&
            ((reinterpret_cast<std::uintptr_t>(cache.scale(0, key, record)) |
              reinterpret_cast<std::uintptr_t>(cache.scale(1, key, record))) &
             15u) == 0;
        if (!vector_scales) {
#pragma unroll
            for (int i = threadIdx.x; i < 2 * K; i += Threads) {
                const int plane = i / K, row = i % K;
                const bool live         = unsigned(row - lower) < unsigned(count - lower);
                dest.scales[plane][row] = live ? *cache.scale(plane, key + row, record) : 0;
            }
            return;
        }
    }
    if (threadIdx.x < 2 * K / 16) {
        const int byte  = threadIdx.x * 16;
        const int plane = byte / K, index = byte % K;
        const auto* scales = cache.scale(plane, key, record);
        mma::cp_async_zfill<16, Cache::cg>(dest.scales[plane] + index, scales + index,
                                           min(16, max(0, count - index)));
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

// The accumulator columns for each query row live in a four-lane quad. Select
// the two source columns from three consecutive MMA fragments, then shear them
// into the QK fragment with quad shuffles.
__device__ __forceinline__ float2 shear_bias(float2 a, float2 b, float2 c, int shift, int pair) {
    const int offset = shift & 7;
    const int block  = shift >> 3;
    const auto pick  = [&](float lo, float mid, float hi, int column) {
        const int n = block + int(column < offset);
        return n == 0 ? lo : n == 1 ? mid : hi;
    };
    const float lo = pick(a.x, b.x, c.x, pair);
    const float hi = pick(a.y, b.y, c.y, pair + 1);
    const int src  = (pair + offset) / 2;
    const bool odd = offset & 1;
    return {__shfl_sync(kWarp, odd ? hi : lo, src & 3, 4),
            __shfl_sync(kWarp, odd ? lo : hi, (src + odd) & 3, 4)};
}

__device__ __forceinline__ unsigned quantize_four(float a, float b, float c, float d,
                                                  float inverse) {
    return unsigned(mma::fp8_quant_code2(a, b, inverse)) |
           (unsigned(mma::fp8_quant_code2(c, d, inverse)) << 16);
}

struct MxQuery {
    unsigned words[4];
    unsigned scale;
};

__device__ __forceinline__ MxQuery quantize_query(Query q, QueryRow a, QueryRow b, int col,
                                                  int lane) {
    float values[2][8]{};
    unsigned exponents[2];
    const QueryRow rows[]{a, b};
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const auto row = rows[r];
        float local    = 0;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const auto raw          = row.position >= 0
                                          ? mma::load_vec<uint2>(q.q + row.index * kDim + col + half * 16)
                                          : make_uint2(0, 0);
            values[r][half * 4]     = __uint_as_float(raw.x << 16);
            values[r][half * 4 + 1] = __uint_as_float(raw.x & 0xffff0000u);
            values[r][half * 4 + 2] = __uint_as_float(raw.y << 16);
            values[r][half * 4 + 3] = __uint_as_float(raw.y & 0xffff0000u);
        }
#pragma unroll
        for (float x : values[r]) local = fmaxf(local, fabsf(x));
        const auto quantization = mma::fp8_ue8m0_scale(quad_max(local));
        exponents[r]            = quantization.scale;
#pragma unroll
        for (float& x : values[r]) x *= quantization.inverse_scale;
    }
    MxQuery result{};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const auto* v   = values[i & 1] + (i >> 1) * 4;
        result.words[i] = quantize_four(v[0], v[1], v[2], v[3], 1.0f);
    }
    result.scale = lane & 1 ? exponents[1] : exponents[0];
    return result;
}

__device__ __forceinline__ void load_transposed_fp8(unsigned (&regs)[4], unsigned address) {
    asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3}, [%4];"
                 : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
                 : "r"(address));
}

__device__ __forceinline__ unsigned join_probability_pairs(unsigned a, unsigned b, int lane) {
    // From two m16n8 output fragments (2 adjacent keys per lane), form a
    // m16n8k32 A register (4 adjacent keys per lane).
    const int src       = (lane & 1) * 2;
    const unsigned both = a | (b << 16);
    const unsigned x    = __shfl_sync(kWarp, both, src, 4);
    const unsigned y    = __shfl_sync(kWarp, both, src + 1, 4);
    return __byte_perm(x, y, lane & 2 ? 0x7632 : 0x5410);
}

struct ResidualProbabilities {
    unsigned hi[4], lo[4];
    unsigned scale_hi, scale_lo;
};

__device__ __forceinline__ ResidualProbabilities
quantize_probabilities(float (*p)[4], const unsigned char* value_scales, int lane) {
    const int pair = (lane & 3) * 2;
    float max0 = 0.0f, max1 = 0.0f;
#pragma unroll
    for (int n = 0; n < 4; ++n) {
        const auto* exponents = value_scales + n * 8 + pair;
        const float s0        = mma::ue8m0_to_float(exponents[0]);
        const float s1        = mma::ue8m0_to_float(exponents[1]);
        p[n][0] *= s0;
        p[n][1] *= s1;
        p[n][2] *= s0;
        p[n][3] *= s1;
        max0 = fmaxf(max0, fmaxf(p[n][0], p[n][1]));
        max1 = fmaxf(max1, fmaxf(p[n][2], p[n][3]));
    }
    const auto hi0   = mma::fp8_ue8m0_scale(quad_max(max0));
    const auto hi1   = mma::fp8_ue8m0_scale(quad_max(max1));
    const float mul0 = mma::ue8m0_to_float(hi0.scale);
    const float mul1 = mma::ue8m0_to_float(hi1.scale);
    unsigned hi_codes[4][2];
    max0 = max1 = 0.0f;
#pragma unroll
    for (int n = 0; n < 4; ++n) {
        hi_codes[n][0]   = mma::fp8_quant_code2(p[n][0], p[n][1], hi0.inverse_scale);
        hi_codes[n][1]   = mma::fp8_quant_code2(p[n][2], p[n][3], hi1.inverse_scale);
        const unsigned x = mma::fp8_e4m3x2_to_bf16x2_bits(hi_codes[n][0]);
        const unsigned y = mma::fp8_e4m3x2_to_bf16x2_bits(hi_codes[n][1]);
        p[n][0]          = fmaf(-__uint_as_float(x << 16), mul0, p[n][0]);
        p[n][1]          = fmaf(-__uint_as_float(x & 0xffff0000u), mul0, p[n][1]);
        p[n][2]          = fmaf(-__uint_as_float(y << 16), mul1, p[n][2]);
        p[n][3]          = fmaf(-__uint_as_float(y & 0xffff0000u), mul1, p[n][3]);
        max0             = fmaxf(max0, fmaxf(fabsf(p[n][0]), fabsf(p[n][1])));
        max1             = fmaxf(max1, fmaxf(fabsf(p[n][2]), fabsf(p[n][3])));
    }
    // The zero-maximum scale is one, so an exactly represented block supplies
    // zero residual codes and contributes zero from the second contraction.
    const auto lo0 = mma::fp8_ue8m0_scale(quad_max(max0));
    const auto lo1 = mma::fp8_ue8m0_scale(quad_max(max1));
    ResidualProbabilities out;
    out.scale_hi = lane & 1 ? hi1.scale : hi0.scale;
    out.scale_lo = lane & 1 ? lo1.scale : lo0.scale;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int n = i & 2, h = i & 1;
        const float inverse = h ? lo1.inverse_scale : lo0.inverse_scale;
        out.hi[i]           = join_probability_pairs(hi_codes[n][h], hi_codes[n + 1][h], lane);
        const auto a        = mma::fp8_quant_code2(p[n][2 * h], p[n][2 * h + 1], inverse);
        const auto b        = mma::fp8_quant_code2(p[n + 1][2 * h], p[n + 1][2 * h + 1], inverse);
        out.lo[i]           = join_probability_pairs(a, b, lane);
    }
    return out;
}

template <bool Fp8, bool Swa, bool Ragged, int Warps, int Threads, bool Direct, int K = kKeyTile,
          int QRows = 16, int HeadGroup = kGroup, bool RegisterBias = false,
          bool Contiguous = false>
__device__ __forceinline__ void tiled_attention(
    Query q, CacheView cache, int sequence, std::int64_t sequence_first, std::int64_t first,
    int live, int width, int head_index, int split, int splits, Partials partials,
    std::size_t total_rows,
    TileStorage<Warps, Ragged || Swa, Fp8, K, QRows, Ragged && HeadGroup != 1, RegisterBias>& smem,
    bool signal = false) {
    static_assert(!Fp8 || K % 32 == 0);
    static_assert(!Contiguous || (Ragged && RegisterBias));
    constexpr bool RegisterQ        = Ragged || Swa;
    constexpr bool SharedHeads      = Ragged && HeadGroup != 1;
    constexpr bool RegisterShear    = SharedHeads || Fp8 || RegisterBias;
    constexpr bool IndependentHeads = Ragged && !SharedHeads;
    constexpr int ScoreTiles        = K / 8;
    constexpr int OutputTiles       = kDim / 8;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    const int quad = lane >> 2, pair = (lane & 3) * 2;
    const int kvhead     = HeadGroup == 1 ? head_index / kGroup : head_index;
    const int first_head = head_index * HeadGroup;
    const int limit      = position_limit<Swa>(cache);
    const auto range     = [&] {
        if constexpr (Contiguous) {
            const int lo = q.positions[first];
            const int hi = lo + (live - 1);
            return KeyRange{Swa ? max(0, lo - (kWindow - 1)) / K : 0, hi / K + 1, lo, hi};
        } else {
            return key_range<Swa, K, QRows>(q.positions + first, live, split, splits, limit);
        }
    }();
    const int row0 = warp * 16 + quad, row1 = row0 + 8;
    const auto row_at = [&](int row) {
        if constexpr (Contiguous) {
            const int token = row % QRows, head = first_head + row / QRows;
            return QueryRow{(first + token) * kHeads + head, token < live ? range.low + token : -1};
        } else {
            return query_row<Ragged, QRows, HeadGroup>(q, first, live, first_head, row, limit);
        }
    };
    const auto qr0   = row_at(row0);
    const auto qr1   = row_at(row1);
    const int local0 = Ragged ? row0 % QRows : row0 / kGroup;
    const int local1 = Ragged ? row1 % QRows : row1 / kGroup;
    const bool owns0 = local0 < width && (IndependentHeads ? warp < Warps
                                          : Ragged         ? row0 < QRows * HeadGroup
                                                           : row0 < kGroup * width);
    const bool owns1 = local1 < width && (IndependentHeads ? warp < Warps
                                          : Ragged         ? row1 < QRows * HeadGroup
                                                           : row1 < kGroup * width);
    // Only a live interval may read table rows or per-record offsets.
    const int high = range.high, low = range.low;
    const int begin         = range.first;
    const int first_visible = Swa ? max(0, low - (kWindow - 1)) : 0;
    const int table         = begin < range.last ? cache_row<Swa>(cache, sequence) : -1;
    float sum0 = 0, sum1 = 0;
    float ref0 = -CUDART_INF_F, ref1 = -CUDART_INF_F;
    float out[OutputTiles][4]{};
    if (table >= 0 && table < cache.table_count) {
        const bool staged_projection = Contiguous || high - low < QRows;
        int warp_low = low, warp_high = high;
        if constexpr (Contiguous) {
            const int warp_first = (warp * 16) % QRows;
            warp_low             = low + min(warp_first, live - 1);
            warp_high = warp_first < live ? warp_low + min(15, live - 1 - warp_first) : -1;
        } else if constexpr (Ragged && QRows > 16) {
            const int t = (warp * 16) % QRows + lane;
            const int p = lane < 16 && t < live ? q.positions[first + t] : -1;
            warp_low    = __reduce_min_sync(kWarp, p >= 0 && p <= limit ? p : INT_MAX);
            warp_high   = __reduce_max_sync(kWarp, p <= limit ? p : -1);
        }
        const bool warp_can_shear  = warp_high >= 0 && warp_high - warp_low < 16;
        const int projection_shift = [&] {
            if constexpr (Contiguous) return QRows - 16 - (warp * 16) % QRows;
            return IndependentHeads || (staged_projection && warp_can_shear)
                       ? QRows - 16 - (warp_low - low)
                       : -1;
        }();
        const bool can_shear =
            Contiguous ||
            (warp_can_shear && (RegisterShear || (staged_projection && projection_shift >= 0)));
        const float scale2 = q.scale * kLog2E;
        int chunk_start    = 0;
        if constexpr (Swa)
            if (cache.chunk.k) chunk_start = q.positions[sequence_first];
        const CacheRows<Swa, Fp8> rows(cache, table, kvhead, sequence_first, chunk_start,
                                       !cache.chunk.k ||
                                           max(begin * K, first_visible) < chunk_start);

        if constexpr (!RegisterQ && !Fp8) {
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
        if constexpr (Fp8)
            stage_fp8<Swa, K, Threads>(smem.raw[0], rows, begin * K, first_visible, high);
        else if constexpr (SharedHeads) {
            stage_16bit<Swa, false, K, Threads>(smem.kv[0].k, rows, begin * K, first_visible, high,
                                                record);
            stage_16bit<Swa, true, K, Threads>(smem.kv[0].v, rows, begin * K, first_visible, high,
                                               record);
        } else
            stage_16bit<Swa, false, K, Threads, Contiguous>(smem.k, rows, begin * K, first_visible,
                                                            high, record);
        if constexpr (RegisterQ)
            stage_projection<K, Threads, QRows>(smem.input.projection[0], q.projection, low,
                                                begin * K, staged_projection);
        mma::cp_commit();

        const int matrix_row   = (lane & 7) + (lane & 8);
        const int matrix_col   = (lane >> 4) * 8;
        const int first_row    = min(warp, Warps - 1) * 16;
        const bool has_queries = warp < Warps && (Ragged || warp * 16 < live * kGroup);
        const auto load_q      = [&](int d, unsigned* a) {
            if constexpr (!RegisterQ && !Fp8) {
                const auto* qrow = smem.input.q + (first_row + matrix_row) * kDim;
                mma::ldmatrix_x4(
                    a[0], a[1], a[2], a[3],
                    mma::smem_addr(qrow + mma::swizzle_xor_64(matrix_row, d + matrix_col)));
            }
        };
        unsigned q_frag[kDim / 16][4];
        MxQuery q_mx[kDim / 32];
        if constexpr (Fp8) {
#pragma unroll
            for (int d = 0; d < kDim / 32; ++d)
                q_mx[d] = quantize_query(q, qr0, qr1, d * 32 + (lane & 3) * 4, lane);
        } else if constexpr (RegisterQ) {
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
            if constexpr (Fp8) {
                if (next_tile < range.last) {
                    const int next = next_tile * K;
                    stage_fp8<Swa, K, Threads>(smem.raw[stage ^ 1], rows, next, first_visible,
                                               high);
                    if constexpr (RegisterQ)
                        stage_projection<K, Threads, QRows>(smem.input.projection[stage ^ 1],
                                                            q.projection, low, next,
                                                            staged_projection);
                    mma::cp_commit();
                }
            } else if constexpr (SharedHeads) {
                if (next_tile < range.last) {
                    const int next         = next_tile * K;
                    const auto next_record = rows.record(next);
                    stage_16bit<Swa, false, K, Threads>(smem.kv[stage ^ 1].k, rows, next,
                                                        first_visible, high, next_record);
                    stage_16bit<Swa, true, K, Threads>(smem.kv[stage ^ 1].v, rows, next,
                                                       first_visible, high, next_record);
                    stage_projection<K, Threads, QRows>(smem.input.projection[stage ^ 1],
                                                        q.projection, low, next, staged_projection);
                    mma::cp_commit();
                }
            } else if constexpr (!Fp8) {
                stage_16bit<Swa, true, K, Threads, Contiguous>(smem.v, rows, key, first_visible,
                                                               high, record);
                mma::cp_commit();
            }
            const bool computes = has_queries && (IndependentHeads || warp_high >= 0) &&
                                  key <= warp_high &&
                                  (!Swa || key + (K - 1) >= warp_low - kWindow + 1);
            unsigned p[K / 16][4];
            ResidualProbabilities p_mx[Fp8 ? K / 32 : 1];
            if (computes) {
                float scores[ScoreTiles][4]{};
                if constexpr (Fp8) {
                    const auto& raw = smem.raw[stage];
                    const int kr    = (lane & 7) + (lane >> 4) * 8;
#pragma unroll
                    for (int d = 0; d < kDim / 32; ++d) {
                        const auto& a = q_mx[d];
#pragma unroll
                        for (int n = 0; n < ScoreTiles; n += 2) {
                            unsigned b[4];
                            const int col = mma::swizzle_xor_128(kr, d * 32 + (lane & 8) * 2);
                            mma::ldmatrix_x4(
                                b[0], b[1], b[2], b[3],
                                mma::smem_addr(raw.codes[0] + (n * 8 + kr) * kDim + col));
#pragma unroll
                            for (int i = 0; i < 2; ++i) {
                                auto& c = scores[n + i];
                                mma::mma_fp8_e4m3_scaled(c[0], c[1], c[2], c[3], a.words[0],
                                                         a.words[1], a.words[2], a.words[3],
                                                         b[i * 2], b[i * 2 + 1], a.scale,
                                                         raw.scales[0][(n + i) * 8 + quad]);
                            }
                        }
                    }
                } else if constexpr (SharedHeads) {
                    const auto* keys = smem.kv[stage].k;
                    const int kr     = (lane & 7) + (lane >> 4) * 8;
#pragma unroll
                    for (int d = 0; d < kDim / 16; ++d) {
                        unsigned a[4];
                        if constexpr (RegisterQ) {
#pragma unroll
                            for (int i = 0; i < 4; ++i) a[i] = q_frag[d][i];
                        } else {
                            load_q(d * 16, a);
                        }
#pragma unroll
                        for (int n = 0; n < ScoreTiles; n += 2) {
                            unsigned b[4];
                            const auto addr =
                                mma::smem_addr(keys + (n * 8 + kr) * kDim +
                                               mma::swizzle_xor_64(kr, d * 16 + (lane & 8)));
                            mma::ldmatrix_x4(b[0], b[1], b[2], b[3], addr);
#pragma unroll
                            for (int i = 0; i < 2; ++i) {
                                auto& c = scores[n + i];
                                mma::mma_bf16(c[0], c[1], c[2], c[3], a[0], a[1], a[2], a[3],
                                              b[i * 2], b[i * 2 + 1]);
                            }
                        }
                    }
                } else {
                    unsigned key_frag[2][ScoreTiles][2];
                    const int kr      = (lane & 7) + (lane >> 4) * 8;
                    const auto load_k = [&](int d, auto& fragments) {
                        const auto* keys = [&] {
                            if constexpr (SharedHeads)
                                return smem.kv[stage].k;
                            else
                                return smem.k;
                        }();
#pragma unroll
                        for (int n = 0; n < ScoreTiles; n += 2)
                            mma::ldmatrix_x4(
                                fragments[n][0], fragments[n][1], fragments[n + 1][0],
                                fragments[n + 1][1],
                                mma::smem_addr(keys + (n * 8 + kr) * kDim +
                                               mma::swizzle_xor_64(kr, d + (lane & 8))));
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
                            mma::mma_bf16(scores[n][0], scores[n][1], scores[n][2], scores[n][3],
                                          a[0], a[1], a[2], a[3], b[0], b[1]);
                        }
                    }
                }

                const bool near = (Swa && Contiguous && kWindow <= kExtent) ||
                                  warp_low - (key + (K - 1)) < kExtent;
                if constexpr (RegisterShear) {
                    const int shift0       = 16 - max(0, qr0.position - warp_low);
                    const int shift1       = 16 - max(0, qr1.position - warp_low);
                    const int min_position = Contiguous ? warp_low : low;
                    const int max_position = Contiguous ? warp_high : high;
                    const bool interior =
                        key + (K - 1) <= min_position && (!Swa || max_position - key < kWindow);
                    const auto mask_and_add = [&](int n, float2 bias0, float2 bias1) {
                        const int dist0  = qr0.position - (key + n * 8 + pair);
                        const int dist1  = qr1.position - (key + n * 8 + pair);
                        const auto score = [&](float x, float y, int distance) {
                            return interior || (distance >= 0 && (!Swa || distance < kWindow))
                                       ? fmaf(x, scale2, y * kLog2E)
                                       : -CUDART_INF_F;
                        };
                        auto& c = scores[n];
                        c[0]    = score(c[0], bias0.x, dist0);
                        c[1]    = score(c[1], bias0.y, dist0 - 1);
                        c[2]    = score(c[2], bias1.x, dist1);
                        c[3]    = score(c[3], bias1.y, dist1 - 1);
                    };
                    if (near && can_shear) {
                        const auto project = [&](int n) {
                            const int distance = warp_low - key + 16 - n * 8 - quad;
                            unsigned b0 = 0, b1 = 0;
                            if constexpr (RegisterQ) {
                                if (Contiguous || projection_shift >= 0) {
                                    const auto* pr =
                                        smem.input
                                            .projection[stage][projection_shift + n * 8 + quad];
                                    b0 = load_pair(pr, pair);
                                    b1 = load_pair(pr, pair + 8);
                                }
                            }
                            if (!Contiguous && (!RegisterQ || projection_shift < 0) &&
                                unsigned(distance) < kExtent) {
                                const auto* pr = q.projection + distance * kRelDim;
                                b0             = load_pair(pr, pair);
                                b1             = load_pair(pr, pair + 8);
                            }
                            float4 result{};
                            mma::mma_bf16(result.x, result.y, result.z, result.w, rel0, rel1, rel2,
                                          rel3, b0, b1);
                            return result;
                        };
                        auto a = project(0), b = project(1);
#pragma unroll
                        for (int n = 0; n < ScoreTiles; ++n) {
                            const auto c = project(n + 2);
                            if constexpr (Contiguous) {
                                // The distances of the two rows differ by eight keys, so
                                // both gathers have the same source lanes within the quad.
                                const int offset = 8 - quad;
                                const bool odd   = offset & 1;
                                const int src    = (pair + offset) / 2;
                                const auto shear = [&](float lo0, float lo1, float hi0, float hi1) {
                                    const float x = pair < offset ? hi0 : lo0;
                                    const float y = pair + 1 < offset ? hi1 : lo1;
                                    return make_float2(
                                        __shfl_sync(kWarp, odd ? y : x, src & 3, 4),
                                        __shfl_sync(kWarp, odd ? x : y, (src + odd) & 3, 4));
                                };
                                mask_and_add(n, shear(b.x, b.y, c.x, c.y),
                                             shear(a.z, a.w, b.z, b.w));
                            } else {
                                mask_and_add(
                                    n, shear_bias({a.x, a.y}, {b.x, b.y}, {c.x, c.y}, shift0, pair),
                                    shear_bias({a.z, a.w}, {b.z, b.w}, {c.z, c.w}, shift1, pair));
                            }
                            a = b;
                            b = c;
                        }
                    } else if (near) {
#pragma unroll
                        for (int n = 0; n < ScoreTiles; ++n) {
                            const int dist0 = qr0.position - (key + n * 8 + pair);
                            const int dist1 = qr1.position - (key + n * 8 + pair);
                            const auto bias = [&](QueryRow row, int distance) {
                                return relative_dot(q.r + row.index * kRelDim, q.projection,
                                                    distance);
                            };
                            mask_and_add(n, {bias(qr0, dist0), bias(qr0, dist0 - 1)},
                                         {bias(qr1, dist1), bias(qr1, dist1 - 1)});
                        }
                    } else {
#pragma unroll
                        for (int n = 0; n < ScoreTiles; ++n)
#pragma unroll
                            for (float& x : scores[n]) x *= scale2;
                    }
                } else {
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
                        const int shift0 = 16 - max(0, qr0.position - warp_low);
                        const int shift1 = 16 - max(0, qr1.position - warp_low);
                        const bool interior =
                            key + (K - 1) <= low && (!Swa || high - key < kWindow);
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
                                        (Contiguous || can_shear)
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
                            if constexpr (!IndependentHeads || Contiguous) {
                                const unsigned quad_mask = 0xfu << (threadIdx.x & 28);
                                mass                     = quad_sum(mass, quad_mask);
                            }
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
                        scores[n][0] = a;
                        scores[n][1] = b;
                        scores[n][2] = c;
                        scores[n][3] = d;
                    } else {
                        p[n / 2][(n % 2) * 2]     = mma::pack_f16x2(a, b);
                        p[n / 2][(n % 2) * 2 + 1] = mma::pack_f16x2(c, d);
                    }
                }
                if constexpr (Fp8) {
#pragma unroll
                    for (int k = 0; k < K / 32; ++k) {
                        p_mx[k] = quantize_probabilities(scores + k * 4,
                                                         smem.raw[stage].scales[1] + k * 32, lane);
                    }
                }
                sum0 =
                    fmaf(sum0, alpha0, IndependentHeads && !Contiguous ? quad_sum(norm0) : norm0);
                sum1 =
                    fmaf(sum1, alpha1, IndependentHeads && !Contiguous ? quad_sum(norm1) : norm1);
                // Once all rows retain their running maxima, carry the previous
                // output across the tile without multiplying every accumulator.
                if ((!SharedHeads && !RegisterBias) ||
                    __any_sync(kWarp, alpha0 != 1.0f || alpha1 != 1.0f)) {
#pragma unroll
                    for (int n = 0; n < OutputTiles; ++n) {
                        out[n][0] *= alpha0;
                        out[n][1] *= alpha0;
                        out[n][2] *= alpha1;
                        out[n][3] *= alpha1;
                    }
                }
            }

            if constexpr (!Fp8 && !SharedHeads) {
                mma::cp_wait<0>();
                __syncthreads();
                if (next_tile < range.last) {
                    const int next = next_tile * K;
                    record         = rows.record(next);
                    stage_16bit<Swa, false, K, Threads, Contiguous>(smem.k, rows, next,
                                                                    first_visible, high, record);
                    if constexpr (RegisterQ)
                        stage_projection<K, Threads, QRows>(smem.input.projection[0], q.projection,
                                                            low, next, staged_projection);
                    mma::cp_commit();
                }
            }
            if (computes) {
                if constexpr (Fp8) {
                    const auto& raw = smem.raw[stage];
#pragma unroll
                    for (int k = 0; k < K / 32; ++k) {
                        const auto& a = p_mx[k];
#pragma unroll
                        for (int n = 0; n < OutputTiles; n += 2) {
                            unsigned b[4];
                            const int col = mma::swizzle_xor_128(lane & 15, n * 8);
                            load_transposed_fp8(
                                b, mma::smem_addr(raw.codes[1] + (k * 32 + lane) * kDim + col));
#pragma unroll
                            for (int i = 0; i < 2; ++i) {
                                auto& c = out[n + i];
                                mma::mma_fp8_e4m3_scaled(c[0], c[1], c[2], c[3], a.hi[0], a.hi[1],
                                                         a.hi[2], a.hi[3], b[i], b[i + 2],
                                                         a.scale_hi, 127u);
                                mma::mma_fp8_e4m3_scaled(c[0], c[1], c[2], c[3], a.lo[0], a.lo[1],
                                                         a.lo[2], a.lo[3], b[i], b[i + 2],
                                                         a.scale_lo, 127u);
                            }
                        }
                    }
                } else {
                    constexpr int Pairs = OutputTiles / 2;
                    constexpr int Loads = (K / 16) * Pairs;
                    const int vr        = lane & 15;
                    const auto load_v   = [&](int step, unsigned* frag) {
                        const int k = step / Pairs, n = step % Pairs * 2;
                        const int vc       = n * 8 + (lane >> 4) * 8;
                        const auto* values = [&] {
                            if constexpr (SharedHeads)
                                return smem.kv[stage].v;
                            else
                                return smem.v;
                        }();
                        mma::ldmatrix_x4_t(frag[0], frag[1], frag[2], frag[3],
                                             mma::smem_addr(values + (k * 16 + vr) * kDim +
                                                            mma::swizzle_xor_64(vr, vc)));
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
                            mma::mma_f16(o[0], o[1], o[2], o[3], p[k][0], p[k][1], p[k][2], p[k][3],
                                         x, y);
                        };
                        const auto& v = value_frag[current];
                        multiply(n, v[0], v[1]);
                        multiply(n + 1, v[2], v[3]);
                    }
                }
            }

            mma::cp_wait<0>();
            __syncthreads();
            if constexpr (SharedHeads || Fp8) stage ^= 1;
            tile = next_tile;
        }
    }

    if constexpr (!Direct)
        if (signal) cudaTriggerProgrammaticLaunchCompletion();
    auto write = [&](bool owns, QueryRow row, float sum, float reference, int half) {
        if (!owns) return;
        const bool visible       = row.position >= 0 && sum > 0;
        const float lse          = visible ? reference + __log2f(sum) : -CUDART_INF_F;
        const float inverse      = visible ? __frcp_rn(sum) : 0.0f;
        const std::size_t record = Direct ? 0 : split * total_rows + row.index;
        if constexpr (Direct) {
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
    write(owns0, qr0, IndependentHeads && !Contiguous ? sum0 : quad_sum(sum0), ref0, 0);
    write(owns1, qr1, IndependentHeads && !Contiguous ? sum1 : quad_sum(sum1), ref1, 2);
}

} // namespace ninfer::ops::relative
