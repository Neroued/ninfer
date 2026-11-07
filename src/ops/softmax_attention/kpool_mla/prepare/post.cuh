#pragma once

#include "ops/softmax_attention/kpool_mla/prepare/coordinates.cuh"
#include "ops/softmax_attention/kpool_mla/index_fp8.cuh"
#include "ops/common/cuda/reduce.cuh"

namespace ninfer::ops::detail::kpool {

__device__ __forceinline__ float projection_value(const PrepareArguments& a, int column, int row) {
    if (a.splits == 1)
        return a.projected[std::int64_t(column) * a.norm_dim + (row < 640 ? row : row - 128)];
    float sum = 0.F;
    for (int s = 0; s < a.splits; ++s)
        sum += a.projected[(std::int64_t(s) * a.stage_columns + column) * a.rows + row];
    return sum;
}

__device__ __forceinline__ __nv_bfloat16* tail_at(const TailField& f, int b, int r) {
    return f.data + f.offsets[b] / 2 + r * f.column_stride;
}

__device__ __forceinline__ __nv_bfloat16* record_at(const PrepareArguments& a, int kind, int b,
                                                    int t) {
    const auto& f = a.records[kind];
    return f.data + b * f.row_stride + (std::int64_t(a.record_offsets[b]) + t) * f.column_stride;
}

__device__ __forceinline__ __nv_bfloat16* carry_at(const PrepareArguments& a, int kind, int b,
                                                   int r) {
    return a.carry + ((b * 2 + kind) * 3 + r) * 128;
}

__device__ __forceinline__ __nv_bfloat16 old_word(const PrepareArguments& a, int kind, int b,
                                                  int t0, int frontier, int position, int feature) {
    const int initial = a.positions[std::int64_t(b) * a.tokens];
    if (!a.update && position >= initial) return record_at(a, kind, b, position - initial)[feature];
    const int r = position - (frontier - frontier % 4);
    if (a.update && t0 > 0) return carry_at(a, kind, b, r)[feature];
    return tail_at(a.before[kind], b, r)[feature];
}

// Decode update CTAs reserve the upper four warps for end-tail work. Norm jobs use
// the lower four warps and an explicit-count barrier; their reduction order is unchanged.
template <bool Subgroup>
__device__ __forceinline__ void prepare_norm_sync() {
    if constexpr (Subgroup)
        asm volatile("bar.sync 1, 128;" ::: "memory");
    else
        __syncthreads();
}

template <bool Subgroup>
__device__ __forceinline__ float prepare_norm_sum(float value, float* reduction) {
    if constexpr (!Subgroup) return common::block_reduce_sum<128>(value, reduction);
    value          = common::warp_reduce_sum(value);
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    if (lane == 0) reduction[warp] = value;
    prepare_norm_sync<true>();
    value = threadIdx.x < 4 ? reduction[lane] : 0.F;
    if (warp == 0) value = common::warp_reduce_sum<4>(value);
    return value;
}

template <int Dimension, int Offset, bool Query, bool Subgroup = false>
__device__ __forceinline__ void rms_job(const PrepareArguments& a, std::int64_t begin, int column,
                                        float* reduction) {
    const auto m = begin + column;
    int b, t;
    coordinate(a, column, b, t);
    const int tid = threadIdx.x;
    if constexpr (Query) {
        if (a.splits > 1 && a.heads && tid < 32)
            a.coefficients[m * 32 + tid] =
                t < a.valid[b] ? projection_value(a, column, 2304 + tid) : 0.F;
    }
    if (t >= a.valid[b]) {
        if constexpr (Query)
            for (int d = tid; d < Dimension; d += 128)
                a.query[m * Dimension + d] = __float2bfloat16_rn(0.F);
        return;
    }
    float values[Dimension / 128], sum = 0.F;
#pragma unroll
    for (int i = 0; i < Dimension / 128; ++i) {
        const float v = projection_value(a, column, Offset + i * 128 + tid);
        values[i]     = v;
        sum           = fmaf(v, v, sum);
    }
    sum = prepare_norm_sum<Subgroup>(sum, reduction);
    if (tid == 0) reduction[0] = rsqrtf(sum / Dimension + a.rms_epsilon);
    prepare_norm_sync<Subgroup>();
    const float inv = reduction[0];
    __nv_bfloat16* dst;
    if constexpr (Query)
        dst = a.query + m * Dimension;
    else {
        const int p       = a.positions[m];
        const auto offset = a.page_offsets[a.table_rows[b] * a.table_stride + p / 64];
        dst               = a.latent + offset / 2 + (p % 64) * 512;
    }
    const auto* gamma = Query ? a.query_norm : a.latent_norm;
#pragma unroll
    for (int i = 0; i < Dimension / 128; ++i) {
        const int d = i * 128 + tid;
        dst[d]      = __float2bfloat16_rn(values[i] * inv * __bfloat162float(gamma[d]));
    }
}

// A warp produces four features per lane with the same BF16 publication boundary
// for records, shared pooling operands, and the request's end tail.
struct RawToken {
    __nv_bfloat16 keys[4], gates[4];

    __device__ void store(int i, __nv_bfloat16 key, __nv_bfloat16 gate) {
        keys[i]  = key;
        gates[i] = gate;
    }
};

struct SharedRawToken {
    __nv_bfloat16 *keys, *gates;

    __device__ void store(int i, __nv_bfloat16 key, __nv_bfloat16 gate) const {
        keys[i * 32]  = key;
        gates[i * 32] = gate;
    }
};

template <class Output>
__device__ __forceinline__ void raw_token(const PrepareArguments& a, std::int64_t begin, int b,
                                          int t0, int frontier, int end, int pool, int r,
                                          bool publish_record, Output& output) {
    const int lane = threadIdx.x & 31, position = pool * 4 + r;
    const int initial = a.positions[std::int64_t(b) * a.tokens];
    if (position < end && position >= frontier) {
        const int t      = position - initial;
        const int column = std::int64_t(b) * a.tokens + t - begin;
        float z[4], sum = 0.F;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            z[i] = projection_value(a, column, 512 + lane + 32 * i);
            sum += z[i];
        }
        const float mean = common::warp_sum(sum) * (1.F / 128.F);
        float variance   = 0.F;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            z[i] -= mean;
            variance = fmaf(z[i], z[i], variance);
        }
        const float inv = rsqrtf(common::warp_sum(variance) * (1.F / 128.F) + a.key_epsilon);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int d     = lane + 32 * i;
            const auto key  = __float2bfloat16_rn(z[i] * inv * __bfloat162float(a.key_norm[d]) +
                                                  __bfloat162float(a.key_bias[d]));
            const auto gate = a.splits > 1
                                  ? __float2bfloat16_rn(projection_value(a, column, 640 + d))
                              : a.update ? a.gates[std::int64_t(column) * 128 + d]
                                         : record_at(a, 1, b, t)[d];
            output.store(i, key, gate);
            if (publish_record) {
                record_at(a, 0, b, t)[d] = key;
                if (a.splits > 1) record_at(a, 1, b, t)[d] = gate;
            }
        }
    } else if (position < frontier && position < end) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int d = lane + 32 * i;
            output.store(i, old_word(a, 0, b, t0, frontier, position, d),
                         old_word(a, 1, b, t0, frontier, position, d));
        }
    }
}

__device__ __forceinline__ void raw_pool(const PrepareArguments& a, std::int64_t begin, int b,
                                         int t0, int frontier, int end, int pool,
                                         bool publish_record, __nv_bfloat16 (&keys)[4][128],
                                         __nv_bfloat16 (&gates)[4][128]) {
    const int r = threadIdx.x / 32, lane = threadIdx.x & 31;
    SharedRawToken output{keys[r] + lane, gates[r] + lane};
    raw_token(a, begin, b, t0, frontier, end, pool, r, publish_record, output);
    __syncthreads();
}

__device__ __forceinline__ float pooled_feature(const PrepareArguments& a, int d,
                                                const __nv_bfloat16 (&keys)[4][128],
                                                const __nv_bfloat16 (&gates)[4][128]) {
    float logit[4], maximum = -CUDART_INF_F;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        logit[r] = __bfloat162float(gates[r][d]) + __bfloat162float(a.ape[r * 128 + d]);
        maximum  = fmaxf(maximum, logit[r]);
    }
    float sum = 0.F, value = 0.F;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const float weight = __expf(logit[r] - maximum);
        sum += weight;
        value = fmaf(weight, __bfloat162float(keys[r][d]), value);
    }
    return value / sum;
}

template <bool Subgroup = false>
__device__ __forceinline__ void latent_fp8_job(const PrepareArguments& a, std::int64_t begin,
                                               int column, float* reduction) {
    int b, t;
    coordinate(a, column, b, t);
    if (t >= a.valid[b]) return;
    const auto m   = begin + column;
    const int lane = threadIdx.x % 32, group = threadIdx.x / 32;
    const int first = group * 128 + lane * 4;
    float values[4], sum = 0.F;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        values[i] = projection_value(a, column, first + i);
        sum       = fmaf(values[i], values[i], sum);
    }
    sum = prepare_norm_sum<Subgroup>(sum, reduction);
    if (threadIdx.x == 0) reduction[0] = rsqrtf(sum / 512 + a.rms_epsilon);
    prepare_norm_sync<Subgroup>();
    float maximum = 0.F;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        values[i] *= reduction[0] * __bfloat162float(a.latent_norm[first + i]);
        maximum = fmaxf(maximum, fabsf(values[i]));
    }
    const auto scale = common::fp8_ue8m0_scale(common::warp_max(maximum));
    const unsigned codes =
        common::fp8_quant_code2(values[0], values[1], scale.inverse_scale) |
        unsigned(common::fp8_quant_code2(values[2], values[3], scale.inverse_scale)) << 16;
    const int position = a.positions[m];
    const auto page =
        a.page_offsets[std::int64_t(a.table_rows[b]) * a.table_stride + position / 64];
    *reinterpret_cast<unsigned*>(a.latent_fp8 + page + (position % 64) * 512 + first) = codes;
    if (lane == 0) a.latent_scales[page + (position % 64) * 4 + group] = scale.scale;
}

template <bool Fp8Index>
__device__ __forceinline__ void publish_pool(const PrepareArguments& a, int b, int pool,
                                             const __nv_bfloat16 (&keys)[4][128],
                                             const __nv_bfloat16 (&gates)[4][128]) {
    const auto offset = a.page_offsets[a.table_rows[b] * a.table_stride + pool / 16];
    if constexpr (Fp8Index) {
        if (threadIdx.x < 32) {
            const int lane = threadIdx.x;
            float values[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) values[j] = pooled_feature(a, lane * 4 + j, keys, gates);
            const auto encoded = encode_index_lane<false>(values);
            auto* codes        = static_cast<std::uint8_t*>(a.index) + offset + (pool % 16) * 128;
            reinterpret_cast<unsigned*>(codes)[lane] = encoded.codes;
            if (lane == 0) a.index_scales[offset + pool % 16] = encoded.scale;
        }
    } else if (threadIdx.x < 128) {
        const int d = threadIdx.x;
        static_cast<__nv_bfloat16*>(a.index)[offset / 2 + (pool % 16) * 128 + d] =
            __float2bfloat16_rn(pooled_feature(a, d, keys, gates));
    }
}

template <bool Fp8Index, bool Fp8Latent = false, bool ParallelTail = false>
__global__ __launch_bounds__(ParallelTail ? 256 : 128) void prepare_post_kernel(PrepareArguments a,
                                                                                std::int64_t begin,
                                                                                int columns) {
    __shared__ float reduction[4];
    __shared__ __nv_bfloat16 keys[4][128], gates[4][128];
    int job = blockIdx.x;
    if (job < columns) {
        if constexpr (ParallelTail)
            if (threadIdx.x >= 128) return;
        if constexpr (Fp8Latent)
            latent_fp8_job<ParallelTail>(a, begin, job, reduction);
        else
            rms_job<512, 0, false, ParallelTail>(a, begin, job, reduction);
        return;
    }
    job -= columns;
    if (a.query_rank) {
        if (job < columns) {
            if constexpr (ParallelTail)
                if (threadIdx.x >= 128) return;
            rms_job<1536, 768, true, ParallelTail>(a, begin, job, reduction);
            return;
        }
        job -= columns;
    }
    const int first_b = a.first_row, last_b = a.last_row;
    int b = first_b, t0 = 0, t1 = 0;
    for (; b <= last_b; ++b) {
        t0                 = b == first_b ? a.first_column : 0;
        t1                 = b == last_b ? a.last_stop : a.tokens;
        const int capacity = (t1 - t0 + 6) / 4;
        if (job < capacity) break;
        job -= capacity;
    }
    if (b > last_b) return;
    const int n = a.valid[b], stop = min(t1, n);
    if (stop <= t0) return;
    const int initial  = a.positions[std::int64_t(b) * a.tokens];
    const int frontier = initial + t0, end = initial + stop;
    const int first_pool = frontier / 4, pool = first_pool + job, full_end = end / 4;
    if constexpr (ParallelTail) {
        // The first pool's CTA owns every old-tail read and the end-tail publication.
        // Its lower warps pool the prefix while three upper warps normalize the suffix.
        if (job != 0 && pool >= full_end) return;
        const int tid = threadIdx.x, lane = tid & 31, remaining = end % 4;
        RawToken tail;
        if (pool < full_end && tid < 128) {
            const int r = tid / 32;
            SharedRawToken output{keys[r] + lane, gates[r] + lane};
            raw_token(a, begin, b, t0, frontier, end, pool, r, false, output);
        }
        if (job == 0 && tid >= 128 && tid < 224 && tid / 32 - 4 < remaining)
            raw_token(a, begin, b, t0, frontier, end, full_end, tid / 32 - 4, false, tail);
        __syncthreads();
        if (pool < full_end) publish_pool<Fp8Index>(a, b, pool, keys, gates);
        if (job == 0 && tid >= 128 && tid < 224) {
            const int r      = tid / 32 - 4;
            const bool final = n <= t1;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int d = lane + i * 32;
                (final ? tail_at(a.after[0], b, r) : carry_at(a, 0, b, r))[d] =
                    r < remaining ? tail.keys[i] : __float2bfloat16_rn(0.F);
                (final ? tail_at(a.after[1], b, r) : carry_at(a, 1, b, r))[d] =
                    r < remaining ? tail.gates[i] : __float2bfloat16_rn(0.F);
            }
        }
        return;
    }
    if (a.update) {
        if (job != 0 && pool >= full_end) return;
    } else if (pool >= (end + 3) / 4)
        return;
    if (pool < full_end || !a.update) {
        raw_pool(a, begin, b, t0, frontier, end, pool, !a.update, keys, gates);
        if (pool < full_end) publish_pool<Fp8Index>(a, b, pool, keys, gates);
    }

    if (a.update && job == 0) {
        // The sole old-tail reader owns the end tail too. Its last raw tokens come from the
        // completed projection, never from another CTA's yet-unpublished normalization.
        __syncthreads();
        const int remaining = end % 4;
        if (remaining) raw_pool(a, begin, b, t0, frontier, end, full_end, false, keys, gates);
        const bool final = n <= t1;
        const int d      = threadIdx.x;
#pragma unroll
        for (int r = 0; r < 3; ++r) {
            const auto k = r < remaining ? keys[r][d] : __float2bfloat16_rn(0.F);
            const auto g = r < remaining ? gates[r][d] : __float2bfloat16_rn(0.F);
            (final ? tail_at(a.after[0], b, r) : carry_at(a, 0, b, r))[d] = k;
            (final ? tail_at(a.after[1], b, r) : carry_at(a, 1, b, r))[d] = g;
        }
    }
}
} // namespace ninfer::ops::detail::kpool
