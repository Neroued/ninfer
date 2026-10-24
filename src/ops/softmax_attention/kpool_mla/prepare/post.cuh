#pragma once

#include "ops/softmax_attention/kpool_mla/prepare/coordinates.cuh"
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

template <int Dimension, int Offset, bool Query>
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
    sum = common::block_reduce_sum<128>(sum, reduction);
    if (tid == 0) reduction[0] = rsqrtf(sum / Dimension + a.rms_epsilon);
    __syncthreads();
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

// One warp owns one token. Published raw words are also the exact operands placed in shared
// memory for pooling; neither a projection nor a LayerNorm value is silently used instead.
__device__ __forceinline__ void raw_pool(const PrepareArguments& a, std::int64_t begin, int b,
                                         int t0, int frontier, int end, int pool,
                                         bool publish_record, __nv_bfloat16 (&keys)[4][128],
                                         __nv_bfloat16 (&gates)[4][128]) {
    const int r = threadIdx.x / 32, lane = threadIdx.x & 31, position = pool * 4 + r;
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
            keys[r][d]      = key;
            const auto gate = a.splits > 1
                                  ? __float2bfloat16_rn(projection_value(a, column, 640 + d))
                              : a.update ? a.gates[std::int64_t(column) * 128 + d]
                                         : record_at(a, 1, b, t)[d];
            gates[r][d]     = gate;
            if (publish_record) {
                record_at(a, 0, b, t)[d] = key;
                if (a.splits > 1) record_at(a, 1, b, t)[d] = gate;
            }
        }
    } else if (position < frontier && position < end) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int d = lane + 32 * i;
            keys[r][d]  = old_word(a, 0, b, t0, frontier, position, d);
            gates[r][d] = old_word(a, 1, b, t0, frontier, position, d);
        }
    }
    __syncthreads();
}

__global__ __launch_bounds__(128) void prepare_post_kernel(PrepareArguments a, std::int64_t begin,
                                                           int columns) {
    __shared__ float reduction[4];
    __shared__ __nv_bfloat16 keys[4][128], gates[4][128];
    int job = blockIdx.x;
    if (job < columns) {
        rms_job<512, 0, false>(a, begin, job, reduction);
        return;
    }
    job -= columns;
    if (a.query_rank) {
        if (job < columns) {
            rms_job<1536, 768, true>(a, begin, job, reduction);
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
    if (a.update) {
        if (job != 0 && pool >= full_end) return;
    } else if (pool >= (end + 3) / 4)
        return;
    if (pool < full_end || !a.update) {
        raw_pool(a, begin, b, t0, frontier, end, pool, !a.update, keys, gates);
        if (pool < full_end) {
            const int d = threadIdx.x;
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
            const auto offset = a.page_offsets[a.table_rows[b] * a.table_stride + pool / 16];
            a.index[offset / 2 + (pool % 16) * 128 + d] = __float2bfloat16_rn(value / sum);
        }
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
