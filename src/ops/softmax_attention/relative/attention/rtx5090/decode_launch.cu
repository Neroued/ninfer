#include "ops/softmax_attention/relative/attention/rtx5090/tiled.cuh"

#include <algorithm>
#include <type_traits>

namespace ninfer::ops::relative {
namespace {

constexpr int kWarps      = 4;
constexpr int kMergeWarps = 4;

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int i = 16; i; i >>= 1) x += __shfl_xor_sync(kWarp, x, i);
    return x;
}

__device__ __forceinline__ common::F32Vector<16> read_bf16(const std::uint16_t* p) {
    const auto lo = common::load_vec<uint4>(p);
    const auto hi = common::load_vec<uint4>(p + 8);
    const unsigned words[]{lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    common::F32Vector<16> values;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        values.word[2 * i]     = __uint_as_float(words[i] << 16);
        values.word[2 * i + 1] = __uint_as_float(words[i] & 0xffff0000u);
    }
    return values;
}

__device__ __forceinline__ common::F32Vector<16> expand_fp8(uint4 codes) {
    common::F32Vector<16> result;
    const auto* pairs = reinterpret_cast<const unsigned short*>(&codes);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const auto decoded     = common::fp8_code2_to_float2(pairs[i]);
        result.word[2 * i]     = decoded.x;
        result.word[2 * i + 1] = decoded.y;
    }
    return result;
}

__device__ __forceinline__ common::F32Vector<16> expand_scaled_key(uint4 codes, unsigned char e) {
    if (e < 10 || e > 245) {
        auto x            = expand_fp8(codes);
        const float scale = common::ue8m0_to_float(e);
#pragma unroll
        for (auto& f : x.word) f *= scale;
        return x;
    }
    // Every E4M3 value times a power of two in this range is exactly representable
    // as a normal BF16 value or zero. Decode and scale both values in one pair.
    const auto* pairs = reinterpret_cast<const unsigned short*>(&codes);
    common::F32Vector<16> result;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const auto decoded     = common::fp8_dequant_bf16x2(pairs[i], e);
        result.word[2 * i]     = __uint_as_float(decoded << 16);
        result.word[2 * i + 1] = __uint_as_float(decoded & 0xffff0000u);
    }
    return result;
}

__device__ __forceinline__ common::F32Vector<16> expand_fp16(uint4 lo, uint4 hi) {
    const unsigned words[]{lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    common::F32Vector<16> values;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const auto pair        = __half22float2(common::load_vec<__half2>(&words[i]));
        values.word[2 * i]     = pair.x;
        values.word[2 * i + 1] = pair.y;
    }
    return values;
}

__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) x = fmaxf(x, __shfl_xor_sync(kWarp, x, offset));
    return x;
}

__device__ __forceinline__ void write_output(Query q, std::int64_t row, int dim, float4 v,
                                             float lse2) {
    ushort4 out{};
    if (lse2 != -CUDART_INF_F) {
        const float factor = sink_factor(q, row, lse2);
        out                = {gated(q, row, dim, v.x, factor), gated(q, row, dim + 1, v.y, factor),
                              gated(q, row, dim + 2, v.z, factor), gated(q, row, dim + 3, v.w, factor)};
    }
    common::store_vec(q.out + row * kDim + dim, out);
}

constexpr int kDirectWarps    = 32;
constexpr int kDirectKeyLanes = 8;
constexpr int kDirectStreams  = kDirectWarps * 32 / kDirectKeyLanes;
constexpr int kDirectFeatures = kDirectSwaDecodeStripe / kDirectKeyLanes;

struct DirectState {
    float maximum, mass;
    float values[kDirectFeatures];
};

template <bool Chunk>
__device__ __forceinline__ DirectState direct_swa_keys(Query q, CacheView cache, std::int64_t row,
                                                       int batch, int head, int position,
                                                       int stripe) {
    const int lane      = threadIdx.x & 31;
    const int stream    = threadIdx.x / kDirectKeyLanes;
    const int feature   = threadIdx.x % kDirectKeyLanes;
    const unsigned mask = 0xffu << (lane & ~7);
    const int first     = max(0, position - (kWindow - 1));
    const CacheRows<true, false> cache_rows(cache, batch, head / kGroup, batch, position,
                                            !Chunk || first < position);
    const auto query    = read_bf16(q.q + row * kDim + feature * 16);
    const auto relative = common::load_vec<unsigned>(q.r + row * kRelDim + feature * 2);
    const float r0      = __uint_as_float(relative << 16);
    const float r1      = __uint_as_float(relative & 0xffff0000u);
    DirectState state{-CUDART_INF_F, 0.0f, {}};

    for (int offset = stream; offset <= position - first; offset += kDirectStreams) {
        const int key      = first + offset;
        const int k_offset = feature * 16;
        const int v_offset = stripe + feature * kDirectFeatures;
        const auto* k      = cache_rows.k;
        const auto* v      = cache_rows.v;
        if constexpr (Chunk) {
            k = cache_rows.codes(false, key, 0);
            v = cache_rows.codes(true, key, 0);
        } else {
            const int slot_bytes = (key & (kWindow - 1)) * (2 * kDim);
            k += slot_bytes;
            v += slot_bytes;
        }
        const auto packed_v = common::load_vec<uint2>(v + v_offset * 2);
        const auto keys     = read_bf16(reinterpret_cast<const std::uint16_t*>(k) + k_offset);
        const auto p =
            common::load_vec<unsigned>(q.projection + (position - key) * kRelDim + feature * 2);
        float dot0 = 0.0f, dot1 = 0.0f;
#pragma unroll
        for (int d = 0; d < 16; d += 2) {
            dot0 = fmaf(query.word[d], keys.word[d], dot0);
            dot1 = fmaf(query.word[d + 1], keys.word[d + 1], dot1);
        }
        const float bias =
            fmaf(r1, __uint_as_float(p & 0xffff0000u), r0 * __uint_as_float(p << 16));
        float score = fmaf(dot0 + dot1, q.scale, bias) * kLog2E;
#pragma unroll
        for (int step = kDirectKeyLanes / 2; step; step /= 2)
            score += __shfl_xor_sync(mask, score, step, kDirectKeyLanes);
        const float next  = fmaxf(state.maximum, score);
        const float alpha = common::exp2_approx(state.maximum - next);
        const float beta  = common::exp2_approx(score - next);
        state.maximum     = next;
        state.mass        = fmaf(state.mass, alpha, beta);
        const auto lo     = __half22float2(common::load_vec<__half2>(&packed_v.x));
        const auto hi     = __half22float2(common::load_vec<__half2>(&packed_v.y));
        const float values[]{lo.x, lo.y, hi.x, hi.y};
#pragma unroll
        for (int d = 0; d < kDirectFeatures; ++d)
            state.values[d] = fmaf(values[d], beta, state.values[d] * alpha);
    }
    return state;
}

__global__ __launch_bounds__(kDirectWarps * 32) void relative_decode_direct_swa(Query q,
                                                                                CacheView cache,
                                                                                const int* valid) {
    constexpr int Stripe = kDirectSwaDecodeStripe;
    static_assert(kDirectFeatures == 4);
    __shared__ float maxima[kDirectWarps], masses[kDirectWarps];
    __shared__ __align__(16) float values[kDirectWarps][Stripe];

    const int batch = blockIdx.z, head = blockIdx.x, stripe = blockIdx.y * Stripe;
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    const auto row     = std::int64_t(batch) * kHeads + head;
    const int position = q.positions[batch];
    if ((valid && valid[batch] <= 0) || position < 0) {
        if (threadIdx.x < Stripe / 4)
            write_output(q, row, stripe + threadIdx.x * 4, {}, -CUDART_INF_F);
        return;
    }
    auto state = cache.chunk.k
                     ? direct_swa_keys<true>(q, cache, row, batch, head, position, stripe)
                     : direct_swa_keys<false>(q, cache, row, batch, head, position, stripe);

    // Each group of eight lanes evaluates independent keys. Merge the four groups
    // in registers before publishing one partial per warp.
    float maximum = state.maximum;
#pragma unroll
    for (int step = 16; step >= kDirectKeyLanes; step /= 2)
        maximum = fmaxf(maximum, __shfl_xor_sync(kWarp, maximum, step));
    const float warp_weight = state.mass > 0 ? common::exp2_approx(state.maximum - maximum) : 0.0f;
    state.mass *= warp_weight;
#pragma unroll
    for (auto& x : state.values) x *= warp_weight;
#pragma unroll
    for (int step = 16; step >= kDirectKeyLanes; step /= 2) {
        state.mass += __shfl_xor_sync(kWarp, state.mass, step);
#pragma unroll
        for (auto& x : state.values) x += __shfl_xor_sync(kWarp, x, step);
    }
    if (lane == 0) {
        maxima[warp] = maximum;
        masses[warp] = state.mass;
    }
    if (lane < kDirectKeyLanes)
        common::store_vec(
            values[warp] + lane * kDirectFeatures,
            make_float4(state.values[0], state.values[1], state.values[2], state.values[3]));
    __syncthreads();

    constexpr int ColumnsPerWarp  = 8;
    constexpr int ReductionGroups = 32 / ColumnsPerWarp;
    if (warp >= Stripe / ColumnsPerWarp) return;
    const float local  = maxima[lane];
    maximum            = warp_max(local);
    const float weight = local == -CUDART_INF_F ? 0.0f : common::exp2_approx(local - maximum);
    const float mass   = warp_sum(weight * masses[lane]);
    const int column   = warp * ColumnsPerWarp + lane % ColumnsPerWarp;
    const int group    = lane / ColumnsPerWarp;
    float output       = 0.0f;
#pragma unroll
    for (int w = 0; w < kDirectWarps; w += ReductionGroups) {
        const float wgt = __shfl_sync(kWarp, weight, w + group);
        output          = fmaf(values[w + group][column], wgt, output);
    }
#pragma unroll
    for (int step = 16; step >= ColumnsPerWarp; step /= 2)
        output += __shfl_xor_sync(kWarp, output, step);
    if (group == 0) {
        const float value     = mass > 0.0f ? output * common::rcp_approx_ftz(mass) : 0.0f;
        const float factor    = mass > 0.0f ? sink_factor(q, row, maximum + __log2f(mass)) : 0.0f;
        const int d           = stripe + column;
        q.out[row * kDim + d] = gated(q, row, d, value, factor);
    }
}

struct SimtStorage {
    // Eight lanes consume one complete key and value, sharing it across all three heads.
    static constexpr int Lanes   = 8;
    static constexpr int Streams = kWarps * 32 / Lanes;
    float values[kGroup][Streams][kDim];
    float maximum[kGroup][Streams], mass[kGroup][Streams];
};

template <bool Swa, bool Fp8, bool Direct>
__device__ __forceinline__ void
simt_attention(Query q, CacheView cache, int batch, int live, int kvhead, int split, int splits,
               Partials partials, std::size_t total_rows, SimtStorage& smem, bool signal) {
    constexpr int Lanes       = SimtStorage::Lanes;
    constexpr int Streams     = SimtStorage::Streams;
    constexpr int Features    = kDim / Lanes;
    constexpr int RelFeatures = kRelDim / Lanes;
    constexpr int Keys        = Swa ? kSwaDecodeKeyTile : kKeyTile;
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    const int group              = threadIdx.x / Lanes;
    const int lane_in_group      = threadIdx.x % Lanes;
    const int feature            = lane_in_group * Features;
    const unsigned mask          = 0xffu << (lane & ~7);
    const auto range             = key_range<Swa, Keys, 1>(q.positions + batch, live, split, splits,
                                                           position_limit<Swa>(cache));
    const int table              = range.first < range.last ? cache_row<Swa>(cache, batch) : -1;
    const std::int64_t first_row = std::int64_t(batch) * kHeads + kvhead * kGroup;

    float acc[kGroup][Features]{};
    float maximum[kGroup], denom[kGroup]{};
#pragma unroll
    for (int h = 0; h < kGroup; ++h) maximum[h] = -CUDART_INF_F;
    if (table >= 0 && table < cache.table_count) {
        const int pos = range.high;
        const CacheRows<Swa, Fp8> data(cache, table, kvhead, batch, pos,
                                       !cache.chunk.k || range.first * Keys < pos);
        const int oldest    = Swa ? max(0, pos - (kWindow - 1)) : 0;
        const bool any_bias = range.last * std::int64_t(Keys) > std::int64_t(pos) - kExtent + 1;
        common::F32Vector<Features> qr[kGroup];
        float rel[kGroup][RelFeatures]{};
#pragma unroll
        for (int h = 0; h < kGroup; ++h) {
            const auto row = first_row + h;
            qr[h]          = read_bf16(q.q + row * kDim + feature);
            if (any_bias) {
#pragma unroll
                for (int j = 0; j < RelFeatures; ++j)
                    rel[h][j] = bf16(q.r[row * kRelDim + lane_in_group * RelFeatures + j]);
            }
        }
        for (int tile = range.first; tile < range.last; ++tile) {
            const int start   = tile * Keys;
            const auto record = data.record(start);
#pragma unroll 1
            for (int i = group; i < Keys; i += Streams) {
                const int key = start + i;
                if (key > pos) break;
                if (key < oldest) continue;
                const int offset = feature * (Fp8 ? 1 : 2);
                const auto* vk   = data.codes(true, key, record) + offset;
                // Make the compressed V load independent of the K dot and softmax.
                uint4 packed_value{};
                unsigned char vs_exp = 127;
                if constexpr (Fp8) {
                    packed_value = common::load_vec<uint4>(vk);
                    vs_exp       = *data.scale(true, key, record);
                }
                uint4 value_lo{}, value_hi{};
                if constexpr (!Fp8) {
                    value_lo = common::load_vec<uint4>(vk);
                    value_hi = common::load_vec<uint4>(vk + sizeof(uint4));
                }
                const auto* kk = data.codes(false, key, record) + offset;
                common::F32Vector<Features> k;
                if constexpr (Fp8) {
                    const auto exponent = *data.scale(false, key, record);
                    k                   = expand_scaled_key(common::load_vec<uint4>(kk), exponent);
                } else {
                    k = read_bf16(reinterpret_cast<const std::uint16_t*>(kk));
                }
                const float scale   = q.scale;
                const int distance  = pos - key;
                const bool has_bias = Swa || distance < kExtent;
                float proj[RelFeatures]{};
                if (has_bias) {
#pragma unroll
                    for (int j = 0; j < RelFeatures; j += 2) {
                        const auto* src =
                            q.projection + distance * kRelDim + lane_in_group * RelFeatures + j;
                        const unsigned packed = common::load_vec<unsigned>(src);
                        proj[j]               = __uint_as_float(packed << 16);
                        proj[j + 1]           = __uint_as_float(packed & 0xffff0000u);
                    }
                }
                float alpha[kGroup], beta[kGroup];
#pragma unroll
                for (int h = 0; h < kGroup; ++h) {
                    float dot = 0;
#pragma unroll
                    for (int f = 0; f < Features; ++f) dot = fmaf(qr[h].word[f], k.word[f], dot);
                    float bias = 0;
                    if (has_bias) {
#pragma unroll
                        for (int j = 0; j < RelFeatures; ++j) bias = fmaf(rel[h][j], proj[j], bias);
                    }
                    float score;
                    if constexpr (Fp8 && !Swa) {
                        if (!has_bias) {
#pragma unroll
                            for (int j = Lanes / 2; j; j >>= 1)
                                dot += __shfl_xor_sync(mask, dot, j, Lanes);
                            score = dot * scale;
                        } else {
                            score = fmaf(dot, scale, bias);
#pragma unroll
                            for (int j = Lanes / 2; j; j >>= 1)
                                score += __shfl_xor_sync(mask, score, j, Lanes);
                        }
                    } else {
                        score = fmaf(dot, scale, bias);
#pragma unroll
                        for (int j = Lanes / 2; j; j >>= 1)
                            score += __shfl_xor_sync(mask, score, j, Lanes);
                    }
                    score *= kLog2E;
                    const float next = fmaxf(maximum[h], score);
                    alpha[h]         = common::exp2_approx(maximum[h] - next);
                    beta[h]          = common::exp2_approx(score - next);
                    maximum[h]       = next;
                    denom[h]         = fmaf(denom[h], alpha[h], beta[h]);
                }
                common::F32Vector<Features> value;
                if constexpr (Fp8) {
                    value               = expand_fp8(packed_value);
                    const auto exponent = vs_exp;
                    const float vs      = common::ue8m0_to_float(exponent);
                    bool fold           = exponent >= 32;
#pragma unroll
                    for (int h = 0; h < kGroup; ++h) fold &= beta[h] == 0.0f || beta[h] >= 0x1p-31f;
                    if (fold) {
#pragma unroll
                        for (int h = 0; h < kGroup; ++h) beta[h] *= vs;
                    } else {
#pragma unroll
                        for (int f = 0; f < Features; ++f) value.word[f] *= vs;
                    }
                } else {
                    value = expand_fp16(value_lo, value_hi);
                }
#pragma unroll
                for (int h = 0; h < kGroup; ++h)
#pragma unroll
                    for (int f = 0; f < Features; ++f)
                        acc[h][f] = fmaf(value.word[f], beta[h], acc[h][f] * alpha[h]);
            }
        }
    }

#pragma unroll
    for (int h = 0; h < kGroup; ++h) {
        if (lane_in_group == 0) {
            smem.maximum[h][group] = maximum[h];
            smem.mass[h][group]    = denom[h];
        }
        if (denom[h] > 0) {
#pragma unroll
            for (int f = 0; f < Features; f += 4)
                common::store_vec(
                    smem.values[h][group] + feature + f,
                    make_float4(acc[h][f], acc[h][f + 1], acc[h][f + 2], acc[h][f + 3]));
        }
    }
    __syncthreads();

    if constexpr (!Direct)
        if (signal) cudaTriggerProgrammaticLaunchCompletion();
    if (warp < kGroup) {
        const auto row         = first_row + warp;
        const float stream_max = lane < Streams ? smem.maximum[warp][lane] : -CUDART_INF_F;
        const float mass       = lane < Streams ? smem.mass[warp][lane] : 0.0f;
        const float maximum    = warp_max(stream_max);
        const float weight     = mass > 0.0f ? common::exp2_approx(stream_max - maximum) : 0.0f;
        const float norm       = warp_sum(weight * mass);
        float4 out{};
#pragma unroll
        for (int i = 0; i < Streams; ++i) {
            const float w = __shfl_sync(kWarp, weight, i);
            if (w == 0) continue;
            const auto v = common::load_vec<float4>(smem.values[warp][i] + lane * 4);
            out.x        = fmaf(v.x, w, out.x);
            out.y        = fmaf(v.y, w, out.y);
            out.z        = fmaf(v.z, w, out.z);
            out.w        = fmaf(v.w, w, out.w);
        }
        const float out_lse = norm > 0 ? maximum + __log2f(norm) : -CUDART_INF_F;
        const float inv     = norm > 0 ? common::rcp_approx_ftz(norm) : 0.0f;
        const auto value    = make_float4(out.x * inv, out.y * inv, out.z * inv, out.w * inv);
        if constexpr (Direct)
            write_output(q, row, lane * 4, value, out_lse);
        else {
            const auto rec = split * total_rows + row;
            if (lane == 0) partials.lse[rec] = out_lse;
            if (norm > 0) common::store_vec(partials.values + rec * kDim + lane * 4, value);
        }
    }
    if (!signal) __syncthreads();
}

template <bool Swa, bool Fp8, bool Direct, bool Simt, int TileWarps = 1, bool Adaptive = false>
__global__ __launch_bounds__((Simt ? kWarps : TileWarps + 1) *
                             32) void relative_decode(Query q, CacheView cache, const int* valid,
                                                      int batch, int width, int splits,
                                                      Partials partials, int streaming_splits) {
    constexpr int Keys = Swa && !Fp8 ? kSwaVerifyKeyTile : kKeyTile;
    using Shared = std::conditional_t<Simt, SimtStorage, TileStorage<TileWarps, Swa, Fp8, Keys>>;
    __shared__ Shared storage;
    const std::size_t total_rows = std::size_t(batch) * width * kHeads;
    for (auto b = std::int64_t(blockIdx.z); b < batch; b += gridDim.z) {
        const int live  = valid ? min(width, max(0, valid[b])) : width;
        const bool last = b + gridDim.z >= batch;
        if constexpr (Simt) {
            int active = splits;
            if constexpr (Adaptive) {
                active = active_decode_splits(std::int64_t(q.positions[b]) + 1, splits,
                                              streaming_splits);
                if (blockIdx.x >= static_cast<unsigned>(active)) continue;
            }
            simt_attention<Swa, Fp8, Direct>(q, cache, int(b), live, blockIdx.y, blockIdx.x, active,
                                             partials, total_rows, storage, last);
        } else {
            tiled_attention<Fp8, Swa, false, TileWarps, (TileWarps + 1) * 32, Direct, Keys, 16,
                            kGroup>(q, cache, int(b), b * width, b * width, live, width, blockIdx.y,
                                    blockIdx.x, splits, partials, total_rows, storage, last);
            if (!last) __syncthreads();
        }
    }
}

template <bool Adaptive = false>
__global__ __launch_bounds__(128) void relative_combine(Query q, const int* valid, int batch,
                                                        int width, int splits, Partials partials,
                                                        int streaming_splits) {
    __shared__ float weights[kMaxSplits];
    __shared__ float maxima[kMergeWarps], sums[kMergeWarps];
    __shared__ float4 values[kMergeWarps][32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    const int local        = blockIdx.x;
    const int dim          = 4 * lane;
    const std::size_t rows = std::size_t(batch) * width * kHeads;
    cudaGridDependencySynchronize();
    for (auto b = std::int64_t(blockIdx.z); b < batch; b += gridDim.z) {
        const auto row = b * width * kHeads + local;
        if (valid && local / kHeads >= valid[b]) {
            if (warp == 0) write_output(q, row, dim, {}, -CUDART_INF_F);
            continue;
        }

        int active = splits;
        if constexpr (Adaptive)
            active =
                active_decode_splits(std::int64_t(q.positions[b]) + 1, splits, streaming_splits);
        const float lse = tid < active ? partials.lse[tid * rows + row] : -CUDART_INF_F;
        float maximum   = lse;
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            maximum = fmaxf(maximum, __shfl_xor_sync(kWarp, maximum, offset));
        if (lane == 0) maxima[warp] = maximum;
        __syncthreads();
        maximum            = fmaxf(fmaxf(maxima[0], maxima[1]), fmaxf(maxima[2], maxima[3]));
        const float weight = maximum > -CUDART_INF_F ? common::exp2_approx(lse - maximum) : 0.0f;
        weights[tid]       = weight;
        const float norm   = warp_sum(weight);
        if (lane == 0) sums[warp] = norm;
        __syncthreads();

        float4 acc{};
        for (int i = warp; i < active; i += kMergeWarps) {
            const float w = weights[i];
            if (w == 0.0f) continue;
            const auto* src = partials.values + (i * rows + row) * kDim + dim;
            const auto v    = common::load_vec<float4>(src);
            acc.x           = fmaf(v.x, w, acc.x);
            acc.y           = fmaf(v.y, w, acc.y);
            acc.z           = fmaf(v.z, w, acc.z);
            acc.w           = fmaf(v.w, w, acc.w);
        }
        values[warp][lane] = acc;
        __syncthreads();
        if (warp == 0) {
            const float sum  = sums[0] + sums[1] + sums[2] + sums[3];
            const float inv  = sum > 0 ? common::rcp_approx_ftz(sum) : 0.0f;
            const float lse2 = sum > 0 ? maximum + __log2f(sum) : -CUDART_INF_F;
            const auto a = values[1][lane], c = values[2][lane], d = values[3][lane];
            write_output(q, row, dim,
                         {(acc.x + a.x + c.x + d.x) * inv, (acc.y + a.y + c.y + d.y) * inv,
                          (acc.z + a.z + c.z + d.z) * inv, (acc.w + a.w + c.w + d.w) * inv},
                         lse2);
        }
        if (b + gridDim.z < batch) __syncthreads();
    }
}

__global__ __launch_bounds__(128) void relative_combine_warp(Query q, const int* valid, int batch,
                                                             int width, int splits,
                                                             Partials partials) {
    const int lane         = threadIdx.x & 31;
    const int local        = blockIdx.x * kMergeWarps + threadIdx.x / 32;
    const int batch_rows   = width * kHeads;
    const std::size_t rows = std::size_t(batch) * batch_rows;
    cudaGridDependencySynchronize();
    if (local >= batch_rows) return;
    for (auto b = std::int64_t(blockIdx.z); b < batch; b += gridDim.z) {
        const auto row = b * batch_rows + local;
        const int dim  = 4 * lane;
        if (valid && local / kHeads >= valid[b]) {
            write_output(q, row, dim, {}, -CUDART_INF_F);
            continue;
        }
        const float lse     = lane < splits ? partials.lse[lane * rows + row] : -CUDART_INF_F;
        const float maximum = warp_max(lse);
        const float weight  = maximum == -CUDART_INF_F ? 0.0f : common::exp2_approx(lse - maximum);
        const float norm    = warp_sum(weight);
        float4 acc{};
        for (int i = 0; i < splits; ++i) {
            const float w = __shfl_sync(kWarp, weight, i);
            if (w == 0.0f) continue;
            const auto v =
                common::load_vec<float4>(partials.values + (i * rows + row) * kDim + dim);
            acc.x = fmaf(w, v.x, acc.x);
            acc.y = fmaf(w, v.y, acc.y);
            acc.z = fmaf(w, v.z, acc.z);
            acc.w = fmaf(w, v.w, acc.w);
        }
        const float inv  = norm > 0 ? common::rcp_approx_ftz(norm) : 0.0f;
        const float lse2 = norm > 0 ? maximum + __log2f(norm) : -CUDART_INF_F;
        write_output(q, row, dim, {acc.x * inv, acc.y * inv, acc.z * inv, acc.w * inv}, lse2);
    }
}

} // namespace

void launch_decode(Query q, CacheView cache, bool swa, KvCacheStorage storage, const int* valid,
                   DecodePlan plan, Partials partials, cudaStream_t stream) {
    if (plan.direct_swa) {
        const dim3 stripes(kHeads, kDim / kDirectSwaDecodeStripe, plan.batch);
        relative_decode_direct_swa<<<stripes, kDirectWarps * 32, 0, stream>>>(q, cache, valid);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const int depth = std::min(plan.batch, 65535);
    const dim3 grid(plan.splits, kKvHeads, depth);
    const auto launch = [&]<bool Swa, bool Fp8, bool Direct>() {
        if (!plan.simt) {
            const auto tiled = [&]<int QueryWarps>() {
                relative_decode<Swa, Fp8, Direct, false, QueryWarps>
                    <<<grid, (QueryWarps + 1) * 32, 0, stream>>>(
                        q, cache, valid, plan.batch, plan.width, plan.splits, partials, 0);
            };
            const int m = kGroup * plan.width;
            if (m <= 16)
                tiled.template operator()<1>();
            else if (m <= 32)
                tiled.template operator()<2>();
            else
                tiled.template operator()<3>();
        } else {
            if constexpr (!Swa && Fp8 && !Direct) {
                if (plan.streaming_splits) {
                    relative_decode<false, true, false, true, 1, true>
                        <<<grid, kWarps * 32, 0, stream>>>(q, cache, valid, plan.batch, 1,
                                                           plan.splits, partials,
                                                           plan.streaming_splits);
                    return;
                }
            }
            relative_decode<Swa, Fp8, Direct, true><<<grid, kWarps * 32, 0, stream>>>(
                q, cache, valid, plan.batch, plan.width, plan.splits, partials, 0);
        }
    };
    const auto format = [&]<bool Swa, bool Direct>() {
        if (storage == KvCacheStorage::BFloat16)
            launch.template operator()<Swa, false, Direct>();
        else
            launch.template operator()<Swa, true, Direct>();
    };
    if (plan.splits == 1) {
        if (swa)
            format.template operator()<true, true>();
        else
            format.template operator()<false, true>();
        CUDA_CHECK(cudaGetLastError());
    } else {
        if (swa)
            format.template operator()<true, false>();
        else
            format.template operator()<false, false>();
        CUDA_CHECK(cudaGetLastError());
        cudaLaunchAttribute dependency{};
        dependency.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        dependency.val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t combine{};
        combine.gridDim  = dim3(plan.width * kHeads, 1, depth);
        combine.blockDim = dim3(kMergeWarps * 32);
        combine.stream   = stream;
        combine.attrs    = &dependency;
        combine.numAttrs = 1;
        if (plan.splits <= 16) {
            combine.gridDim.x = (combine.gridDim.x + kMergeWarps - 1) / kMergeWarps;
            CUDA_CHECK(cudaLaunchKernelEx(&combine, relative_combine_warp, q, valid, plan.batch,
                                          plan.width, plan.splits, partials));
        } else if (plan.streaming_splits) {
            CUDA_CHECK(cudaLaunchKernelEx(&combine, relative_combine<true>, q, valid, plan.batch,
                                          plan.width, plan.splits, partials,
                                          plan.streaming_splits));
        } else {
            CUDA_CHECK(cudaLaunchKernelEx(&combine, relative_combine<false>, q, valid, plan.batch,
                                          plan.width, plan.splits, partials, 0));
        }
    }
}

} // namespace ninfer::ops::relative
