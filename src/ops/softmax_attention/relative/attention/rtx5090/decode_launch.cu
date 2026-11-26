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

template <int Features>
__device__ __forceinline__ common::F32Vector<Features> read_bf16(const std::uint16_t* p) {
    const auto words =
        common::Bf16Vector<Features>::load(reinterpret_cast<const __nv_bfloat16*>(p));
    common::F32Vector<Features> values;
#pragma unroll
    for (int f = 0; f < Features; ++f) values.word[f] = __bfloat162float(words.word[f]);
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

template <int Features>
__device__ __forceinline__ common::F32Vector<Features> read_fp16(const unsigned char* p) {
    const auto words =
        common::Bf16Vector<Features>::load(reinterpret_cast<const __nv_bfloat16*>(p));
    const auto* bits = reinterpret_cast<const unsigned short*>(&words);
    common::F32Vector<Features> result;
#pragma unroll
    for (int f = 0; f < Features; ++f) result.word[f] = __half2float(__ushort_as_half(bits[f]));
    return result;
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

struct SimtStorage {
    // Eight lanes consume one complete key and value, sharing it across all three heads.
    static constexpr int Lanes   = 8;
    static constexpr int Streams = kWarps * 32 / Lanes;
    float values[kGroup][Streams][kDim];
    float lse[kGroup][Streams];
};

template <bool Swa, bool Fp8, bool Direct, bool Chunk = false>
__device__ __forceinline__ void simt_attention(Query q, CacheView cache, int batch, int live,
                                               int kvhead, int split, int splits, Partials partials,
                                               std::size_t total_rows, SimtStorage& smem) {
    constexpr int Lanes       = SimtStorage::Lanes;
    constexpr int Streams     = SimtStorage::Streams;
    constexpr int Features    = kDim / Lanes;
    constexpr int RelFeatures = kRelDim / Lanes;
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    const int group         = threadIdx.x / Lanes;
    const int lane_in_group = threadIdx.x % Lanes;
    const int feature       = lane_in_group * Features;
    const unsigned mask     = 0xffu << (lane & ~7);
    const auto range =
        key_range<Swa>(q.positions + batch, live, split, splits, position_limit<Swa>(cache));
    const int table              = range.first < range.last ? cache_row<Swa>(cache, batch) : -1;
    const std::int64_t first_row = std::int64_t(batch) * kHeads + kvhead * kGroup;

    float acc[kGroup][Features]{};
    float maximum[kGroup], denom[kGroup]{};
#pragma unroll
    for (int h = 0; h < kGroup; ++h) maximum[h] = -CUDART_INF_F;
    if (table >= 0 && table < cache.table_count) {
        const int pos = range.high;
        const CacheRows<Swa, Fp8, Chunk> data(cache, table, kvhead, batch, pos,
                                              !Chunk || range.first * kKeyTile < pos);
        const int oldest    = Swa ? max(0, pos - (kWindow - 1)) : 0;
        const bool any_bias = range.last * std::int64_t(kKeyTile) > std::int64_t(pos) - kExtent + 1;
        common::F32Vector<Features> qr[kGroup];
        float rel[kGroup][RelFeatures]{};
#pragma unroll
        for (int h = 0; h < kGroup; ++h) {
            const auto row = first_row + h;
            qr[h]          = read_bf16<Features>(q.q + row * kDim + feature);
            if (any_bias) {
#pragma unroll
                for (int j = 0; j < RelFeatures; ++j)
                    rel[h][j] = bf16(q.r[row * kRelDim + lane_in_group * RelFeatures + j]);
            }
        }
        for (int tile = range.first; tile < range.last; ++tile) {
            const int start   = tile * kKeyTile;
            const auto record = data.record(start);
#pragma unroll 1
            for (int i = group; i < kKeyTile; i += Streams) {
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
                const auto* kk = data.codes(false, key, record) + offset;
                common::F32Vector<Features> k;
                if constexpr (Fp8) {
                    const auto exponent = *data.scale(false, key, record);
                    k                   = expand_scaled_key(common::load_vec<uint4>(kk), exponent);
                } else {
                    k = read_bf16<Features>(reinterpret_cast<const std::uint16_t*>(kk));
                }
                const float scale   = q.scale;
                const int distance  = pos - key;
                const bool has_bias = Swa || distance < kExtent;
                float proj[RelFeatures]{};
                if (has_bias) {
#pragma unroll
                    for (int j = 0; j < RelFeatures; ++j)
                        proj[j] = bf16(
                            q.projection[distance * kRelDim + lane_in_group * RelFeatures + j]);
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
                    value = read_fp16<Features>(vk);
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
        const float count = denom[h];
        if (lane_in_group == 0)
            smem.lse[h][group] = count > 0 ? maximum[h] + __log2f(count) : -CUDART_INF_F;
        if (count > 0) {
            const float inv = common::rcp_approx_ftz(count);
#pragma unroll
            for (int f = 0; f < Features; f += 4)
                common::store_vec(smem.values[h][group] + feature + f,
                                  make_float4(acc[h][f] * inv, acc[h][f + 1] * inv,
                                              acc[h][f + 2] * inv, acc[h][f + 3] * inv));
        }
    }
    __syncthreads();

    if (warp < kGroup) {
        const auto row      = first_row + warp;
        const float lse     = lane < Streams ? smem.lse[warp][lane] : -CUDART_INF_F;
        const float max_lse = warp_max(lse);
        const float weight  = max_lse > -CUDART_INF_F ? common::exp2_approx(lse - max_lse) : 0;
        const float norm    = warp_sum(weight);
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
        const float out_lse = norm > 0 ? max_lse + __log2f(norm) : -CUDART_INF_F;
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
    __syncthreads();
}

template <bool Swa, bool Fp8, bool Direct, bool Simt, int TileWarps = 1>
__global__ __launch_bounds__((Simt ? kWarps : TileWarps + 1) *
                             32) void relative_decode(Query q, CacheView cache, const int* valid,
                                                      int batch, int width, int splits,
                                                      Partials partials) {
    using Shared =
        std::conditional_t<Simt, SimtStorage,
                           TileStorage<TileWarps, Swa, Fp8, Swa ? kSwaVerifyKeyTile : kKeyTile>>;
    __shared__ Shared storage;
    if constexpr (!Direct) cudaTriggerProgrammaticLaunchCompletion();
    const std::size_t total_rows = std::size_t(batch) * width * kHeads;
    for (auto b = std::int64_t(blockIdx.z); b < batch; b += gridDim.z) {
        const int live = valid ? min(width, max(0, valid[b])) : width;
        if constexpr (Simt) {
            if constexpr (Swa) {
                // Same graph node for both W=1 forms. Select the address rule once per CTA.
                if (cache.chunk.k) {
                    simt_attention<Swa, Fp8, Direct, true>(q, cache, int(b), live, blockIdx.y,
                                                           blockIdx.x, splits, partials, total_rows,
                                                           storage);
                    continue;
                }
            }
            simt_attention<Swa, Fp8, Direct>(q, cache, int(b), live, blockIdx.y, blockIdx.x, splits,
                                             partials, total_rows, storage);
        } else {
            tiled_attention<Fp8, Swa, false, TileWarps, (TileWarps + 1) * 32, Direct,
                            Swa ? kSwaVerifyKeyTile : kKeyTile>(
                q, cache, int(b), b * width, b * width, live, width, blockIdx.y, blockIdx.x, splits,
                partials, total_rows, storage);
            __syncthreads();
        }
    }
}

__global__ __launch_bounds__(128) void relative_combine(Query q, const int* valid, int batch,
                                                        int width, int splits, Partials partials) {
    __shared__ float weights[kMaxSplits];
    __shared__ float maxima[kMergeWarps], sums[kMergeWarps];
    __shared__ float4 values[kMergeWarps][32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    const int local        = blockIdx.x;
    const int dim          = 4 * lane;
    const std::size_t rows = std::size_t(batch) * width * kHeads;
    for (auto b = std::int64_t(blockIdx.z); b < batch; b += gridDim.z) {
        const auto row = b * width * kHeads + local;
        if (valid && local / kHeads >= valid[b]) {
            if (warp == 0) write_output(q, row, dim, {}, -CUDART_INF_F);
            continue;
        }

        cudaGridDependencySynchronize();
        const float lse = tid < splits ? partials.lse[tid * rows + row] : -CUDART_INF_F;
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
        for (int i = warp; i < splits; i += kMergeWarps) {
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
        __syncthreads();
    }
}

} // namespace

void launch_decode(Query q, CacheView cache, bool swa, KvCacheStorage storage, const int* valid,
                   DecodePlan plan, Partials partials, cudaStream_t stream) {
    const int depth = std::min(plan.batch, 65535);
    const dim3 grid(plan.splits, kKvHeads, depth);
    const auto launch = [&]<bool Swa, bool Fp8, bool Direct>() {
        if (!plan.simt) {
            const auto tiled = [&]<int QueryWarps>() {
                relative_decode<Swa, Fp8, Direct, false, QueryWarps>
                    <<<grid, (QueryWarps + 1) * 32, 0, stream>>>(q, cache, valid, plan.batch,
                                                                 plan.width, plan.splits, partials);
            };
            const int m = kGroup * plan.width;
            if (m <= 16)
                tiled.template operator()<1>();
            else if (m <= 32)
                tiled.template operator()<2>();
            else
                tiled.template operator()<3>();
        } else {
            relative_decode<Swa, Fp8, Direct, true><<<grid, kWarps * 32, 0, stream>>>(
                q, cache, valid, plan.batch, plan.width, plan.splits, partials);
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
        CUDA_CHECK(cudaLaunchKernelEx(&combine, relative_combine, q, valid, plan.batch, plan.width,
                                      plan.splits, partials));
    }
}

} // namespace ninfer::ops::relative
