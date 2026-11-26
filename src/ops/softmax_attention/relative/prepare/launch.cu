#include "kernels.cuh"

#include "ops/common/cuda/mma.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/quant/fp8.cuh"

#include <algorithm>

namespace ninfer::ops::relative_prepare {
namespace {

enum class Projection { Aligned, Strided, Disabled };

using common::load_vec;
using common::store_vec;
using common::warp_sum;

constexpr int kWarp            = common::kWarpSize;
constexpr unsigned kWarpMask   = common::kFullWarpMask;
constexpr int kSinkTokens      = 32;
constexpr int kSinkK           = 64;
constexpr int kSinkStages      = 2;
constexpr int kSinkSharedBytes = (kSinkTokens + kQueryHeads) * kSinkK * kSinkStages * sizeof(Bf16);

struct Token {
    int sequence, column, length;
};

template <bool Ragged>
__device__ __forceinline__ Token locate(const Arguments& a, int token) {
    if constexpr (Ragged) {
        const int seq = a.token_to_seq[token];
        return {seq, token - a.cu_seqlens[seq], a.seq_lengths[seq]};
    } else {
        const int seq = token / a.width;
        return {seq, token - seq * a.width, a.valid ? a.valid[seq] : a.width};
    }
}

__device__ __forceinline__ float square_sum(const float (&x)[4]) {
    return warp_sum(fmaf(x[0], x[0], fmaf(x[1], x[1], fmaf(x[2], x[2], x[3] * x[3]))));
}

__device__ __forceinline__ float clip_norm(float norm, float threshold, float eps) {
    const float denom = norm + eps;
    return denom > threshold ? __fdiv_rn(threshold, denom) : 1.0f;
}

// For an underflowed square sum, nonzero FP32 components are in [2^-149, 2^-75]
// in magnitude. Shifting up by 112 makes every nonzero square normal. Shifting
// down by 112 instead leaves the square sum of an overflowed finite vector normal.
template <int N>
__device__ __forceinline__ float normalize(float (&x)[N], float sum, float threshold, float eps,
                                           bool value = false) {
    const bool tiny = sum == 0.0f;
    // When epsilon dominates, a zero FP32 sum contributes less than 2^-50 of it.
    if ((tiny || !isfinite(sum)) && !(tiny && value && eps >= 0x1p-100f)) {
        const float inverse = tiny ? 0x1p112f : 0x1p-112f;
        const float unit    = tiny ? 0x1p-112f : 0x1p112f;
        float scaled[N];
#pragma unroll
        for (int i = 0; i < N; ++i) scaled[i] = x[i] * inverse;
        float scaled_sum;
        if constexpr (N == 4)
            scaled_sum = square_sum(scaled);
        else
            scaled_sum = warp_sum(scaled[0] * scaled[0]);
        if (tiny) {
            // Every nonzero square survives this shift, so zero still means an all-zero vector.
            if (!value) return clip_norm(sqrtf(scaled_sum) * unit, threshold, eps);
            const float scaled_eps = (eps * inverse) * inverse;
            return rsqrtf(fmaf(scaled_sum, 1.0f / kDim, scaled_eps)) * inverse;
        }
        if (isfinite(scaled_sum)) {
            const float scaled_eps = eps * inverse;
            const float scaled_gain =
                value ? rsqrtf(fmaf(scaled_sum, 1.0f / kDim, scaled_eps * inverse))
                      : __fdiv_rn(threshold, sqrtf(scaled_sum) + scaled_eps);
            const float gain = scaled_gain * inverse;
            // A normal gain can be applied to the original vector, preserving small coordinates.
            if (gain >= 0x1p-126f) return value ? gain : fminf(1.0f, gain);
#pragma unroll
            for (int i = 0; i < N; ++i) x[i] = scaled[i];
            return scaled_gain;
        }
    }
    return value ? rsqrtf(fmaf(sum, 1.0f / kDim, eps)) : clip_norm(sqrtf(sum), threshold, eps);
}

__device__ __forceinline__ void unpack(uint2 raw, float (&x)[4]) {
    x[0] = __uint_as_float(raw.x << 16);
    x[1] = __uint_as_float(raw.x & 0xffff0000u);
    x[2] = __uint_as_float(raw.y << 16);
    x[3] = __uint_as_float(raw.y & 0xffff0000u);
}

__device__ __forceinline__ uint2 pack_bf16(const float (&x)[4]) {
    const auto lo = __floats2bfloat162_rn(x[0], x[1]);
    const auto hi = __floats2bfloat162_rn(x[2], x[3]);
    return make_uint2(load_vec<unsigned>(&lo), load_vec<unsigned>(&hi));
}

__device__ __forceinline__ uint2 pack_fp16(const float (&x)[4]) {
    const auto lo = __floats2half2_rn(x[0], x[1]);
    const auto hi = __floats2half2_rn(x[2], x[3]);
    return make_uint2(load_vec<unsigned>(&lo), load_vec<unsigned>(&hi));
}

__device__ __forceinline__ void prepare_query(const Arguments& a, int token, int head, int lane,
                                              bool live) {
    const auto q_index = (std::int64_t(token) * kQueryHeads + head) * kDim + lane * 4;
    const auto r_index = (std::int64_t(token) * kQueryHeads + head) * kRelDim + lane;
    if (!live) {
        store_vec(a.q_out + q_index, make_uint2(0, 0));
        if (lane < kRelDim) a.r_out[r_index] = __ushort_as_bfloat16(0);
        return;
    }

    float tau = 1.0f;
    if (a.params.use_log_scaling) {
        const auto next_pos = static_cast<unsigned>(a.positions[token]) + 1u;
        if (next_pos > static_cast<unsigned>(a.params.log_scaling_floor)) {
            const float ratio = __uint2float_rn(next_pos) / float(a.params.log_scaling_floor);
            tau += a.params.log_scaling_alpha * logf(ratio);
        }
    }
    float q[4], scale[4];
    unpack(load_vec<uint2>(a.q + q_index), q);
    unpack(load_vec<uint2>(a.q_scale + head * kDim + lane * 4), scale);
#pragma unroll
    for (int i = 0; i < 4; ++i) q[i] *= scale[i];

    float r[1]{};
    if (lane < kRelDim)
        r[0] = __bfloat162float(a.r[r_index]) * __bfloat162float(a.r_scale[head * kRelDim + lane]);
    const float r_clip =
        normalize(r, warp_sum(r[0] * r[0]), a.params.rel_clip_threshold, a.params.clip_eps);

    const float q_clip = normalize(q, square_sum(q), a.params.qk_clip_threshold, a.params.clip_eps);

#pragma unroll
    for (auto& x : q) x *= q_clip;
    auto q_bits = pack_bf16(q);
    if (tau != 1.0f) {
        unpack(q_bits, q);
#pragma unroll
        for (auto& x : q) x *= tau;
        q_bits = pack_bf16(q);
    }
    store_vec(a.q_out + q_index, q_bits);
    if (lane < kRelDim) {
        auto r_bits = __float2bfloat16_rn(r[0] * r_clip);
        if (tau != 1.0f) r_bits = __float2bfloat16_rn(__bfloat162float(r_bits) * tau);
        a.r_out[r_index] = r_bits;
    }
}

template <Destination Dest, bool Fp8>
__device__ __forceinline__ void store_cache(const Conv& conv, std::int64_t offset, int slot,
                                            std::int64_t head_row, int sequence, int lane,
                                            bool value, const float (&x)[4]) {
    if constexpr (Fp8) {
        const float local = fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fmaxf(fabsf(x[2]), fabsf(x[3])));
        // The unsigned bit ordering of finite, nonnegative FP32 values is monotonic.
        const float maximum = __uint_as_float(__reduce_max_sync(kWarpMask, __float_as_uint(local)));
        const auto scale    = common::fp8_ue8m0_scale(maximum);
        unsigned codes      = 0;
        if (maximum != 0) {
            const auto lo = common::fp8_quant_code2(x[0], x[1], scale.inverse_scale);
            const auto hi = common::fp8_quant_code2(x[2], x[3], scale.inverse_scale);
            codes         = unsigned(lo) | (unsigned(hi) << 16);
        }
        store_vec(conv.cache.base + offset + (head_row + slot) * kDim + lane * 4, codes);
        if (lane == 0) {
            auto scale_offset = offset;
            if constexpr (Dest == Destination::Cyclic)
                if (conv.scales.offsets) scale_offset = conv.scales.offsets[sequence];
            reinterpret_cast<std::uint8_t*>(conv.scales.base)[scale_offset + head_row + slot] =
                scale.scale;
        }
    } else {
        store_vec(conv.cache.base + offset + ((head_row + slot) * kDim + lane * 4) * 2,
                  value ? pack_fp16(x) : pack_bf16(x));
    }
}

__device__ __forceinline__ void load_weights(const Conv& conv, int channel,
                                             uint2 (&weights)[kTaps]) {
#pragma unroll
    for (int tap = 0; tap < kTaps; ++tap)
        weights[tap] = load_vec<uint2>(conv.weight + tap * kChannels + channel);
}

template <Destination Dest, bool Fp8, bool Dynamic>
__device__ __forceinline__ void
convolve_to_cache(const Arguments& a, const Conv& conv, int sequence, int token, int head, int lane,
                  bool value, const uint2 (&raw)[kTaps], const uint2 (&weights)[kTaps]) {
    float dyn[kTaps]{};
    if constexpr (Dynamic)
        unpack(load_vec<uint2>(conv.dynamic + (std::int64_t(token) * kKvHeads + head) * kTaps),
               dyn);

    float sum[4]{};
    float x[4], w[4];
#pragma unroll
    for (int tap = 0; tap < kTaps; ++tap) {
        const int lag = Dynamic ? tap : kHistory - tap;
        unpack(raw[lag], x);
        unpack(weights[tap], w);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if constexpr (Dynamic) w[i] += dyn[tap];
            sum[i] = fmaf(x[i], w[i], sum[i]);
        }
    }
    unpack(raw[0], x);
    constexpr float conv_scale = Dynamic ? 1.0f / 3.0f : 0.44721359549995793928f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float conv = (sum[i] + x[i]) * conv_scale;
        // The residual addition can overflow while its scaled result is finite.
        if (!isfinite(conv)) conv = fmaf(sum[i], conv_scale, x[i] * conv_scale);
        x[i] = __bfloat162float(__float2bfloat16_rn(conv));
    }
    const float sum_sq = square_sum(x);
    const float factor = normalize(x, sum_sq, a.params.qk_clip_threshold,
                                   value ? a.params.v_norm_eps : a.params.clip_eps, value);
#pragma unroll
    for (auto& f : x) f *= factor;

    int slot            = 0;
    std::int64_t offset = 0, head_row;
    if constexpr (Dest == Destination::Chunk) {
        head_row = std::int64_t(token) * kKvHeads + head;
    } else if constexpr (Dest == Destination::Paged) {
        const int pos  = a.positions[token];
        const auto row = a.table_rows[sequence];
        offset   = conv.cache.offsets[std::int64_t(row) * a.cache_rows + pos / kPagedKVPageSize];
        slot     = pos % kPagedKVPageSize;
        head_row = head * kPagedKVPageSize;
    } else {
        offset   = conv.cache.offsets[sequence];
        slot     = a.positions[token] & (kCyclicCapacity - 1);
        head_row = std::int64_t(head) * a.cache_rows;
    }
    store_cache<Dest, Fp8>(conv, offset, slot, head_row, sequence, lane, value, x);
}

template <Destination Dest, bool Fp8, bool Dynamic, bool Snapshot, bool HasHistory>
__device__ __forceinline__ void prepare_kv(const Arguments& a, const Conv& conv, Token t, int token,
                                           int head, int lane, bool value, const Bf16* before) {
    const int channel   = head * kDim + lane * 4;
    const auto* current = conv.input + std::int64_t(token) * kChannels + channel;
    uint2 raw[kTaps];
#pragma unroll
    for (int lag = 0; lag < kTaps; ++lag) {
        if constexpr (HasHistory) {
            if (t.column < lag) {
                raw[lag] = load_vec<uint2>(before + (kHistory + t.column - lag) * kChannels);
                continue;
            }
        }
        raw[lag] = load_vec<uint2>(current - lag * kChannels);
    }

    if constexpr (Snapshot) {
        const auto stride = value ? a.snapshot_stride_v : a.snapshot_stride_k;
        const auto slot   = std::int64_t(a.snapshot_slots[t.sequence]) + t.column;
        auto* dest =
            reinterpret_cast<Bf16*>(reinterpret_cast<char*>(conv.after.base) + slot * stride) +
            channel;
#pragma unroll
        for (int j = 0; j < kHistory; ++j) store_vec(dest + j * kChannels, raw[kHistory - 1 - j]);
    }

    uint2 weights[kTaps];
    load_weights(conv, channel, weights);
    convolve_to_cache<Dest, Fp8, Dynamic>(a, conv, t.sequence, token, head, lane, value, raw,
                                          weights);
}

template <Destination Dest, bool Fp8, bool Dynamic, bool Snapshot>
__device__ __forceinline__ void prepare_conv(const Arguments& a, Token t, int token, int head,
                                             int lane, bool value) {
    const Conv& conv = value ? a.v : a.k;
    if (t.column >= kHistory) {
        prepare_kv<Dest, Fp8, Dynamic, Snapshot, false>(a, conv, t, token, head, lane, value,
                                                        nullptr);
    } else if constexpr (Snapshot) {
        const auto* before = reinterpret_cast<const Bf16*>(
            reinterpret_cast<const char*>(conv.before.base) + conv.before.offsets[t.sequence]);
        prepare_kv<Dest, Fp8, Dynamic, Snapshot, true>(a, conv, t, token, head, lane, value,
                                                       before + head * kDim + lane * 4);
    } else if (t.column == 0) {
        const int channel = head * kDim + lane * 4;
        const auto* before =
            reinterpret_cast<const Bf16*>(reinterpret_cast<const char*>(conv.before.base) +
                                          conv.before.offsets[t.sequence]) +
            channel;
        // This warp owns every read of its head's history and the final history update.
        // Later columns only consume raw input, so direct in-place state needs no grid barrier.
        const int prefix = min(t.length, kHistory);
#pragma unroll 1
        for (int j = 0; j < prefix; ++j)
            prepare_kv<Dest, Fp8, Dynamic, Snapshot, true>(a, conv, {t.sequence, j, t.length},
                                                           token + j, head, lane, value, before);

        uint2 tail[kHistory];
#pragma unroll
        for (int j = 0; j < kHistory; ++j) {
            const int column = t.length - kHistory + j;
            const auto* src =
                column >= 0 ? conv.input + (std::int64_t(token) + column) * kChannels + channel
                            : before + (column + kHistory) * kChannels;
            tail[j] = load_vec<uint2>(src);
        }
        auto* after = reinterpret_cast<Bf16*>(reinterpret_cast<char*>(conv.after.base) +
                                              conv.after.offsets[t.sequence]) +
                      channel;
#pragma unroll
        for (int j = 0; j < kHistory; ++j) store_vec(after + j * kChannels, tail[j]);
    }
}

template <Destination Dest, bool Fp8, bool Dynamic>
__device__ __forceinline__ void prepare_ragged_pair(const Arguments& a, int token, int head,
                                                    int lane, bool value) {
    const auto t       = locate<true>(a, token);
    const bool regular = t.column >= kHistory && t.column + 1 < t.length;
    if (!regular) {
        prepare_conv<Dest, Fp8, Dynamic, false>(a, t, token, head, lane, value);
        if (token + 1 < a.columns)
            prepare_conv<Dest, Fp8, Dynamic, false>(a, locate<true>(a, token + 1), token + 1, head,
                                                    lane, value);
        return;
    }

    // Adjacent interior tokens share three raw taps and all four weight vectors.
    const auto& conv  = value ? a.v : a.k;
    const int channel = head * kDim + lane * 4;
    const auto* input = conv.input + std::int64_t(token) * kChannels + channel;
    uint2 raw[kTaps], weights[kTaps];
    load_weights(conv, channel, weights);
#pragma unroll
    for (int lag = 0; lag < kTaps; ++lag) raw[lag] = load_vec<uint2>(input - lag * kChannels);
#pragma unroll 1
    for (int j = 0; j < 2; ++j) {
        convolve_to_cache<Dest, Fp8, Dynamic>(a, conv, t.sequence, token + j, head, lane, value,
                                              raw, weights);
        if (j == 0) {
#pragma unroll
            for (int lag = kHistory; lag > 0; --lag) raw[lag] = raw[lag - 1];
            raw[0] = load_vec<uint2>(input + kChannels);
        }
    }
}

__device__ __forceinline__ void project_sink(const Arguments& a, int token, int head, int lane,
                                             bool live) {
    float sum = 0;
    if (live) {
        const auto* input  = a.sink_x + std::int64_t(token) * a.sink_channels;
        const auto* weight = a.sink_weight + std::int64_t(head) * a.sink_channels;
        if (a.sink_channels % 8 != 0) {
            for (std::int64_t i = lane; i < a.sink_channels; i += kWarp)
                sum = fmaf(__bfloat162float(input[i]), __bfloat162float(weight[i]), sum);
            sum = warp_sum(sum);
        } else {
            float partial[4]{};
            for (std::int64_t i = lane * 8; i < a.sink_channels; i += kWarp * 8) {
                const auto x = load_vec<uint4>(input + i);
                const auto w = load_vec<uint4>(weight + i);
                float x0[4], x1[4], w0[4], w1[4];
                unpack(make_uint2(x.x, x.y), x0);
                unpack(make_uint2(x.z, x.w), x1);
                unpack(make_uint2(w.x, w.y), w0);
                unpack(make_uint2(w.z, w.w), w1);
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    partial[j] = fmaf(x1[j], w1[j], fmaf(x0[j], w0[j], partial[j]));
            }
            sum = warp_sum((partial[0] + partial[1]) + (partial[2] + partial[3]));
        }
    }
    if (lane == 0) a.sink_out[std::int64_t(token) * kQueryHeads + head] = __float2bfloat16_rn(sum);
}

template <bool Ragged>
__device__ __forceinline__ bool sink_live(const Arguments& a, std::int64_t token) {
    if (token >= a.columns) return false;
    if constexpr (Ragged) return true;
    const auto t = locate<false>(a, static_cast<int>(token));
    return t.column < t.length;
}

template <bool Ragged, int Warps, Projection Sink>
__device__ __forceinline__ void project_sink_tile(const Arguments& a, std::int64_t tile, int warp,
                                                  int lane, Bf16* shared) {
    constexpr int Rows          = kSinkTokens + kQueryHeads;
    const auto token_begin      = tile * kSinkTokens;
    const std::int64_t channels = a.sink_channels;
    const int stages            = (channels + kSinkK - 1) / kSinkK;
    constexpr int Copies        = (Rows * kSinkK / 8 + Warps * kWarp - 1) / (Warps * kWarp);

    struct TileCopy {
        const Bf16* global;
        unsigned shared;
        int channel;
        bool live;
    };

    TileCopy copies[Copies];
#pragma unroll
    for (int j = 0; j < Copies; ++j) {
        const int i   = threadIdx.x + j * Warps * kWarp;
        const int row = i / (kSinkK / 8), channel = i % (kSinkK / 8) * 8;
        const bool input = row < kSinkTokens;
        const auto* base = input ? a.sink_x : a.sink_weight;
        const auto index = input ? token_begin + row : row - kSinkTokens;
        const bool live  = row < Rows && (!input || sink_live<Ragged>(a, index));
        copies[j]        = {
            live ? base + index * channels : base,
            common::smem_addr(shared + row * kSinkK + common::swizzle_xor_64(row, channel)),
            channel, live};
    }
    auto stage = [&](int step) {
        const unsigned shared_stage = (step % kSinkStages) * Rows * kSinkK * sizeof(Bf16);
        const auto start            = std::int64_t(step) * kSinkK;
#pragma unroll
        for (int j = 0; j < Copies; ++j) {
            const int i = threadIdx.x + j * Warps * kWarp;
            if (i >= Rows * kSinkK / 8) continue;
            const auto& copy   = copies[j];
            const auto channel = start + copy.channel;
            const bool live    = copy.live && channel < channels;
            const auto* src    = live ? copy.global + channel : copy.global;
            if constexpr (Sink == Projection::Aligned) {
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                             :
                             : "r"(copy.shared + shared_stage), "l"(src), "r"(live ? 16 : 0));
            } else {
                auto dst = copy.shared + shared_stage;
#pragma unroll 1
                for (int d = 0; d < 8; ++d, dst += sizeof(Bf16)) {
                    const auto word = live && channel + d < channels ? __bfloat16_as_ushort(src[d])
                                                                     : std::uint16_t{0};
                    asm volatile("st.shared.b16 [%0], %1;" : : "r"(dst), "h"(word) : "memory");
                }
            }
        }
        common::cp_commit();
    };

    constexpr int HeadTiles = kQueryHeads / 8;
    const int token_warp    = warp / HeadTiles * 16;
    const int head_warp     = warp % HeadTiles * 8;
    const bool compute      = warp < kSinkTokens / 16 * HeadTiles;
    float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    stage(0);
    if (stages > 1) stage(1);
    for (int step = 0; step < stages; ++step) {
        if (step + 1 < stages)
            common::cp_wait<1>();
        else
            common::cp_wait<0>();
        __syncthreads();
        if (compute) {
            const auto* buf = shared + (step % kSinkStages) * Rows * kSinkK;
            const int ar    = token_warp + (lane & 7) + ((lane & 8) ? 8 : 0);
            const int br    = kSinkTokens + head_warp + (lane & 7);
#pragma unroll
            for (int k = 0; k < kSinkK; k += 16) {
                unsigned av[4], bv[2];
                const int ak = k + ((lane & 16) ? 8 : 0);
                const int bk = k + ((lane & 8) ? 8 : 0);
                common::ldmatrix_x4(
                    av[0], av[1], av[2], av[3],
                    common::smem_addr(buf + ar * kSinkK + common::swizzle_xor_64(ar, ak)));
                common::ldmatrix_x2(
                    bv[0], bv[1],
                    common::smem_addr(buf + br * kSinkK + common::swizzle_xor_64(br, bk)));
                common::mma_bf16(c0, c1, c2, c3, av[0], av[1], av[2], av[3], bv[0], bv[1]);
            }
        }
        __syncthreads();
        if (step + kSinkStages < stages) stage(step + kSinkStages);
    }
    if (compute) {
        const auto token = token_begin + token_warp + lane / 4;
        const int head   = head_warp + lane % 4 * 2;
        const auto lo    = __floats2bfloat162_rn(c0, c1);
        const auto hi    = __floats2bfloat162_rn(c2, c3);
        if (token < a.columns)
            store_vec(a.sink_out + token * kQueryHeads + head,
                      sink_live<Ragged>(a, token) ? load_vec<unsigned>(&lo) : 0u);
        if (token + 8 < a.columns)
            store_vec(a.sink_out + (token + 8) * kQueryHeads + head,
                      sink_live<Ragged>(a, token + 8) ? load_vec<unsigned>(&hi) : 0u);
    }
}

template <Destination Dest, bool Ragged, int Warps, Projection Sink>
struct PreparationTile {
    // Pairs reuse input taps. Dense sink-free chunk stores are faster with one token per CTA.
    static constexpr bool paired      = Ragged && Warps == 8 && Sink != Projection::Disabled;
    static constexpr int columns      = paired ? 2 : 1;
    static constexpr int query_groups = kQueryHeads / Warps * columns;
    static constexpr int groups       = query_groups + 2 * kKvHeads / Warps;
};

// The paged pair needs registers for two cache addresses and shared tap values.
template <Destination Dest, bool Dynamic, bool Ragged, int Warps, Projection Sink>
constexpr int min_resident_blocks() {
    if constexpr (Warps == 4) return 8;
    if constexpr (Sink == Projection::Strided) return 3;
    if constexpr (Dest == Destination::Cyclic) return 5;
    // With only contiguous stores to issue, the sink-free chunk kernel benefits from more warps.
    if constexpr (Sink == Projection::Disabled) return 5;
    // Static pairs keep taps and weights live across both normalizations.
    if constexpr (Ragged && !Dynamic) return 3;
    return 4;
}

template <Destination Dest, bool Fp8, bool Dynamic, bool Ragged, bool Snapshot, int Warps,
          Projection Sink>
__global__ __launch_bounds__(
    Warps* kWarp,
    (min_resident_blocks<Dest, Dynamic, Ragged, Warps,
                         Sink>())) void prepare_kernel(const __grid_constant__ Arguments a,
                                                       std::int64_t sink_blocks,
                                                       std::int64_t blocks) {
    using Tile     = PreparationTile<Dest, Ragged, Warps, Sink>;
    const int lane = threadIdx.x % kWarp;
    const int warp = threadIdx.x / kWarp;
    for (auto job = std::int64_t(blockIdx.x); job < blocks; job += gridDim.x) {
        if constexpr (Sink != Projection::Disabled) {
            if (job < sink_blocks) {
                if constexpr (Warps == 8) {
                    extern __shared__ __align__(16) Bf16 shared[];
                    project_sink_tile<Ragged, Warps, Sink>(a, job, warp, lane, shared);
                } else {
                    constexpr int groups = kQueryHeads / Warps;
                    const int token      = job / groups;
                    project_sink(a, token, (job % groups) * Warps + warp, lane,
                                 sink_live<Ragged>(a, token));
                }
                continue;
            }
        }
        const auto prep_job = job - sink_blocks;
        const int group     = prep_job % Tile::groups;
        const int tile      = prep_job / Tile::groups;
        const int token     = tile * Tile::columns;
        if (group < Tile::query_groups) {
            const int q_token = token + (Tile::paired ? group / (kQueryHeads / Warps) : 0);
            const int head    = group * Warps % kQueryHeads + warp;
            bool live         = true;
            if constexpr (!Ragged) {
                const auto t = locate<false>(a, q_token);
                live         = t.column < t.length;
            }
            if (q_token < a.columns) prepare_query(a, q_token, head, lane, live);
        } else {
            const int kv_head = (group - Tile::query_groups) * Warps + warp;
            if constexpr (Tile::paired) {
                prepare_ragged_pair<Dest, Fp8, Dynamic>(a, token, kv_head % kKvHeads, lane,
                                                        kv_head >= kKvHeads);
            } else {
                const auto t = locate<Ragged>(a, token);
                if (t.column < t.length)
                    prepare_conv<Dest, Fp8, Dynamic, Snapshot>(a, t, token, kv_head % kKvHeads,
                                                               lane, kv_head >= kKvHeads);
            }
        }
    }
}

template <Destination Dest, bool Fp8, bool Dynamic, bool Ragged, bool Snapshot, int Warps,
          Projection Sink = Projection::Aligned>
void submit(const Arguments& a, CudaExecutionView execution) {
    using Tile         = PreparationTile<Dest, Ragged, Warps, Sink>;
    const auto columns = std::int64_t(a.columns);
    const auto prep    = (columns + Tile::columns - 1) / Tile::columns * Tile::groups;
    const auto sink    = !a.sink_channels ? 0
                         : Warps == 8     ? (columns + kSinkTokens - 1) / kSinkTokens
                                          : columns * (kQueryHeads / Warps);
    const auto blocks  = prep + sink;
    const auto grid    = static_cast<unsigned>(std::min<std::int64_t>(blocks, INT32_MAX));
    const int shared   = Warps == 8 && sink ? kSinkSharedBytes : 0;
    prepare_kernel<Dest, Fp8, Dynamic, Ragged, Snapshot, Warps, Sink>
        <<<grid, Warps * kWarp, shared, execution.stream()>>>(a, sink, blocks);
}

template <Destination Dest, bool Fp8, bool Dynamic, bool Ragged, bool Snapshot>
void submit_extent(const Arguments& a, CudaExecutionView execution) {
    // Four-warp CTAs expose more parallelism on small/medium grids (especially dynamic conv
    // + sink projection where 8-warp sink tiles under-occupy SMs below 512 tokens).
    if (a.columns < 512) {
        submit<Dest, Fp8, Dynamic, Ragged, Snapshot, 4>(a, execution);
        return;
    }
    if constexpr (Dest == Destination::Chunk && Ragged) {
        if (!a.sink_channels) {
            submit<Dest, Fp8, Dynamic, Ragged, Snapshot, 8, Projection::Disabled>(a, execution);
            return;
        }
    }
    if (a.sink_channels % 8 == 0) {
        submit<Dest, Fp8, Dynamic, Ragged, Snapshot, 8>(a, execution);
    } else {
        submit<Dest, Fp8, Dynamic, Ragged, Snapshot, 8, Projection::Strided>(a, execution);
    }
}

template <Destination Dest, bool Fp8, bool Dynamic>
void submit_layout(const Arguments& a, CudaExecutionView execution) {
    if constexpr (Dest != Destination::Cyclic) {
        if (a.token_to_seq) {
            submit_extent<Dest, Fp8, Dynamic, true, false>(a, execution);
            return;
        }
        if (a.snapshot_slots) {
            submit_extent<Dest, Fp8, Dynamic, false, true>(a, execution);
            return;
        }
    }
    submit_extent<Dest, Fp8, Dynamic, false, false>(a, execution);
}

template <Destination Dest, bool Fp8>
void submit_conv(const Arguments& a, CudaExecutionView execution) {
    if (a.k.dynamic)
        submit_layout<Dest, Fp8, true>(a, execution);
    else
        submit_layout<Dest, Fp8, false>(a, execution);
}

// Copy one 16-byte slice per lane; FP8 uses half the lanes for each plane.
template <bool Fp8>
__device__ __forceinline__ void commit_plane(const CommitPlane& plane, int token, int lane,
                                             int sequence, int slot, int rows) {
    constexpr int VectorBytes  = Fp8 ? kDim : 2 * kDim;
    constexpr int UnitsPerHead = VectorBytes / sizeof(uint4);
    const int head             = lane / UnitsPerHead;
    const auto cache_vector    = std::int64_t(head) * rows + slot;
    const auto chunk_vector    = std::int64_t(token) * kKvHeads + head;
    const int feature          = (lane % UnitsPerHead) * sizeof(uint4);
    const auto offset          = plane.cache.offsets[sequence];
    store_vec(plane.cache.base + offset + cache_vector * VectorBytes + feature,
              load_vec<uint4>(plane.data + chunk_vector * VectorBytes + feature));
    if constexpr (Fp8) {
        if (feature == 0) {
            const auto scale_offset =
                plane.cache_scales.offsets ? plane.cache_scales.offsets[sequence] : offset;
            plane.cache_scales.base[scale_offset + cache_vector] = plane.scales[chunk_vector];
        }
    }
}

template <bool Ragged, bool Fp8>
__global__ void commit_kernel(const __grid_constant__ CommitArguments a) {
    const int tail_idx = blockIdx.x;
    const int lane     = threadIdx.x;
    unsigned seq       = blockIdx.y;
    // grid.y never exceeds the sequence count. The stride is at most 65535,
    // so even the final, out-of-range addition fits an unsigned index.
    do {
        const int sequence = static_cast<int>(seq);
        int length, end;
        if constexpr (Ragged) {
            length = a.seq_lengths[sequence];
            end    = a.cu_seqlens[sequence + 1];
        } else {
            length = a.valid ? a.valid[sequence] : a.width;
            end    = sequence * a.width + length;
        }
        const int surviving = min(length, kCyclicCapacity);
        if (tail_idx < surviving) {
            const int token = end - surviving + tail_idx;
            const int slot  = a.positions[token] & (kCyclicCapacity - 1);
            if constexpr (Fp8) {
                constexpr int PlaneThreads = kKvHeads * kDim / sizeof(uint4);
                const auto& plane          = lane < PlaneThreads ? a.k : a.v;
                commit_plane<true>(plane, token, lane % PlaneThreads, sequence, slot, a.cache_rows);
            } else {
                commit_plane<false>(a.k, token, lane, sequence, slot, a.cache_rows);
                commit_plane<false>(a.v, token, lane, sequence, slot, a.cache_rows);
            }
        }
    } while ((seq += gridDim.y) < static_cast<unsigned>(a.sequences));
}

} // namespace

void launch(const Arguments& args, Destination dest, bool fp8, CudaExecutionView execution) {
    switch (dest) {
    case Destination::Cyclic:
        if (fp8)
            submit_conv<Destination::Cyclic, true>(args, execution);
        else
            submit_conv<Destination::Cyclic, false>(args, execution);
        break;
    case Destination::Chunk:
        if (fp8)
            submit_conv<Destination::Chunk, true>(args, execution);
        else
            submit_conv<Destination::Chunk, false>(args, execution);
        break;
    case Destination::Paged:
        if (fp8)
            submit_conv<Destination::Paged, true>(args, execution);
        else
            submit_conv<Destination::Paged, false>(args, execution);
        break;
    }
    CUDA_CHECK(cudaGetLastError());
}

void launch_commit(const CommitArguments& args, bool fp8, CudaExecutionView execution) {
    constexpr int Threads = 2 * kKvHeads * kDim / sizeof(uint4);
    const dim3 grid(args.tail_slots, std::min(args.sequences, 65535));
    if (args.seq_lengths) {
        if (fp8)
            commit_kernel<true, true><<<grid, Threads, 0, execution.stream()>>>(args);
        else
            commit_kernel<true, false><<<grid, Threads, 0, execution.stream()>>>(args);
    } else {
        if (fp8)
            commit_kernel<false, true><<<grid, Threads, 0, execution.stream()>>>(args);
        else
            commit_kernel<false, false><<<grid, Threads, 0, execution.stream()>>>(args);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::relative_prepare
