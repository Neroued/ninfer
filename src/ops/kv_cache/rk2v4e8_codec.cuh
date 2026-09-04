#pragma once

// rk2v4-e8 asymmetric KV codec.
//
//   * Key: the fixed normalized D256 Hadamard rotation followed by the E8-root "cylinder"
//     encoder. Every 8 rotated dimensions collapse to one (root, log-radius|axis) byte pair,
//     so the persistent key plane is quarter width (64 bytes / token / head). Decode expands
//     a pair back to eight signed int8 codes that share the per-64-group FP16 scale, which
//     keeps the INT8 m16n8k32 QK path of the G64 codec unchanged.
//   * Value: a per-64-group Hadamard rotation followed by packed int4 codes at half width
//     (128 bytes / token / head), also with a per-64-group FP16 scale. Because V is encoded
//     in the rotated basis, PV accumulates in rotated-V coordinates and the attention output
//     must be inverse-rotated (the 64-dim Hadamard is its own inverse) before it leaves the
//     kernel family. rk2v4e8_inverse_rotate_output_kernel owns that step.
//
// 64 + 8 + 128 + 8 = 208 B / token / head at head_dim 256.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/e8_lattice.cuh"
#include "ops/kernel/e8_root_codec.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kRk2v4E8HeadDim    = 256;
inline constexpr int kRk2v4E8Group      = 64;
inline constexpr int kRk2v4E8Groups     = kRk2v4E8HeadDim / kRk2v4E8Group;
inline constexpr int kRk2v4E8KCodeWidth = kRk2v4E8HeadDim / 4;
inline constexpr int kRk2v4E8VCodeWidth = kRk2v4E8HeadDim / 2;

static_assert(kRk2v4E8Groups == 4);
static_assert(kRk2v4E8KCodeWidth == 64);
static_assert(kRk2v4E8VCodeWidth == 128);

// Byte offset of the (root, rad|axis) pair covering dimension d. Every 8 dimensions occupy
// two consecutive bytes, so the pair for d starts at 2 * (d / 8).
template <typename Geometry>
__device__ __forceinline__ std::int64_t rk2v4e8_k_code_index(int physical_page, int kv_head, int d,
                                                             int page_offset) {
    return paged_kv_element_offset<kRk2v4E8KCodeWidth, Geometry::KVHeads>(physical_page, kv_head,
                                                                          page_offset, (d >> 3) * 2);
}

// Byte offset of the packed int4 pair covering dimensions (d, d+1) for even d.
template <typename Geometry>
__device__ __forceinline__ std::int64_t rk2v4e8_v_code_index(int physical_page, int kv_head, int d,
                                                             int page_offset) {
    return paged_kv_element_offset<kRk2v4E8VCodeWidth, Geometry::KVHeads>(physical_page, kv_head,
                                                                          page_offset, d >> 1);
}

// Both scale planes carry one FP16 per 64-dimension group; K and V use the same extent.
template <typename Geometry>
__device__ __forceinline__ std::int64_t rk2v4e8_scale_index(int physical_page, int kv_head,
                                                            int group, int page_offset) {
    return paged_kv_element_offset<kRk2v4E8Groups, Geometry::KVHeads>(physical_page, kv_head,
                                                                      page_offset, group);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t rk2v4e8_src_index(int kv_head, int d, int token) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kRk2v4E8HeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(Geometry::KVHeads) * token);
}

// 64-dimensional normalized Hadamard for the int4 Value basis. Lane l carries dimensions
// (l, l + 32) of one group, so the five XOR butterflies plus the final H2 leaf realize the
// natural-order H64 scaled by 1/8. Orthogonal and symmetric, hence its own inverse.
__device__ __forceinline__ void rk2v4e8_hadamard64(float& x0, float& x1, int lane,
                                                   unsigned mask = 0xffffffffu) {
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const float y0 = __shfl_xor_sync(mask, x0, offset);
        const float y1 = __shfl_xor_sync(mask, x1, offset);
        const bool high = (lane & offset) != 0;
        x0              = high ? y0 - x0 : x0 + y0;
        x1              = high ? y1 - x1 : x1 + y1;
    }
    const float a = x0;
    const float b = x1;
    x0            = (a + b) * 0.125f;
    x1            = (a - b) * 0.125f;
}

// Quantize one rotated value to a signed int4 code in [-7, 7]; int4's symmetric range is < 8,
// so the group scale uses a /7 denominator rather than the int8 /127.
__device__ __forceinline__ std::int8_t rk2v4e8_quant_i4_code(float x, float inv_scale) {
    if (inv_scale == 0.0f) { return static_cast<std::int8_t>(0); }
    int q = __float2int_rn(x * inv_scale);
    q     = max(-7, min(7, q));
    return static_cast<std::int8_t>(q);
}

__device__ __forceinline__ std::uint8_t rk2v4e8_pack_i4(std::int8_t lo, std::int8_t hi) {
    return static_cast<std::uint8_t>((static_cast<unsigned>(lo) & 0x0fu) |
                                     ((static_cast<unsigned>(hi) & 0x0fu) << 4));
}

// Unpack one nibble to a SIGNED int8 in [-8, 7] via XOR-carry sign extension. Nibbles are
// stored unsigned (0..15); (nibble ^ 8) - 8 maps them back to -8..7. Without this a
// high-valence code reads back non-negative and V quality collapses.
__device__ __forceinline__ std::int8_t rk2v4e8_unpack_i4(std::uint8_t packed, bool high) {
    const unsigned nibble = high ? (packed >> 4) : (packed & 0x0fu);
    return static_cast<std::int8_t>(static_cast<int>(nibble ^ 8u) - 8);
}

// Expand the four code bytes covering 16 consecutive (16-aligned) dimensions into int8 codes.
// `codes4` is 4-byte aligned by construction; `out` must be 8-byte aligned.
__device__ __forceinline__ void rk2v4e8_decode_k_16d(const std::uint8_t* codes4, std::int8_t* out) {
    const std::uint32_t src = *reinterpret_cast<const std::uint32_t*>(codes4);
    e8_root_decode_8d_int8(static_cast<std::uint8_t>(src & 0xffu),
                           static_cast<std::uint8_t>((src >> 8) & 0xffu), out);
    e8_root_decode_8d_int8(static_cast<std::uint8_t>((src >> 16) & 0xffu),
                           static_cast<std::uint8_t>((src >> 24) & 0xffu), out + 8);
}

// Expand the eight packed bytes covering 16 consecutive (16-aligned) dimensions into int8 codes.
__device__ __forceinline__ void rk2v4e8_unpack_v_16d(const std::uint8_t* packed8, std::int8_t* out) {
    const int2 raw              = load_vec<int2>(packed8);
    const std::uint8_t* packed  = reinterpret_cast<const std::uint8_t*>(&raw);
#pragma unroll
    for (int e = 0; e < 16; e += 2) {
        const std::uint8_t byte = packed[e >> 1];
        out[e]                  = rk2v4e8_unpack_i4(byte, false);
        out[e + 1]              = rk2v4e8_unpack_i4(byte, true);
    }
}

// Encode one (token, kv_head) K and V row. One full warp owns the row: lane l carries
// dimensions l + 32r in values[r], so each 8-lane subgroup of the warp holds exactly the eight
// consecutive dimensions the E8 encoder contracts over. Every lane must reach both encoder
// calls - the encoder is warp-collective over the full mask and diverging there is undefined.
// Writes each of the four K and four V group scales exactly once, from lane 0.
template <typename Geometry>
__device__ __forceinline__ void
rk2v4e8_append_row(const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
                   std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
                   __half* __restrict__ scale_k, __half* __restrict__ scale_v, int token,
                   int kv_head, int physical_page, int page_offset, int lane) {
    constexpr unsigned FullMask = 0xffffffffu;

    float values[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        values[r] =
            __bfloat162float(k[rk2v4e8_src_index<Geometry>(kv_head, lane + 32 * r, token)]);
    }
    normalized_hadamard_d256_inplace(values, lane);

    float k_scale[kRk2v4E8Groups];
#pragma unroll
    for (int g = 0; g < kRk2v4E8Groups; ++g) {
        const float absmax =
            warp_max(fmaxf(fabsf(values[2 * g]), fabsf(values[2 * g + 1])), FullMask);
        k_scale[g] = __half2float(__float2half_rn(absmax > 0.0f ? absmax / 7.0f : 0.0f));
    }

    const std::int64_t k_base =
        paged_kv_page_head_offset<kRk2v4E8KCodeWidth, Geometry::KVHeads>(physical_page, kv_head) +
        static_cast<std::int64_t>(page_offset) * kRk2v4E8KCodeWidth;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        std::uint8_t root = 0;
        std::uint8_t axis = 0;
        e8_encode_cylinder_8d_warp(values[r], k_scale[r >> 1], root, axis, lane);
        if ((lane & 7) == 0) {
            // Dimension lane + 32r sits in 8-dim block 4r + (lane >> 3); its pair starts at
            // twice that index.
            const int byte             = 8 * r + 2 * (lane >> 3);
            cache_k[k_base + byte]     = root;
            cache_k[k_base + byte + 1] = axis;
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int g = 0; g < kRk2v4E8Groups; ++g) {
            scale_k[rk2v4e8_scale_index<Geometry>(physical_page, kv_head, g, page_offset)] =
                __float2half_rn(k_scale[g]);
        }
    }

#pragma unroll
    for (int r = 0; r < 8; ++r) {
        values[r] =
            __bfloat162float(v[rk2v4e8_src_index<Geometry>(kv_head, lane + 32 * r, token)]);
    }
    // Rotate before the group absmax so the stored scale describes the encoded magnitudes.
#pragma unroll
    for (int g = 0; g < kRk2v4E8Groups; ++g) {
        rk2v4e8_hadamard64(values[2 * g], values[2 * g + 1], lane, FullMask);
    }
    float v_scale[kRk2v4E8Groups];
    float v_inverse[kRk2v4E8Groups];
#pragma unroll
    for (int g = 0; g < kRk2v4E8Groups; ++g) {
        const float absmax =
            warp_max(fmaxf(fabsf(values[2 * g]), fabsf(values[2 * g + 1])), FullMask);
        v_scale[g]   = __half2float(__float2half_rn(absmax > 0.0f ? absmax / 7.0f : 0.0f));
        v_inverse[g] = v_scale[g] > 0.0f ? 1.0f / v_scale[g] : 0.0f;
    }

    const std::int64_t v_base =
        paged_kv_page_head_offset<kRk2v4E8VCodeWidth, Geometry::KVHeads>(physical_page, kv_head) +
        static_cast<std::int64_t>(page_offset) * kRk2v4E8VCodeWidth;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        // Dimensions (d, d+1) live in adjacent lanes of the same register slot; the even lane
        // of each pair writes the packed byte.
        const float odd = __shfl_down_sync(FullMask, values[r], 1);
        if ((lane & 1) == 0) {
            const int d = lane + 32 * r;
            cache_v[v_base + (d >> 1)] =
                rk2v4e8_pack_i4(rk2v4e8_quant_i4_code(values[r], v_inverse[r >> 1]),
                                rk2v4e8_quant_i4_code(odd, v_inverse[r >> 1]));
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int g = 0; g < kRk2v4E8Groups; ++g) {
            scale_v[rk2v4e8_scale_index<Geometry>(physical_page, kv_head, g, page_offset)] =
                __float2half_rn(v_scale[g]);
        }
    }
}

// Inverse-rotate attention output rows back to the original V coordinates. PV accumulates in
// the rotated-V basis (V is Hadamard-rotated before the int4 encode), so re-applying the same
// 64-dim butterfly to each {group, lane} output row undoes it. MUST run after the final `out`
// tile has been written. One 32-lane block owns one (row, q_head, group).
template <int QHeads>
__launch_bounds__(32) __global__
    void rk2v4e8_inverse_rotate_output_kernel(__nv_bfloat16* output, int width, int full_width,
                                              int column_begin,
                                              const std::int32_t* valid_columns) {
    const int unit   = static_cast<int>(blockIdx.x);
    const int lane   = static_cast<int>(threadIdx.x);
    const int group  = unit % kRk2v4E8Groups;
    const int rest   = unit / kRk2v4E8Groups;
    const int q_head = rest % QHeads;
    const int row    = rest / QHeads;
    const int batch  = row / width;
    const int token  = row - batch * width;
    const int column = column_begin + token;
    // Uniform across the block, so the whole warp converges on the butterfly below.
    if (token >= width || (valid_columns != nullptr && column >= valid_columns[batch])) { return; }
    const int d0 = group * kRk2v4E8Group + lane;
    const int d1 = d0 + 32;
    const std::int64_t base =
        static_cast<std::int64_t>(kRk2v4E8HeadDim) *
        (static_cast<std::int64_t>(q_head) +
         static_cast<std::int64_t>(QHeads) *
             (static_cast<std::int64_t>(column) + static_cast<std::int64_t>(full_width) * batch));
    float x0 = __bfloat162float(output[base + d0]);
    float x1 = __bfloat162float(output[base + d1]);
    rk2v4e8_hadamard64(x0, x1, lane);
    output[base + d0] = __float2bfloat16(x0);
    output[base + d1] = __float2bfloat16(x1);
}

} // namespace ninfer::ops
