#pragma once
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/linear/nvfp4/nvfp4_layout.h"

namespace ninfer::ops::detail {
// The TMA route reads a whole [128 or 256 tokens, kNvfp4ScaleTileGroups groups] tile of
// activation scales per request. Row-major by token that tile is 16 bytes per row at the plane's
// row stride, so one stage would require a small request for each token. Writing the same bytes
// tile-contiguous makes the request wide. The byte order inside a tile is unchanged, so the shared
// image the MMA reads is identical.
//
// It addresses whole tiles, so a ragged token count is padded up to one and that padding is
// written with zeroes below. launch_nvfp4_a4_quantize checks that the plane was allocated
// for the padded count before this layout is selected.
template <class Geometry, Nvfp4ScaleLayout Layout>
__device__ __forceinline__ std::int64_t nvfp4_tiled_scale_offset(int token, int group) {
    constexpr int kGroupsPerTile = kNvfp4ScaleTileGroups;
    constexpr int kTilesPerPlane = Geometry::kGroupsPerRow / kGroupsPerTile;
    constexpr int kBlockTokens   = kNvfp4ScaleTileTokens<Layout>;
    const int token_tile         = token / kBlockTokens;
    const int group_tile         = group / kGroupsPerTile;
    const int tile               = token_tile * kTilesPerPlane + group_tile;
    return static_cast<std::int64_t>(tile) * kBlockTokens * kGroupsPerTile +
           static_cast<std::int64_t>(token - token_tile * kBlockTokens) * kGroupsPerTile +
           (group - group_tile * kGroupsPerTile);
}

template <class Geometry, int Threads, Nvfp4ScaleLayout Layout>
__global__ __launch_bounds__(Threads, 512 / Threads) void nvfp4_a4_quantize_kernel(
    const __nv_bfloat16* __restrict__ input, std::uint8_t* __restrict__ codes,
    std::uint8_t* __restrict__ scales, std::int32_t tokens, std::int32_t written_tokens,
    float input_scale_divisor) {
    static_assert(Threads == 128 || Threads == 256 || Threads == 512);
    static_assert(Layout == Nvfp4ScaleLayout::RowMajor ||
                  (Geometry::kInputRows / 16) % kNvfp4ScaleTileGroups == 0);
    constexpr int kGroupsPerRow = Geometry::kInputRows / 16;
    const int task =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int tasks = tokens * kGroupsPerRow;
    if (task >= tasks) {
        // The tiled plane is addressed in whole tiles, so the launch covers the padding of the last
        // one. A padded token owns no input and no code byte, only the scale the consumer's tile
        // will read; zero it so it is defined rather than whatever the arena last held.
        // written_tokens is the extent the plane was allocated for and bounds this store; it equals
        // tokens for the row-major layout, which stops at the real count.
        if constexpr (Layout != Nvfp4ScaleLayout::RowMajor) {
            const int pad_token = task / kGroupsPerRow;
            if (pad_token < written_tokens) {
                const int pad_group = task - pad_token * kGroupsPerRow;
                scales[nvfp4_tiled_scale_offset<Geometry, Layout>(pad_token, pad_group)] = 0;
            }
        }
        return;
    }

    const int token                   = task / kGroupsPerRow;
    const int group                   = task - token * kGroupsPerRow;
    const Nvfp4QuantizedK16 quantized = quantize_nvfp4_k16(
        input + static_cast<std::int64_t>(token) * Geometry::kInputRows + group * 16,
        input_scale_divisor);
    auto* code_destination =
        codes + static_cast<std::int64_t>(token) * Geometry::kCodeBytesPerRow + group * 8;
    store_vec(code_destination, make_uint2(quantized.codes_lo, quantized.codes_hi));
    if constexpr (Layout != Nvfp4ScaleLayout::RowMajor) {
        scales[nvfp4_tiled_scale_offset<Geometry, Layout>(token, group)] = quantized.scale;
    } else {
        scales[static_cast<std::int64_t>(token) * kGroupsPerRow + group] = quantized.scale;
    }
}

} // namespace ninfer::ops::detail
