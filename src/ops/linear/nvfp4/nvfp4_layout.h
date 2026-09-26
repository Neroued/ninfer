#pragma once
#include "ops/linear/nvfp4/nvfp4_geometry.h"

namespace ninfer::ops::detail {
inline constexpr int kNvfp4ScaleTileGroups = 16;
enum class Nvfp4ScaleLayout : std::uint8_t { RowMajor, Tiled128, Tiled256 };
template <Nvfp4ScaleLayout Layout>
inline constexpr int kNvfp4ScaleTileTokens = Layout == Nvfp4ScaleLayout::Tiled128   ? 128
                                             : Layout == Nvfp4ScaleLayout::Tiled256 ? 256
                                                                                    : 1;

inline constexpr int nvfp4_scale_tile_tokens(Nvfp4ScaleLayout layout) {
    return layout == Nvfp4ScaleLayout::Tiled128   ? 128
           : layout == Nvfp4ScaleLayout::Tiled256 ? 256
                                                  : 1;
}

inline constexpr int nvfp4_a4_padded_tokens(int tokens,
                                            Nvfp4ScaleLayout layout = Nvfp4ScaleLayout::Tiled256) {
    const int tile = nvfp4_scale_tile_tokens(layout);
    return ((tokens + tile - 1) / tile) * tile;
}
} // namespace ninfer::ops::detail
