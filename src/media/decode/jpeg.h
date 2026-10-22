#pragma once

#include "media/decode/decode.h"

namespace ninfer::media::decode::detail {

// Stored geometry and RGB8 pixels, before EXIF orientation is applied.
ImageInfo inspect_jpeg(std::span<const std::uint8_t> bytes, const Policy& policy);
Image decode_jpeg(std::span<const std::uint8_t> bytes, const Policy& policy);

} // namespace ninfer::media::decode::detail
