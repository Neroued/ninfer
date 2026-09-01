// API-compatible stand-in for media/decode/decode.cpp in builds configured
// with NINFER_BUILD_MEDIA=OFF (no FFMPEG). The public API is preserved so the
// vision frontend compiles unchanged; every entry point throws at runtime.
// Text-only servers never reach these calls: the generation service rejects
// media requests when started without --vision.

#include "media/decode/decode.h"

#include <stdexcept>
#include <string>

namespace ninfer::media::decode {

namespace {
[[noreturn]] void unavailable() {
    throw std::runtime_error(
        "media decode is unavailable in this build; configure with "
        "NINFER_BUILD_MEDIA=ON (requires FFMPEG) to serve vision models");
}
} // namespace

Image decode_image(std::span<const std::uint8_t>, const Policy&) {
    unavailable();
}

Video decode_video(std::span<const std::uint8_t>, const Policy&, double, int, int) {
    unavailable();
}

} // namespace ninfer::media::decode
