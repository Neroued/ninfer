// API-compatible stand-in for product/media_acquire/acquire.cpp in builds
// configured with NINFER_BUILD_MEDIA=OFF (no libcurl). The public API is
// preserved so the serve layer and prompt input compile unchanged; every
// entry point throws at runtime. Text-only servers never reach these calls:
// the generation service rejects media requests when started without
// --vision.

#include "product/media_acquire/acquire.h"

#include <stdexcept>
#include <string>

namespace ninfer::product::media_acquire {

[[noreturn]] static void unavailable() {
    throw std::runtime_error(
        "media acquisition is unavailable in this build; configure with "
        "NINFER_BUILD_MEDIA=ON (requires libcurl) to serve vision models");
}

std::vector<std::uint8_t> acquire_bytes(const Source&, const Policy&) {
    unavailable();
}

} // namespace ninfer::product::media_acquire
