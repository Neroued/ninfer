#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::models::frontend {

struct PreparedMediaPayload {
    // Exact row-major BF16 input consumed by the Vision patch projection.
    std::unique_ptr<std::uint16_t[]> patches;
    std::size_t patch_elements = 0;

    [[nodiscard]] std::span<std::uint16_t> mutable_span() noexcept {
        return {patches.get(), patch_elements};
    }

    [[nodiscard]] std::span<const std::uint16_t> span() const noexcept {
        return {patches.get(), patch_elements};
    }
};

enum class Modality : std::uint8_t {
    Image = 1,
    Video = 2,
};

struct VisionGrid {
    std::int32_t temporal = 0;
    std::int32_t height   = 0;
    std::int32_t width    = 0;
};

struct TokenSpan {
    std::size_t begin = 0;
    std::size_t count = 0;
};

struct VisionItem {
    Modality modality = Modality::Image;
    VisionGrid grid;
    std::size_t patch_begin = 0;
    std::size_t patch_count = 0;
    // SHA-256 of the owned encoded media bytes. Grid/modality/span identity is carried
    // separately so this digest binds the content without retaining the request payload.
    std::array<std::uint8_t, 32> content_digest{};
    std::vector<double> timestamps;
    std::vector<TokenSpan> token_spans;
};

} // namespace ninfer::models::frontend
