#pragma once

#include <ninfer/targets/qwen3_6/frontend.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6 {

enum class PromptModality : std::uint8_t {
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
    PromptModality modality = PromptModality::Image;
    VisionGrid grid;
    std::size_t patch_begin = 0;
    std::size_t patch_count = 0;
    // SHA-256 of the owned encoded media bytes. Grid/modality/span identity is carried
    // separately so this digest binds the content without retaining the request payload.
    std::array<std::uint8_t, 32> content_digest{};
    std::vector<double> timestamps;
    std::vector<TokenSpan> token_spans;
};

struct PromptIdentity {
    bool reusable = true;
    std::optional<std::uint32_t> assistant_content_boundary;
};

struct PrepareStats {
    double seconds                = 0.0;
    std::size_t media_items       = 0;
    std::uint64_t raw_patches     = 0;
    std::uint64_t vision_tokens   = 0;
    std::uint64_t attention_pairs = 0;
    std::size_t patch_bytes       = 0;
};

struct PreparedPromptData {
    std::vector<TokenId> token_ids;
    std::vector<std::uint8_t> token_types;
    std::vector<std::int32_t> positions;
    std::int32_t rope_delta = 0;
    std::vector<float> patches;
    std::vector<VisionItem> vision_items;
    PromptIdentity identity;
    bool starts_in_reasoning = false;
    PrepareStats prepare;

    [[nodiscard]] std::span<const std::int32_t> position_axis(int axis) const;

    [[nodiscard]] bool has_media() const noexcept { return !vision_items.empty(); }
};

class PreparedPromptAccess {
public:
    [[nodiscard]] static const PreparedPromptData& view(const PreparedPrompt& prompt);
    [[nodiscard]] static PreparedPromptData take(PreparedPrompt&& prompt);
};

} // namespace ninfer::targets::qwen3_6
