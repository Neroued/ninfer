#pragma once
#include "models/glm5next/frontend/prepared_prompt.h"

namespace ninfer::models::glm5next {
struct InputIdentity {
    std::vector<TokenId> tokens;
    std::vector<MediaInputIdentity> media;
    std::vector<std::uint32_t> execution_frontiers;
    PrefixDigests digests;
    std::uint32_t tag = 0;

    InputIdentity() = default;
    explicit InputIdentity(const PreparedPromptData&, std::uint32_t tag);

    [[nodiscard]] InputPrefixView view() const noexcept {
        return {tokens, media, execution_frontiers};
    }

    void reserve(std::size_t count);
    void prepare_generated(std::size_t count, std::optional<std::uint32_t> split);
    void append_generated(std::span<const TokenId>, std::optional<std::uint32_t> split);
    void truncate(std::uint32_t frontier);
};
} // namespace ninfer::models::glm5next
