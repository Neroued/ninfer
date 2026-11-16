#pragma once
#include "models/common/context/identity.h"
#include "models/glm5next/frontend/prepared_prompt.h"

namespace ninfer::models::glm5next {
struct InputIdentity {
    std::vector<TokenId> tokens;
    std::vector<MediaInputIdentity> media;
    PrefixDigests digests;
    std::uint32_t tag = 0;
    context::IdentityMetadata retention;

    InputIdentity() = default;
    explicit InputIdentity(const PreparedPromptData&, std::uint32_t tag);

    [[nodiscard]] InputPrefixView view() const noexcept { return {tokens, media}; }

    void reserve(std::size_t count);
    void prepare_generated(std::size_t count);
    void append_generated(std::span<const TokenId>);
    void truncate(std::uint32_t frontier);

    [[nodiscard]] std::size_t metadata_bytes() const noexcept {
        auto bytes = sizeof(*this) + tokens.capacity() * sizeof(TokenId) +
                     media.capacity() * sizeof(MediaInputIdentity) + digests.metadata_bytes();
        for (const auto& entry : media)
            bytes += entry.item.timestamps.capacity() * sizeof(double) +
                     entry.item.token_spans.capacity() * sizeof(models::frontend::TokenSpan);
        return bytes;
    }

    [[nodiscard]] static std::size_t cache_bytes(std::size_t tokens, std::size_t concurrency) {
        return context::IdentityMetadata::cache_bytes<InputIdentity>(
            tokens, concurrency, sizeof(TokenId) + sizeof(std::array<std::uint64_t, 2>));
    }
};
} // namespace ninfer::models::glm5next
