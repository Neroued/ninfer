#pragma once

#include "models/common/context/prefix.h"
#include "models/glm5next/frontend/frontend.h"
#include "models/glm5next/frontend/output.h"
#include "models/glm5next/prefix_identity.h"

namespace ninfer::models::glm5next {

struct PromptIdentity {
    bool reusable = true;
    std::optional<models::frontend::RewriteCheckpointSpec> rewrite_checkpoint;
    std::vector<MediaInputIdentity> media;
};

struct PreparedPromptData : models::frontend::PromptData {
    PromptIdentity identity;
    PreparedOutput output;
    SamplingPreset sampling;
    std::vector<std::shared_ptr<const models::frontend::PreparedMediaPayload>> media_payloads;

    [[nodiscard]] InputPrefixView prefix_view() const noexcept {
        return {token_ids, identity.media};
    }
};

class PreparedPromptAccess {
public:
    static const PreparedPromptData& view(const PreparedPrompt& prompt);
    static std::shared_ptr<const PreparedPromptData> shared(const PreparedPrompt& prompt);
    static PreparedPromptData take(PreparedPrompt&& prompt);
};

class PromptPrefixKeyer {
public:
    PromptPrefixKeyer() = default;
    PromptPrefixKeyer(const PreparedPrompt& prompt, std::uint32_t domain);

    [[nodiscard]] std::uint32_t max_frontier() const noexcept { return max_frontier_; }

    [[nodiscard]] runtime::PrefixKey key_at(std::uint32_t frontier) const;

private:
    std::shared_ptr<const PreparedPromptData> data_;
    std::uint32_t domain_       = 0;
    std::uint32_t max_frontier_ = 0;
    mutable std::vector<context::MediaIdentityView> media_;
    mutable bool media_ready_ = false;
    context::PrefixFingerprints fingerprints_;
};

} // namespace ninfer::models::glm5next
