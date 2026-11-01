#pragma once

#include "models/glm5next/frontend/frontend.h"
#include "models/glm5next/frontend/output.h"
#include "models/glm5next/prefix_identity.h"

namespace ninfer::models::glm5next {

struct PromptIdentity {
    bool reusable = true;
    std::optional<models::frontend::RewriteCheckpointSpec> rewrite_checkpoint;
    std::vector<std::uint32_t> execution_frontiers;
    std::vector<MediaInputIdentity> media;
};

struct PreparedPromptData : models::frontend::PromptData {
    PromptIdentity identity;
    PreparedOutput output;
    SamplingPreset sampling;
    std::vector<std::shared_ptr<const models::frontend::PreparedMediaPayload>> media_payloads;

    [[nodiscard]] InputPrefixView prefix_view() const noexcept {
        return {token_ids, identity.media, identity.execution_frontiers};
    }
};

class PreparedPromptAccess {
public:
    static const PreparedPromptData& view(const PreparedPrompt& prompt);
    static PreparedPromptData take(PreparedPrompt&& prompt);
};

} // namespace ninfer::models::glm5next
