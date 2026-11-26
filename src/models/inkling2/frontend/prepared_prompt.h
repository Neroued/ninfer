#pragma once

#include "models/inkling2/frontend/frontend.h"
#include "models/inkling2/frontend/output.h"
#include "models/inkling2/prefix_identity.h"

#include <memory>
#include <optional>
#include <vector>

namespace ninfer::models::inkling2 {

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
    static PreparedPromptData take(PreparedPrompt&& prompt);
};

} // namespace ninfer::models::inkling2
