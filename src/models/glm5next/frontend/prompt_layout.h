#pragma once

#include "models/common/frontend/prompt.h"
#include "models/glm5next/frontend/chat_template.h"

#include <functional>

namespace ninfer::models::glm5next {

struct RewriteByteSpec {
    models::frontend::RewriteCheckpointKind kind;
    std::size_t frontier;
};

struct PromptBoundaries {
    std::vector<std::optional<std::size_t>> markers;
    std::vector<std::size_t> semantic;
    std::vector<std::size_t> structural;
    std::vector<std::size_t> execution;
    std::optional<RewriteByteSpec> rewrite;
};

struct TurnProbe {
    bool retains_response;
    text::TemplateOutput output;
};

struct PromptProbes {
    std::function<RenderedChat(const PromptCacheMarker&)> content_prefix;
    std::function<TurnProbe()> next_turn;
};

// Coordinates refer to this rendering, including its literal interpretation and input order.
PromptBoundaries resolve_prompt_boundaries(const RenderedChat& rendered,
                                           std::span<const ChatMessage> messages, bool continuation,
                                           std::span<const PromptCacheMarker> markers,
                                           const PromptProbes& probes);

} // namespace ninfer::models::glm5next
