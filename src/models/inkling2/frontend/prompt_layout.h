#pragma once

#include "models/common/frontend/chat_template.h"
#include "models/common/frontend/prompt.h"
#include "ninfer/types.h"
#include "text/jinja.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninfer::models::inkling2 {

enum class InitialBlock : std::uint8_t {
    BetweenBlocks,
    MessageHeader,
    Text,
    Thinking,
};

struct TextSource {
    std::uint32_t tag = 0;
    text::ByteSpan bytes;
};

struct MessageSources {
    std::optional<TextSource> content;
    std::vector<std::optional<TextSource>> parts;
    TextSource reasoning;
    std::vector<std::uint32_t> tool_calls;
};

struct MediaPlaceholder {
    std::size_t message = 0;
    std::size_t part    = 0;
    MediaKind kind      = MediaKind::Image;
    text::ByteSpan bytes;
};

struct RenderedChat {
    text::TemplateOutput output;
    std::vector<MessageSources> messages;
    std::vector<std::uint32_t> tools;
    std::vector<MediaPlaceholder> media;
    InitialBlock initial_block = InitialBlock::BetweenBlocks;
    bool thinking_enabled      = true;
    std::string continuation_content;
};

struct RewriteByteSpec {
    models::frontend::RewriteCheckpointKind kind =
        models::frontend::RewriteCheckpointKind::ResponseReplay;
    std::size_t frontier = 0;
};

struct PromptBoundaries {
    std::vector<std::size_t> semantic;
    std::vector<std::size_t> structural;
    std::vector<std::optional<std::size_t>> markers;
    std::optional<RewriteByteSpec> rewrite;
};

struct TurnProbe {
    bool retains_response = true;
    text::TemplateOutput output;
};

struct PromptProbes {
    std::function<RenderedChat(const PromptCacheMarker&)> content_prefix;
    std::function<TurnProbe()> next_turn;
};

PromptBoundaries resolve_prompt_boundaries(const RenderedChat& rendered,
                                           std::span<const ChatMessage> messages, bool continuation,
                                           std::span<const PromptCacheMarker> markers,
                                           const PromptProbes& probes);

} // namespace ninfer::models::inkling2
