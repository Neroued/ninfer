#pragma once

#include "models/common/frontend/chat_template.h"
#include "ninfer/types.h"

#include <optional>
#include <span>
#include <utility>

namespace ninfer::models::glm5next {

struct PromptBoundaries;

// Byte coordinates refer to the tagged template input, before trimming or serialization.
struct TextSource {
    std::uint32_t tag = 0;
    text::ByteSpan bytes;
};

struct MessageSources {
    std::optional<TextSource> content;
    // Indexed by the original message's parts, including canonical media wrappers.
    std::vector<std::optional<TextSource>> parts;
    TextSource reasoning;
    std::vector<std::uint32_t> tool_calls;
};

struct MediaPlaceholder {
    std::size_t message;
    std::size_t part;
    MediaKind kind;
    text::ByteSpan bytes;
};

struct RenderedChat {
    text::TemplateOutput output;
    std::vector<MessageSources> messages;
    std::vector<std::uint32_t> tools;
    std::vector<MediaPlaceholder> media;
    bool starts_in_reasoning = false;
    std::string continuation_content;
};

class CompiledChatTemplate {
public:
    explicit CompiledChatTemplate(models::frontend::ChatTemplate compiled)
        : compiled_(std::move(compiled)) {}

    // New turns start in reasoning. Continuation resumes the final assistant's text after its
    // completed reasoning block. All returned text, source mappings and literal spans are owned.
    [[nodiscard]] RenderedChat render(std::span<const ChatMessage> messages,
                                      const PromptOptions& options      = {},
                                      const PreparationControl& control = {}) const;

    [[nodiscard]] RenderedChat render(std::span<const ChatMessage> messages,
                                      const PromptOptions& options,
                                      std::span<const PromptCacheMarker> markers,
                                      PromptBoundaries& boundaries,
                                      const PreparationControl& control = {}) const;

private:
    RenderedChat render_impl(std::span<const ChatMessage> messages, const PromptOptions& options,
                             std::span<const PromptCacheMarker> markers,
                             PromptBoundaries* boundaries, const PreparationControl& control) const;
    models::frontend::ChatTemplate compiled_;
};

} // namespace ninfer::models::glm5next
