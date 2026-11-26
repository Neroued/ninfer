#pragma once

#include "models/common/frontend/chat_template.h"
#include "models/inkling2/frontend/prompt_layout.h"
#include "models/inkling2/frontend/resources.h"

#include <optional>
#include <span>
#include <string>

namespace ninfer::models::inkling2 {

class CompiledChatTemplate {
public:
    CompiledChatTemplate(models::frontend::ChatTemplate compiled, std::string image_token)
        : compiled_(std::move(compiled)), image_token_(std::move(image_token)) {}

    [[nodiscard]] RenderedChat render(std::span<const ChatMessage> messages,
                                      const PromptOptions& options,
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
    std::string image_token_;
};

} // namespace ninfer::models::inkling2
