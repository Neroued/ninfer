#pragma once

#include "models/common/qwen/frontend/chat_template.h"

namespace ninfer::models::qwen::frontend {

struct MessageLayout {
    ChatRole role             = ChatRole::User;
    std::size_t begin         = 0;
    std::size_t content_begin = 0;
    std::size_t content_end   = 0;
    std::size_t end           = 0;
    bool closed               = false;
};

struct PromptLayout {
    std::vector<MessageLayout> messages;
    std::vector<MediaPlaceholderByteSpec> media_placeholders;
    std::vector<std::size_t> execution_boundaries;
    bool starts_in_reasoning = false;
};

[[nodiscard]] PromptLayout inspect_prompt_layout(const text::TemplateOutput& rendered,
                                                 std::span<const models::frontend::Modality> media);
} // namespace ninfer::models::qwen::frontend
