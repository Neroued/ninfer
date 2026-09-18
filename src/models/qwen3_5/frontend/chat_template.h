#pragma once

#include "models/qwen3_5/frontend/tokenizer.h"
#include "text/jinja.h"

#include "models/qwen3_5/frontend/prepared_prompt.h"
#include <ninfer/types.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::frontend {

inline constexpr std::string_view kCanonicalReasoningCloseSerialization = "\n</think>\n\n";

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

enum class ChatPartKind {
    Text,
    Image,
    Video,
};

enum class Modality : std::uint8_t {
    Image = 1,
    Video = 2,
};

struct MediaPlaceholderByteSpec {
    text::ByteSpan bytes;
    Modality modality      = Modality::Image;
    std::size_t item_index = 0;
};

struct MediaTokenRunByteSpec {
    text::ByteSpan bytes;
    Modality modality       = Modality::Image;
    std::size_t item_index  = 0;
    std::size_t frame_index = 0;
};

struct MediaData {
    std::vector<std::uint8_t> bytes;
    std::string media_type;
    std::string source_name;
    ImageResizePolicy image_resize_policy = ImageResizePolicy::Downsize;
};

struct ChatPart {
    ChatPartKind kind = ChatPartKind::Text;
    std::string text;
    MediaData media;

    static ChatPart text_part(std::string value) {
        ChatPart part;
        part.text = std::move(value);
        return part;
    }

    static ChatPart image(MediaData value) {
        ChatPart part;
        part.kind  = ChatPartKind::Image;
        part.media = std::move(value);
        return part;
    }

    static ChatPart video(MediaData value) {
        ChatPart part;
        part.kind  = ChatPartKind::Video;
        part.media = std::move(value);
        return part;
    }
};

struct ChatMessage {
    ChatRole role = ChatRole::User;
    std::vector<ChatPart> parts;
    std::string reasoning_content;
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;

    [[nodiscard]] bool has_media() const noexcept;
};

struct ChatRenderOptions {
    PromptContinuationMode continuation = PromptContinuationMode::NewAssistantTurn;
    // Internal renderer control used by frontend qualification. Product PromptInput always
    // selects either a new assistant turn or continuation of the final assistant.
    bool add_generation_prompt = true;
    std::optional<bool> enable_thinking;
    std::optional<ReasoningEffort> reasoning_effort;
    std::optional<bool> preserve_thinking;
    std::string chat_template_kwargs_json;
    bool add_vision_id = false;
    std::vector<std::string> tool_jsons;
    std::vector<PromptCacheMarker> cache_markers;
};

struct RewriteCheckpointByteSpec {
    RewriteCheckpointKind kind = RewriteCheckpointKind::TurnClosure;
    std::size_t offset         = 0;
};

struct RenderedChat {
    std::string text;
    std::vector<text::ByteSpan> literal_spans;
    bool starts_in_reasoning = false;
    std::vector<MediaPlaceholderByteSpec> media_placeholders;
    std::vector<MediaTokenRunByteSpec> media_token_runs;
    std::optional<RewriteCheckpointByteSpec> rewrite_checkpoint;
    std::vector<std::size_t> rewrite_execution_boundaries;
    // Index n is the exact byte frontier after serializing the first n input messages. A missing
    // value means the template has no independent boundary there (for example, before a leading
    // instruction message folded into the system preamble).
    std::vector<std::optional<std::size_t>> message_boundaries;
    // One rendered byte boundary per requested cache marker.
    std::vector<std::optional<std::size_t>> cache_boundaries;
};

class CompiledChatTemplate {
public:
    [[nodiscard]] static CompiledChatTemplate
    resolve(std::string_view source, std::string source_name = "chat_template.jinja",
            nlohmann::ordered_json special_tokens = nlohmann::ordered_json::object());
    [[nodiscard]] RenderedChat render(const std::vector<ChatMessage>& messages,
                                      ChatRenderOptions options         = {},
                                      const PreparationControl& control = {}) const;

private:
    CompiledChatTemplate(text::JinjaTemplate compiled, nlohmann::ordered_json special_tokens)
        : compiled_(std::move(compiled)), special_tokens_(std::move(special_tokens)) {}

    text::JinjaTemplate compiled_;
    nlohmann::ordered_json special_tokens_;
};

} // namespace ninfer::models::qwen3_5::frontend
