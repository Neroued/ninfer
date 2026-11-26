#pragma once

#include "models/common/frontend/resources.h"
#include "models/inkling2/frontend/processor_config.h"

#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace ninfer::artifact {
class Binder;
}

namespace ninfer::models::inkling2 {

inline constexpr std::string_view kMessageUser            = "<|message_user|>";
inline constexpr std::string_view kMessageModel           = "<|message_model|>";
inline constexpr std::string_view kMessageSystem          = "<|message_system|>";
inline constexpr std::string_view kMessageTool            = "<|message_tool|>";
inline constexpr std::string_view kContentText            = "<|content_text|>";
inline constexpr std::string_view kContentImage           = "<|content_image|>";
inline constexpr std::string_view kContentThinking        = "<|content_thinking|>";
inline constexpr std::string_view kContentXml             = "<|content_xml|>";
inline constexpr std::string_view kContentToolError       = "<|content_tool_error|>";
inline constexpr std::string_view kContentInvokeToolJson  = "<|content_invoke_tool_json|>";
inline constexpr std::string_view kContentInvokeToolText  = "<|content_invoke_tool_text|>";
inline constexpr std::string_view kEndMessage             = "<|end_message|>";
inline constexpr std::string_view kContentModelEndSample  = "<|content_model_end_sampling|>";
inline constexpr std::string_view kDefaultImagePadToken   = "<|unused_200054|>";

// Minimum image span: <|content_image|> + 1 pad token + <|end_message|>.
inline constexpr std::uint32_t kMinimumMediaSpanTokens = 3;

struct FramingTokens {
    TokenId message_model            = -1;
    TokenId content_thinking         = -1;
    TokenId content_text             = -1;
    TokenId content_invoke_tool_json = -1;
    TokenId end_message              = -1;
    TokenId content_image            = -1;
    TokenId image_pad                = -1;
    std::optional<TokenId> content_tool_error;
    std::optional<TokenId> content_invoke_tool_text;
};

struct FrontendResources {
    models::frontend::TextResources text;
    std::string_view processor_config_json;
    SamplingPreset sampling;
    FramingTokens framing;
    std::shared_ptr<const std::vector<TokenId>> control_to_text;
    std::shared_ptr<const std::vector<TokenId>> control_to_header;
    std::optional<ProcessorConfig> processor;

    [[nodiscard]] std::uint32_t control_token_limit() const;
};

void parse_resources(FrontendResources& resources, std::uint32_t embedding_rows);
FrontendResources bind_resources(artifact::Binder& binder, std::uint32_t embedding_rows,
                                 bool vision);

} // namespace ninfer::models::inkling2
