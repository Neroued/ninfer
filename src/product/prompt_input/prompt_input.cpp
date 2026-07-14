#include "product/prompt_input/prompt_input.h"

#include "product/media_acquire/acquire.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace ninfer::product {
namespace {

using Json = nlohmann::json;

bool supported_role(std::string_view role) {
    return role == "system" || role == "developer" || role == "user" || role == "assistant" ||
           role == "tool";
}

std::string media_value(const Json& part, MediaKind kind) {
    const char* direct = kind == MediaKind::Image ? "image" : "video";
    const char* url    = kind == MediaKind::Image ? "image_url" : "video_url";
    if (part.contains(direct) && part.at(direct).is_string()) {
        return part.at(direct).get<std::string>();
    }
    if (part.contains(url)) {
        const Json& value = part.at(url);
        if (value.is_string()) { return value.get<std::string>(); }
        if (value.is_object() && value.contains("url") && value.at("url").is_string()) {
            return value.at("url").get<std::string>();
        }
    }
    return {};
}

std::string data_media_type(std::string_view value) {
    if (!value.starts_with("data:")) { return {}; }
    const std::size_t separator = value.find(';', 5);
    const std::size_t comma     = value.find(',', 5);
    const std::size_t end       = separator == std::string_view::npos ? comma : separator;
    if (end == std::string_view::npos || end == 5) { return {}; }
    return std::string(value.substr(5, end - 5));
}

OwnedMedia acquire_media(const Json& part, MediaKind kind, std::size_t message_index,
                         std::size_t part_index) {
    std::string value = media_value(part, kind);
    std::string media_type;
    if (part.contains("media_type") && part.at("media_type").is_string()) {
        media_type = part.at("media_type").get<std::string>();
    }
    if (value.empty()) {
        throw std::invalid_argument("message " + std::to_string(message_index) + " content part " +
                                    std::to_string(part_index) + " has no media source");
    }

    media_acquire::Source source;
    source.media_type = media_type;
    if (value.starts_with("http://") || value.starts_with("https://")) {
        source.kind = media_acquire::SourceKind::Url;
    } else if (value.starts_with("data:")) {
        source.kind = media_acquire::SourceKind::Data;
        if (media_type.empty()) { media_type = data_media_type(value); }
    } else {
        source.kind = media_acquire::SourceKind::Path;
        if (value.starts_with("file://")) { value.erase(0, 7); }
    }
    source.value = value;

    media_acquire::Policy policy;
    // Local product inputs include loopback URLs just as naturally as filesystem paths.
    policy.allow_private_network    = true;
    std::vector<std::uint8_t> bytes = media_acquire::acquire_bytes(source, policy);

    OwnedMedia result;
    result.kind       = kind;
    result.media_type = std::move(media_type);
    result.source_name =
        source.kind == media_acquire::SourceKind::Data ? "inline data URI" : source.value;
    result.bytes = std::move(bytes);
    return result;
}

std::vector<MessagePart> parse_content(const Json& content, std::size_t message_index) {
    if (content.is_null()) { return {}; }
    if (content.is_string()) {
        MessagePart part;
        part.text = content.get<std::string>();
        return {std::move(part)};
    }
    if (!content.is_array() || content.empty()) {
        throw std::invalid_argument("message " + std::to_string(message_index) +
                                    " content must be a string or non-empty array");
    }

    std::vector<MessagePart> result;
    result.reserve(content.size());
    for (std::size_t i = 0; i < content.size(); ++i) {
        const Json& item = content.at(i);
        if (!item.is_object()) {
            throw std::invalid_argument("message " + std::to_string(message_index) +
                                        " content parts must be objects");
        }
        const std::string type = item.value("type", "");
        MessagePart part;
        if (type == "text" || item.contains("text")) {
            if (!item.contains("text") || !item.at("text").is_string()) {
                throw std::invalid_argument("text content part must contain string text");
            }
            part.text = item.at("text").get<std::string>();
        } else if (type == "image" || type == "image_url" || item.contains("image") ||
                   item.contains("image_url")) {
            part.kind  = MessagePartKind::Media;
            part.media = acquire_media(item, MediaKind::Image, message_index, i);
        } else if (type == "video" || type == "video_url" || item.contains("video") ||
                   item.contains("video_url")) {
            part.kind  = MessagePartKind::Media;
            part.media = acquire_media(item, MediaKind::Video, message_index, i);
        } else {
            throw std::invalid_argument("message " + std::to_string(message_index) +
                                        " has unsupported content part type: " + type);
        }
        result.push_back(std::move(part));
    }
    return result;
}

std::vector<ToolCall> parse_tool_calls(const Json& value, std::size_t message_index) {
    if (!value.is_array() || value.empty()) {
        throw std::invalid_argument("message " + std::to_string(message_index) +
                                    " tool_calls must be a non-empty array");
    }
    std::vector<ToolCall> result;
    result.reserve(value.size());
    for (const Json& item : value) {
        if (!item.is_object()) { throw std::invalid_argument("tool call must be an object"); }
        const Json& function = item.contains("function") ? item.at("function") : item;
        if (!function.is_object() || !function.contains("name") ||
            !function.at("name").is_string() || !function.contains("arguments")) {
            throw std::invalid_argument("tool call must contain name and arguments");
        }
        ToolCall call;
        if (item.contains("id") && item.at("id").is_string()) {
            call.id = item.at("id").get<std::string>();
        }
        call.name             = function.at("name").get<std::string>();
        const Json& arguments = function.at("arguments");
        if (arguments.is_object()) {
            call.arguments_json = arguments.dump();
        } else if (arguments.is_string()) {
            call.arguments_json = arguments.get<std::string>();
            const Json parsed   = Json::parse(call.arguments_json, nullptr, false);
            if (parsed.is_discarded() || !parsed.is_object()) {
                throw std::invalid_argument("tool call arguments string must contain an object");
            }
        } else {
            throw std::invalid_argument("tool call arguments must be an object or JSON string");
        }
        result.push_back(std::move(call));
    }
    return result;
}

ChatMessage parse_message(const Json& item, std::size_t index) {
    if (!item.is_object() || !item.contains("role") || !item.contains("content")) {
        throw std::invalid_argument("message " + std::to_string(index) +
                                    " must be an object containing role and content");
    }
    for (auto field = item.begin(); field != item.end(); ++field) {
        if (field.key() != "role" && field.key() != "content" &&
            field.key() != "reasoning_content" && field.key() != "tool_calls" &&
            field.key() != "tool_call_id") {
            throw std::invalid_argument("message " + std::to_string(index) +
                                        " contains unsupported field: " + field.key());
        }
    }
    if (!item.at("role").is_string()) {
        throw std::invalid_argument("message " + std::to_string(index) + " role must be a string");
    }

    ChatMessage message;
    message.role = item.at("role").get<std::string>();
    if (!supported_role(message.role)) {
        throw std::invalid_argument("unsupported chat role: " + message.role);
    }
    message.parts = parse_content(item.at("content"), index);

    if (item.contains("reasoning_content")) {
        if (!item.at("reasoning_content").is_string()) {
            throw std::invalid_argument("reasoning_content must be a string");
        }
        message.reasoning_content = item.at("reasoning_content").get<std::string>();
    }
    if (item.contains("tool_calls")) {
        if (message.role != "assistant") {
            throw std::invalid_argument("only assistant messages may contain tool_calls");
        }
        message.tool_calls = parse_tool_calls(item.at("tool_calls"), index);
    }
    if (item.contains("tool_call_id")) {
        if (message.role != "tool" || !item.at("tool_call_id").is_string()) {
            throw std::invalid_argument("tool_call_id must be a string on a tool message");
        }
        message.tool_call_id = item.at("tool_call_id").get<std::string>();
    }
    if (message.parts.empty() && message.tool_calls.empty()) {
        throw std::invalid_argument("message " + std::to_string(index) + " has empty content");
    }
    return message;
}

} // namespace

PromptInput prompt_from_text(std::string text, bool enable_thinking) {
    if (text.empty()) { throw std::invalid_argument("--prompt text is empty"); }
    MessagePart part;
    part.text = std::move(text);
    ChatMessage message;
    message.role = "user";
    message.parts.push_back(std::move(part));
    PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = enable_thinking;
    return input;
}

PromptInput prompt_from_messages(const std::filesystem::path& path, bool enable_thinking) {
    std::ifstream stream(path);
    if (!stream) { throw std::runtime_error("failed to open messages JSON: " + path.string()); }

    Json root;
    try {
        stream >> root;
    } catch (const nlohmann::json::exception& error) {
        throw std::invalid_argument(std::string("failed to parse messages JSON: ") + error.what());
    }

    PromptInput input;
    input.options.enable_thinking = enable_thinking;
    if (root.is_object()) {
        if (root.contains("tools")) {
            if (!root.at("tools").is_array()) {
                throw std::invalid_argument("messages JSON tools must be an array");
            }
            for (const Json& tool : root.at("tools")) {
                input.options.tool_jsons.push_back(tool.dump());
            }
        }
        if (!root.contains("messages")) {
            throw std::invalid_argument("messages JSON object must contain messages");
        }
        root = root.at("messages");
    }
    if (!root.is_array() || root.empty()) {
        throw std::invalid_argument("messages JSON must be a non-empty array");
    }
    input.messages.reserve(root.size());
    for (std::size_t i = 0; i < root.size(); ++i) {
        input.messages.push_back(parse_message(root.at(i), i));
    }
    return input;
}

} // namespace ninfer::product
