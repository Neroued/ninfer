#include "serve/anthropic_messages.h"
#include "serve/request_validation.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

constexpr std::size_t kMaxToolNameLength = 128;

enum class ParsePurpose {
    Messages,
    CountTokens,
};

[[noreturn]] void invalid_tool_history(std::string message) {
    bad_request(std::move(message), "messages", "invalid_tool_history");
}

[[noreturn]] void invalid_thinking_signature() {
    bad_request("assistant Thinking blocks must include the signature returned by NInfer and be "
                "passed back unmodified",
                "messages", "invalid_thinking_signature");
}

const Json& require_object(const Json& body) {
    if (!body.is_object()) { bad_request("request body must be a JSON object"); }
    return body;
}

std::string require_string(const Json& object, const char* key, const char* param,
                           const char* description) {
    if (!object.contains(key) || !object.at(key).is_string()) {
        bad_request(std::string(description) + " must contain a string '" + key + "'", param);
    }
    return object.at(key).get<std::string>();
}

std::string require_block_type(const Json& block, const char* param) {
    if (!block.is_object() || !block.contains("type") || !block.at("type").is_string()) {
        bad_request("content blocks must be objects with a string 'type'", param);
    }
    return block.at("type").get<std::string>();
}

std::optional<CacheBoundary::Ttl> cache_boundary(const Json& value, const char* param) {
    if (!value.is_object() || !value.contains("cache_control") ||
        value.at("cache_control").is_null()) {
        return std::nullopt;
    }
    const Json& control = value.at("cache_control");
    if (!control.is_object()) {
        bad_request("cache_control must be an object", param, "invalid_cache_control");
    }
    if (!control.contains("type") || !control.at("type").is_string() ||
        control.at("type").get<std::string>() != "ephemeral") {
        bad_request("cache_control.type must be 'ephemeral'", param, "invalid_cache_control");
    }
    if (!control.contains("ttl") || control.at("ttl").is_null()) {
        return CacheBoundary::Ttl::FiveMinutes;
    }
    if (!control.at("ttl").is_string()) {
        bad_request("cache_control.ttl must be '5m' or '1h'", param, "invalid_cache_control");
    }
    const std::string ttl = control.at("ttl").get<std::string>();
    if (ttl == "5m") { return CacheBoundary::Ttl::FiveMinutes; }
    if (ttl == "1h") { return CacheBoundary::Ttl::OneHour; }
    bad_request("cache_control.ttl must be '5m' or '1h'", param, "invalid_cache_control");
}

CacheBoundary explicit_cache_boundary(CacheBoundary::Ttl ttl) {
    return CacheBoundary{.kind     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
                         .evidence = ninfer::SharedCandidateEvidence::ExplicitBoundary,
                         .ttl      = ttl};
}

std::string require_tool_name(const Json& object, const char* param) {
    if (!object.contains("name") || !object.at("name").is_string()) {
        bad_request("tool name must be a string", param);
    }
    std::string name = object.at("name").get<std::string>();
    if (!valid_tool_name(name, kMaxToolNameLength)) {
        bad_request("tool name must match [A-Za-z0-9_-]{1,128}", param);
    }
    return name;
}

ninfer::product::media_acquire::Source parse_image_source(const Json& block) {
    if (!block.contains("source") || !block.at("source").is_object()) {
        bad_request("image block must contain a source object", "messages");
    }
    const Json& source     = block.at("source");
    const std::string type = require_string(source, "type", "messages", "image source");
    ninfer::product::media_acquire::Source result;
    if (type == "base64") {
        result.media_type = require_string(source, "media_type", "messages", "base64 image source");
        const std::string data = require_string(source, "data", "messages", "base64 image source");
        if (data.empty()) { bad_request("base64 image data must not be empty", "messages"); }
        result.kind  = ninfer::product::media_acquire::SourceKind::Data;
        result.value = "data:" + result.media_type + ";base64," + data;
        return result;
    }
    if (type == "url") {
        result.value = require_string(source, "url", "messages", "URL image source");
        if (!result.value.starts_with("http://") && !result.value.starts_with("https://")) {
            bad_request("image URL must use HTTP(S)", "messages");
        }
        result.kind = ninfer::product::media_acquire::SourceKind::Url;
        return result;
    }
    if (type == "file") {
        bad_request("image file sources require an Anthropic Files API, which NInfer does not "
                    "provide",
                    "messages", "files_not_supported");
    }
    bad_request("unsupported image source type: " + type, "messages");
}

ContentPart parse_image(const Json& block) {
    ContentPart result;
    result.kind     = ContentKind::Image;
    result.type_raw = "image";
    result.source   = parse_image_source(block);
    if (const auto ttl = cache_boundary(block, "messages")) {
        result.cache_boundary_after = explicit_cache_boundary(*ttl);
    }
    if (!block.contains("transformations") || block.at("transformations").is_null()) {
        return result;
    }
    const Json& transformations = block.at("transformations");
    if (!transformations.is_object()) {
        bad_request("image transformations must be an object", "messages");
    }
    if (!transformations.contains("oversized_image") ||
        transformations.at("oversized_image").is_null()) {
        return result;
    }
    if (!transformations.at("oversized_image").is_string()) {
        bad_request("transformations.oversized_image must be 'downsize' or 'error'", "messages");
    }
    const std::string policy = transformations.at("oversized_image").get<std::string>();
    if (policy == "downsize") { return result; }
    if (policy == "error") {
        result.image_resize_policy = ninfer::ImageResizePolicy::RejectOversized;
        return result;
    }
    bad_request("transformations.oversized_image must be 'downsize' or 'error'", "messages");
}

ContentPart text_part(std::string value,
                      std::optional<CacheBoundary::Ttl> boundary = std::nullopt) {
    ContentPart part;
    part.kind     = ContentKind::Text;
    part.type_raw = "text";
    part.text     = std::move(value);
    if (boundary) { part.cache_boundary_after = explicit_cache_boundary(*boundary); }
    return part;
}

bool is_server_tool_block(std::string_view type) {
    return type == "server_tool_use" || type == "mcp_tool_use" || type == "mcp_tool_result" ||
           type == "container_upload" || type == "tool_reference" ||
           (type.ends_with("_tool_result") && type != "tool_result");
}

void append_turn(ChatTurn& target, ChatTurn turn) {
    if (target.cache_boundary_after && !target.content.empty()) {
        target.content.back().cache_boundary_after = target.cache_boundary_after;
        target.cache_boundary_after.reset();
    }
    target.content.insert(target.content.end(), std::make_move_iterator(turn.content.begin()),
                          std::make_move_iterator(turn.content.end()));
    target.tool_calls.insert(target.tool_calls.end(),
                             std::make_move_iterator(turn.tool_calls.begin()),
                             std::make_move_iterator(turn.tool_calls.end()));
    target.reasoning_content += turn.reasoning_content;
    target.cache_boundary_after = turn.cache_boundary_after;
}

void merge_turn(GenerationRequest& request, ChatTurn turn) {
    if (request.messages.empty() ||
        (turn.role != ChatRole::User && turn.role != ChatRole::Assistant) ||
        request.messages.back().role != turn.role) {
        request.messages.push_back(std::move(turn));
        return;
    }
    append_turn(request.messages.back(), std::move(turn));
}

struct ParsedToolResult {
    std::string tool_use_id;
    std::vector<ContentPart> content;
    bool is_error = false;
    std::optional<CacheBoundary::Ttl> cache_boundary_after;
};

using ParsedUserBlock = std::variant<ContentPart, ParsedToolResult>;

struct ParsedMessage {
    ChatRole role = ChatRole::User;
    ChatTurn turn;
    std::vector<ParsedUserBlock> user_blocks;
};

std::string require_tool_history_id(const Json& object, const char* key,
                                    std::string_view block_name) {
    if (!object.contains(key) || !object.at(key).is_string() ||
        object.at(key).get_ref<const std::string&>().empty()) {
        invalid_tool_history(std::string(block_name) + " " + key + " must be a non-empty string");
    }
    return object.at(key).get<std::string>();
}

std::vector<ContentPart> parse_tool_result_content(const Json& block) {
    if (!block.contains("content") || block.at("content").is_null()) { return {text_part({})}; }
    const Json& content = block.at("content");
    if (content.is_string()) { return {text_part(content.get<std::string>())}; }
    if (!content.is_array()) {
        bad_request("tool_result content must be a string or array", "messages");
    }
    std::vector<ContentPart> result;
    for (const Json& part : content) {
        const std::string type = require_block_type(part, "messages");
        if (type == "text") {
            result.push_back(text_part(require_string(part, "text", "messages", "text block"),
                                       cache_boundary(part, "messages")));
        } else if (type == "image") {
            result.push_back(parse_image(part));
        } else {
            bad_request("unsupported tool_result content block: " + type, "messages",
                        "content_block_not_supported");
        }
    }
    if (result.empty()) { result.push_back(text_part({})); }
    return result;
}

ToolCall parse_tool_use(const Json& block) {
    ToolCall call;
    call.id   = require_tool_history_id(block, "id", "tool_use block");
    call.name = require_tool_name(block, "messages");
    if (!block.contains("input") || !block.at("input").is_object()) {
        bad_request("tool_use blocks must contain an input object", "messages");
    }
    call.arguments_json = block.at("input").dump();
    return call;
}

ParsedToolResult parse_tool_result(const Json& block) {
    ParsedToolResult result;
    result.tool_use_id = require_tool_history_id(block, "tool_use_id", "tool_result block");
    result.content     = parse_tool_result_content(block);
    if (block.contains("is_error") && !block.at("is_error").is_null()) {
        if (!block.at("is_error").is_boolean()) {
            bad_request("tool_result.is_error must be a boolean", "messages");
        }
        result.is_error = block.at("is_error").get<bool>();
    }
    result.cache_boundary_after = cache_boundary(block, "messages");
    return result;
}

std::vector<ParsedUserBlock> parse_user_blocks(const Json& content) {
    std::vector<ParsedUserBlock> result;
    result.reserve(content.size());
    for (const Json& block : content) {
        const std::string type = require_block_type(block, "messages");
        if (type == "text") {
            result.emplace_back(text_part(require_string(block, "text", "messages", "text block"),
                                          cache_boundary(block, "messages")));
        } else if (type == "image") {
            result.emplace_back(parse_image(block));
        } else if (type == "tool_result") {
            result.emplace_back(parse_tool_result(block));
        } else if (type == "document") {
            bad_request("document blocks require document and citation semantics that NInfer does "
                        "not provide",
                        "messages", "documents_not_supported");
        } else if (type == "search_result") {
            bad_request("search_result blocks require source and citation semantics that NInfer "
                        "does not provide",
                        "messages", "search_results_not_supported");
        } else if (is_server_tool_block(type)) {
            bad_request("server tool result blocks require an executor that NInfer does not "
                        "provide",
                        "messages", "server_tools_not_supported");
        } else {
            bad_request("unsupported user content block: " + type, "messages",
                        "content_block_not_supported");
        }
    }
    return result;
}

ChatTurn parse_assistant_blocks(const Json& content, const AnthropicThinkingSigner& signer) {
    ChatTurn assistant;
    assistant.role = ChatRole::Assistant;
    for (std::size_t index = 0; index < content.size(); ++index) {
        const Json& block      = content[index];
        const std::string type = require_block_type(block, "messages");
        if (type == "text") {
            assistant.content.push_back(
                text_part(require_string(block, "text", "messages", "text block"),
                          cache_boundary(block, "messages")));
        } else if (type == "thinking") {
            if (cache_boundary(block, "messages")) {
                bad_request("cache_control is not valid on thinking blocks", "messages",
                            "invalid_cache_control");
            }
            const std::string thinking =
                require_string(block, "thinking", "messages", "thinking block");
            if (!block.contains("signature") || !block.at("signature").is_string()) {
                invalid_thinking_signature();
            }
            const std::string signature = block.at("signature").get<std::string>();
            if (signature.empty() || !signer.verify(thinking, index, signature)) {
                invalid_thinking_signature();
            }
            assistant.reasoning_content += thinking;
        } else if (type == "redacted_thinking") {
            if (cache_boundary(block, "messages")) {
                bad_request("cache_control is not valid on redacted_thinking blocks", "messages",
                            "invalid_cache_control");
            }
            // Claude-encrypted reasoning is intentionally opaque to the local model.
        } else if (type == "tool_use") {
            assistant.tool_calls.push_back(parse_tool_use(block));
            if (const auto ttl = cache_boundary(block, "messages")) {
                if (index + 1U == content.size()) {
                    assistant.cache_boundary_after = explicit_cache_boundary(*ttl);
                }
                // A valid non-terminal tool_use breakpoint cannot be represented by the Qwen
                // flattened assistant turn. It is advisory, so execution continues without it.
            }
        } else if (is_server_tool_block(type)) {
            bad_request("server tool content blocks require an executor that NInfer does not "
                        "provide",
                        "messages", "server_tools_not_supported");
        } else {
            bad_request("unsupported assistant content block: " + type, "messages",
                        "content_block_not_supported");
        }
    }
    return assistant;
}

ChatTurn parse_system_value(const Json& value, const char* param) {
    ChatTurn system;
    system.role = ChatRole::System;
    if (value.is_string()) {
        system.content.push_back(text_part(value.get<std::string>()));
        return system;
    }
    if (!value.is_array()) {
        bad_request("system content must be a string or an array of text blocks", param);
    }
    for (const Json& block : value) {
        if (require_block_type(block, param) != "text") {
            bad_request("only text system blocks are supported", param,
                        "content_block_not_supported");
        }
        system.content.push_back(
            text_part(require_string(block, "text", param, "system text block"),
                      cache_boundary(block, param)));
    }
    return system;
}

void parse_system(const Json& body, GenerationRequest& request) {
    if (!body.contains("system") || body.at("system").is_null()) { return; }
    request.messages.push_back(parse_system_value(body.at("system"), "system"));
}

std::vector<ParsedMessage> normalize_same_role_messages(std::vector<ParsedMessage> messages) {
    std::vector<ParsedMessage> normalized;
    normalized.reserve(messages.size());
    for (ParsedMessage& message : messages) {
        if (normalized.empty() || message.role == ChatRole::System ||
            normalized.back().role != message.role) {
            normalized.push_back(std::move(message));
            continue;
        }
        ParsedMessage& target = normalized.back();
        if (message.role == ChatRole::User) {
            target.user_blocks.insert(target.user_blocks.end(),
                                      std::make_move_iterator(message.user_blocks.begin()),
                                      std::make_move_iterator(message.user_blocks.end()));
        } else {
            append_turn(target.turn, std::move(message.turn));
        }
    }
    return normalized;
}

std::size_t validate_user_result_prefix(const ParsedMessage& message) {
    bool saw_user_content    = false;
    std::size_t result_count = 0;
    std::unordered_set<std::string> result_ids;
    for (const ParsedUserBlock& block : message.user_blocks) {
        if (std::holds_alternative<ContentPart>(block)) {
            saw_user_content = true;
            continue;
        }
        const ParsedToolResult& result = std::get<ParsedToolResult>(block);
        if (saw_user_content) {
            invalid_tool_history(
                "tool_result blocks must precede all text and image blocks in a user message");
        }
        if (!result_ids.insert(result.tool_use_id).second) {
            invalid_tool_history("duplicate tool_result for tool_use id '" + result.tool_use_id +
                                 "'");
        }
        ++result_count;
    }
    return result_count;
}

void validate_assistant_tool_ids(const ParsedMessage& message) {
    std::unordered_set<std::string> ids;
    for (const ToolCall& call : message.turn.tool_calls) {
        if (!ids.insert(call.id).second) {
            invalid_tool_history("duplicate tool_use id '" + call.id +
                                 "' in one assistant message");
        }
    }
}

void normalize_tool_history(std::vector<ParsedMessage>& messages) {
    std::vector<std::size_t> result_counts(messages.size(), 0);
    std::vector<bool> paired_result_turn(messages.size(), false);
    for (std::size_t index = 0; index < messages.size(); ++index) {
        if (messages[index].role == ChatRole::User) {
            result_counts[index] = validate_user_result_prefix(messages[index]);
        } else if (messages[index].role == ChatRole::Assistant) {
            validate_assistant_tool_ids(messages[index]);
        }
    }

    for (std::size_t index = 0; index < messages.size(); ++index) {
        ParsedMessage& assistant = messages[index];
        if (assistant.role != ChatRole::Assistant || assistant.turn.tool_calls.empty()) {
            continue;
        }
        if (index + 1U >= messages.size() || messages[index + 1U].role != ChatRole::User) {
            invalid_tool_history("tool_result for visible tool_use id '" +
                                 assistant.turn.tool_calls.front().id +
                                 "' must immediately follow its assistant message");
        }

        ParsedMessage& user            = messages[index + 1U];
        const std::size_t result_count = result_counts[index + 1U];
        std::unordered_map<std::string, std::size_t> result_by_id;
        result_by_id.reserve(result_count);
        for (std::size_t result_index = 0; result_index < result_count; ++result_index) {
            const ParsedToolResult& result =
                std::get<ParsedToolResult>(user.user_blocks[result_index]);
            result_by_id.emplace(result.tool_use_id, result_index);
        }

        std::unordered_set<std::string> call_ids;
        call_ids.reserve(assistant.turn.tool_calls.size());
        for (const ToolCall& call : assistant.turn.tool_calls) { call_ids.insert(call.id); }
        for (std::size_t result_index = 0; result_index < result_count; ++result_index) {
            const ParsedToolResult& result =
                std::get<ParsedToolResult>(user.user_blocks[result_index]);
            if (!call_ids.contains(result.tool_use_id)) {
                invalid_tool_history("unknown tool_result id '" + result.tool_use_id +
                                     "' after visible assistant tool_use");
            }
        }
        for (const ToolCall& call : assistant.turn.tool_calls) {
            if (!result_by_id.contains(call.id)) {
                invalid_tool_history("missing tool_result for visible tool_use id '" + call.id +
                                     "'");
            }
        }

        std::vector<ParsedUserBlock> ordered;
        ordered.reserve(user.user_blocks.size());
        for (const ToolCall& call : assistant.turn.tool_calls) {
            ordered.push_back(std::move(user.user_blocks[result_by_id.at(call.id)]));
        }
        for (std::size_t block_index = result_count; block_index < user.user_blocks.size();
             ++block_index) {
            ordered.push_back(std::move(user.user_blocks[block_index]));
        }
        user.user_blocks               = std::move(ordered);
        paired_result_turn[index + 1U] = true;
    }

    for (std::size_t index = 0; index < messages.size(); ++index) {
        if (result_counts[index] == 0 || paired_result_turn[index]) { continue; }
        if (index == 0 && messages[index].role == ChatRole::User) { continue; }
        const ParsedToolResult& result =
            std::get<ParsedToolResult>(messages[index].user_blocks.front());
        invalid_tool_history("tool_result id '" + result.tool_use_id +
                             "' does not immediately follow a visible assistant tool_use");
    }
}

void lower_messages(std::vector<ParsedMessage> messages, GenerationRequest& request) {
    for (ParsedMessage& message : messages) {
        if (message.role != ChatRole::User) {
            merge_turn(request, std::move(message.turn));
            continue;
        }

        ChatTurn user;
        user.role = ChatRole::User;
        for (ParsedUserBlock& block : message.user_blocks) {
            if (std::holds_alternative<ContentPart>(block)) {
                user.content.push_back(std::move(std::get<ContentPart>(block)));
                continue;
            }
            ParsedToolResult& result = std::get<ParsedToolResult>(block);
            ChatTurn tool;
            tool.role                 = ChatRole::Tool;
            tool.tool_call_id         = std::move(result.tool_use_id);
            tool.content              = std::move(result.content);
            tool.tool_result_is_error = result.is_error;
            if (result.cache_boundary_after) {
                tool.cache_boundary_after = explicit_cache_boundary(*result.cache_boundary_after);
            }
            request.messages.push_back(std::move(tool));
        }
        if (!user.content.empty() || message.user_blocks.empty()) {
            merge_turn(request, std::move(user));
        }
    }
}

void parse_messages(const Json& body, GenerationRequest& request,
                    const AnthropicThinkingSigner& signer) {
    if (!body.contains("messages")) { bad_request("missing required field: messages", "messages"); }
    const Json& messages = body.at("messages");
    if (!messages.is_array() || messages.empty()) {
        bad_request("messages must be a non-empty array", "messages");
    }

    std::vector<ChatRole> roles;
    roles.reserve(messages.size());
    for (std::size_t index = 0; index < messages.size(); ++index) {
        const Json& item = messages[index];
        if (!item.is_object() || !item.contains("role") || !item.at("role").is_string()) {
            bad_request("each message must be an object with a string role", "messages");
        }
        const std::string role = item.at("role").get<std::string>();
        if (role == "user") {
            roles.push_back(ChatRole::User);
        } else if (role == "assistant") {
            roles.push_back(ChatRole::Assistant);
        } else if (role == "system") {
            roles.push_back(ChatRole::System);
        } else {
            bad_request("message role must be 'user', 'assistant', or 'system'", "messages",
                        "unsupported_role");
        }
        if (!item.contains("content") || item.at("content").is_null()) {
            bad_request("each message must contain content", "messages");
        }
    }

    for (std::size_t index = 0; index < roles.size();) {
        if (roles[index] != ChatRole::System) {
            ++index;
            continue;
        }
        const std::size_t begin = index;
        while (index < roles.size() && roles[index] == ChatRole::System) { ++index; }
        if (begin == 0 || roles[begin - 1U] != ChatRole::User ||
            (index < roles.size() && roles[index] != ChatRole::Assistant)) {
            bad_request("system messages must follow a user message and be final or precede an "
                        "assistant message",
                        "messages", "invalid_message_order");
        }
    }

    std::vector<ParsedMessage> parsed;
    parsed.reserve(messages.size());
    for (std::size_t index = 0; index < messages.size(); ++index) {
        const Json& content = messages[index].at("content");
        const ChatRole role = roles[index];
        ParsedMessage message;
        message.role      = role;
        message.turn.role = role;
        if (role == ChatRole::System) {
            message.turn = parse_system_value(content, "messages");
            parsed.push_back(std::move(message));
            continue;
        }
        if (content.is_string()) {
            if (role == ChatRole::Assistant) {
                message.turn.content.push_back(text_part(content.get<std::string>()));
            } else {
                message.user_blocks.emplace_back(text_part(content.get<std::string>()));
            }
            parsed.push_back(std::move(message));
            continue;
        }
        if (!content.is_array()) {
            bad_request("message content must be a string or an array", "messages");
        }
        if (role == ChatRole::Assistant) {
            message.turn = parse_assistant_blocks(content, signer);
        } else {
            message.user_blocks = parse_user_blocks(content);
        }
        parsed.push_back(std::move(message));
    }

    parsed = normalize_same_role_messages(std::move(parsed));
    normalize_tool_history(parsed);
    lower_messages(std::move(parsed), request);

    if (!request.messages.empty() && request.messages.back().role == ChatRole::Assistant) {
        const ChatTurn& final = request.messages.back();
        if (final.content.empty() || !final.reasoning_content.empty() ||
            !final.tool_calls.empty() ||
            std::any_of(final.content.begin(), final.content.end(),
                        [](const ContentPart& part) { return part.kind != ContentKind::Text; })) {
            bad_request("a final assistant prefill must contain only text", "messages",
                        "assistant_prefill_not_supported");
        }
        request.continuation = ninfer::PromptContinuationMode::ContinueFinalAssistant;
    }
}

enum class ToolSelectionKind {
    Auto,
    None,
    Any,
    Named,
};

struct ToolSelection {
    ToolSelectionKind kind = ToolSelectionKind::Auto;
    std::string name;
    bool disable_parallel = false;
};

ToolSelection parse_tool_choice(const Json& body) {
    ToolSelection result;
    if (!body.contains("tool_choice") || body.at("tool_choice").is_null()) { return result; }
    const Json& choice = body.at("tool_choice");
    if (!choice.is_object() || !choice.contains("type") || !choice.at("type").is_string()) {
        bad_request("tool_choice must be an object with a string type", "tool_choice");
    }
    const std::string type = choice.at("type").get<std::string>();
    if (type == "auto") {
        result.kind = ToolSelectionKind::Auto;
    } else if (type == "none") {
        result.kind = ToolSelectionKind::None;
    } else if (type == "any") {
        result.kind = ToolSelectionKind::Any;
    } else if (type == "tool") {
        result.kind = ToolSelectionKind::Named;
        result.name = require_tool_name(choice, "tool_choice");
    } else {
        bad_request("unsupported tool_choice type: " + type, "tool_choice");
    }
    result.disable_parallel = optional_bool(choice, "disable_parallel_tool_use", false);
    return result;
}

enum class ToolSource {
    UserDefined,
    AnthropicProvided,
    Toolset,
};

struct ParsedTool {
    ToolDefinition definition;
    ToolSource source = ToolSource::UserDefined;
    std::string source_type;
    bool strict        = false;
    bool defer_loading = false;
    std::optional<std::vector<std::string>> allowed_callers;
};

std::vector<ParsedTool> parse_tool_definitions(const Json& body) {
    std::vector<ParsedTool> result;
    if (!body.contains("tools") || body.at("tools").is_null()) { return result; }
    if (!body.at("tools").is_array()) { bad_request("tools must be an array", "tools"); }
    std::unordered_set<std::string> names;
    for (const Json& item : body.at("tools")) {
        if (!item.is_object()) { bad_request("tools entries must be objects", "tools"); }
        ParsedTool parsed;
        if (item.contains("type") && !item.at("type").is_null()) {
            if (!item.at("type").is_string()) {
                bad_request("tool type must be a string or null", "tools");
            }
            parsed.source_type = item.at("type").get<std::string>();
            if (parsed.source_type == "custom") {
                parsed.source = ToolSource::UserDefined;
            } else if (parsed.source_type == "toolset") {
                parsed.source = ToolSource::Toolset;
            } else {
                parsed.source = ToolSource::AnthropicProvided;
            }
        }

        if (parsed.source == ToolSource::UserDefined) {
            parsed.definition.name = require_tool_name(item, "tools");
            if (!names.insert(parsed.definition.name).second) {
                bad_request("duplicate tool name: " + parsed.definition.name, "tools");
            }
            if (item.contains("description") && !item.at("description").is_null()) {
                if (!item.at("description").is_string()) {
                    bad_request("tool description must be a string", "tools");
                }
                parsed.definition.description = item.at("description").get<std::string>();
            }
            if (!item.contains("input_schema") || !item.at("input_schema").is_object()) {
                bad_request("tool input_schema must be a JSON object", "tools");
            }
            parsed.definition.input_schema_json = item.at("input_schema").dump();
            if (item.contains("input_examples") && !item.at("input_examples").is_null()) {
                if (!item.at("input_examples").is_array()) {
                    bad_request("tool input_examples must be an array", "tools");
                }
                parsed.definition.input_examples_json = item.at("input_examples").dump();
            }
            if (const auto ttl = cache_boundary(item, "tools")) {
                parsed.definition.cache_boundary_after = explicit_cache_boundary(*ttl);
            }
        } else if (item.contains("name") && item.at("name").is_string()) {
            parsed.definition.name = item.at("name").get<std::string>();
        }

        if (item.contains("strict") && !item.at("strict").is_null()) {
            if (!item.at("strict").is_boolean()) {
                bad_request("tool strict must be a boolean", "tools");
            }
            parsed.strict = item.at("strict").get<bool>();
        }
        if (item.contains("defer_loading") && !item.at("defer_loading").is_null()) {
            if (!item.at("defer_loading").is_boolean()) {
                bad_request("tool defer_loading must be a boolean", "tools");
            }
            parsed.defer_loading = item.at("defer_loading").get<bool>();
        }
        if (item.contains("eager_input_streaming") && !item.at("eager_input_streaming").is_null() &&
            !item.at("eager_input_streaming").is_boolean()) {
            bad_request("tool eager_input_streaming must be a boolean", "tools");
        }
        if (item.contains("allowed_callers") && !item.at("allowed_callers").is_null()) {
            if (!item.at("allowed_callers").is_array()) {
                bad_request("tool allowed_callers must be an array", "tools");
            }
            parsed.allowed_callers.emplace();
            for (const Json& caller : item.at("allowed_callers")) {
                if (!caller.is_string()) {
                    bad_request("tool allowed_callers entries must be strings", "tools");
                }
                parsed.allowed_callers->push_back(caller.get<std::string>());
            }
        }
        result.push_back(std::move(parsed));
    }
    return result;
}

void lower_tools(const Json& body, GenerationRequest& request) {
    const ToolSelection selection       = parse_tool_choice(body);
    std::vector<ParsedTool> definitions = parse_tool_definitions(body);
    const auto named                    = [&](const ParsedTool& tool) {
        return tool.definition.name == selection.name;
    };

    if (selection.kind == ToolSelectionKind::Named) {
        if (std::none_of(definitions.begin(), definitions.end(), named)) {
            bad_request("tool_choice references unknown tool: " + selection.name, "tool_choice");
        }
        bad_request("tool_choice.type='tool' requires that exact tool to be called, which NInfer "
                    "cannot guarantee",
                    "tool_choice", "tool_choice_not_supported");
    }
    if (selection.kind == ToolSelectionKind::Any) {
        if (definitions.empty()) { bad_request("tool_choice requires tools", "tool_choice"); }
        bad_request("tool_choice.type='any' requires at least one tool call, which NInfer cannot "
                    "guarantee",
                    "tool_choice", "tool_choice_not_supported");
    }

    request.tool_choice.mode =
        selection.kind == ToolSelectionKind::None ? ToolChoiceMode::None : ToolChoiceMode::Auto;
    if (selection.kind == ToolSelectionKind::None) {
        for (ParsedTool& tool : definitions) {
            if (tool.source == ToolSource::UserDefined) {
                request.tools.push_back(std::move(tool.definition));
            }
        }
        return;
    }

    for (ParsedTool& tool : definitions) {
        if (tool.source == ToolSource::Toolset) {
            bad_request("Anthropic toolsets require a tool loader that NInfer does not provide",
                        "tools", "toolsets_not_supported");
        }
        if (tool.source == ToolSource::AnthropicProvided) {
            bad_request("Anthropic-provided tool type '" + tool.source_type +
                            "' requires its predefined prompt schema or server executor, which "
                            "NInfer does not provide",
                        "tools", "anthropic_tools_not_supported");
        }
        if (tool.strict) {
            bad_request("strict=true requires generated tool input to satisfy the declared JSON "
                        "Schema, which NInfer cannot guarantee",
                        "tools", "strict_tools_not_supported");
        }
        if (tool.defer_loading) {
            bad_request("defer_loading=true requires a deferred tool loader that NInfer does not "
                        "provide",
                        "tools", "deferred_tools_not_supported");
        }
        if (tool.allowed_callers &&
            std::find(tool.allowed_callers->begin(), tool.allowed_callers->end(), "direct") ==
                tool.allowed_callers->end()) {
            bad_request("tool allowed_callers excludes direct model calls, and NInfer provides no "
                        "alternate caller",
                        "tools", "tool_caller_not_supported");
        }
        request.tools.push_back(std::move(tool.definition));
    }
    if (selection.disable_parallel && !request.tools.empty()) {
        bad_request("disable_parallel_tool_use=true requires at most one tool call, which NInfer "
                    "cannot guarantee",
                    "tool_choice", "parallel_tool_use_not_supported");
    }
}

void parse_thinking(const Json& body, GenerationRequest& request, ParsePurpose purpose,
                    int effective_max_tokens) {
    if (!body.contains("thinking") || body.at("thinking").is_null()) { return; }
    const Json& thinking = body.at("thinking");
    if (!thinking.is_object() || !thinking.contains("type") || !thinking.at("type").is_string()) {
        bad_request("thinking must be an object with a string type", "thinking");
    }
    const std::string type = thinking.at("type").get<std::string>();
    if (type == "disabled") {
        request.enable_thinking = false;
    } else if (type == "adaptive") {
        request.enable_thinking = true;
    } else if (type == "enabled") {
        request.enable_thinking         = true;
        const std::optional<int> budget = optional_int(thinking, "budget_tokens");
        if (!budget || *budget < 1024) {
            bad_request("thinking.budget_tokens must be an integer of at least 1024", "thinking");
        }
        if (purpose == ParsePurpose::Messages && *budget >= effective_max_tokens) {
            bad_request("thinking.budget_tokens must be less than max_tokens", "thinking");
        }
        request.thinking_budget = static_cast<std::uint32_t>(*budget);
    } else {
        bad_request("thinking.type must be 'disabled', 'adaptive', or 'enabled'", "thinking");
    }

    if (thinking.contains("display") && !thinking.at("display").is_null()) {
        if (!thinking.at("display").is_string()) {
            bad_request("thinking.display must be 'summarized' or 'omitted'", "thinking");
        }
        const std::string display = thinking.at("display").get<std::string>();
        if (display != "summarized" && display != "omitted") {
            bad_request("thinking.display must be 'summarized' or 'omitted'", "thinking");
        }
        if (type == "disabled") {
            bad_request("thinking.display is valid only when thinking is adaptive or enabled",
                        "thinking");
        }
        if (purpose == ParsePurpose::Messages && display == "omitted") {
            bad_request("thinking.display='omitted' requires encrypted hidden-reasoning restore "
                        "semantics that NInfer does not provide",
                        "thinking", "thinking_display_not_supported");
        }
    }
}

void parse_effort(const Json& body, GenerationRequest& request, ParsePurpose purpose) {
    if (!body.contains("output_config") || body.at("output_config").is_null()) { return; }
    const Json& config = body.at("output_config");
    if (!config.is_object()) { bad_request("output_config must be an object", "output_config"); }
    if (purpose == ParsePurpose::Messages && config.contains("format") &&
        !config.at("format").is_null()) {
        bad_request("output_config.format requires constrained decoding, which NInfer does not "
                    "provide",
                    "output_config.format", "output_config_format_not_supported");
    }
    if (!config.contains("effort") || config.at("effort").is_null()) { return; }
    if (!config.at("effort").is_string()) {
        bad_request("output_config.effort must be a string", "output_config.effort");
    }
    const std::string value = config.at("effort").get<std::string>();
    const auto effort       = parse_requested_reasoning_effort(value);
    if (!effort) {
        bad_request("output_config.effort is not a recognized effort value",
                    "output_config.effort");
    }
    request.reasoning_effort = *effort;
}

void parse_generation_fields(const Json& body, GenerationRequest& request) {
    if (body.contains("stop_sequences") && !body.at("stop_sequences").is_null()) {
        if (!body.at("stop_sequences").is_array()) {
            bad_request("stop_sequences must be an array of strings", "stop_sequences");
        }
        for (const Json& value : body.at("stop_sequences")) {
            if (!value.is_string() || value.get<std::string>().empty()) {
                bad_request("stop_sequences entries must be non-empty strings", "stop_sequences");
            }
            request.stop_strings.push_back(value.get<std::string>());
        }
    }

    request.sampling.temperature = optional_number(body, "temperature");
    request.sampling.top_p       = optional_number(body, "top_p");
    request.sampling.top_k       = optional_int(body, "top_k");
    if (request.sampling.temperature &&
        (*request.sampling.temperature < 0.0 || *request.sampling.temperature > 1.0)) {
        bad_request("temperature must be in [0,1]", "temperature");
    }
    if (request.sampling.top_p &&
        (*request.sampling.top_p < 0.0 || *request.sampling.top_p > 1.0)) {
        bad_request("top_p must be in [0,1]", "top_p");
    }
    if (request.sampling.top_k && (*request.sampling.top_k < 0 || *request.sampling.top_k > 20)) {
        bad_request("top_k must be in [0,20]", "top_k");
    }
}

std::string parse_model(const Json& body) {
    if (!body.contains("model") || !body.at("model").is_string() ||
        body.at("model").get<std::string>().empty()) {
        bad_request("missing required field: model", "model");
    }
    return body.at("model").get<std::string>();
}

void apply_anthropic_prompt_cache_policy(const Json& body, GenerationRequest& request) {
    std::vector<std::optional<CacheBoundary>*> explicit_boundaries;
    for (ToolDefinition& tool : request.tools) {
        if (tool.cache_boundary_after) {
            explicit_boundaries.push_back(&tool.cache_boundary_after);
        }
    }
    for (ChatTurn& turn : request.messages) {
        for (ContentPart& part : turn.content) {
            if (part.cache_boundary_after) {
                explicit_boundaries.push_back(&part.cache_boundary_after);
            }
        }
        if (turn.cache_boundary_after) {
            explicit_boundaries.push_back(&turn.cache_boundary_after);
        }
    }
    if (explicit_boundaries.size() > 4U) {
        bad_request("at most four block-level cache_control breakpoints are supported per request",
                    "cache_control", "too_many_cache_breakpoints");
    }

    const std::optional<CacheBoundary::Ttl> automatic_ttl = cache_boundary(body, "cache_control");
    if (!automatic_ttl) { return; }
    request.allow_engine_automatic_shared_prefixes = false;

    std::optional<CacheBoundary>* automatic_target = nullptr;
    for (auto turn = request.messages.rbegin(); turn != request.messages.rend(); ++turn) {
        if (!turn->tool_calls.empty()) {
            automatic_target = &turn->cache_boundary_after;
            break;
        }
        if (!turn->content.empty()) {
            automatic_target = &turn->content.back().cache_boundary_after;
            break;
        }
    }
    if (automatic_target == nullptr && !request.tools.empty()) {
        automatic_target = &request.tools.back().cache_boundary_after;
    }
    if (automatic_target == nullptr) { return; }

    if (*automatic_target) {
        if (automatic_target->value().ttl != *automatic_ttl) {
            bad_request("request-level cache_control and the final block-level cache_control "
                        "target the same boundary with different ttl values",
                        "cache_control", "conflicting_cache_ttl");
        }
        automatic_target->value().evidence |= ninfer::SharedCandidateEvidence::RequestedAutomatic;
        return;
    }
    if (explicit_boundaries.size() == 4U) {
        bad_request("request-level cache_control needs a fifth distinct cache write after four "
                    "block-level breakpoints",
                    "cache_control", "too_many_cache_breakpoints");
    }
    *automatic_target =
        CacheBoundary{.kind     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
                      .evidence = ninfer::SharedCandidateEvidence::RequestedAutomatic,
                      .ttl      = *automatic_ttl};
}

void parse_common_prompt(const Json& body, GenerationRequest& request, ParsePurpose purpose,
                         int effective_max_tokens, const AnthropicThinkingSigner& signer) {
    lower_tools(body, request);
    parse_system(body, request);
    parse_messages(body, request, signer);
    parse_thinking(body, request, purpose, effective_max_tokens);
    parse_effort(body, request, purpose);
    apply_anthropic_prompt_cache_policy(body, request);
    if (body.contains("container") && !body.at("container").is_null()) {
        bad_request("container requires an external execution environment that NInfer does not "
                    "provide",
                    "container", "container_not_supported");
    }
    if (body.contains("preserve_thinking") && !body.at("preserve_thinking").is_null()) {
        if (!body.at("preserve_thinking").is_boolean()) {
            bad_request("preserve_thinking must be a boolean", "preserve_thinking");
        }
        request.preserve_thinking = body.at("preserve_thinking").get<bool>();
    }
}

} // namespace

AnthropicMessagesRequest parse_anthropic_messages_request(const Json& body,
                                                          const RequestLimits& limits,
                                                          const AnthropicThinkingSigner& signer) {
    require_object(body);
    AnthropicMessagesRequest result;
    result.model                           = parse_model(body);
    result.generation.tool_name_max_length = kMaxToolNameLength;
    result.stream                          = optional_bool(body, "stream", false);

    const std::optional<int> max_tokens = optional_int(body, "max_tokens");
    if (max_tokens) {
        result.output_tokens_explicit = true;
        if (*max_tokens == 0) {
            bad_request("max_tokens=0 requires a completed cache prewarm lifecycle that NInfer "
                        "does not provide",
                        "max_tokens", "cache_prewarm_not_supported");
        }
        if (*max_tokens < 0) { bad_request("max_tokens must be positive", "max_tokens"); }
        result.generation.max_tokens = *max_tokens;
    } else {
        result.generation.max_tokens = limits.default_max_tokens;
    }

    parse_common_prompt(body, result.generation, ParsePurpose::Messages,
                        result.generation.max_tokens, signer);
    parse_generation_fields(body, result.generation);
    return result;
}

AnthropicCountTokensRequest
parse_anthropic_count_tokens_request(const Json& body, const AnthropicThinkingSigner& signer) {
    require_object(body);
    AnthropicCountTokensRequest result;
    result.model                           = parse_model(body);
    result.generation.tool_name_max_length = kMaxToolNameLength;
    parse_common_prompt(body, result.generation, ParsePurpose::CountTokens,
                        std::numeric_limits<int>::max(), signer);
    return result;
}

} // namespace ninfer::serve
