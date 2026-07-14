#include "serve/anthropic_schema.h"

#include <array>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <utility>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

[[noreturn]] void bad_request(std::string message, std::string param = {}, std::string code = {}) {
    ApiError error;
    error.status  = 400;
    error.type    = "invalid_request_error";
    error.message = std::move(message);
    error.param   = std::move(param);
    error.code    = std::move(code);
    throw ApiException(std::move(error));
}

const Json& require_object(const Json& body) {
    if (!body.is_object()) { bad_request("request body must be a JSON object"); }
    return body;
}

bool get_bool(const Json& obj, const char* key, bool fallback) {
    if (!obj.contains(key) || obj.at(key).is_null()) { return fallback; }
    if (!obj.at(key).is_boolean()) { bad_request(std::string(key) + " must be a boolean", key); }
    return obj.at(key).get<bool>();
}

std::optional<double> get_number(const Json& obj, const char* key) {
    if (!obj.contains(key) || obj.at(key).is_null()) { return std::nullopt; }
    if (!obj.at(key).is_number()) { bad_request(std::string(key) + " must be a number", key); }
    return obj.at(key).get<double>();
}

std::optional<int> get_int(const Json& obj, const char* key) {
    if (!obj.contains(key) || obj.at(key).is_null()) { return std::nullopt; }
    if (!obj.at(key).is_number_integer()) {
        bad_request(std::string(key) + " must be an integer", key);
    }
    if (obj.at(key).is_number_unsigned()) {
        const std::uint64_t value = obj.at(key).get<std::uint64_t>();
        if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
            bad_request(std::string(key) + " is out of range", key);
        }
        return static_cast<int>(value);
    }
    const std::int64_t value = obj.at(key).get<std::int64_t>();
    if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max()) {
        bad_request(std::string(key) + " is out of range", key);
    }
    return static_cast<int>(value);
}

bool is_valid_function_name(const std::string& name) {
    if (name.empty() || name.size() > 64) { return false; }
    for (const unsigned char c : name) {
        if (std::isalnum(c) == 0 && c != '_' && c != '-') { return false; }
    }
    return true;
}

std::string require_function_name(const Json& obj, const char* param) {
    if (!obj.contains("name") || !obj.at("name").is_string()) {
        bad_request("tool name must be a string", param);
    }
    std::string name = obj.at("name").get<std::string>();
    if (!is_valid_function_name(name)) {
        bad_request("tool name must match [A-Za-z0-9_-]{1,64}", param);
    }
    return name;
}

bool has_tool_named(const GenerationRequest& req, const std::string& name) {
    for (const ToolDefinition& tool : req.tools) {
        if (tool.name == name) { return true; }
    }
    return false;
}

void append_text(std::string& acc, const std::string& piece) {
    if (piece.empty()) { return; }
    if (!acc.empty()) { acc += '\n'; }
    acc += piece;
}

std::string require_block_type(const Json& block, const char* param) {
    if (!block.is_object() || !block.contains("type") || !block.at("type").is_string()) {
        bad_request("content blocks must be objects with a string 'type'", param);
    }
    return block.at("type").get<std::string>();
}

std::string require_string_field(const Json& block, const char* field, const char* what) {
    if (!block.contains(field) || !block.at(field).is_string()) {
        bad_request(std::string(what) + " must contain a string '" + field + "'", "messages");
    }
    return block.at(field).get<std::string>();
}

ninfer::product::media_acquire::Source parse_image_source(const Json& block) {
    if (!block.contains("source") || !block.at("source").is_object()) {
        bad_request("image block must contain a source object", "messages");
    }
    const Json& source     = block.at("source");
    const std::string type = require_string_field(source, "type", "image source");
    ninfer::product::media_acquire::Source out;
    if (type == "base64") {
        out.media_type         = require_string_field(source, "media_type", "base64 image source");
        const std::string data = require_string_field(source, "data", "base64 image source");
        if (data.empty()) { bad_request("base64 image data must not be empty", "messages"); }
        out.kind  = ninfer::product::media_acquire::SourceKind::Data;
        out.value = "data:" + out.media_type + ";base64," + data;
    } else if (type == "url") {
        out.value = require_string_field(source, "url", "URL image source");
        if (!out.value.starts_with("http://") && !out.value.starts_with("https://")) {
            bad_request("image URL must use HTTP(S)", "messages");
        }
        out.kind = ninfer::product::media_acquire::SourceKind::Url;
    } else {
        bad_request("unsupported image source type: " + type, "messages");
    }
    return out;
}

// --- request parsing --------------------------------------------------------

void parse_tools(const Json& body, GenerationRequest& out) {
    if (!body.contains("tools") || body.at("tools").is_null()) { return; }
    const Json& tools = body.at("tools");
    if (!tools.is_array()) { bad_request("tools must be an array", "tools"); }
    out.tools.reserve(tools.size());
    for (const Json& item : tools) {
        if (!item.is_object()) { bad_request("tools entries must be objects", "tools"); }
        // Anthropic server/built-in tools carry a `type` and no `input_schema`; we
        // only support client tools (name + input_schema), which is what Claude
        // Code sends for its own tools.
        if (!item.contains("input_schema") || item.at("input_schema").is_null()) {
            bad_request("only client tools with an input_schema are supported", "tools",
                        "tool_type_not_supported");
        }
        if (!item.at("input_schema").is_object()) {
            bad_request("tool input_schema must be a JSON object", "tools");
        }
        ToolDefinition tool;
        tool.name     = require_function_name(item, "tools");
        Json function = Json{{"name", tool.name}};
        if (item.contains("description") && !item.at("description").is_null()) {
            if (!item.at("description").is_string()) {
                bad_request("tool description must be a string", "tools");
            }
            tool.description        = item.at("description").get<std::string>();
            function["description"] = tool.description;
        }
        const Json& parameters = item.at("input_schema");
        tool.parameters_json   = parameters.dump();
        function["parameters"] = parameters;
        function["strict"]     = false;
        // Normalized OpenAI-style function tool object consumed verbatim by the
        // Qwen chat template's <tools> block (render_tools_system_block).
        tool.definition_json = Json{{"type", "function"}, {"function", std::move(function)}}.dump();
        out.tools.push_back(std::move(tool));
    }
}

void parse_tool_choice(const Json& body, GenerationRequest& out) {
    if (!body.contains("tool_choice") || body.at("tool_choice").is_null()) { return; }
    const Json& choice = body.at("tool_choice");
    if (!choice.is_object() || !choice.contains("type") || !choice.at("type").is_string()) {
        bad_request("tool_choice must be an object with a string type", "tool_choice");
    }
    const std::string type = choice.at("type").get<std::string>();
    if (type == "auto") {
        out.tool_choice.mode = ToolChoiceMode::Auto;
    } else if (type == "any") {
        out.tool_choice.mode = ToolChoiceMode::Required;
    } else if (type == "none") {
        out.tool_choice.mode = ToolChoiceMode::None;
    } else if (type == "tool") {
        out.tool_choice.mode = ToolChoiceMode::Named;
        out.tool_choice.name = require_function_name(choice, "tool_choice");
    } else {
        bad_request("unsupported tool_choice type: " + type, "tool_choice");
    }
    if (out.tool_choice.mode != ToolChoiceMode::None && out.tools.empty()) {
        bad_request("tool_choice requires tools", "tool_choice");
    }
    if (out.tool_choice.mode == ToolChoiceMode::Named &&
        !has_tool_named(out, out.tool_choice.name)) {
        bad_request("tool_choice references unknown tool: " + out.tool_choice.name, "tool_choice");
    }
}

// Flatten a system value (string or array of text blocks) into plain text. Used
// for both the top-level `system` field and any system-role turn Claude Code
// injects into the messages array.
std::string flatten_system_value(const Json& value, const char* param) {
    std::string text;
    if (value.is_string()) {
        text = value.get<std::string>();
    } else if (value.is_array()) {
        for (const Json& block : value) {
            if (require_block_type(block, param) != "text") {
                bad_request("only text system blocks are supported", param);
            }
            append_text(text, require_string_field(block, "text", "system text block"));
        }
    } else {
        bad_request("system content must be a string or an array of text blocks", param);
    }
    return text;
}

void parse_system(const Json& body, std::string& system_text) {
    if (!body.contains("system") || body.at("system").is_null()) { return; }
    append_text(system_text, flatten_system_value(body.at("system"), "system"));
}

ToolCall parse_tool_use(const Json& block) {
    ToolCall call;
    if (!block.contains("id") || !block.at("id").is_string() ||
        block.at("id").get<std::string>().empty()) {
        bad_request("tool_use blocks must contain a string id", "messages");
    }
    call.id   = block.at("id").get<std::string>();
    call.name = require_function_name(block, "messages");
    if (!block.contains("input") || !block.at("input").is_object()) {
        bad_request("tool_use blocks must contain an input object", "messages");
    }
    call.arguments_json = block.at("input").dump();
    return call;
}

std::vector<ContentPart> parse_tool_result_content(const Json& block) {
    if (!block.contains("content") || block.at("content").is_null()) {
        return {ContentPart{ContentKind::Text, {}, "text"}};
    }
    const Json& content = block.at("content");
    if (content.is_string()) {
        return {ContentPart{ContentKind::Text, content.get<std::string>(), "text"}};
    }
    if (content.is_array()) {
        std::vector<ContentPart> parts;
        for (const Json& part : content) {
            const std::string type = require_block_type(part, "messages");
            if (type == "text") {
                parts.push_back(ContentPart{
                    ContentKind::Text, require_string_field(part, "text", "text block"), "text"});
            } else if (type == "image") {
                ContentPart image;
                image.kind     = ContentKind::Image;
                image.type_raw = "image";
                image.source   = parse_image_source(part);
                parts.push_back(std::move(image));
            } else {
                bad_request("unsupported tool_result content block: " + type, "messages");
            }
        }
        return parts;
    }
    bad_request("tool_result content must be a string or array of blocks", "messages");
    return {};
}

void parse_assistant_content(const Json& content, GenerationRequest& out) {
    ChatTurn turn;
    turn.role = "assistant";
    std::string text;
    for (const Json& block : content) {
        const std::string type = require_block_type(block, "messages");
        if (type == "text") {
            append_text(text, require_string_field(block, "text", "text block"));
        } else if (type == "thinking") {
            if (block.contains("thinking") && block.at("thinking").is_string()) {
                append_text(turn.reasoning_content, block.at("thinking").get<std::string>());
            }
        } else if (type == "redacted_thinking") {
            // Opaque redacted reasoning: nothing usable for the text frontend.
        } else if (type == "tool_use") {
            turn.tool_calls.push_back(parse_tool_use(block));
        } else {
            bad_request("unsupported assistant content block: " + type, "messages");
        }
    }
    if (!text.empty()) {
        turn.content.push_back(ContentPart{ContentKind::Text, std::move(text), "text"});
    }
    if (turn.content.empty() && turn.tool_calls.empty() && turn.reasoning_content.empty()) {
        bad_request("assistant message must have text, thinking, or tool_use content", "messages");
    }
    out.messages.push_back(std::move(turn));
}

void parse_user_content(const Json& content, GenerationRequest& out) {
    // A user turn may carry tool_result blocks (each -> its own tool turn) and/or
    // text blocks (folded into one user turn emitted after the tool turns, matching
    // the Qwen template's assistant-tool_calls -> tool-responses -> user order).
    ChatTurn user_turn;
    user_turn.role = "user";
    for (const Json& block : content) {
        const std::string type = require_block_type(block, "messages");
        if (type == "text") {
            std::string text = require_string_field(block, "text", "text block");
            user_turn.content.push_back(ContentPart{ContentKind::Text, std::move(text), "text"});
        } else if (type == "tool_result") {
            if (!block.contains("tool_use_id") || !block.at("tool_use_id").is_string() ||
                block.at("tool_use_id").get<std::string>().empty()) {
                bad_request("tool_result blocks must contain a string tool_use_id", "messages");
            }
            ChatTurn tool_turn;
            tool_turn.role         = "tool";
            tool_turn.tool_call_id = block.at("tool_use_id").get<std::string>();
            tool_turn.content      = parse_tool_result_content(block);
            out.messages.push_back(std::move(tool_turn));
        } else if (type == "image") {
            ContentPart part;
            part.kind     = ContentKind::Image;
            part.type_raw = "image";
            part.source   = parse_image_source(block);
            user_turn.content.push_back(std::move(part));
        } else {
            bad_request("unsupported user content block: " + type, "messages");
        }
    }
    if (!user_turn.content.empty()) { out.messages.push_back(std::move(user_turn)); }
}

void parse_messages(const Json& body, GenerationRequest& out, std::string& system_text) {
    if (!body.contains("messages")) { bad_request("missing required field: messages", "messages"); }
    const Json& messages = body.at("messages");
    if (!messages.is_array() || messages.empty()) {
        bad_request("messages must be a non-empty array", "messages");
    }
    for (std::size_t i = 0; i < messages.size(); ++i) {
        const Json& item = messages.at(i);
        if (!item.is_object()) {
            bad_request("message " + std::to_string(i) + " must be an object", "messages");
        }
        if (!item.contains("role") || !item.at("role").is_string()) {
            bad_request("message " + std::to_string(i) + " must have a string role", "messages");
        }
        const std::string role = item.at("role").get<std::string>();
        if (role != "user" && role != "assistant" && role != "system") {
            bad_request("message role must be 'user', 'assistant', or 'system'", "messages",
                        "unsupported_role");
        }
        if (!item.contains("content") || item.at("content").is_null()) {
            bad_request("message " + std::to_string(i) + " must have content", "messages");
        }
        const Json& content = item.at("content");
        if (role == "system") {
            // Claude Code injects system reminders as system-role messages inside
            // the messages array. Fold them into the leading system block: the Qwen
            // chat template only honors leading system turns and drops the rest.
            append_text(system_text, flatten_system_value(content, "messages"));
            continue;
        }
        if (content.is_string()) {
            ChatTurn turn;
            turn.role = role;
            turn.content.push_back(
                ContentPart{ContentKind::Text, content.get<std::string>(), "text"});
            out.messages.push_back(std::move(turn));
        } else if (content.is_array()) {
            if (role == "assistant") {
                parse_assistant_content(content, out);
            } else {
                parse_user_content(content, out);
            }
        } else {
            bad_request("message " + std::to_string(i) + " content must be a string or array",
                        "messages");
        }
    }
}

void parse_stop_sequences(const Json& body, GenerationRequest& out) {
    if (!body.contains("stop_sequences") || body.at("stop_sequences").is_null()) { return; }
    const Json& stop = body.at("stop_sequences");
    if (!stop.is_array()) {
        bad_request("stop_sequences must be an array of strings", "stop_sequences");
    }
    for (const Json& s : stop) {
        if (!s.is_string()) {
            bad_request("stop_sequences entries must be strings", "stop_sequences");
        }
        if (!s.get<std::string>().empty()) { out.stop_strings.push_back(s.get<std::string>()); }
    }
}

void parse_sampling(const Json& body, GenerationRequest& out) {
    SamplingParams& s = out.sampling;
    s.temperature     = get_number(body, "temperature");
    s.top_p           = get_number(body, "top_p");
    s.top_k           = get_int(body, "top_k");
    // The Anthropic API has no presence/frequency penalty or seed; those keep the
    // server defaults resolved in translate.cpp.
}

void parse_thinking(const Json& body, GenerationRequest& out) {
    if (!body.contains("thinking") || body.at("thinking").is_null()) { return; }
    const Json& thinking = body.at("thinking");
    if (!thinking.is_object() || !thinking.contains("type") || !thinking.at("type").is_string()) {
        bad_request("thinking must be an object with a string type", "thinking");
    }
    // Anthropic thinking modes: "disabled" turns reasoning off; "enabled" and
    // "adaptive" (Claude Code's default extended-thinking mode) turn it on. The
    // Qwen template only exposes an on/off toggle, so any non-"disabled" mode maps
    // to thinking-on. Unknown future modes default to on rather than 400 so the
    // adapter tolerates Claude Code's evolving thinking vocabulary.
    const std::string type = thinking.at("type").get<std::string>();
    out.enable_thinking    = (type != "disabled");
}

const char* anthropic_error_type(int status) {
    switch (status) {
    case 400:
        return "invalid_request_error";
    case 401:
        return "authentication_error";
    case 403:
        return "permission_error";
    case 404:
        return "not_found_error";
    case 429:
        return "rate_limit_error";
    default:
        return status >= 500 ? "api_error" : "invalid_request_error";
    }
}

std::string sse(const char* type, const Json& payload) {
    return std::string("event: ") + type + "\ndata: " + payload.dump() + "\n\n";
}

} // namespace

GenerationRequest parse_messages_request(const Json& body, const RequestLimits& limits) {
    require_object(body);

    GenerationRequest out;
    if (!body.contains("model") || !body.at("model").is_string() ||
        body.at("model").get<std::string>().empty()) {
        bad_request("missing required field: model", "model");
    }
    out.model = body.at("model").get<std::string>();

    parse_tools(body, out);
    parse_tool_choice(body, out);
    std::string system_text;
    parse_system(body, system_text);
    parse_messages(body, out, system_text);
    // The leading system turn combines the top-level `system` field with any
    // system-role reminders folded out of the messages array.
    if (!system_text.empty()) {
        ChatTurn turn;
        turn.role = "system";
        turn.content.push_back(ContentPart{ContentKind::Text, std::move(system_text), "text"});
        out.messages.insert(out.messages.begin(), std::move(turn));
    }
    parse_stop_sequences(body, out);
    parse_sampling(body, out);
    parse_thinking(body, out);

    out.stream = get_bool(body, "stream", false);

    const std::optional<int> max_tokens = get_int(body, "max_tokens");
    if (max_tokens) {
        if (*max_tokens <= 0) { bad_request("max_tokens must be positive", "max_tokens"); }
        out.max_tokens     = *max_tokens;
        out.max_tokens_set = true;
    } else {
        out.max_tokens     = limits.default_max_tokens;
        out.max_tokens_set = false;
    }
    return out;
}

const char* messages_stop_reason(ninfer::FinishReason reason, bool has_tool_calls) {
    if (has_tool_calls) { return "tool_use"; }
    switch (reason) {
    case ninfer::FinishReason::OutputLimit:
    case ninfer::FinishReason::ContextCapacity:
        return "max_tokens";
    case ninfer::FinishReason::None:
    case ninfer::FinishReason::StopToken:
    case ninfer::FinishReason::StopString:
    case ninfer::FinishReason::Cancelled:
        return "end_turn";
    }
    return "end_turn";
}

std::string make_messages_response(const std::string& id, const std::string& model,
                                   const std::string& content, const std::string& reasoning,
                                   const std::vector<ToolCall>& tool_calls, const char* stop_reason,
                                   const CompletionUsage& usage) {
    Json blocks = Json::array();
    if (!reasoning.empty()) {
        blocks.push_back(Json{{"type", "thinking"}, {"thinking", reasoning}, {"signature", ""}});
    }
    if (!content.empty()) { blocks.push_back(Json{{"type", "text"}, {"text", content}}); }
    for (const ToolCall& call : tool_calls) {
        Json input = Json::parse(call.arguments_json, nullptr, false);
        if (input.is_discarded() || !input.is_object()) { input = Json::object(); }
        blocks.push_back(Json{{"type", "tool_use"},
                              {"id", call.id},
                              {"name", call.name},
                              {"input", std::move(input)}});
    }
    if (blocks.empty()) { blocks.push_back(Json{{"type", "text"}, {"text", ""}}); }
    const Json payload = {{"id", id},
                          {"type", "message"},
                          {"role", "assistant"},
                          {"model", model},
                          {"content", std::move(blocks)},
                          {"stop_reason", stop_reason},
                          {"stop_sequence", nullptr},
                          {"usage", Json{{"input_tokens", usage.prompt_tokens},
                                         {"output_tokens", usage.completion_tokens}}}};
    return payload.dump();
}

std::string make_message_start(const std::string& id, const std::string& model, int input_tokens) {
    const Json message = {{"id", id},
                          {"type", "message"},
                          {"role", "assistant"},
                          {"model", model},
                          {"content", Json::array()},
                          {"stop_reason", nullptr},
                          {"stop_sequence", nullptr},
                          {"usage", Json{{"input_tokens", input_tokens}, {"output_tokens", 0}}}};
    return sse("message_start", Json{{"type", "message_start"}, {"message", message}});
}

std::string make_content_block_start_text(int index) {
    return sse("content_block_start",
               Json{{"type", "content_block_start"},
                    {"index", index},
                    {"content_block", Json{{"type", "text"}, {"text", ""}}}});
}

std::string make_content_block_start_thinking(int index) {
    return sse("content_block_start",
               Json{{"type", "content_block_start"},
                    {"index", index},
                    {"content_block", Json{{"type", "thinking"}, {"thinking", ""}}}});
}

std::string make_content_block_start_tool_use(int index, const ToolCall& call) {
    return sse("content_block_start", Json{{"type", "content_block_start"},
                                           {"index", index},
                                           {"content_block", Json{{"type", "tool_use"},
                                                                  {"id", call.id},
                                                                  {"name", call.name},
                                                                  {"input", Json::object()}}}});
}

std::string make_content_block_delta_text(int index, const std::string& delta_text) {
    return sse("content_block_delta",
               Json{{"type", "content_block_delta"},
                    {"index", index},
                    {"delta", Json{{"type", "text_delta"}, {"text", delta_text}}}});
}

std::string make_content_block_delta_thinking(int index, const std::string& delta_text) {
    return sse("content_block_delta",
               Json{{"type", "content_block_delta"},
                    {"index", index},
                    {"delta", Json{{"type", "thinking_delta"}, {"thinking", delta_text}}}});
}

std::string make_content_block_delta_tool_json(int index, const std::string& partial_json) {
    return sse("content_block_delta",
               Json{{"type", "content_block_delta"},
                    {"index", index},
                    {"delta", Json{{"type", "input_json_delta"}, {"partial_json", partial_json}}}});
}

std::string make_content_block_stop(int index) {
    return sse("content_block_stop", Json{{"type", "content_block_stop"}, {"index", index}});
}

std::string make_message_delta(const char* stop_reason, int output_tokens) {
    return sse("message_delta",
               Json{{"type", "message_delta"},
                    {"delta", Json{{"stop_reason", stop_reason}, {"stop_sequence", nullptr}}},
                    {"usage", Json{{"output_tokens", output_tokens}}}});
}

std::string make_message_stop() { return sse("message_stop", Json{{"type", "message_stop"}}); }

std::string make_messages_ping() { return sse("ping", Json{{"type", "ping"}}); }

std::string make_messages_error_body(const ApiError& error) {
    const Json payload = {
        {"type", "error"},
        {"error", Json{{"type", anthropic_error_type(error.status)}, {"message", error.message}}}};
    return payload.dump();
}

std::string messages_sse_error_event(const ApiError& error) {
    return std::string("event: error\ndata: ") + make_messages_error_body(error) + "\n\n";
}

std::string make_count_tokens_response(int input_tokens) {
    return Json{{"input_tokens", input_tokens}}.dump();
}

std::string new_message_id() {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<std::uint64_t> dist;
    std::array<char, 32> buf{};
    std::snprintf(buf.data(), buf.size(), "%016llx", static_cast<unsigned long long>(dist(rng)));
    return "msg_" + std::string(buf.data());
}

} // namespace ninfer::serve
