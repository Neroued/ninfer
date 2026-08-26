#include "serve/responses_schema.h"

#include "serve/generation_service.h"
#include "serve/openai_schema.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
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

bool optional_bool(const Json& object, const char* key, bool fallback) {
    if (!object.contains(key) || object.at(key).is_null()) { return fallback; }
    if (!object.at(key).is_boolean()) { bad_request(std::string(key) + " must be a boolean", key); }
    return object.at(key).get<bool>();
}

std::optional<double> optional_number(const Json& object, const char* key) {
    if (!object.contains(key) || object.at(key).is_null()) { return std::nullopt; }
    if (!object.at(key).is_number()) { bad_request(std::string(key) + " must be a number", key); }
    const double value = object.at(key).get<double>();
    if (!std::isfinite(value)) { bad_request(std::string(key) + " must be finite", key); }
    return value;
}

std::optional<int> optional_int(const Json& object, const char* key) {
    if (!object.contains(key) || object.at(key).is_null()) { return std::nullopt; }
    if (!object.at(key).is_number_integer()) {
        bad_request(std::string(key) + " must be an integer", key);
    }
    if (object.at(key).is_number_unsigned()) {
        const std::uint64_t value = object.at(key).get<std::uint64_t>();
        if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
            bad_request(std::string(key) + " is out of range", key);
        }
        return static_cast<int>(value);
    }
    const std::int64_t value = object.at(key).get<std::int64_t>();
    if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max()) {
        bad_request(std::string(key) + " is out of range", key);
    }
    return static_cast<int>(value);
}

bool valid_function_name(const std::string& name) {
    if (name.empty() || name.size() > 64) { return false; }
    for (const unsigned char c : name) {
        if (std::isalnum(c) == 0 && c != '_' && c != '-') { return false; }
    }
    return true;
}

std::string require_function_name(const Json& object, const char* param) {
    if (!object.contains("name") || !object.at("name").is_string()) {
        bad_request("function name must be a string", param);
    }
    std::string name = object.at("name").get<std::string>();
    if (!valid_function_name(name)) {
        bad_request("function name must match [A-Za-z0-9_-]{1,64}", param);
    }
    return name;
}

std::string item_id(const Json& item, const char* prefix, const char* param) {
    if (!item.contains("id") || item.at("id").is_null()) { return new_response_item_id(prefix); }
    if (!item.at("id").is_string() || item.at("id").get<std::string>().empty()) {
        bad_request("input Item id must be a non-empty string", param);
    }
    return item.at("id").get<std::string>();
}

ninfer::product::media_acquire::Source parse_image_source(const Json& part) {
    if (part.contains("file_id") && !part.at("file_id").is_null()) {
        bad_request("input_image.file_id is not supported; use image_url", "input",
                    "file_inputs_not_supported");
    }
    if (!part.contains("image_url") || !part.at("image_url").is_string() ||
        part.at("image_url").get<std::string>().empty()) {
        bad_request("input_image must contain a non-empty image_url", "input");
    }
    if (part.contains("detail") && !part.at("detail").is_null()) {
        if (!part.at("detail").is_string()) {
            bad_request("input_image.detail must be a string", "input");
        }
        const std::string detail = part.at("detail").get<std::string>();
        if (detail != "auto") {
            bad_request("only input_image detail 'auto' is supported", "input",
                        "image_detail_not_supported");
        }
    }

    ninfer::product::media_acquire::Source source;
    source.value = part.at("image_url").get<std::string>();
    if (source.value.starts_with("data:")) {
        source.kind = ninfer::product::media_acquire::SourceKind::Data;
    } else if (source.value.starts_with("http://") || source.value.starts_with("https://")) {
        source.kind = ninfer::product::media_acquire::SourceKind::Url;
    } else {
        bad_request("input_image.image_url must use HTTP(S) or a data URI", "input");
    }
    return source;
}

ninfer::product::media_acquire::Source parse_video_source(const Json& part) {
    if (!part.contains("video_url") || !part.at("video_url").is_string() ||
        part.at("video_url").get<std::string>().empty()) {
        bad_request("input_video must contain a non-empty video_url", "input");
    }
    ninfer::product::media_acquire::Source source;
    source.value = part.at("video_url").get<std::string>();
    if (source.value.starts_with("data:")) {
        source.kind = ninfer::product::media_acquire::SourceKind::Data;
    } else if (source.value.starts_with("http://") || source.value.starts_with("https://")) {
        source.kind = ninfer::product::media_acquire::SourceKind::Url;
    } else {
        bad_request("input_video.video_url must use HTTP(S) or a data URI", "input");
    }
    return source;
}

struct ParsedMessage {
    ChatTurn turn;
    Json canonical;
};

ParsedMessage parse_message_item(const Json& item, std::size_t index) {
    if (!item.contains("role") || !item.at("role").is_string()) {
        bad_request("input message " + std::to_string(index) + " must contain a string role",
                    "input");
    }
    const std::string role = item.at("role").get<std::string>();
    ChatRole parsed_role;
    if (role == "user") {
        parsed_role = ChatRole::User;
    } else if (role == "assistant") {
        parsed_role = ChatRole::Assistant;
    } else if (role == "system") {
        parsed_role = ChatRole::System;
    } else if (role == "developer") {
        parsed_role = ChatRole::Developer;
    } else {
        bad_request("unsupported input message role: " + role, "input", "unsupported_role");
    }
    if (item.contains("phase") && !item.at("phase").is_null()) {
        bad_request("message phase is not supported", "input", "phase_not_supported");
    }
    if (item.contains("status") && !item.at("status").is_null()) {
        if (!item.at("status").is_string() || item.at("status").get<std::string>() != "completed") {
            bad_request("input message status must be 'completed'", "input");
        }
    }
    if (!item.contains("content") || item.at("content").is_null()) {
        bad_request("input message " + std::to_string(index) + " must contain content", "input");
    }

    ParsedMessage parsed;
    parsed.turn.role       = parsed_role;
    Json content           = Json::array();
    const auto append_text = [&](const std::string& text, const std::string& wire_type) {
        ContentPart part;
        part.kind     = ContentKind::Text;
        part.text     = text;
        part.type_raw = wire_type;
        parsed.turn.content.push_back(std::move(part));
        Json canonical = {{"type", wire_type}, {"text", text}};
        if (wire_type == "output_text") { canonical["annotations"] = Json::array(); }
        content.push_back(std::move(canonical));
    };

    if (item.at("content").is_string()) {
        append_text(item.at("content").get<std::string>(),
                    parsed_role == ChatRole::Assistant ? "output_text" : "input_text");
    } else if (item.at("content").is_array()) {
        for (const Json& value : item.at("content")) {
            if (!value.is_object() || !value.contains("type") || !value.at("type").is_string()) {
                bad_request("input message content parts must have a string type", "input");
            }
            const std::string type = value.at("type").get<std::string>();
            if (type == "input_text" || type == "output_text") {
                if (!value.contains("text") || !value.at("text").is_string()) {
                    bad_request(type + " must contain a string text", "input");
                }
                if (type == "output_text" && parsed_role != ChatRole::Assistant) {
                    bad_request("output_text is only valid on assistant messages", "input");
                }
                append_text(value.at("text").get<std::string>(), type);
            } else if (type == "input_image") {
                if (parsed_role != ChatRole::User) {
                    bad_request("input_image is only supported on user messages", "input");
                }
                ContentPart part;
                part.kind     = ContentKind::Image;
                part.type_raw = type;
                part.source   = parse_image_source(value);
                parsed.turn.content.push_back(std::move(part));
                content.push_back(Json{{"type", "input_image"},
                                       {"image_url", value.at("image_url")},
                                       {"detail", "auto"}});
            } else if (type == "input_video") {
                if (parsed_role != ChatRole::User) {
                    bad_request("input_video is only supported on user messages", "input");
                }
                ContentPart part;
                part.kind     = ContentKind::Video;
                part.type_raw = type;
                part.source   = parse_video_source(value);
                parsed.turn.content.push_back(std::move(part));
                content.push_back(
                    Json{{"type", "input_video"}, {"video_url", value.at("video_url")}});
            } else if (type == "input_file") {
                bad_request("input_file is not supported", "input", "file_inputs_not_supported");
            } else if (type == "input_audio") {
                bad_request("input_audio is not supported", "input", "audio_inputs_not_supported");
            } else {
                bad_request("unsupported message content type: " + type, "input",
                            "modality_not_supported");
            }
        }
    } else {
        bad_request("input message content must be a string or array", "input");
    }
    if (parsed.turn.content.empty()) {
        bad_request("input message content must not be empty", "input");
    }

    parsed.canonical = {{"id", item_id(item, "msg", "input")},
                        {"type", "message"},
                        {"role", role},
                        {"content", std::move(content)}};
    return parsed;
}

std::string parse_reasoning_item(const Json& item, Json& canonical) {
    if (item.contains("encrypted_content") && !item.at("encrypted_content").is_null()) {
        bad_request("encrypted reasoning content is not supported", "input",
                    "encrypted_reasoning_not_supported");
    }
    if (item.contains("summary") && !item.at("summary").is_null()) {
        if (!item.at("summary").is_array()) {
            bad_request("reasoning summary must be an array", "input");
        }
        if (!item.at("summary").empty()) {
            bad_request("reasoning summaries are not supported; replay raw reasoning_text", "input",
                        "reasoning_summary_not_supported");
        }
    }
    if (!item.contains("content") || !item.at("content").is_array()) {
        bad_request("reasoning Item must contain a content array", "input");
    }
    std::string text;
    Json content = Json::array();
    for (const Json& part : item.at("content")) {
        if (!part.is_object() || !part.contains("type") || !part.at("type").is_string() ||
            part.at("type").get<std::string>() != "reasoning_text" || !part.contains("text") ||
            !part.at("text").is_string()) {
            bad_request("reasoning content only supports reasoning_text parts", "input");
        }
        text += part.at("text").get<std::string>();
        content.push_back(Json{{"type", "reasoning_text"}, {"text", part.at("text")}});
    }
    canonical = {{"id", item_id(item, "rs", "input")},
                 {"type", "reasoning"},
                 {"summary", Json::array()},
                 {"content", std::move(content)}};
    return text;
}

ToolCall parse_function_call_item(const Json& item, Json& canonical) {
    ToolCall call;
    if (!item.contains("call_id") || !item.at("call_id").is_string() ||
        item.at("call_id").get<std::string>().empty()) {
        bad_request("function_call must contain a non-empty call_id", "input");
    }
    call.id   = item.at("call_id").get<std::string>();
    call.name = require_function_name(item, "input");
    if (!item.contains("arguments") || !item.at("arguments").is_string()) {
        bad_request("function_call arguments must be a JSON string", "input");
    }
    call.arguments_json  = item.at("arguments").get<std::string>();
    const Json arguments = Json::parse(call.arguments_json, nullptr, false);
    if (arguments.is_discarded() || !arguments.is_object()) {
        bad_request("function_call arguments must encode a JSON object", "input");
    }
    if (item.contains("status") && !item.at("status").is_null() &&
        (!item.at("status").is_string() || item.at("status").get<std::string>() != "completed")) {
        bad_request("function_call status must be 'completed'", "input");
    }
    canonical = {{"id", item_id(item, "fc", "input")},
                 {"type", "function_call"},
                 {"status", "completed"},
                 {"call_id", call.id},
                 {"name", call.name},
                 {"arguments", call.arguments_json}};
    return call;
}

ChatTurn parse_function_call_output_item(const Json& item, Json& canonical) {
    if (!item.contains("call_id") || !item.at("call_id").is_string() ||
        item.at("call_id").get<std::string>().empty()) {
        bad_request("function_call_output must contain a non-empty call_id", "input");
    }
    if (!item.contains("output") || !item.at("output").is_string()) {
        bad_request("function_call_output output must be a string", "input");
    }
    if (item.contains("status") && !item.at("status").is_null() &&
        (!item.at("status").is_string() || item.at("status").get<std::string>() != "completed")) {
        bad_request("function_call_output status must be 'completed'", "input");
    }
    ChatTurn turn;
    turn.role         = ChatRole::Tool;
    turn.tool_call_id = item.at("call_id").get<std::string>();
    ContentPart content;
    content.kind     = ContentKind::Text;
    content.type_raw = "input_text";
    content.text     = item.at("output").get<std::string>();
    turn.content.push_back(std::move(content));
    canonical = {{"id", item_id(item, "fco", "input")},
                 {"type", "function_call_output"},
                 {"status", "completed"},
                 {"call_id", turn.tool_call_id},
                 {"output", item.at("output")}};
    return turn;
}

void parse_input(const Json& input, ResponsesRequest& out) {
    Json values;
    if (input.is_string()) {
        values = Json::array({Json{{"type", "message"}, {"role", "user"}, {"content", input}}});
    } else if (input.is_array()) {
        values = input;
    } else {
        bad_request("input must be a string or an array of Items", "input");
    }
    if (values.empty()) { bad_request("input must not be empty", "input"); }

    std::string pending_reasoning;
    bool pending_reasoning_present = false;
    bool can_group_function_calls  = false;
    std::unordered_set<std::string> ids;
    for (std::size_t index = 0; index < values.size(); ++index) {
        const Json& item = values.at(index);
        if (!item.is_object()) {
            bad_request("input Item " + std::to_string(index) + " must be an object", "input");
        }
        std::string type;
        if (item.contains("type") && !item.at("type").is_null()) {
            if (!item.at("type").is_string()) {
                bad_request("input Item type must be a string", "input");
            }
            type = item.at("type").get<std::string>();
        } else if (item.contains("role")) {
            type = "message";
        } else {
            bad_request("input Item must contain type", "input");
        }

        Json canonical;
        if (type == "message") {
            ParsedMessage message = parse_message_item(item, index);
            if (pending_reasoning_present) {
                if (message.turn.role != ChatRole::Assistant) {
                    bad_request("a reasoning Item must be followed by an assistant output Item",
                                "input");
                }
                message.turn.reasoning_content = std::move(pending_reasoning);
                pending_reasoning.clear();
                pending_reasoning_present = false;
            }
            out.input_turns.push_back(std::move(message.turn));
            canonical                = std::move(message.canonical);
            can_group_function_calls = false;
        } else if (type == "reasoning") {
            if (pending_reasoning_present) {
                bad_request("adjacent reasoning Items are not supported", "input");
            }
            pending_reasoning         = parse_reasoning_item(item, canonical);
            pending_reasoning_present = true;
            can_group_function_calls  = false;
        } else if (type == "function_call") {
            ToolCall call = parse_function_call_item(item, canonical);
            if (can_group_function_calls && !pending_reasoning_present &&
                !out.input_turns.empty() && out.input_turns.back().role == ChatRole::Assistant &&
                out.input_turns.back().content.empty() &&
                !out.input_turns.back().tool_calls.empty()) {
                out.input_turns.back().tool_calls.push_back(std::move(call));
            } else {
                ChatTurn turn;
                turn.role              = ChatRole::Assistant;
                turn.reasoning_content = std::move(pending_reasoning);
                pending_reasoning.clear();
                pending_reasoning_present = false;
                turn.tool_calls.push_back(std::move(call));
                out.input_turns.push_back(std::move(turn));
            }
            can_group_function_calls = true;
        } else if (type == "function_call_output") {
            if (pending_reasoning_present) {
                bad_request("a reasoning Item must be followed by an assistant output Item",
                            "input");
            }
            out.input_turns.push_back(parse_function_call_output_item(item, canonical));
            can_group_function_calls = false;
        } else if (type == "input_file") {
            bad_request("input_file is not supported", "input", "file_inputs_not_supported");
        } else {
            bad_request("unsupported input Item type: " + type, "input", "item_type_not_supported");
        }

        const std::string id = canonical.at("id").get<std::string>();
        if (!ids.insert(id).second) { bad_request("duplicate input Item id: " + id, "input"); }
        out.input_items.push_back(std::move(canonical));
    }
    if (pending_reasoning_present) {
        bad_request("a reasoning Item must be followed by an assistant output Item", "input");
    }
}

void parse_tools(const Json& body, ResponsesRequest& out) {
    if (!body.contains("tools") || body.at("tools").is_null()) { return; }
    if (!body.at("tools").is_array()) { bad_request("tools must be an array", "tools"); }
    std::unordered_set<std::string> names;
    for (const Json& item : body.at("tools")) {
        if (!item.is_object() || !item.contains("type") || !item.at("type").is_string()) {
            bad_request("tools entries must be objects with a string type", "tools");
        }
        if (item.at("type").get<std::string>() != "function") {
            bad_request("only function tools are supported", "tools", "tool_type_not_supported");
        }
        ToolDefinition tool;
        tool.name = require_function_name(item, "tools");
        if (!names.insert(tool.name).second) {
            bad_request("duplicate function tool name: " + tool.name, "tools");
        }
        if (item.contains("description") && !item.at("description").is_null()) {
            if (!item.at("description").is_string()) {
                bad_request("function description must be a string", "tools");
            }
            tool.description = item.at("description").get<std::string>();
        }
        Json parameters = Json{{"type", "object"}, {"properties", Json::object()}};
        if (item.contains("parameters") && !item.at("parameters").is_null()) {
            if (!item.at("parameters").is_object()) {
                bad_request("function parameters must be a JSON object", "tools");
            }
            parameters = item.at("parameters");
        }
        if (item.contains("strict") && !item.at("strict").is_null()) {
            if (!item.at("strict").is_boolean()) {
                bad_request("function strict must be a boolean", "tools");
            }
            if (item.at("strict").get<bool>()) {
                bad_request("strict function schema enforcement is not supported", "tools",
                            "strict_tools_not_supported");
            }
        }
        tool.strict          = false;
        tool.parameters_json = parameters.dump();
        Json canonical       = {{"type", "function"},
                                {"name", tool.name},
                                {"parameters", parameters},
                                {"strict", false}};
        if (!tool.description.empty()) { canonical["description"] = tool.description; }
        Json nested = {
            {"type", "function"},
            {"function", Json{{"name", tool.name}, {"parameters", parameters}, {"strict", false}}}};
        if (!tool.description.empty()) { nested["function"]["description"] = tool.description; }
        tool.definition_json = nested.dump();
        out.generation.tools.push_back(std::move(tool));
        out.tools.push_back(std::move(canonical));
    }
}

void parse_tool_choice(const Json& body, ResponsesRequest& out) {
    if (!body.contains("tool_choice") || body.at("tool_choice").is_null()) {
        out.tool_choice = "auto";
        return;
    }
    const Json& choice = body.at("tool_choice");
    if (choice.is_string()) {
        const std::string value = choice.get<std::string>();
        if (value == "auto") {
            out.generation.tool_choice.mode = ToolChoiceMode::Auto;
        } else if (value == "none") {
            out.generation.tool_choice.mode = ToolChoiceMode::None;
        } else if (value == "required") {
            bad_request("tool_choice 'required' is not supported", "tool_choice",
                        "tool_choice_not_supported");
        } else {
            bad_request("tool_choice must be 'auto' or 'none'", "tool_choice");
        }
        out.tool_choice = value;
    } else if (choice.is_object()) {
        bad_request("named tool_choice is not supported", "tool_choice",
                    "tool_choice_not_supported");
    } else {
        bad_request("tool_choice must be a string or object", "tool_choice");
    }
}

void parse_reasoning(const Json& body, ResponsesRequest& out) {
    if (!body.contains("reasoning") || body.at("reasoning").is_null()) { return; }
    const Json& reasoning = body.at("reasoning");
    if (!reasoning.is_object()) { bad_request("reasoning must be an object", "reasoning"); }
    for (auto it = reasoning.begin(); it != reasoning.end(); ++it) {
        if (it.key() != "effort" && !it.value().is_null()) {
            bad_request("reasoning." + it.key() + " is not supported", "reasoning",
                        "reasoning_option_not_supported");
        }
    }
    if (!reasoning.contains("effort") || reasoning.at("effort").is_null()) { return; }
    if (!reasoning.at("effort").is_string()) {
        bad_request("reasoning.effort must be a string", "reasoning");
    }
    const std::string value = reasoning.at("effort").get<std::string>();
    const std::optional<RequestedReasoningEffort> effort = parse_requested_reasoning_effort(value);
    if (!effort) {
        bad_request("reasoning.effort must be one of none, minimal, low, medium, high, xhigh, or "
                    "max",
                    "reasoning");
    }
    out.generation.reasoning_effort       = *effort;
    out.generation.reasoning_effort_param = "reasoning.effort";
}

void validate_metadata(const Json& body, ResponsesRequest& out) {
    if (!body.contains("metadata") || body.at("metadata").is_null()) { return; }
    if (!body.at("metadata").is_object()) { bad_request("metadata must be an object", "metadata"); }
    if (body.at("metadata").size() > 16) {
        bad_request("metadata supports at most 16 entries", "metadata");
    }
    for (auto it = body.at("metadata").begin(); it != body.at("metadata").end(); ++it) {
        if (it.key().size() > 64 || !it.value().is_string() ||
            it.value().get_ref<const std::string&>().size() > 512) {
            bad_request("metadata keys must be at most 64 characters and string values at most "
                        "512 characters",
                        "metadata");
        }
    }
    out.metadata = body.at("metadata");
}

void reject_unknown_top_level(const Json& body) {
    static const std::unordered_set<std::string> allowed = {
        "background",
        "chat_template_kwargs",
        "context_management",
        "conversation",
        "include",
        "input",
        "instructions",
        "max_output_tokens",
        "max_tool_calls",
        "metadata",
        "model",
        "moderation",
        "parallel_tool_calls",
        "previous_response_id",
        "preserve_thinking",
        "prompt",
        "prompt_cache_key",
        "prompt_cache_options",
        "prompt_cache_retention",
        "reasoning",
        "safety_identifier",
        "service_tier",
        "store",
        "stream",
        "stream_options",
        "temperature",
        "text",
        "tool_choice",
        "tools",
        "top_logprobs",
        "top_p",
        "truncation",
        "user",
    };
    for (auto it = body.begin(); it != body.end(); ++it) {
        if (!allowed.contains(it.key())) {
            bad_request("unknown parameter: " + it.key(), it.key(), "unknown_parameter");
        }
    }
}

void reject_server_managed_features(const Json& body) {
    for (const char* key : {"context_management", "conversation", "max_tool_calls", "moderation",
                            "prompt", "prompt_cache_key", "prompt_cache_options",
                            "prompt_cache_retention", "safety_identifier", "user"}) {
        if (body.contains(key) && !body.at(key).is_null()) {
            bad_request(std::string(key) + " is not supported", key, "parameter_not_supported");
        }
    }
    if (body.contains("background") && !body.at("background").is_null()) {
        if (!body.at("background").is_boolean()) {
            bad_request("background must be a boolean", "background");
        }
        if (body.at("background").get<bool>()) {
            bad_request("background responses are not supported", "background",
                        "background_not_supported");
        }
    }
    if (body.contains("include") && !body.at("include").is_null()) {
        if (!body.at("include").is_array()) { bad_request("include must be an array", "include"); }
        if (!body.at("include").empty()) {
            bad_request("additional response fields are not supported", "include",
                        "include_not_supported");
        }
    }
    if (body.contains("parallel_tool_calls") && !body.at("parallel_tool_calls").is_null()) {
        if (!body.at("parallel_tool_calls").is_boolean()) {
            bad_request("parallel_tool_calls must be a boolean", "parallel_tool_calls");
        }
        if (!body.at("parallel_tool_calls").get<bool>()) {
            bad_request("parallel_tool_calls=false cannot be enforced", "parallel_tool_calls",
                        "parallel_tool_calls_not_supported");
        }
    }
    if (body.contains("top_logprobs") && !body.at("top_logprobs").is_null()) {
        const std::optional<int> value = optional_int(body, "top_logprobs");
        if (!value || *value != 0) {
            bad_request("top_logprobs is not supported", "top_logprobs", "logprobs_not_supported");
        }
    }
    if (body.contains("truncation") && !body.at("truncation").is_null()) {
        if (!body.at("truncation").is_string() ||
            body.at("truncation").get<std::string>() != "disabled") {
            bad_request("only truncation 'disabled' is supported", "truncation",
                        "truncation_not_supported");
        }
    }
    if (body.contains("service_tier") && !body.at("service_tier").is_null()) {
        if (!body.at("service_tier").is_string()) {
            bad_request("service_tier must be a string", "service_tier");
        }
        const std::string tier = body.at("service_tier").get<std::string>();
        if (tier != "auto" && tier != "default") {
            bad_request("only service_tier 'auto' or 'default' is supported", "service_tier",
                        "service_tier_not_supported");
        }
    }
    if (body.contains("stream_options") && !body.at("stream_options").is_null()) {
        if (!body.at("stream_options").is_object()) {
            bad_request("stream_options must be an object", "stream_options");
        }
        for (auto it = body.at("stream_options").begin(); it != body.at("stream_options").end();
             ++it) {
            if (it.key() != "include_obfuscation" || !it.value().is_boolean() ||
                it.value().get<bool>()) {
                bad_request("stream_options only supports include_obfuscation=false",
                            "stream_options", "stream_option_not_supported");
            }
        }
    }
    if (body.contains("text") && !body.at("text").is_null()) {
        if (!body.at("text").is_object()) { bad_request("text must be an object", "text"); }
        for (auto it = body.at("text").begin(); it != body.at("text").end(); ++it) {
            if (it.key() != "format" && !it.value().is_null()) {
                bad_request("text." + it.key() + " is not supported", "text",
                            "text_option_not_supported");
            }
        }
        if (body.at("text").contains("format") && !body.at("text").at("format").is_null()) {
            const Json& format = body.at("text").at("format");
            if (!format.is_object() || !format.contains("type") || !format.at("type").is_string() ||
                format.at("type").get<std::string>() != "text") {
                bad_request("only text.format {type:'text'} is supported", "text",
                            "structured_outputs_not_supported");
            }
            for (auto it = format.begin(); it != format.end(); ++it) {
                if (it.key() != "type") {
                    bad_request("only text.format {type:'text'} is supported", "text",
                                "structured_outputs_not_supported");
                }
            }
        }
    }
}

ResponsesRequest parse_request_impl(const Json& body, const RequestLimits& limits) {
    require_object(body);
    reject_unknown_top_level(body);
    reject_server_managed_features(body);

    ResponsesRequest out;
    if (!body.contains("model") || !body.at("model").is_string() ||
        body.at("model").get<std::string>().empty()) {
        bad_request("missing required field: model", "model");
    }
    out.generation.model = body.at("model").get<std::string>();
    if (!body.contains("input")) { bad_request("missing required field: input", "input"); }
    parse_input(body.at("input"), out);

    if (body.contains("instructions") && !body.at("instructions").is_null()) {
        if (!body.at("instructions").is_string()) {
            bad_request("instructions must be a string", "instructions");
        }
        out.instructions = body.at("instructions").get<std::string>();
    }
    if (body.contains("previous_response_id") && !body.at("previous_response_id").is_null()) {
        if (!body.at("previous_response_id").is_string() ||
            body.at("previous_response_id").get<std::string>().empty()) {
            bad_request("previous_response_id must be a non-empty string", "previous_response_id");
        }
        out.previous_response_id = body.at("previous_response_id").get<std::string>();
    }

    out.store             = optional_bool(body, "store", true);
    out.stream            = optional_bool(body, "stream", false);
    out.generation.stream = out.stream;
    validate_metadata(body, out);
    parse_tools(body, out);
    parse_tool_choice(body, out);
    parse_reasoning(body, out);
    out.generation.preserve_thinking = parse_openai_preserve_thinking(body);

    if (const std::optional<double> temperature = optional_number(body, "temperature")) {
        if (*temperature < 0.0 || *temperature > 2.0) {
            bad_request("temperature must be in [0,2]", "temperature");
        }
        out.generation.sampling.temperature = *temperature;
    }
    if (const std::optional<double> top_p = optional_number(body, "top_p")) {
        if (*top_p < 0.0 || *top_p > 1.0) { bad_request("top_p must be in [0,1]", "top_p"); }
        out.generation.sampling.top_p = *top_p;
    }

    if (const std::optional<int> max_output = optional_int(body, "max_output_tokens")) {
        if (*max_output < 16) {
            bad_request("max_output_tokens must be at least 16", "max_output_tokens",
                        "invalid_value");
        }
        out.generation.max_tokens     = *max_output;
        out.generation.max_tokens_set = true;
    } else {
        out.generation.max_tokens     = limits.default_max_tokens;
        out.generation.max_tokens_set = false;
    }
    out.generation.messages = out.input_turns;
    return out;
}

std::string random_id(const char* prefix) {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<std::uint64_t> distribution;
    std::array<char, 48> buffer{};
    std::snprintf(buffer.data(), buffer.size(), "%016llx%016llx",
                  static_cast<unsigned long long>(distribution(rng)),
                  static_cast<unsigned long long>(distribution(rng)));
    return std::string(prefix) + "_" + buffer.data();
}

std::int64_t completion_time_now() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string response_status(ninfer::FinishReason reason) {
    switch (reason) {
    case ninfer::FinishReason::OutputLimit:
    case ninfer::FinishReason::ContextCapacity:
        return "incomplete";
    case ninfer::FinishReason::Cancelled:
        return "cancelled";
    case ninfer::FinishReason::None:
    case ninfer::FinishReason::StopToken:
    case ninfer::FinishReason::StopString:
        return "completed";
    }
    return "failed";
}

struct ItemIds {
    std::string reasoning;
    std::string message;
    std::vector<std::string> function_calls;
};

Json response_common(const std::string& id, std::int64_t created_at,
                     const ResponsesRequest& request, const ResponsesRuntimeValues& runtime) {
    const Json reasoning = {
        {"effort", request.generation.reasoning_effort
                       ? Json(requested_reasoning_effort_name(*request.generation.reasoning_effort))
                       : Json(nullptr)},
        {"summary", nullptr}};
    return Json{
        {"id", id},
        {"object", "response"},
        {"created_at", created_at},
        {"background", false},
        {"instructions", request.instructions ? Json(*request.instructions) : Json(nullptr)},
        {"max_output_tokens", request.generation.max_tokens},
        {"max_tool_calls", nullptr},
        {"metadata", request.metadata},
        {"model", request.generation.model},
        {"parallel_tool_calls", true},
        {"previous_response_id",
         request.previous_response_id ? Json(*request.previous_response_id) : Json(nullptr)},
        {"reasoning", reasoning},
        {"service_tier", "default"},
        {"store", request.store},
        {"temperature", runtime.temperature},
        {"text", Json{{"format", Json{{"type", "text"}}}}},
        {"tool_choice", request.tool_choice},
        {"tools", request.tools},
        {"top_logprobs", 0},
        {"top_p", runtime.top_p},
        {"truncation", "disabled"}};
}

BuiltResponse build_response(const std::string& id, std::int64_t created_at,
                             const ResponsesRequest& request, const ResponsesRuntimeValues& runtime,
                             const GenerationOutcome& outcome, ItemIds ids) {
    BuiltResponse built;
    const std::string status      = response_status(outcome.finish_reason);
    const std::string item_status = status == "completed" ? "completed" : "incomplete";

    if (!outcome.reasoning.empty()) {
        if (ids.reasoning.empty()) { ids.reasoning = new_response_item_id("rs"); }
        const char* reasoning_status = (!outcome.text.empty() || !outcome.tool_calls.empty())
                                           ? "completed"
                                           : item_status.c_str();
        built.output_items.push_back(
            Json{{"id", ids.reasoning},
                 {"type", "reasoning"},
                 {"status", reasoning_status},
                 {"summary", Json::array()},
                 {"content",
                  Json::array({Json{{"type", "reasoning_text"}, {"text", outcome.reasoning}}})}});
    }

    if (!outcome.text.empty() || outcome.tool_calls.empty()) {
        if (ids.message.empty()) { ids.message = new_response_item_id("msg"); }
        built.output_items.push_back(
            Json{{"id", ids.message},
                 {"type", "message"},
                 {"status", item_status},
                 {"role", "assistant"},
                 {"content", Json::array({Json{{"type", "output_text"},
                                               {"annotations", Json::array()},
                                               {"text", outcome.text}}})}});
    }

    ids.function_calls.resize(outcome.tool_calls.size());
    for (std::size_t index = 0; index < outcome.tool_calls.size(); ++index) {
        if (ids.function_calls[index].empty()) {
            ids.function_calls[index] = new_response_item_id("fc");
        }
        const ToolCall& call = outcome.tool_calls[index];
        built.output_items.push_back(Json{{"id", ids.function_calls[index]},
                                          {"type", "function_call"},
                                          {"status", "completed"},
                                          {"call_id", call.id},
                                          {"name", call.name},
                                          {"arguments", call.arguments_json}});
    }

    ChatTurn history;
    history.role              = ChatRole::Assistant;
    history.reasoning_content = outcome.reasoning;
    history.tool_calls        = outcome.tool_calls;
    if (!outcome.text.empty()) {
        ContentPart part;
        part.kind     = ContentKind::Text;
        part.type_raw = "output_text";
        part.text     = outcome.text;
        history.content.push_back(std::move(part));
    }
    built.output_history.push_back(std::move(history));

    Json response            = response_common(id, created_at, request, runtime);
    response["status"]       = status;
    response["completed_at"] = completion_time_now();
    response["error"]        = nullptr;
    response["output"]       = built.output_items;
    response["incomplete_details"] =
        status == "incomplete" ? Json{{"reason", "max_output_tokens"}} : Json(nullptr);
    const int observed_cached = std::max(runtime.cached_input_tokens,
                                         static_cast<int>(outcome.metrics.prefix_cache_hit_tokens));
    const int cached_tokens   = std::clamp(observed_cached, 0, outcome.prompt_tokens);
    response["usage"] =
        Json{{"input_tokens", outcome.prompt_tokens},
             {"input_tokens_details", Json{{"cached_tokens", cached_tokens}}},
             {"output_tokens", outcome.completion_tokens},
             {"output_tokens_details", Json{{"reasoning_tokens", outcome.reasoning_tokens}}},
             {"total_tokens", outcome.prompt_tokens + outcome.completion_tokens}};
    built.body = std::move(response);
    return built;
}

std::string sse(const Json& event) {
    return "event: " + event.at("type").get<std::string>() + "\n" + "data: " + event.dump() +
           "\n\n";
}

Json in_progress_response(const std::string& id, std::int64_t created_at,
                          const ResponsesRequest& request, const ResponsesRuntimeValues& runtime) {
    Json response                  = response_common(id, created_at, request, runtime);
    response["status"]             = "in_progress";
    response["completed_at"]       = nullptr;
    response["error"]              = nullptr;
    response["incomplete_details"] = nullptr;
    response["output"]             = Json::array();
    response["usage"]              = nullptr;
    return response;
}

} // namespace

ResponsesRequest parse_responses_request(const Json& body, const RequestLimits& limits) {
    return parse_request_impl(body, limits);
}

ResponsesRequest parse_response_input_tokens_request(const Json& body,
                                                     const RequestLimits& limits) {
    require_object(body);
    for (auto it = body.begin(); it != body.end(); ++it) {
        if (it.key() != "model" && it.key() != "input" && it.key() != "chat_template_kwargs" &&
            it.key() != "preserve_thinking") {
            bad_request("unknown parameter: " + it.key(), it.key(), "unknown_parameter");
        }
    }
    ResponsesRequest parsed  = parse_request_impl(body, limits);
    parsed.store             = false;
    parsed.stream            = false;
    parsed.generation.stream = false;
    return parsed;
}

void inherit_responses_preserve_thinking(ResponsesRequest& request, bool parent_value) {
    if (request.generation.preserve_thinking) {
        request.generation.preserve_thinking_semantic_change =
            *request.generation.preserve_thinking != parent_value;
        return;
    }
    request.generation.preserve_thinking = parent_value;
}

void compose_responses_generation_messages(ResponsesRequest& request,
                                           std::vector<ChatTurn> previous_context) {
    std::vector<ChatTurn> current_input = std::move(request.generation.messages);
    std::vector<ChatTurn> messages;
    messages.reserve((request.instructions ? 1U : 0U) + previous_context.size() +
                     current_input.size());
    if (request.instructions) {
        ChatTurn instructions;
        instructions.role = ChatRole::Developer;
        ContentPart part;
        part.kind     = ContentKind::Text;
        part.type_raw = "input_text";
        part.text     = *request.instructions;
        instructions.content.push_back(std::move(part));
        messages.push_back(std::move(instructions));
    }
    messages.insert(messages.end(), std::make_move_iterator(previous_context.begin()),
                    std::make_move_iterator(previous_context.end()));
    messages.insert(messages.end(), std::make_move_iterator(current_input.begin()),
                    std::make_move_iterator(current_input.end()));
    request.generation.messages = std::move(messages);
}

BuiltResponse make_response_object(const std::string& id, std::int64_t created_at,
                                   const ResponsesRequest& request,
                                   const ResponsesRuntimeValues& runtime,
                                   const GenerationOutcome& outcome) {
    return build_response(id, created_at, request, runtime, outcome, {});
}

std::string make_response_input_tokens_body(int input_tokens) {
    return Json{{"object", "response.input_tokens"}, {"input_tokens", input_tokens}}.dump();
}

class ResponsesEventStream::Impl {
public:
    Impl(std::string response_id, std::int64_t created_at_, ResponsesRequest request_,
         ResponsesRuntimeValues runtime_)
        : id(std::move(response_id)), created_at(created_at_), request(std::move(request_)),
          runtime(runtime_) {}

    Json event(std::string type, Json fields = Json::object()) {
        fields["type"]            = std::move(type);
        fields["sequence_number"] = sequence++;
        return fields;
    }

    std::vector<std::string> ensure_reasoning() {
        if (reasoning_started) { return {}; }
        reasoning_started = true;
        ids.reasoning     = new_response_item_id("rs");
        reasoning_index   = next_output_index++;
        const Json item   = {{"id", ids.reasoning},
                             {"type", "reasoning"},
                             {"status", "in_progress"},
                             {"summary", Json::array()},
                             {"content", Json::array()}};
        const Json part   = {{"type", "reasoning_text"}, {"text", ""}};
        return {sse(event("response.output_item.added",
                          Json{{"output_index", reasoning_index}, {"item", item}})),
                sse(event("response.content_part.added", Json{{"item_id", ids.reasoning},
                                                              {"output_index", reasoning_index},
                                                              {"content_index", 0},
                                                              {"part", part}}))};
    }

    std::vector<std::string> close_reasoning(const std::string& final_text,
                                             const char* item_status = "completed") {
        if (!reasoning_started || reasoning_done) { return {}; }
        reasoning_done  = true;
        reasoning_text  = final_text;
        const Json part = {{"type", "reasoning_text"}, {"text", reasoning_text}};
        const Json item = {{"id", ids.reasoning},
                           {"type", "reasoning"},
                           {"status", item_status},
                           {"summary", Json::array()},
                           {"content", Json::array({part})}};
        return {sse(event("response.reasoning_text.done", Json{{"item_id", ids.reasoning},
                                                               {"output_index", reasoning_index},
                                                               {"content_index", 0},
                                                               {"text", reasoning_text}})),
                sse(event("response.content_part.done", Json{{"item_id", ids.reasoning},
                                                             {"output_index", reasoning_index},
                                                             {"content_index", 0},
                                                             {"part", part}})),
                sse(event("response.output_item.done",
                          Json{{"output_index", reasoning_index}, {"item", item}}))};
    }

    std::vector<std::string> ensure_message() {
        if (message_started) { return {}; }
        message_started = true;
        ids.message     = new_response_item_id("msg");
        message_index   = next_output_index++;
        const Json item = {{"id", ids.message},
                           {"type", "message"},
                           {"status", "in_progress"},
                           {"role", "assistant"},
                           {"content", Json::array()}};
        const Json part = {{"type", "output_text"}, {"annotations", Json::array()}, {"text", ""}};
        return {sse(event("response.output_item.added",
                          Json{{"output_index", message_index}, {"item", item}})),
                sse(event("response.content_part.added", Json{{"item_id", ids.message},
                                                              {"output_index", message_index},
                                                              {"content_index", 0},
                                                              {"part", part}}))};
    }

    std::vector<std::string> close_message(const std::string& final_text,
                                           const char* item_status = "completed") {
        if (!message_started || message_done) { return {}; }
        message_done    = true;
        content_text    = final_text;
        const Json part = {
            {"type", "output_text"}, {"annotations", Json::array()}, {"text", content_text}};
        const Json item = {{"id", ids.message},
                           {"type", "message"},
                           {"status", item_status},
                           {"role", "assistant"},
                           {"content", Json::array({part})}};
        return {sse(event("response.output_text.done", Json{{"item_id", ids.message},
                                                            {"output_index", message_index},
                                                            {"content_index", 0},
                                                            {"text", content_text},
                                                            {"logprobs", Json::array()}})),
                sse(event("response.content_part.done", Json{{"item_id", ids.message},
                                                             {"output_index", message_index},
                                                             {"content_index", 0},
                                                             {"part", part}})),
                sse(event("response.output_item.done",
                          Json{{"output_index", message_index}, {"item", item}}))};
    }

    std::string id;
    std::int64_t created_at = 0;
    ResponsesRequest request;
    ResponsesRuntimeValues runtime;
    std::uint64_t sequence = 0;
    int next_output_index  = 0;
    int reasoning_index    = -1;
    int message_index      = -1;
    bool started           = false;
    bool reasoning_started = false;
    bool reasoning_done    = false;
    bool message_started   = false;
    bool message_done      = false;
    bool finish_built      = false;
    bool terminal_emitted  = false;
    std::string reasoning_text;
    std::string content_text;
    ItemIds ids;
};

ResponsesEventStream::ResponsesEventStream(std::string response_id, std::int64_t created_at,
                                           ResponsesRequest request, ResponsesRuntimeValues runtime)
    : impl_(std::make_unique<Impl>(std::move(response_id), created_at, std::move(request),
                                   runtime)) {}

ResponsesEventStream::~ResponsesEventStream()                                          = default;
ResponsesEventStream::ResponsesEventStream(ResponsesEventStream&&) noexcept            = default;
ResponsesEventStream& ResponsesEventStream::operator=(ResponsesEventStream&&) noexcept = default;

std::vector<std::string> ResponsesEventStream::start() {
    if (impl_->started) { throw std::logic_error("Responses event stream already started"); }
    impl_->started = true;
    const Json response =
        in_progress_response(impl_->id, impl_->created_at, impl_->request, impl_->runtime);
    return {sse(impl_->event("response.created", Json{{"response", response}})),
            sse(impl_->event("response.in_progress", Json{{"response", response}}))};
}

std::vector<std::string> ResponsesEventStream::reasoning_delta(const std::string& text) {
    if (!impl_->started || impl_->finish_built) {
        throw std::logic_error("invalid reasoning delta event state");
    }
    if (text.empty()) { return {}; }
    std::vector<std::string> events = impl_->ensure_reasoning();
    impl_->reasoning_text += text;
    events.push_back(sse(
        impl_->event("response.reasoning_text.delta", Json{{"item_id", impl_->ids.reasoning},
                                                           {"output_index", impl_->reasoning_index},
                                                           {"content_index", 0},
                                                           {"delta", text}})));
    return events;
}

std::vector<std::string> ResponsesEventStream::content_delta(const std::string& text) {
    if (!impl_->started || impl_->finish_built) {
        throw std::logic_error("invalid content delta event state");
    }
    if (text.empty()) { return {}; }
    std::vector<std::string> events = impl_->close_reasoning(impl_->reasoning_text);
    std::vector<std::string> added  = impl_->ensure_message();
    events.insert(events.end(), std::make_move_iterator(added.begin()),
                  std::make_move_iterator(added.end()));
    impl_->content_text += text;
    events.push_back(
        sse(impl_->event("response.output_text.delta", Json{{"item_id", impl_->ids.message},
                                                            {"output_index", impl_->message_index},
                                                            {"content_index", 0},
                                                            {"delta", text},
                                                            {"logprobs", Json::array()}})));
    return events;
}

ResponsesStreamFinish ResponsesEventStream::finish(const GenerationOutcome& outcome) {
    if (!impl_->started || impl_->finish_built) {
        throw std::logic_error("invalid Responses stream finish state");
    }
    impl_->finish_built = true;
    ResponsesStreamFinish finished;
    const std::string status = response_status(outcome.finish_reason);
    const char* item_status  = status == "completed" ? "completed" : "incomplete";

    auto append = [&](std::vector<std::string> events) {
        finished.events_before_terminal.insert(finished.events_before_terminal.end(),
                                               std::make_move_iterator(events.begin()),
                                               std::make_move_iterator(events.end()));
    };
    if (!outcome.reasoning.empty() && !impl_->reasoning_started) {
        append(impl_->ensure_reasoning());
        impl_->reasoning_text = outcome.reasoning;
        finished.events_before_terminal.push_back(sse(impl_->event(
            "response.reasoning_text.delta", Json{{"item_id", impl_->ids.reasoning},
                                                  {"output_index", impl_->reasoning_index},
                                                  {"content_index", 0},
                                                  {"delta", outcome.reasoning}})));
    }
    const char* reasoning_status =
        (!outcome.text.empty() || !outcome.tool_calls.empty()) ? "completed" : item_status;
    append(impl_->close_reasoning(outcome.reasoning, reasoning_status));

    const bool needs_message = !outcome.text.empty() || outcome.tool_calls.empty();
    if (needs_message) {
        append(impl_->ensure_message());
        if (outcome.text != impl_->content_text) {
            if (!outcome.text.starts_with(impl_->content_text)) {
                throw std::logic_error("streamed content does not match terminal content");
            }
            const std::string suffix = outcome.text.substr(impl_->content_text.size());
            if (!suffix.empty()) {
                impl_->content_text += suffix;
                finished.events_before_terminal.push_back(sse(impl_->event(
                    "response.output_text.delta", Json{{"item_id", impl_->ids.message},
                                                       {"output_index", impl_->message_index},
                                                       {"content_index", 0},
                                                       {"delta", suffix},
                                                       {"logprobs", Json::array()}})));
            }
        }
        append(impl_->close_message(outcome.text, item_status));
    }

    impl_->ids.function_calls.reserve(outcome.tool_calls.size());
    for (const ToolCall& call : outcome.tool_calls) {
        const std::string item_id = new_response_item_id("fc");
        impl_->ids.function_calls.push_back(item_id);
        const int output_index = impl_->next_output_index++;
        const Json added_item  = {{"id", item_id},           {"type", "function_call"},
                                  {"status", "in_progress"}, {"call_id", call.id},
                                  {"name", call.name},       {"arguments", ""}};
        finished.events_before_terminal.push_back(
            sse(impl_->event("response.output_item.added",
                             Json{{"output_index", output_index}, {"item", added_item}})));
        if (!call.arguments_json.empty()) {
            finished.events_before_terminal.push_back(sse(impl_->event(
                "response.function_call_arguments.delta", Json{{"item_id", item_id},
                                                               {"output_index", output_index},
                                                               {"delta", call.arguments_json}})));
        }
        finished.events_before_terminal.push_back(sse(impl_->event(
            "response.function_call_arguments.done", Json{{"item_id", item_id},
                                                          {"output_index", output_index},
                                                          {"name", call.name},
                                                          {"arguments", call.arguments_json}})));
        const Json done_item = {{"id", item_id},         {"type", "function_call"},
                                {"status", "completed"}, {"call_id", call.id},
                                {"name", call.name},     {"arguments", call.arguments_json}};
        finished.events_before_terminal.push_back(
            sse(impl_->event("response.output_item.done",
                             Json{{"output_index", output_index}, {"item", done_item}})));
    }

    finished.response = build_response(impl_->id, impl_->created_at, impl_->request, impl_->runtime,
                                       outcome, impl_->ids);
    return finished;
}

std::string ResponsesEventStream::terminal(const BuiltResponse& response) {
    if (!impl_->finish_built || impl_->terminal_emitted) {
        throw std::logic_error("invalid Responses terminal event state");
    }
    impl_->terminal_emitted  = true;
    const std::string status = response.body.at("status").get<std::string>();
    const std::string type   = status == "completed"    ? "response.completed"
                               : status == "incomplete" ? "response.incomplete"
                               : status == "cancelled"  ? "response.cancelled"
                                                        : "response.failed";
    return sse(impl_->event(type, Json{{"response", response.body}}));
}

std::string ResponsesEventStream::failed(const ApiError& error) {
    if (!impl_->started || impl_->terminal_emitted) {
        throw std::logic_error("invalid Responses failed event state");
    }
    impl_->terminal_emitted = true;
    Json response =
        in_progress_response(impl_->id, impl_->created_at, impl_->request, impl_->runtime);
    response["status"]       = "failed";
    response["completed_at"] = completion_time_now();
    response["error"]        = Json{{"code", error.code.empty() ? Json(nullptr) : Json(error.code)},
                                    {"message", error.message}};
    return sse(impl_->event("response.failed", Json{{"response", response}}));
}

std::string new_response_id() { return random_id("resp"); }

std::string new_response_item_id(const char* prefix) { return random_id(prefix); }

} // namespace ninfer::serve
