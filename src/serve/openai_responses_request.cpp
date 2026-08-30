#include "serve/openai_responses.h"
#include "serve/openai_common.h"
#include "serve/request_validation.h"

#include <algorithm>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

void require_object(const Json& value, std::string_view name = "request body") {
    if (!value.is_object()) { bad_request(std::string(name) + " must be a JSON object"); }
}

std::string require_function_name(const Json& object, const char* param) {
    if (!object.contains("name") || !object.at("name").is_string()) {
        bad_request("function name must be a string", param);
    }
    const std::string name = object.at("name").get<std::string>();
    if (!valid_tool_name(name, 64)) {
        bad_request("function name must match [A-Za-z0-9_-]{1,64}", param);
    }
    return name;
}

std::optional<std::string> optional_namespace_name(const Json& object, const char* param) {
    if (!object.contains("namespace") || object.at("namespace").is_null()) { return std::nullopt; }
    if (!object.at("namespace").is_string()) {
        bad_request("function namespace must be a string or null", param);
    }
    const std::string name = object.at("namespace").get<std::string>();
    if (!valid_tool_name(name, 64)) {
        bad_request("function namespace must match [A-Za-z0-9_-]{1,64}", param,
                    "invalid_tool_name");
    }
    return name;
}

std::string require_namespace_tool_name(const Json& object) {
    if (!object.contains("name") || !object.at("name").is_string()) {
        bad_request("namespace name must be a string", "tools");
    }
    const std::string name = object.at("name").get<std::string>();
    if (!valid_tool_name(name, 64)) {
        bad_request("namespace name must match [A-Za-z0-9_-]{1,64}", "tools", "invalid_tool_name");
    }
    return name;
}

OpenAIResponsesFunctionIdentity function_identity(const Json& object, const char* param) {
    return OpenAIResponsesFunctionIdentity{.name = require_function_name(object, param),
                                           .wire_namespace =
                                               optional_namespace_name(object, param)};
}

std::string lower_function_identity(
    const OpenAIResponsesFunctionIdentity& identity,
    std::unordered_map<std::string, OpenAIResponsesFunctionIdentity>& identities,
    const char* param) {
    const std::string engine_name =
        identity.wire_namespace ? *identity.wire_namespace + "__" + identity.name : identity.name;
    if (!valid_tool_name(engine_name, 64)) {
        bad_request("flattened function identity '" + engine_name +
                        "' exceeds the Engine tool-name contract [A-Za-z0-9_-]{1,64}",
                    param, "invalid_tool_name");
    }
    const auto [position, inserted] = identities.emplace(engine_name, identity);
    if (!inserted && position->second != identity) {
        bad_request("function identity collision after namespace translation: '" + engine_name +
                        "'",
                    param, "duplicate_tool_name");
    }
    return engine_name;
}

void add_wire_function_identity(Json& object, const OpenAIResponsesFunctionIdentity& identity) {
    object["name"] = identity.name;
    if (identity.wire_namespace) { object["namespace"] = *identity.wire_namespace; }
}

std::string item_id(const Json& item, const char* prefix) {
    if (!item.contains("id") || item.at("id").is_null()) {
        return new_openai_response_item_id(prefix);
    }
    if (!item.at("id").is_string() || item.at("id").get_ref<const std::string&>().empty()) {
        bad_request("input Item id must be a non-empty string", "input");
    }
    return item.at("id").get<std::string>();
}

ninfer::product::media_acquire::Source parse_image_source(const Json& part) {
    if (part.contains("file_id") && !part.at("file_id").is_null()) {
        bad_request("input_image.file_id requires a Files API, which NInfer does not provide",
                    "input", "file_inputs_not_supported");
    }
    if (!part.contains("image_url") || !part.at("image_url").is_string() ||
        part.at("image_url").get_ref<const std::string&>().empty()) {
        bad_request("input_image must contain a non-empty image_url", "input");
    }
    if (part.contains("detail") && !part.at("detail").is_null()) {
        if (!part.at("detail").is_string()) {
            bad_request("input_image.detail must be a string", "input");
        }
        if (part.at("detail").get<std::string>() != "auto") {
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
        part.at("video_url").get_ref<const std::string&>().empty()) {
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

void apply_shared_breakpoint(ContentPart& part, const Json& wire, std::size_t& breakpoint_count) {
    if (!parse_openai_prompt_cache_breakpoint(wire, "input")) { return; }
    ++breakpoint_count;
    part.cache_boundary_after = CacheBoundary{};
}

struct ParsedMessage {
    ChatTurn turn;
    Json canonical;
};

void append_message_text(ParsedMessage& parsed, Json& content, const std::string& text,
                         const std::string& wire_type, const Json* wire,
                         std::size_t& breakpoint_count) {
    ContentPart part;
    part.kind     = ContentKind::Text;
    part.text     = text;
    part.type_raw = wire_type;
    if (wire != nullptr) { apply_shared_breakpoint(part, *wire, breakpoint_count); }
    parsed.turn.content.push_back(std::move(part));

    Json canonical;
    if (wire_type == "refusal") {
        canonical = Json{{"type", "refusal"}, {"refusal", text}};
    } else {
        canonical = Json{{"type", wire_type}, {"text", text}};
        if (wire_type == "output_text") {
            canonical["annotations"] = wire != nullptr && wire->contains("annotations")
                                           ? wire->at("annotations")
                                           : Json::array();
            if (wire != nullptr && wire->contains("logprobs")) {
                canonical["logprobs"] = wire->at("logprobs");
            }
        }
    }
    if (wire != nullptr && wire->contains("prompt_cache_breakpoint")) {
        canonical["prompt_cache_breakpoint"] = wire->at("prompt_cache_breakpoint");
    }
    content.push_back(std::move(canonical));
}

void parse_message_content_part(const Json& value, ChatRole role, ParsedMessage& parsed,
                                Json& content, std::size_t& breakpoint_count) {
    if (!value.is_object() || !value.contains("type") || !value.at("type").is_string()) {
        bad_request("input message content parts must have a string type", "input");
    }
    const std::string type = value.at("type").get<std::string>();
    if (type == "input_text") {
        if (!value.contains("text") || !value.at("text").is_string()) {
            bad_request("input_text must contain a string text", "input");
        }
        append_message_text(parsed, content, value.at("text").get<std::string>(), type, &value,
                            breakpoint_count);
        return;
    }
    if (type == "output_text") {
        if (role != ChatRole::Assistant) {
            bad_request("output_text is only valid on assistant messages", "input");
        }
        if (!value.contains("text") || !value.at("text").is_string()) {
            bad_request("output_text must contain a string text", "input");
        }
        if (value.contains("annotations") && !value.at("annotations").is_null() &&
            !value.at("annotations").is_array()) {
            bad_request("output_text.annotations must be an array", "input");
        }
        if (value.contains("logprobs") && !value.at("logprobs").is_null() &&
            !value.at("logprobs").is_array()) {
            bad_request("output_text.logprobs must be an array", "input");
        }
        append_message_text(parsed, content, value.at("text").get<std::string>(), type, &value,
                            breakpoint_count);
        return;
    }
    if (type == "refusal") {
        if (role != ChatRole::Assistant) {
            bad_request("refusal is only valid on assistant messages", "input");
        }
        if (!value.contains("refusal") || !value.at("refusal").is_string()) {
            bad_request("refusal must contain a string refusal", "input");
        }
        append_message_text(parsed, content, value.at("refusal").get<std::string>(), type, &value,
                            breakpoint_count);
        return;
    }
    if (type == "input_image") {
        if (role != ChatRole::User && role != ChatRole::Assistant) {
            bad_request("input_image is only supported on user or assistant messages", "input");
        }
        ContentPart part;
        part.kind     = ContentKind::Image;
        part.type_raw = type;
        part.source   = parse_image_source(value);
        apply_shared_breakpoint(part, value, breakpoint_count);
        parsed.turn.content.push_back(std::move(part));
        Json canonical{
            {"type", "input_image"}, {"image_url", value.at("image_url")}, {"detail", "auto"}};
        if (value.contains("prompt_cache_breakpoint")) {
            canonical["prompt_cache_breakpoint"] = value.at("prompt_cache_breakpoint");
        }
        content.push_back(std::move(canonical));
        return;
    }
    if (type == "input_video") {
        if (role != ChatRole::User) {
            bad_request("input_video is only supported on user messages", "input");
        }
        ContentPart part;
        part.kind     = ContentKind::Video;
        part.type_raw = type;
        part.source   = parse_video_source(value);
        parsed.turn.content.push_back(std::move(part));
        content.push_back(Json{{"type", "input_video"}, {"video_url", value.at("video_url")}});
        return;
    }
    if (type == "input_file") {
        bad_request("input_file requires a Files API, which NInfer does not provide", "input",
                    "file_inputs_not_supported");
    }
    if (type == "input_audio") {
        bad_request("input_audio is not supported by the Engine", "input",
                    "audio_inputs_not_supported");
    }
    bad_request("unsupported message content type: " + type, "input", "modality_not_supported");
}

ParsedMessage parse_message_item(const Json& item, std::size_t index,
                                 std::size_t& breakpoint_count) {
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

    if (item.contains("status") && !item.at("status").is_null() && !item.at("status").is_string()) {
        bad_request("input message status must be a string", "input");
    }
    if (item.contains("phase") && !item.at("phase").is_null() && !item.at("phase").is_string()) {
        bad_request("input message phase must be a string", "input");
    }
    if (!item.contains("content") || item.at("content").is_null()) {
        bad_request("input message " + std::to_string(index) + " must contain content", "input");
    }

    ParsedMessage parsed;
    parsed.turn.role = parsed_role;
    Json content     = Json::array();

    if (item.at("content").is_string()) {
        append_message_text(parsed, content, item.at("content").get<std::string>(),
                            parsed_role == ChatRole::Assistant ? "output_text" : "input_text",
                            nullptr, breakpoint_count);
    } else if (item.at("content").is_array()) {
        for (const Json& value : item.at("content")) {
            parse_message_content_part(value, parsed_role, parsed, content, breakpoint_count);
        }
    } else {
        bad_request("input message content must be a string or array", "input");
    }
    if (parsed.turn.content.empty()) {
        bad_request("input message content must not be empty", "input");
    }

    parsed.canonical = {{"id", item_id(item, "msg")},
                        {"type", "message"},
                        {"role", role},
                        {"content", std::move(content)}};
    if (item.contains("status") && !item.at("status").is_null()) {
        parsed.canonical["status"] = item.at("status");
    }
    if (item.contains("phase") && !item.at("phase").is_null()) {
        parsed.canonical["phase"] = item.at("phase");
    }
    return parsed;
}

std::string parse_reasoning_item(const Json& item, Json& canonical) {
    const bool has_encrypted =
        item.contains("encrypted_content") && !item.at("encrypted_content").is_null();
    if (has_encrypted && !item.at("encrypted_content").is_string()) {
        bad_request("reasoning encrypted_content must be a string", "input");
    }
    if (item.contains("summary") && !item.at("summary").is_null() &&
        !item.at("summary").is_array()) {
        bad_request("reasoning summary must be an array", "input");
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
    if (text.empty() &&
        (has_encrypted || (item.contains("summary") && !item.at("summary").is_null() &&
                           !item.at("summary").empty()))) {
        bad_request("reasoning Items require raw reasoning_text; summary or encrypted content "
                    "cannot reconstruct the model context",
                    "input", "reasoning_content_not_supported");
    }

    canonical = {{"id", item_id(item, "rs")},
                 {"type", "reasoning"},
                 {"summary", item.contains("summary") && !item.at("summary").is_null()
                                 ? item.at("summary")
                                 : Json::array()},
                 {"content", std::move(content)}};
    if (has_encrypted) { canonical["encrypted_content"] = item.at("encrypted_content"); }
    return text;
}

void reject_nonnull_unknown_members(const Json& object,
                                    const std::unordered_set<std::string>& allowed,
                                    const char* param) {
    for (auto iterator = object.begin(); iterator != object.end(); ++iterator) {
        if (!allowed.contains(iterator.key()) && !iterator.value().is_null()) {
            bad_request("unsupported " + std::string(param) + " member: " + iterator.key(), param,
                        "parameter_not_supported");
        }
    }
}

ToolCall parse_function_call_item(
    const Json& item, Json& canonical,
    std::unordered_map<std::string, OpenAIResponsesFunctionIdentity>& identities) {
    static const std::unordered_set<std::string> allowed = {
        "id", "type", "call_id", "name", "arguments", "status", "caller", "namespace"};
    reject_nonnull_unknown_members(item, allowed, "input");
    if (item.contains("caller") && !item.at("caller").is_null()) {
        bad_request("function_call.caller is not supported", "input",
                    "tool_relationship_not_supported");
    }

    ToolCall call;
    if (!item.contains("call_id") || !item.at("call_id").is_string() ||
        item.at("call_id").get_ref<const std::string&>().empty()) {
        bad_request("function_call must contain a non-empty call_id", "input");
    }
    call.id                                        = item.at("call_id").get<std::string>();
    const OpenAIResponsesFunctionIdentity identity = function_identity(item, "input");
    call.name = lower_function_identity(identity, identities, "input");
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
        bad_request("partial function_call Items cannot be represented in model history", "input",
                    "partial_tool_call_not_supported");
    }
    canonical = {{"id", item_id(item, "fc")},
                 {"type", "function_call"},
                 {"status", "completed"},
                 {"call_id", call.id},
                 {"arguments", call.arguments_json}};
    add_wire_function_identity(canonical, identity);
    return call;
}

ContentPart tool_output_text(std::string text, const Json* wire, std::size_t& breakpoint_count) {
    ContentPart part;
    part.kind     = ContentKind::Text;
    part.type_raw = "input_text";
    part.text     = std::move(text);
    if (wire != nullptr) { apply_shared_breakpoint(part, *wire, breakpoint_count); }
    return part;
}

ChatTurn parse_function_call_output_item(
    const Json& item, Json& canonical, std::size_t& breakpoint_count,
    std::unordered_map<std::string, OpenAIResponsesFunctionIdentity>& identities) {
    static const std::unordered_set<std::string> allowed = {
        "id", "type", "call_id", "output", "status", "caller", "name", "namespace"};
    reject_nonnull_unknown_members(item, allowed, "input");
    if (item.contains("caller") && !item.at("caller").is_null()) {
        bad_request("function_call_output.caller is not supported", "input",
                    "tool_relationship_not_supported");
    }
    if (!item.contains("call_id") || !item.at("call_id").is_string() ||
        item.at("call_id").get_ref<const std::string&>().empty()) {
        bad_request("function_call_output must contain a non-empty call_id", "input");
    }
    if (!item.contains("output")) {
        bad_request("function_call_output must contain output", "input");
    }
    if (item.contains("status") && !item.at("status").is_null() &&
        (!item.at("status").is_string() || item.at("status").get<std::string>() != "completed")) {
        bad_request("partial function_call_output Items cannot be represented in model history",
                    "input", "partial_tool_result_not_supported");
    }

    ChatTurn turn;
    turn.role                = ChatRole::Tool;
    turn.tool_call_id        = item.at("call_id").get<std::string>();
    const bool has_name      = item.contains("name") && !item.at("name").is_null();
    const bool has_namespace = item.contains("namespace") && !item.at("namespace").is_null();
    if (has_namespace && !has_name) {
        bad_request("function_call_output.namespace requires function_call_output.name", "input",
                    "invalid_tool_history");
    }
    std::optional<OpenAIResponsesFunctionIdentity> asserted_identity;
    if (has_name) {
        asserted_identity     = function_identity(item, "input");
        turn.tool_result_name = lower_function_identity(*asserted_identity, identities, "input");
    }
    if (item.at("output").is_string()) {
        turn.content.push_back(
            tool_output_text(item.at("output").get<std::string>(), nullptr, breakpoint_count));
    } else if (item.at("output").is_array()) {
        for (const Json& value : item.at("output")) {
            if (!value.is_object() || !value.contains("type") || !value.at("type").is_string()) {
                bad_request("function_call_output content parts must have a string type", "input");
            }
            const std::string type = value.at("type").get<std::string>();
            if (type == "input_text") {
                if (!value.contains("text") || !value.at("text").is_string()) {
                    bad_request("tool result input_text must contain a string text", "input");
                }
                turn.content.push_back(tool_output_text(value.at("text").get<std::string>(), &value,
                                                        breakpoint_count));
            } else if (type == "input_image") {
                ContentPart part;
                part.kind     = ContentKind::Image;
                part.type_raw = type;
                part.source   = parse_image_source(value);
                apply_shared_breakpoint(part, value, breakpoint_count);
                turn.content.push_back(std::move(part));
            } else if (type == "input_file") {
                bad_request("tool result input_file requires a Files API", "input",
                            "file_inputs_not_supported");
            } else {
                bad_request("unsupported function_call_output content type: " + type, "input",
                            "modality_not_supported");
            }
        }
        if (turn.content.empty()) {
            bad_request("function_call_output content must not be empty", "input");
        }
    } else {
        bad_request("function_call_output output must be a string or content array", "input");
    }

    canonical = {{"id", item_id(item, "fco")},
                 {"type", "function_call_output"},
                 {"status", "completed"},
                 {"call_id", turn.tool_call_id},
                 {"output", item.at("output")}};
    if (asserted_identity) { add_wire_function_identity(canonical, *asserted_identity); }
    return turn;
}

enum class AssistantInputPhase {
    Empty,
    Reasoning,
    Content,
    Calls,
};

[[noreturn]] void invalid_assistant_history(std::size_t index, std::string message) {
    bad_request("input Item " + std::to_string(index) + " " + std::move(message) +
                    "; supported assistant Item order is reasoning, message content, then "
                    "function calls",
                "input", "invalid_assistant_history");
}

struct AssistantInputRun {
    AssistantInputRun() { reset(); }

    void append_reasoning(std::string reasoning, std::size_t index) {
        switch (phase) {
        case AssistantInputPhase::Empty:
            turn.reasoning_content = std::move(reasoning);
            phase                  = AssistantInputPhase::Reasoning;
            return;
        case AssistantInputPhase::Reasoning:
            invalid_assistant_history(index, "reasoning cannot follow another reasoning Item");
        case AssistantInputPhase::Content:
            invalid_assistant_history(index, "reasoning cannot follow assistant message content");
        case AssistantInputPhase::Calls:
            invalid_assistant_history(index, "reasoning cannot follow function_call Items");
        }
        throw std::logic_error("unreachable assistant input phase");
    }

    void append_message(ChatTurn message, std::size_t index) {
        if (message.role != ChatRole::Assistant) {
            throw std::logic_error("assistant input run received a non-assistant message");
        }
        if (phase == AssistantInputPhase::Calls) {
            invalid_assistant_history(
                index, "assistant message content cannot follow function_call Items");
        }
        turn.content.insert(turn.content.end(), std::make_move_iterator(message.content.begin()),
                            std::make_move_iterator(message.content.end()));
        phase = AssistantInputPhase::Content;
    }

    void append_call(ToolCall call) {
        turn.tool_calls.push_back(std::move(call));
        phase = AssistantInputPhase::Calls;
    }

    void flush(std::vector<ChatTurn>& turns) {
        if (phase == AssistantInputPhase::Empty) { return; }
        if (!turn.reasoning_content.empty() || !turn.content.empty() || !turn.tool_calls.empty()) {
            turns.push_back(std::move(turn));
        }
        reset();
    }

private:
    void reset() {
        turn      = ChatTurn{};
        turn.role = ChatRole::Assistant;
        phase     = AssistantInputPhase::Empty;
    }

    ChatTurn turn;
    AssistantInputPhase phase = AssistantInputPhase::Empty;
};

void parse_input(const Json& input, OpenAIResponsesPromptRequest& out,
                 std::unordered_map<std::string, OpenAIResponsesFunctionIdentity>& identities) {
    Json values;
    if (input.is_string()) {
        values = Json::array({Json{{"type", "message"}, {"role", "user"}, {"content", input}}});
    } else if (input.is_array()) {
        values = input;
    } else {
        bad_request("input must be a string or an array of Items", "input");
    }

    AssistantInputRun assistant;
    std::size_t breakpoint_count = 0;
    std::unordered_set<std::string> item_ids;
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
            ParsedMessage message = parse_message_item(item, index, breakpoint_count);
            canonical             = std::move(message.canonical);
            if (message.turn.role == ChatRole::Assistant) {
                assistant.append_message(std::move(message.turn), index);
            } else {
                assistant.flush(out.input_turns);
                out.input_turns.push_back(std::move(message.turn));
            }
        } else if (type == "reasoning") {
            assistant.append_reasoning(parse_reasoning_item(item, canonical), index);
        } else if (type == "function_call") {
            assistant.append_call(parse_function_call_item(item, canonical, identities));
        } else if (type == "function_call_output") {
            ChatTurn result =
                parse_function_call_output_item(item, canonical, breakpoint_count, identities);
            assistant.flush(out.input_turns);
            out.input_turns.push_back(std::move(result));
        } else if (type == "input_file") {
            bad_request("input_file requires a Files API, which NInfer does not provide", "input",
                        "file_inputs_not_supported");
        } else {
            bad_request("unsupported input Item type: " + type, "input", "item_type_not_supported");
        }

        const std::string id = canonical.at("id").get<std::string>();
        if (!item_ids.insert(id).second) { bad_request("duplicate input Item id: " + id, "input"); }
        out.input_items.push_back(std::move(canonical));
    }
    assistant.flush(out.input_turns);
}

struct ParsedPromptFields {
    OpenAIResponsesPromptRequest prompt;
    Json wire_tools          = Json::array();
    Json wire_tool_choice    = "auto";
    bool parallel_tool_calls = true;
    std::unordered_map<std::string, OpenAIResponsesFunctionIdentity> tool_identities;
};

struct ParsedFunctionTool {
    ToolDefinition definition;
    Json canonical;
    std::string engine_name;
};

ParsedFunctionTool
parse_function_tool(const Json& item, std::optional<std::string> wire_namespace,
                    std::string_view namespace_description,
                    std::unordered_map<std::string, OpenAIResponsesFunctionIdentity>& identities) {
    static const std::unordered_set<std::string> allowed_members = {
        "type",          "name",         "description", "parameters", "strict", "allowed_callers",
        "defer_loading", "output_schema"};
    reject_nonnull_unknown_members(item, allowed_members, "tools");

    const OpenAIResponsesFunctionIdentity identity{.name = require_function_name(item, "tools"),
                                                   .wire_namespace = std::move(wire_namespace)};
    ParsedFunctionTool parsed;
    parsed.engine_name     = lower_function_identity(identity, identities, "tools");
    parsed.definition.name = parsed.engine_name;

    std::string function_description;
    if (item.contains("description") && !item.at("description").is_null()) {
        if (!item.at("description").is_string()) {
            bad_request("function description must be a string", "tools");
        }
        function_description = item.at("description").get<std::string>();
    }
    if (!namespace_description.empty()) {
        parsed.definition.description = std::string(namespace_description);
        if (!function_description.empty()) {
            parsed.definition.description += "\n\n" + function_description;
        }
    } else {
        parsed.definition.description = function_description;
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
            bad_request("strict function schema enforcement requires constrained decoding, "
                        "which the Engine does not provide",
                        "tools", "strict_tools_not_supported");
        }
    }
    if (item.contains("defer_loading") && !item.at("defer_loading").is_null()) {
        if (!item.at("defer_loading").is_boolean()) {
            bad_request("function defer_loading must be a boolean", "tools");
        }
        if (item.at("defer_loading").get<bool>()) {
            bad_request("deferred tool loading is not supported", "tools",
                        "deferred_tools_not_supported");
        }
    }
    if (item.contains("allowed_callers") && !item.at("allowed_callers").is_null()) {
        const Json& callers = item.at("allowed_callers");
        if (!callers.is_array()) {
            bad_request("function allowed_callers must be an array", "tools");
        }
        bool direct = false;
        for (const Json& caller : callers) {
            if (!caller.is_string()) {
                bad_request("function allowed_callers entries must be strings", "tools");
            }
            direct = direct || caller.get<std::string>() == "direct";
        }
        if (!direct) {
            bad_request("function allowed_callers must permit direct invocation", "tools",
                        "tool_caller_not_supported");
        }
    }
    if (item.contains("output_schema") && !item.at("output_schema").is_null()) {
        bad_request("function output_schema cannot be enforced", "tools",
                    "tool_output_schema_not_supported");
    }

    parsed.definition.input_schema_json = parameters.dump();
    parsed.canonical                    = {{"type", "function"},
                                           {"name", identity.name},
                                           {"parameters", parameters},
                                           {"strict", false}};
    if (!function_description.empty()) {
        parsed.canonical["description"] = std::move(function_description);
    }
    if (item.contains("allowed_callers") && !item.at("allowed_callers").is_null()) {
        parsed.canonical["allowed_callers"] = item.at("allowed_callers");
    }
    if (item.contains("defer_loading") && !item.at("defer_loading").is_null()) {
        parsed.canonical["defer_loading"] = false;
    }
    return parsed;
}

void parse_tools(const Json& body, ParsedPromptFields& out) {
    if (!body.contains("tools") || body.at("tools").is_null()) { return; }
    if (!body.at("tools").is_array()) { bad_request("tools must be an array", "tools"); }

    static const std::unordered_set<std::string> namespace_members = {"type", "name", "description",
                                                                      "tools"};
    std::unordered_set<std::string> declared_names;
    std::unordered_set<std::string> namespace_names;
    const auto append_function = [&](ParsedFunctionTool parsed) {
        if (!declared_names.insert(parsed.engine_name).second) {
            bad_request("duplicate function tool identity: " + parsed.engine_name, "tools",
                        "duplicate_tool_name");
        }
        out.prompt.generation.tools.push_back(std::move(parsed.definition));
        return std::move(parsed.canonical);
    };

    for (const Json& item : body.at("tools")) {
        if (!item.is_object() || !item.contains("type") || !item.at("type").is_string()) {
            bad_request("tools entries must be objects with a string type", "tools");
        }
        const std::string type = item.at("type").get<std::string>();
        if (type == "function") {
            out.wire_tools.push_back(
                append_function(parse_function_tool(item, std::nullopt, {}, out.tool_identities)));
            continue;
        }
        if (type != "namespace") {
            bad_request("tool type '" + type +
                            "' requires an executor that NInfer does not provide",
                        "tools", "tool_type_not_supported");
        }

        // OpenAI Responses beta groups functions/custom tools under a namespace. NInfer lowers
        // only nested functions because custom tools require unsupported free-form decoding.
        reject_nonnull_unknown_members(item, namespace_members, "tools");
        const std::string namespace_name = require_namespace_tool_name(item);
        if (!namespace_names.insert(namespace_name).second) {
            bad_request("duplicate namespace tool name: " + namespace_name, "tools",
                        "duplicate_tool_name");
        }
        std::string namespace_description;
        if (item.contains("description") && !item.at("description").is_null()) {
            if (!item.at("description").is_string()) {
                bad_request("namespace description must be a string", "tools");
            }
            namespace_description = item.at("description").get<std::string>();
        }
        if (!item.contains("tools") || !item.at("tools").is_array()) {
            bad_request("namespace tools must contain a tools array", "tools");
        }
        Json canonical = {
            {"type", "namespace"}, {"name", namespace_name}, {"tools", Json::array()}};
        if (!namespace_description.empty()) { canonical["description"] = namespace_description; }
        for (const Json& nested : item.at("tools")) {
            if (!nested.is_object() || !nested.contains("type") || !nested.at("type").is_string()) {
                bad_request("namespace tool entries must be objects with a string type", "tools");
            }
            const std::string nested_type = nested.at("type").get<std::string>();
            if (nested_type != "function") {
                bad_request("nested tool type '" + nested_type +
                                "' cannot be represented by the Engine",
                            "tools", "tool_type_not_supported");
            }
            canonical["tools"].push_back(append_function(parse_function_tool(
                nested, namespace_name, namespace_description, out.tool_identities)));
        }
        out.wire_tools.push_back(std::move(canonical));
    }
}

void filter_allowed_tools(const Json& choice, ParsedPromptFields& out) {
    static const std::unordered_set<std::string> allowed_choice = {"type", "mode", "tools"};
    reject_nonnull_unknown_members(choice, allowed_choice, "tool_choice");
    if (!choice.contains("mode") || !choice.at("mode").is_string()) {
        bad_request("allowed_tools tool_choice must contain a string mode", "tool_choice");
    }
    if (choice.at("mode").get<std::string>() != "auto") {
        bad_request("allowed_tools mode 'required' cannot be enforced", "tool_choice",
                    "tool_choice_not_supported");
    }
    if (!choice.contains("tools") || !choice.at("tools").is_array()) {
        bad_request("allowed_tools tool_choice must contain a tools array", "tool_choice");
    }

    std::unordered_set<std::string> declared;
    for (const ToolDefinition& tool : out.prompt.generation.tools) { declared.insert(tool.name); }
    std::unordered_set<std::string> selected;
    for (const Json& item : choice.at("tools")) {
        if (!item.is_object() || !item.contains("type") || !item.at("type").is_string() ||
            item.at("type").get<std::string>() != "function") {
            bad_request("allowed_tools only supports function entries", "tool_choice",
                        "tool_choice_not_supported");
        }
        static const std::unordered_set<std::string> allowed_entry = {"type", "name", "namespace"};
        reject_nonnull_unknown_members(item, allowed_entry, "tool_choice");
        const OpenAIResponsesFunctionIdentity identity = function_identity(item, "tool_choice");
        const std::string name =
            lower_function_identity(identity, out.tool_identities, "tool_choice");
        if (!declared.contains(name)) {
            const std::string wire_name = identity.wire_namespace
                                              ? *identity.wire_namespace + "." + identity.name
                                              : identity.name;
            bad_request("allowed_tools references undeclared function '" + wire_name + "'",
                        "tool_choice", "invalid_tool_choice");
        }
        selected.insert(name);
    }

    std::vector<ToolDefinition> effective;
    effective.reserve(selected.size());
    for (ToolDefinition& tool : out.prompt.generation.tools) {
        if (selected.contains(tool.name)) { effective.push_back(std::move(tool)); }
    }
    out.prompt.generation.tools = std::move(effective);
}

void parse_tool_choice(const Json& body, ParsedPromptFields& out) {
    if (!body.contains("tool_choice") || body.at("tool_choice").is_null()) {
        out.wire_tool_choice = "auto";
        return;
    }
    const Json& choice = body.at("tool_choice");
    if (choice.is_string()) {
        const std::string value = choice.get<std::string>();
        if (value == "auto") {
            out.prompt.generation.tool_choice.mode = ToolChoiceMode::Auto;
        } else if (value == "none") {
            out.prompt.generation.tool_choice.mode = ToolChoiceMode::None;
        } else if (value == "required") {
            bad_request("tool_choice 'required' cannot be guaranteed by the Engine", "tool_choice",
                        "tool_choice_not_supported");
        } else {
            bad_request("tool_choice must be 'auto', 'none', or a supported object", "tool_choice");
        }
        out.wire_tool_choice = value;
        return;
    }
    if (!choice.is_object() || !choice.contains("type") || !choice.at("type").is_string()) {
        bad_request("tool_choice must be a string or typed object", "tool_choice");
    }
    if (choice.at("type").get<std::string>() != "allowed_tools") {
        bad_request("named or hosted tool_choice cannot be enforced", "tool_choice",
                    "tool_choice_not_supported");
    }
    filter_allowed_tools(choice, out);
    out.wire_tool_choice = choice;
}

void parse_reasoning(const Json& body, OpenAIResponsesPromptRequest& out) {
    if (!body.contains("reasoning") || body.at("reasoning").is_null()) { return; }
    const Json& reasoning = body.at("reasoning");
    if (!reasoning.is_object()) { bad_request("reasoning must be an object", "reasoning"); }
    static const std::unordered_set<std::string> allowed = {"effort", "context", "summary",
                                                            "generate_summary", "mode"};
    reject_nonnull_unknown_members(reasoning, allowed, "reasoning");
    for (const char* key : {"context", "summary", "generate_summary", "mode"}) {
        if (reasoning.contains(key) && !reasoning.at(key).is_null()) {
            bad_request("reasoning." + std::string(key) +
                            " changes reasoning input or output and is not supported",
                        "reasoning", "reasoning_option_not_supported");
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
    out.generation.reasoning_effort = *effort;
}

void parse_text(const Json& body) {
    if (!body.contains("text") || body.at("text").is_null()) { return; }
    const Json& text = body.at("text");
    if (!text.is_object()) { bad_request("text must be an object", "text"); }
    static const std::unordered_set<std::string> allowed = {"format", "verbosity"};
    reject_nonnull_unknown_members(text, allowed, "text");
    if (text.contains("format") && !text.at("format").is_null()) {
        const Json& format = text.at("format");
        if (!format.is_object() || !format.contains("type") || !format.at("type").is_string()) {
            bad_request("text.format must be a typed object", "text");
        }
        if (format.at("type").get<std::string>() != "text" || format.size() != 1) {
            bad_request("structured text output requires constrained decoding, which the Engine "
                        "does not provide",
                        "text", "structured_outputs_not_supported");
        }
    }
    if (text.contains("verbosity") && !text.at("verbosity").is_null()) {
        if (!text.at("verbosity").is_string()) {
            bad_request("text.verbosity must be a string", "text");
        }
        const std::string verbosity = text.at("verbosity").get<std::string>();
        if (verbosity != "medium") {
            bad_request("text.verbosity '" + verbosity + "' cannot be enforced by the Engine",
                        "text", "verbosity_not_supported");
        }
    }
}

void parse_preserve_thinking(const Json& body, OpenAIResponsesPromptRequest& out) {
    if (body.contains("preserve_thinking") && !body.at("preserve_thinking").is_null()) {
        if (!body.at("preserve_thinking").is_boolean()) {
            bad_request("preserve_thinking must be a boolean or null", "preserve_thinking");
        }
        out.generation.preserve_thinking = body.at("preserve_thinking").get<bool>();
    }
    if (!body.contains("chat_template_kwargs") || body.at("chat_template_kwargs").is_null()) {
        return;
    }
    const Json& kwargs = body.at("chat_template_kwargs");
    if (!kwargs.is_object()) {
        bad_request("chat_template_kwargs must be an object", "chat_template_kwargs");
    }
    for (auto iterator = kwargs.begin(); iterator != kwargs.end(); ++iterator) {
        if (iterator.key() != "preserve_thinking" && !iterator.value().is_null()) {
            bad_request("chat_template_kwargs." + iterator.key() + " is not supported",
                        "chat_template_kwargs", "chat_template_option_not_supported");
        }
    }
    if (!kwargs.contains("preserve_thinking") || kwargs.at("preserve_thinking").is_null()) {
        return;
    }
    if (!kwargs.at("preserve_thinking").is_boolean()) {
        bad_request("chat_template_kwargs.preserve_thinking must be a boolean or null",
                    "chat_template_kwargs");
    }
    const bool nested = kwargs.at("preserve_thinking").get<bool>();
    if (out.generation.preserve_thinking && *out.generation.preserve_thinking != nested) {
        bad_request("conflicting preserve_thinking values", "preserve_thinking",
                    "conflicting_template_option");
    }
    out.generation.preserve_thinking = nested;
}

void parse_truncation(const Json& body) {
    if (!body.contains("truncation") || body.at("truncation").is_null()) { return; }
    if (!body.at("truncation").is_string()) {
        bad_request("truncation must be a string", "truncation");
    }
    if (body.at("truncation").get<std::string>() != "disabled") {
        bad_request("truncation 'auto' would discard input Items and is not supported",
                    "truncation", "truncation_not_supported");
    }
}

ParsedPromptFields parse_prompt_fields(const Json& body, const RequestLimits& limits) {
    ParsedPromptFields out;
    if (!body.contains("model") || !body.at("model").is_string() ||
        body.at("model").get_ref<const std::string&>().empty()) {
        bad_request("missing required field: model", "model");
    }
    out.prompt.model = body.at("model").get<std::string>();
    if (body.contains("input") && !body.at("input").is_null()) {
        parse_input(body.at("input"), out.prompt, out.tool_identities);
    }
    if (body.contains("instructions") && !body.at("instructions").is_null()) {
        if (!body.at("instructions").is_string()) {
            bad_request("instructions must be a string", "instructions");
        }
        out.prompt.instructions = body.at("instructions").get<std::string>();
    }
    if (body.contains("previous_response_id") && !body.at("previous_response_id").is_null()) {
        if (!body.at("previous_response_id").is_string() ||
            body.at("previous_response_id").get_ref<const std::string&>().empty()) {
            bad_request("previous_response_id must be a non-empty string", "previous_response_id");
        }
        out.prompt.previous_response_id = body.at("previous_response_id").get<std::string>();
    }

    parse_tools(body, out);
    parse_tool_choice(body, out);
    out.parallel_tool_calls = optional_bool(body, "parallel_tool_calls", true);
    if (!out.parallel_tool_calls && out.prompt.generation.uses_tools()) {
        bad_request("parallel_tool_calls=false cannot be guaranteed when callable tools are "
                    "present",
                    "parallel_tool_calls", "parallel_tool_calls_not_supported");
    }
    parse_reasoning(body, out.prompt);
    parse_text(body);
    parse_truncation(body);
    parse_preserve_thinking(body, out.prompt);
    out.prompt.generation.max_tokens = limits.default_max_tokens;
    return out;
}

void validate_metadata(const Json& body, Json& metadata) {
    if (!body.contains("metadata") || body.at("metadata").is_null()) { return; }
    if (!body.at("metadata").is_object()) { bad_request("metadata must be an object", "metadata"); }
    if (body.at("metadata").size() > 16) {
        bad_request("metadata supports at most 16 entries", "metadata");
    }
    for (auto iterator = body.at("metadata").begin(); iterator != body.at("metadata").end();
         ++iterator) {
        if (iterator.key().size() > 64 || !iterator.value().is_string() ||
            iterator.value().get_ref<const std::string&>().size() > 512) {
            bad_request("metadata keys must be at most 64 characters and string values at most "
                        "512 characters",
                        "metadata");
        }
    }
    metadata = body.at("metadata");
}

void reject_unsupported_platform_fields(const Json& body) {
    const struct {
        const char* field;
        const char* code;
        const char* reason;
    } unsupported[] = {
        {"conversation", "conversations_not_supported",
         "conversation requires an OpenAI Conversations resource"},
        {"prompt", "prompt_templates_not_supported",
         "prompt requires an OpenAI prompt-template resource"},
        {"context_management", "context_management_not_supported",
         "context_management requires an API compaction pipeline"},
        {"moderation", "moderation_not_supported",
         "moderation changes request acceptance and output but no moderator is configured"},
    };

    for (const auto& entry : unsupported) {
        if (body.contains(entry.field) && !body.at(entry.field).is_null()) {
            bad_request(entry.reason, entry.field, entry.code);
        }
    }
}

void validate_common_top_level(const Json& body, bool create) {
    static const std::unordered_set<std::string> create_fields = {"background",
                                                                  "chat_template_kwargs",
                                                                  "client_metadata",
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
                                                                  "user"};
    static const std::unordered_set<std::string> count_fields  = {"chat_template_kwargs",
                                                                  "conversation",
                                                                  "input",
                                                                  "instructions",
                                                                  "model",
                                                                  "parallel_tool_calls",
                                                                  "personality",
                                                                  "previous_response_id",
                                                                  "preserve_thinking",
                                                                  "reasoning",
                                                                  "text",
                                                                  "tool_choice",
                                                                  "tools",
                                                                  "truncation"};
    const auto& allowed = create ? create_fields : count_fields;
    for (auto iterator = body.begin(); iterator != body.end(); ++iterator) {
        if (!allowed.contains(iterator.key())) {
            bad_request("unknown parameter: " + iterator.key(), iterator.key(),
                        "unknown_parameter");
        }
    }
}

} // namespace

OpenAIResponsesCreateRequest parse_openai_responses_create_request(const Json& body,
                                                                   const RequestLimits& limits) {
    require_object(body);
    validate_common_top_level(body, true);
    reject_unsupported_platform_fields(body);
    const OpenAIPromptCachePolicy cache_policy = parse_openai_prompt_cache_policy(body);

    ParsedPromptFields parsed = parse_prompt_fields(body, limits);
    apply_openai_prompt_cache_policy(parsed.prompt.generation, cache_policy);
    OpenAIResponsesCreateRequest out;
    out.prompt              = std::move(parsed.prompt);
    out.tools               = std::move(parsed.wire_tools);
    out.tool_choice         = std::move(parsed.wire_tool_choice);
    out.tool_identities     = std::move(parsed.tool_identities);
    out.parallel_tool_calls = parsed.parallel_tool_calls;
    out.store               = optional_bool(body, "store", true);
    out.stream              = optional_bool(body, "stream", false);
    validate_metadata(body, out.metadata);

    // Codex attaches per-request tracing information here. It is an opaque client hint and has no
    // Engine, cache-identity, response-store, or response-body semantics.
    if (body.contains("client_metadata") && !body.at("client_metadata").is_null() &&
        !body.at("client_metadata").is_object()) {
        bad_request("client_metadata must be an object or null", "client_metadata", "invalid_type");
    }

    if (body.contains("background") && !body.at("background").is_null()) {
        if (!body.at("background").is_boolean()) {
            bad_request("background must be a boolean", "background");
        }
        if (body.at("background").get<bool>()) {
            bad_request("background execution is not supported", "background",
                        "background_not_supported");
        }
    }
    if (body.contains("include") && !body.at("include").is_null()) {
        if (!body.at("include").is_array()) { bad_request("include must be an array", "include"); }
        if (!body.at("include").empty()) {
            bad_request("the requested additional response fields have no available response "
                        "representation",
                        "include", "include_not_supported");
        }
    }
    if (body.contains("stream_options") && !body.at("stream_options").is_null()) {
        const Json& options = body.at("stream_options");
        if (!options.is_object()) {
            bad_request("stream_options must be an object", "stream_options");
        }
        static const std::unordered_set<std::string> allowed = {"include_obfuscation"};
        reject_nonnull_unknown_members(options, allowed, "stream_options");
        if (options.contains("include_obfuscation") &&
            !options.at("include_obfuscation").is_null() &&
            !options.at("include_obfuscation").is_boolean()) {
            bad_request("stream_options.include_obfuscation must be a boolean", "stream_options");
        }
    }
    if (body.contains("service_tier") && !body.at("service_tier").is_null()) {
        if (!body.at("service_tier").is_string()) {
            bad_request("service_tier must be a string", "service_tier");
        }
        const std::string tier = body.at("service_tier").get<std::string>();
        if (tier != "auto" && tier != "default") {
            bad_request("the requested service tier is not provided by this local server",
                        "service_tier", "service_tier_not_supported");
        }
    }
    if (const std::optional<int> top_logprobs = optional_int(body, "top_logprobs")) {
        if (*top_logprobs < 0 || *top_logprobs > 20) {
            bad_request("top_logprobs must be in [0,20]", "top_logprobs");
        }
        if (*top_logprobs != 0) {
            bad_request("the Engine does not return token log probabilities", "top_logprobs",
                        "logprobs_not_supported");
        }
    }
    if (const std::optional<int> max_tool_calls = optional_int(body, "max_tool_calls")) {
        if (*max_tool_calls < 0) {
            bad_request("max_tool_calls must be non-negative", "max_tool_calls");
        }
        out.max_tool_calls = *max_tool_calls;
    }

    if (const std::optional<double> temperature = optional_number(body, "temperature")) {
        if (*temperature < 0.0 || *temperature > 2.0) {
            bad_request("temperature must be in [0,2]", "temperature");
        }
        out.prompt.generation.sampling.temperature = *temperature;
    }
    if (const std::optional<double> top_p = optional_number(body, "top_p")) {
        if (*top_p < 0.0 || *top_p > 1.0) { bad_request("top_p must be in [0,1]", "top_p"); }
        out.prompt.generation.sampling.top_p = *top_p;
    }
    if (const std::optional<int> max_output = optional_int(body, "max_output_tokens")) {
        if (*max_output < 0) {
            bad_request("max_output_tokens must be non-negative", "max_output_tokens");
        }
        out.requested_max_output_tokens  = *max_output;
        out.prompt.generation.max_tokens = *max_output;
    }
    return out;
}

OpenAIResponsesPromptRequest
parse_openai_responses_input_tokens_request(const Json& body, const RequestLimits& limits) {
    require_object(body);
    validate_common_top_level(body, false);
    reject_unsupported_platform_fields(body);
    if (body.contains("personality") && !body.at("personality").is_null()) {
        bad_request("personality changes prompt construction and is not supported", "personality",
                    "personality_not_supported");
    }
    return std::move(parse_prompt_fields(body, limits).prompt);
}

} // namespace ninfer::serve
