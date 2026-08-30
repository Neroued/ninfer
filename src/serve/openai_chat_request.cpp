#include "serve/openai_chat.h"
#include "serve/openai_common.h"
#include "serve/request_validation.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

void require_object(const Json& value, const char* message, const char* param = nullptr) {
    if (!value.is_object()) { bad_request(message, param == nullptr ? "" : param); }
}

bool get_bool(const Json& object, const char* key, bool fallback) {
    if (!object.contains(key) || object.at(key).is_null()) { return fallback; }
    if (!object.at(key).is_boolean()) { bad_request(std::string(key) + " must be a boolean", key); }
    return object.at(key).get<bool>();
}

std::optional<bool> get_optional_bool(const Json& object, const char* key) {
    if (!object.contains(key) || object.at(key).is_null()) { return std::nullopt; }
    if (!object.at(key).is_boolean()) {
        bad_request(std::string(key) + " must be a boolean or null", key);
    }
    return object.at(key).get<bool>();
}

std::optional<double> get_number(const Json& object, const char* key) {
    if (!object.contains(key) || object.at(key).is_null()) { return std::nullopt; }
    if (!object.at(key).is_number()) { bad_request(std::string(key) + " must be a number", key); }
    return object.at(key).get<double>();
}

std::optional<std::uint64_t> get_seed(const Json& object) {
    if (!object.contains("seed") || object.at("seed").is_null()) { return std::nullopt; }
    const Json& value = object.at("seed");
    if (!value.is_number_integer()) { bad_request("seed must be an integer", "seed"); }
    if (value.is_number_unsigned()) { return value.get<std::uint64_t>(); }
    return static_cast<std::uint64_t>(value.get<std::int64_t>());
}

ChatRole parse_message_role(const std::string& role) {
    if (role == "system") { return ChatRole::System; }
    if (role == "developer") { return ChatRole::Developer; }
    if (role == "user") { return ChatRole::User; }
    if (role == "assistant") { return ChatRole::Assistant; }
    if (role == "tool") { return ChatRole::Tool; }
    bad_request("unsupported role: " + role, "messages", "unsupported_role");
}

std::string require_function_name(const Json& object, const char* param) {
    if (!object.contains("name") || !object.at("name").is_string()) {
        bad_request("function name must be a string", param);
    }
    std::string name = object.at("name").get<std::string>();
    if (!valid_tool_name(name, 64)) {
        bad_request("function name must match [A-Za-z0-9_-]{1,64}", param);
    }
    return name;
}

void validate_standard_output_controls(const Json& body) {
    if (body.contains("functions") && !body.at("functions").is_null()) {
        const Json& functions = body.at("functions");
        if (!functions.is_array()) { bad_request("functions must be an array", "functions"); }
        if (!functions.empty()) {
            bad_request(
                "non-empty legacy functions require the legacy prompt and single-function-call "
                "response contract, which NInfer does not expose; use tools instead",
                "functions", "legacy_tools_not_supported");
        }
    }
    if (body.contains("function_call") && !body.at("function_call").is_null()) {
        const Json& choice = body.at("function_call");
        if (choice.is_string()) {
            const std::string value = choice.get<std::string>();
            if (value != "none" && value != "auto") {
                bad_request("function_call must be 'none', 'auto', or a named function choice",
                            "function_call");
            }
        } else if (choice.is_object()) {
            bad_request(
                "a named legacy function_call requires forced tool invocation, which NInfer "
                "cannot guarantee",
                "function_call", "legacy_tools_not_supported");
        } else {
            bad_request("function_call must be 'none', 'auto', or an object", "function_call");
        }
    }

    if (body.contains("logit_bias") && !body.at("logit_bias").is_null()) {
        const Json& biases = body.at("logit_bias");
        if (!biases.is_object()) { bad_request("logit_bias must be an object", "logit_bias"); }
        for (auto iterator = biases.begin(); iterator != biases.end(); ++iterator) {
            if (!iterator.value().is_number()) {
                bad_request("logit_bias values must be numbers", "logit_bias");
            }
            const double value = iterator.value().get<double>();
            if (!std::isfinite(value) || value != 0.0) {
                bad_request(
                    "nonzero logit_bias requires per-token logit modification, which NInfer "
                    "does not provide",
                    "logit_bias", "logit_bias_not_supported");
            }
        }
    }
    if (body.contains("logprobs") && !body.at("logprobs").is_null()) {
        if (!body.at("logprobs").is_boolean()) {
            bad_request("logprobs must be a boolean", "logprobs");
        }
        if (body.at("logprobs").get<bool>()) {
            bad_request("logprobs=true requires per-token log probabilities in the response, which "
                        "NInfer does not provide",
                        "logprobs", "logprobs_not_supported");
        }
    }
    if (const std::optional<int> top_logprobs = optional_int(body, "top_logprobs")) {
        if (*top_logprobs != 0) {
            bad_request(
                "nonzero top_logprobs requires alternative-token probabilities in the response, "
                "which NInfer does not provide",
                "top_logprobs", "logprobs_not_supported");
        }
    }

    if (body.contains("response_format") && !body.at("response_format").is_null()) {
        const Json& format = body.at("response_format");
        if (!format.is_object() || !format.contains("type") || !format.at("type").is_string()) {
            bad_request("response_format must contain a string type", "response_format");
        }
        if (format.at("type").get<std::string>() != "text") {
            bad_request(
                "this response_format requires constrained output, which NInfer cannot guarantee; "
                "only {\"type\":\"text\"} is available",
                "response_format", "response_format_not_supported");
        }
    }

    if (body.contains("modalities") && !body.at("modalities").is_null()) {
        const Json& modalities = body.at("modalities");
        if (!modalities.is_array() || modalities.empty()) {
            bad_request("modalities must be a non-empty array", "modalities");
        }
        for (const Json& modality : modalities) {
            if (!modality.is_string()) {
                bad_request("modalities entries must be strings", "modalities");
            }
            if (modality.get<std::string>() != "text") {
                bad_request(
                    "the requested output modality requires non-text generation, while NInfer "
                    "produces text only",
                    "modalities", "modality_not_supported");
            }
        }
    }

    if (body.contains("web_search_options") && !body.at("web_search_options").is_null()) {
        bad_request(
            "web_search_options requests hosted web search and citations, which NInfer does not "
            "provide",
            "web_search_options", "web_search_not_supported");
    }
    if (body.contains("moderation") && !body.at("moderation").is_null()) {
        bad_request(
            "moderation requests input/output moderation behavior, which NInfer does not provide",
            "moderation", "moderation_not_supported");
    }
    if (body.contains("verbosity") && !body.at("verbosity").is_null()) {
        if (!body.at("verbosity").is_string()) {
            bad_request("verbosity must be 'low', 'medium', or 'high'", "verbosity");
        }
        const std::string value = body.at("verbosity").get<std::string>();
        if (value != "low" && value != "medium" && value != "high") {
            bad_request("verbosity must be 'low', 'medium', or 'high'", "verbosity");
        }
        if (value != "medium") {
            bad_request(
                "verbosity='" + value +
                    "' requires an output-length style constraint that NInfer cannot guarantee; "
                    "the default 'medium' value is accepted",
                "verbosity", "verbosity_not_supported");
        }
    }

    if (body.contains("store") && !body.at("store").is_null()) {
        if (!body.at("store").is_boolean()) { bad_request("store must be a boolean", "store"); }
        if (body.at("store").get<bool>()) {
            bad_request(
                "store=true requires a retrievable stored Chat Completion, which NInfer does not "
                "provide",
                "store", "store_not_supported");
        }
    }
}

void validate_constrained_decoding_extensions(const Json& body) {
    // llama.cpp exposes grammar; vLLM uses structured_outputs and previously exposed the
    // guided_* spellings. Each promises constrained generation rather than an advisory hint.
    static constexpr const char* fields[] = {
        "grammar",      "structured_outputs", "guided_json",
        "guided_regex", "guided_choice",      "guided_grammar",
    };
    for (const char* field : fields) {
        if (!body.contains(field) || body.at(field).is_null()) { continue; }
        const Json& value = body.at(field);
        if (std::string_view(field) == "grammar" && value.is_string() &&
            value.get_ref<const std::string&>().empty()) {
            continue;
        }
        bad_request(std::string(field) +
                        " requests constrained decoding, which NInfer does not provide",
                    field, "constrained_decoding_not_supported");
    }
}

void validate_compatibility_hints(const Json& body) {
    // vLLM exposes repetition_penalty, but NInfer's Engine intentionally has no such sampler.
    // The neutral value is accepted so common client defaults remain harmless.
    if (body.contains("repetition_penalty") && !body.at("repetition_penalty").is_null()) {
        if (!body.at("repetition_penalty").is_number()) {
            bad_request("repetition_penalty must be a number", "repetition_penalty");
        }
        const double value = body.at("repetition_penalty").get<double>();
        if (!std::isfinite(value) || value != 1.0) {
            bad_request(
                "a non-neutral repetition_penalty requires a sampler transform that NInfer does "
                "not provide; only repetition_penalty=1 is accepted",
                "repetition_penalty", "repetition_penalty_not_supported");
        }
    }

    // vLLM/SGLang expose processor-specific kwargs. They cannot be honored by NInfer's fixed
    // Vision frontend, so only omitted/null-valued overrides are semantically neutral.
    if (body.contains("mm_processor_kwargs") && !body.at("mm_processor_kwargs").is_null()) {
        if (!body.at("mm_processor_kwargs").is_object()) {
            bad_request("mm_processor_kwargs must be an object", "mm_processor_kwargs");
        }
        for (auto iterator = body.at("mm_processor_kwargs").begin();
             iterator != body.at("mm_processor_kwargs").end(); ++iterator) {
            if (!iterator.value().is_null()) {
                bad_request(
                    "mm_processor_kwargs contains a non-null preprocessing override that NInfer's "
                    "fixed Vision frontend cannot apply",
                    "mm_processor_kwargs", "mm_processor_kwargs_not_supported");
            }
        }
    }
}

ninfer::product::media_acquire::Source parse_media_url(const Json& part, const char* field,
                                                       bool image) {
    if (!part.contains(field)) {
        bad_request(std::string(field) + " content part must contain " + field, "messages");
    }
    const Json& value = part.at(field);
    std::string url;
    if (value.is_string()) {
        // vLLM/SGLang accept this shorthand in addition to OpenAI's object form.
        url = value.get<std::string>();
    } else if (value.is_object()) {
        if (!value.contains("url") || !value.at("url").is_string()) {
            bad_request(std::string(field) + " must contain a string url", "messages");
        }
        url = value.at("url").get<std::string>();
        if (image && value.contains("detail") && !value.at("detail").is_null()) {
            if (!value.at("detail").is_string()) {
                bad_request("image_url.detail must be a string", "messages");
            }
            const std::string detail = value.at("detail").get<std::string>();
            if (detail != "auto") {
                bad_request(
                    "image_url.detail='" + detail +
                        "' requests an explicit preprocessing profile that NInfer's fixed Vision "
                        "frontend cannot apply; use 'auto'",
                    "messages", "image_detail_not_supported");
            }
        }
    } else {
        bad_request(std::string(field) + " must be a URL string or object", "messages");
    }
    if (url.empty()) { bad_request(std::string(field) + " URL must not be empty", "messages"); }

    ninfer::product::media_acquire::Source source;
    source.value = std::move(url);
    if (source.value.starts_with("data:")) {
        source.kind = ninfer::product::media_acquire::SourceKind::Data;
    } else if (source.value.starts_with("http://") || source.value.starts_with("https://")) {
        source.kind = ninfer::product::media_acquire::SourceKind::Url;
    } else {
        bad_request(std::string(field) + " must use HTTP(S) or a data URI", "messages");
    }
    return source;
}

void parse_content_parts(const Json& content, ChatTurn& turn, std::size_t index) {
    if (content.is_string()) {
        turn.content.push_back(ContentPart{
            .kind = ContentKind::Text, .text = content.get<std::string>(), .type_raw = "text"});
        return;
    }
    if (!content.is_array()) {
        bad_request("message " + std::to_string(index) + " content must be a string or array",
                    "messages");
    }
    for (const Json& part : content) {
        if (!part.is_object() || !part.contains("type") || !part.at("type").is_string()) {
            bad_request("message content parts must contain a string type", "messages");
        }
        const std::string type = part.at("type").get<std::string>();
        ContentPart parsed;
        parsed.type_raw = type;
        if (type == "text") {
            if (!part.contains("text") || !part.at("text").is_string()) {
                bad_request("text content part must contain a string text", "messages");
            }
            parsed.kind = ContentKind::Text;
            parsed.text = part.at("text").get<std::string>();
        } else if (type == "refusal") {
            if (turn.role != ChatRole::Assistant) {
                bad_request("refusal content is only valid on assistant messages", "messages");
            }
            if (!part.contains("refusal") || !part.at("refusal").is_string()) {
                bad_request("refusal content part must contain a string refusal", "messages");
            }
            parsed.kind = ContentKind::Text;
            parsed.text = part.at("refusal").get<std::string>();
        } else if (type == "image_url") {
            // VS Code Copilot-compatible screenshot results use image_url parts on tool turns.
            if (turn.role != ChatRole::User && turn.role != ChatRole::Tool) {
                bad_request("image_url is only supported on user or tool messages", "messages",
                            "modality_not_supported");
            }
            parsed.kind   = ContentKind::Image;
            parsed.source = parse_media_url(part, "image_url", true);
        } else if (type == "video_url") {
            // Qwen, vLLM, and SGLang use video_url as a Chat Completions extension for
            // multimodal models. NInfer maps it to the Engine's native Video input.
            if (turn.role != ChatRole::User) {
                bad_request("video_url is only supported on user messages", "messages",
                            "modality_not_supported");
            }
            parsed.kind   = ContentKind::Video;
            parsed.source = parse_media_url(part, "video_url", false);
        } else {
            bad_request("content type '" + type + "' is not supported", "messages",
                        "modality_not_supported");
        }
        if (parse_openai_prompt_cache_breakpoint(part, "messages")) {
            parsed.cache_boundary_after = CacheBoundary{};
        }
        turn.content.push_back(std::move(parsed));
    }
}

std::vector<ToolCall> parse_assistant_tool_calls(const Json& message, std::size_t index) {
    std::vector<ToolCall> calls;
    if (!message.contains("tool_calls") || message.at("tool_calls").is_null()) { return calls; }
    const Json& values = message.at("tool_calls");
    if (!values.is_array()) {
        bad_request("assistant message " + std::to_string(index) + " tool_calls must be an array",
                    "messages");
    }
    calls.reserve(values.size());
    for (const Json& value : values) {
        if (!value.is_object() || !value.contains("id") || !value.at("id").is_string()) {
            bad_request("tool_calls entries must contain a string id", "messages");
        }
        if (!value.contains("type") || !value.at("type").is_string() ||
            value.at("type").get<std::string>() != "function") {
            bad_request("only function tool_calls are supported", "messages",
                        "tool_type_not_supported");
        }
        if (!value.contains("function") || !value.at("function").is_object()) {
            bad_request("tool_calls entries must contain a function object", "messages");
        }
        const Json& function = value.at("function");
        if (!function.contains("arguments") || !function.at("arguments").is_string()) {
            bad_request("function tool_calls must contain string arguments", "messages");
        }
        calls.push_back(ToolCall{.id             = value.at("id").get<std::string>(),
                                 .name           = require_function_name(function, "messages"),
                                 .arguments_json = function.at("arguments").get<std::string>()});
    }
    return calls;
}

std::optional<ToolCall> parse_legacy_assistant_function_call(const Json& message,
                                                             std::size_t index) {
    if (!message.contains("function_call") || message.at("function_call").is_null()) {
        return std::nullopt;
    }
    const Json& call = message.at("function_call");
    if (!call.is_object()) {
        bad_request("assistant message " + std::to_string(index) +
                        " function_call must be an object",
                    "messages");
    }
    if (!call.contains("arguments") || !call.at("arguments").is_string()) {
        bad_request("assistant function_call must contain string arguments", "messages");
    }
    return ToolCall{.id             = {},
                    .name           = require_function_name(call, "messages"),
                    .arguments_json = call.at("arguments").get<std::string>()};
}

std::optional<std::string> parse_assistant_reasoning(const Json& message, std::size_t index) {
    auto parse = [&](const char* key) -> std::optional<std::string> {
        if (!message.contains(key) || message.at(key).is_null()) { return std::nullopt; }
        if (!message.at(key).is_string()) {
            bad_request("assistant message " + std::to_string(index) + " " + key +
                            " must be a string",
                        "messages");
        }
        return message.at(key).get<std::string>();
    };
    // vLLM historically used reasoning_content and now also accepts reasoning. They carry the
    // same assistant-history semantics, so equal aliases normalize to one Engine field.
    std::optional<std::string> content   = parse("reasoning_content");
    std::optional<std::string> reasoning = parse("reasoning");
    if (content && content->empty()) { content.reset(); }
    if (reasoning && reasoning->empty()) { reasoning.reset(); }
    if (content && reasoning && *content != *reasoning) {
        bad_request("conflicting assistant reasoning and reasoning_content values", "messages",
                    "conflicting_template_option");
    }
    return reasoning ? reasoning : content;
}

void validate_message_name(const Json& item, ChatRole role) {
    if (!item.contains("name") || item.at("name").is_null()) { return; }
    if (!item.at("name").is_string()) { bad_request("message name must be a string", "messages"); }
    const std::string& name = item.at("name").get_ref<const std::string&>();
    // Some OpenAI-compatible clients mirror the function name onto role=tool messages. Accept
    // that non-standard field as an ignored compatibility hint; it never reaches the Engine or
    // prompt renderer.
    if (!name.empty() && role != ChatRole::Tool) {
        bad_request("a non-empty message name changes participant identity, which NInfer's chat "
                    "template cannot represent",
                    "messages", "message_name_not_supported");
    }
}

void validate_non_assistant_fields(const Json& item, ChatRole role) {
    if (role != ChatRole::Assistant && item.contains("function_call") &&
        !item.at("function_call").is_null()) {
        bad_request("function_call is only valid on assistant messages", "messages");
    }
    if (role == ChatRole::Assistant) { return; }
    for (const char* key : {"reasoning", "reasoning_content"}) {
        if (!item.contains(key) || item.at(key).is_null()) { continue; }
        if (!item.at(key).is_string()) {
            bad_request(std::string(key) + " must be a string", "messages");
        }
        if (!item.at(key).get<std::string>().empty()) {
            bad_request(std::string(key) + " is only valid on assistant messages", "messages");
        }
    }
}

void validate_non_tool_call_id(const Json& item) {
    if (!item.contains("tool_call_id") || item.at("tool_call_id").is_null()) { return; }
    if (!item.at("tool_call_id").is_string()) {
        bad_request("tool_call_id must be a string", "messages");
    }
    if (!item.at("tool_call_id").get<std::string>().empty()) {
        bad_request("a non-empty tool_call_id is only valid on tool messages", "messages");
    }
}

ChatTurn parse_tool_message(const Json& item, std::size_t index, bool legacy_function) {
    if (item.contains("tool_calls") && !item.at("tool_calls").is_null()) {
        if (!item.at("tool_calls").is_array()) {
            bad_request("tool_calls must be an array", "messages");
        }
        if (!item.at("tool_calls").empty()) {
            bad_request("tool messages cannot contain tool_calls", "messages");
        }
    }
    if (!legacy_function &&
        (!item.contains("tool_call_id") || !item.at("tool_call_id").is_string())) {
        bad_request("tool messages must contain a string tool_call_id", "messages");
    }
    if (item.contains("tool_call_id") && !item.at("tool_call_id").is_null() &&
        !item.at("tool_call_id").is_string()) {
        bad_request("tool_call_id must be a string", "messages");
    }
    if (!item.contains("content") || item.at("content").is_null()) {
        bad_request("tool messages must contain content", "messages");
    }

    ChatTurn turn;
    turn.role = ChatRole::Tool;
    if (item.contains("tool_call_id") && item.at("tool_call_id").is_string()) {
        turn.tool_call_id = item.at("tool_call_id").get<std::string>();
    }
    parse_content_parts(item.at("content"), turn, index);
    return turn;
}

ChatTurn parse_assistant_message(const Json& item, std::size_t index) {
    if (item.contains("audio") && !item.at("audio").is_null()) {
        bad_request("assistant audio history requires resolving a previous audio response, which "
                    "NInfer cannot provide",
                    "messages", "assistant_history_not_supported");
    }

    ChatTurn turn;
    turn.role       = ChatRole::Assistant;
    turn.tool_calls = parse_assistant_tool_calls(item, index);
    if (std::optional<ToolCall> legacy_call = parse_legacy_assistant_function_call(item, index)) {
        turn.tool_calls.insert(turn.tool_calls.begin(), std::move(*legacy_call));
    }
    const std::optional<std::string> reasoning = parse_assistant_reasoning(item, index);
    if (item.contains("content") && !item.at("content").is_null()) {
        parse_content_parts(item.at("content"), turn, index);
    }
    if (item.contains("refusal") && !item.at("refusal").is_null()) {
        if (!item.at("refusal").is_string()) {
            bad_request("assistant refusal must be a string", "messages");
        }
        std::string refusal = item.at("refusal").get<std::string>();
        if (!refusal.empty()) {
            turn.content.push_back(ContentPart{
                .kind = ContentKind::Text, .text = std::move(refusal), .type_raw = "refusal"});
        }
    }
    if (reasoning) { turn.reasoning_content = *reasoning; }
    return turn;
}

ChatTurn parse_regular_message(const Json& item, std::size_t index, ChatRole role) {
    if (item.contains("tool_calls") && !item.at("tool_calls").is_null()) {
        if (!item.at("tool_calls").is_array()) {
            bad_request("tool_calls must be an array", "messages");
        }
        if (!item.at("tool_calls").empty()) {
            bad_request("non-empty tool_calls are only valid on assistant messages", "messages");
        }
    }
    if (!item.contains("content") || item.at("content").is_null()) {
        bad_request("message " + std::to_string(index) + " must have content", "messages");
    }

    ChatTurn turn;
    turn.role = role;
    parse_content_parts(item.at("content"), turn, index);
    return turn;
}

ChatTurn parse_message(const Json& item, std::size_t index) {
    if (!item.is_object() || !item.contains("role") || !item.at("role").is_string()) {
        bad_request("message " + std::to_string(index) + " must be an object with a string role",
                    "messages");
    }
    const std::string role_name = item.at("role").get<std::string>();
    const bool legacy_function  = role_name == "function";
    const ChatRole role         = legacy_function ? ChatRole::Tool : parse_message_role(role_name);

    validate_message_name(item, role);
    if (legacy_function) { (void)require_function_name(item, "messages"); }
    validate_non_assistant_fields(item, role);

    if (role == ChatRole::Tool) { return parse_tool_message(item, index, legacy_function); }
    validate_non_tool_call_id(item);
    if (role == ChatRole::Assistant) { return parse_assistant_message(item, index); }
    return parse_regular_message(item, index, role);
}

void parse_messages(const Json& body, GenerationRequest& output) {
    if (!body.contains("messages")) { bad_request("missing required field: messages", "messages"); }
    const Json& messages = body.at("messages");
    if (!messages.is_array() || messages.empty()) {
        bad_request("messages must be a non-empty array", "messages");
    }
    output.messages.reserve(messages.size());
    for (std::size_t index = 0; index < messages.size(); ++index) {
        output.messages.push_back(parse_message(messages.at(index), index));
    }
}

void parse_tools(const Json& body, GenerationRequest& output) {
    if (!body.contains("tools") || body.at("tools").is_null()) { return; }
    const Json& tools = body.at("tools");
    if (!tools.is_array()) { bad_request("tools must be an array", "tools"); }
    output.tools.reserve(tools.size());
    for (const Json& item : tools) {
        if (!item.is_object() || !item.contains("type") || !item.at("type").is_string()) {
            bad_request("tools entries must contain a string type", "tools");
        }
        const std::string type = item.at("type").get<std::string>();
        if (type != "function") {
            bad_request(
                "tool type '" + type +
                    "' requires a non-function output contract that NInfer does not provide",
                "tools", "tool_type_not_supported");
        }
        if (!item.contains("function") || !item.at("function").is_object()) {
            bad_request("function tools must contain a function object", "tools");
        }
        const Json& function = item.at("function");
        ToolDefinition tool;
        tool.name = require_function_name(function, "tools");
        if (function.contains("description") && !function.at("description").is_null()) {
            if (!function.at("description").is_string()) {
                bad_request("function description must be a string", "tools");
            }
            tool.description = function.at("description").get<std::string>();
        }
        if (!function.contains("parameters") || function.at("parameters").is_null()) {
            tool.input_schema_json =
                Json{{"type", "object"}, {"properties", Json::object()}}.dump();
        } else if (!function.at("parameters").is_object()) {
            bad_request("function parameters must be a JSON object", "tools");
        } else {
            tool.input_schema_json = function.at("parameters").dump();
        }
        if (function.contains("strict") && !function.at("strict").is_null()) {
            if (!function.at("strict").is_boolean()) {
                bad_request("function strict must be a boolean", "tools");
            }
            if (function.at("strict").get<bool>()) {
                bad_request(
                    "strict=true requires generated function arguments to satisfy the declared "
                    "JSON Schema, which NInfer cannot guarantee",
                    "tools", "strict_tools_not_supported");
            }
        }
        output.tools.push_back(std::move(tool));
    }
}

void apply_allowed_tools(const Json& config, GenerationRequest& output) {
    if (!config.is_object()) {
        bad_request("tool_choice.allowed_tools must be an object", "tool_choice");
    }
    if (!config.contains("mode") || !config.at("mode").is_string()) {
        bad_request("tool_choice.allowed_tools.mode must be 'auto' or 'required'", "tool_choice");
    }
    const std::string mode = config.at("mode").get<std::string>();
    if (mode != "auto" && mode != "required") {
        bad_request("tool_choice.allowed_tools.mode must be 'auto' or 'required'", "tool_choice");
    }
    if (!config.contains("tools") || !config.at("tools").is_array()) {
        bad_request("tool_choice.allowed_tools.tools must be an array", "tool_choice");
    }

    std::vector<std::string> allowed_names;
    allowed_names.reserve(config.at("tools").size());
    for (const Json& item : config.at("tools")) {
        if (!item.is_object() || !item.contains("type") || !item.at("type").is_string()) {
            bad_request("allowed tool entries must contain a string type", "tool_choice");
        }
        if (item.at("type").get<std::string>() != "function") {
            bad_request(
                "allowed_tools can select only function tools because NInfer does not provide "
                "custom tool output",
                "tool_choice", "tool_type_not_supported");
        }
        const std::string name = require_function_name(item, "tool_choice");
        const bool declared =
            std::any_of(output.tools.begin(), output.tools.end(),
                        [&](const ToolDefinition& tool) { return tool.name == name; });
        if (!declared) {
            bad_request("allowed tool '" + name + "' is not present in tools", "tool_choice");
        }
        if (std::find(allowed_names.begin(), allowed_names.end(), name) == allowed_names.end()) {
            allowed_names.push_back(name);
        }
    }

    if (mode == "required") {
        bad_request(
            "tool_choice.allowed_tools mode='required' requires at least one tool call, which "
            "NInfer cannot guarantee",
            "tool_choice", "tool_choice_not_supported");
    }

    std::erase_if(output.tools, [&](const ToolDefinition& tool) {
        return std::find(allowed_names.begin(), allowed_names.end(), tool.name) ==
               allowed_names.end();
    });
    output.tool_choice.mode = ToolChoiceMode::Auto;
}

void parse_tool_choice(const Json& body, GenerationRequest& output) {
    if (!body.contains("tool_choice") || body.at("tool_choice").is_null()) { return; }
    const Json& choice = body.at("tool_choice");
    if (choice.is_string()) {
        const std::string value = choice.get<std::string>();
        if (value == "auto") {
            output.tool_choice.mode = ToolChoiceMode::Auto;
        } else if (value == "none") {
            output.tool_choice.mode = ToolChoiceMode::None;
        } else if (value == "required") {
            bad_request(
                "tool_choice='required' requires at least one tool call, which NInfer cannot "
                "guarantee",
                "tool_choice", "tool_choice_not_supported");
        } else {
            bad_request("tool_choice must be 'auto', 'none', 'required', or a function choice",
                        "tool_choice");
        }
    } else if (choice.is_object()) {
        if (!choice.contains("type") || !choice.at("type").is_string()) {
            bad_request("tool_choice objects must contain a string type", "tool_choice");
        }
        const std::string type = choice.at("type").get<std::string>();
        if (type == "allowed_tools") {
            apply_allowed_tools(
                choice.contains("allowed_tools") ? choice.at("allowed_tools") : choice, output);
        } else if (type == "function") {
            if (!choice.contains("function") || !choice.at("function").is_object()) {
                bad_request("function tool_choice must contain a function object", "tool_choice");
            }
            const std::string name = require_function_name(choice.at("function"), "tool_choice");
            bad_request(
                "tool_choice for function '" + name +
                    "' requires that exact function to be called, which NInfer cannot guarantee",
                "tool_choice", "tool_choice_not_supported");
        } else if (type == "custom") {
            bad_request(
                "custom tool_choice requires custom tool output, which NInfer does not provide",
                "tool_choice", "tool_type_not_supported");
        } else {
            bad_request("unsupported tool_choice type: " + type, "tool_choice");
        }
    } else {
        bad_request("tool_choice must be a string or object", "tool_choice");
    }
}

void parse_parallel_tool_calls(const Json& body, const GenerationRequest& output) {
    if (!body.contains("parallel_tool_calls") || body.at("parallel_tool_calls").is_null()) {
        return;
    }
    if (!body.at("parallel_tool_calls").is_boolean()) {
        bad_request("parallel_tool_calls must be a boolean", "parallel_tool_calls");
    }
    if (!body.at("parallel_tool_calls").get<bool>() && output.uses_tools()) {
        bad_request(
            "parallel_tool_calls=false requires the model to emit at most one tool call, which "
            "NInfer cannot guarantee while tools are enabled",
            "parallel_tool_calls", "parallel_tool_calls_not_supported");
    }
}

void parse_stop(const Json& body, GenerationRequest& output) {
    if (!body.contains("stop") || body.at("stop").is_null()) { return; }
    output.stop_strings_apply_to_reasoning = true;
    const Json& stop                       = body.at("stop");
    auto append                            = [&](const Json& value) {
        if (!value.is_string()) { bad_request("stop entries must be strings", "stop"); }
        std::string text = value.get<std::string>();
        if (text.empty()) { bad_request("stop strings must not be empty", "stop"); }
        output.stop_strings.push_back(std::move(text));
    };
    if (stop.is_string()) {
        append(stop);
    } else if (stop.is_array()) {
        if (stop.size() > 4) { bad_request("stop supports at most four strings", "stop"); }
        for (const Json& value : stop) { append(value); }
    } else {
        bad_request("stop must be a string or array of strings", "stop");
    }
}

void parse_sampling(const Json& body, GenerationRequest& output) {
    SamplingParams& sampling   = output.sampling;
    sampling.temperature       = get_number(body, "temperature");
    sampling.top_p             = get_number(body, "top_p");
    sampling.presence_penalty  = get_number(body, "presence_penalty");
    sampling.frequency_penalty = get_number(body, "frequency_penalty");
    sampling.seed              = get_seed(body);

    // vLLM and SGLang expose top_k/min_p on their OpenAI-compatible endpoints; both map directly
    // to NInfer's native sampler and are useful for Qwen's published sampling presets.
    sampling.top_k = optional_int(body, "top_k");
    sampling.min_p = get_number(body, "min_p");

    if (const std::optional<int> count = optional_int(body, "n")) {
        if (*count != 1) {
            bad_request("n requests multiple completions, while NInfer produces one completion per "
                        "request; only n=1 is supported",
                        "n", "n_not_supported");
        }
    }
}

struct TemplateOptions {
    std::optional<bool> enable_thinking;
    std::optional<bool> preserve_thinking;
};

TemplateOptions parse_template_options(const Json& body) {
    // Qwen's deployment examples use both top-level aliases and
    // chat_template_kwargs. vLLM/SGLang expose the kwargs form. They are normalized here so the
    // Engine sees one value and conflicting aliases fail before execution.
    TemplateOptions output;
    output.enable_thinking   = get_optional_bool(body, "enable_thinking");
    output.preserve_thinking = get_optional_bool(body, "preserve_thinking");

    if (!body.contains("chat_template_kwargs") || body.at("chat_template_kwargs").is_null()) {
        return output;
    }
    const Json& kwargs = body.at("chat_template_kwargs");
    if (!kwargs.is_object()) {
        bad_request("chat_template_kwargs must be an object", "chat_template_kwargs");
    }
    for (auto iterator = kwargs.begin(); iterator != kwargs.end(); ++iterator) {
        if (iterator.key() != "enable_thinking" && iterator.key() != "preserve_thinking" &&
            !iterator.value().is_null()) {
            bad_request("chat_template_kwargs." + iterator.key() + " is not supported",
                        "chat_template_kwargs", "chat_template_option_not_supported");
        }
    }
    auto merge = [&](const char* key, std::optional<bool>& top_level) {
        const std::optional<bool> nested = get_optional_bool(kwargs, key);
        if (top_level && nested && *top_level != *nested) {
            bad_request(std::string("conflicting ") + key + " values", key,
                        "conflicting_template_option");
        }
        if (nested) { top_level = nested; }
    };
    merge("enable_thinking", output.enable_thinking);
    merge("preserve_thinking", output.preserve_thinking);
    return output;
}

void parse_reasoning_effort(const Json& body, GenerationRequest& output) {
    if (!body.contains("reasoning_effort") || body.at("reasoning_effort").is_null()) { return; }
    if (!body.at("reasoning_effort").is_string()) {
        bad_request("reasoning_effort must be a string or null", "reasoning_effort");
    }
    const std::string value = body.at("reasoning_effort").get<std::string>();
    const std::optional<RequestedReasoningEffort> parsed = parse_requested_reasoning_effort(value);
    if (!parsed) {
        bad_request("reasoning_effort must be one of none, minimal, low, medium, high, xhigh, or "
                    "max",
                    "reasoning_effort");
    }
    output.reasoning_effort = *parsed;
}

void parse_stream_options(const Json& body, OpenAIChatRequest& output) {
    output.stream = get_bool(body, "stream", false);
    if (!body.contains("stream_options") || body.at("stream_options").is_null()) { return; }
    const Json& options = body.at("stream_options");
    if (!options.is_object()) { bad_request("stream_options must be an object", "stream_options"); }
    output.include_usage = get_bool(options, "include_usage", false);
    if (options.contains("include_obfuscation") && !options.at("include_obfuscation").is_null() &&
        !options.at("include_obfuscation").is_boolean()) {
        bad_request("include_obfuscation must be a boolean", "stream_options");
    }
}

void parse_output_limit(const Json& body, const RequestLimits& limits, OpenAIChatRequest& output) {
    std::optional<int> limit = optional_int(body, "max_completion_tokens");
    const char* param        = "max_completion_tokens";
    if (!limit) {
        limit = optional_int(body, "max_tokens");
        param = "max_tokens";
    }
    if (limit) {
        if (*limit < 0) { bad_request(std::string(param) + " must be nonnegative", param); }
        output.generation.max_tokens  = *limit;
        output.output_tokens_explicit = true;
    } else {
        output.generation.max_tokens = limits.default_max_tokens;
    }
}

} // namespace

OpenAIChatRequest parse_chat_completion_request(const Json& body, const RequestLimits& limits) {
    require_object(body, "request body must be a JSON object");
    validate_standard_output_controls(body);
    validate_constrained_decoding_extensions(body);
    validate_compatibility_hints(body);

    OpenAIChatRequest output;
    if (!body.contains("model") || !body.at("model").is_string() ||
        body.at("model").get<std::string>().empty()) {
        bad_request("missing required field: model", "model");
    }
    output.model = body.at("model").get<std::string>();

    const OpenAIPromptCachePolicy cache_policy = parse_openai_prompt_cache_policy(body);

    parse_tools(body, output.generation);
    parse_tool_choice(body, output.generation);
    parse_parallel_tool_calls(body, output.generation);
    parse_messages(body, output.generation);
    parse_stop(body, output.generation);
    parse_sampling(body, output.generation);
    parse_stream_options(body, output);
    parse_output_limit(body, limits, output);
    parse_reasoning_effort(body, output.generation);
    const TemplateOptions template_options = parse_template_options(body);
    output.generation.enable_thinking      = template_options.enable_thinking;
    output.generation.preserve_thinking    = template_options.preserve_thinking;
    apply_openai_prompt_cache_policy(output.generation, cache_policy);
    return output;
}

} // namespace ninfer::serve
