#include "qus/serve/openai_schema.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <random>
#include <string>

namespace qus::serve {
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
    return obj.at(key).get<int>();
}

void parse_content_parts(const Json& content, ChatTurn& turn, std::size_t index) {
    if (content.is_string()) {
        turn.content.push_back(ContentPart{ContentKind::Text, content.get<std::string>(), "text"});
        return;
    }
    if (!content.is_array()) {
        bad_request("message " + std::to_string(index) + " content must be a string or array",
                    "messages");
    }
    for (const Json& part : content) {
        if (!part.is_object() || !part.contains("type") || !part.at("type").is_string()) {
            bad_request("message " + std::to_string(index) +
                            " content parts must be objects with a string 'type'",
                        "messages");
        }
        const std::string type = part.at("type").get<std::string>();
        ContentPart out;
        out.type_raw = type;
        if (type == "text") {
            if (!part.contains("text") || !part.at("text").is_string()) {
                bad_request("text content part must contain a string 'text'", "messages");
            }
            out.kind = ContentKind::Text;
            out.text = part.at("text").get<std::string>();
        } else if (type == "image_url") {
            out.kind = ContentKind::ImageUrl;
        } else if (type == "input_audio") {
            out.kind = ContentKind::InputAudio;
        } else {
            out.kind = ContentKind::Unsupported;
        }
        turn.content.push_back(std::move(out));
    }
    if (turn.content.empty()) {
        bad_request("message " + std::to_string(index) + " content must not be empty", "messages");
    }
}

void parse_messages(const Json& body, GenerationRequest& out) {
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
        if (item.contains("tool_calls") || item.contains("tool_call_id") ||
            item.contains("function_call")) {
            ApiError error;
            error.message = "tool/function messages are not supported yet";
            error.param   = "messages";
            error.code    = "tools_not_supported";
            throw ApiException(std::move(error));
        }
        if (!item.contains("role") || !item.at("role").is_string()) {
            bad_request("message " + std::to_string(i) + " must have a string role", "messages");
        }
        const std::string role = item.at("role").get<std::string>();
        if (role == "tool" || role == "function") {
            ApiError error;
            error.message = "role '" + role + "' is not supported yet";
            error.param   = "messages";
            error.code    = "unsupported_role";
            throw ApiException(std::move(error));
        }
        if (!item.contains("content") || item.at("content").is_null()) {
            bad_request("message " + std::to_string(i) + " must have content", "messages");
        }
        ChatTurn turn;
        turn.role = role;
        parse_content_parts(item.at("content"), turn, i);
        out.messages.push_back(std::move(turn));
    }
}

void parse_stop(const Json& body, GenerationRequest& out) {
    if (!body.contains("stop") || body.at("stop").is_null()) { return; }
    const Json& stop = body.at("stop");
    if (stop.is_string()) {
        if (!stop.get<std::string>().empty()) { out.stop_strings.push_back(stop.get<std::string>()); }
        return;
    }
    if (stop.is_array()) {
        for (const Json& s : stop) {
            if (!s.is_string()) { bad_request("stop entries must be strings", "stop"); }
            if (!s.get<std::string>().empty()) { out.stop_strings.push_back(s.get<std::string>()); }
        }
        return;
    }
    bad_request("stop must be a string or array of strings", "stop");
}

void parse_sampling(const Json& body, GenerationRequest& out) {
    SamplingParams& s = out.sampling;
    s.temperature       = get_number(body, "temperature");
    s.top_p             = get_number(body, "top_p");
    s.top_k             = get_int(body, "top_k");
    s.presence_penalty  = get_number(body, "presence_penalty");
    s.frequency_penalty = get_number(body, "frequency_penalty");
    if (const std::optional<int> seed = get_int(body, "seed")) {
        s.seed = static_cast<std::uint64_t>(*seed);
    }
    if (body.contains("logit_bias") && !body.at("logit_bias").is_null()) {
        const Json& bias = body.at("logit_bias");
        if (!bias.is_object()) { bad_request("logit_bias must be an object", "logit_bias"); }
        for (auto it = bias.begin(); it != bias.end(); ++it) {
            if (!it.value().is_number()) {
                bad_request("logit_bias values must be numbers", "logit_bias");
            }
            try {
                s.logit_bias.emplace(std::stoi(it.key()), it.value().get<double>());
            } catch (const std::exception&) {
                bad_request("logit_bias keys must be integer token ids", "logit_bias");
            }
        }
    }
    if (const std::optional<int> n = get_int(body, "n")) {
        s.n = *n;
        if (s.n != 1) {
            ApiError error;
            error.message = "n>1 is not supported yet";
            error.param   = "n";
            error.code    = "n_not_supported";
            throw ApiException(std::move(error));
        }
    }
}

void reject_unsupported_features(const Json& body) {
    for (const char* key : {"tools", "functions", "tool_choice", "function_call"}) {
        if (body.contains(key) && !body.at(key).is_null()) {
            ApiError error;
            error.message = std::string(key) + " is not supported yet";
            error.param   = key;
            error.code    = "tools_not_supported";
            throw ApiException(std::move(error));
        }
    }
    if (body.contains("response_format") && !body.at("response_format").is_null()) {
        const Json& fmt = body.at("response_format");
        std::string type = fmt.is_object() && fmt.contains("type") && fmt.at("type").is_string()
                               ? fmt.at("type").get<std::string>()
                               : std::string();
        if (type != "text") {
            ApiError error;
            error.message = "only response_format {type:text} is supported";
            error.param   = "response_format";
            error.code    = "response_format_not_supported";
            throw ApiException(std::move(error));
        }
    }
}

Json base_chunk(const std::string& id, const std::string& model, std::int64_t created) {
    return Json{{"id", id},
                {"object", "chat.completion.chunk"},
                {"created", created},
                {"model", model}};
}

std::string sse_event(const Json& payload) { return "data: " + payload.dump() + "\n\n"; }

} // namespace

GenerationRequest parse_chat_completion_request(const Json& body, const RequestLimits& limits) {
    require_object(body);
    reject_unsupported_features(body);

    GenerationRequest out;
    if (!body.contains("model") || !body.at("model").is_string() ||
        body.at("model").get<std::string>().empty()) {
        bad_request("missing required field: model", "model");
    }
    out.model = body.at("model").get<std::string>();

    parse_messages(body, out);
    parse_stop(body, out);
    parse_sampling(body, out);

    out.stream = get_bool(body, "stream", false);
    if (body.contains("stream_options") && body.at("stream_options").is_object()) {
        out.include_usage = get_bool(body.at("stream_options"), "include_usage", false);
    }
    if (body.contains("enable_thinking") && !body.at("enable_thinking").is_null()) {
        out.enable_thinking = get_bool(body, "enable_thinking", false);
    }

    std::optional<int> max_tokens = get_int(body, "max_completion_tokens");
    if (!max_tokens) { max_tokens = get_int(body, "max_tokens"); }
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

std::string make_chat_completion_response(const std::string& id, const std::string& model,
                                          std::int64_t created, const std::string& content,
                                          const char* finish_reason,
                                          const CompletionUsage& usage) {
    const Json payload = {
        {"id", id},
        {"object", "chat.completion"},
        {"created", created},
        {"model", model},
        {"choices",
         Json::array({Json{{"index", 0},
                           {"message", Json{{"role", "assistant"}, {"content", content}}},
                           {"finish_reason", finish_reason}}})},
        {"usage",
         Json{{"prompt_tokens", usage.prompt_tokens},
              {"completion_tokens", usage.completion_tokens},
              {"total_tokens", usage.prompt_tokens + usage.completion_tokens}}}};
    return payload.dump();
}

std::string make_chat_chunk_role(const std::string& id, const std::string& model,
                                 std::int64_t created) {
    Json payload            = base_chunk(id, model, created);
    payload["choices"]      = Json::array({Json{{"index", 0},
                                                {"delta", Json{{"role", "assistant"}, {"content", ""}}},
                                                {"finish_reason", nullptr}}});
    return sse_event(payload);
}

std::string make_chat_chunk_content(const std::string& id, const std::string& model,
                                     std::int64_t created, const std::string& delta_text) {
    Json payload       = base_chunk(id, model, created);
    payload["choices"] = Json::array({Json{{"index", 0},
                                           {"delta", Json{{"content", delta_text}}},
                                           {"finish_reason", nullptr}}});
    return sse_event(payload);
}

std::string make_chat_chunk_final(const std::string& id, const std::string& model,
                                   std::int64_t created, const char* finish_reason,
                                   const CompletionUsage* usage) {
    Json payload       = base_chunk(id, model, created);
    payload["choices"] = Json::array(
        {Json{{"index", 0}, {"delta", Json::object()}, {"finish_reason", finish_reason}}});
    if (usage != nullptr) {
        payload["usage"] = Json{{"prompt_tokens", usage->prompt_tokens},
                                {"completion_tokens", usage->completion_tokens},
                                {"total_tokens", usage->prompt_tokens + usage->completion_tokens}};
    }
    return sse_event(payload);
}

std::string sse_done() { return "data: [DONE]\n\n"; }

std::string make_models_list(const std::string& model_id, std::int64_t created) {
    const Json payload = {
        {"object", "list"},
        {"data", Json::array({Json{{"id", model_id},
                                   {"object", "model"},
                                   {"created", created},
                                   {"owned_by", "qus"}}})}};
    return payload.dump();
}

std::string make_model_object(const std::string& model_id, std::int64_t created) {
    const Json payload = {{"id", model_id},
                          {"object", "model"},
                          {"created", created},
                          {"owned_by", "qus"}};
    return payload.dump();
}

std::string make_error_body(const ApiError& error) {
    Json err = {{"message", error.message}, {"type", error.type}};
    err["param"] = error.param.empty() ? Json(nullptr) : Json(error.param);
    err["code"]  = error.code.empty() ? Json(nullptr) : Json(error.code);
    return Json{{"error", err}}.dump();
}

std::string new_chat_completion_id() {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<std::uint64_t> dist;
    std::array<char, 32> buf{};
    std::snprintf(buf.data(), buf.size(), "%016llx",
                  static_cast<unsigned long long>(dist(rng)));
    return "chatcmpl-" + std::string(buf.data());
}

std::int64_t unix_time_now() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

} // namespace qus::serve
