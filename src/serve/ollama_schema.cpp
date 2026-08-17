#include "serve/ollama_schema.h"

#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

constexpr std::size_t kMaxToolNameLength = 64;

[[noreturn]] void bad_request(std::string message, std::string param = {}, std::string code = {}) {
    ApiError error;
    error.status  = 400;
    error.type    = "invalid_request_error";
    error.message = std::move(message);
    error.param   = std::move(param);
    error.code    = std::move(code);
    throw ApiException(std::move(error));
}

std::optional<double> get_number(const Json& obj, const char* key, const char* param) {
    if (!obj.contains(key) || obj.at(key).is_null()) { return std::nullopt; }
    if (!obj.at(key).is_number()) { bad_request(std::string(param) + " must be a number", param); }
    return obj.at(key).get<double>();
}

std::optional<int> get_int(const Json& obj, const char* key, const char* param) {
    if (!obj.contains(key) || obj.at(key).is_null()) { return std::nullopt; }
    const Json& value = obj.at(key);
    if (value.is_number_integer()) {
        const std::int64_t v = value.get<std::int64_t>();
        if (v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max()) {
            bad_request(std::string(param) + " is out of range", param);
        }
        return static_cast<int>(v);
    }
    if (value.is_number_float()) {
        // Ollama options are typed loosely by clients; accept integral floats.
        const double v = value.get<double>();
        if (std::floor(v) != v || v < std::numeric_limits<int>::min() ||
            v > std::numeric_limits<int>::max()) {
            bad_request(std::string(param) + " must be an integer", param);
        }
        return static_cast<int>(v);
    }
    bad_request(std::string(param) + " must be an integer", param);
}

std::optional<std::uint64_t> get_u64(const Json& obj, const char* key, const char* param) {
    if (!obj.contains(key) || obj.at(key).is_null()) { return std::nullopt; }
    const Json& value = obj.at(key);
    if (!value.is_number_integer()) {
        bad_request(std::string(param) + " must be a nonnegative integer", param);
    }
    if (value.is_number_unsigned()) { return value.get<std::uint64_t>(); }
    const std::int64_t v = value.get<std::int64_t>();
    if (v < 0) { bad_request(std::string(param) + " must be nonnegative", param); }
    return static_cast<std::uint64_t>(v);
}

bool is_valid_function_name(const std::string& name) {
    if (name.empty() || name.size() > kMaxToolNameLength) { return false; }
    for (const unsigned char c : name) {
        if (std::isalnum(c) == 0 && c != '_' && c != '-') { return false; }
    }
    return true;
}

std::string require_function_name(const Json& obj, const char* param) {
    if (!obj.contains("name") || !obj.at("name").is_string()) {
        bad_request("function name must be a string", param);
    }
    std::string name = obj.at("name").get<std::string>();
    if (!is_valid_function_name(name)) {
        bad_request("function name must match [A-Za-z0-9_-]{1,64}", param);
    }
    return name;
}

// --- request parsing --------------------------------------------------------

void parse_tools(const Json& body, GenerationRequest& out) {
    if (!body.contains("tools") || body.at("tools").is_null()) { return; }
    const Json& tools = body.at("tools");
    if (!tools.is_array()) { bad_request("tools must be an array", "tools"); }
    out.tools.reserve(tools.size());
    for (const Json& item : tools) {
        if (!item.is_object() || !item.contains("type") || !item.at("type").is_string() ||
            item.at("type").get<std::string>() != "function") {
            bad_request("tools entries must be objects with type 'function'", "tools",
                        "tool_type_not_supported");
        }
        if (!item.contains("function") || !item.at("function").is_object()) {
            bad_request("function tools must contain a function object", "tools");
        }
        const Json& fn = item.at("function");
        ToolDefinition tool;
        tool.name     = require_function_name(fn, "tools");
        Json function = Json{{"name", tool.name}};
        if (fn.contains("description") && !fn.at("description").is_null()) {
            if (!fn.at("description").is_string()) {
                bad_request("function description must be a string", "tools");
            }
            tool.description        = fn.at("description").get<std::string>();
            function["description"] = tool.description;
        }
        // Same default as the OpenAI surfaces so one tool definition renders
        // identically into the chat template regardless of endpoint.
        Json parameters = Json{{"type", "object"}, {"properties", Json::object()}};
        if (fn.contains("parameters") && !fn.at("parameters").is_null()) {
            if (!fn.at("parameters").is_object()) {
                bad_request("function parameters must be a JSON object", "tools");
            }
            parameters = fn.at("parameters");
        }
        tool.parameters_json   = parameters.dump();
        function["parameters"] = std::move(parameters);
        function["strict"]     = false;
        // Normalized OpenAI-style function tool object consumed verbatim by the
        // Qwen chat template's <tools> block.
        tool.definition_json = Json{{"type", "function"}, {"function", std::move(function)}}.dump();
        out.tools.push_back(std::move(tool));
    }
}

// Ollama's chat contract carries only these four roles; 'developer' is an OpenAI
// extension its clients never send, so it stays rejected here.
ChatRole parse_ollama_role(const std::string& role) {
    if (role == "system") { return ChatRole::System; }
    if (role == "user") { return ChatRole::User; }
    if (role == "assistant") { return ChatRole::Assistant; }
    if (role == "tool") { return ChatRole::Tool; }
    bad_request("message role must be 'system', 'user', 'assistant', or 'tool'", "messages",
                "unsupported_role");
}

ContentPart parse_image(const Json& image, std::size_t index) {
    // Ollama carries images as bare base64 payloads without a media type; the
    // decoder identifies the container from the bytes.
    if (!image.is_string() || image.get<std::string>().empty()) {
        bad_request("message " + std::to_string(index) + " images must be non-empty base64 strings",
                    "messages");
    }
    ContentPart part;
    part.kind         = ContentKind::Image;
    part.type_raw     = "image";
    part.source.kind  = ninfer::product::media_acquire::SourceKind::Data;
    part.source.value = "data:application/octet-stream;base64," + image.get<std::string>();
    return part;
}

std::vector<ToolCall> parse_assistant_tool_calls(const Json& item, std::size_t index) {
    std::vector<ToolCall> calls;
    if (!item.contains("tool_calls") || item.at("tool_calls").is_null()) { return calls; }
    const Json& tool_calls = item.at("tool_calls");
    if (!tool_calls.is_array()) {
        bad_request("assistant message " + std::to_string(index) + " tool_calls must be an array",
                    "messages");
    }
    calls.reserve(tool_calls.size());
    for (const Json& entry : tool_calls) {
        if (!entry.is_object() || !entry.contains("function") ||
            !entry.at("function").is_object()) {
            bad_request("tool_calls entries must contain a function object", "messages");
        }
        const Json& fn = entry.at("function");
        ToolCall call;
        if (entry.contains("id") && entry.at("id").is_string()) {
            call.id = entry.at("id").get<std::string>();
        }
        call.name = require_function_name(fn, "messages");
        if (!fn.contains("arguments") || fn.at("arguments").is_null()) {
            call.arguments_json = "{}";
        } else if (fn.at("arguments").is_object()) {
            call.arguments_json = fn.at("arguments").dump();
        } else {
            bad_request("tool call arguments must be a JSON object", "messages");
        }
        calls.push_back(std::move(call));
    }
    return calls;
}

void parse_messages(const Json& body, GenerationRequest& out) {
    if (!body.contains("messages")) { bad_request("missing required field: messages", "messages"); }
    const Json& messages = body.at("messages");
    if (!messages.is_array() || messages.empty()) {
        bad_request("messages must be a non-empty array", "messages");
    }
    out.messages.reserve(messages.size());
    for (std::size_t i = 0; i < messages.size(); ++i) {
        const Json& item = messages.at(i);
        if (!item.is_object()) {
            bad_request("message " + std::to_string(i) + " must be an object", "messages");
        }
        if (!item.contains("role") || !item.at("role").is_string()) {
            bad_request("message " + std::to_string(i) + " must have a string role", "messages");
        }
        ChatTurn turn;
        turn.role = parse_ollama_role(item.at("role").get<std::string>());

        std::string content;
        bool has_content = false;
        if (item.contains("content") && !item.at("content").is_null()) {
            if (!item.at("content").is_string()) {
                bad_request("message " + std::to_string(i) + " content must be a string",
                            "messages");
            }
            content     = item.at("content").get<std::string>();
            has_content = true;
        }
        // A tool result may legitimately be empty; the template still renders the
        // tool turn (same as the OpenAI and Anthropic surfaces).
        if (!content.empty() || (has_content && turn.role == ChatRole::Tool)) {
            turn.content.push_back(ContentPart{ContentKind::Text, std::move(content), "text"});
        }

        if (item.contains("images") && !item.at("images").is_null()) {
            if (turn.role != ChatRole::User) {
                bad_request("only user messages may carry images", "messages");
            }
            const Json& images = item.at("images");
            if (!images.is_array()) {
                bad_request("message " + std::to_string(i) + " images must be an array",
                            "messages");
            }
            for (const Json& image : images) { turn.content.push_back(parse_image(image, i)); }
        }

        if (turn.role == ChatRole::Assistant) {
            if (item.contains("thinking") && !item.at("thinking").is_null()) {
                if (!item.at("thinking").is_string()) {
                    bad_request("assistant thinking must be a string", "messages");
                }
                turn.reasoning_content = item.at("thinking").get<std::string>();
            }
            turn.tool_calls = parse_assistant_tool_calls(item, i);
            if (turn.content.empty() && turn.tool_calls.empty() && turn.reasoning_content.empty()) {
                bad_request("assistant message " + std::to_string(i) +
                                " must have content, thinking, or tool_calls",
                            "messages");
            }
        } else if (turn.role == ChatRole::Tool) {
            if (!has_content) {
                bad_request("tool message " + std::to_string(i) + " must have string content",
                            "messages");
            }
        } else if (turn.content.empty()) {
            bad_request("message " + std::to_string(i) + " content must not be empty", "messages");
        }
        // Ollama tool messages carry no call id (`tool_name` is informational);
        // the Qwen template renders tool responses positionally.
        out.messages.push_back(std::move(turn));
    }
}

void parse_stop(const Json& stop, GenerationRequest& out) {
    if (stop.is_string()) {
        if (!stop.get<std::string>().empty()) {
            out.stop_strings.push_back(stop.get<std::string>());
        }
        return;
    }
    if (stop.is_array()) {
        for (const Json& s : stop) {
            if (!s.is_string()) {
                bad_request("options.stop entries must be strings", "options.stop");
            }
            if (!s.get<std::string>().empty()) { out.stop_strings.push_back(s.get<std::string>()); }
        }
        return;
    }
    bad_request("options.stop must be a string or array of strings", "options.stop");
}

constexpr const char* kMappedOptions[] = {
    "temperature",       "top_p", "top_k", "min_p",   "presence_penalty",
    "frequency_penalty", "seed",  "stop",  "num_ctx", "num_predict",
};

// Ollama runner/loader knobs. Residency, threading, and batching are startup
// properties of ninfer-serve, so these are accepted without effect.
constexpr const char* kIgnoredRunnerOptions[] = {
    "num_gpu",   "main_gpu", "num_thread", "num_batch", "num_keep",   "use_mmap",
    "use_mlock", "numa",     "low_vram",   "f16_kv",    "vocab_only", "logits_all",
};

template <std::size_t N>
bool contains_key(const char* const (&keys)[N], const std::string& key) {
    for (const char* candidate : keys) {
        if (key == candidate) { return true; }
    }
    return false;
}

void parse_options(const Json& body, GenerationRequest& out, const RequestLimits& limits) {
    if (!body.contains("options") || body.at("options").is_null()) { return; }
    const Json& options = body.at("options");
    if (!options.is_object()) { bad_request("options must be an object", "options"); }

    for (auto it = options.begin(); it != options.end(); ++it) {
        if (it.value().is_null() || contains_key(kMappedOptions, it.key()) ||
            contains_key(kIgnoredRunnerOptions, it.key())) {
            continue;
        }
        // Sampler knobs with no Engine counterpart (repeat_penalty, mirostat, ...)
        // and unknown keys alike: ignoring them would silently change the
        // requested output, so they are rejected.
        bad_request("options." + it.key() + " is not supported", "options." + it.key(),
                    "option_not_supported");
    }

    SamplingParams& s   = out.sampling;
    s.temperature       = get_number(options, "temperature", "options.temperature");
    s.top_p             = get_number(options, "top_p", "options.top_p");
    s.top_k             = get_int(options, "top_k", "options.top_k");
    s.min_p             = get_number(options, "min_p", "options.min_p");
    s.presence_penalty  = get_number(options, "presence_penalty", "options.presence_penalty");
    s.frequency_penalty = get_number(options, "frequency_penalty", "options.frequency_penalty");
    s.seed              = get_u64(options, "seed", "options.seed");
    if (options.contains("stop") && !options.at("stop").is_null()) {
        parse_stop(options.at("stop"), out);
    }

    if (const std::optional<int> num_ctx = get_int(options, "num_ctx", "options.num_ctx")) {
        if (*num_ctx <= 0) { bad_request("options.num_ctx must be positive", "options.num_ctx"); }
        // The context window is frozen at server startup; a request may ask for
        // less (no effect) but not for more.
        if (limits.max_context > 0 && *num_ctx > limits.max_context) {
            bad_request("options.num_ctx exceeds the server context window of " +
                            std::to_string(limits.max_context),
                        "options.num_ctx", "context_length_exceeded");
        }
    }

    if (const std::optional<int> num_predict =
            get_int(options, "num_predict", "options.num_predict")) {
        // Ollama: -1 => no explicit limit, -2 => fill the context. Both resolve to
        // the server's default output budget; the Engine always bounds output.
        if (*num_predict > 0) {
            out.max_tokens     = *num_predict;
            out.max_tokens_set = true;
        } else if (*num_predict != -1 && *num_predict != -2) {
            bad_request("options.num_predict must be positive, -1, or -2", "options.num_predict");
        }
    }
}

void parse_think(const Json& body, GenerationRequest& out) {
    if (!body.contains("think") || body.at("think").is_null()) { return; }
    const Json& think = body.at("think");
    if (think.is_boolean()) {
        out.enable_thinking = think.get<bool>();
        return;
    }
    if (think.is_string()) {
        const std::string value = think.get<std::string>();
        const std::optional<RequestedReasoningEffort> effort =
            parse_requested_reasoning_effort(value);
        if (!effort || (value != "low" && value != "medium" && value != "high")) {
            bad_request("think must be a boolean or one of low, medium, high", "think");
        }
        out.reasoning_effort       = *effort;
        out.reasoning_effort_param = "think";
        return;
    }
    bad_request("think must be a boolean or one of low, medium, high", "think");
}

Json tool_calls_json(const std::vector<ToolCall>& tool_calls) {
    Json out = Json::array();
    for (std::size_t i = 0; i < tool_calls.size(); ++i) {
        const ToolCall& call = tool_calls[i];
        Json arguments       = Json::parse(call.arguments_json, nullptr, false);
        if (arguments.is_discarded() || !arguments.is_object()) { arguments = Json::object(); }
        Json function = {{"index", static_cast<int>(i)},
                         {"name", call.name},
                         {"arguments", std::move(arguments)}};
        Json item     = Json{{"function", std::move(function)}};
        if (!call.id.empty()) { item["id"] = call.id; }
        out.push_back(std::move(item));
    }
    return out;
}

Json message_json(const std::string& content, const std::string& thinking,
                  const std::vector<ToolCall>& tool_calls) {
    Json message = {{"role", "assistant"}, {"content", content}};
    if (!thinking.empty()) { message["thinking"] = thinking; }
    if (!tool_calls.empty()) { message["tool_calls"] = tool_calls_json(tool_calls); }
    return message;
}

std::int64_t nanoseconds(double seconds) {
    if (!(seconds > 0.0)) { return 0; }
    return static_cast<std::int64_t>(std::llround(seconds * 1e9));
}

void add_timings(Json& frame, const OllamaTimings& timings) {
    frame["total_duration"]       = nanoseconds(timings.total_seconds);
    frame["load_duration"]        = nanoseconds(timings.load_seconds);
    frame["prompt_eval_count"]    = timings.prompt_eval_count;
    frame["prompt_eval_duration"] = nanoseconds(timings.prompt_eval_seconds);
    frame["eval_count"]           = timings.eval_count;
    frame["eval_duration"]        = nanoseconds(timings.eval_seconds);
}

Json base_frame(const std::string& model, const std::string& created_at) {
    return Json{{"model", model}, {"created_at", created_at}};
}

std::string ndjson(const Json& frame) { return frame.dump() + "\n"; }

Json details_json(const OllamaModelFacts& facts) {
    return Json{{"parent_model", ""},
                {"format", "ninfer"},
                {"family", facts.target},
                {"families", Json::array({facts.target})},
                {"quantization_level", facts.weights_id}};
}

Json model_entry(const OllamaModelFacts& facts) {
    return Json{{"name", facts.model_id},
                {"model", facts.model_id},
                {"modified_at", facts.modified_at},
                {"size", facts.size_bytes},
                {"details", details_json(facts)}};
}

std::string require_model(const Json& body) {
    if (!body.is_object()) { bad_request("request body must be a JSON object"); }
    if (!body.contains("model") || !body.at("model").is_string() ||
        body.at("model").get<std::string>().empty()) {
        bad_request("missing required field: model", "model");
    }
    return body.at("model").get<std::string>();
}

} // namespace

std::optional<std::string> ollama_preload_model(const Json& body) {
    if (!body.is_object() || !body.contains("messages") || !body.at("messages").is_array() ||
        !body.at("messages").empty()) {
        return std::nullopt;
    }
    return require_model(body);
}

bool ollama_model_matches(const std::string& requested, const std::string& model_id) {
    static constexpr std::string_view kLatest = ":latest";
    if (requested == model_id) { return true; }
    return requested.size() == model_id.size() + kLatest.size() &&
           requested.compare(0, model_id.size(), model_id) == 0 &&
           requested.compare(model_id.size(), kLatest.size(), kLatest) == 0;
}

GenerationRequest parse_ollama_chat_request(const Json& body, const RequestLimits& limits) {
    GenerationRequest out;
    out.tool_name_max_length = kMaxToolNameLength;
    out.model                = require_model(body);

    // Structured output requires constrained decoding, which the Engine does not
    // implement; a request that asks for it is rejected rather than served loosely.
    if (body.contains("format") && !body.at("format").is_null()) {
        bad_request("format is not supported", "format", "format_not_supported");
    }
    // `keep_alive` describes runner residency, which is fixed for the server
    // lifetime; the field is accepted without effect.

    parse_tools(body, out);
    parse_messages(body, out);
    out.max_tokens     = limits.default_max_tokens;
    out.max_tokens_set = false;
    parse_options(body, out, limits);
    parse_think(body, out);

    if (body.contains("stream") && !body.at("stream").is_null()) {
        if (!body.at("stream").is_boolean()) { bad_request("stream must be a boolean", "stream"); }
        out.stream = body.at("stream").get<bool>();
    } else {
        out.stream = true;
    }
    return out;
}

std::string make_ollama_chat_response(const std::string& model, const std::string& created_at,
                                      const std::string& content, const std::string& thinking,
                                      const std::vector<ToolCall>& tool_calls,
                                      const char* done_reason, const OllamaTimings& timings) {
    Json body           = base_frame(model, created_at);
    body["message"]     = message_json(content, thinking, tool_calls);
    body["done"]        = true;
    body["done_reason"] = done_reason;
    add_timings(body, timings);
    return body.dump();
}

std::string make_ollama_chat_load_response(const std::string& model,
                                           const std::string& created_at) {
    Json body           = base_frame(model, created_at);
    body["message"]     = message_json({}, {}, {});
    body["done"]        = true;
    body["done_reason"] = "load";
    return body.dump();
}

std::string make_ollama_chat_content_frame(const std::string& model, const std::string& created_at,
                                           const std::string& delta_text) {
    Json frame       = base_frame(model, created_at);
    frame["message"] = message_json(delta_text, {}, {});
    frame["done"]    = false;
    return ndjson(frame);
}

std::string make_ollama_chat_thinking_frame(const std::string& model, const std::string& created_at,
                                            const std::string& delta_text) {
    Json frame       = base_frame(model, created_at);
    frame["message"] = message_json({}, delta_text, {});
    frame["done"]    = false;
    return ndjson(frame);
}

std::string make_ollama_chat_tool_calls_frame(const std::string& model,
                                              const std::string& created_at,
                                              const std::vector<ToolCall>& tool_calls) {
    Json frame       = base_frame(model, created_at);
    frame["message"] = message_json({}, {}, tool_calls);
    frame["done"]    = false;
    return ndjson(frame);
}

std::string make_ollama_chat_done_frame(const std::string& model, const std::string& created_at,
                                        const char* done_reason, const OllamaTimings& timings) {
    Json frame           = base_frame(model, created_at);
    frame["message"]     = message_json({}, {}, {});
    frame["done"]        = true;
    frame["done_reason"] = done_reason;
    add_timings(frame, timings);
    return ndjson(frame);
}

std::string make_ollama_tags(const OllamaModelFacts& facts) {
    return Json{{"models", Json::array({model_entry(facts)})}}.dump();
}

std::string make_ollama_show(const OllamaModelFacts& facts) {
    Json capabilities = Json::array({"completion", "tools"});
    if (facts.vision) { capabilities.push_back("vision"); }
    if (facts.thinking) { capabilities.push_back("thinking"); }
    const Json model_info = {{"general.architecture", facts.target},
                             {"general.name", facts.model_id},
                             {facts.target + ".context_length", facts.max_context}};
    return Json{{"modified_at", facts.modified_at},
                {"details", details_json(facts)},
                {"model_info", model_info},
                {"capabilities", std::move(capabilities)}}
        .dump();
}

std::string make_ollama_ps(const OllamaModelFacts& facts) {
    Json entry              = model_entry(facts);
    entry["size_vram"]      = facts.size_vram_bytes;
    entry["context_length"] = facts.max_context;
    return Json{{"models", Json::array({std::move(entry)})}}.dump();
}

std::string make_ollama_version() { return Json{{"version", kOllamaCompatVersion}}.dump(); }

std::string make_ollama_error_body(const ApiError& error) {
    return Json{{"error", error.message}}.dump();
}

std::string make_ollama_error_frame(const ApiError& error) {
    return make_ollama_error_body(error) + "\n";
}

std::string ollama_timestamp(std::int64_t unix_nanoseconds) {
    const std::time_t seconds = static_cast<std::time_t>(unix_nanoseconds / 1'000'000'000);
    const long nanos          = static_cast<long>(unix_nanoseconds % 1'000'000'000);
    std::tm utc{};
#if defined(_WIN32)
    gmtime_s(&utc, &seconds);
#else
    gmtime_r(&seconds, &utc);
#endif
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%04d-%02d-%02dT%02d:%02d:%02d.%09ldZ",
                  utc.tm_year + 1900, utc.tm_mon + 1, utc.tm_mday, utc.tm_hour, utc.tm_min,
                  utc.tm_sec, nanos);
    return buffer;
}

std::string ollama_timestamp_now() {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return ollama_timestamp(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

} // namespace ninfer::serve
