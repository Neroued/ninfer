#include "serve/tool_call_parser.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cctype>
#include <cstdio>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

std::string trim_ascii(std::string_view text) {
    std::size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
        ++begin;
    }
    std::size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) { --end; }
    return std::string(text.substr(begin, end - begin));
}

std::string rtrim_ascii(std::string_view text) {
    std::size_t end = text.size();
    while (end != 0 && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) { --end; }
    return std::string(text.substr(0, end));
}

void skip_ws(std::string_view text, std::size_t& pos) {
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) != 0) { ++pos; }
}

bool starts_with_at(std::string_view text, std::size_t pos, std::string_view prefix) {
    return pos <= text.size() && text.substr(pos, prefix.size()) == prefix;
}

std::size_t longest_suffix_prefix(std::string_view text, std::string_view marker) {
    const std::size_t maximum = std::min(text.size(), marker.size() - 1);
    for (std::size_t size = maximum; size != 0; --size) {
        if (text.substr(text.size() - size) == marker.substr(0, size)) { return size; }
    }
    return 0;
}

bool valid_function_name(std::string_view name, std::size_t max_name_length) {
    if (name.empty() || name.size() > max_name_length) { return false; }
    for (const unsigned char c : name) {
        if (std::isalnum(c) == 0 && c != '_' && c != '-') { return false; }
    }
    return true;
}

std::string new_tool_call_id() {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<std::uint64_t> dist;
    std::array<char, 32> buf{};
    std::snprintf(buf.data(), buf.size(), "call_%016llx",
                  static_cast<unsigned long long>(dist(rng)));
    return std::string(buf.data());
}

const std::unordered_map<std::string, std::vector<std::string>>* tool_param_types(
    const ToolParamTypeMap& map, const std::string& tool_name) {
    const auto it = map.find(tool_name);
    return it == map.end() ? nullptr : &it->second;
}

// The full set of declared non-string types recorded for (tool_name, param),
// or nullptr when the parameter has no non-string schema permission.
const std::vector<std::string>* param_declared_types(const ToolParamTypeMap& map,
                                                     const std::string& tool_name,
                                                     const std::string& param) {
    const auto* params = tool_param_types(map, tool_name);
    if (params == nullptr) { return nullptr; }
    const auto it = params->find(param);
    return it == params->end() ? nullptr : &it->second;
}

// vLLM's qwen3coder coercion for boolean-declared parameters: the model may
// emit Python-style scalars (True/False, 1/0) that are not valid JSON.
// `value` is the lowercased raw text; "true"/"1" -> true, "false"/"0" ->
// false; anything else is not a boolean and stays raw text.
std::optional<bool> coerce_boolean(std::string_view value) {
    if (value == "true" || value == "1") { return true; }
    if (value == "false" || value == "0") { return false; }
    return std::nullopt;
}

} // namespace

namespace {

// Valid JSON Schema "type" values that are not string. A parameter is only
// allowed to deserialize when every type it declares is in this set.
const std::unordered_set<std::string>& non_string_schema_types() {
    static const std::unordered_set<std::string> types = {"integer", "number", "boolean",
                                                          "array", "object", "null"};
    return types;
}

// Classify a parameter's schema "type" (a string or an array of strings) into
// the set of declared types. Returns false if "type" is absent or not a
// string/array; in that case classification is uncertain and the caller
// preserves raw text. Returns true and fills `declared` otherwise.
bool classify_param_type(const Json& spec, std::vector<std::string>& declared) {
    const auto type_it = spec.find("type");
    if (type_it == spec.end()) { return false; }
    if (type_it->is_string()) {
        declared.push_back(type_it->get<std::string>());
        return true;
    }
    if (type_it->is_array()) {
        for (const Json& t : *type_it) {
            if (!t.is_string()) { return false; }
            declared.push_back(t.get<std::string>());
        }
        // An empty type array (e.g. "type":[]) is uncertain, not a positive
        // declaration of a non-string type; returning false here preserves
        // raw text and prevents all_non_string_types from succeeding vacuously
        // on an empty set (which would record the parameter with no declared
        // types).
        if (declared.empty()) { return false; }
        return true;
    }
    return false;
}

// Whether every declared type is a valid non-string JSON Schema type. If any
// declared type is "string" or unknown/invalid, the schema permits (or may
// permit) a string value, so the parser must preserve raw text.
bool all_non_string_types(const std::vector<std::string>& declared) {
    const auto& valid = non_string_schema_types();
    for (const std::string& type : declared) {
        if (valid.find(type) == valid.end()) { return false; }
    }
    return true;
}

} // namespace

ToolParamTypeMap build_tool_param_type_map(const std::vector<ToolDefinition>& tools) {
    ToolParamTypeMap map;
    for (const ToolDefinition& tool : tools) {
        // Replace any prior entry for this tool name first, before any early
        // exit, so a redefinition with an empty/malformed/no-properties schema
        // cannot leak stale non-string permissions from a previous definition.
        map[tool.name] = {};
        if (tool.parameters_json.empty()) { continue; }
        const Json schema = Json::parse(tool.parameters_json, nullptr, false);
        if (!schema.is_object()) { continue; }
        const auto props_it = schema.find("properties");
        if (props_it == schema.end() || !props_it->is_object()) { continue; }
        // The entry for tool.name was already reset to empty at the top of
        // the loop; populate it only from this definition's properties.
        auto& inner = map[tool.name];
        for (const auto& [name, spec] : props_it->items()) {
            if (!spec.is_object()) { continue; }
            std::vector<std::string> declared;
            if (!classify_param_type(spec, declared)) { continue; }
            // Record only when every declared type is a valid non-string type;
            // string-allowed, unknown/invalid, and absent-type params are left
            // out so the parser preserves raw text for them. Store the full
            // declared set (not just the first element) so the parser can
            // reason about nullable types (e.g. ["boolean","null"])
            // independently of the type-array order.
            if (all_non_string_types(declared)) { inner[name] = declared; }
        }
    }
    return map;
}

namespace {

bool parse_parameter(std::string_view inner, std::size_t& pos, Json& args,
                     const std::string& tool_name, const ToolParamTypeMap& param_types) {
    constexpr std::string_view kParamOpen  = "<parameter=";
    constexpr std::string_view kParamClose = "</parameter>";
    if (!starts_with_at(inner, pos, kParamOpen)) { return false; }
    const std::size_t name_begin = pos + kParamOpen.size();
    const std::size_t name_end   = inner.find('>', name_begin);
    if (name_end == std::string_view::npos || name_end == name_begin) { return false; }
    const std::string key       = std::string(inner.substr(name_begin, name_end - name_begin));
    pos                         = name_end + 1;
    const std::size_t value_end = inner.find(kParamClose, pos);
    if (value_end == std::string_view::npos) { return false; }
    const std::string raw_value = trim_ascii(inner.substr(pos, value_end - pos));
    const std::vector<std::string>* declared = param_declared_types(param_types, tool_name, key);
    const bool is_boolean =
        declared != nullptr &&
        std::find(declared->begin(), declared->end(), "boolean") != declared->end();
    if (is_boolean) {
        // Boolean-declared params: the model may emit Python-style scalars
        // (True/False, 1/0) that are not valid JSON, so coerce the raw text
        // instead of adopting a parsed value. The literal null is JSON null:
        // a valid value for a nullable boolean, and the faithful reading of
        // the token otherwise (matching the fallthrough for other nullable
        // types). Any other value stays raw text for the client to validate.
        // Both comparisons are case-insensitive, matching the model's
        // Python-style emissions (True/TRUE, Null/NULL).
        std::string lower;
        lower.reserve(raw_value.size());
        for (const char c : raw_value) {
            lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
        }
        if (std::optional<bool> coerced = coerce_boolean(lower)) {
            args[key] = *coerced;
        } else if (lower == "null") {
            args[key] = nullptr;
        } else {
            args[key] = Json(raw_value);
        }
        pos = value_end + kParamClose.size();
        return true;
    }
    // Only adopt the deserialized JSON type when the schema explicitly
    // permits a non-string type. For string-typed, unknown, or absent
    // params, keep the raw text so the model's value reaches the client
    // with its type intact; the client owns the schema and validates.
    Json parsed = Json::parse(raw_value, nullptr, false);
    const bool can_deserialize = declared != nullptr;
    args[key] = (parsed.is_discarded() || !can_deserialize) ? Json(raw_value) : parsed;
    pos                         = value_end + kParamClose.size();
    return true;
}

Json typed_json_argument(const std::string& tool_name, const std::string& key, const Json& value,
                         const ToolParamTypeMap& param_types) {
    const std::vector<std::string>* declared = param_declared_types(param_types, tool_name, key);
    const bool is_boolean =
        declared != nullptr &&
        std::find(declared->begin(), declared->end(), "boolean") != declared->end();
    if (is_boolean) {
        if (value.is_boolean() || value.is_null()) { return value; }
        if (value.is_string()) {
            const std::string raw = value.get<std::string>();
            std::string lower;
            lower.reserve(raw.size());
            for (const char c : raw) {
                lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
            }
            if (std::optional<bool> coerced = coerce_boolean(lower)) { return *coerced; }
            if (lower == "null") { return nullptr; }
            return value;
        }
        if (value.is_number_integer() && !value.is_number_float()) {
            const auto n = value.get<std::int64_t>();
            if (n == 1) { return true; }
            if (n == 0) { return false; }
        }
        return Json(value.dump());
    }
    if (declared != nullptr) { return value; }
    if (value.is_string()) { return value; }
    return Json(value.dump());
}

std::size_t json_object_end(std::string_view text, std::size_t pos) {
    if (pos >= text.size() || text[pos] != '{') { return std::string_view::npos; }
    int depth         = 0;
    bool in_string    = false;
    bool escaped      = false;
    for (std::size_t i = pos; i < text.size(); ++i) {
        const char c = text[i];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        if (c == '"') {
            in_string = true;
        } else if (c == '{') {
            ++depth;
        } else if (c == '}') {
            --depth;
            if (depth == 0) { return i + 1; }
        }
    }
    return std::string_view::npos;
}

bool parse_json_argument_object(std::string_view inner, std::size_t& pos, Json& args,
                                const std::string& tool_name,
                                const ToolParamTypeMap& param_types) {
    const std::size_t end = json_object_end(inner, pos);
    if (end == std::string_view::npos) { return false; }
    Json payload = Json::parse(inner.substr(pos, end - pos), nullptr, false);
    if (payload.is_discarded() || !payload.is_object()) { return false; }
    for (auto it = payload.begin(); it != payload.end(); ++it) {
        args[it.key()] = typed_json_argument(tool_name, it.key(), it.value(), param_types);
    }
    pos = end;
    return true;
}

bool parse_one_tool_call(std::string_view block, std::size_t max_name_length,
                         const ToolParamTypeMap& param_types, ToolCall& out,
                         std::size_t& consumed) {
    constexpr std::string_view kFunctionOpen  = "<function=";
    constexpr std::string_view kFunctionClose = "</function>";
    std::size_t pos                           = 0;
    skip_ws(block, pos);
    if (!starts_with_at(block, pos, kFunctionOpen)) { return false; }
    const std::size_t name_begin = pos + kFunctionOpen.size();
    const std::size_t name_end   = block.find('>', name_begin);
    if (name_end == std::string_view::npos || name_end == name_begin) { return false; }
    const std::string name = std::string(block.substr(name_begin, name_end - name_begin));
    if (!valid_function_name(name, max_name_length)) { return false; }
    pos = name_end + 1;

    Json args = Json::object();
    skip_ws(block, pos);
    if (pos < block.size() && block[pos] == '{') {
        if (!parse_json_argument_object(block, pos, args, name, param_types)) { return false; }
        skip_ws(block, pos);
        if (!starts_with_at(block, pos, kFunctionClose)) { return false; }
        pos += kFunctionClose.size();
    } else {
        const std::size_t function_end = block.find(kFunctionClose, pos);
        if (function_end == std::string_view::npos) { return false; }
        const std::string_view params = block.substr(pos, function_end - pos);
        std::size_t param_pos         = 0;
        for (;;) {
            skip_ws(params, param_pos);
            if (param_pos >= params.size()) { break; }
            if (!parse_parameter(params, param_pos, args, name, param_types)) { return false; }
        }
        pos = function_end + kFunctionClose.size();
    }

    out.id             = new_tool_call_id();
    out.name           = name;
    out.arguments_json = args.dump();
    consumed           = pos;
    return true;
}

ParsedToolCallOutput fallback(const std::string& text) {
    ParsedToolCallOutput out;
    out.content = text;
    return out;
}

} // namespace

ParsedToolCallOutput parse_qwen_tool_call_output(const std::string& text,
                                                 std::size_t max_tool_name_length, bool tolerant) {
    return parse_qwen_tool_call_output(text, max_tool_name_length, {}, tolerant);
}

ParsedToolCallOutput parse_qwen_tool_call_output(const std::string& text,
                                                 std::size_t max_tool_name_length,
                                                 const ToolParamTypeMap& param_types,
                                                 bool tolerant) {
    constexpr std::string_view kToolOpen  = "<tool_call>";
    constexpr std::string_view kToolClose = "</tool_call>";

    const std::size_t first = text.find(kToolOpen);
    if (first == std::string::npos) { return fallback(text); }

    ParsedToolCallOutput out;
    // Text before the first <tool_call> is ordinary content and is RETAINED here.
    // Suppressing it in the parser makes the terminal body shorter than what a
    // streaming response may already have emitted, which trips the
    // streamed-vs-terminal invariant and kills the request mid-stream. The caller
    // suppresses it instead, because only the caller knows how many bytes actually
    // left the process.
    out.content = rtrim_ascii(std::string_view(text).substr(0, first));

    std::size_t pos = first;
    while (pos < text.size()) {
        skip_ws(text, pos);
        if (pos >= text.size()) { break; }
        if (!starts_with_at(text, pos, kToolOpen)) {
            if (tolerant && !out.tool_calls.empty()) { break; }
            return fallback(text);
        }
        const std::size_t inner_begin = pos + kToolOpen.size();
        ToolCall call;
        std::size_t consumed = 0;
        if (!parse_one_tool_call(std::string_view(text).substr(inner_begin), max_tool_name_length,
                                 param_types, call, consumed)) {
            // Once one complete call has been recovered, do not discard it just
            // because Qwen started a malformed second call or added a suffix.
            if (tolerant && !out.tool_calls.empty()) { break; }
            return fallback(text);
        }
        pos = inner_begin + consumed;
        skip_ws(text, pos);
        if (starts_with_at(text, pos, kToolClose)) {
            pos += kToolClose.size();
        } else if (!tolerant) {
            return fallback(text);
        } else {
            out.tool_calls.push_back(std::move(call));
            break;
        }
        out.tool_calls.push_back(std::move(call));
    }

    if (out.tool_calls.empty()) { return fallback(text); }
    out.is_tool_call_response = true;
    return out;
}

std::string ToolCallStreamFilter::feed(std::string_view text) {
    if (finished_) { throw std::logic_error("tool-call stream filter is already finished"); }
    if (text.empty()) { return {}; }
    if (saw_tool_marker_) {
        tool_region_.append(text);
        return {};
    }

    constexpr std::string_view kToolOpen = "<tool_call>";
    pending_.append(text);
    const std::size_t marker = pending_.find(kToolOpen);
    if (marker != std::string::npos) {
        std::size_t safe_end = marker;
        while (safe_end != 0 &&
               std::isspace(static_cast<unsigned char>(pending_[safe_end - 1])) != 0) {
            --safe_end;
        }
        held_prefix_     = pending_.substr(0, safe_end);
        tool_region_     = pending_.substr(safe_end);
        pending_.clear();
        saw_tool_marker_ = true;
        return {};
    }

    const std::size_t prefix = longest_suffix_prefix(pending_, kToolOpen);
    std::size_t safe_end     = pending_.size() - prefix;
    while (safe_end != 0 && std::isspace(static_cast<unsigned char>(pending_[safe_end - 1])) != 0) {
        --safe_end;
    }
    std::string visible = pending_.substr(0, safe_end);
    pending_.erase(0, safe_end);
    emitted_bytes_ += visible.size();
    return visible;
}

std::string ToolCallStreamFilter::finish(bool is_tool_call_response) {
    if (finished_) { throw std::logic_error("tool-call stream filter is already finished"); }
    finished_ = true;
    if (is_tool_call_response) {
        pending_.clear();
        held_prefix_.clear();
        tool_region_.clear();
        return {};
    }
    std::string tail = std::move(held_prefix_);
    tail += pending_;
    tail += tool_region_;
    pending_.clear();
    tool_region_.clear();
    emitted_bytes_ += tail.size();
    return tail;
}

} // namespace ninfer::serve
