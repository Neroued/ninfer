#pragma once

#include "serve/request.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ninfer::serve {

struct ParsedToolCallOutput {
    bool is_tool_call_response = false;
    std::string content;
    std::vector<ToolCall> tool_calls;
};

// Per-tool parameter deserialization allow-list distilled from the request
// ToolDefinition list. Outer key: tool name. Inner key: parameter name; the
// inner value is the full set of declared non-string types, in schema order.
// Only parameters whose JSON Schema "type" is a valid non-string type (or an
// array of valid non-string types) are recorded. The parser deserializes
// recorded parameters and, for sets containing "boolean", coerces
// Python-style scalars (True/False, 1/0) to JSON booleans and the literal
// null to JSON null. A parameter absent from the inner map (and a tool
// absent from the outer map) has no schema permission to deserialize: the
// parser preserves raw text and the client owns type interpretation.
using ToolParamTypeMap =
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<std::string>>>;

// Distill the request ToolDefinition list into a per-tool parameter
// deserialization allow-list. Each ToolDefinition::parameters_json is a JSON
// Schema object; its "properties" object maps each parameter name to an
// object whose "type" field (a string or an array of strings) declares the
// schema type(s). A parameter is recorded only when every declared type is
// one of the valid non-string JSON Schema types {integer, number, boolean,
// array, object, null}; otherwise (string allowed, unknown/invalid type, or
// absent "type") it is omitted so the parser preserves raw text. A tool
// name seen again replaces its entry so a redefinition cannot leak stale
// non-string permissions from a prior definition.
ToolParamTypeMap build_tool_param_type_map(const std::vector<ToolDefinition>& tools);

// Parse Qwen's XML-like tool-call format. In tolerant mode, a complete function
// call is recovered even when the model adds wrapper garbage or suffix text.
// A successful parse RETAINS any preamble before <tool_call> as content. It is
// not user-visible assistant text for a tool turn (OpenAI sends content=null
// when tool_calls exist), but suppressing it here would make the terminal body
// shorter than what a streaming response may already have emitted, which aborts
// the request mid-stream. GenerationService drops it instead, once it knows
// streamed_content_bytes == 0.
ParsedToolCallOutput parse_qwen_tool_call_output(const std::string& text,
                                                 std::size_t max_tool_name_length,
                                                 bool tolerant = false);
ParsedToolCallOutput parse_qwen_tool_call_output(const std::string& text,
                                                 std::size_t max_tool_name_length,
                                                 const ToolParamTypeMap& param_types,
                                                 bool tolerant = false);

// Incrementally publishes text that is provably outside a possible Qwen
// <tool_call> suffix. At terminal time, a valid tool response discards the
// buffered tool region; malformed/non-tool output flushes it verbatim.
class ToolCallStreamFilter {
public:
    std::string feed(std::string_view text);
    std::string finish(bool is_tool_call_response);

    [[nodiscard]] std::size_t emitted_bytes() const noexcept { return emitted_bytes_; }

private:
    std::string pending_;
    std::string held_prefix_;
    std::string tool_region_;
    std::size_t emitted_bytes_ = 0;
    bool saw_tool_marker_      = false;
    bool finished_             = false;
};

} // namespace ninfer::serve
