#pragma once

#include "qus/media/source.h"

// Internal, wire-format-independent representation of a generation request.
//
// This is deliberately decoupled from the OpenAI JSON schema (openai_schema.h)
// and from the qus_core text runner (translate.h). New wire protocols map into
// this struct; new engine capabilities read from it. In particular the full
// SamplingParams surface is parsed and carried here even though the greedy
// engine ignores it today, so adding a real sampler later touches only the
// engine hook, not the protocol or translation layers.

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace qus::serve {

// A structured API error mapped onto an error object + HTTP status. Wire-format
// independent: each protocol layer renders it into its own error body shape.
struct ApiError {
    int status       = 400;
    std::string type = "invalid_request_error";
    std::string message;
    std::string param; // optional
    std::string code;  // optional
};

class ApiException : public std::runtime_error {
public:
    explicit ApiException(ApiError error)
        : std::runtime_error(error.message), error_(std::move(error)) {}

    [[nodiscard]] const ApiError& error() const noexcept { return error_; }

private:
    ApiError error_;
};

// Server-side context needed while parsing/validating a request.
struct RequestLimits {
    int default_max_tokens    = 512;
    std::uint32_t max_context = 8192;
};

struct CompletionUsage {
    int prompt_tokens     = 0;
    int completion_tokens = 0;
};

enum class ContentKind {
    Text,
    Image,
    Video,
    InputAudio,
    Unsupported,
};

struct ContentPart {
    ContentKind kind = ContentKind::Text;
    std::string text;     // populated for Text
    std::string type_raw; // original OpenAI "type" string (diagnostics / future use)
    qus::media::Source source;
};

struct ToolDefinition {
    std::string name;
    std::string description;
    std::string parameters_json;
    std::string definition_json; // normalized OpenAI function-tool object for Qwen prompt rendering
    bool strict = false;
};

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

enum class ToolChoiceMode {
    Auto,
    None,
    Required,
    Named,
};

struct ToolChoice {
    ToolChoiceMode mode = ToolChoiceMode::Auto;
    std::string name;
};

struct ChatTurn {
    std::string role; // system | user | assistant | tool (validated in translate)
    std::vector<ContentPart>
        content; // one or more parts; assistant content may be empty with tool_calls
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;      // populated for role=tool
    std::string reasoning_content; // assistant thinking carried across turns (round-tripped to the
                                   // template)
};

// The complete OpenAI sampling surface. Parsed now, honored when the sampler
// lands. `logit_bias` maps token id -> bias.
struct SamplingParams {
    std::optional<double> temperature;
    std::optional<double> top_p;
    std::optional<int> top_k;
    std::optional<double> presence_penalty;
    std::optional<double> frequency_penalty;
    std::optional<std::uint64_t> seed;
    std::unordered_map<int, double> logit_bias;
    int n = 1;
};

struct GenerationRequest {
    std::string model;
    std::vector<ChatTurn> messages;
    std::vector<ToolDefinition> tools;
    ToolChoice tool_choice;
    std::vector<std::string> stop_strings;
    int max_tokens      = 0; // 0 => use server default
    bool max_tokens_set = false;
    bool stream         = false;
    bool include_usage  = false;
    std::optional<bool> enable_thinking; // non-standard extension; falls back to server default
    SamplingParams sampling;

    [[nodiscard]] bool uses_tools() const noexcept {
        return !tools.empty() && tool_choice.mode != ToolChoiceMode::None;
    }

    [[nodiscard]] bool has_tool_history() const noexcept {
        for (const ChatTurn& message : messages) {
            if (!message.tool_calls.empty() || message.role == "tool") { return true; }
        }
        return false;
    }
};

} // namespace qus::serve
