#pragma once

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
#include <string>
#include <unordered_map>
#include <vector>

namespace qus::serve {

// A single piece of message content. Only Text is executable today; the other
// kinds are modeled so the DTO layer can carry multimodal content and the
// translation layer can reject it with a precise error until it is supported.
enum class ContentKind {
    Text,
    ImageUrl,
    InputAudio,
    Unsupported,
};

struct ContentPart {
    ContentKind kind = ContentKind::Text;
    std::string text;      // populated for Text
    std::string type_raw;  // original OpenAI "type" string (diagnostics / future use)
};

struct ChatTurn {
    std::string role;                  // system | user | assistant (validated in translate)
    std::vector<ContentPart> content;  // one or more parts
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
    std::vector<std::string> stop_strings;
    int max_tokens        = 0;  // 0 => use server default
    bool max_tokens_set   = false;
    bool stream           = false;
    bool include_usage    = false;
    std::optional<bool> enable_thinking;  // non-standard extension; falls back to server default
    SamplingParams sampling;
};

} // namespace qus::serve
