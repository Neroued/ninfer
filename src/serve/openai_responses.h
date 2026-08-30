#pragma once

// OpenAI Responses protocol adapter. Request parsing, state normalization, response encoding and
// HTTP transport live in separate translation units; only wire-independent GenerationRequest is
// passed to GenerationService.

#include "serve/openai_responses_store.h"
#include "serve/request.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace ninfer::serve {

struct GenerationOutcome;

struct OpenAIResponsesFunctionIdentity {
    std::string name;
    std::optional<std::string> wire_namespace;

    bool operator==(const OpenAIResponsesFunctionIdentity&) const = default;
};

struct OpenAIResponsesPromptRequest {
    std::string model;
    GenerationRequest generation;
    std::vector<ChatTurn> input_turns;
    std::vector<nlohmann::json> input_items;
    std::optional<std::string> instructions;
    std::optional<std::string> previous_response_id;
};

struct OpenAIResponsesCreateRequest {
    OpenAIResponsesPromptRequest prompt;
    nlohmann::json metadata    = nlohmann::json::object();
    nlohmann::json tools       = nlohmann::json::array();
    nlohmann::json tool_choice = "auto";
    // Responses beta namespace tools are flattened for the Engine and restored only at the wire
    // boundary. Never infer a namespace by splitting an Engine function name.
    std::unordered_map<std::string, OpenAIResponsesFunctionIdentity> tool_identities;
    std::optional<int> requested_max_output_tokens;
    std::optional<int> max_tool_calls;
    bool parallel_tool_calls = true;
    bool store               = true;
    bool stream              = false;
};

struct OpenAIResponsesResolvedPrompt {
    GenerationRequest generation;
    OpenAIResponseContext parent;
    std::optional<std::string> session_key;
    ContextCacheHints cache_hints;
    bool preserve_thinking_semantic_change = false;
};

struct OpenAIResponsesRuntimeValues {
    float temperature       = 1.0F;
    float top_p             = 1.0F;
    int cached_input_tokens = 0;
};

struct BuiltOpenAIResponse {
    nlohmann::json body;
    std::vector<nlohmann::json> output_items;
    std::vector<ChatTurn> output_history;
};

OpenAIResponsesCreateRequest parse_openai_responses_create_request(const nlohmann::json& body,
                                                                   const RequestLimits& limits);

OpenAIResponsesPromptRequest
parse_openai_responses_input_tokens_request(const nlohmann::json& body,
                                            const RequestLimits& limits);

OpenAIResponsesResolvedPrompt
resolve_openai_responses_prompt(const OpenAIResponsesPromptRequest& request,
                                OpenAIResponsesStore& store, std::optional<std::string> response_id,
                                bool store_response);

BuiltOpenAIResponse make_openai_response_object(const std::string& id, std::int64_t created_at,
                                                const OpenAIResponsesCreateRequest& request,
                                                const OpenAIResponsesRuntimeValues& runtime,
                                                const GenerationOutcome& outcome);

std::string make_openai_response_input_tokens_body(int input_tokens);

struct OpenAIResponsesStreamFinish {
    BuiltOpenAIResponse response;
    std::vector<std::string> events_before_terminal;
};

class OpenAIResponsesEventStream {
public:
    OpenAIResponsesEventStream(std::string response_id, std::int64_t created_at,
                               OpenAIResponsesCreateRequest request,
                               OpenAIResponsesRuntimeValues runtime);
    ~OpenAIResponsesEventStream();
    OpenAIResponsesEventStream(OpenAIResponsesEventStream&&) noexcept;
    OpenAIResponsesEventStream& operator=(OpenAIResponsesEventStream&&) noexcept;

    OpenAIResponsesEventStream(const OpenAIResponsesEventStream&)            = delete;
    OpenAIResponsesEventStream& operator=(const OpenAIResponsesEventStream&) = delete;

    std::vector<std::string> start();
    std::vector<std::string> reasoning_delta(const std::string& text);
    std::vector<std::string> content_delta(const std::string& text);
    OpenAIResponsesStreamFinish finish(const GenerationOutcome& outcome);
    std::string terminal(const BuiltOpenAIResponse& response);
    std::string failed(const ApiError& error);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::serve
