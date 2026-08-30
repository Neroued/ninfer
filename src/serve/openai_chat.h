#pragma once

// OpenAI Chat Completions wire adapter. Parsing produces one protocol envelope plus an executable
// GenerationRequest; response builders consume protocol-neutral GenerationOutcome values.

#include "serve/request.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace ninfer::serve {

struct GenerationOutcome;

struct OpenAIChatRequest {
    std::string model;
    GenerationRequest generation;
    bool stream                 = false;
    bool include_usage          = false;
    bool output_tokens_explicit = false;
};

OpenAIChatRequest parse_chat_completion_request(const nlohmann::json& body,
                                                const RequestLimits& limits);

struct OpenAIChatResponseIdentity {
    std::string id;
    std::string model;
    std::int64_t created = 0;
};

OpenAIChatResponseIdentity make_openai_chat_response_identity(std::string model);
std::string make_chat_completion_response(const OpenAIChatResponseIdentity& identity,
                                          const GenerationOutcome& outcome);

class OpenAIChatStream {
public:
    OpenAIChatStream(OpenAIChatResponseIdentity identity, bool include_usage);

    std::string start();
    std::string reasoning_delta(const std::string& text);
    std::string content_delta(const std::string& text);
    std::vector<std::string> finish(const GenerationOutcome& outcome);

private:
    OpenAIChatResponseIdentity identity_;
    std::string reasoning_;
    std::string content_;
    bool include_usage_   = false;
    bool started_         = false;
    bool content_started_ = false;
    bool finished_        = false;
};

} // namespace ninfer::serve
