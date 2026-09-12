#pragma once

// OpenAI Chat Completions wire adapter. Parsing produces one protocol envelope plus an executable
// GenerationRequest; response builders consume protocol-neutral GenerationOutcome values.

#include "serve/request.h"
#include "serve/request_json.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
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
    // llama.cpp-compatible response observations. Final timings remain unconditional;
    // timings_per_token controls only cumulative timing snapshots on streamed output chunks.
    bool timings_per_token = false;
    bool return_progress   = false;
};

OpenAIChatRequest parse_chat_completion_request(const RequestJson& body,
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
    OpenAIChatStream(OpenAIChatResponseIdentity identity, bool include_usage,
                     bool timings_per_token = false, bool return_progress = false);

    std::string start();
    void note_start(const ninfer::GenerationStart& start);
    std::string initial_prompt_progress();
    std::string prompt_progress(const ninfer::PromptProgress& progress);
    void note_timing(const ninfer::GenerationTimingObservation& timing);
    std::string reasoning_delta(const std::string& text);
    std::string content_delta(const std::string& text);
    std::vector<std::string> finish(const GenerationOutcome& outcome);

private:
    nlohmann::json live_timings_json() const;

    OpenAIChatResponseIdentity identity_;
    std::string reasoning_;
    std::string content_;
    std::optional<ninfer::GenerationTimingObservation> live_timing_;
    std::uint32_t prompt_tokens_            = 0;
    std::uint32_t cached_tokens_            = 0;
    std::uint32_t last_progress_tokens_     = 0;
    std::uint64_t last_progress_elapsed_ns_ = 0;
    bool include_usage_                     = false;
    bool timings_per_token_                 = false;
    bool return_progress_                   = false;
    bool started_                           = false;
    bool admitted_                          = false;
    bool progress_started_                  = false;
    bool content_started_                   = false;
    bool finished_                          = false;
};

} // namespace ninfer::serve
