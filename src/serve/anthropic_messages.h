#pragma once

// Anthropic Messages wire adapter. Request parsing lowers executable semantics into the common
// GenerationRequest; response construction owns Anthropic aggregate, SSE, usage, and error shapes.

#include "serve/request.h"
#include "serve/anthropic_thinking_signature.h"

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <vector>

namespace ninfer::serve {

struct GenerationOutcome;

struct AnthropicMessagesRequest {
    std::string model;
    GenerationRequest generation;
    bool stream                 = false;
    bool output_tokens_explicit = false;
};

struct AnthropicCountTokensRequest {
    std::string model;
    GenerationRequest generation;
};

AnthropicMessagesRequest parse_anthropic_messages_request(const nlohmann::json& body,
                                                          const RequestLimits& limits,
                                                          const AnthropicThinkingSigner& signer);
AnthropicCountTokensRequest
parse_anthropic_count_tokens_request(const nlohmann::json& body,
                                     const AnthropicThinkingSigner& signer);

struct AnthropicResponseIdentity {
    std::string request_id;
    std::string message_id;
    std::string model;
};

std::string new_anthropic_request_id();
AnthropicResponseIdentity make_anthropic_response_identity(std::string request_id,
                                                           std::string model);

ApiError normalize_anthropic_error(ApiError error);
std::string make_anthropic_error_body(const ApiError& error, const std::string& request_id);
std::string make_anthropic_sse_error(const ApiError& error, const std::string& request_id);

std::string make_anthropic_messages_response(const AnthropicResponseIdentity& identity,
                                             const GenerationOutcome& outcome,
                                             const AnthropicThinkingSigner& signer);
std::string make_anthropic_count_tokens_response(int input_tokens);

class AnthropicMessagesStream {
public:
    AnthropicMessagesStream(AnthropicResponseIdentity identity, int input_tokens,
                            AnthropicThinkingSigner signer);

    // The Engine start event is exact for normal streams. The no-argument form is reserved for an
    // error raised before admission, so an Anthropic error event still has a valid stream prefix.
    std::string start();
    std::string start(const ninfer::GenerationStart& generation);

    [[nodiscard]] bool started() const noexcept { return started_; }

    std::vector<std::string> reasoning_delta(const std::string& text);
    std::vector<std::string> content_delta(const std::string& text);
    std::vector<std::string> finish(const GenerationOutcome& outcome);
    std::string error(const ApiError& api_error) const;

private:
    std::string start_with_cache(std::optional<int> cache_read_input_tokens);
    std::vector<std::string> close_thinking();
    std::vector<std::string> close_text();

    AnthropicResponseIdentity identity_;
    AnthropicThinkingSigner signer_;
    std::string reasoning_;
    std::string content_;
    int input_tokens_    = 0;
    int next_index_      = 0;
    int thinking_index_  = -1;
    int text_index_      = -1;
    bool started_        = false;
    bool finished_       = false;
    bool thinking_open_  = false;
    bool signature_sent_ = false;
    bool text_open_      = false;
};

} // namespace ninfer::serve
