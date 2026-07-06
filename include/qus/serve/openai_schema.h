#pragma once

// OpenAI wire-format layer: parses request JSON into the internal GenerationRequest
// and serializes internal results back into OpenAI Chat Completions bodies/chunks.
// This layer knows nothing about the engine; it only speaks the OpenAI schema.

#include "qus/serve/request.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace qus::serve {

// A structured API error mapped 1:1 onto an OpenAI error object + HTTP status.
struct ApiError {
    int status          = 400;
    std::string type    = "invalid_request_error";
    std::string message;
    std::string param;  // optional
    std::string code;   // optional
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

// Parse an already-decoded JSON body into a GenerationRequest. Throws ApiException
// on malformed or unsupported requests (n>1, tools, non-text response_format, ...).
GenerationRequest parse_chat_completion_request(const nlohmann::json& body,
                                                const RequestLimits& limits);

// Non-streaming chat completion response body (JSON string).
std::string make_chat_completion_response(const std::string& id, const std::string& model,
                                          std::int64_t created, const std::string& content,
                                          const char* finish_reason, const CompletionUsage& usage);

// Streaming SSE event strings ("data: {...}\n\n"). The first chunk carries the
// assistant role; content chunks carry text deltas; the final chunk carries the
// finish_reason with an empty delta. Per the OpenAI stream_options contract, when
// usage reporting is enabled every content-bearing chunk carries `usage: null`
// and a single dedicated usage chunk (empty choices) is emitted before [DONE];
// pass include_usage accordingly.
std::string make_chat_chunk_role(const std::string& id, const std::string& model,
                                 std::int64_t created, bool include_usage);
std::string make_chat_chunk_content(const std::string& id, const std::string& model,
                                     std::int64_t created, const std::string& delta_text,
                                     bool include_usage);
std::string make_chat_chunk_final(const std::string& id, const std::string& model,
                                   std::int64_t created, const char* finish_reason,
                                   bool include_usage);
// Dedicated usage chunk: `choices: []` with the request's token usage. Emitted
// only when stream_options.include_usage is true.
std::string make_chat_chunk_usage(const std::string& id, const std::string& model,
                                  std::int64_t created, const CompletionUsage& usage);
std::string sse_done();

// /v1/models payloads.
std::string make_models_list(const std::string& model_id, std::int64_t created);
std::string make_model_object(const std::string& model_id, std::int64_t created);

// Error object body.
std::string make_error_body(const ApiError& error);

// Identifiers / timestamps.
std::string new_chat_completion_id();
std::int64_t unix_time_now();

} // namespace qus::serve
