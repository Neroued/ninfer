#pragma once

// OpenAI wire-format layer: parses request JSON into the internal GenerationRequest
// and serializes internal results back into OpenAI Chat Completions bodies/chunks.
// This layer knows nothing about the engine; it only speaks the OpenAI schema.

#include "serve/request.h"
#include "serve/serve_options.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <string>

namespace ninfer::serve {

// ApiError, ApiException, RequestLimits, and CompletionUsage are the wire-format
// independent request/error types; they live in request.h and are shared by the
// OpenAI and Anthropic schema layers.

// Parse an already-decoded JSON body into a GenerationRequest. Throws ApiException
// on malformed or unsupported requests (n>1, tools, non-text response_format, ...).
// `default_model_id` fills in when the request omits `model` (clients such as the
// llama.cpp webui run a single loaded model and do not send it).
GenerationRequest parse_chat_completion_request(const nlohmann::json& body,
                                                const RequestLimits& limits,
                                                const std::string& default_model_id = {});

// llama.cpp webui dialect, mapped onto the public thinking controls:
// chat_template_kwargs.enable_thinking and reasoning_effort=low|medium are
// accepted instead of being rejected. `conflict_param` names the source field of
// any clash with `reasoning_effort` ("enable_thinking" or "reasoning_effort").
void parse_openai_chat_thinking(const nlohmann::json& body,
                                std::optional<bool>* enable_thinking,
                                std::optional<RequestedReasoningEffort>* reasoning_effort,
                                std::string* reasoning_effort_param,
                                const std::string& conflict_param);

std::optional<bool> parse_openai_preserve_thinking(const nlohmann::json& body);

// llama.cpp webui dialect: /props payload derived from the process configuration.
// The webui probes it for role, context size, default params, and the chat
// template; only process-level overrides are reported, everything else is
// neutral so client defaults never swallow a user-set request parameter.
nlohmann::json make_props_stub(const ServeOptions& options,
                               const std::string& model_id);

// Non-streaming chat completion response body (JSON string). When `reasoning` is
// non-empty it is attached as `message.reasoning_content` (the DeepSeek/vLLM-style
// convention consumed by Chatbox, Open WebUI, etc.), leaving `content` = answer.
std::string make_chat_completion_response(const std::string& id, const std::string& model,
                                          std::int64_t created, const std::string& content,
                                          const std::string& reasoning, const char* finish_reason,
                                          const CompletionUsage& usage);
std::string make_chat_completion_tool_response(const std::string& id, const std::string& model,
                                               std::int64_t created, const std::string& content,
                                               const std::string& reasoning,
                                               const std::vector<ToolCall>& tool_calls,
                                               const CompletionUsage& usage);

// Streaming SSE event strings ("data: {...}\n\n"). The first chunk carries the
// assistant role; reasoning chunks carry `reasoning_content` deltas (the <think>
// block), content chunks carry `content` deltas; the final chunk carries the
// finish_reason with an empty delta. Per the OpenAI stream_options contract, when
// usage reporting is enabled every content-bearing chunk carries `usage: null`
// and a single dedicated usage chunk (empty choices) is emitted before [DONE];
// pass include_usage accordingly.
std::string make_chat_chunk_role(const std::string& id, const std::string& model,
                                 std::int64_t created, bool include_usage);
std::string make_chat_chunk_reasoning(const std::string& id, const std::string& model,
                                      std::int64_t created, const std::string& delta_text,
                                      bool include_usage);
std::string make_chat_chunk_content(const std::string& id, const std::string& model,
                                    std::int64_t created, const std::string& delta_text,
                                    bool include_usage);
std::string make_chat_chunk_tool_calls(const std::string& id, const std::string& model,
                                       std::int64_t created,
                                       const std::vector<ToolCall>& tool_calls, bool include_usage);
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

} // namespace ninfer::serve
