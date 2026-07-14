#pragma once

// Anthropic Messages API wire-format layer: parses /v1/messages request JSON into
// the internal GenerationRequest and serializes internal results back into
// Anthropic message bodies / SSE events. This is a sibling of openai_schema.h;
// both map to the same wire-agnostic GenerationRequest / GenerationOutcome, so the
// engine and generation service below know nothing about either protocol.

#include "serve/request.h"
#include "ninfer/types.h"

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace ninfer::serve {

// Parse an already-decoded Anthropic Messages body into a GenerationRequest.
// `system` becomes a leading system turn; user tool_result blocks become tool
// turns; assistant tool_use blocks become tool calls. Throws ApiException on
// malformed / unsupported requests. The `model` field is accepted verbatim (any
// Claude model name) and echoed back, never validated against the loaded model.
GenerationRequest parse_messages_request(const nlohmann::json& body, const RequestLimits& limits);

// Map an internal finish reason (+ whether tool calls were produced) onto the
// Anthropic stop_reason wire value.
const char* messages_stop_reason(ninfer::FinishReason reason, bool has_tool_calls);

// Non-streaming Messages response body (JSON string). Content blocks are emitted
// in order: an optional `thinking` block (from reasoning), an optional `text`
// block (from content), then a `tool_use` block per tool call. When nothing was
// produced an empty text block is emitted so `content` is never empty.
std::string make_messages_response(const std::string& id, const std::string& model,
                                   const std::string& content, const std::string& reasoning,
                                   const std::vector<ToolCall>& tool_calls, const char* stop_reason,
                                   const CompletionUsage& usage);

// Streaming SSE event strings ("event: <type>\ndata: {...}\n\n"). The transport
// drives the block state machine (open/close, running index) and calls these pure
// builders; thinking blocks precede the text block, tool_use blocks follow.
std::string make_message_start(const std::string& id, const std::string& model, int input_tokens);
std::string make_content_block_start_text(int index);
std::string make_content_block_start_thinking(int index);
std::string make_content_block_start_tool_use(int index, const ToolCall& call);
std::string make_content_block_delta_text(int index, const std::string& delta_text);
std::string make_content_block_delta_thinking(int index, const std::string& delta_text);
std::string make_content_block_delta_tool_json(int index, const std::string& partial_json);
std::string make_content_block_stop(int index);
std::string make_message_delta(const char* stop_reason, int output_tokens);
std::string make_message_stop();
std::string make_messages_ping();

// Error object body (Anthropic shape) and its SSE `event: error` form.
std::string make_messages_error_body(const ApiError& error);
std::string messages_sse_error_event(const ApiError& error);

// /v1/messages/count_tokens response body.
std::string make_count_tokens_response(int input_tokens);

// Message identifier ("msg_...").
std::string new_message_id();

} // namespace ninfer::serve
