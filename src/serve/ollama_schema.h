#pragma once

// Ollama API wire-format layer: parses /api/chat request JSON into the internal
// GenerationRequest and serializes internal results back into Ollama chat bodies
// and NDJSON stream frames, plus the read-only model inventory endpoints Ollama
// clients probe (/api/tags, /api/show, /api/ps, /api/version). This is a sibling
// of openai_schema.h and anthropic_schema.h; all three map onto the same
// wire-agnostic GenerationRequest / GenerationOutcome, so the engine and the
// generation service know nothing about the protocol.
//
// The Ollama contract carries per-request phase timings in its terminal chat
// frame. Those are what Ollama-aware clients (Open WebUI and others) render as
// prompt and response token/s; the OpenAI and Anthropic surfaces have no
// standard equivalent.

#include "serve/request.h"
#include "ninfer/types.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::serve {

// Advertised Ollama API level. Ollama clients probe /api/version to decide which
// request fields they may send; the parser below understands the chat contract
// at this level (`think`, assistant `thinking`, tool_calls with object
// arguments, message `images`).
constexpr const char* kOllamaCompatVersion = "0.11.0";

// Parse an already-decoded Ollama /api/chat body into a GenerationRequest.
// `stream` defaults to true. `options` sampling keys map onto SamplingParams;
// sampler knobs the Engine does not implement (repeat_penalty, mirostat, ...)
// and non-null `format` are rejected, while Ollama runner/loader knobs
// (num_gpu, num_thread, keep_alive, ...) are accepted without effect because
// residency is a startup property of the server. `options.num_ctx` must not
// exceed `limits.max_context`. Throws ApiException on malformed or unsupported
// requests.
GenerationRequest parse_ollama_chat_request(const nlohmann::json& body,
                                            const RequestLimits& limits);

// Ollama defines POST /api/chat with an empty `messages` array as a model
// preload request that returns immediately with `done_reason: "load"`. Returns
// the requested model name when `body` has that shape; throws ApiException when
// `model` is missing or not a string. Callers still validate the model name.
std::optional<std::string> ollama_preload_model(const nlohmann::json& body);

// True when a client-supplied model name addresses the served model. Ollama
// clients normalize untagged names to `name:latest`, so that spelling is
// accepted alongside the bare id advertised by /api/tags.
bool ollama_model_matches(const std::string& requested, const std::string& model_id);

// Read-only model facts rendered by /api/tags, /api/show, and /api/ps.
struct OllamaModelFacts {
    std::string model_id;              // public model id, also the Ollama model name
    std::string target;                // registered target (family)
    std::string weights_id;            // weight profile, reported as quantization_level
    std::string modified_at;           // RFC 3339 artifact timestamp
    std::uint64_t size_bytes      = 0; // artifact bytes
    std::uint64_t size_vram_bytes = 0; // resident device bytes
    std::uint32_t max_context     = 0;
    bool vision                   = false;
    bool thinking                 = false;
};

// Terminal timing fields of an Ollama chat response. Durations are wall seconds
// and are serialized as integer nanoseconds. `load_seconds` is always zero for
// a resident model but the field is part of the contract.
struct OllamaTimings {
    double total_seconds       = 0.0;
    double load_seconds        = 0.0;
    double prompt_eval_seconds = 0.0; // computed-prefill phase
    double eval_seconds        = 0.0; // decode phase
    int prompt_eval_count      = 0; // prompt tokens actually computed (excludes prefix-cache hits)
    int eval_count             = 0; // accepted generated tokens
};

// Non-streaming /api/chat response body (one JSON object with the complete
// message, `done: true`, `done_reason`, and timings).
std::string make_ollama_chat_response(const std::string& model, const std::string& created_at,
                                      const std::string& content, const std::string& thinking,
                                      const std::vector<ToolCall>& tool_calls,
                                      const char* done_reason, const OllamaTimings& timings);

// Reply to a preload request: an empty assistant message with `done: true` and
// `done_reason: "load"`. The model is always resident, so no work is done.
std::string make_ollama_chat_load_response(const std::string& model, const std::string& created_at);

// NDJSON stream frames ("{...}\n"). Content and thinking deltas are separate
// frames; tool calls are emitted in one frame once they are complete; the final
// frame carries an empty message, `done: true`, `done_reason`, and timings.
std::string make_ollama_chat_content_frame(const std::string& model, const std::string& created_at,
                                           const std::string& delta_text);
std::string make_ollama_chat_thinking_frame(const std::string& model, const std::string& created_at,
                                            const std::string& delta_text);
std::string make_ollama_chat_tool_calls_frame(const std::string& model,
                                              const std::string& created_at,
                                              const std::vector<ToolCall>& tool_calls);
std::string make_ollama_chat_done_frame(const std::string& model, const std::string& created_at,
                                        const char* done_reason, const OllamaTimings& timings);

// Inventory endpoints.
std::string make_ollama_tags(const OllamaModelFacts& facts);
std::string make_ollama_show(const OllamaModelFacts& facts);
std::string make_ollama_ps(const OllamaModelFacts& facts);
std::string make_ollama_version();

// Error body ({"error": "..."}) shared by HTTP errors and mid-stream error frames.
std::string make_ollama_error_body(const ApiError& error);
std::string make_ollama_error_frame(const ApiError& error);

// RFC 3339 UTC timestamp with nanosecond precision, the Ollama `created_at` shape.
std::string ollama_timestamp_now();
std::string ollama_timestamp(std::int64_t unix_nanoseconds);

} // namespace ninfer::serve
