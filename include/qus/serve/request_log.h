#pragma once

// Human-facing per-request console lines for qus-serve. Pure formatting: given a
// request's parameters and its GenerationOutcome, produce one log line. The HTTP
// layer owns the request id, the "qus-serve: " prefix, and serialization of
// writes across threads.

#include "qus/serve/generation_service.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace qus::serve {

// Logged right after a request is validated: what it asked for. `client_set`
// distinguishes a client-provided max_tokens from the server default.
std::string format_request_start(std::uint64_t id, bool stream, std::size_t n_messages,
                                 int max_tokens, bool client_set, std::size_t n_tools,
                                 const ToolChoice& tool_choice, bool has_tool_history);

// Logged when generation finishes (or is cancelled): tokens, TTFT, prefill/decode
// throughput, wall time, and speculative-decoding acceptance.
std::string format_request_done(std::uint64_t id, const GenerationOutcome& outcome);

// Logged when a request fails with an error before/while generating.
std::string format_request_error(std::uint64_t id, const std::string& message);

} // namespace qus::serve
