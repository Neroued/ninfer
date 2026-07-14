#pragma once

// Human-facing per-request console lines for ninfer-serve. Pure formatting: given a
// request's parameters and its GenerationOutcome, produce one log line. The HTTP
// layer owns the request id, the "ninfer-serve: " prefix, and serialization of
// writes across threads.

#include "ninfer/kernels/sampling.h"
#include "ninfer/serve/generation_service.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace ninfer::serve {

// Logged right after a request is validated. `requested_max_tokens` is the
// client/default upper bound and `effective_max_tokens` is the context-fitted
// value. `client_set` distinguishes a client-provided value from the server
// default. `sampling` is the resolved per-request sampler config.
std::string format_request_start(std::uint64_t id, bool stream, std::size_t n_messages,
                                 int requested_max_tokens, int effective_max_tokens,
                                 bool client_set, std::size_t n_tools,
                                 const ToolChoice& tool_choice, bool has_tool_history,
                                 const ninfer::kernels::SamplingConfig& sampling);

// Logged when generation finishes (or is cancelled): tokens, TTFT, prefill/decode
// throughput, wall time, and speculative-decoding acceptance.
std::string format_request_done(std::uint64_t id, const GenerationOutcome& outcome);

// Logged when a request fails with an error before/while generating.
std::string format_request_error(std::uint64_t id, const std::string& message);

} // namespace ninfer::serve
