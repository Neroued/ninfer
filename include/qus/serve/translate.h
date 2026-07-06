#pragma once

// Adapter between the internal GenerationRequest and the qus_core text runner.
// This is the single place that flattens structured content into the string the
// Qwen chat template consumes and that rejects not-yet-supported modalities, so
// adding images later means extending only this layer (and the template).

#include "qus/serve/request.h"
#include "qus/serve/serve_options.h"
#include "qus/text/chat_template.h"
#include "qus/text/text_runner.h"

#include <vector>

namespace qus::serve {

// Flatten each turn's text parts into a ChatMessage. Throws ApiException for
// unsupported roles or non-text content.
std::vector<qus::text::ChatMessage> to_chat_messages(const GenerationRequest& req);

// Build runner options (max_new_tokens, thinking, clean output, sampler). The
// sampler is resolved from the request's SamplingParams over the server defaults;
// --greedy on the server forces exact argmax regardless of the request.
qus::text::TextGenerationOptions to_generation_options(const GenerationRequest& req,
                                                       const ServeOptions& server);

// Map an internal finish reason onto the OpenAI wire value. Cancelled maps to
// "stop" (a disconnected client is not an error state on the wire).
const char* finish_reason_wire(qus::text::FinishReason reason);

} // namespace qus::serve
