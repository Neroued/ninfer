#pragma once

// Adapter between the wire-independent request and the structured Qwen chat model.

#include "qus/serve/request.h"
#include "qus/serve/serve_options.h"
#include "qus/text/chat_template.h"
#include "qus/text/text_runner.h"

#include <vector>

namespace qus::serve {

// Preserve text/image/video part order while mapping wire roles and tool calls.
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
