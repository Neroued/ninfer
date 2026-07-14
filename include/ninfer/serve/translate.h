#pragma once

// Adapter between the wire-independent request and the structured Qwen chat model.

#include "ninfer/serve/request.h"
#include "ninfer/serve/serve_options.h"
#include "ninfer/text/chat_template.h"
#include "ninfer/text/text_runner.h"

#include <vector>

namespace ninfer::serve {

// Preserve text/image/video part order while mapping wire roles and tool calls.
std::vector<ninfer::text::ChatMessage> to_chat_messages(const GenerationRequest& req);

// Build runner options (max_new_tokens, thinking, clean output, sampler). The
// sampler is resolved from the request's SamplingParams over the server defaults;
// --greedy on the server forces exact argmax regardless of the request.
ninfer::text::TextGenerationOptions to_generation_options(const GenerationRequest& req,
                                                       const ServeOptions& server);

// Map an internal finish reason onto the OpenAI wire value. Cancelled maps to
// "stop" (a disconnected client is not an error state on the wire).
const char* finish_reason_wire(ninfer::text::FinishReason reason);

} // namespace ninfer::serve
