#include "serve/translate.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::serve {
namespace {

std::uint64_t random_seed() {
    static thread_local std::mt19937_64 rng(std::random_device{}());
    return rng();
}

[[noreturn]] void invalid_sampling(std::string message, std::string param) {
    ApiError error;
    error.message = std::move(message);
    error.param   = std::move(param);
    throw ApiException(std::move(error));
}

[[noreturn]] void invalid_prompt_option(std::string message, std::string param, std::string code) {
    ApiError error;
    error.message = std::move(message);
    error.param   = std::move(param);
    error.code    = std::move(code);
    throw ApiException(std::move(error));
}

ninfer::SamplingOverrides resolve_sampling_overrides(const SamplingParams& request,
                                                     const ServeOptions& server) {
    ninfer::SamplingOverrides sampling = server.sampling_overrides;
    if (request.temperature) { sampling.temperature = static_cast<float>(*request.temperature); }
    if (request.top_p) { sampling.top_p = static_cast<float>(*request.top_p); }
    if (request.top_k) { sampling.top_k = static_cast<std::int32_t>(*request.top_k); }
    if (request.presence_penalty) {
        sampling.presence_penalty = static_cast<float>(*request.presence_penalty);
    }
    if (request.frequency_penalty) {
        sampling.frequency_penalty = static_cast<float>(*request.frequency_penalty);
    }
    if (request.seed) {
        sampling.seed = *request.seed;
    } else if (server.sampling_overrides.seed) {
        sampling.seed = *server.sampling_overrides.seed;
    } else {
        sampling.seed = random_seed();
    }

    const auto finite = [](const std::optional<float>& value) {
        return !value || std::isfinite(*value);
    };
    if (!finite(sampling.temperature) || !finite(sampling.top_p) || !finite(sampling.min_p) ||
        !finite(sampling.presence_penalty) || !finite(sampling.frequency_penalty)) {
        invalid_sampling("sampling parameters must be finite", "sampling");
    }
    if (sampling.temperature && (*sampling.temperature < 0.0F || *sampling.temperature > 2.0F)) {
        invalid_sampling("temperature must be in [0,2]", "temperature");
    }
    if (sampling.top_p && (*sampling.top_p < 0.0F || *sampling.top_p > 1.0F)) {
        invalid_sampling("top_p must be in [0,1]", "top_p");
    }
    if (sampling.top_k && *sampling.top_k < 0) {
        invalid_sampling("top_k must be nonnegative", "top_k");
    }
    if (sampling.min_p && (*sampling.min_p < 0.0F || *sampling.min_p > 1.0F)) {
        invalid_sampling("min_p must be in [0,1]", "min_p");
    }
    if (sampling.presence_penalty &&
        (*sampling.presence_penalty < -2.0F || *sampling.presence_penalty > 2.0F)) {
        invalid_sampling("presence_penalty must be in [-2,2]", "presence_penalty");
    }
    if (sampling.frequency_penalty &&
        (*sampling.frequency_penalty < -2.0F || *sampling.frequency_penalty > 2.0F)) {
        invalid_sampling("frequency_penalty must be in [-2,2]", "frequency_penalty");
    }
    if (server.greedy) { sampling.temperature = 0.0F; }
    return sampling;
}

std::vector<std::string> effective_tool_jsons(const GenerationRequest& request) {
    std::vector<std::string> tools;
    if (!request.uses_tools()) { return tools; }
    if (request.tool_choice.mode == ToolChoiceMode::Named) {
        for (const ToolDefinition& tool : request.tools) {
            if (tool.name == request.tool_choice.name) {
                tools.push_back(tool.definition_json);
                break;
            }
        }
        return tools;
    }
    tools.reserve(request.tools.size());
    for (const ToolDefinition& tool : request.tools) { tools.push_back(tool.definition_json); }
    return tools;
}

} // namespace

ResolvedPromptSemantics resolve_prompt_semantics(const GenerationRequest& request,
                                                 const ServeOptions& server,
                                                 const ninfer::PromptCapabilities& capabilities) {
    ResolvedPromptSemantics result{
        .enable_thinking   = request.enable_thinking.value_or(server.enable_thinking),
        .reasoning_effort  = std::nullopt,
        .preserve_thinking = request.preserve_thinking.value_or(server.preserve_thinking),
    };
    if (!request.reasoning_effort) { return result; }

    const RequestedReasoningEffort requested = *request.reasoning_effort;
    const bool enables_thinking              = requested != RequestedReasoningEffort::None;
    if (request.enable_thinking && *request.enable_thinking != enables_thinking) {
        invalid_prompt_option("reasoning effort conflicts with enable_thinking",
                              request.reasoning_effort_param, "conflicting_template_option");
    }
    result.enable_thinking = enables_thinking;

    if (requested == RequestedReasoningEffort::None) {
        if (!capabilities.enable_thinking) {
            invalid_prompt_option("the loaded chat template cannot disable thinking",
                                  request.reasoning_effort_param, "reasoning_effort_not_supported");
        }
        return result;
    }

    switch (requested) {
    case RequestedReasoningEffort::Low:
        result.reasoning_effort = ninfer::ReasoningEffort::Low;
        break;
    case RequestedReasoningEffort::Medium:
        result.reasoning_effort = ninfer::ReasoningEffort::Medium;
        break;
    case RequestedReasoningEffort::XHigh:
        result.reasoning_effort = ninfer::ReasoningEffort::XHigh;
        break;
    case RequestedReasoningEffort::Minimal:
    case RequestedReasoningEffort::High:
    case RequestedReasoningEffort::Max:
        invalid_prompt_option("reasoning effort '" +
                                  std::string(requested_reasoning_effort_name(requested)) +
                                  "' is not supported by the loaded chat template",
                              request.reasoning_effort_param, "reasoning_effort_not_supported");
    case RequestedReasoningEffort::None:
        break;
    }

    if (!capabilities.reasoning_effort.supports(*result.reasoning_effort)) {
        invalid_prompt_option("reasoning effort '" +
                                  std::string(requested_reasoning_effort_name(requested)) +
                                  "' is not supported by the loaded chat template",
                              request.reasoning_effort_param, "reasoning_effort_not_supported");
    }
    return result;
}

// JSON response formats have no token-level constraint in this engine, so they
// are enforced by an instruction injected into the prompt — the same soft-JSON
// approach llama.cpp pairs with its json_object grammar. JsonObject asks for a
// bare object; JsonSchema additionally carries the client's schema.
std::string structured_output_instruction(const StructuredOutput& output) {
    std::string text = "You must respond with a single JSON object and nothing else: "
                       "no prose, no explanations, no markdown code fences.";
    if (output.type == StructuredOutputType::JsonSchema) {
        text += " The object must conform to this JSON schema:\n" + output.schema_json;
    }
    return text;
}

ninfer::PromptInput to_prompt_input(const GenerationRequest& request,
                                    const ResolvedPromptSemantics& semantics,
                                    const MediaAcquirer& acquire_media) {
    std::vector<ChatTurn> turns = request.messages;
    const StructuredOutput& output = request.structured_output;
    if (output.type == StructuredOutputType::JsonObject ||
        output.type == StructuredOutputType::JsonSchema) {
        ContentPart instruction;
        instruction.kind = ContentKind::Text;
        instruction.text = structured_output_instruction(output);
        if (!turns.empty() && turns.front().role == ChatRole::System) {
            turns.front().content.push_back(std::move(instruction));
        } else {
            ChatTurn system;
            system.role    = ChatRole::System;
            system.content = std::vector<ContentPart>{std::move(instruction)};
            turns.insert(turns.begin(), std::move(system));
        }
    }

    ninfer::PromptInput input;
    input.messages.reserve(turns.size());
    for (const ChatTurn& turn : turns) {
        ninfer::ChatMessage message;
        message.role              = turn.role;
        message.reasoning_content = turn.reasoning_content;
        message.tool_call_id      = turn.tool_call_id;
        message.tool_calls.reserve(turn.tool_calls.size());
        for (const ToolCall& call : turn.tool_calls) {
            message.tool_calls.push_back(ninfer::ToolCall{call.id, call.name, call.arguments_json});
        }

        for (const ContentPart& part : turn.content) {
            if (part.kind == ContentKind::Text) {
                if (!message.parts.empty() && !part.text.empty() &&
                    message.parts.back().kind == ninfer::MessagePartKind::Text) {
                    ninfer::MessagePart newline;
                    newline.text = "\n";
                    message.parts.push_back(std::move(newline));
                }
                ninfer::MessagePart text;
                text.text = part.text;
                message.parts.push_back(std::move(text));
                continue;
            }
            if (part.kind == ContentKind::Image || part.kind == ContentKind::Video) {
                if (!acquire_media) {
                    throw std::logic_error("media acquisition callback is not configured");
                }
                ninfer::MessagePart media;
                media.kind  = ninfer::MessagePartKind::Media;
                media.media = acquire_media(part);
                message.parts.push_back(std::move(media));
                continue;
            }

            ApiError error;
            error.message = "content type '" + part.type_raw + "' is not supported";
            error.param   = "messages";
            error.code    = "modality_not_supported";
            throw ApiException(std::move(error));
        }
        input.messages.push_back(std::move(message));
    }

    input.options.add_generation_prompt = true;
    input.options.enable_thinking       = semantics.enable_thinking;
    input.options.reasoning_effort      = semantics.reasoning_effort;
    input.options.preserve_thinking     = semantics.preserve_thinking;
    input.options.add_vision_id         = false;
    input.options.tool_jsons            = effective_tool_jsons(request);
    return input;
}

ninfer::RequestOptions to_request_options(const GenerationRequest& request,
                                          const ServeOptions& server) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = static_cast<std::uint32_t>(request.max_tokens);
    options.execution.allow_prefix_reuse      = server.allow_prefix_reuse;
    options.execution.sampling             = resolve_sampling_overrides(request.sampling, server);
    options.output.raw                     = false;
    options.output.preserve_special_tokens = request.uses_tools() || request.has_tool_history();
    options.stop.strings.reserve(request.stop_strings.size());
    for (const std::string& stop : request.stop_strings) {
        if (!stop.empty()) {
            options.stop.strings.push_back(
                ninfer::StopString{.text              = stop,
                                   .channel           = ninfer::OutputChannel::Content,
                                   .include_in_output = false});
        }
    }
    return options;
}

const char* finish_reason_wire(ninfer::FinishReason reason) {
    switch (reason) {
    case ninfer::FinishReason::OutputLimit:
    case ninfer::FinishReason::ContextCapacity:
        return "length";
    case ninfer::FinishReason::None:
    case ninfer::FinishReason::StopToken:
    case ninfer::FinishReason::StopString:
    case ninfer::FinishReason::Cancelled:
        return "stop";
    }
    return "stop";
}

} // namespace ninfer::serve
