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

ninfer::SamplingParameters resolve_sampling(const SamplingParams& request,
                                            const ServeOptions& server) {
    ninfer::SamplingParameters sampling;
    sampling.temperature =
        static_cast<float>(request.temperature.value_or(server.sampling_temperature));
    sampling.top_p = static_cast<float>(request.top_p.value_or(server.sampling_top_p));
    sampling.top_k = request.top_k.value_or(server.sampling_top_k);
    sampling.presence_penalty =
        static_cast<float>(request.presence_penalty.value_or(server.sampling_presence_penalty));
    sampling.frequency_penalty =
        static_cast<float>(request.frequency_penalty.value_or(server.sampling_frequency_penalty));
    sampling.seed = request.seed.value_or(server.sampling_seed.value_or(random_seed()));

    if (!std::isfinite(sampling.temperature) || !std::isfinite(sampling.top_p) ||
        !std::isfinite(sampling.min_p) || !std::isfinite(sampling.presence_penalty) ||
        !std::isfinite(sampling.frequency_penalty)) {
        invalid_sampling("sampling parameters must be finite", "sampling");
    }
    if (sampling.temperature < 0.0F || sampling.temperature > 2.0F) {
        invalid_sampling("temperature must be in [0,2]", "temperature");
    }
    if (sampling.top_p < 0.0F || sampling.top_p > 1.0F) {
        invalid_sampling("top_p must be in [0,1]", "top_p");
    }
    if (sampling.top_k < 0) { invalid_sampling("top_k must be nonnegative", "top_k"); }
    if (sampling.presence_penalty < -2.0F || sampling.presence_penalty > 2.0F) {
        invalid_sampling("presence_penalty must be in [-2,2]", "presence_penalty");
    }
    if (sampling.frequency_penalty < -2.0F || sampling.frequency_penalty > 2.0F) {
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

std::string normalized_role(const std::string& role) {
    if (role == "developer") { return "system"; }
    if (role == "system" || role == "user" || role == "assistant" || role == "tool") {
        return role;
    }
    ApiError error;
    error.message = "unsupported role: " + role;
    error.param   = "messages";
    error.code    = "unsupported_role";
    throw ApiException(std::move(error));
}

} // namespace

ninfer::PromptInput to_prompt_input(const GenerationRequest& request, const ServeOptions& server,
                                    const MediaAcquirer& acquire_media) {
    ninfer::PromptInput input;
    input.messages.reserve(request.messages.size());
    for (const ChatTurn& turn : request.messages) {
        ninfer::ChatMessage message;
        message.role              = normalized_role(turn.role);
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
    input.options.enable_thinking       = request.enable_thinking.value_or(server.enable_thinking);
    input.options.preserve_thinking     = false;
    input.options.add_vision_id         = false;
    input.options.tool_jsons            = effective_tool_jsons(request);
    return input;
}

ninfer::RequestOptions to_request_options(const GenerationRequest& request,
                                          const ServeOptions& server) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = static_cast<std::uint32_t>(request.max_tokens);
    options.execution.allow_prefix_reuse      = true;
    options.execution.sampling                = resolve_sampling(request.sampling, server);
    options.output.raw                        = false;
    options.output.preserve_special_tokens    = request.uses_tools() || request.has_tool_history();
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
