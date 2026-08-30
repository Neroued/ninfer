#include "serve/translate.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
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
    if (request.min_p) { sampling.min_p = static_cast<float>(*request.min_p); }
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
    if (sampling.top_k && (*sampling.top_k < 0 || *sampling.top_k > 20)) {
        invalid_sampling("top_k must be in [0,20]", "top_k");
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

std::vector<const ToolDefinition*> effective_tools(const GenerationRequest& request) {
    std::vector<const ToolDefinition*> tools;
    if (!request.uses_tools()) { return tools; }
    tools.reserve(request.tools.size());
    for (const ToolDefinition& tool : request.tools) { tools.push_back(&tool); }
    return tools;
}

std::string render_tool_definition(const ToolDefinition& tool) {
    using Json  = nlohmann::json;
    Json schema = Json::parse(tool.input_schema_json);
    Json function{{"name", tool.name}, {"parameters", std::move(schema)}, {"strict", false}};
    if (!tool.description.empty()) { function["description"] = tool.description; }
    if (tool.input_examples_json) {
        function["input_examples"] = Json::parse(*tool.input_examples_json);
    }
    return Json{{"type", "function"}, {"function", std::move(function)}}.dump();
}

} // namespace

ResolvedPromptSemantics resolve_prompt_semantics(const GenerationRequest& request,
                                                 const ServeOptions& server,
                                                 const ninfer::PromptCapabilities& capabilities) {
    ResolvedPromptSemantics result{
        .enable_thinking            = request.enable_thinking.value_or(server.enable_thinking),
        .reasoning_effort           = std::nullopt,
        .effective_reasoning_effort = std::nullopt,
        .preserve_thinking          = request.preserve_thinking.value_or(server.preserve_thinking),
    };
    const auto complete = [&]() {
        if (request.continuation == ninfer::PromptContinuationMode::ContinueFinalAssistant &&
            result.enable_thinking) {
            invalid_prompt_option("assistant prefill cannot be combined with enabled thinking",
                                  "messages", "assistant_prefill_not_supported");
        }
        if (result.enable_thinking) {
            result.effective_reasoning_effort = result.reasoning_effort
                                                    ? result.reasoning_effort
                                                    : capabilities.reasoning_effort.default_effort;
        }
        return result;
    };
    if (!request.reasoning_effort) { return complete(); }

    const RequestedReasoningEffort requested = *request.reasoning_effort;
    const bool enables_thinking              = requested != RequestedReasoningEffort::None;
    if (request.enable_thinking && *request.enable_thinking != enables_thinking) {
        invalid_prompt_option("reasoning effort conflicts with enable_thinking", "reasoning_effort",
                              "conflicting_template_option");
    }
    result.enable_thinking = enables_thinking;

    if (requested == RequestedReasoningEffort::None) {
        if (!capabilities.enable_thinking) {
            invalid_prompt_option("the loaded chat template cannot disable thinking",
                                  "reasoning_effort", "reasoning_effort_not_supported");
        }
        return complete();
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
                              "reasoning_effort", "reasoning_effort_not_supported");
    case RequestedReasoningEffort::None:
        break;
    }

    if (!capabilities.reasoning_effort.supports(*result.reasoning_effort)) {
        invalid_prompt_option("reasoning effort '" +
                                  std::string(requested_reasoning_effort_name(requested)) +
                                  "' is not supported by the loaded chat template",
                              "reasoning_effort", "reasoning_effort_not_supported");
    }
    return complete();
}

ninfer::PromptInput to_prompt_input(const GenerationRequest& request,
                                    const ResolvedPromptSemantics& semantics,
                                    const MediaAcquirer& acquire_media) {
    ninfer::PromptInput input;
    input.messages.reserve(request.messages.size());
    for (std::size_t turn_index = 0; turn_index < request.messages.size(); ++turn_index) {
        const ChatTurn& turn = request.messages[turn_index];
        ninfer::ChatMessage message;
        message.role              = turn.role;
        message.reasoning_content = turn.reasoning_content;
        message.tool_call_id      = turn.tool_call_id;
        message.tool_calls.reserve(turn.tool_calls.size());
        for (const ToolCall& call : turn.tool_calls) {
            message.tool_calls.push_back(ninfer::ToolCall{call.id, call.name, call.arguments_json});
        }

        if (turn.role == ChatRole::Tool && turn.tool_result_is_error) {
            ninfer::MessagePart error;
            error.text = "[tool_error]\n";
            message.parts.push_back(std::move(error));
        }

        std::uint64_t text_bytes = 0;
        for (std::size_t part_index = 0; part_index < turn.content.size(); ++part_index) {
            const ContentPart& part = turn.content[part_index];
            if (part.kind == ContentKind::Text) {
                ninfer::MessagePart text;
                text.text = part.text;
                message.parts.push_back(std::move(text));
                if (part.text.size() > std::numeric_limits<std::uint32_t>::max() - text_bytes) {
                    throw std::invalid_argument("cacheable instruction text exceeds uint32");
                }
                text_bytes += part.text.size();
            } else if (part.kind == ContentKind::Image || part.kind == ContentKind::Video) {
                if (!acquire_media) {
                    throw std::logic_error("media acquisition callback is not configured");
                }
                ninfer::MessagePart media;
                media.kind  = ninfer::MessagePartKind::Media;
                media.media = acquire_media(part);
                message.parts.push_back(std::move(media));
            } else {
                ApiError error;
                error.message = "content type '" + part.type_raw + "' is not supported";
                error.param   = "messages";
                error.code    = "modality_not_supported";
                throw ApiException(std::move(error));
            }
            if (part.cache_boundary_after) {
                if (turn_index >= std::numeric_limits<std::uint32_t>::max() ||
                    part_index >= std::numeric_limits<std::uint32_t>::max()) {
                    throw std::overflow_error("conversation cache boundary exceeds uint32");
                }
                const bool leading_instruction =
                    turn_index == 0 &&
                    (turn.role == ChatRole::System || turn.role == ChatRole::Developer) &&
                    part.kind == ContentKind::Text;
                if (leading_instruction) {
                    input.context_cache.markers.push_back(ninfer::PromptCacheMarker{
                        .kind     = part.cache_boundary_after->kind,
                        .evidence = part.cache_boundary_after->evidence,
                        .location = ninfer::PromptCacheMarkerLocation::LeadingInstructionBoundary,
                        .leading_instruction_bytes = static_cast<std::uint32_t>(text_bytes),
                    });
                } else {
                    input.context_cache.markers.push_back(ninfer::PromptCacheMarker{
                        .after_message_count = static_cast<std::uint32_t>(turn_index + 1U),
                        .kind                = part.cache_boundary_after->kind,
                        .evidence            = part.cache_boundary_after->evidence,
                        .location = ninfer::PromptCacheMarkerLocation::MessagePartBoundary,
                        .after_message_part_count =
                            static_cast<std::uint32_t>(message.parts.size()),
                    });
                }
            }
        }
        input.messages.push_back(std::move(message));
        if (turn.cache_boundary_after) {
            if (input.messages.size() > std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error("conversation cache boundary exceeds uint32");
            }
            input.context_cache.markers.push_back(ninfer::PromptCacheMarker{
                .after_message_count = static_cast<std::uint32_t>(input.messages.size()),
                .kind                = turn.cache_boundary_after->kind,
                .evidence            = turn.cache_boundary_after->evidence,
                .location            = ninfer::PromptCacheMarkerLocation::MessageBoundary,
            });
        }
    }

    input.options.continuation                     = request.continuation;
    input.options.enable_thinking                  = semantics.enable_thinking;
    input.options.reasoning_effort                 = semantics.reasoning_effort;
    input.options.preserve_thinking                = semantics.preserve_thinking;
    input.options.add_vision_id                    = false;
    const std::vector<const ToolDefinition*> tools = effective_tools(request);
    input.options.tool_jsons.reserve(tools.size());
    for (std::size_t index = 0; index < tools.size(); ++index) {
        input.options.tool_jsons.push_back(render_tool_definition(*tools[index]));
        if (tools[index]->cache_boundary_after) {
            input.context_cache.markers.push_back(ninfer::PromptCacheMarker{
                .kind             = tools[index]->cache_boundary_after->kind,
                .evidence         = tools[index]->cache_boundary_after->evidence,
                .location         = ninfer::PromptCacheMarkerLocation::ToolBoundary,
                .after_tool_count = static_cast<std::uint32_t>(index + 1U),
            });
        }
    }
    input.context_cache.allow_engine_automatic_shared_prefixes =
        request.allow_engine_automatic_shared_prefixes;
    return input;
}

ninfer::RequestOptions to_request_options(const GenerationRequest& request,
                                          const ServeOptions& server,
                                          const ResolvedPromptSemantics& semantics,
                                          bool allow_prefix_reuse) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = static_cast<std::uint32_t>(request.max_tokens);
    options.execution.allow_prefix_reuse      = allow_prefix_reuse;
    if (semantics.enable_thinking) {
        options.execution.thinking.budget =
            request.thinking_budget ? request.thinking_budget : server.default_thinking_budget;
    }
    options.execution.sampling             = resolve_sampling_overrides(request.sampling, server);
    options.output.raw                     = false;
    options.output.preserve_special_tokens = request.uses_tools() || request.has_tool_history();
    options.output.tool_name_max_length = static_cast<std::uint32_t>(request.tool_name_max_length);
    options.stop.strings.reserve(request.stop_strings.size() *
                                 (request.stop_strings_apply_to_reasoning ? 2U : 1U));
    for (const std::string& stop : request.stop_strings) {
        if (!stop.empty()) {
            options.stop.strings.push_back(
                ninfer::StopString{.text              = stop,
                                   .channel           = ninfer::OutputChannel::Content,
                                   .include_in_output = false});
            if (request.stop_strings_apply_to_reasoning) {
                options.stop.strings.push_back(
                    ninfer::StopString{.text              = stop,
                                       .channel           = ninfer::OutputChannel::Reasoning,
                                       .include_in_output = false});
            }
        }
    }
    return options;
}

} // namespace ninfer::serve
