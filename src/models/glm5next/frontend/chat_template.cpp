#include "models/glm5next/frontend/chat_template.h"
#include "models/glm5next/frontend/prompt_layout.h"

#include "models/common/frontend/media_cache.h"
#include "text/unicode.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::glm5next {
namespace {
using Json = nlohmann::ordered_json;

Json template_parameters(const PromptOptions& options, const Json& tokens) {
    auto context = models::frontend::template_context(options.chat_template_kwargs_json, tokens);
    for (const char* key : {"enable_thinking", "preserve_thinking", "clear_thinking",
                            "reasoning_effort", "add_vision_id"})
        if (context.contains(key) && context[key].is_null()) context.erase(key);
    if (options.enable_thinking)
        models::frontend::merge_template_option(context, "enable_thinking",
                                                *options.enable_thinking);
    if (options.preserve_thinking)
        models::frontend::merge_template_option(context, "preserve_thinking",
                                                *options.preserve_thinking);
    if (options.reasoning_effort) {
        const auto name = reasoning_effort_name(*options.reasoning_effort);
        if (name.empty()) throw std::invalid_argument("invalid reasoning effort");
        models::frontend::merge_template_option(context, "reasoning_effort", name);
    }
    if (options.add_vision_id)
        models::frontend::merge_template_option(context, "add_vision_id", true);
    for (const char* key :
         {"enable_thinking", "preserve_thinking", "clear_thinking", "add_vision_id"})
        if (context.contains(key) && !context[key].is_boolean())
            throw std::invalid_argument(std::string(key) + " must be a boolean");

    if (context.contains("reasoning_effort")) {
        const auto& effort = context["reasoning_effort"];
        if (effort != "low" && effort != "high" && effort != "max")
            throw std::invalid_argument("GLM reasoning_effort must be low, high or max");
        models::frontend::merge_template_option(context, "enable_thinking", true);
    }
    if (!context.contains("enable_thinking") && options.defaults.enable_thinking)
        context["enable_thinking"] = *options.defaults.enable_thinking;
    if (!context.value("enable_thinking", true))
        throw std::invalid_argument("GLM does not support disabled thinking");
    if (context.value("add_vision_id", false))
        throw std::invalid_argument("GLM does not support add_vision_id");

    if (context.contains("preserve_thinking")) {
        models::frontend::merge_template_option(context, "clear_thinking",
                                                !context["preserve_thinking"].get<bool>());
        context.erase("preserve_thinking");
    } else if (!context.contains("clear_thinking") && options.defaults.preserve_thinking) {
        context["clear_thinking"] = !*options.defaults.preserve_thinking;
    }
    context.update(tokens);
    return context;
}

std::string_view role_name(ChatRole role) {
    switch (role) {
    case ChatRole::System:
    case ChatRole::Developer:
        return "system";
    case ChatRole::User:
        return "user";
    case ChatRole::Assistant:
        return "assistant";
    case ChatRole::Tool:
        return "tool";
    }
    throw std::invalid_argument("invalid chat role");
}

bool has_media(const ChatMessage& message) {
    return std::any_of(message.parts.begin(), message.parts.end(),
                       [](const auto& part) { return part.kind == MessagePartKind::Media; });
}

struct TemplateInput {
    Json context;
    std::vector<text::TemplateInputRegion> regions;
    std::vector<MessageSources> messages;
    std::vector<std::uint32_t> tools;

    std::uint32_t tag(std::string pointer) {
        const auto id = static_cast<std::uint32_t>(regions.size() + 1);
        regions.push_back({std::move(pointer), id});
        return id;
    }

    void append_message(const ChatMessage& message) {
        const bool media = has_media(message);
        if (message.role != ChatRole::Assistant &&
            (!message.reasoning_content.empty() || !message.tool_calls.empty()))
            throw std::invalid_argument("reasoning and tool calls require an assistant message");
        if (message.role != ChatRole::Tool &&
            (!message.tool_call_id.empty() || message.tool_result_is_error))
            throw std::invalid_argument("tool result fields require a tool message");
        if ((message.role == ChatRole::System || message.role == ChatRole::Developer) && media)
            throw std::invalid_argument("system and developer messages may contain only text");

        const auto pointer = "/messages/" + std::to_string(messages.size());
        Json value{{"role", role_name(message.role)}};
        MessageSources sources;
        sources.parts.resize(message.parts.size());
        const std::string error_prefix = message.tool_result_is_error ? "[tool_error]\n" : "";
        if (!media) {
            std::string content = error_prefix;
            const auto id       = tag(pointer + "/content");
            for (std::size_t i = 0; i < message.parts.size(); ++i) {
                const auto& part = message.parts[i];
                if (part.kind != MessagePartKind::Text)
                    throw std::invalid_argument("invalid message part kind");
                const auto begin = content.size();
                content += part.text;
                sources.parts[i] = TextSource{id, {begin, content.size()}};
            }
            sources.content  = TextSource{id, {0, content.size()}};
            value["content"] = std::move(content);
        } else {
            value["content"] = Json::array();
            if (!error_prefix.empty())
                value["content"].push_back({{"type", "text"}, {"text", error_prefix}});
            for (std::size_t i = 0; i < message.parts.size(); ++i) {
                const auto& part = message.parts[i];
                const auto item  = pointer + "/content/" + std::to_string(value["content"].size());
                switch (part.kind) {
                case MessagePartKind::Text:
                    value["content"].push_back({{"type", "text"}, {"text", part.text}});
                    sources.parts[i] = TextSource{tag(item + "/text"), {0, part.text.size()}};
                    break;
                case MessagePartKind::Media:
                    switch (part.media.kind) {
                    case MediaKind::Image:
                        value["content"].push_back({{"type", "image"}});
                        break;
                    case MediaKind::Video:
                        value["content"].push_back({{"type", "video"}});
                        break;
                    default:
                        throw std::invalid_argument("invalid media kind");
                    }
                    break;
                default:
                    throw std::invalid_argument("invalid message part kind");
                }
            }
        }
        if (message.role == ChatRole::Assistant) {
            // An explicit empty field prevents the source template from interpreting literal
            // <think> text in the structured content as imported reasoning.
            value["reasoning_content"] = message.reasoning_content;
            sources.reasoning          = {tag(pointer + "/reasoning_content"),
                                          {0, message.reasoning_content.size()}};
        }
        if (!message.tool_calls.empty()) {
            value["tool_calls"] = Json::array();
            for (const auto& call : message.tool_calls) {
                auto arguments =
                    call.arguments_json.empty() ? Json::object() : Json::parse(call.arguments_json);
                if (!arguments.is_object())
                    throw std::invalid_argument("tool call arguments must be a JSON object");
                sources.tool_calls.push_back(
                    tag(pointer + "/tool_calls/" + std::to_string(value["tool_calls"].size())));
                value["tool_calls"].push_back(
                    {{"id", call.id},
                     {"type", "function"},
                     {"function", {{"name", call.name}, {"arguments", std::move(arguments)}}}});
            }
        }
        if (!message.tool_call_id.empty()) value["tool_call_id"] = message.tool_call_id;
        context["messages"].push_back(std::move(value));
        messages.push_back(std::move(sources));
    }
};

std::size_t find_control(const text::TemplateOutput& output, std::string_view marker,
                         std::size_t begin = 0) {
    for (auto pos = output.text.find(marker, begin); pos != std::string::npos;
         pos      = output.text.find(marker, pos + marker.size()))
        if (!text::overlaps(output.literal_spans, pos, pos + marker.size())) return pos;
    return std::string::npos;
}

std::size_t final_assistant_body(const text::TemplateOutput& output) {
    std::size_t last = std::string::npos;
    std::string_view role;
    for (const auto marker : {"<|system|>", "<|user|>", "<|assistant|>", "<|observation|>"}) {
        for (auto pos = find_control(output, marker); pos != std::string::npos;
             pos      = find_control(output, marker, pos + 1)) {
            if (last == std::string::npos || pos > last) {
                last = pos;
                role = marker;
            }
        }
    }
    if (last == std::string::npos || role != "<|assistant|>")
        throw std::invalid_argument("GLM chat template must end in an assistant message");
    const auto body = last + role.size();
    if (find_control(output, "<think>", body) != body)
        throw std::invalid_argument("GLM assistant message must begin with a reasoning block");
    return body;
}

void truncate(text::TemplateOutput& output, std::size_t end) {
    output.text.resize(end);
    std::erase_if(output.regions, [&](const auto& region) { return region.begin > end; });
    for (auto& region : output.regions) region.end = std::min(region.end, end);
    std::erase_if(output.literal_spans, [&](auto span) { return span.begin >= end; });
    if (!output.literal_spans.empty())
        output.literal_spans.back().end = std::min(output.literal_spans.back().end, end);
    std::erase_if(output.boundary_mappings,
                  [&](const auto& mapping) { return mapping.offset > end; });
}

void finish_continuation(RenderedChat& result, std::string_view content, std::size_t body) {
    const auto close = find_control(result.output, "</think>", body + 7);
    if (close == std::string::npos)
        throw RequestError(RequestErrorKind::UnsupportedPromptContinuation,
                           "GLM continuation requires completed reasoning");
    const auto content_begin = close + std::string_view("</think>").size();
    const auto trim          = text::unicode_internal::trim_utf8(
        content, true, true, text::unicode_internal::is_python_whitespace,
        "assistant continuation");
    std::size_t end = content_begin;
    if (trim.begin != trim.end) {
        const auto region = models::frontend::unique_output_region(
            result.output, result.messages.back().content->tag);
        if (!region || region->begin < content_begin ||
            std::string_view(result.output.text)
                    .substr(region->begin, region->end - region->begin) !=
                content.substr(trim.begin, trim.end - trim.begin))
            throw RequestError(RequestErrorKind::UnsupportedPromptContinuation,
                               "GLM chat template must expose the final assistant text once");
        result.continuation_content =
            result.output.text.substr(region->begin, region->end - region->begin);
        end = region->end;
    }
    truncate(result.output, end);
}

RenderedChat render_input(const TemplateInput& input,
                          const models::frontend::ChatTemplate& compiled, bool continuation,
                          const PreparationControl& control) {
    std::vector<std::string> control_variables;
    for (const auto& item : compiled.special_tokens.items())
        control_variables.push_back(item.key());
    const text::TemplateRenderOptions execution{
        .checkpoint =
            [&] { models::frontend::check_preparation_control(control, "GLM chat template"); },
        .regions           = input.regions,
        .control_variables = control_variables};
    RenderedChat result{.output   = compiled.program.render(input.context, execution),
                        .messages = input.messages,
                        .tools    = input.tools};
    const auto body = final_assistant_body(result.output);
    if (continuation)
        finish_continuation(
            result, input.context["messages"].back()["content"].get_ref<const std::string&>(),
            body);
    else {
        if (find_control(result.output, "</think>", body + 7) != std::string::npos)
            throw std::invalid_argument("GLM generation prefix must leave reasoning open");
        result.starts_in_reasoning = true;
    }
    return result;
}
} // namespace

RenderedChat CompiledChatTemplate::render(std::span<const ChatMessage> messages,
                                          const PromptOptions& options,
                                          const PreparationControl& control) const {
    return render_impl(messages, options, {}, nullptr, control);
}

RenderedChat CompiledChatTemplate::render(std::span<const ChatMessage> messages,
                                          const PromptOptions& options,
                                          std::span<const PromptCacheMarker> markers,
                                          PromptBoundaries& boundaries,
                                          const PreparationControl& control) const {
    return render_impl(messages, options, markers, &boundaries, control);
}

RenderedChat CompiledChatTemplate::render_impl(std::span<const ChatMessage> messages,
                                               const PromptOptions& options,
                                               std::span<const PromptCacheMarker> markers,
                                               PromptBoundaries* boundaries,
                                               const PreparationControl& control) const {
    models::frontend::check_preparation_control(control, "GLM chat template");
    if (messages.empty()) throw std::invalid_argument("chat requires at least one message");
    bool continuation = false;
    switch (options.continuation) {
    case PromptContinuationMode::NewAssistantTurn:
        break;
    case PromptContinuationMode::ContinueFinalAssistant:
        continuation = true;
        if (messages.back().role != ChatRole::Assistant || has_media(messages.back()) ||
            !messages.back().tool_calls.empty())
            throw RequestError(RequestErrorKind::UnsupportedPromptContinuation,
                               "GLM continuation requires a final text-only assistant message");
        break;
    default:
        throw std::invalid_argument("invalid prompt continuation mode");
    }
    TemplateInput input{.context = template_parameters(options, compiled_.special_tokens)};
    input.context["messages"]               = Json::array();
    input.context["add_generation_prompt"]  = !continuation;
    input.context["continue_final_message"] = continuation;
    for (const auto& message : messages) input.append_message(message);
    input.context["tools"] = Json::array();
    for (const auto& source : options.tool_jsons) {
        auto tool = Json::parse(source);
        if (!tool.is_object())
            throw std::invalid_argument("tool declaration must be a JSON object");
        input.tools.push_back(input.tag("/tools/" + std::to_string(input.tools.size())));
        input.context["tools"].push_back(std::move(tool));
    }
    auto result = render_input(input, compiled_, continuation, control);
    if (boundaries) {
        PromptProbes probes;
        probes.content_prefix = [&](const PromptCacheMarker& marker) {
            auto probe       = input;
            const auto index = marker.after_message_count - 1;
            auto& source     = probe.messages[index];
            auto& value      = probe.context["messages"][index];
            const auto bytes = marker.after_message_part_count
                                   ? source.parts[marker.after_message_part_count - 1]->bytes.end
                                   : 0;
            value["content"].get_ref<std::string&>().resize(bytes);
            source.content->bytes.end = bytes;
            source.parts.resize(marker.after_message_part_count);
            value["reasoning_content"].get_ref<std::string&>().resize(marker.reasoning_bytes);
            source.reasoning.bytes.end = marker.reasoning_bytes;
            if (value.contains("tool_calls")) {
                auto& calls = value["tool_calls"];
                calls.erase(calls.begin() + marker.after_tool_call_count, calls.end());
            }
            source.tool_calls.resize(marker.after_tool_call_count);
            return render_input(probe, compiled_, continuation, control);
        };
        probes.next_turn = [&] {
            bool has_response = false;
            for (const auto& message : messages) {
                if (message.role == ChatRole::User)
                    has_response = false;
                else if (message.role == ChatRole::Assistant)
                    has_response = true;
            }
            auto probe = input;
            std::optional<RenderedChat> synthetic;
            if (!has_response) {
                probe.append_message(ChatMessage{.role              = ChatRole::Assistant,
                                                 .parts             = {{.text = "response"}},
                                                 .reasoning_content = "reasoning"});
                synthetic = render_input(probe, compiled_, false, control);
            }
            const auto& before = synthetic ? *synthetic : result;
            const auto end =
                final_assistant_body(before.output) - std::string_view("<|assistant|>").size();
            probe.append_message(ChatMessage{.role = ChatRole::User, .parts = {{.text = ""}}});
            auto after         = render_input(probe, compiled_, false, control);
            const auto retains = models::frontend::same_prefix(before.output, after.output, end);
            return TurnProbe{retains, std::move(after.output)};
        };
        *boundaries = resolve_prompt_boundaries(result, messages, continuation, markers, probes);
    }
    return result;
}

} // namespace ninfer::models::glm5next
