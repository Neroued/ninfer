#include "models/inkling2/frontend/chat_template.h"

#include "models/common/frontend/media_cache.h"
#include "models/inkling2/frontend/resources.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace ninfer::models::inkling2 {
namespace {
using Json = nlohmann::ordered_json;

constexpr double kDefaultEffort = 0.9;
constexpr double kMaxEffort     = 0.99;

std::optional<double> named_effort(std::string_view name) {
    if (name == "none") return 0.0;
    if (name == "minimal") return 0.1;
    if (name == "low") return 0.2;
    if (name == "medium") return 0.7;
    if (name == "high") return 0.9;
    if (name == "xhigh" || name == "max") return 0.99;
    return std::nullopt;
}

double resolve_effort_value(const Json& value) {
    if (value.is_string()) {
        std::string_view text = value.get_ref<const std::string&>();
        while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front())))
            text.remove_prefix(1);
        while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())))
            text.remove_suffix(1);
        if (const auto named = named_effort(text)) return *named;
        // Also accept numeric string values like "0.5" (matching vLLM Inkling2 renderer).
        std::string copy(text);
        char* end         = nullptr;
        const double parsed = std::strtod(copy.c_str(), &end);
        if (!copy.empty() && end == copy.c_str() + copy.size() && std::isfinite(parsed)) {
            if (parsed < 0.0 || parsed > kMaxEffort)
                throw std::invalid_argument("Inkling2 reasoning_effort must be in [0.0, 0.99]");
            return parsed;
        }
        throw std::invalid_argument("Unknown Inkling2 reasoning_effort: " +
                                    value.get<std::string>());
    }
    if (value.is_number() && !value.is_boolean()) {
        const double num = value.get<double>();
        if (!std::isfinite(num) || num < 0.0 || num > kMaxEffort)
            throw std::invalid_argument("Inkling2 reasoning_effort must be in [0.0, 0.99]");
        return num;
    }
    throw std::invalid_argument("Inkling2 reasoning_effort must be a named level or number");
}

struct NormalizedParams {
    Json context;
    bool thinking_enabled  = true;
    bool preserve_thinking = true;
};

NormalizedParams template_parameters(const PromptOptions& options, const Json& tokens) {
    auto context = models::frontend::template_context(options.chat_template_kwargs_json, tokens);
    for (const char* key :
         {"enable_thinking", "preserve_thinking", "reasoning_effort", "add_vision_id"}) {
        if (context.contains(key) && context[key].is_null()) context.erase(key);
    }
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

    for (const char* key : {"enable_thinking", "preserve_thinking", "add_vision_id"}) {
        if (context.contains(key) && !context[key].is_boolean())
            throw std::invalid_argument(std::string(key) + " must be a boolean");
    }
    if (context.value("add_vision_id", false))
        throw std::invalid_argument("Inkling2 does not support add_vision_id");

    std::optional<double> effort;
    if (context.contains("reasoning_effort")) {
        effort = resolve_effort_value(context["reasoning_effort"]);
        // Pass numeric effort directly when parsed from a numeric string so chat_template.jinja
        // (which only looks up named strings in effort_map) always formats it cleanly.
        if (context["reasoning_effort"].is_string()) {
            std::string_view s = context["reasoning_effort"].get_ref<const std::string&>();
            while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front())))
                s.remove_prefix(1);
            while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))
                s.remove_suffix(1);
            if (!named_effort(s)) context["reasoning_effort"] = *effort;
        }
    }

    std::optional<bool> enable_thinking;
    if (context.contains("enable_thinking")) {
        enable_thinking = context["enable_thinking"].get<bool>();
    } else if (options.defaults.enable_thinking) {
        enable_thinking = *options.defaults.enable_thinking;
    }

    if (effort && enable_thinking) {
        const bool effort_on = *effort > 0.0;
        if (effort_on != *enable_thinking) {
            throw RequestError(RequestErrorKind::ConflictingPromptOptions,
                               "reasoning_effort conflicts with enable_thinking",
                               "reasoning_effort");
        }
    } else if (!effort && enable_thinking && !*enable_thinking) {
        effort                      = 0.0;
        context["reasoning_effort"] = 0.0;
    }

    const double effective_effort = effort.value_or(kDefaultEffort);
    bool preserve                 = true;
    if (context.contains("preserve_thinking"))
        preserve = context["preserve_thinking"].get<bool>();
    else if (options.defaults.preserve_thinking)
        preserve = *options.defaults.preserve_thinking;

    context.update(tokens);
    return {.context           = std::move(context),
            .thinking_enabled  = effective_effort > 0.0,
            .preserve_thinking = preserve};
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
    std::vector<std::uint32_t> tool_error_tags;

    struct MediaSource {
        std::size_t message = 0;
        std::size_t part    = 0;
        std::string pointer;
        MediaKind kind    = MediaKind::Image;
        std::uint32_t tag = 0;
    };

    std::vector<MediaSource> media;

    std::uint32_t tag(std::string pointer) {
        const auto id = static_cast<std::uint32_t>(regions.size() + 1);
        regions.push_back({std::move(pointer), id});
        return id;
    }

    void append_message(const ChatMessage& message, bool keep_reasoning) {
        const bool media_present = has_media(message);
        if (message.role != ChatRole::Assistant &&
            (!message.reasoning_content.empty() || !message.tool_calls.empty()))
            throw std::invalid_argument("reasoning and tool calls require an assistant message");
        if (message.role != ChatRole::Tool &&
            (!message.tool_call_id.empty() || message.tool_result_is_error))
            throw std::invalid_argument("tool result fields require a tool message");
        if ((message.role == ChatRole::System || message.role == ChatRole::Developer ||
             message.role == ChatRole::Assistant) &&
            media_present)
            throw std::invalid_argument("Inkling2 supports images only in user and tool messages");

        const auto pointer = "/messages/" + std::to_string(messages.size());
        Json value{{"role", role_name(message.role)}};
        MessageSources sources;
        sources.parts.resize(message.parts.size());

        if (!media_present) {
            std::string content;
            const auto id = tag(pointer + "/content");
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
            if (message.role == ChatRole::Tool && message.tool_result_is_error)
                tool_error_tags.push_back(id);
        } else {
            value["content"] = Json::array();
            for (std::size_t i = 0; i < message.parts.size(); ++i) {
                const auto& part = message.parts[i];
                const auto item  = pointer + "/content/" + std::to_string(value["content"].size());
                switch (part.kind) {
                case MessagePartKind::Text: {
                    const auto id = tag(item + "/text");
                    value["content"].push_back({{"type", "text"}, {"text", part.text}});
                    sources.parts[i] = TextSource{id, {0, part.text.size()}};
                    if (message.role == ChatRole::Tool && message.tool_result_is_error)
                        tool_error_tags.push_back(id);
                    break;
                }
                case MessagePartKind::Media: {
                    if (part.media.kind != MediaKind::Image) {
                        throw RequestError(RequestErrorKind::InvalidMedia,
                                           "Inkling2 supports image media only");
                    }
                    const auto id = tag(item + "/text");
                    this->media.push_back({messages.size(), i, item, part.media.kind, id});
                    sources.parts[i] = TextSource{id, {0, 1}};
                    value["content"].push_back({{"type", "image"}});
                    break;
                }
                default:
                    throw std::invalid_argument("invalid message part kind");
                }
            }
        }

        if (message.role == ChatRole::Assistant && keep_reasoning &&
            !message.reasoning_content.empty()) {
            value["reasoning_content"] = message.reasoning_content;
            sources.reasoning          = {tag(pointer + "/reasoning_content"),
                                          {0, message.reasoning_content.size()}};
        }
        if (!message.tool_calls.empty()) {
            value["tool_calls"] = Json::array();
            for (std::size_t j = 0; j < message.tool_calls.size(); ++j) {
                const auto& call = message.tool_calls[j];
                auto arguments   = models::frontend::parse_tool_history_arguments(
                    call.arguments_json, call.name, messages.size(), j);
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

bool ends_with_control(const text::TemplateOutput& output, std::string_view marker) {
    if (output.text.size() < marker.size()) return false;
    const std::size_t pos = output.text.size() - marker.size();
    return std::string_view(output.text).substr(pos) == marker &&
           !text::overlaps(output.literal_spans, pos, output.text.size());
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

void rewrite_tool_error_markers(text::TemplateOutput& output,
                                std::span<const std::uint32_t> error_tags) {
    if (error_tags.empty()) return;
    struct Replacement {
        std::size_t begin = 0;
        std::size_t end   = 0;
    };
    std::vector<Replacement> targets;
    for (const std::uint32_t tag : error_tags) {
        const auto region = models::frontend::unique_output_region(output, tag);
        if (!region || region->begin < kContentText.size()) continue;
        const std::size_t marker_begin = region->begin - kContentText.size();
        if (std::string_view(output.text).substr(marker_begin, kContentText.size()) ==
                kContentText &&
            !text::overlaps(output.literal_spans, marker_begin, region->begin)) {
            targets.push_back({marker_begin, region->begin});
        }
    }
    if (targets.empty()) return;
    std::sort(targets.begin(), targets.end(),
              [](const auto& a, const auto& b) { return a.begin < b.begin; });

    struct Edit {
        text::ByteSpan before;
        text::ByteSpan after;
    };
    std::vector<Edit> edits;
    std::string rebuilt;
    std::size_t cursor = 0;
    for (const auto& target : targets) {
        if (target.begin < cursor) continue;
        rebuilt.append(output.text, cursor, target.begin - cursor);
        const std::size_t new_begin = rebuilt.size();
        rebuilt.append(kContentToolError);
        edits.push_back({{target.begin, target.end}, {new_begin, rebuilt.size()}});
        cursor = target.end;
    }
    rebuilt.append(output.text, cursor, std::string::npos);

    const auto map = [&](std::size_t pos) {
        std::size_t old_end = 0, new_end = 0;
        for (const auto& edit : edits) {
            if (pos <= edit.before.begin) break;
            if (pos < edit.before.end) return edit.after.begin;
            old_end = edit.before.end;
            new_end = edit.after.end;
        }
        return new_end + (pos - old_end);
    };
    for (auto& span : output.literal_spans) span = {map(span.begin), map(span.end)};
    for (auto& region : output.regions) {
        region.begin = map(region.begin);
        region.end   = map(region.end);
    }
    for (auto& bm : output.boundary_mappings) bm.offset = map(bm.offset);
    output.text = std::move(rebuilt);
}

void locate_media(RenderedChat& result, const TemplateInput& input,
                  const models::frontend::ChatTemplate& compiled,
                  text::TemplateRenderOptions execution, std::string_view image_token) {
    if (input.media.empty()) return;
    auto context = input.context;
    for (const auto& source : input.media) {
        context[Json::json_pointer(source.pointer)] =
            Json{{"type", "text"}, {"text", "NInferMedia" + std::to_string(source.tag) + "End"}};
    }
    auto probe = compiled.program.render(context, execution);
    rewrite_tool_error_markers(probe, input.tool_error_tags);

    struct Located {
        const TemplateInput::MediaSource* source = nullptr;
        text::ByteSpan probe_replace_span;
    };

    std::vector<Located> located;
    located.reserve(input.media.size());
    for (const auto& source : input.media) {
        const auto region  = models::frontend::unique_output_region(probe, source.tag);
        const auto locator = "NInferMedia" + std::to_string(source.tag) + "End";
        if (!region || probe.text.substr(region->begin, region->end - region->begin) != locator)
            throw std::invalid_argument("Inkling2 template must emit each image item exactly once");
        if (region->begin < kContentText.size() ||
            std::string_view(probe.text).substr(region->begin - kContentText.size(),
                                                kContentText.size()) != kContentText ||
            text::overlaps(probe.literal_spans, region->begin - kContentText.size(),
                           region->begin)) {
            throw std::invalid_argument(
                "Inkling2 template must wrap image probe inside <|content_text|>");
        }
        located.push_back({&source, {region->begin - kContentText.size(), region->end}});
    }
    std::sort(located.begin(), located.end(), [](const auto& a, const auto& b) {
        return a.probe_replace_span.begin < b.probe_replace_span.begin;
    });

    const std::string image_wrapper = std::string(kContentImage) + std::string(image_token);
    text::TemplateOutput restored;
    std::size_t cursor         = 0;
    std::size_t literal_cursor = 0;
    const auto append_slice    = [&](std::size_t begin, std::size_t end) {
        const auto base = restored.text.size();
        restored.text.append(probe.text, begin, end - begin);
        while (literal_cursor < probe.literal_spans.size() &&
               probe.literal_spans[literal_cursor].end <= begin)
            ++literal_cursor;
        for (auto i = literal_cursor;
             i < probe.literal_spans.size() && probe.literal_spans[i].begin < end; ++i) {
            const auto literal = probe.literal_spans[i];
            const auto left = std::max(begin, literal.begin), right = std::min(end, literal.end);
            if (left < right)
                restored.literal_spans.push_back({base + left - begin, base + right - begin});
        }
    };

    for (const auto& entry : located) {
        append_slice(cursor, entry.probe_replace_span.begin);
        const auto begin = restored.text.size();
        restored.text += image_wrapper;
        const auto end = restored.text.size();
        result.media.push_back(
            {entry.source->message, entry.source->part, entry.source->kind, {begin, end}});
        result.output.regions.push_back({entry.source->tag, begin, end, std::nullopt});
        result.output.boundary_mappings.push_back({entry.source->tag, 0, 0, begin});
        result.output.boundary_mappings.push_back({entry.source->tag, 1, 1, end});
        cursor = entry.probe_replace_span.end;
    }
    append_slice(cursor, probe.text.size());
    if (restored.text.size() != result.output.text.size() ||
        !models::frontend::same_prefix(restored, result.output, restored.text.size())) {
        throw std::invalid_argument(
            "Inkling2 template cannot map image placeholders to their input sources");
    }
}

void finish_continuation(RenderedChat& result, std::string_view final_content) {
    if (!ends_with_control(result.output, kContentModelEndSample)) {
        throw RequestError(RequestErrorKind::UnsupportedPromptContinuation,
                           "Inkling2 continuation requires a closed final assistant turn");
    }
    truncate(result.output, result.output.text.size() - kContentModelEndSample.size());

    if (!final_content.empty()) {
        if (!ends_with_control(result.output, kEndMessage)) {
            throw RequestError(RequestErrorKind::UnsupportedPromptContinuation,
                               "Inkling2 continuation requires a text block closing marker");
        }
        truncate(result.output, result.output.text.size() - kEndMessage.size());
        const auto region = models::frontend::unique_output_region(
            result.output, result.messages.back().content->tag);
        if (!region || region->end != result.output.text.size() ||
            std::string_view(result.output.text)
                    .substr(region->begin, region->end - region->begin) != final_content) {
            throw RequestError(RequestErrorKind::UnsupportedPromptContinuation,
                               "Inkling2 template must expose the final assistant text at the end");
        }
        result.continuation_content = std::string(final_content);
        result.initial_block        = InitialBlock::Text;
        return;
    }

    // Empty final assistant content: either an empty text block was emitted (strip <|end_message|>)
    // or only reasoning was emitted (open a new <|message_model|><|content_text|> block).
    const std::string empty_text_block =
        std::string(kMessageModel) + std::string(kContentText) + std::string(kEndMessage);
    if (ends_with_control(result.output, empty_text_block)) {
        truncate(result.output, result.output.text.size() - kEndMessage.size());
    } else {
        result.output.text.append(kMessageModel);
        result.output.text.append(kContentText);
    }
    result.continuation_content.clear();
    result.initial_block = InitialBlock::Text;
}

RenderedChat render_input(const TemplateInput& input,
                          const models::frontend::ChatTemplate& compiled, bool continuation,
                          bool thinking_enabled, text::TemplateRenderOptions execution,
                          std::string_view image_token, std::size_t* generation_begin = nullptr) {
    execution.regions = input.regions;
    RenderedChat result{.output           = compiled.program.render(input.context, execution),
                        .messages         = input.messages,
                        .tools            = input.tools,
                        .thinking_enabled = thinking_enabled};
    rewrite_tool_error_markers(result.output, input.tool_error_tags);
    locate_media(result, input, compiled, execution, image_token);
    std::size_t prefix_begin = result.output.text.size();
    if (continuation) {
        finish_continuation(
            result, input.context["messages"].back()["content"].get_ref<const std::string&>());
        prefix_begin = result.output.text.size();
    } else if (!thinking_enabled && input.tools.empty()) {
        // Inkling2 treats "Thinking effort level: 0" as a soft hint only. When thinking is
        // disabled without active tools, prefill <|message_model|><|content_text|> so the model
        // emits answer text directly without entering a thinking block.
        if (ends_with_control(result.output, kMessageModel)) {
            prefix_begin = result.output.text.size() - kMessageModel.size();
            result.output.text.append(kContentText);
        } else {
            prefix_begin = result.output.text.size();
            result.output.text.append(kMessageModel);
            result.output.text.append(kContentText);
        }
        result.initial_block = InitialBlock::Text;
    } else if (ends_with_control(result.output, kMessageModel)) {
        prefix_begin         = result.output.text.size() - kMessageModel.size();
        result.initial_block = InitialBlock::MessageHeader;
    } else {
        result.initial_block = InitialBlock::BetweenBlocks;
    }
    if (generation_begin) *generation_begin = prefix_begin;
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
    models::frontend::check_preparation_control(control, "Inkling2 chat template");
    if (messages.empty()) throw std::invalid_argument("chat requires at least one message");

    bool continuation = false;
    switch (options.continuation) {
    case PromptContinuationMode::NewAssistantTurn:
        break;
    case PromptContinuationMode::ContinueFinalAssistant:
        continuation = true;
        if (messages.back().role != ChatRole::Assistant || has_media(messages.back()) ||
            !messages.back().tool_calls.empty()) {
            throw RequestError(
                RequestErrorKind::UnsupportedPromptContinuation,
                "Inkling2 continuation requires a final text-only assistant message");
        }
        break;
    default:
        throw std::invalid_argument("invalid prompt continuation mode");
    }

    auto params = template_parameters(options, compiled_.special_tokens);
    std::size_t last_user = 0;
    bool saw_user         = false;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        if (messages[i].role == ChatRole::User) {
            last_user = i;
            saw_user  = true;
        }
    }

    TemplateInput input{.context = std::move(params.context)};
    input.context["messages"]               = Json::array();
    input.context["add_generation_prompt"]  = !continuation;
    input.context["continue_final_message"] = continuation;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        const bool keep_reasoning = params.preserve_thinking || !saw_user || i > last_user;
        input.append_message(messages[i], keep_reasoning);
    }

    if (!options.tool_jsons.empty()) {
        input.context["tools"] = Json::array();
        for (const auto& source : options.tool_jsons) {
            auto tool = Json::parse(source);
            if (!tool.is_object())
                throw std::invalid_argument("tool declaration must be a JSON object");
            input.tools.push_back(input.tag("/tools/" + std::to_string(input.tools.size())));
            input.context["tools"].push_back(std::move(tool));
        }
    }

    std::vector<std::string> control_variables;
    for (const auto& item : compiled_.special_tokens.items())
        control_variables.push_back(item.key());
    const text::TemplateRenderOptions execution{
        .checkpoint =
            [&] { models::frontend::check_preparation_control(control, "Inkling2 chat template"); },
        .control_variables = control_variables};

    std::size_t generation_begin = 0;
    auto result = render_input(input, compiled_, continuation, params.thinking_enabled, execution,
                               image_token_, &generation_begin);

    if (boundaries) {
        PromptProbes probes;
        probes.content_prefix = [&](const PromptCacheMarker& marker) {
            auto probe       = input;
            const auto index = marker.after_message_count - 1;
            auto& source     = probe.messages[index];
            auto& value      = probe.context["messages"][index];
            if (source.content) {
                const auto bytes =
                    marker.after_message_part_count
                        ? source.parts[marker.after_message_part_count - 1]->bytes.end
                        : 0;
                value["content"].get_ref<std::string&>().resize(bytes);
                source.content->bytes.end = bytes;
            } else {
                auto& content = value["content"];
                content.erase(content.begin() + marker.after_message_part_count, content.end());
                std::erase_if(probe.media, [&](const auto& media) {
                    return media.message == index && media.part >= marker.after_message_part_count;
                });
            }
            source.parts.resize(marker.after_message_part_count);
            if (value.contains("reasoning_content")) {
                value["reasoning_content"].get_ref<std::string&>().resize(marker.reasoning_bytes);
                source.reasoning.bytes.end = marker.reasoning_bytes;
            }
            if (value.contains("tool_calls")) {
                auto& calls = value["tool_calls"];
                calls.erase(calls.begin() + marker.after_tool_call_count, calls.end());
            }
            source.tool_calls.resize(marker.after_tool_call_count);
            return render_input(probe, compiled_, continuation, params.thinking_enabled, execution,
                                image_token_);
        };
        probes.next_turn = [&] {
            auto probe = input;
            probe.append_message(ChatMessage{.role = ChatRole::User, .parts = {{.text = ""}}},
                                 params.preserve_thinking);
            auto after = compiled_.program.render(probe.context, execution);
            rewrite_tool_error_markers(after, probe.tool_error_tags);
            const bool retains =
                models::frontend::same_prefix(result.output, after, generation_begin);
            return TurnProbe{retains, std::move(after)};
        };
        *boundaries = resolve_prompt_boundaries(result, messages, continuation, markers, probes);
    }
    return result;
}

} // namespace ninfer::models::inkling2
