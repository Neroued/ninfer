#include "models/common/qwen/frontend/chat_template.h"

#include "models/common/frontend/media_cache.h"
#include "models/common/qwen/frontend/prompt_layout.h"
#include "text/unicode.h"

#include <algorithm>
#include <map>
#include <stdexcept>

namespace ninfer::models::qwen::frontend {
namespace {

using Json = nlohmann::ordered_json;

std::string_view role_name(ChatRole role) {
    switch (role) {
    case ChatRole::System:
        return "system";
    case ChatRole::Developer:
        return "developer";
    case ChatRole::User:
        return "user";
    case ChatRole::Assistant:
        return "assistant";
    case ChatRole::Tool:
        return "tool";
    }
    throw std::invalid_argument("invalid chat role");
}

bool instruction(ChatRole role) { return role == ChatRole::System || role == ChatRole::Developer; }

struct ContentSource {
    std::uint32_t tag = 0;
    std::vector<std::size_t> part_ends;
    std::vector<std::uint32_t> part_tags;
    std::vector<std::optional<std::size_t>> part_media;
    std::uint32_t reasoning_tag = 0;
};

Json template_parameters(const ChatRenderOptions& options, const Json& special_tokens) {
    Json context =
        models::frontend::template_context(options.chat_template_kwargs_json, special_tokens);
    for (const char* name :
         {"enable_thinking", "preserve_thinking", "reasoning_effort", "add_vision_id"}) {
        if (context.contains(name) && context[name].is_null()) context.erase(name);
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
    for (const char* name : {"enable_thinking", "preserve_thinking", "add_vision_id"}) {
        if (context.contains(name) && !context[name].is_boolean()) {
            throw std::invalid_argument(std::string(name) + " must be a boolean");
        }
    }
    if (context.contains("reasoning_effort")) {
        const auto& effort = context["reasoning_effort"];
        if (!effort.is_string() ||
            (effort != "none" && effort != "minimal" && effort != "low" && effort != "medium" &&
             effort != "high" && effort != "xhigh" && effort != "max")) {
            throw std::invalid_argument("invalid reasoning_effort template parameter");
        }
        const bool thinking = effort != "none";
        if (context.contains("enable_thinking") && context["enable_thinking"] != thinking) {
            throw RequestError(RequestErrorKind::ConflictingPromptOptions,
                               "reasoning_effort conflicts with enable_thinking",
                               "reasoning_effort");
        }
        context["enable_thinking"] = thinking;
    }
    if (!context.contains("enable_thinking") && options.defaults.enable_thinking)
        context["enable_thinking"] = *options.defaults.enable_thinking;
    if (!context.contains("preserve_thinking") && options.defaults.preserve_thinking)
        context["preserve_thinking"] = *options.defaults.preserve_thinking;
    context.update(special_tokens);
    return context;
}

std::optional<std::size_t> containing_message(const PromptLayout& layout, text::ByteSpan region) {
    for (std::size_t i = 0; i < layout.messages.size(); ++i) {
        const auto& message = layout.messages[i];
        if (region.begin >= message.content_begin && region.end <= message.content_end) return i;
    }
    return std::nullopt;
}

bool real_user(const ChatMessage& message) {
    if (message.role != ChatRole::User) return false;
    if (message.has_media()) return true;
    std::string content;
    for (const auto& part : message.parts) content += part.text;
    const auto bytes = text::unicode_internal::trim_utf8(
        content, true, true, text::unicode_internal::is_whitespace, "message content");
    const auto body = std::string_view(content).substr(bytes.begin, bytes.end - bytes.begin);
    return !(body.starts_with("<tool_response>") && body.ends_with("</tool_response>"));
}

} // namespace

bool ChatMessage::has_media() const noexcept {
    return std::any_of(parts.begin(), parts.end(),
                       [](const ChatPart& part) { return part.kind != ChatPartKind::Text; });
}

CompiledChatTemplate CompiledChatTemplate::resolve(std::string_view source, std::string source_name,
                                                   Json special_tokens) {
    return CompiledChatTemplate({text::JinjaTemplate(std::string(source), std::move(source_name)),
                                 std::move(special_tokens)});
}

RenderedChat CompiledChatTemplate::render(const std::vector<ChatMessage>& messages,
                                          ChatRenderOptions options,
                                          const PreparationControl& control) const {
    if (messages.empty()) throw std::invalid_argument("chat requires at least one message");
    models::frontend::check_preparation_control(control, "chat template");
    const bool continuation =
        options.continuation == PromptContinuationMode::ContinueFinalAssistant;
    Json context = template_parameters(options, compiled_.special_tokens);
    if (continuation) {
        if (context.value("enable_thinking", false))
            throw RequestError(RequestErrorKind::UnsupportedPromptContinuation,
                               "assistant prefill cannot be combined with enabled thinking");
        const auto& final = messages.back();
        if (final.role != ChatRole::Assistant || final.has_media() ||
            !final.reasoning_content.empty() || !final.tool_calls.empty()) {
            throw std::invalid_argument("assistant continuation requires a final text-only "
                                        "assistant message and disabled thinking");
        }
    }
    context["continue_final_message"] = continuation;
    context["messages"]               = Json::array();
    context["add_generation_prompt"]  = !continuation && options.add_generation_prompt;
    std::vector<text::TemplateInputRegion> regions;
    std::uint32_t next_tag = 1;
    const auto tag         = [&](std::string pointer) {
        const auto id = next_tag++;
        regions.push_back({std::move(pointer), id});
        return id;
    };
    std::vector<ContentSource> sources(messages.size());
    std::vector<models::frontend::Modality> media;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        const auto& message = messages[i];
        if (instruction(message.role) &&
            (message.has_media() || !message.reasoning_content.empty() ||
             !message.tool_calls.empty() || !message.tool_call_id.empty())) {
            throw std::invalid_argument("system and developer messages may contain only text");
        }
        Json value{{"role", role_name(message.role)}};
        const auto pointer = "/messages/" + std::to_string(i);
        auto& origin       = sources[i];
        if (!message.has_media()) {
            std::string content;
            if (message.role == ChatRole::Tool && message.tool_result_is_error)
                content = "[tool_error]\n";
            for (const auto& part : message.parts) {
                content += part.text;
                origin.part_ends.push_back(content.size());
            }
            value["content"] = std::move(content);
            origin.tag       = tag(pointer + "/content");
        } else {
            value["content"] = Json::array();
            if (message.role == ChatRole::Tool && message.tool_result_is_error)
                value["content"].push_back({{"type", "text"}, {"text", "[tool_error]\n"}});
            for (std::size_t j = 0; j < message.parts.size(); ++j) {
                const auto& part = message.parts[j];
                const auto item  = pointer + "/content/" + std::to_string(value["content"].size());
                if (part.kind == ChatPartKind::Text) {
                    value["content"].push_back({{"type", "text"}, {"text", part.text}});
                    origin.part_tags.push_back(tag(item + "/text"));
                    origin.part_media.push_back(std::nullopt);
                } else {
                    const bool image = part.kind == ChatPartKind::Image;
                    value["content"].push_back({{"type", image ? "image" : "video"}});
                    origin.part_media.push_back(media.size());
                    media.push_back(image ? models::frontend::Modality::Image
                                          : models::frontend::Modality::Video);
                    origin.part_tags.push_back(0);
                }
            }
        }
        if (!message.reasoning_content.empty()) {
            value["reasoning_content"] = message.reasoning_content;
            origin.reasoning_tag       = tag(pointer + "/reasoning_content");
        }
        if (!message.tool_calls.empty()) {
            value["tool_calls"] = Json::array();
            for (const auto& call : message.tool_calls) {
                value["tool_calls"].push_back(
                    {{"id", call.id},
                     {"type", "function"},
                     {"function",
                      {{"name", call.name},
                       {"arguments",
                        (call.arguments_json.empty() ? Json::object()
                                                     : Json::parse(call.arguments_json))}}}});
            }
        }
        if (!message.tool_call_id.empty()) value["tool_call_id"] = message.tool_call_id;
        context["messages"].push_back(std::move(value));
    }
    std::vector<std::uint32_t> tool_tags;
    if (!options.tool_jsons.empty()) {
        context["tools"] = Json::array();
        for (std::size_t i = 0; i < options.tool_jsons.size(); ++i) {
            context["tools"].push_back(Json::parse(options.tool_jsons[i]));
            tool_tags.push_back(tag("/tools/" + std::to_string(i)));
        }
    }
    std::vector<std::string> control_variables;
    for (const auto& item : compiled_.special_tokens.items())
        control_variables.push_back(item.key());
    text::TemplateRenderOptions execution{
        .checkpoint =
            [&] { models::frontend::check_preparation_control(control, "chat template"); },
        .regions           = regions,
        .control_variables = control_variables};
    auto output = compiled_.program.render(context, execution);
    auto layout = inspect_prompt_layout(output, media);
    std::string continuation_content;
    if (continuation) {
        const auto content = models::frontend::unique_output_region(output, sources.back().tag);
        if (!content || layout.messages.empty())
            throw std::invalid_argument(
                "chat template does not expose the final assistant content for continuation");
        const auto& final = layout.messages.back();
        if (final.role != ChatRole::Assistant || content->begin < final.content_begin ||
            content->end > final.content_end) {
            throw std::invalid_argument(
                "chat template cannot continue the final assistant content unambiguously");
        }
        continuation_content = output.text.substr(content->begin, content->end - content->begin);
        output.text.resize(content->end);
        std::erase_if(output.regions,
                      [&](const auto& region) { return region.begin > output.text.size(); });
        for (auto& region : output.regions) region.end = std::min(region.end, output.text.size());
        std::erase_if(output.literal_spans,
                      [&](auto span) { return span.begin >= output.text.size(); });
        if (!output.literal_spans.empty())
            output.literal_spans.back().end =
                std::min(output.literal_spans.back().end, output.text.size());
        layout = inspect_prompt_layout(output, media);
    }
    RenderedChat result;
    if (context.contains("enable_thinking"))
        result.enable_thinking = context["enable_thinking"].get<bool>();
    result.continuation_content         = std::move(continuation_content);
    result.starts_in_reasoning          = layout.starts_in_reasoning;
    result.media_placeholders           = layout.media_placeholders;
    result.rewrite_execution_boundaries = layout.execution_boundaries;
    result.message_boundaries.resize(messages.size() + 1);
    result.cache_boundaries.resize(options.cache_markers.size());

    // Only requested/structural boundaries need proof, independent of history length.
    std::map<std::size_t, std::optional<std::size_t>> prefix_cache;
    auto prefix = [&](std::size_t count) -> std::optional<std::size_t> {
        if (count > messages.size()) return std::nullopt;
        if (const auto it = prefix_cache.find(count); it != prefix_cache.end()) return it->second;
        std::optional<std::size_t> boundary;
        Json probe = context;
        probe["messages"].erase(probe["messages"].begin() + static_cast<std::ptrdiff_t>(count),
                                probe["messages"].end());
        probe["add_generation_prompt"] = false;
        try {
            const auto rendered = compiled_.program.render(probe, execution);
            if (models::frontend::same_prefix(output, rendered, rendered.text.size()))
                boundary = rendered.text.size();
        } catch (const RequestError&) {
            throw;
        } catch (const std::invalid_argument&) { /* This subset has no independent serialization. */
        }
        prefix_cache.emplace(count, boundary);
        result.message_boundaries[count] = boundary;
        return boundary;
    };
    // Associate actual input regions with complete ChatML blocks. Several tool results may
    // share one user block; their individual closing tags provide the intermediate boundaries.
    std::vector<std::optional<std::size_t>> message_blocks(messages.size());
    std::vector<std::optional<text::ByteSpan>> content_regions(messages.size());
    std::vector<std::size_t> block_users(layout.messages.size());
    for (std::size_t i = 0; i < sources.size(); ++i) {
        const auto& origin = sources[i];
        std::optional<text::ByteSpan> extent;
        const auto include = [&](text::ByteSpan bytes) {
            if (!extent)
                extent = bytes;
            else {
                extent->begin = std::min(extent->begin, bytes.begin);
                extent->end   = std::max(extent->end, bytes.end);
            }
        };
        for (const auto& region : output.regions) {
            if (region.tag == origin.tag || region.tag == origin.reasoning_tag ||
                std::find(origin.part_tags.begin(), origin.part_tags.end(), region.tag) !=
                    origin.part_tags.end()) {
                include({region.begin, region.end});
            }
        }
        for (const auto index : origin.part_media) {
            if (index) include(layout.media_placeholders[*index].bytes);
        }
        content_regions[i] = extent;
        if (extent) message_blocks[i] = containing_message(layout, *extent);
        if (message_blocks[i]) ++block_users[*message_blocks[i]];
    }
    for (std::size_t i = 0; i < messages.size(); ++i) {
        if (!message_blocks[i]) continue;
        const auto& block = layout.messages[*message_blocks[i]];
        if (block.closed && block_users[*message_blocks[i]] == 1)
            result.message_boundaries[i + 1] = block.end;
        if (messages[i].role == ChatRole::Tool && content_regions[i]) {
            constexpr std::string_view close = "\n</tool_response>";
            const auto pos                   = output.text.find(close, content_regions[i]->end);
            if (pos != std::string::npos && pos + close.size() <= block.content_end) {
                const auto end = pos + close.size();
                const bool sourced =
                    text::overlaps(output.literal_spans, content_regions[i]->end, end);
                if (!sourced)
                    result.message_boundaries[i + 1] =
                        end == block.content_end && block.closed ? block.end : end;
            }
        }
    }
    if (instruction(messages.front().role) && !result.message_boundaries[1]) (void)prefix(1);

    const auto generation_begin = continuation
                                      ? std::optional<std::size_t>(layout.messages.back().begin)
                                  : options.add_generation_prompt ? prefix(messages.size())
                                                                  : std::nullopt;
    if (!options.add_generation_prompt && !continuation)
        result.message_boundaries.back() = output.text.size();
    std::optional<std::size_t> first_tail_assistant;
    if (!continuation) {
        std::optional<std::size_t> query_message;
        for (std::size_t i = messages.size(); i > 0; --i) {
            if (real_user(messages[i - 1])) {
                query_message = i - 1;
                break;
            }
            // Imported history may begin with a tool result after its user turn was removed.
            if (!query_message && messages[i - 1].role == ChatRole::Tool) query_message = i - 1;
        }
        if (query_message && message_blocks[*query_message]) {
            for (std::size_t j = *message_blocks[*query_message] + 1; j < layout.messages.size();
                 ++j) {
                if (layout.messages[j].role == ChatRole::Assistant) {
                    first_tail_assistant = j;
                    break;
                }
            }
        }
    }
    // Select a recovery point from the template's observed serialization. A single ordinary
    // next-turn probe distinguishes retained history from a rewritten open turn, including
    // templates whose defaults or aliases differ from the typed request hint. This only chooses
    // the checkpoint to retain: subsequent requests still require exact token/state identity.
    bool retain_open_turn = context.value("preserve_thinking", false);
    if (first_tail_assistant && layout.messages[*first_tail_assistant].closed) {
        Json probe = context;
        probe["messages"].push_back({{"role", "user"}, {"content", ""}});
        probe["add_generation_prompt"] = false;
        auto probe_options             = execution;
        probe_options.regions          = {};
        try {
            const auto next_turn     = compiled_.program.render(probe, probe_options);
            const auto history_bytes = generation_begin.value_or(output.text.size());
            retain_open_turn = models::frontend::same_prefix(next_turn, output, history_bytes);
        } catch (const RequestError&) { throw; } catch (const std::invalid_argument&) {
            retain_open_turn = false;
        }
    }
    if (generation_begin && *generation_begin > 0) {
        result.rewrite_checkpoint = RewriteCheckpointByteSpec{
            .kind   = continuation || retain_open_turn ? RewriteCheckpointKind::ResponseReplay
                                                       : RewriteCheckpointKind::TurnClosure,
            .offset = *generation_begin,
            .recovery_offset = *generation_begin};
    }
    if (!continuation && !retain_open_turn && first_tail_assistant) {
        const auto begin = layout.messages[*first_tail_assistant].begin;
        if (begin > 0)
            result.rewrite_checkpoint =
                RewriteCheckpointByteSpec{.kind            = RewriteCheckpointKind::TurnClosure,
                                          .offset          = begin,
                                          .recovery_offset = begin};
    }
    const auto part_boundary = [&](std::size_t message,
                                   std::size_t part) -> std::optional<std::size_t> {
        const auto& origin = sources[message];
        if (part < origin.part_ends.size())
            return models::frontend::source_boundary(output, origin.tag, origin.part_ends[part]);
        if (part < origin.part_tags.size()) {
            if (origin.part_tags[part]) {
                if (const auto region =
                        models::frontend::unique_output_region(output, origin.part_tags[part]))
                    return region->end;
            } else if (origin.part_media[part]) {
                return layout.media_placeholders[*origin.part_media[part]].bytes.end +
                       std::string_view("<|vision_end|>").size();
            }
        }
        return std::nullopt;
    };
    if (result.rewrite_checkpoint) {
        auto& checkpoint = *result.rewrite_checkpoint;
        for (std::size_t i = 0; i < messages.size(); ++i) {
            const auto& message = messages[i];
            if ((message.role != ChatRole::User && !instruction(message.role)) ||
                message.parts.empty() || !message_blocks[i]) {
                continue;
            }
            const auto block_index = *message_blocks[i];
            const auto& block      = layout.messages[block_index];
            if (!block.closed || block_users[block_index] != 1 || block.end != checkpoint.offset) {
                continue;
            }
            const auto content_end = part_boundary(i, message.parts.size() - 1U);
            if (content_end && *content_end > block.content_begin &&
                *content_end == block.content_end &&
                !text::overlaps(output.literal_spans, *content_end, block.end)) {
                checkpoint.recovery_offset = *content_end;
                break;
            }
        }
    }
    for (std::size_t i = 0; i < options.cache_markers.size(); ++i) {
        const auto& marker = options.cache_markers[i];
        switch (marker.location) {
        case PromptCacheMarkerLocation::MessageBoundary:
            if (marker.after_message_count < result.message_boundaries.size()) {
                auto& boundary = result.message_boundaries[marker.after_message_count];
                if (!boundary) boundary = prefix(marker.after_message_count);
                result.cache_boundaries[i] = boundary;
                const bool automatic =
                    has_shared_candidate_evidence(
                        marker.evidence, SharedCandidateEvidence::DefaultAutomatic |
                                             SharedCandidateEvidence::RequestedAutomatic) &&
                    !has_shared_candidate_evidence(marker.evidence,
                                                   SharedCandidateEvidence::ExplicitBoundary);
                if (!boundary && automatic && marker.after_message_count) {
                    // Some templates have no independently serializable message closing.
                    // An automatic hint can still retain the proven content prefix.
                    const auto message = marker.after_message_count - 1U;
                    if (!messages[message].parts.empty())
                        result.cache_boundaries[i] =
                            part_boundary(message, messages[message].parts.size() - 1U);
                }
            }
            break;
        case PromptCacheMarkerLocation::LeadingInstructionBoundary:
            if (instruction(messages.front().role))
                result.cache_boundaries[i] = models::frontend::source_boundary(
                    output, sources.front().tag, marker.leading_instruction_bytes);
            break;
        case PromptCacheMarkerLocation::ToolBoundary:
            if (marker.after_tool_count && marker.after_tool_count <= tool_tags.size()) {
                if (const auto region = models::frontend::unique_output_region(
                        output, tool_tags[marker.after_tool_count - 1]))
                    result.cache_boundaries[i] = region->end;
            }
            break;
        case PromptCacheMarkerLocation::MessageContentBoundary: {
            if (!marker.after_message_count || marker.after_message_count > messages.size()) break;
            const auto index    = marker.after_message_count - 1U;
            const auto& message = messages[index];
            if (message.role != ChatRole::Assistant || message.has_media() ||
                marker.after_message_part_count > message.parts.size() ||
                marker.after_tool_call_count > message.tool_calls.size() ||
                marker.reasoning_bytes > message.reasoning_content.size())
                break;
            Json probe  = context;
            auto& turns = probe["messages"];
            turns.erase(turns.begin() + index + 1U, turns.end());
            auto& tail = turns[index];
            std::string content;
            for (std::size_t part = 0; part < marker.after_message_part_count; ++part)
                content += message.parts[part].text;
            tail["content"] = std::move(content);
            if (tail.contains("reasoning_content"))
                tail["reasoning_content"] =
                    message.reasoning_content.substr(0, marker.reasoning_bytes);
            if (tail.contains("tool_calls")) {
                auto& calls = tail["tool_calls"];
                calls.erase(calls.begin() + marker.after_tool_call_count, calls.end());
            }
            probe["add_generation_prompt"] = false;
            try {
                const auto partial      = compiled_.program.render(probe, execution);
                std::size_t media_count = 0;
                for (std::size_t i = 0; i <= index; ++i)
                    media_count +=
                        std::count_if(sources[i].part_media.begin(), sources[i].part_media.end(),
                                      [](const auto& item) { return item.has_value(); });
                const auto partial_layout = inspect_prompt_layout(
                    partial, std::span<const models::frontend::Modality>(media).first(media_count));
                if (!partial_layout.messages.empty()) {
                    const auto& last = partial_layout.messages.back();
                    if (last.role == ChatRole::Assistant && last.closed &&
                        models::frontend::same_prefix(output, partial, last.content_end))
                        result.cache_boundaries[i] = last.content_end;
                }
            } catch (const RequestError&) { throw; } catch (const std::invalid_argument&) {
            }
            break;
        }
        case PromptCacheMarkerLocation::MessagePartBoundary:
            if (marker.after_message_count && marker.after_message_count <= sources.size() &&
                marker.after_message_part_count) {
                result.cache_boundaries[i] = part_boundary(marker.after_message_count - 1U,
                                                           marker.after_message_part_count - 1U);
            }
            break;
        }
    }
    result.text          = std::move(output.text);
    result.literal_spans = std::move(output.literal_spans);
    return result;
}

} // namespace ninfer::models::qwen::frontend
