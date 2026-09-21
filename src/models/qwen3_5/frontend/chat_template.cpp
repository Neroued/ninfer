#include "models/qwen3_5/frontend/chat_template.h"

#include "models/qwen3_5/frontend/media_cache.h"
#include "models/qwen3_5/frontend/prompt_layout.h"
#include "text/unicode.h"

#include <algorithm>
#include <map>
#include <stdexcept>

namespace ninfer::models::qwen3_5::frontend {
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

void merge_option(Json& context, std::string_view key, const Json& value) {
    const std::string name(key);
    if (context.contains(name) && !context[name].is_null() && context[name] != value) {
        throw std::invalid_argument("conflicting chat template parameter: " + name);
    }
    context[name] = value;
}

Json template_parameters(const ChatRenderOptions& options, const Json& special_tokens) {
    Json context = options.chat_template_kwargs_json.empty()
                       ? Json::object()
                       : Json::parse(options.chat_template_kwargs_json);
    if (!context.is_object()) throw std::invalid_argument("chat_template_kwargs must be an object");
    for (const auto& item : context.items()) {
        const auto& key = item.key();
        if (key == "messages" || key == "tools" || key == "add_generation_prompt" ||
            key == "continue_final_message" || key == "bos_token" || key == "eos_token" ||
            key == "pad_token" || key == "unk_token" || key == "sep_token" || key == "cls_token" ||
            key == "mask_token" || key == "additional_special_tokens" ||
            special_tokens.contains(key)) {
            throw std::invalid_argument("chat_template_kwargs cannot override " + key);
        }
    }
    for (const char* name :
         {"enable_thinking", "preserve_thinking", "reasoning_effort", "add_vision_id"}) {
        if (context.contains(name) && context[name].is_null()) context.erase(name);
    }
    if (options.enable_thinking) merge_option(context, "enable_thinking", *options.enable_thinking);
    if (options.preserve_thinking)
        merge_option(context, "preserve_thinking", *options.preserve_thinking);
    if (options.reasoning_effort) {
        const auto name = reasoning_effort_name(*options.reasoning_effort);
        if (name.empty()) throw std::invalid_argument("invalid reasoning effort");
        merge_option(context, "reasoning_effort", name);
    }
    if (options.add_vision_id) merge_option(context, "add_vision_id", true);
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
            throw std::invalid_argument("reasoning_effort conflicts with enable_thinking");
        }
        // The protocol's 'none' is an explicit disable. Other efforts retain template defaults.
        if (!thinking) context["enable_thinking"] = false;
    }
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

// Equal text alone does not prove equal control-token interpretation.
bool same_prefix(const text::TemplateOutput& lhs, const text::TemplateOutput& rhs,
                 std::size_t size) {
    if (lhs.text.size() < size || rhs.text.size() < size ||
        std::string_view(lhs.text).substr(0, size) != std::string_view(rhs.text).substr(0, size))
        return false;
    for (std::size_t i = 0;; ++i) {
        const bool left  = i < lhs.literal_spans.size() && lhs.literal_spans[i].begin < size;
        const bool right = i < rhs.literal_spans.size() && rhs.literal_spans[i].begin < size;
        if (!left || !right) return left == right;
        const auto a = lhs.literal_spans[i];
        const auto b = rhs.literal_spans[i];
        if (a.begin != b.begin || std::min(a.end, size) != std::min(b.end, size)) return false;
    }
}

bool real_user(const ChatMessage& message) {
    if (message.role != ChatRole::User) return false;
    if (message.has_media()) return true;
    std::string content;
    for (const auto& part : message.parts) content += part.text;
    const auto chars  = text::unicode_internal::utf8_codepoints(content, "message content");
    std::size_t begin = 0, end = chars.size();
    while (begin < end && text::unicode_internal::is_whitespace(chars[begin].value)) ++begin;
    while (end > begin && text::unicode_internal::is_whitespace(chars[end - 1].value)) --end;
    const auto offset = begin < chars.size() ? chars[begin].offset : content.size();
    const auto limit  = end < chars.size() ? chars[end].offset : content.size();
    const auto body   = std::string_view(content).substr(offset, limit - offset);
    return !(body.starts_with("<tool_response>") && body.ends_with("</tool_response>"));
}

} // namespace

bool ChatMessage::has_media() const noexcept {
    return std::any_of(parts.begin(), parts.end(),
                       [](const ChatPart& part) { return part.kind != ChatPartKind::Text; });
}

CompiledChatTemplate CompiledChatTemplate::resolve(std::string_view source, std::string source_name,
                                                   Json special_tokens) {
    return CompiledChatTemplate(text::JinjaTemplate(std::string(source), std::move(source_name)),
                                std::move(special_tokens));
}

RenderedChat CompiledChatTemplate::render(const std::vector<ChatMessage>& messages,
                                          ChatRenderOptions options,
                                          const PreparationControl& control) const {
    if (messages.empty()) throw std::invalid_argument("chat requires at least one message");
    check_preparation_control(control, "chat template");
    const bool continuation =
        options.continuation == PromptContinuationMode::ContinueFinalAssistant;
    Json context = template_parameters(options, special_tokens_);
    if (continuation) {
        const auto& final = messages.back();
        if (final.role != ChatRole::Assistant || final.has_media()) {
            throw std::invalid_argument("assistant continuation requires a final assistant message without media");
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
    std::vector<Modality> media;
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
            for (const auto& part : message.parts) {
                content += part.text;
                origin.part_ends.push_back(content.size());
            }
            value["content"] = std::move(content);
            origin.tag       = tag(pointer + "/content");
        } else {
            value["content"] = Json::array();
            for (std::size_t j = 0; j < message.parts.size(); ++j) {
                const auto& part = message.parts[j];
                const auto item  = pointer + "/content/" + std::to_string(j);
                if (part.kind == ChatPartKind::Text) {
                    value["content"].push_back({{"type", "text"}, {"text", part.text}});
                    origin.part_tags.push_back(tag(item + "/text"));
                    origin.part_media.push_back(std::nullopt);
                } else {
                    const bool image = part.kind == ChatPartKind::Image;
                    value["content"].push_back({{"type", image ? "image" : "video"}});
                    origin.part_media.push_back(media.size());
                    media.push_back(image ? Modality::Image : Modality::Video);
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
    for (const auto& item : special_tokens_.items()) control_variables.push_back(item.key());
    text::TemplateRenderOptions execution{
        .checkpoint        = [&] { check_preparation_control(control, "chat template"); },
        .regions           = regions,
        .control_variables = control_variables};
    auto output = compiled_.render(context, execution);
    auto layout = inspect_prompt_layout(output, media);
    if (continuation) {
        const auto content = unique_output_region(output, sources.back().tag);
        if (!content || layout.messages.empty())
            throw std::invalid_argument(
                "chat template does not expose the final assistant content for continuation");
        const auto& final = layout.messages.back();
        if (final.role != ChatRole::Assistant || content->begin < final.content_begin ||
            content->end > final.content_end) {
            throw std::invalid_argument(
                "chat template cannot continue the final assistant content unambiguously");
        }
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
            const auto rendered = compiled_.render(probe, execution);
            if (same_prefix(output, rendered, rendered.text.size()))
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
            const auto next_turn     = compiled_.render(probe, probe_options);
            const auto history_bytes = generation_begin.value_or(output.text.size());
            retain_open_turn         = same_prefix(next_turn, output, history_bytes);
        } catch (const RequestError&) { throw; } catch (const std::invalid_argument&) {
            retain_open_turn = false;
        }
    }
    if (generation_begin && *generation_begin > 0) {
        result.rewrite_checkpoint = RewriteCheckpointByteSpec{
            .kind   = continuation || retain_open_turn ? RewriteCheckpointKind::ResponseReplay
                                                       : RewriteCheckpointKind::TurnClosure,
            .offset = *generation_begin};
    }
    if (!continuation && !retain_open_turn && first_tail_assistant) {
        const auto begin = layout.messages[*first_tail_assistant].begin;
        if (begin > 0)
            result.rewrite_checkpoint = RewriteCheckpointByteSpec{
                .kind = RewriteCheckpointKind::TurnClosure, .offset = begin};
    }
    for (std::size_t i = 0; i < options.cache_markers.size(); ++i) {
        const auto& marker = options.cache_markers[i];
        switch (marker.location) {
        case PromptCacheMarkerLocation::MessageBoundary:
            if (marker.after_message_count < result.message_boundaries.size()) {
                auto& boundary = result.message_boundaries[marker.after_message_count];
                if (!boundary) boundary = prefix(marker.after_message_count);
                result.cache_boundaries[i] = boundary;
            }
            break;
        case PromptCacheMarkerLocation::LeadingInstructionBoundary:
            if (instruction(messages.front().role))
                result.cache_boundaries[i] =
                    source_boundary(output, sources.front().tag, marker.leading_instruction_bytes);
            break;
        case PromptCacheMarkerLocation::ToolBoundary:
            if (marker.after_tool_count && marker.after_tool_count <= tool_tags.size()) {
                if (const auto region =
                        unique_output_region(output, tool_tags[marker.after_tool_count - 1]))
                    result.cache_boundaries[i] = region->end;
            }
            break;
        case PromptCacheMarkerLocation::MessagePartBoundary:
            if (marker.after_message_count && marker.after_message_count <= sources.size() &&
                marker.after_message_part_count) {
                const auto& origin = sources[marker.after_message_count - 1];
                const auto part    = marker.after_message_part_count - 1;
                if (part < origin.part_ends.size())
                    result.cache_boundaries[i] =
                        source_boundary(output, origin.tag, origin.part_ends[part]);
                else if (part < origin.part_tags.size()) {
                    if (origin.part_tags[part]) {
                        if (const auto region =
                                unique_output_region(output, origin.part_tags[part]))
                            result.cache_boundaries[i] = region->end;
                    } else if (origin.part_media[part]) {
                        result.cache_boundaries[i] =
                            layout.media_placeholders[*origin.part_media[part]].bytes.end +
                            std::string_view("<|vision_end|>").size();
                    }
                }
            }
            break;
        }
    }
    result.text          = std::move(output.text);
    result.literal_spans = std::move(output.literal_spans);
    return result;
}

} // namespace ninfer::models::qwen3_5::frontend
