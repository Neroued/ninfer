#include "models/glm5next/frontend/prompt_layout.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::glm5next {
namespace {
using Span            = text::ByteSpan;
constexpr auto absent = std::string::npos;

std::size_t control(const text::TemplateOutput& output, std::string_view marker, std::size_t begin,
                    std::size_t end) {
    for (auto p = output.text.find(marker, begin); p != absent && p + marker.size() <= end;
         p      = output.text.find(marker, p + marker.size()))
        if (!text::overlaps(output.literal_spans, p, p + marker.size())) return p;
    return absent;
}

std::vector<Span> enclosed(const text::TemplateOutput& output, Span range, std::string_view open,
                           std::string_view close) {
    std::vector<Span> result;
    for (auto p = control(output, open, range.begin, range.end); p != absent;) {
        const auto q = control(output, close, p + open.size(), range.end);
        if (q == absent) break;
        result.push_back({p, q + close.size()});
        p = control(output, open, q + close.size(), range.end);
    }
    return result;
}

bool contains(Span outer, Span inner) {
    return outer.begin <= inner.begin && inner.end <= outer.end;
}

void include(std::optional<Span>& extent, std::optional<Span> value) {
    if (!value) return;
    if (!extent)
        extent = value;
    else {
        extent->begin = std::min(extent->begin, value->begin);
        extent->end   = std::max(extent->end, value->end);
    }
}

class Sources {
public:
    explicit Sources(const text::TemplateOutput& output) {
        for (const auto& region : output.regions) {
            if (region.tag >= values.size()) values.resize(std::size_t(region.tag) + 1);
            include(values[region.tag], Span{region.begin, region.end});
        }
    }

    std::optional<Span> at(std::uint32_t tag) const {
        return tag && tag < values.size() ? values[tag] : std::nullopt;
    }

    std::optional<Span> message(const MessageSources& source) const {
        std::optional<Span> result;
        if (source.content) include(result, at(source.content->tag));
        for (const auto& part : source.parts)
            if (part) include(result, at(part->tag));
        include(result, at(source.reasoning.tag));
        for (const auto tag : source.tool_calls) include(result, at(tag));
        return result;
    }
private:
    std::vector<std::optional<Span>> values;
};

struct Block {
    ChatRole role;
    Span bytes;
    std::size_t content;
};

struct Layout {
    std::vector<Block> blocks;
    std::vector<std::optional<Span>> messages;
    std::vector<std::optional<Span>> tools;
    std::vector<std::size_t> semantic;
    std::vector<std::size_t> execution;
};

Layout inspect(const RenderedChat& rendered, std::span<const ChatMessage> messages,
               bool continuation) {
    const auto& out = rendered.output;
    Layout layout;
    for (std::size_t p = 0; (p = out.text.find("<|", p)) != absent; ++p) {
        const auto end = out.text.find("|>", p + 2);
        if (end == absent) break;
        if (text::overlaps(out.literal_spans, p, end + 2)) continue;
        const auto name = std::string_view(out.text).substr(p + 2, end - p - 2);
        std::optional<ChatRole> role;
        if (name == "system")
            role = ChatRole::System;
        else if (name == "user")
            role = ChatRole::User;
        else if (name == "assistant")
            role = ChatRole::Assistant;
        else if (name == "observation")
            role = ChatRole::Tool;
        if (!role) continue;
        if (!layout.blocks.empty()) layout.blocks.back().bytes.end = p;
        layout.blocks.push_back({*role, {p, out.text.size()}, end + 2});
    }
    const Sources sources(out);
    layout.messages.resize(messages.size());
    layout.tools.resize(rendered.tools.size());
    std::vector<Span> declarations;

    struct Group {
        std::size_t begin, end;
        ChatRole role;
    };

    std::vector<Group> groups;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        const auto role =
            messages[i].role == ChatRole::Developer ? ChatRole::System : messages[i].role;
        if (role == ChatRole::Tool && !groups.empty() && groups.back().role == role)
            groups.back().end = i + 1;
        else
            groups.push_back({i, i + 1, role});
    }
    const auto input_blocks = layout.blocks.size() - (!continuation && !layout.blocks.empty());
    if (groups.size() <= input_blocks) {
        const auto first = input_blocks - groups.size();
        for (std::size_t g = 0; g < groups.size(); ++g) {
            const auto group  = groups[g];
            const auto& block = layout.blocks[first + g];
            if (block.role != group.role) continue;
            const auto responses =
                block.role == ChatRole::Tool
                    ? enclosed(out, block.bytes, "<tool_response>", "</tool_response>")
                    : std::vector<Span>{};
            for (auto i = group.begin; i < group.end; ++i) {
                const auto extent = sources.message(rendered.messages[i]);
                if (extent && !contains(block.bytes, *extent)) continue;
                layout.messages[i] = block.bytes;
                if (block.role == ChatRole::Tool && extent) {
                    const auto part =
                        std::lower_bound(responses.begin(), responses.end(), extent->begin,
                                         [](Span span, std::size_t pos) { return span.end < pos; });
                    if (part != responses.end() && contains(*part, *extent))
                        layout.messages[i] = *part;
                }
            }
        }
    }
    for (std::size_t i = 0; i < layout.blocks.size(); ++i) {
        const auto& block = layout.blocks[i];
        if (continuation || i + 1 < layout.blocks.size())
            layout.semantic.push_back(block.bytes.end);
        if (block.role == ChatRole::Assistant) {
            layout.execution.push_back(block.content);
            if (control(out, "<think>", block.content, block.bytes.end) == block.content) {
                layout.execution.push_back(block.content + 7);
                const auto close = control(out, "</think>", block.content + 7, block.bytes.end);
                if (close != absent) layout.execution.push_back(close + 8);
            }
            for (auto span : enclosed(out, block.bytes, "<tool_call>", "</tool_call>"))
                layout.semantic.push_back(span.end);
        } else if (block.role == ChatRole::Tool) {
            for (auto span : enclosed(out, block.bytes, "<tool_response>", "</tool_response>"))
                layout.semantic.push_back(span.end);
        } else if (block.role == ChatRole::System) {
            for (auto list : enclosed(out, block.bytes, "<tools>", "</tools>")) {
                // Values and keys are literal input; these braces belong to the template.
                const auto objects = enclosed(out, list, "{", "}");
                declarations.insert(declarations.end(), objects.begin(), objects.end());
            }
        }
    }
    for (std::size_t t = 0; t < rendered.tools.size(); ++t) {
        const auto extent = sources.at(rendered.tools[t]);
        if (!extent) continue;
        const auto object =
            std::lower_bound(declarations.begin(), declarations.end(), extent->begin,
                             [](Span span, std::size_t pos) { return span.end < pos; });
        if (object != declarations.end() && contains(*object, *extent)) layout.tools[t] = *object;
    }
    return layout;
}

bool ordered_prefix(const Layout& layout, std::size_t count, std::size_t end, bool partial) {
    for (std::size_t i = 0; i < layout.messages.size(); ++i) {
        if (!layout.messages[i]) return false;
        const auto span = *layout.messages[i];
        if (i < count - std::size_t(partial) && span.end > end) return false;
        if (i >= count && span.begin < end) return false;
    }
    return true;
}

std::optional<std::size_t> message_end(const Layout& layout, std::size_t count) {
    if (!count) return {};
    std::size_t end = 0;
    for (std::size_t i = 0; i < count; ++i) {
        if (!layout.messages[i]) return {};
        end = std::max(end, layout.messages[i]->end);
    }
    return ordered_prefix(layout, count, end, false) ? std::optional(end) : std::nullopt;
}

std::optional<std::size_t> tool_end(const Layout& layout, std::size_t count) {
    std::optional<std::size_t> end;
    for (std::size_t i = 0; i < count; ++i)
        if (layout.tools[i]) end = std::max(end.value_or(0), layout.tools[i]->end);
    if (end)
        for (std::size_t i = count; i < layout.tools.size(); ++i)
            if (layout.tools[i] && layout.tools[i]->begin < *end) return {};
    return end;
}

std::optional<std::size_t> resolve_marker(const RenderedChat& rendered, const Layout& layout,
                                          std::span<const ChatMessage> messages, bool continuation,
                                          const PromptCacheMarker& marker,
                                          const PromptProbes& probes) {
    if (marker.location == PromptCacheMarkerLocation::ToolBoundary) {
        if (marker.after_tool_count > layout.tools.size())
            throw std::invalid_argument("cache tool marker is outside the declarations");
        return tool_end(layout, marker.after_tool_count);
    }
    if (marker.location == PromptCacheMarkerLocation::LeadingInstructionBoundary) {
        if (messages.empty() || (messages.front().role != ChatRole::System &&
                                 messages.front().role != ChatRole::Developer))
            throw std::invalid_argument(
                "leading-instruction marker requires a leading instruction");
        const auto& source = rendered.messages.front().content;
        if (!source || marker.leading_instruction_bytes > source->bytes.end)
            throw std::invalid_argument("cache leading-instruction marker is outside the text");
        const auto end = models::frontend::source_boundary(rendered.output, source->tag,
                                                           marker.leading_instruction_bytes);
        return end && ordered_prefix(layout, 1, *end, true) ? end : std::nullopt;
    }
    const auto count = std::size_t(marker.after_message_count);
    if (count > messages.size())
        throw std::invalid_argument("cache message marker is outside the input");
    if (marker.location == PromptCacheMarkerLocation::MessageBoundary)
        return message_end(layout, count);
    if (!count) throw std::invalid_argument("cache marker requires a message");
    const auto& message = messages[count - 1];
    const auto& sources = rendered.messages[count - 1];
    if (marker.after_message_part_count > message.parts.size())
        throw std::invalid_argument("cache part marker is outside the message");
    if (marker.location == PromptCacheMarkerLocation::MessagePartBoundary) {
        if (!marker.after_message_part_count)
            throw std::invalid_argument("cache part count is zero");
        const auto source = sources.parts[marker.after_message_part_count - 1];
        if (!source) return {};
        const auto end =
            models::frontend::source_boundary(rendered.output, source->tag, source->bytes.end);
        return end && ordered_prefix(layout, count, *end, true) ? end : std::nullopt;
    }
    if (marker.location != PromptCacheMarkerLocation::MessageContentBoundary)
        throw std::invalid_argument("invalid cache marker location");
    if (message.role != ChatRole::Assistant ||
        marker.after_tool_call_count > message.tool_calls.size() ||
        marker.reasoning_bytes > message.reasoning_content.size())
        throw std::invalid_argument("cache assistant-content marker is outside the message");
    if (!sources.content) return {};
    const auto prefix  = probes.content_prefix(marker);
    const auto partial = inspect(prefix, messages, continuation);
    const auto end     = message_end(partial, count);
    return end && ordered_prefix(layout, count, *end, true) &&
                   models::frontend::same_prefix(rendered.output, prefix.output, *end)
               ? end
               : std::nullopt;
}
} // namespace

PromptBoundaries resolve_prompt_boundaries(const RenderedChat& rendered,
                                           std::span<const ChatMessage> messages, bool continuation,
                                           std::span<const PromptCacheMarker> markers,
                                           const PromptProbes& probes) {
    const auto layout = inspect(rendered, messages, continuation);
    PromptBoundaries result;
    result.semantic  = layout.semantic;
    result.execution = layout.execution;
    for (const auto& marker : markers)
        result.markers.push_back(
            resolve_marker(rendered, layout, messages, continuation, marker, probes));
    if (const auto end = tool_end(layout, layout.tools.size())) result.structural.push_back(*end);
    std::size_t leading = 0;
    while (leading < messages.size() && (messages[leading].role == ChatRole::System ||
                                         messages[leading].role == ChatRole::Developer))
        ++leading;
    if (const auto end = message_end(layout, leading)) result.structural.push_back(*end);
    if (!layout.blocks.empty()) {
        auto frontier = layout.blocks.back().bytes.begin;
        auto kind     = models::frontend::RewriteCheckpointKind::ResponseReplay;
        std::optional<TurnProbe> next_turn;
        if (!continuation) next_turn = probes.next_turn();
        if (next_turn && !next_turn->retains_response) {
            kind              = models::frontend::RewriteCheckpointKind::TurnClosure;
            std::size_t first = 0;
            for (std::size_t i = 0; i < messages.size(); ++i)
                if (messages[i].role == ChatRole::User) first = i + 1;
            for (auto i = first; i < messages.size(); ++i)
                if (messages[i].role == ChatRole::Assistant && layout.messages[i]) {
                    frontier = layout.messages[i]->begin;
                    break;
                }
        }
        if (frontier && (!next_turn || models::frontend::same_prefix(rendered.output,
                                                                     next_turn->output, frontier)))
            result.rewrite = RewriteByteSpec{kind, frontier};
    }
    return result;
}

} // namespace ninfer::models::glm5next
