#include "models/inkling2/frontend/prompt_layout.h"

#include "models/inkling2/frontend/resources.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::inkling2 {
namespace {
using Span            = text::ByteSpan;
constexpr auto absent = std::string::npos;

std::size_t find_control(const text::TemplateOutput& output, std::string_view marker,
                         std::size_t begin, std::size_t end) {
    for (auto p = output.text.find(marker, begin); p != absent && p + marker.size() <= end;
         p      = output.text.find(marker, p + marker.size())) {
        if (!text::overlaps(output.literal_spans, p, p + marker.size())) return p;
    }
    return absent;
}

bool contains(Span outer, Span inner) {
    return outer.begin <= inner.begin && inner.end <= outer.end;
}

void include(std::optional<Span>& extent, std::optional<Span> value) {
    if (!value) return;
    if (!extent) {
        extent = value;
    } else {
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
    ChatRole role = ChatRole::User;
    Span bytes;
    bool closed_turn = false;
};

struct Layout {
    std::vector<Block> blocks;
    std::vector<std::optional<Span>> messages;
    std::vector<std::optional<Span>> tools;
    std::vector<std::size_t> semantic;
};

Layout inspect(const RenderedChat& rendered, std::span<const ChatMessage> messages,
               bool continuation) {
    const auto& out = rendered.output;
    Layout layout;

    for (std::size_t p = 0; (p = out.text.find("<|message_", p)) != absent; ++p) {
        const auto end = out.text.find("|>", p + 10);
        if (end == absent) break;
        const std::size_t marker_end = end + 2;
        if (text::overlaps(out.literal_spans, p, marker_end)) continue;
        const auto token = std::string_view(out.text).substr(p, marker_end - p);
        std::optional<ChatRole> role;
        if (token == kMessageSystem)
            role = ChatRole::System;
        else if (token == kMessageUser)
            role = ChatRole::User;
        else if (token == kMessageModel)
            role = ChatRole::Assistant;
        else if (token == kMessageTool)
            role = ChatRole::Tool;
        if (!role) continue;
        if (!layout.blocks.empty()) layout.blocks.back().bytes.end = p;
        layout.blocks.push_back({*role, {p, out.text.size()}, false});
    }

    // Mark blocks closed by <|content_model_end_sampling|> or <|end_message|>.
    for (auto& block : layout.blocks) {
        const auto end_sample =
            find_control(out, kContentModelEndSample, block.bytes.begin, block.bytes.end);
        if (end_sample != absent) block.closed_turn = true;
    }

    const Sources sources(out);
    layout.messages.resize(messages.size());
    layout.tools.resize(rendered.tools.size());

    // Map each input message to the span covering all of its rendered Inkling blocks.
    for (std::size_t i = 0; i < messages.size(); ++i) {
        const auto extent = sources.message(rendered.messages[i]);
        if (!extent) continue;
        std::optional<Span> msg_span;
        for (const auto& block : layout.blocks) {
            if (block.bytes.end > extent->begin && block.bytes.begin < extent->end) {
                include(msg_span, block.bytes);
            }
        }
        if (msg_span) layout.messages[i] = msg_span;
    }

    const bool trailing_open_assistant =
        continuation || (rendered.initial_block != InitialBlock::BetweenBlocks &&
                         !layout.blocks.empty() && layout.blocks.back().role == ChatRole::Assistant &&
                         !layout.blocks.back().closed_turn);
    for (std::size_t i = 0; i < layout.blocks.size(); ++i) {
        const auto& block = layout.blocks[i];
        if (trailing_open_assistant && i + 1 == layout.blocks.size()) continue;
        layout.semantic.push_back(block.bytes.end);
    }

    for (std::size_t t = 0; t < rendered.tools.size(); ++t) {
        const auto extent = sources.at(rendered.tools[t]);
        if (extent) layout.tools[t] = extent;
    }
    return layout;
}

bool ordered_prefix(const Layout& layout, std::size_t count, std::size_t end, bool partial) {
    for (std::size_t i = 0; i < layout.messages.size(); ++i) {
        if (!layout.messages[i]) continue;
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
    if (end) {
        for (std::size_t i = count; i < layout.tools.size(); ++i)
            if (layout.tools[i] && layout.tools[i]->begin < *end) return {};
    }
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
    result.semantic = layout.semantic;
    for (std::size_t i = 0; i < markers.size(); ++i) {
        const auto same_location = [&](PromptCacheMarker other) {
            other.kind     = markers[i].kind;
            other.evidence = markers[i].evidence;
            return other == markers[i];
        };
        const auto previous = std::find_if(markers.begin(), markers.begin() + i, same_location);
        result.markers.push_back(
            previous != markers.begin() + i
                ? result.markers[previous - markers.begin()]
                : resolve_marker(rendered, layout, messages, continuation, markers[i], probes));
    }
    if (!rendered.tools.empty()) {
        // The tool declaration block ends at the first <|end_message|> after <|content_xml|>.
        const auto xml_pos =
            find_control(rendered.output, kContentXml, 0, rendered.output.text.size());
        if (xml_pos != absent) {
            const auto end_pos = find_control(rendered.output, kEndMessage, xml_pos,
                                              rendered.output.text.size());
            if (end_pos != absent) result.structural.push_back(end_pos + kEndMessage.size());
        }
    }
    std::size_t leading = 0;
    while (leading < messages.size() && (messages[leading].role == ChatRole::System ||
                                         messages[leading].role == ChatRole::Developer)) {
        ++leading;
    }
    if (const auto end = message_end(layout, leading)) result.structural.push_back(*end);
    if (!layout.blocks.empty()) {
        const bool trailing_open_assistant =
            continuation || (rendered.initial_block != InitialBlock::BetweenBlocks &&
                             layout.blocks.back().role == ChatRole::Assistant &&
                             !layout.blocks.back().closed_turn);
        auto frontier = trailing_open_assistant ? layout.blocks.back().bytes.begin
                                                : rendered.output.text.size();
        auto kind     = models::frontend::RewriteCheckpointKind::ResponseReplay;
        std::optional<TurnProbe> next_turn;
        if (!continuation && probes.next_turn) next_turn = probes.next_turn();
        if (next_turn && !next_turn->retains_response) {
            kind              = models::frontend::RewriteCheckpointKind::TurnClosure;
            std::size_t first = 0;
            for (std::size_t i = 0; i < messages.size(); ++i)
                if (messages[i].role == ChatRole::User) first = i + 1;
            for (auto i = first; i < messages.size(); ++i) {
                if (messages[i].role == ChatRole::Assistant && layout.messages[i]) {
                    frontier = layout.messages[i]->begin;
                    break;
                }
            }
        }
        if (frontier && (!next_turn || models::frontend::same_prefix(rendered.output,
                                                                     next_turn->output, frontier)))
            result.rewrite = RewriteByteSpec{kind, frontier};
    }
    return result;
}

} // namespace ninfer::models::inkling2
