#include "models/common/frontend/prompt.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::frontend {
namespace {
void validate_session_key(const std::optional<std::string>& key) {
    if (key && (key->empty() || key->size() > kPreparedSessionKeyCapacity))
        throw std::invalid_argument("context cache session_key must contain 1 to 256 bytes");
}
} // namespace

void validate_context_cache_hints(const ContextCacheHints& hints, std::size_t message_count) {
    if (hints.markers.size() > kMaximumExplicitPromptCacheMarkers)
        throw std::invalid_argument("PromptInput supports at most four explicit cache markers");
    validate_session_key(hints.session_key);
    for (const auto marker : hints.markers) {
        if (marker.kind != PromptCacheMarkerKind::SharedStablePrefix &&
            marker.kind != PromptCacheMarkerKind::PrivateLongAnchor)
            throw std::invalid_argument("context cache marker kind is invalid");
        if (marker.kind == PromptCacheMarkerKind::SharedStablePrefix &&
            marker.evidence == SharedCandidateEvidence::None)
            throw std::invalid_argument("shared context cache marker evidence is empty");
        if (marker.after_message_count > message_count)
            throw std::invalid_argument("context cache message marker is outside the input");
        if (marker.location != PromptCacheMarkerLocation::MessageContentBoundary &&
            (marker.after_tool_call_count || marker.reasoning_bytes))
            throw std::invalid_argument("content coordinates require a message-content marker");
        switch (marker.location) {
        case PromptCacheMarkerLocation::MessageBoundary:
            if (marker.leading_instruction_bytes || marker.after_tool_count ||
                marker.after_message_part_count)
                throw std::invalid_argument("context cache message marker is invalid");
            break;
        case PromptCacheMarkerLocation::MessageContentBoundary:
            if (!marker.after_message_count || marker.leading_instruction_bytes ||
                marker.after_tool_count)
                throw std::invalid_argument("context cache message-content marker is invalid");
            break;
        case PromptCacheMarkerLocation::MessagePartBoundary:
            if (!marker.after_message_count || !marker.after_message_part_count ||
                marker.leading_instruction_bytes || marker.after_tool_count)
                throw std::invalid_argument("context cache message-part marker is invalid");
            break;
        case PromptCacheMarkerLocation::LeadingInstructionBoundary:
            if (marker.after_message_count || !marker.leading_instruction_bytes ||
                marker.after_tool_count || marker.after_message_part_count)
                throw std::invalid_argument("context cache leading-instruction marker is invalid");
            break;
        case PromptCacheMarkerLocation::ToolBoundary:
            if (marker.after_message_count || marker.leading_instruction_bytes ||
                !marker.after_tool_count || marker.after_message_part_count)
                throw std::invalid_argument("context cache tool marker is invalid");
            break;
        default:
            throw std::invalid_argument("context cache marker location is invalid");
        }
    }
}

PreparedContextCache
prepare_context_cache(const ContextCacheHints& hints,
                      std::span<const std::optional<std::uint32_t>> marker_frontiers,
                      std::span<const std::uint32_t> semantic_frontiers,
                      std::span<const std::uint32_t> structural_frontiers) {
    if (hints.markers.size() != marker_frontiers.size())
        throw std::logic_error("resolved cache marker count changed during preparation");
    PreparedContextCache result;
    result.automatic_checkpoints = hints.allow_engine_automatic_shared_prefixes;
    result.update_session_index  = hints.update_session_index;
    if (hints.session_key) {
        validate_session_key(hints.session_key);
        PreparedSessionKey key;
        key.size = static_cast<std::uint16_t>(hints.session_key->size());
        std::copy(hints.session_key->begin(), hints.session_key->end(), key.bytes.begin());
        result.session_key = key;
    }
    const auto add = [&](PromptCacheMarkerKind kind, SharedCandidateEvidence evidence,
                         std::uint32_t frontier, std::uint32_t order) {
        if (!frontier) return;
        const auto same = std::find_if(
            result.opportunities.begin(), result.opportunities.end(),
            [&](const auto& item) { return item.kind == kind && item.frontier == frontier; });
        if (same == result.opportunities.end())
            result.opportunities.push_back({kind, evidence, frontier, order});
        else if (kind == PromptCacheMarkerKind::SharedStablePrefix)
            same->evidence |= evidence;
    };
    for (std::size_t i = 0; i < marker_frontiers.size(); ++i)
        if (marker_frontiers[i])
            add(hints.markers[i].kind, hints.markers[i].evidence, *marker_frontiers[i], i);
    if (result.automatic_checkpoints) {
        result.semantic_boundaries.assign(semantic_frontiers.begin(), semantic_frontiers.end());
        std::erase(result.semantic_boundaries, 0);
        std::sort(result.semantic_boundaries.begin(), result.semantic_boundaries.end());
        result.semantic_boundaries.erase(
            std::unique(result.semantic_boundaries.begin(), result.semantic_boundaries.end()),
            result.semantic_boundaries.end());
        auto order = static_cast<std::uint32_t>(hints.markers.size());
        for (const auto frontier : structural_frontiers)
            add(PromptCacheMarkerKind::SharedStablePrefix,
                SharedCandidateEvidence::EngineStructural, frontier, order++);
    }
    return result;
}

PromptPreparationStats PrepareStats::public_stats() const noexcept {
    return {.seconds                       = seconds,
            .media_preprocess_seconds      = media_preprocess_seconds,
            .media_preprocess_work_seconds = media_preprocess_work_seconds,
            .tokenize_seconds              = tokenize_seconds,
            .media_items                   = media_items,
            .media_bytes                   = media_bytes,
            .raw_patches                   = raw_patches,
            .vision_tokens                 = vision_tokens,
            .patch_bytes                   = patch_bytes,
            .media_cache_hits              = media_cache_hits,
            .media_cache_misses            = media_cache_misses,
            .media_singleflight_waits      = media_singleflight_waits,
            .built_patch_bytes             = built_patch_bytes,
            .reused_patch_bytes            = reused_patch_bytes};
}

} // namespace ninfer::models::frontend
