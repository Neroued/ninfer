#include "targets/qwen3_6_27b_rtx5090/impl/schedule/schedule.h"

#include "targets/qwen3_6_27b_rtx5090/impl/schedule/vision_context.h"

#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_6_27b_rtx5090::detail::schedule {
namespace {

PreprocessStats legacy_stats(const PreparedPromptData& prompt) {
    PreprocessStats stats;
    stats.media_items     = prompt.prepare.media_items;
    stats.raw_patches     = prompt.prepare.raw_patches;
    stats.vision_tokens   = prompt.prepare.vision_tokens;
    stats.attention_pairs = prompt.prepare.attention_pairs;
    stats.prompt_tokens   = prompt.token_ids.size();
    stats.patch_bytes     = prompt.prepare.patch_bytes;
    return stats;
}

Modality legacy_modality(PromptModality modality) {
    switch (modality) {
    case PromptModality::Image:
        return Modality::Image;
    case PromptModality::Video:
        return Modality::Video;
    }
    throw std::logic_error("unknown prompt modality");
}

VisionItem convert_vision_item(ninfer::targets::qwen3_6_27b_rtx5090::detail::VisionItem item) {
    VisionItem converted;
    converted.modality    = legacy_modality(item.modality);
    converted.grid        = VisionGrid{item.grid.temporal, item.grid.height, item.grid.width};
    converted.patch_begin = item.patch_begin;
    converted.patch_count = item.patch_count;
    converted.timestamps  = std::move(item.timestamps);
    converted.token_spans.reserve(item.token_spans.size());
    for (const auto span : item.token_spans) {
        converted.token_spans.push_back(TokenSpan{span.begin, span.count});
    }
    return converted;
}

} // namespace

std::size_t vision_workspace_bytes(const PreparedPromptData& prompt) {
    if (!prompt.has_media()) { return 0; }
    ProcessedInput dimensions;
    dimensions.input_ids.assign(prompt.token_ids.begin(), prompt.token_ids.end());
    dimensions.token_types = prompt.token_types;
    dimensions.stats       = legacy_stats(prompt);
    dimensions.vision_items.reserve(prompt.vision_items.size());
    for (const auto& item : prompt.vision_items) {
        dimensions.vision_items.push_back(convert_vision_item(item));
    }
    return VisionContext::workspace_bytes(dimensions);
}

ProcessedInput take_processed_prompt(PreparedPromptData&& prompt) {
    ProcessedInput out;
    out.input_ids           = std::move(prompt.token_ids);
    out.token_types         = std::move(prompt.token_types);
    out.positions           = std::move(prompt.positions);
    out.rope_delta          = prompt.rope_delta;
    out.patches             = std::move(prompt.patches);
    out.stats               = legacy_stats(prompt);
    out.stats.prompt_tokens = out.input_ids.size();
    out.vision_items.reserve(prompt.vision_items.size());
    for (auto& item : prompt.vision_items) {
        out.vision_items.push_back(convert_vision_item(std::move(item)));
    }
    return out;
}

Tensor encode_vision(State& state, const ProcessedInput& prompt,
                     runtime::TransientRegion transient) {
    const std::size_t required = VisionContext::workspace_bytes(prompt);
    if (transient.data == nullptr || transient.size < required || transient.alignment < 256) {
        throw std::invalid_argument("Vision transient region is too small or misaligned");
    }
    WorkspaceArena workspace(DeviceSpan{transient.data, transient.size});
    VisionContext vision(state.device, state.model);
    void* tap_context = state.diagnostic_vision_tap != nullptr ? state.diagnostic_context : nullptr;
    return vision.encode(prompt, workspace, tap_context, state.diagnostic_vision_tap);
}

} // namespace ninfer::targets::qwen3_6_27b_rtx5090::detail::schedule
