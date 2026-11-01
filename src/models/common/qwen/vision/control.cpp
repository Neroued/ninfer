#include "models/common/qwen/vision/control.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen {
namespace {

std::int32_t checked_i32(std::size_t value, const char* label) {
    if (value > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(std::string("vision control ") + label + " exceeds int32");
    }
    return static_cast<std::int32_t>(value);
}

float coordinate(std::int32_t index, std::int32_t size, std::int32_t position_side) {
    return size <= 1 ? 0.0F
                     : static_cast<float>(index) * static_cast<float>(position_side - 1) /
                           static_cast<float>(size - 1);
}

std::size_t checked_mul(std::size_t left, std::size_t right, const char* label) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::overflow_error(std::string("vision control ") + label + " exceeds size_t");
    }
    return left * right;
}

} // namespace

bool VisionExecutionInput::can_checkpoint(std::uint32_t frontier) const noexcept {
    return std::none_of(items.begin(), items.end(), [frontier](const auto& item) {
        return item.begin < frontier && frontier < item.end;
    });
}

std::shared_ptr<const VisionExecutionInput> prepare_vision_input(const PreparedPromptData& prompt,
                                                                 const VisionConfig& config,
                                                                 std::uint32_t capacity) {
    const auto merge = checked_i32(config.spatial_merge_size, "merge size");
    if (merge <= 0 || !config.position_grid_side ||
        prompt.token_ids.size() != prompt.token_types.size() ||
        prompt.media_payloads.size() != prompt.vision_items.size())
        throw std::invalid_argument("Vision prepared input has incomplete geometry or media");
    auto out                 = std::make_shared<VisionExecutionInput>();
    out->config              = config;
    out->vision_items        = prompt.vision_items;
    out->media_payloads      = prompt.media_payloads;
    std::size_t patch_cursor = 0, token_count = 0, previous_end = 0;
    for (std::size_t index = 0; index < prompt.vision_items.size(); ++index) {
        const auto& item = prompt.vision_items[index];
        const auto& grid = item.grid;
        if (grid.temporal <= 0 || grid.height <= 0 || grid.width <= 0 || grid.height % merge ||
            grid.width % merge ||
            (item.modality == models::frontend::Modality::Image && grid.temporal != 1))
            throw std::invalid_argument("Vision grid must be positive and merge-aligned");
        const auto patches =
            checked_mul(checked_mul(grid.temporal, grid.height, "grid"), grid.width, "grid");
        if (item.patch_begin != patch_cursor || item.patch_count != patches ||
            item.token_spans.size() != static_cast<std::size_t>(grid.temporal))
            throw std::invalid_argument("Vision patch ranges or temporal spans are incomplete");
        const auto& payload = prompt.media_payloads[index];
        if (!payload || payload->features_per_patch != config.patch_width() ||
            payload->span().size() != checked_mul(patches, config.patch_width(), "patch payload"))
            throw std::invalid_argument("Vision item has an invalid patch payload");
        for (const auto& span : item.token_spans) {
            if (!span.count || span.begin < previous_end ||
                span.begin > prompt.token_types.size() ||
                span.count > prompt.token_types.size() - span.begin)
                throw std::invalid_argument("Vision token spans are not ordered prompt ranges");
            const auto expected = static_cast<std::uint8_t>(item.modality);
            if (!std::all_of(prompt.token_types.begin() + span.begin,
                             prompt.token_types.begin() + span.begin + span.count,
                             [expected](auto value) { return value == expected; }))
                throw std::invalid_argument("Vision token span modality mismatch");
            token_count += span.count;
            previous_end = span.begin + span.count;
        }
        out->items.push_back(
            {static_cast<std::uint32_t>(checked_i32(item.token_spans.front().begin, "item begin")),
             static_cast<std::uint32_t>(checked_i32(previous_end, "item end"))});
        patch_cursor += patches;
    }
    if (patch_cursor != prompt.prepare.raw_patches || token_count != prompt.prepare.vision_tokens)
        throw std::invalid_argument("Vision metadata does not cover the prepared prompt");
    out->units = vision_units(prompt.vision_items, config);
    vision::validate_units(out->units, capacity, prompt.token_ids.size());
    return out;
}

std::vector<vision::Unit> vision_units(std::span<const models::frontend::VisionItem> items,
                                       const VisionConfig& config) {
    std::vector<vision::Unit> out;
    const auto merge = std::uint64_t(config.spatial_merge_size) * config.spatial_merge_size;
    for (std::uint32_t i = 0; i < items.size(); ++i) {
        const auto& item = items[i];
        const auto patches =
            checked_i32(std::uint64_t(item.grid.height) * item.grid.width, "segment patches");
        if (item.token_spans.size() != static_cast<std::size_t>(item.grid.temporal) ||
            patches % merge != 0)
            throw std::invalid_argument("Vision temporal grids differ from their token spans");
        for (std::uint32_t t = 0; t < item.token_spans.size(); ++t) {
            const auto& span = item.token_spans[t];
            if (span.count != patches / merge)
                throw std::invalid_argument("Vision grid does not cover its feature span");
            const auto linear =
                std::uint64_t(4) * config.hidden_size + std::uint64_t(2) * config.intermediate_size;
            out.push_back({i, t, std::uint64_t(t) * patches, static_cast<std::uint32_t>(patches),
                           static_cast<std::uint32_t>(checked_i32(span.begin, "token position")),
                           static_cast<std::uint32_t>(span.count),
                           std::uint64_t(patches) * (linear + patches)});
        }
    }
    return out;
}

VisionEncoderBatch encoder_batch(const VisionExecutionInput& input, const vision::WindowPlan& plan,
                                 std::uint32_t rank) {
    VisionEncoderBatch out;
    out.cu_seqlens.push_back(0);
    std::vector<std::int32_t> x_positions;
    for (const auto& placement : plan.placements) {
        if (placement.rank != rank) continue;
        const auto& unit    = plan.units.at(placement.unit);
        const auto& source  = input.vision_items.at(unit.item);
        const auto& payload = input.media_payloads.at(unit.item);
        const auto first    = static_cast<std::size_t>(unit.patch_begin);
        const auto count    = static_cast<std::size_t>(unit.patches);
        if (!payload || first > source.patch_count || count > source.patch_count - first)
            throw std::invalid_argument("Vision encoder unit exceeds its prepared input");
        out.patches.push_back(payload->span().subspan(first * payload->features_per_patch,
                                                      count * payload->features_per_patch));
        const auto merge = static_cast<std::int32_t>(input.config.spatial_merge_size);
        const auto side  = static_cast<std::int32_t>(input.config.position_grid_side);
        const auto h = source.grid.height, w = source.grid.width;
        for (std::int32_t by = 0; by < h / merge; ++by)
            for (std::int32_t bx = 0; bx < w / merge; ++bx)
                for (std::int32_t iy = 0; iy < merge; ++iy)
                    for (std::int32_t ix = 0; ix < merge; ++ix) {
                        const auto y = by * merge + iy, x = bx * merge + ix;
                        out.positions.push_back(y);
                        x_positions.push_back(x);
                        const auto yf = coordinate(y, h, side), xf = coordinate(x, w, side);
                        const auto y0 = static_cast<std::int32_t>(yf),
                                   x0 = static_cast<std::int32_t>(xf);
                        const auto y1 = std::min(y0 + 1, side - 1), x1 = std::min(x0 + 1, side - 1);
                        const auto wy = yf - y0, wx = xf - x0;
                        out.position_indices.insert(
                            out.position_indices.end(),
                            {y0 * side + x0, y0 * side + x1, y1 * side + x0, y1 * side + x1});
                        out.position_weights.insert(
                            out.position_weights.end(),
                            {(1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx});
                    }
        out.cu_seqlens.push_back(out.cu_seqlens.back() + unit.patches);
        out.features += unit.features;
    }
    out.positions.insert(out.positions.end(), x_positions.begin(), x_positions.end());
    return out;
}

VisionWork prepare_vision_work(vision::WindowUse use,
                               std::shared_ptr<const VisionExecutionInput> source,
                               std::uint32_t ranks) {
    VisionWork out{std::move(use), {}};
    if (!out.window.produce) return out;
    if (!source) throw std::logic_error("Vision production has no prepared input");
    auto input = std::make_shared<VisionWindowInput>();
    for (std::uint32_t rank = 0; rank < ranks; ++rank)
        input->batches.push_back(encoder_batch(*source, *out.window.plan, rank));
    input->owner = std::move(source);
    out.input    = std::move(input);
    return out;
}

} // namespace ninfer::models::qwen
