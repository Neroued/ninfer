#include "models/inkling2/vision/control.h"

#include "models/inkling2/frontend/prepared_prompt.h"

#include <algorithm>

namespace ninfer::models::inkling2::vision {
namespace {

struct Axis {
    std::int32_t first, second;
    float fraction;
};

Axis source_coords(std::int32_t index, std::int32_t extent, std::int32_t grid) {
    const float coord = std::clamp(float(grid) * (float(index) + 0.5F) / float(extent) - 0.5F, 0.0F,
                                   float(grid - 1));
    const auto first  = static_cast<std::int32_t>(coord);
    return {first, std::min(first + 1, grid - 1), coord - float(first)};
}

} // namespace

std::uint64_t patches_per_feature(const VisionConfig& c) noexcept {
    return std::uint64_t(c.merge_size) * c.merge_size;
}

std::vector<models::vision::Unit> units(std::span<const models::frontend::VisionItem> items,
                                        const VisionConfig& c) {
    std::vector<models::vision::Unit> out;
    out.reserve(items.size());
    for (const auto& item : items) {
        const auto& g = item.grid;
        if (item.modality != models::frontend::Modality::Image || g.temporal != 1 ||
            g.height <= 0 || g.width <= 0 || !c.merge_size || g.height % c.merge_size ||
            g.width % c.merge_size || item.token_spans.size() != 1)
            throw std::invalid_argument(
                "Inkling2 Vision requires one merge-aligned grid per image");
        const auto patches = dimension(std::uint64_t(g.height) * g.width);
        const auto tokens  = dimension(std::uint64_t(patches) + 1);
        const auto& span   = item.token_spans.front();
        if (item.patch_count != std::size_t(patches) ||
            span.count != patches / patches_per_feature(c) || span.begin > INT32_MAX)
            throw std::invalid_argument(
                "Inkling2 Vision image grid differs from its patch or token span");
        out.push_back({static_cast<std::uint32_t>(out.size()), 0, 0,
                       static_cast<std::uint32_t>(patches), static_cast<std::uint32_t>(span.begin),
                       static_cast<std::uint32_t>(span.count), std::uint64_t(tokens) * tokens});
    }
    return out;
}

EncoderBatch prepare_batch(const ExecutionInput& input, const models::vision::WindowPlan& plan) {
    EncoderBatch out;
    out.features = plan.features;
    const auto patches =
        dimension(std::uint64_t(plan.features) * patches_per_feature(input.config));
    const auto tokens = dimension(std::uint64_t(patches) + plan.placements.size());
    out.patches.reserve(plan.placements.size());
    out.cu_seqlens.reserve(plan.placements.size() + 1);
    out.cu_seqlens.push_back(0);
    out.positions.resize(2ULL * tokens);
    out.patch_columns.reserve(patches);
    out.pool_rows.reserve(patches);
    out.position_indices.reserve(patches);
    out.position_weights.reserve(patches);
    const auto merge = dimension(input.config.merge_size);
    const auto grid  = dimension(input.config.posemb_grid_size);
    for (const auto& placement : plan.placements) {
        const auto& unit = plan.units.at(placement.unit);
        const auto& g    = input.items.at(unit.item).grid;
        const auto cls   = out.cu_seqlens.back();
        const auto first = cls + 1;
        out.patches.push_back(input.payloads.at(unit.item)->span());
        for (int y = 0; y < g.height; ++y) {
            const auto [y0, y1, wy] = source_coords(y, g.height, grid);
            for (int x = 0; x < g.width; ++x) {
                const auto [x0, x1, wx]                  = source_coords(x, g.width, grid);
                const auto col                           = first + y * g.width + x;
                out.positions[col]                       = y + 1;
                out.positions[std::size_t(tokens) + col] = x + 1;
                out.patch_columns.push_back(col);
                out.position_indices.push_back(
                    {y0 * grid + x0, y0 * grid + x1, y1 * grid + x0, y1 * grid + x1});
                out.position_weights.push_back(
                    {(1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx});
            }
        }
        for (int by = 0; by < g.height; by += merge)
            for (int bx = 0; bx < g.width; bx += merge)
                for (int dy = 0; dy < merge; ++dy)
                    for (int dx = 0; dx < merge; ++dx)
                        out.pool_rows.push_back(first + (by + dy) * g.width + bx + dx);
        out.cu_seqlens.push_back(first + dimension(unit.patches));
    }
    return out;
}

std::shared_ptr<const ExecutionInput> prepare_input(const PreparedPromptData& prompt,
                                                    const VisionConfig& config,
                                                    std::uint32_t capacity) {
    if (prompt.media_payloads.size() != prompt.identity.media.size())
        throw std::invalid_argument("Inkling2 Vision input has incomplete media payloads");
    auto out      = std::make_shared<ExecutionInput>();
    out->config   = config;
    out->payloads = prompt.media_payloads;
    out->items.reserve(prompt.identity.media.size());
    for (const auto& media : prompt.identity.media) out->items.push_back(media.item);
    out->units = units(out->items, config);
    models::vision::validate_units(out->units, capacity, dimension(prompt.token_ids.size()));

    std::size_t patches = 0, features = 0;
    const auto width = config.patch_width();
    for (const auto& unit : out->units) {
        const auto& item    = out->items[unit.item];
        const auto& payload = out->payloads[unit.item];
        if (!payload || item.patch_begin != patches || payload->features_per_patch != width ||
            payload->span().size() != std::uint64_t(unit.patches) * width)
            throw std::invalid_argument(
                "Inkling2 Vision patch payload differs from its media grid");
        patches += unit.patches;
        features += unit.features;
    }
    if (patches != prompt.prepare.raw_patches || features != prompt.prepare.vision_tokens)
        throw std::invalid_argument("Inkling2 Vision metadata does not cover its prepared prompt");
    return out;
}

Work prepare_work(models::vision::WindowUse use, std::shared_ptr<const ExecutionInput> source) {
    Work out{std::move(use), {}};
    if (!out.window.produce) return out;
    if (!source) throw std::logic_error("Inkling2 Vision production has no prepared input");
    auto batch = prepare_batch(*source, *out.window.plan);
    out.input  = std::make_shared<WindowInput>(WindowInput{std::move(batch), std::move(source)});
    return out;
}

} // namespace ninfer::models::inkling2::vision
