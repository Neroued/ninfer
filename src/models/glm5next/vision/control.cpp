#include "models/glm5next/vision/control.h"

#include "models/glm5next/frontend/prepared_prompt.h"

#include <algorithm>

namespace ninfer::models::glm5next::vision {

std::vector<models::vision::Unit> units(std::span<const models::frontend::VisionItem> items,
                                        const VisionConfig& c) {
    std::vector<models::vision::Unit> out;
    for (std::uint32_t i = 0; i < items.size(); ++i) {
        const auto& item = items[i];
        const auto n     = dimension(std::uint64_t(item.grid.height) * item.grid.width);
        if (item.token_spans.size() != static_cast<std::size_t>(item.grid.temporal))
            throw std::invalid_argument("GLM Vision temporal grids differ from token spans");
        for (std::uint32_t t = 0; t < item.token_spans.size(); ++t) {
            const auto& span = item.token_spans[t];
            if (span.count * c.merge_size * c.merge_size != std::size_t(n))
                throw std::invalid_argument("GLM Vision grid differs from its feature span");
            if (span.begin > INT32_MAX)
                throw std::invalid_argument("GLM Vision position exceeds int32");
            out.push_back({i, t, std::uint64_t(t) * n, static_cast<std::uint32_t>(n),
                           static_cast<std::uint32_t>(span.begin),
                           static_cast<std::uint32_t>(span.count),
                           std::uint64_t(n) * (4ULL * c.hidden + 3ULL * c.intermediate + n)});
        }
    }
    return out;
}

EncoderBatch prepare_batch(
    std::span<const models::frontend::VisionItem> items,
    std::span<const std::shared_ptr<const models::frontend::PreparedMediaPayload>> payloads,
    const models::vision::WindowPlan& plan, std::uint32_t rank, const VisionConfig& c) {
    EncoderBatch out;
    out.cu_seqlens.push_back(0);
    std::vector<std::int32_t> xs;
    for (const auto& placement : plan.placements) {
        if (placement.rank != rank) continue;
        const auto& unit = plan.units.at(placement.unit);
        if (unit.item >= items.size() || unit.item >= payloads.size() || !payloads[unit.item])
            throw std::invalid_argument("GLM Vision unit has no prepared payload");
        const auto& grid    = items[unit.item].grid;
        const auto& payload = *payloads[unit.item];
        if (payload.features_per_patch != c.patch_width() ||
            unit.index >= std::uint32_t(grid.temporal) ||
            unit.patches != std::uint64_t(grid.height) * grid.width ||
            unit.patch_begin != std::uint64_t(unit.index) * unit.patches)
            throw std::invalid_argument("GLM Vision unit differs from its source geometry");
        const auto begin = unit.patch_begin * c.patch_width();
        const auto count = std::uint64_t(unit.patches) * c.patch_width();
        if (begin > payload.span().size() || count > payload.span().size() - begin)
            throw std::invalid_argument("GLM Vision patch span exceeds its prepared payload");
        out.patches.push_back(payload.span().subspan(begin, count));
        for (int by = 0; by < grid.height; by += c.merge_size)
            for (int bx = 0; bx < grid.width; bx += c.merge_size)
                for (std::uint32_t y = 0; y < c.merge_size; ++y)
                    for (std::uint32_t x = 0; x < c.merge_size; ++x) {
                        out.positions.push_back(by + y);
                        xs.push_back(bx + x);
                    }
        out.cu_seqlens.push_back(out.cu_seqlens.back() + unit.patches);
        out.features += unit.features;
    }
    out.positions.insert(out.positions.end(), xs.begin(), xs.end());
    return out;
}

std::shared_ptr<const ExecutionInput> prepare_input(const PreparedPromptData& prompt,
                                                    const VisionConfig& config,
                                                    std::uint32_t capacity) {
    if (!config.merge_size || prompt.media_payloads.size() != prompt.identity.media.size())
        throw std::invalid_argument("GLM Vision input has incomplete geometry or media");
    auto out            = std::make_shared<ExecutionInput>();
    out->config         = config;
    out->payloads       = prompt.media_payloads;
    std::size_t patches = 0, features = 0, previous_end = 0;
    for (std::size_t i = 0; i < prompt.identity.media.size(); ++i) {
        const auto& item = prompt.identity.media[i].item;
        const auto& grid = item.grid;
        if (grid.temporal <= 0 || grid.height <= 0 || grid.width <= 0 ||
            grid.height % config.merge_size || grid.width % config.merge_size ||
            (item.modality == models::frontend::Modality::Image && grid.temporal != 1))
            throw std::invalid_argument("GLM Vision grid must be positive and merge-aligned");
        const auto count    = std::uint64_t(grid.temporal) * grid.height * grid.width;
        const auto& payload = out->payloads[i];
        if (item.patch_begin != patches || item.patch_count != count ||
            item.token_spans.size() != std::size_t(grid.temporal) || !payload ||
            payload->features_per_patch != config.patch_width() ||
            payload->span().size() / config.patch_width() != count ||
            payload->span().size() % config.patch_width())
            throw std::invalid_argument("GLM Vision patch payload differs from its media grid");
        for (const auto& span : item.token_spans) {
            if (!span.count || span.begin < previous_end || span.begin > prompt.token_ids.size() ||
                span.count > prompt.token_ids.size() - span.begin)
                throw std::invalid_argument("GLM Vision spans are not ordered prompt ranges");
            features += span.count;
            previous_end = span.begin + span.count;
        }
        patches += count;
        out->items.push_back(item);
    }
    if (patches != prompt.prepare.raw_patches || features != prompt.prepare.vision_tokens)
        throw std::invalid_argument("GLM Vision metadata does not cover its prepared prompt");
    out->units = units(out->items, config);
    models::vision::validate_units(out->units, capacity, prompt.token_ids.size());
    return out;
}

Work prepare_work(models::vision::WindowUse use, std::shared_ptr<const ExecutionInput> source,
                  std::uint32_t ranks) {
    Work out{std::move(use), {}};
    if (!out.window.produce) return out;
    if (!source) throw std::logic_error("GLM Vision production has no prepared input");
    auto input = std::make_shared<WindowInput>();
    for (std::uint32_t rank = 0; rank < ranks; ++rank)
        input->batches.push_back(
            prepare_batch(source->items, source->payloads, *out.window.plan, rank, source->config));
    input->owner = std::move(source);
    out.input    = std::move(input);
    return out;
}

} // namespace ninfer::models::glm5next::vision
