#include <ninfer/targets/qwen3_6/vision_control.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen3_6 {
namespace {

constexpr std::int32_t kMerge        = 2;
constexpr std::int32_t kPositionSide = 48;

std::int32_t checked_i32(std::size_t value, const char* label) {
    if (value > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(std::string("vision control ") + label + " exceeds int32");
    }
    return static_cast<std::int32_t>(value);
}

float coordinate(std::int32_t index, std::int32_t size) {
    return size <= 1 ? 0.0F
                     : static_cast<float>(index) * static_cast<float>(kPositionSide - 1) /
                           static_cast<float>(size - 1);
}

std::size_t checked_mul(std::size_t left, std::size_t right, const char* label) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::overflow_error(std::string("vision control ") + label + " exceeds size_t");
    }
    return left * right;
}

} // namespace

VisionControlPlan plan_vision_control(const PreparedPromptData& prompt) {
    if (prompt.token_ids.size() != prompt.token_types.size()) {
        throw std::invalid_argument("vision control token types must cover the prompt");
    }
    VisionControlPlan plan;
    plan.items.reserve(prompt.vision_items.size());
    std::size_t patch_cursor    = 0;
    std::size_t token_cursor    = 0;
    std::size_t next_span_begin = 0;
    for (const VisionItem& item : prompt.vision_items) {
        const std::int32_t t = item.grid.temporal;
        const std::int32_t h = item.grid.height;
        const std::int32_t w = item.grid.width;
        if (t <= 0 || h <= 0 || w <= 0 || h % kMerge != 0 || w % kMerge != 0) {
            throw std::invalid_argument(
                "vision control grid must be positive and merge-aligned: " + std::to_string(t) +
                "x" + std::to_string(h) + "x" + std::to_string(w));
        }
        const std::size_t item_patches =
            checked_mul(checked_mul(static_cast<std::size_t>(t), static_cast<std::size_t>(h),
                                    "temporal/spatial patch product"),
                        static_cast<std::size_t>(w), "patch product");
        if (item.patch_begin != patch_cursor || item.patch_count != item_patches) {
            throw std::invalid_argument("vision control patch ranges are not canonical");
        }
        const std::size_t expected_spans =
            item.modality == PromptModality::Video ? static_cast<std::size_t>(t) : 1;
        if (item.token_spans.size() != expected_spans) {
            throw std::invalid_argument("vision control token spans do not match modality grid");
        }
        std::size_t item_tokens = 0;
        for (const TokenSpan& span : item.token_spans) {
            if (span.count == 0 || span.begin > prompt.token_types.size() ||
                span.count > prompt.token_types.size() - span.begin) {
                throw std::invalid_argument("vision control token span exceeds prompt");
            }
            if (span.begin < next_span_begin) {
                throw std::invalid_argument("vision control token spans are not ordered");
            }
            if (span.count > std::numeric_limits<std::size_t>::max() - item_tokens) {
                throw std::overflow_error("vision control merged token count exceeds size_t");
            }
            item_tokens += span.count;
            next_span_begin = span.begin + span.count;
        }
        if (item_tokens != item_patches / static_cast<std::size_t>(kMerge * kMerge)) {
            throw std::invalid_argument("vision control token spans do not cover merged patches");
        }
        if (item.token_spans.front().begin > std::numeric_limits<std::uint32_t>::max() ||
            item.token_spans.back().begin + item.token_spans.back().count >
                std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("vision control token frontier exceeds uint32");
        }
        plan.items.push_back(VisionItemControlPlan{
            .token_begin  = static_cast<std::uint32_t>(item.token_spans.front().begin),
            .token_end    = static_cast<std::uint32_t>(item.token_spans.back().begin +
                                                       item.token_spans.back().count),
            .merged_count = item_tokens,
        });
        token_cursor += item_tokens;
        patch_cursor += item_patches;
    }
    if (static_cast<std::uint64_t>(patch_cursor) != prompt.prepare.raw_patches ||
        static_cast<std::uint64_t>(token_cursor) != prompt.prepare.vision_tokens ||
        plan.items.size() != prompt.vision_items.size()) {
        throw std::invalid_argument("vision control metadata does not cover prepared prompt");
    }
    return plan;
}

VisionControl build_vision_control(const PreparedPromptData& prompt, const VisionControlPlan& plan,
                                   std::uint32_t prepared_item_begin) {
    if (plan.items.size() != prompt.vision_items.size() ||
        prepared_item_begin > prompt.vision_items.size()) {
        throw std::invalid_argument("vision control plan does not cover the prepared media");
    }
    VisionControl out;
    out.prepared_item_begin = prepared_item_begin;
    out.items.reserve(prompt.vision_items.size() - prepared_item_begin);
    for (std::size_t item_index = prepared_item_begin; item_index < prompt.vision_items.size();
         ++item_index) {
        const VisionItem& item             = prompt.vision_items[item_index];
        const VisionItemControlPlan& input = plan.items[item_index];
        const std::int32_t t               = item.grid.temporal;
        const std::int32_t h               = item.grid.height;
        const std::int32_t w               = item.grid.width;
        const std::size_t item_patches     = item.patch_count;

        VisionItemControl control;
        control.modality       = item.modality;
        control.grid           = item.grid;
        control.patch_begin    = item.patch_begin;
        control.patch_count    = item.patch_count;
        control.merged_count   = input.merged_count;
        control.segment_length = checked_i32(
            checked_mul(static_cast<std::size_t>(h), static_cast<std::size_t>(w), "segment length"),
            "segment length");
        control.segment_count = t;
        control.position_ids.resize(checked_mul(item_patches, 2, "position id count"));
        control.position_table_indices.reserve(
            checked_mul(item_patches, 4, "position table index count"));
        control.position_table_weights.reserve(
            checked_mul(item_patches, 4, "position table weight count"));
        const auto expected = static_cast<std::uint8_t>(item.modality);
        for (const TokenSpan& span : item.token_spans) {
            if (!std::all_of(prompt.token_types.begin() + static_cast<std::ptrdiff_t>(span.begin),
                             prompt.token_types.begin() +
                                 static_cast<std::ptrdiff_t>(span.begin + span.count),
                             [expected](std::uint8_t value) { return value == expected; })) {
                throw std::invalid_argument("vision control token span modality mismatch");
            }
            for (std::size_t i = 0; i < span.count; ++i) {
                control.scatter_indices.push_back(checked_i32(span.begin + i, "scatter index"));
            }
        }

        std::size_t position_cursor = 0;
        for (std::int32_t temporal = 0; temporal < t; ++temporal) {
            for (std::int32_t block_y = 0; block_y < h / kMerge; ++block_y) {
                for (std::int32_t block_x = 0; block_x < w / kMerge; ++block_x) {
                    for (std::int32_t inner_y = 0; inner_y < kMerge; ++inner_y) {
                        for (std::int32_t inner_x = 0; inner_x < kMerge; ++inner_x) {
                            const std::int32_t y                  = block_y * kMerge + inner_y;
                            const std::int32_t x                  = block_x * kMerge + inner_x;
                            control.position_ids[position_cursor] = y;
                            control.position_ids[item_patches + position_cursor] = x;
                            ++position_cursor;

                            const float yf        = coordinate(y, h);
                            const float xf        = coordinate(x, w);
                            const auto y0         = static_cast<std::int32_t>(yf);
                            const auto x0         = static_cast<std::int32_t>(xf);
                            const std::int32_t y1 = std::min(y0 + 1, kPositionSide - 1);
                            const std::int32_t x1 = std::min(x0 + 1, kPositionSide - 1);
                            const float wy        = yf - static_cast<float>(y0);
                            const float wx        = xf - static_cast<float>(x0);
                            control.position_table_indices.insert(
                                control.position_table_indices.end(),
                                {y0 * kPositionSide + x0, y0 * kPositionSide + x1,
                                 y1 * kPositionSide + x0, y1 * kPositionSide + x1});
                            control.position_table_weights.insert(
                                control.position_table_weights.end(),
                                {(1.0F - wy) * (1.0F - wx), (1.0F - wy) * wx, wy * (1.0F - wx),
                                 wy * wx});
                        }
                    }
                }
            }
        }

        if (position_cursor != item_patches || control.position_ids.size() != item_patches * 2 ||
            control.position_table_indices.size() != item_patches * 4 ||
            control.position_table_weights.size() != item_patches * 4 ||
            control.scatter_indices.size() != input.merged_count ||
            checked_mul(static_cast<std::size_t>(control.segment_length),
                        static_cast<std::size_t>(control.segment_count),
                        "segmented patch count") != item_patches) {
            throw std::invalid_argument("vision item control metadata is incomplete");
        }
        out.items.push_back(std::move(control));
    }
    if (out.items.size() != prompt.vision_items.size() - prepared_item_begin) {
        throw std::invalid_argument("vision control metadata does not cover prepared prompt");
    }
    return out;
}

} // namespace ninfer::targets::qwen3_6
