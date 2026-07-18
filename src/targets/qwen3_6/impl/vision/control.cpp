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

} // namespace

VisionControl build_vision_control(const PreparedPromptData& prompt) {
    if (prompt.token_ids.size() != prompt.token_types.size()) {
        throw std::invalid_argument("vision control token types must cover the prompt");
    }
    const std::size_t patches = static_cast<std::size_t>(prompt.prepare.raw_patches);
    const std::size_t tokens  = static_cast<std::size_t>(prompt.prepare.vision_tokens);
    VisionControl out;
    out.position_ids.resize(patches * 2);
    out.position_table_indices.reserve(patches * 4);
    out.position_table_weights.reserve(patches * 4);
    out.scatter_indices.reserve(tokens);
    out.items.reserve(prompt.vision_items.size());
    out.cu_seqlens.push_back(0);

    std::size_t patch_cursor    = 0;
    std::size_t position_cursor = 0;
    std::size_t token_cursor    = 0;
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
            static_cast<std::size_t>(t) * static_cast<std::size_t>(h) * static_cast<std::size_t>(w);
        if (item.patch_begin != patch_cursor || item.patch_count != item_patches) {
            throw std::invalid_argument("vision control patch ranges are not canonical");
        }
        const std::size_t expected_spans =
            item.modality == PromptModality::Video ? static_cast<std::size_t>(t) : 1;
        if (item.token_spans.size() != expected_spans) {
            throw std::invalid_argument("vision control token spans do not match modality grid");
        }

        const std::size_t scatter_begin = out.scatter_indices.size();
        std::size_t item_tokens         = 0;
        std::size_t next_span_begin =
            out.scatter_indices.empty() ? 0
                                        : static_cast<std::size_t>(out.scatter_indices.back()) + 1;
        for (const TokenSpan& span : item.token_spans) {
            if (span.count == 0 || span.begin > prompt.token_types.size() ||
                span.count > prompt.token_types.size() - span.begin) {
                throw std::invalid_argument("vision control token span exceeds prompt");
            }
            if (span.begin < next_span_begin) {
                throw std::invalid_argument("vision control token spans are not ordered");
            }
            const auto expected = static_cast<std::uint8_t>(item.modality);
            if (!std::all_of(prompt.token_types.begin() + static_cast<std::ptrdiff_t>(span.begin),
                             prompt.token_types.begin() +
                                 static_cast<std::ptrdiff_t>(span.begin + span.count),
                             [expected](std::uint8_t value) { return value == expected; })) {
                throw std::invalid_argument("vision control token span modality mismatch");
            }
            item_tokens += span.count;
            next_span_begin = span.begin + span.count;
        }
        if (item_tokens != item_patches / static_cast<std::size_t>(kMerge * kMerge)) {
            throw std::invalid_argument("vision control token spans do not cover merged patches");
        }

        for (std::int32_t temporal = 0; temporal < t; ++temporal) {
            const std::int64_t next =
                static_cast<std::int64_t>(out.cu_seqlens.back()) + static_cast<std::int64_t>(h) * w;
            if (next > std::numeric_limits<std::int32_t>::max()) {
                throw std::overflow_error("vision control cu_seqlens exceeds int32");
            }
            out.cu_seqlens.push_back(static_cast<std::int32_t>(next));
            for (std::int32_t block_y = 0; block_y < h / kMerge; ++block_y) {
                for (std::int32_t block_x = 0; block_x < w / kMerge; ++block_x) {
                    for (std::int32_t inner_y = 0; inner_y < kMerge; ++inner_y) {
                        for (std::int32_t inner_x = 0; inner_x < kMerge; ++inner_x) {
                            const std::int32_t y              = block_y * kMerge + inner_y;
                            const std::int32_t x              = block_x * kMerge + inner_x;
                            out.position_ids[position_cursor] = y;
                            out.position_ids[patches + position_cursor] = x;
                            ++position_cursor;

                            const float yf        = coordinate(y, h);
                            const float xf        = coordinate(x, w);
                            const auto y0         = static_cast<std::int32_t>(yf);
                            const auto x0         = static_cast<std::int32_t>(xf);
                            const std::int32_t y1 = std::min(y0 + 1, kPositionSide - 1);
                            const std::int32_t x1 = std::min(x0 + 1, kPositionSide - 1);
                            const float wy        = yf - static_cast<float>(y0);
                            const float wx        = xf - static_cast<float>(x0);
                            out.position_table_indices.insert(
                                out.position_table_indices.end(),
                                {y0 * kPositionSide + x0, y0 * kPositionSide + x1,
                                 y1 * kPositionSide + x0, y1 * kPositionSide + x1});
                            out.position_table_weights.insert(out.position_table_weights.end(),
                                                              {(1.0F - wy) * (1.0F - wx),
                                                               (1.0F - wy) * wx, wy * (1.0F - wx),
                                                               wy * wx});
                        }
                    }
                }
            }
        }

        for (const TokenSpan& span : item.token_spans) {
            for (std::size_t i = 0; i < span.count; ++i) {
                out.scatter_indices.push_back(checked_i32(span.begin + i, "scatter index"));
            }
            token_cursor += span.count;
        }
        out.items.push_back(VisionItemControl{
            .patch_begin    = item.patch_begin,
            .patch_count    = item.patch_count,
            .merged_begin   = token_cursor - item_tokens,
            .merged_count   = item_tokens,
            .segment_length = h * w,
            .segment_count  = t,
            .scatter_begin  = scatter_begin,
            .scatter_count  = item_tokens,
        });
        patch_cursor += item_patches;
    }

    if (patch_cursor != patches || position_cursor != patches || token_cursor != tokens ||
        out.position_ids.size() != patches * 2 ||
        out.position_table_indices.size() != patches * 4 ||
        out.position_table_weights.size() != patches * 4 || out.scatter_indices.size() != tokens ||
        out.items.size() != prompt.vision_items.size() ||
        out.cu_seqlens.back() != checked_i32(patches, "patch count")) {
        throw std::invalid_argument("vision control metadata does not cover prepared prompt");
    }
    return out;
}

} // namespace ninfer::targets::qwen3_6
