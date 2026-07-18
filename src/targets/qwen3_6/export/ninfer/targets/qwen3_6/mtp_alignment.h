#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::targets::qwen3_6 {

struct MtpAlignmentWindow {
    std::uint32_t hidden_begin             = 0;
    std::uint32_t position_begin           = 0;
    std::uint32_t shifted_embedding_begin  = 0;
    std::uint32_t columns                  = 0;
    bool final_column_uses_generated_token = false;
};

[[nodiscard]] inline MtpAlignmentWindow plan_mtp_alignment_window(std::uint32_t prompt_tokens,
                                                                  std::uint32_t chunk_begin,
                                                                  std::uint32_t columns) {
    if (columns == 0 || chunk_begin >= prompt_tokens || columns > prompt_tokens - chunk_begin) {
        throw std::invalid_argument("MTP alignment window is outside the prompt");
    }
    if (chunk_begin == std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("MTP shifted embedding position exceeds uint32");
    }
    return MtpAlignmentWindow{
        .hidden_begin                      = chunk_begin,
        .position_begin                    = chunk_begin,
        .shifted_embedding_begin           = chunk_begin + 1,
        .columns                           = columns,
        .final_column_uses_generated_token = columns == prompt_tokens - chunk_begin,
    };
}

struct MtpVisualOverlap {
    std::size_t source_begin = 0;
    std::vector<std::int32_t> destination_columns;

    [[nodiscard]] std::size_t size() const noexcept { return destination_columns.size(); }

    [[nodiscard]] bool empty() const noexcept { return destination_columns.empty(); }
};

[[nodiscard]] inline MtpVisualOverlap
shifted_visual_overlap(std::span<const std::int32_t> scatter_indices, std::uint32_t prompt_tokens,
                       const MtpAlignmentWindow& window) {
    if (!std::is_sorted(scatter_indices.begin(), scatter_indices.end()) ||
        std::adjacent_find(scatter_indices.begin(), scatter_indices.end()) !=
            scatter_indices.end()) {
        throw std::invalid_argument("MTP visual scatter indices must be sorted and unique");
    }
    if (!scatter_indices.empty() && scatter_indices.front() < 0) {
        throw std::invalid_argument("MTP visual scatter index is negative");
    }
    if (window.columns == 0 || window.shifted_embedding_begin > prompt_tokens ||
        window.columns > prompt_tokens - window.hidden_begin ||
        window.shifted_embedding_begin != window.hidden_begin + 1) {
        throw std::invalid_argument("MTP visual overlap received an invalid alignment window");
    }

    MtpVisualOverlap overlap;
    if (scatter_indices.empty() || window.shifted_embedding_begin >= prompt_tokens) {
        return overlap;
    }
    const std::uint64_t raw_end = static_cast<std::uint64_t>(window.shifted_embedding_begin) +
                                  static_cast<std::uint64_t>(window.columns);
    const std::uint32_t shifted_end =
        static_cast<std::uint32_t>(std::min<std::uint64_t>(prompt_tokens, raw_end));
    const auto begin = std::lower_bound(scatter_indices.begin(), scatter_indices.end(),
                                        static_cast<std::int32_t>(window.shifted_embedding_begin));
    const auto end =
        std::lower_bound(begin, scatter_indices.end(), static_cast<std::int32_t>(shifted_end));
    overlap.source_begin = static_cast<std::size_t>(begin - scatter_indices.begin());
    overlap.destination_columns.reserve(static_cast<std::size_t>(end - begin));
    for (auto it = begin; it != end; ++it) {
        overlap.destination_columns.push_back(
            *it - static_cast<std::int32_t>(window.shifted_embedding_begin));
    }
    return overlap;
}

} // namespace ninfer::targets::qwen3_6
