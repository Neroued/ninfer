#pragma once

#include "ninfer/ops/context_bucket.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::bench::workload {

enum class LengthDist : std::uint8_t { Uniform, Skewed1Long, Staggered, SharedPrefix };
enum class ValidDist : std::uint8_t { Full, Ragged, WithInert };

namespace detail {
inline constexpr std::array<std::string_view, 4> kLengthDistNames{
    "uniform",
    "skewed-1long",
    "staggered",
    "shared-prefix",
};
inline constexpr std::array<std::string_view, 3> kValidDistNames{"full", "ragged", "with-inert"};
} // namespace detail

[[nodiscard]] constexpr std::string_view length_dist_name(LengthDist dist) {
    const auto index = static_cast<std::size_t>(dist);
    if (index >= detail::kLengthDistNames.size())
        throw std::invalid_argument("invalid length distribution");
    return detail::kLengthDistNames[index];
}

[[nodiscard]] constexpr std::string_view valid_dist_name(ValidDist dist) {
    const auto index = static_cast<std::size_t>(dist);
    if (index >= detail::kValidDistNames.size())
        throw std::invalid_argument("invalid validity distribution");
    return detail::kValidDistNames[index];
}

[[nodiscard]] inline LengthDist parse_length_dist(std::string_view text) {
    for (std::size_t i = 0; i < detail::kLengthDistNames.size(); ++i) {
        if (detail::kLengthDistNames[i] == text) return static_cast<LengthDist>(i);
    }
    throw std::invalid_argument("invalid length distribution: " + std::string(text));
}

[[nodiscard]] inline ValidDist parse_valid_dist(std::string_view text) {
    for (std::size_t i = 0; i < detail::kValidDistNames.size(); ++i) {
        if (detail::kValidDistNames[i] == text) return static_cast<ValidDist>(i);
    }
    throw std::invalid_argument("invalid validity distribution: " + std::string(text));
}

[[nodiscard]] inline std::optional<ops::ContextBucket>
parse_bucket_override(std::string_view text) {
    if (text == "auto") return std::nullopt;
    return ops::parse_context_bucket(text);
}

struct BatchWorkload {
    std::vector<std::int32_t> valid_columns;
    std::vector<std::int32_t> row_contexts;
    std::int32_t shared_prefix_tokens = 0;
    std::uint32_t max_visible_tokens  = 0;
    ops::ContextBucket natural_bucket = ops::ContextBucket::K2;
};

// A live row's context is its exclusive end position, including valid query columns.
// Empty batches and zero-column batches have zero visible tokens.
// The target must fit in a bucket and hold all query columns.
[[nodiscard]] inline BatchWorkload generate_batch_workload(std::int32_t target_context,
                                                           std::int32_t tokens, std::int32_t batch,
                                                           LengthDist length_dist,
                                                           ValidDist valid_dist,
                                                           std::int32_t page_tokens = 64) {
    if (tokens < 0 || target_context < tokens || batch < 0 || page_tokens <= 0 ||
        static_cast<std::uint32_t>(target_context) >
            ops::context_bucket_tokens(ops::ContextBucket::M1)) {
        throw std::invalid_argument("invalid context, token, batch, or page size for workload");
    }
    (void)length_dist_name(length_dist);
    (void)valid_dist_name(valid_dist);

    const bool has_inert_row = valid_dist == ValidDist::WithInert && batch >= 2;
    const auto live_rows     = tokens == 0 ? 0 : batch - static_cast<int>(has_inert_row);

    BatchWorkload workload;
    // Row zero always reaches the target if the batch has any live rows.
    workload.max_visible_tokens = live_rows > 0 ? static_cast<std::uint32_t>(target_context) : 0U;
    workload.natural_bucket     = ops::select_context_bucket(workload.max_visible_tokens);
    workload.valid_columns.resize(static_cast<std::size_t>(batch));
    workload.row_contexts.resize(static_cast<std::size_t>(batch));
    if (length_dist == LengthDist::SharedPrefix && target_context / 2 >= page_tokens) {
        workload.shared_prefix_tokens = (target_context * 3 / 4) / page_tokens * page_tokens;
    }
    constexpr std::array<std::int32_t, 7> short_lengths{512, 1024, 384, 1536, 768, 256, 1280};
    for (std::int32_t row = 0; row < live_rows; ++row) {
        const auto index = static_cast<std::size_t>(row);
        const auto valid = valid_dist == ValidDist::Full ? tokens : tokens - row % tokens;
        workload.valid_columns[index] = valid;

        auto context = target_context;
        if (row > 0) {
            switch (length_dist) {
            case LengthDist::Uniform:
                break;
            case LengthDist::Skewed1Long:
                context = short_lengths[(index - 1) % short_lengths.size()];
                break;
            case LengthDist::Staggered:
            case LengthDist::SharedPrefix: {
                const auto prefix = workload.shared_prefix_tokens;
                const auto suffix = static_cast<std::int64_t>(target_context - prefix);
                context = prefix + static_cast<std::int32_t>(suffix * (batch - row) / batch);
                break;
            }
            }
        }
        workload.row_contexts[index] = std::clamp(context, valid, target_context);
    }
    return workload;
}

} // namespace ninfer::bench::workload
