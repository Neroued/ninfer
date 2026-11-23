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

[[nodiscard]] inline int shared_prefix_length(int target, LengthDist dist, int page_tokens) {
    return dist == LengthDist::SharedPrefix && target / 2 >= page_tokens
               ? (target * 3 / 4) / page_tokens * page_tokens
               : 0;
}

[[nodiscard]] inline int distributed_length(int target, int prefix, int row, int batch,
                                            LengthDist dist) {
    if (row == 0 || dist == LengthDist::Uniform) return target;
    if (dist == LengthDist::Skewed1Long) {
        constexpr std::array short_lengths{512, 1024, 384, 1536, 768, 256, 1280};
        return std::min(target, short_lengths[(row - 1) % short_lengths.size()]);
    }
    return prefix + static_cast<int>(std::int64_t(target - prefix) * (batch - row) / batch);
}

[[nodiscard]] inline int distributed_valid(int tokens, int row, int batch, ValidDist dist) {
    if (tokens == 0 || (dist == ValidDist::WithInert && batch >= 2 && row == batch - 1)) return 0;
    return dist == ValidDist::Full ? tokens : tokens - row % tokens;
}
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

    BatchWorkload workload;
    // Row zero always reaches the target if the batch has any live rows.
    workload.max_visible_tokens =
        batch > 0 && tokens > 0 ? static_cast<std::uint32_t>(target_context) : 0U;
    workload.natural_bucket = ops::select_context_bucket(workload.max_visible_tokens);
    workload.valid_columns.resize(static_cast<std::size_t>(batch));
    workload.row_contexts.resize(static_cast<std::size_t>(batch));
    workload.shared_prefix_tokens =
        detail::shared_prefix_length(target_context, length_dist, page_tokens);
    for (std::int32_t row = 0; row < batch; ++row) {
        const int valid = detail::distributed_valid(tokens, row, batch, valid_dist);
        if (valid == 0) continue;
        workload.valid_columns[row] = valid;
        workload.row_contexts[row]  = std::max(
            valid, detail::distributed_length(target_context, workload.shared_prefix_tokens, row,
                                               batch, length_dist));
    }
    return workload;
}

struct ContextQueryOptions {
    std::optional<ops::ContextBucket> bucket;
    std::optional<LengthDist> length_dist;
    std::optional<ValidDist> valid_dist;
    std::string_view lengths = "uniform";
    std::string_view valid   = "full";
    int envelope_max         = -1;

    [[nodiscard]] std::string_view length_name() const {
        return length_dist ? length_dist_name(*length_dist) : lengths;
    }

    [[nodiscard]] std::string_view valid_name() const {
        return valid_dist ? valid_dist_name(*valid_dist) : valid;
    }

    [[nodiscard]] ops::ContextBucket select_bucket(std::uint32_t required_tokens) const {
        const auto selected =
            bucket ? *bucket
                   : ops::select_context_bucket(envelope_max >= 0 ? envelope_max : required_tokens);
        if (ops::context_bucket_tokens(selected) < required_tokens)
            throw std::invalid_argument("selected bucket is too small for the live rows");
        return selected;
    }
};

struct ContextQueryWorkload {
    std::vector<std::int32_t> lengths;
    std::vector<std::int32_t> valid_columns;
    int shared_prefix_tokens         = 0;
    std::uint32_t max_history        = 0;
    std::uint32_t max_visible_tokens = 0;
};

// The context-query benchmarks take committed history as --context. Distribute it independently
// of query validity, and expose both bounds because the cyclic bucket includes the live queries.
[[nodiscard]] inline ContextQueryWorkload
generate_context_query_workload(int context, int tokens, int batch,
                                const ContextQueryOptions& options, int page_tokens = 64) {
    ContextQueryWorkload workload;
    workload.lengths.resize(batch);
    workload.valid_columns.resize(batch);
    const auto dist  = options.length_dist.value_or(LengthDist::Uniform);
    const int prefix = detail::shared_prefix_length(context, dist, page_tokens);
    for (int row = 0; row < batch; ++row) {
        int valid = tokens;
        if (options.valid_dist)
            valid = detail::distributed_valid(tokens, row, batch, *options.valid_dist);
        else if (options.valid == "mixed")
            valid = static_cast<int>((std::int64_t(row) * 7 + tokens) % (tokens + 1));
        else if (options.valid == "half")
            valid = (tokens + 1) / 2;
        else if (options.valid == "zero")
            valid = 0;
        if (valid == 0) continue;

        int length = detail::distributed_length(context, prefix, row, batch, dist);
        if (!options.length_dist && options.lengths == "mixed") {
            if (row % 3 == 1) length = std::max(0, context - 63);
            if (row % 3 == 2) length = std::min(context, 128);
        }
        workload.lengths[row]         = length;
        workload.valid_columns[row]   = valid;
        workload.shared_prefix_tokens = prefix;
        workload.max_history = std::max(workload.max_history, static_cast<std::uint32_t>(length));
        workload.max_visible_tokens =
            std::max(workload.max_visible_tokens, static_cast<std::uint32_t>(length + valid));
    }
    return workload;
}

} // namespace ninfer::bench::workload
