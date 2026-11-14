#include "models/common/batch.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace ninfer::models {
namespace {
std::uint32_t ceil_div(std::uint32_t n, std::uint32_t d) { return n / d + (n % d != 0); }

void validate_spec(const BatchExecutionSpec& spec) {
    if (!spec.resident_capacity || !spec.attention_owners)
        throw std::invalid_argument(
            "batch execution requires positive request and owner capacities");
    std::uint32_t previous = 0;
    for (auto rows : spec.graph_rows) {
        if (rows <= previous || rows > spec.resident_capacity)
            throw std::invalid_argument(
                "batch graph rows must be strictly increasing within capacity");
        previous = rows;
    }
    if (!spec.graph_rows.empty() && previous != spec.resident_capacity)
        throw std::invalid_argument("batch graph rows do not cover request capacity");
}

std::uint32_t lower_bound_rows(std::span<const std::uint32_t> rows, std::uint32_t value) {
    const auto found = std::lower_bound(rows.begin(), rows.end(), value);
    if (found == rows.end() || *found != value)
        throw std::invalid_argument("batch shape is absent from the graph catalog");
    return found == rows.begin() ? 1 : *(found - 1) + 1;
}
} // namespace

BatchExecutionSpec make_batch_execution_spec(std::uint32_t capacity, std::uint32_t owners,
                                             bool capture) {
    BatchExecutionSpec spec{capacity, owners, {}};
    validate_spec(spec);
    if (!capture) return spec;
    for (std::uint32_t rows = 1; rows <= std::min(capacity, 8U); ++rows)
        spec.graph_rows.push_back(rows);
    for (std::uint64_t base = 8; base < capacity; base *= 2) {
        for (std::uint64_t step = 1; step <= 4; ++step) {
            const auto rows = base + base * step / 4;
            if (rows > capacity) break;
            spec.graph_rows.push_back(static_cast<std::uint32_t>(rows));
        }
    }
    if (spec.graph_rows.back() != capacity) spec.graph_rows.push_back(capacity);
    return spec;
}

BatchExecutionPlan::BatchExecutionPlan(BatchExecutionSpec spec) : spec_(std::move(spec)) {
    validate_spec(spec_);
    std::uint32_t lower = 1;
    for (auto rows : spec_.graph_rows) {
        const auto minimum_owner = ceil_div(lower, spec_.attention_owners);
        auto local =
            std::lower_bound(spec_.graph_rows.begin(), spec_.graph_rows.end(), minimum_owner);
        for (; local != spec_.graph_rows.end() && *local <= rows; ++local)
            graph_shapes_.push_back({rows, *local});
        // The final bucket may end at UINT32_MAX; its successor is never needed.
        if (rows != spec_.resident_capacity) lower = rows + 1;
    }
}

PhysicalBatchShape BatchExecutionPlan::exact(std::uint32_t rows, std::uint32_t local) const {
    if (!rows || rows > spec_.resident_capacity || local > rows ||
        local < ceil_div(rows, spec_.attention_owners))
        throw std::invalid_argument("batch membership exceeds its request or owner geometry");
    return {rows, local};
}

PhysicalBatchShape BatchExecutionPlan::decode(std::uint32_t rows, std::uint32_t local) const {
    (void)exact(rows, local);
    if (spec_.graph_rows.empty()) return {rows, local};
    const auto bucket = [&](std::uint32_t value) {
        return *std::lower_bound(spec_.graph_rows.begin(), spec_.graph_rows.end(), value);
    };
    return {bucket(rows), bucket(local)};
}

std::uint32_t BatchExecutionPlan::minimum_rows(std::uint32_t rows) const {
    if (!rows || rows > spec_.resident_capacity)
        throw std::invalid_argument("physical rows exceed prepared capacity");
    return spec_.graph_rows.empty() ? rows : lower_bound_rows(spec_.graph_rows, rows);
}

std::vector<std::uint32_t>
BatchExecutionPlan::representative_owners(PhysicalBatchShape shape) const {
    if (!std::binary_search(graph_shapes_.begin(), graph_shapes_.end(), shape))
        throw std::invalid_argument("batch shape is absent from the graph catalog");
    const auto first       = lower_bound_rows(spec_.graph_rows, shape.rows);
    const auto local_first = lower_bound_rows(spec_.graph_rows, shape.attention_rows);
    const auto maximum     = std::max(local_first, ceil_div(first, spec_.attention_owners));
    const auto rows        = std::max(first, maximum);
    std::vector<std::uint32_t> result(spec_.attention_owners);
    auto remaining = rows;
    for (auto& count : result) {
        count = std::min(remaining, maximum);
        remaining -= count;
    }
    if (remaining || decode(rows, maximum) != shape)
        throw std::logic_error("batch graph representative does not reproduce its shape");
    return result;
}

} // namespace ninfer::models
