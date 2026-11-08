#pragma once

#include "core/weight.h"
#include "ninfer/ops/linear.h"

#include <optional>
#include <variant>

namespace ninfer::ops {

// Address-independent matrix description used by execution planning.
struct LinearProfile {
    QType qtype;
    std::int32_t n, k;
    LinearPolicy policy = LinearPolicy::A16Only;
    QuantLayout layout  = QuantLayout::Contiguous;

    friend bool operator==(const LinearProfile&, const LinearProfile&) = default;
};

struct ProjectionProfile {
    LinearProfile first;
    std::optional<LinearProfile> second;

    friend bool operator==(const ProjectionProfile&, const ProjectionProfile&) = default;
};

// Prepared, non-owning operands. The consuming Op defines the logical row order.
struct SingleProjectionWeight {
    Weight weight;
    LinearPolicy policy = LinearPolicy::A16Only;
};

// Each parent retains the activation permission of its own mathematical Uses.
struct PairedProjectionWeights {
    SingleProjectionWeight first, second;
};

using ProjectionWeights = std::variant<SingleProjectionWeight, PairedProjectionWeights>;

[[nodiscard]] inline LinearProfile linear_profile(const SingleProjectionWeight& p) {
    return {p.weight.qtype, p.weight.n, p.weight.k, p.policy, p.weight.layout};
}

[[nodiscard]] inline LinearProfile linear_profile(LinearProfile p) { return p; }

[[nodiscard]] inline ProjectionProfile projection_profile(const ProjectionWeights& weights) {
    if (const auto* single = std::get_if<SingleProjectionWeight>(&weights)) {
        return {linear_profile(*single), {}};
    }
    const auto& pair = std::get<PairedProjectionWeights>(weights);
    return {linear_profile(pair.first), linear_profile(pair.second)};
}

} // namespace ninfer::ops
