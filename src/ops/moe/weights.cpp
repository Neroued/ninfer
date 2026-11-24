#include "ninfer/ops/moe.h"

#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {
std::int32_t dimension(std::uint64_t value) {
    if (!value || value > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("SwiGLU bank dimension is outside the Tensor domain");
    }
    return static_cast<std::int32_t>(value);
}

QuantLayout matrix_layout(QuantLayout layout) {
    return layout == QuantLayout::BatchedBlockScaleK16M128x4 ? QuantLayout::BlockScaleK16M128x4
                                                             : layout;
}
} // namespace

void ExpertInputUsage::include(const WeightInput& input) {
    if (!valid_linear_policy(input.policy))
        throw std::invalid_argument("expert has invalid activation policy");
    if (input.policy == LinearPolicy::A16Only || policy_ == LinearPolicy::A16Only)
        policy_ = LinearPolicy::A16Only;
    else if (input.policy == LinearPolicy::AllowA8)
        policy_ = LinearPolicy::AllowA8;
    if (!input.activation_input_divisor) {
        missing_divisor_ = true;
        return;
    }
    const float value = *input.activation_input_divisor;
    if (!std::isfinite(value) || value <= 0)
        throw std::invalid_argument("expert activation divisor must be finite and positive");
    if (divisor_)
        unequal_divisors_ |=
            std::bit_cast<std::uint32_t>(*divisor_) != std::bit_cast<std::uint32_t>(value);
    else
        divisor_ = value;
}

std::optional<float> ExpertInputUsage::activation_divisor(QType format) const {
    if (format != QType::NVFP4) return std::nullopt;
    if (allows_a4(policy_) && (!divisor_ || missing_divisor_ || unequal_divisors_))
        throw std::invalid_argument("NVFP4 expert A4 requires a common activation divisor");
    return unequal_divisors_ || missing_divisor_ ? std::nullopt : divisor_;
}

SwiGLUProfile swiglu_profile(const SwiGLUWeights& weights) {
    return swiglu_profile(linear_profile(weights.gate_up), linear_profile(weights.down));
}

SwiGLUProfile swiglu_profile(LinearProfile gu, LinearProfile down) {
    if (gu.n <= 0 || gu.n % 2 || gu.k <= 0 || down.n != gu.k || down.k != gu.n / 2 ||
        !valid_linear_policy(gu.policy) || !valid_linear_policy(down.policy)) {
        throw std::invalid_argument("SwiGLU matrices have inconsistent geometry or policy");
    }
    return {gu.k, gu.n / 2, gu.qtype, down.qtype, gu.layout, down.layout, gu.policy, down.policy};
}

RoutedSwiGLUProfile routed_swiglu_profile(const WeightBankLayout& gu, const WeightBankLayout& dn,
                                          std::int32_t top_k, LinearPolicy gate_up_policy,
                                          LinearPolicy down_policy) {
    if (gu.region.rows % 2 || gu.region.count != dn.region.count || gu.columns != dn.region.rows ||
        gu.region.rows / 2 != dn.columns || top_k <= 0 || std::uint32_t(top_k) > gu.region.count ||
        !valid_linear_policy(gate_up_policy) || !valid_linear_policy(down_policy)) {
        throw std::invalid_argument("routed SwiGLU banks have inconsistent geometry or policy");
    }
    return {{dimension(gu.columns), dimension(dn.columns), gu.format, dn.format,
             matrix_layout(gu.layout), matrix_layout(dn.layout), gate_up_policy, down_policy},
            dimension(gu.region.count),
            top_k};
}

RoutedSwiGLUProfile routed_swiglu_profile(const WeightBankView& gate_up, const WeightBankView& down,
                                          std::int32_t top_k, LinearPolicy gu, LinearPolicy dn) {
    if (!gate_up.parent || !down.parent) {
        throw std::invalid_argument("routed SwiGLU banks have no parents");
    }
    return routed_swiglu_profile(gate_up.layout, down.layout, top_k, gu, dn);
}
} // namespace ninfer::ops
