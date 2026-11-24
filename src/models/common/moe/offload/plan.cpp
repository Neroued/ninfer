#include "models/common/moe/offload/plan.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace ninfer::models::moe::offload {
namespace {

std::int32_t hybrid_token_limit(HybridPolicy policy, std::uint32_t workers) {
    if (policy.max_tokens <= 0 || policy.reference_workers == 0 || workers == 0)
        throw std::invalid_argument("MoE hybrid policy requires positive limits and team size");
    const auto maximum = static_cast<std::uint64_t>(policy.max_tokens);
    return static_cast<std::int32_t>(
        std::clamp(maximum * workers / policy.reference_workers, std::uint64_t{1}, maximum));
}

void size_resources(Plan& plan, std::int32_t max_tokens) {
    plan.bridge       = {};
    std::size_t masks = 0;
    for (std::size_t i = 0; i < plan.layers.size(); ++i) {
        const auto& layer = plan.layers[i];
        if (!layer.spec.profile.routed.experts) continue;
        plan.bridge.include(describe_execution(layer.spec.profile, ops::ExpertSubsetKind::MaskOnes,
                                               layer.spec.epilogue, 1,
                                               std::min(max_tokens, layer.hybrid_max_tokens)));
        masks += std::size_t(layer.spec.profile.routed.experts) * sizeof(std::int32_t);
    }
    plan.device_bytes = masks + plan.bridge.device_frame_bytes + plan.cache.config.capacity_bytes +
                        plan.cache.offset_bytes + plan.cache.divisor_bytes;
}

struct CacheAllocation {
    std::vector<std::uint32_t> experts;
    std::size_t bytes = 0;
};

CacheAllocation allocate_cache(const Plan& plan, std::size_t capacity,
                               std::span<const std::uint32_t> minimum_counts) {
    if (!minimum_counts.empty() && minimum_counts.size() != plan.layers.size())
        throw std::invalid_argument("MoE minimum cache counts differ from the layer directory");
    const auto count = plan.layers.size();
    std::vector<std::size_t> entry(count), minimum(count), maximum(count);
    std::size_t lower_bytes = 0, upper_share = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const auto& layer = plan.layers[i];
        const auto forced = minimum_counts.empty() ? 0 : minimum_counts[i];
        if (forced > std::uint32_t(layer.spec.profile.routed.experts))
            throw std::invalid_argument("MoE minimum cache count exceeds its bank");
        if (!layer.spec.profile.routed.experts) continue;
        entry[i]   = plan.cache.banks[layer.spec.bank.index].entry_bytes;
        minimum[i] = entry[i] * forced;
        maximum[i] = entry[i] * layer.spec.profile.routed.experts;
        if (minimum[i] > capacity - lower_bytes)
            throw std::invalid_argument("MoE minimum cache counts exceed the byte budget");
        lower_bytes += minimum[i];
        upper_share = std::max(upper_share, maximum[i]);
    }
    const auto fits = [&](std::size_t share) {
        std::size_t used = 0;
        for (std::size_t i = 0; i < count; ++i) {
            const auto bytes = std::max(minimum[i], std::min(maximum[i], share));
            if (bytes > capacity - used) return false;
            used += bytes;
        }
        return true;
    };
    std::size_t share = 0;
    while (share < upper_share) {
        const auto middle = share + (upper_share - share + 1) / 2;
        if (fits(middle))
            share = middle;
        else
            upper_share = middle - 1;
    }
    CacheAllocation out;
    out.experts.resize(count);
    std::vector<std::size_t> candidates;
    std::size_t unit = 0;
    for (std::size_t i = 0; i < count; ++i) {
        if (!entry[i]) continue;
        const auto quota = std::max(minimum[i], std::min(maximum[i], share));
        out.experts[i]   = static_cast<std::uint32_t>(quota / entry[i]);
        out.bytes += out.experts[i] * entry[i];
        if (out.experts[i] < std::uint32_t(plan.layers[i].spec.profile.routed.experts) &&
            minimum[i] <= share) {
            candidates.push_back(i);
            unit = std::gcd(unit, entry[i]);
        }
    }
    if (candidates.empty()) return out;
    const auto remainder = (capacity - out.bytes) / unit;
    std::vector<std::int32_t> selected(remainder + 1, -1);
    selected[0] = -2;
    for (const auto i : candidates) {
        const auto cost = entry[i] / unit;
        for (auto value = remainder; value >= cost; --value) {
            if (selected[value] == -1 && selected[value - cost] != -1)
                selected[value] = static_cast<std::int32_t>(i);
        }
    }
    auto used = remainder;
    while (selected[used] == -1) --used;
    out.bytes += used * unit;
    while (used) {
        const auto i = static_cast<std::size_t>(selected[used]);
        ++out.experts[i];
        used -= entry[i] / unit;
    }
    return out;
}

std::size_t missing_bytes(const Plan& plan, std::span<const std::uint32_t> counts) {
    std::size_t largest = 0;
    for (std::size_t i = 0; i < plan.layers.size(); ++i) {
        const auto& layer = plan.layers[i];
        if (layer.spec.profile.routed.experts)
            largest = std::max(largest, (layer.spec.profile.routed.experts - counts[i]) *
                                            plan.cache.banks[layer.spec.bank.index].entry_bytes);
    }
    return largest;
}

std::optional<CacheAllocation> fixed_allocation(const Plan& plan, std::uint32_t windows) {
    const auto capacity = plan.cache.config.capacity_bytes;
    std::vector<std::size_t> shares{0};
    for (const auto& layer : plan.layers) {
        if (!layer.spec.profile.routed.experts) continue;
        const auto bytes = plan.cache.banks[layer.spec.bank.index].entry_bytes;
        for (std::uint32_t n = 1; n <= std::uint32_t(layer.spec.profile.routed.experts); ++n)
            shares.push_back(n * bytes);
    }
    std::sort(shares.begin(), shares.end());
    shares.erase(std::unique(shares.begin(), shares.end()), shares.end());
    std::optional<CacheAllocation> best;
    CacheAllocation candidate;
    candidate.experts.resize(plan.layers.size());
    for (const auto share : shares) {
        candidate.bytes = 0;
        for (std::size_t i = 0; i < plan.layers.size(); ++i) {
            const auto& layer = plan.layers[i];
            if (!layer.spec.profile.routed.experts) continue;
            const auto bytes     = plan.cache.banks[layer.spec.bank.index].entry_bytes;
            candidate.experts[i] = static_cast<std::uint32_t>(
                std::min<std::size_t>(layer.spec.profile.routed.experts, share / bytes));
            candidate.bytes += candidate.experts[i] * bytes;
        }
        if (candidate.bytes > capacity) break;
        const auto window = missing_bytes(plan, candidate.experts);
        if (window <= (capacity - candidate.bytes) / windows) best = candidate;
    }
    if (!best) return {};
    // Whole-expert rounding can free additional streaming capacity. Keep the fixed allocation
    // monotone while redistributing that remainder under the same balanced-share rule.
    for (;;) {
        const auto window = missing_bytes(plan, best->experts);
        auto next         = allocate_cache(plan, capacity - windows * window, best->experts);
        if (next.bytes == best->bytes) break;
        best = std::move(next);
    }
    return best;
}

void append_layout(std::vector<ExpertPlacement>& out, ExpertBankId bank,
                   std::span<const std::uint32_t> ids, std::size_t bytes, std::size_t& offset) {
    std::vector<std::uint32_t> ordered(ids.begin(), ids.end());
    std::sort(ordered.begin(), ordered.end());
    for (const auto id : ordered) {
        out.push_back({{bank, id}, offset});
        offset += bytes;
    }
}

} // namespace

bool plan_cache(Plan& plan) {
    auto fixed        = fixed_allocation(plan, 2);
    plan.window_count = 2;
    if (!fixed) {
        fixed             = fixed_allocation(plan, 1);
        plan.window_count = 1;
    }
    if (!fixed) return false;
    const auto capacity           = plan.cache.config.capacity_bytes;
    const auto target             = allocate_cache(plan, capacity, fixed->experts);
    plan.cache.config.fixed_bytes = fixed->bytes;
    plan.target_bytes             = target.bytes;
    plan.fixed_layout.clear();
    plan.extra_layout.clear();
    plan.windows     = {};
    const auto span  = capacity - fixed->bytes;
    const auto first = span / plan.window_count / 256 * 256;
    plan.windows[0]  = {fixed->bytes, plan.window_count == 1 ? span : first, 256};
    if (plan.window_count == 2) plan.windows[1] = {fixed->bytes + first, span - first, 256};
    std::size_t fixed_offset = 0, extra_offset = 0;
    for (std::size_t i = 0; i < plan.layers.size(); ++i) {
        auto& layer         = plan.layers[i];
        layer.fixed_experts = fixed->experts[i];
        layer.cache_experts = target.experts[i];
        layer.stream_layout.clear();
        if (!layer.spec.profile.routed.experts) continue;
        const auto bytes = plan.cache.banks[layer.spec.bank.index].entry_bytes;
        const std::span priority(layer.spec.priority);
        append_layout(plan.fixed_layout, layer.spec.bank, priority.first(layer.fixed_experts),
                      bytes, fixed_offset);
        append_layout(
            plan.extra_layout, layer.spec.bank,
            priority.subspan(layer.fixed_experts, layer.cache_experts - layer.fixed_experts), bytes,
            extra_offset);
        std::size_t offset = 0;
        append_layout(layer.stream_layout, layer.spec.bank, priority.subspan(layer.fixed_experts),
                      bytes, offset);
    }
    return true;
}

Plan describe_plan(std::span<const ExpertBankCacheSpec> banks, std::span<const LayerSpec> layers) {
    Plan out;
    out.cache = describe_expert_cache(banks, {});
    out.layer_indices_.assign(banks.size(), SIZE_MAX);
    for (const auto& spec : layers) {
        const auto count = spec.profile.routed.experts;
        if (spec.bank.index >= banks.size() || out.layer_indices_[spec.bank.index] != SIZE_MAX ||
            count <= 0 || banks[spec.bank.index].resident ||
            banks[spec.bank.index].gate_up.region.count != std::uint32_t(count) ||
            spec.priority.size() != std::size_t(count) || spec.hybrid.max_tokens <= 0 ||
            spec.hybrid.reference_workers == 0)
            throw std::invalid_argument("MoE layer does not match its scheduled expert bank");
        std::vector<bool> seen(count);
        for (const auto e : spec.priority) {
            if (e >= std::uint32_t(count) || seen[e])
                throw std::invalid_argument("MoE expert priority is not a permutation");
            seen[e] = true;
        }
        out.layer_indices_[spec.bank.index] = out.layers.size();
        out.layers.push_back({spec, spec.hybrid.max_tokens});
    }
    return out;
}

std::size_t Plan::layer_index(ExpertBankId bank) const {
    if (bank.index >= layer_indices_.size() || layer_indices_[bank.index] == SIZE_MAX)
        throw std::invalid_argument("expert bank is outside the MoE traversal");
    return layer_indices_[bank.index];
}

const LayerPlan& Plan::layer(ExpertBankId bank) const { return layers[layer_index(bank)]; }

std::optional<Plan> resolve_plan(Plan out, std::span<const ExpertBankCacheSpec> specs,
                                 std::size_t available, std::int32_t max_tokens,
                                 const CpuTopology& topology, std::uint32_t requested_workers) {
    const auto order = cpu_worker_order(topology.info());
    if (order.empty()) throw std::runtime_error("no CPU available for MoE execution");
    const auto workers       = requested_workers ? requested_workers
                                                 : static_cast<std::uint32_t>(std::min<std::size_t>(
                                                 16, std::max<std::size_t>(1, order.size() - 1)));
    std::size_t largest_bank = 0;
    for (auto& layer : out.layers) {
        if (!layer.spec.profile.routed.experts) continue;
        layer.hybrid_max_tokens = hybrid_token_limit(layer.spec.hybrid, workers);
        largest_bank = std::max(largest_bank, out.cache.banks[layer.spec.bank.index].entry_bytes *
                                                  std::size_t(layer.spec.profile.routed.experts));
    }
    size_resources(out, max_tokens);
    if (!largest_bank || out.device_bytes > available ||
        available - out.device_bytes < largest_bank)
        return std::nullopt;
    ExpertCacheConfig config;
    config.capacity_bytes = (available - out.device_bytes) / 256 * 256;
    config.upload_slots   = std::max<std::uint32_t>(
        3, static_cast<std::uint32_t>((2 * largest_bank + config.max_batch_bytes - 1) /
                                        config.max_batch_bytes) +
               2);
    config.max_in_flight_uses = static_cast<std::uint32_t>(out.layers.size() + 8);
    out.cache                 = describe_expert_cache(specs, config);
    if (!plan_cache(out)) return std::nullopt;
    size_resources(out, max_tokens);
    out.team = plan_cpu_team(topology.info(), workers);
    if (workers < order.size()) out.caller_cpus = {order[workers]};
    return out;
}

} // namespace ninfer::models::moe::offload
