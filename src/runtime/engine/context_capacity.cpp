#include "runtime/engine/context_capacity.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::runtime {
namespace {

std::size_t checked_add(std::size_t a, std::size_t b) {
    if (b > SIZE_MAX - a) throw std::overflow_error("context reservation exceeds size_t");
    return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b) {
    if (b && a > SIZE_MAX / b) throw std::overflow_error("context reservation exceeds size_t");
    return a * b;
}

void validate_curve(const ContextCapacityCurve& c) {
    if (!c.primary_page_tokens || !c.minimum_primary_page_groups ||
        c.minimum_primary_page_groups > c.maximum_primary_page_groups ||
        !c.minimum_device_reservation_bytes || !c.minimum_state_slots ||
        !c.bytes_per_additional_state_slot ||
        (c.minimum_primary_page_groups < c.maximum_primary_page_groups &&
         !c.bytes_per_additional_primary_page_group))
        throw std::invalid_argument("context capacity curve is invalid");
}

template <class Predicate>
std::uint32_t largest(std::uint32_t low, std::uint32_t high, Predicate fits) {
    while (low < high) {
        const auto middle = low + (std::uint64_t(high) - low + 1) / 2;
        if (fits(static_cast<std::uint32_t>(middle)))
            low = middle;
        else
            high = middle - 1;
    }
    return low;
}

} // namespace

std::size_t ContextCapacityCurve::reservation_bytes(ContextCapacity capacity) const {
    validate_curve(*this);
    if (capacity.primary_page_groups < minimum_primary_page_groups ||
        capacity.primary_page_groups > maximum_primary_page_groups ||
        capacity.device_state_slots < minimum_state_slots ||
        capacity.device_state_slots > INT32_MAX)
        throw std::invalid_argument("context capacity is outside the storage domain");
    return checked_add(
        checked_add(minimum_device_reservation_bytes,
                    checked_mul(capacity.primary_page_groups - minimum_primary_page_groups,
                                bytes_per_additional_primary_page_group)),
        checked_mul(capacity.device_state_slots - minimum_state_slots,
                    bytes_per_additional_state_slot));
}

std::size_t ContextCapacityCurve::control_bytes(ContextCapacity capacity) const {
    (void)reservation_bytes(capacity);
    return checked_add(
        checked_add(minimum_control_bytes,
                    checked_mul(capacity.primary_page_groups - minimum_primary_page_groups,
                                control_bytes_per_page_group)),
        checked_mul(capacity.device_state_slots - minimum_state_slots,
                    control_bytes_per_state_slot));
}

std::uint32_t ContextCapacityCurve::resolved_tokens(std::uint32_t pages) const {
    validate_curve(*this);
    if (pages < minimum_primary_page_groups || pages > maximum_primary_page_groups)
        throw std::invalid_argument("Main KV page count is outside the storage domain");
    const auto tokens = std::uint64_t(pages) * primary_page_tokens;
    if (tokens > UINT32_MAX) throw std::overflow_error("resolved KV capacity exceeds uint32");
    return static_cast<std::uint32_t>(tokens);
}

std::vector<ContextCapacityResolution>
resolve_group_context_capacity(const KvCapacityPolicy& policy, const ContextCacheOptions& cache,
                               std::span<const RankCapacityOffer> offers) {
    if (offers.empty()) throw std::invalid_argument("context capacity requires memory owners");
    if (!cache.checkpoint_interval)
        throw std::invalid_argument("checkpoint interval must be positive");
    if (policy.mode != KvCapacityMode::Automatic && policy.mode != KvCapacityMode::Explicit)
        throw std::invalid_argument("invalid KV capacity policy");
    if ((policy.mode == KvCapacityMode::Explicit &&
         (!policy.explicit_tokens || policy.automatic_headroom_bytes)) ||
        (policy.mode == KvCapacityMode::Automatic && policy.explicit_tokens))
        throw std::invalid_argument("inconsistent KV capacity policy");

    const auto page_tokens = offers.front().curve.primary_page_tokens;
    ContextCapacity minimum;
    std::uint32_t maximum_pages = UINT32_MAX;
    for (const auto& offer : offers) {
        validate_curve(offer.curve);
        if (offer.curve.primary_page_tokens != page_tokens)
            throw std::invalid_argument("context owners disagree on logical page extent");
        minimum.primary_page_groups =
            std::max(minimum.primary_page_groups, offer.curve.minimum_primary_page_groups);
        minimum.device_state_slots =
            std::max(minimum.device_state_slots, offer.curve.minimum_state_slots);
        maximum_pages = std::min(maximum_pages, offer.curve.maximum_primary_page_groups);
    }
    if (minimum.primary_page_groups > maximum_pages)
        throw std::invalid_argument("context owners have no common page capacity");
    ContextCapacity selected = minimum;
    if (policy.mode == KvCapacityMode::Explicit) {
        selected.primary_page_groups = 1U + (policy.explicit_tokens - 1U) / page_tokens;
        if (selected.primary_page_groups < minimum.primary_page_groups ||
            selected.primary_page_groups > maximum_pages)
            throw std::invalid_argument("explicit KV capacity is outside the storage domain");
    }
    const auto explicit_state = cache.extra_state_slots;
    if (explicit_state) {
        const auto slots = std::uint64_t(minimum.device_state_slots) + *explicit_state;
        if (slots > INT32_MAX) throw std::invalid_argument("StateImage slots exceed int32");
        selected.device_state_slots = slots;
    }
    const auto fits = [&](ContextCapacity capacity) {
        for (const auto& offer : offers) {
            if (offer.available_bytes < policy.automatic_headroom_bytes) return false;
            try {
                if (checked_add(offer.curve.reservation_bytes(capacity),
                                offer.curve.control_bytes(capacity)) >
                    offer.available_bytes - policy.automatic_headroom_bytes)
                    return false;
            } catch (const std::overflow_error&) { return false; }
        }
        return true;
    };
    if (!fits(selected))
        throw std::invalid_argument(
            "required context capacity exceeds the available memory budget");

    const auto target_states = [&](std::uint32_t pages) {
        if (explicit_state || !cache.enabled) return selected.device_state_slots;
        const auto tokens = std::uint64_t(pages) * page_tokens;
        const auto target =
            2ULL * minimum.device_state_slots + (1 + (tokens - 1) / cache.checkpoint_interval);
        return static_cast<std::uint32_t>(std::min<std::uint64_t>(target, INT32_MAX));
    };
    const auto target           = target_states(selected.primary_page_groups);
    selected.device_state_slots = largest(selected.device_state_slots, target, [&](auto slots) {
        return fits({selected.primary_page_groups, slots});
    });
    if (policy.mode == KvCapacityMode::Automatic && selected.device_state_slots == target) {
        selected.primary_page_groups =
            largest(selected.primary_page_groups, maximum_pages,
                    [&](auto pages) { return fits({pages, target_states(pages)}); });
        selected.device_state_slots = target_states(selected.primary_page_groups);
        // State slots are discrete. Use the remaining bytes for pages without allocating
        // another image solely to preserve the heuristic density at its next rounding step.
        selected.primary_page_groups =
            largest(selected.primary_page_groups, maximum_pages,
                    [&](auto pages) { return fits({pages, selected.device_state_slots}); });
    }

    std::vector<ContextCapacityResolution> result;
    result.reserve(offers.size());
    for (const auto& offer : offers) {
        const auto bytes = offer.curve.reservation_bytes(selected);
        result.push_back({
            .mode                        = policy.mode,
            .primary_page_groups         = selected.primary_page_groups,
            .maximum_primary_page_groups = offer.curve.maximum_primary_page_groups,
            .resolved_tokens = offer.curve.resolved_tokens(selected.primary_page_groups),
            .minimum_runtime_reservation_bytes = offer.curve.minimum_device_reservation_bytes,
            .bytes_per_additional_primary_page_group =
                offer.curve.bytes_per_additional_primary_page_group,
            .runtime_reservation_bytes     = bytes,
            .available_after_weights_bytes = offer.available_bytes,
            .automatic_headroom_bytes      = policy.automatic_headroom_bytes,
            .planned_slack_bytes =
                offer.available_bytes - bytes - offer.curve.control_bytes(selected),
            .device_state_slots        = selected.device_state_slots,
            .control_reservation_bytes = offer.curve.control_bytes(selected),
        });
    }
    return result;
}

ContextCapacityResolution resolve_context_capacity(const KvCapacityPolicy& policy,
                                                   const ContextCacheOptions& cache,
                                                   const ContextCapacityCurve& curve,
                                                   std::size_t available) {
    const std::array offers{RankCapacityOffer{curve, available}};
    return resolve_group_context_capacity(policy, cache, offers).front();
}

} // namespace ninfer::runtime
