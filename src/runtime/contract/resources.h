#pragma once

#include "core/math_util.h"
#include "core/transfer_work.h"
#include "runtime/contract/request.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>

namespace ninfer {
struct KVPageGeometry;
}

namespace ninfer::runtime {

enum class CheckpointRole : std::uint8_t { InputReplay, LongAnchor, SharedPrefix, Continuation };

struct KvPoolId {
    std::uint16_t index                                           = 0;
    friend constexpr bool operator==(KvPoolId, KvPoolId) noexcept = default;
};

// Pool IDs are contiguous and fixed for a Program's lifetime. Labels are diagnostic only.
struct KvPoolDescriptor {
    KvPoolId id;
    std::string_view label;
    const KVPageGeometry* geometry   = nullptr;
    std::uint32_t device_page_groups = 0;
    std::size_t device_page_bytes    = 0;
    std::size_t host_page_stride     = 0;
};

// Physical quantities are sampled from stores. Each architecture declares its pool capacity;
// inactive pools have zero capacity. Page groups from different pools never offset each other.
template <std::size_t PoolCapacity>
struct ContextResources {
    static constexpr std::size_t pool_capacity = PoolCapacity;
    std::uint32_t state_slots                  = 0;
    std::array<std::uint32_t, PoolCapacity> kv_page_groups{};
    std::size_t host_bytes = 0;

    [[nodiscard]] std::optional<KvPoolId> first_kv_deficit() const noexcept {
        for (std::size_t i = 0; i < PoolCapacity; ++i) {
            if (kv_page_groups[i]) return KvPoolId{static_cast<std::uint16_t>(i)};
        }
        return std::nullopt;
    }

    [[nodiscard]] std::uint64_t useful(const ContextResources& shortage) const noexcept {
        if (shortage.state_slots) return std::min(state_slots, shortage.state_slots);
        if (const auto pool = shortage.first_kv_deficit()) {
            return std::min(kv_page_groups[pool->index], shortage.kv_page_groups[pool->index]);
        }
        return std::min(host_bytes, shortage.host_bytes);
    }

    [[nodiscard]] std::array<std::uint64_t, PoolCapacity + 2> quantities() const noexcept {
        std::array<std::uint64_t, PoolCapacity + 2> result{};
        result[0] = state_slots;
        for (std::size_t i = 0; i < PoolCapacity; ++i) result[i + 1] = kv_page_groups[i];
        result.back() = host_bytes;
        return result;
    }

    friend bool operator==(const ContextResources&, const ContextResources&) noexcept = default;
};

template <class Resources>
struct ResourceReservation {
    bool reserved = false;
    Resources shortage;

    explicit operator bool() const noexcept { return reserved; }
};

// Exact features for the startup-selected static prefill cost model. They describe only the
// suffix rebuilt after a selected prefix and remain separate from Scheduler service work.
struct PrefillWork {
    std::uint64_t chunks          = 0;
    std::uint64_t tokens          = 0;
    std::uint64_t attention_pairs = 0;
    std::uint64_t vision_items    = 0;
    std::uint64_t vision_patches  = 0;

    [[nodiscard]] friend constexpr bool operator==(PrefillWork, PrefillWork) noexcept = default;
};

// Exact prefill feature definition for a suffix beginning after prefix_tokens. Attention work is
// prefix*suffix + suffix*(suffix+1)/2 and all arithmetic saturates.
[[nodiscard]] inline PrefillWork make_prefill_work(std::uint64_t prefix_tokens,
                                                   std::uint64_t suffix_tokens,
                                                   std::uint64_t vision_items,
                                                   std::uint64_t vision_patches,
                                                   std::uint32_t prefill_chunk) noexcept {
    PrefillWork result;
    result.chunks =
        suffix_tokens == 0 || prefill_chunk == 0 ? 0 : 1U + (suffix_tokens - 1U) / prefill_chunk;
    result.tokens                       = suffix_tokens;
    result.vision_items                 = vision_items;
    result.vision_patches               = vision_patches;
    // attention_pairs = min(max64, prefix*suffix + suffix*(suffix+1)/2), computed with
    // u128 limbs so the 128-bit sum saturates at max64 exactly as the old __int128 code did.
    std::uint64_t linear_high = 0;
    const std::uint64_t linear_low = core::u128_mul(prefix_tokens, suffix_tokens, &linear_high);
    // One of suffix and suffix+1 is even, so halve before multiplying to stay in 64 bits.
    // suffix+1 wraps only for suffix == max64 (odd), whose half is 2^63 and must be special-cased.
    const bool suffix_even          = (suffix_tokens & 1U) == 0U;
    const std::uint64_t first       = suffix_even ? suffix_tokens / 2U : suffix_tokens;
    const std::uint64_t second      = suffix_even
                                          ? suffix_tokens + 1U
                                          : (suffix_tokens == std::numeric_limits<std::uint64_t>::max()
                                                 ? (std::numeric_limits<std::uint64_t>::max() >> 1U) + 1U
                                                 : (suffix_tokens + 1U) / 2U);
    std::uint64_t triangular_high = 0;
    const std::uint64_t triangular_low = core::u128_mul(first, second, &triangular_high);
    const std::uint64_t sum_low  = linear_low + triangular_low;
    const std::uint64_t sum_high = linear_high + triangular_high + (sum_low < linear_low ? 1U : 0U);
    result.attention_pairs = sum_high != 0 ? std::numeric_limits<std::uint64_t>::max() : sum_low;
    return result;
}

struct ContextTransferResource {
    std::optional<KvPoolId> pool;

    [[nodiscard]] constexpr bool is_state() const noexcept { return !pool; }

    [[nodiscard]] constexpr std::size_t index() const noexcept {
        return pool ? 1U + pool->index : 0;
    }

    friend constexpr bool operator==(ContextTransferResource,
                                     ContextTransferResource) noexcept = default;
};

enum class ContextTransferDirection : std::uint8_t {
    DeviceToHost,
    HostToDevice,
    DeviceToDevice,
};

struct ContextTransferObservation {
    ContextTransferResource resource;
    ContextTransferDirection direction = ContextTransferDirection::DeviceToHost;
    std::uint32_t image_count          = 0;
    std::uint32_t page_groups          = 0;
    TransferWork work;
    std::uint64_t elapsed_ns = 0;
};

struct ContextTransferRequirement {
    ContextTransferResource resource;
    ContextTransferDirection direction = ContextTransferDirection::DeviceToHost;
    std::uint32_t image_count          = 0;
    std::uint32_t page_groups          = 0;
    TransferWork work;

    [[nodiscard]] friend constexpr bool operator==(ContextTransferRequirement,
                                                   ContextTransferRequirement) noexcept = default;
};

struct ContextOperationCounts {
    std::uint64_t state_moves            = 0;
    std::uint64_t state_forks            = 0;
    std::uint64_t state_restores         = 0;
    std::uint64_t pressure_spill_pages   = 0;
    std::uint64_t partial_tail_cow_pages = 0;
};

struct SequenceCapacityCurve {
    std::uint32_t primary_page_tokens                   = 0;
    std::uint32_t minimum_primary_page_groups           = 0;
    std::uint32_t maximum_primary_page_groups           = 0;
    std::size_t minimum_device_reservation_bytes        = 0;
    std::size_t bytes_per_additional_primary_page_group = 0;

    [[nodiscard]] std::size_t reservation_bytes(std::uint32_t primary_page_groups) const;
    [[nodiscard]] std::uint32_t resolved_tokens(std::uint32_t primary_page_groups) const;
};

struct KvCapacityResolution {
    KvCapacityMode mode                                 = KvCapacityMode::Explicit;
    std::uint32_t primary_page_groups                   = 0;
    std::uint32_t maximum_primary_page_groups           = 0;
    std::uint32_t resolved_tokens                       = 0;
    std::size_t minimum_runtime_reservation_bytes       = 0;
    std::size_t bytes_per_additional_primary_page_group = 0;
    std::size_t runtime_reservation_bytes               = 0;
    std::size_t available_after_weights_bytes           = 0;
    std::size_t available_after_startup_bytes           = 0;
    std::size_t automatic_headroom_bytes                = 0;
    std::size_t planned_slack_bytes                     = 0;
};

} // namespace ninfer::runtime
