#pragma once

#include "core/transfer_work.h"
#include "core/context_pool.h"
#include "runtime/contract/request.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <string>
#include <vector>

namespace ninfer::runtime {

enum class CheckpointRole : std::uint8_t { InputReplay, LongAnchor, SharedPrefix, Continuation };
enum class CheckpointOrigin : std::uint8_t { Semantic, Intermediate };
enum class CaptureSkip : std::uint8_t { Unavailable, Retained };
enum class ContextStart : std::uint8_t { Started, Deferred, Stale };

// Additive contributions in the shortfall domains of one resource assessment.
struct ContextReleaseMeasure {
    std::vector<std::uint64_t> bytes;
    std::size_t metadata_objects = 0;

    void merge(const ContextReleaseMeasure& other) {
        if (bytes.size() < other.bytes.size()) bytes.resize(other.bytes.size());
        for (std::size_t i = 0; i < other.bytes.size(); ++i) bytes[i] += other.bytes[i];
        metadata_objects += other.metadata_objects;
    }

    void subtract(const ContextReleaseMeasure& other) {
        for (std::size_t i = 0; i < other.bytes.size(); ++i) bytes[i] -= other.bytes[i];
        metadata_objects -= other.metadata_objects;
    }

    [[nodiscard]] std::uint64_t total_bytes() const noexcept {
        std::uint64_t total = 0;
        for (const auto value : bytes) total += value;
        return total;
    }

    [[nodiscard]] std::uint64_t useful_bytes(std::span<const std::uint64_t> limits) const {
        std::uint64_t total = 0;
        for (std::size_t i = 0; i < bytes.size(); ++i) total += std::min(bytes[i], limits[i]);
        return total;
    }
};

struct KvPoolId {
    std::uint16_t index                                           = 0;
    friend constexpr bool operator==(KvPoolId, KvPoolId) noexcept = default;
};

// IDs describe logical history kinds, not physical memory partitions.
struct KvPoolDescriptor {
    KvPoolId id;
    std::string_view label;
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
    const unsigned __int128 suffix      = suffix_tokens;
    const unsigned __int128 linear      = static_cast<unsigned __int128>(prefix_tokens) * suffix;
    const unsigned __int128 triangular  = suffix * (suffix + 1U) / 2U;
    constexpr unsigned __int128 maximum = ~static_cast<unsigned __int128>(0);
    const unsigned __int128 attention =
        triangular > maximum - linear ? maximum : linear + triangular;
    result.attention_pairs = attention > std::numeric_limits<std::uint64_t>::max()
                                 ? std::numeric_limits<std::uint64_t>::max()
                                 : static_cast<std::uint64_t>(attention);
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
    HostToHost,
    PeerDeviceToDevice,
};

struct ContextPeerTransfer {
    std::uint32_t source_rank = 0, destination_rank = 0;

    friend constexpr bool operator==(ContextPeerTransfer, ContextPeerTransfer) noexcept = default;
};

struct ContextTransferObservation {
    ContextTransferResource resource;
    ContextTransferDirection direction = ContextTransferDirection::DeviceToHost;
    std::uint32_t image_count          = 0;
    std::uint32_t page_groups          = 0;
    TransferWork work;
    std::uint64_t elapsed_ns = 0;
    std::uint32_t rank       = 0;
    // Both endpoints report their own elapsed time. Payload is counted at the source only.
    std::optional<ContextPeerTransfer> peer;
};

struct ContextTransferRequirement {
    ContextTransferResource resource;
    ContextTransferDirection direction = ContextTransferDirection::DeviceToHost;
    std::uint32_t image_count          = 0;
    std::uint32_t page_groups          = 0;
    TransferWork work;
    std::uint32_t rank = 0;
    std::optional<ContextPeerTransfer> peer;

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

struct ContextCapacity {
    std::size_t device_bytes = 0;

    friend bool operator==(ContextCapacity, ContextCapacity) = default;
};

struct ContextStartupWorkingSet {
    std::string name;
    std::vector<RecordRequest> records;
};

// The directory supplies the cost of its own nodes. Classes follow the canonical pool geometry.
struct ContextMetadataPlan {
    std::size_t fixed_bytes     = 0;
    std::size_t bytes_per_block = 0;
    std::vector<std::size_t> bytes_per_record;
};

// Rank-local cost of the model's balanced, full-context reference workload. Observation only.
struct ContextCapacityReference {
    std::uint32_t tokens_per_request = 0;
    std::uint32_t request_count      = 0;
    std::uint64_t device_bytes       = 0;

    friend bool operator==(const ContextCapacityReference&,
                           const ContextCapacityReference&) = default;
};

struct ContextCapacityPlan {
    std::vector<RecordSpec> records;
    std::vector<ContextStartupWorkingSet> working_sets;
    // Device reservation includes colocated Host buffers on a shared-memory device.
    std::size_t fixed_device_bytes = 0;
    std::size_t fixed_host_bytes   = 0;
    ContextMetadataPlan metadata;
    ContextCapacityReference reference;

    [[nodiscard]] ContextCapacity minimum_capacity() const;
    [[nodiscard]] std::size_t reservation_bytes(ContextCapacity capacity) const;
    [[nodiscard]] std::size_t control_bytes(ContextCapacity capacity) const;
};

struct ContextCapacityResolution {
    ContextCapacityMode mode                      = ContextCapacityMode::Automatic;
    std::size_t device_bytes                      = 0;
    std::size_t minimum_device_bytes              = 0;
    std::size_t block_bytes                       = 0;
    std::size_t minimum_runtime_reservation_bytes = 0;
    std::size_t runtime_reservation_bytes         = 0;
    std::size_t available_after_weights_bytes     = 0;
    std::size_t available_after_startup_bytes     = 0;
    std::size_t planned_slack_bytes               = 0;
    // Directory budget generated by this rank's schema, and its physical owner charge.
    std::size_t metadata_budget_bytes     = 0;
    std::size_t control_reservation_bytes = 0;
    ContextCapacityReference reference;

    [[nodiscard]] ContextCapacity capacity() const noexcept { return {device_bytes}; }
};

} // namespace ninfer::runtime
