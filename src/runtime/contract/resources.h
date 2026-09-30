#pragma once

#include "runtime/contract/request.h"
#include "core/transfer_work.h"
#include <cstddef>
#include <cstdint>
#include <limits>

namespace ninfer::runtime {

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

enum class ContextResourceClass : std::uint8_t {
    State,
    MainKV,
    BackendKV,
};

enum class ContextTransferDirection : std::uint8_t {
    DeviceToHost,
    HostToDevice,
    DeviceToDevice,
};

struct ContextTransferObservation {
    ContextResourceClass resource      = ContextResourceClass::State;
    ContextTransferDirection direction = ContextTransferDirection::DeviceToHost;
    std::uint64_t units                = 0; // State images for State; bytes for typed KV.
    std::uint32_t page_count           = 0;
    TransferWork work;
    std::uint64_t elapsed_ns = 0;
};

enum class Readiness : std::uint8_t {
    Ready,
    TemporarilyBlocked,
    PermanentlyInfeasible,
};

enum class ContextTransactionStatus : std::uint8_t {
    InProgress,
    Published,
    Aborted,
};

enum class ContextTransactionReserveStatus : std::uint8_t {
    Reserved,
    Aborted,
};

struct ContextTransactionInProgress {};

enum class ContextTransactionKind : std::uint8_t {
    Materialization,
};

enum class RetentionClass : std::uint8_t {
    SharedStable,
    LiveSession,
    RecentPrivate,
    Disposable,
};

struct ProgramResourceRevision {
    std::uint64_t value = 0;

    [[nodiscard]] friend constexpr bool operator==(ProgramResourceRevision,
                                                   ProgramResourceRevision) noexcept = default;
};

// Target-produced affine reservation curve for one Main KV physical-capacity axis. The byte
// values come from complete target physical layout plans, not from a model geometry formula in
// the common runtime.
struct SequenceCapacityCurve {
    std::uint32_t main_page_tokens                   = 0;
    std::uint32_t minimum_main_page_groups           = 0;
    std::uint32_t maximum_main_page_groups           = 0;
    std::size_t minimum_device_reservation_bytes     = 0;
    std::size_t bytes_per_additional_main_page_group = 0;

    [[nodiscard]] std::size_t reservation_bytes(std::uint32_t main_page_groups) const;
    [[nodiscard]] std::uint32_t resolved_tokens(std::uint32_t main_page_groups) const;
};

struct KvCapacityResolution {
    KvCapacityMode mode                              = KvCapacityMode::Explicit;
    std::uint32_t main_page_groups                   = 0;
    std::uint32_t maximum_main_page_groups           = 0;
    std::uint32_t resolved_tokens                    = 0;
    std::size_t minimum_runtime_reservation_bytes    = 0;
    std::size_t bytes_per_additional_main_page_group = 0;
    std::size_t runtime_reservation_bytes            = 0;
    std::size_t available_after_weights_bytes        = 0;
    std::size_t available_after_startup_bytes        = 0;
    std::size_t automatic_headroom_bytes             = 0;
    std::size_t planned_slack_bytes                  = 0;
};

} // namespace ninfer::runtime
