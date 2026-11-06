#pragma once

#include "runtime/contract/resources.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <variant>
#include <vector>

namespace ninfer::models::context {

enum class ContextMemory : std::uint8_t { Device, Host };
enum class ContextUse : std::uint8_t { Request, Cache };

struct ContextRange {
    ContextMemory memory = ContextMemory::Device;
    std::int64_t offset  = -1;
    std::size_t bytes    = 0;

    friend bool operator==(const ContextRange&, const ContextRange&) = default;
};

struct RecordClear {
    ContextRange destination;
};

struct RecordCopy {
    ContextRange source;
    ContextRange destination;
    runtime::ContextTransferResource resource;
    std::uint32_t image_count = 0;
    std::uint32_t page_groups = 0;
};

// A matched peer transfer describes only this rank's ranges, concatenated in order. Both
// endpoints have the same byte count. The directory orders all peer edges globally and
// retains address borrows until both endpoints retire; no remote address crosses the wire.
struct PeerCopy {
    std::uint32_t peer = 0;
    bool send          = false;
    std::vector<ContextRange> ranges;
    runtime::ContextTransferResource resource;
    std::uint32_t image_count = 0;
    std::uint32_t page_groups = 0;
};

struct MappingUpdate {
    std::uint32_t pool_id       = 0;
    std::int32_t row            = -1;
    std::uint32_t logical_begin = 0;
    std::vector<std::int64_t> offsets;
};

using ContextAction = std::variant<RecordClear, RecordCopy, MappingUpdate, PeerCopy>;

// Rank-local byte actions execute in order. The directory retains source borrows and
// destination reservations until execution retires, then publishes the logical result.
struct ContextActions {
    std::vector<ContextAction> operations;

    [[nodiscard]] bool empty() const noexcept { return operations.empty(); }
};

inline constexpr std::size_t kContextActionBatchOperations = 1024;
inline constexpr std::size_t kContextActionBatchOffsets    = 4096;

[[nodiscard]] inline std::size_t action_batch_storage_capacity() {
    const auto ordinary = kContextActionBatchOperations * sizeof(ContextAction) +
                          kContextActionBatchOffsets * sizeof(std::int64_t);
    const auto peer = sizeof(ContextAction) + kContextActionBatchOffsets * sizeof(ContextRange);
    return sizeof(ContextActions) + (ordinary > peer ? ordinary : peer);
}

struct ContextActionCursor {
    std::size_t operation      = 0;
    std::size_t mapping_offset = 0;
    std::size_t peer_range     = 0;
    std::size_t peer_offset    = 0;

    [[nodiscard]] bool complete(const ContextActions& actions) const noexcept {
        return operation == actions.operations.size() && mapping_offset == 0 && peer_range == 0 &&
               peer_offset == 0;
    }
};

// Produces the next bounded Context-frame payload without changing action order. The full
// transaction keeps every source and destination alive until all batches have retired.
[[nodiscard]] ContextActions take_action_batch(const ContextActions&, ContextActionCursor&);

struct ContextActionBatchSource {
    std::uint32_t rank;
    const ContextActions* actions;
    ContextActionCursor* cursor;
};

// Advances every cursor atomically. Matched peer endpoints are emitted in the same batch,
// with equal byte counts even when their physical ranges have different boundaries.
[[nodiscard]] std::vector<ContextActions>
    take_action_batches(std::span<const ContextActionBatchSource>);

} // namespace ninfer::models::context
