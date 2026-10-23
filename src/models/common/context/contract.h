#pragma once

#include "runtime/contract/resources.h"

#include <cstddef>
#include <cstdint>
#include <variant>
#include <vector>

namespace ninfer::models::context {

enum class ContextMemory : std::uint8_t { Device, Host };

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

struct MappingUpdate {
    std::uint32_t pool_id       = 0;
    std::int32_t row            = -1;
    std::uint32_t logical_begin = 0;
    std::vector<std::int64_t> offsets;
};

using ContextAction = std::variant<RecordClear, RecordCopy, MappingUpdate>;

// Rank-local byte actions execute in order. The directory retains source borrows and
// destination reservations until execution retires, then publishes the logical result.
struct ContextActions {
    std::vector<ContextAction> operations;

    [[nodiscard]] bool empty() const noexcept { return operations.empty(); }
};

inline constexpr std::size_t kContextActionBatchOperations = 1024;
inline constexpr std::size_t kContextActionBatchOffsets    = 4096;

struct ContextActionCursor {
    std::size_t operation      = 0;
    std::size_t mapping_offset = 0;

    [[nodiscard]] bool complete(const ContextActions& actions) const noexcept {
        return operation == actions.operations.size() && mapping_offset == 0;
    }
};

// Produces the next bounded Context-frame payload without changing action order. The full
// transaction keeps every source and destination alive until all batches have retired.
[[nodiscard]] ContextActions take_action_batch(const ContextActions&, ContextActionCursor&);

} // namespace ninfer::models::context
