#pragma once

#include "models/common/context/directory/history.h"
#include "models/common/context/directory/state.h"

#include <cstddef>
#include <cstdint>
#include <variant>
#include <vector>

namespace ninfer::models::context {

struct StateClear {
    std::int32_t slot = -1;
};

struct StateCopy {
    std::int32_t source      = -1;
    std::int32_t destination = -1;
    // Zero bytes copies the complete image. A field group is relative to one image.
    std::size_t offset = 0;
    std::size_t bytes  = 0;
};

struct PageCopy {
    PageSlot source;
    PageSlot destination;
};

struct PageTransfer {
    StateCopyDirection direction = StateCopyDirection::DeviceToHost;
    HostExtent host;
    std::vector<PageSlot> pages;
};

struct MappingUpdate {
    TableRow row;
    std::uint32_t logical_begin = 0;
    std::vector<PageSlot> pages;
};

using ContextAction = std::variant<StateClear, StateCopy, StateTransferDescription, PageCopy,
                                   PageTransfer, MappingUpdate>;

// Owns all descriptors and mapping arrays. Source leases and destination owners remain in the
// corresponding control-side transaction until its execution dependency has retired.
struct ContextActions {
    std::vector<ContextAction> operations;

    [[nodiscard]] bool empty() const noexcept { return operations.empty(); }
};

} // namespace ninfer::models::context
