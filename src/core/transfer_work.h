#pragma once

#include <cstdint>

namespace ninfer {

// Physical byte spans moved by a transfer. Copied record padding is included; unused arena
// capacity is not. copy_operations counts individual byte-copy spans, including Host moves.
struct TransferWork {
    std::uint64_t payload_bytes   = 0;
    std::uint32_t copy_operations = 0;

    [[nodiscard]] friend constexpr bool operator==(TransferWork, TransferWork) noexcept = default;
};

} // namespace ninfer
