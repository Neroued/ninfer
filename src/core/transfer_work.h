#pragma once

#include <cstdint>

namespace ninfer {

// Physical work represented by one logical context transfer. Payload bytes exclude host-arena
// padding. Completed observations count actual cudaMemcpy/cudaMemcpy2D calls; a plan whose Device
// destination has not been allocated yet uses the known Host coalescing runs as its nominal count.
struct TransferWork {
    std::uint64_t payload_bytes   = 0;
    std::uint32_t copy_operations = 0;

    [[nodiscard]] friend constexpr bool operator==(TransferWork, TransferWork) noexcept = default;
};

} // namespace ninfer
