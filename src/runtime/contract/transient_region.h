#pragma once

#include <cstddef>
#include <span>

namespace ninfer::runtime {

struct TransientRegion {
    std::byte* data       = nullptr;
    std::size_t size      = 0;
    std::size_t alignment = 1;

    [[nodiscard]] explicit operator bool() const noexcept { return data != nullptr; }

    [[nodiscard]] std::span<std::byte> bytes() const noexcept { return {data, size}; }
};

} // namespace ninfer::runtime
