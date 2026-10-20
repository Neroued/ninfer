#pragma once

#include <cstdint>
#include <optional>
#include <span>

namespace ninfer::runtime {

// Semantic captures retain their exact position. Intermediate captures prefer a nearby
// proven message boundary, otherwise the end of the next ordinary execution window.
[[nodiscard]] std::optional<std::uint32_t>
intermediate_capture(std::uint32_t begin, std::uint32_t end, std::uint32_t last_saved,
                     std::optional<std::uint32_t> next_capture,
                     std::span<const std::uint32_t> semantic_boundaries, std::uint32_t interval);

} // namespace ninfer::runtime
