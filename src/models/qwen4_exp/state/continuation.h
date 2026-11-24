#pragma once

#include <array>
#include <cstdint>
#include <optional>

namespace ninfer::models::qwen4_exp {

struct ContinuationMetadata {
    std::uint32_t main_frontier = 0;
    std::int32_t rope_delta     = 0;
    std::array<std::int32_t, 2> ple_raw_tail{};
    std::optional<std::uint32_t> mtp_frontier;
};

[[nodiscard]] inline std::uint32_t mtp_history_frontier(std::uint32_t frontier) {
    return frontier ? frontier - 1 : 0;
}

} // namespace ninfer::models::qwen4_exp
