#pragma once
#include "models/qwen3_5/program/program.h"

namespace ninfer::models::qwen3_5::detail {

[[nodiscard]] std::vector<GraphExecutionProfile> ordinary_graph_profiles(std::uint32_t capacity);
[[nodiscard]] std::vector<GraphExecutionProfile> mtp_graph_profiles(std::uint32_t capacity,
                                                                    std::uint32_t draft_window);
[[nodiscard]] std::vector<GraphExecutionProfile> dflash_graph_profiles(SpeculativeBackend backend,
                                                                       std::uint32_t capacity,
                                                                       std::uint32_t draft_window,
                                                                       std::uint32_t batch_size);

} // namespace ninfer::models::qwen3_5::detail
