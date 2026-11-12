#pragma once

#include "runtime/contract/resources.h"
#include <cstddef>
#include <cstdint>
#include <chrono>

namespace ninfer::models::context {

class HistoryDirectory;
struct HistoryBindingDemand;

using Clock = std::chrono::steady_clock;

std::uint64_t elapsed_ns(Clock::time_point started) noexcept;

std::int32_t checked_i32(std::uint32_t value, const char* label);

std::uint32_t kv_pages_for_frontier(std::uint32_t frontier) noexcept;

runtime::ContextTransferRequirement
state_transfer_requirement(TransferWork work, runtime::ContextTransferDirection direction);

runtime::ContextTransferRequirement
kv_transfer_requirement(runtime::ContextTransferResource resource,
                        runtime::ContextTransferDirection direction, std::uint32_t pages,
                        TransferWork work);

// Copy work selected by the same binding demand that prepares a writable history.
[[nodiscard]] std::vector<runtime::ContextTransferRequirement>
history_tail_requirements(const HistoryDirectory&, const HistoryBindingDemand&);

} // namespace ninfer::models::context
