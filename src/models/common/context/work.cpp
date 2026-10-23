#include "models/common/context/work.h"
#include "models/common/context/contract.h"
#include "core/paged_kv_cache.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>

namespace ninfer::models::context {

ContextActions take_action_batch(const ContextActions& actions, ContextActionCursor& cursor) {
    if (cursor.operation > actions.operations.size() ||
        (cursor.operation == actions.operations.size() && cursor.mapping_offset)) {
        throw std::out_of_range("Context action cursor is outside its plan");
    }
    auto next = cursor;
    ContextActions result;
    result.operations.reserve(
        std::min(kContextActionBatchOperations, actions.operations.size() - next.operation));
    std::size_t mapping_elements = 0;
    while (next.operation < actions.operations.size() &&
           result.operations.size() < kContextActionBatchOperations) {
        const auto& action = actions.operations[next.operation];
        if (const auto* mapping = std::get_if<MappingUpdate>(&action)) {
            if (next.mapping_offset > mapping->offsets.size()) {
                throw std::out_of_range("Context mapping cursor is outside its update");
            }
            const auto remaining = mapping->offsets.size() - next.mapping_offset;
            if (remaining && mapping_elements == kContextActionBatchOffsets) break;
            const auto count = std::min(remaining, kContextActionBatchOffsets - mapping_elements);
            if (next.mapping_offset > UINT32_MAX - mapping->logical_begin) {
                throw std::overflow_error("Context mapping cursor exceeds its logical range");
            }
            const auto begin = mapping->offsets.begin() + next.mapping_offset;
            result.operations.emplace_back(MappingUpdate{
                mapping->pool_id,
                mapping->row,
                mapping->logical_begin + static_cast<std::uint32_t>(next.mapping_offset),
                {begin, begin + count}});
            mapping_elements += count;
            next.mapping_offset += count;
            if (next.mapping_offset == mapping->offsets.size()) {
                ++next.operation;
                next.mapping_offset = 0;
            }
        } else {
            if (next.mapping_offset) {
                throw std::invalid_argument("Context copy cursor has a mapping offset");
            }
            result.operations.push_back(action);
            ++next.operation;
        }
    }
    cursor = next;
    return result;
}

std::uint64_t elapsed_ns(Clock::time_point started) noexcept {
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - started).count();
    return elapsed > 0 ? static_cast<std::uint64_t>(elapsed) : 0;
}

std::int32_t checked_i32(std::uint32_t value, const char* label) {
    if (value > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(label);
    }
    return static_cast<std::int32_t>(value);
}

std::uint32_t kv_pages_for_frontier(std::uint32_t frontier) noexcept {
    return frontier == 0 ? 0U : 1U + (frontier - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

runtime::ContextTransferRequirement
state_transfer_requirement(TransferWork work, runtime::ContextTransferDirection direction) {
    return runtime::ContextTransferRequirement{
        .resource    = {},
        .direction   = direction,
        .image_count = 1,
        .page_groups = 0,
        .work        = work,
    };
}

runtime::ContextTransferRequirement
kv_transfer_requirement(runtime::ContextTransferResource resource,
                        runtime::ContextTransferDirection direction, std::uint32_t pages,
                        TransferWork work) {
    return runtime::ContextTransferRequirement{
        .resource    = resource,
        .direction   = direction,
        .page_groups = pages,
        .work        = work,
    };
}

} // namespace ninfer::models::context
