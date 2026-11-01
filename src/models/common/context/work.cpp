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
        (cursor.operation == actions.operations.size() && !cursor.complete(actions)) ||
        cursor.peer_range || cursor.peer_offset) {
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
        if (std::holds_alternative<PeerCopy>(action)) {
            if (result.empty())
                throw std::logic_error("Context peer actions require coordinated batching");
            break;
        }
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

namespace {

std::size_t peer_bytes(const PeerCopy& action, const ContextActionCursor& cursor,
                       std::size_t maximum_ranges) {
    if (cursor.mapping_offset || cursor.peer_range >= action.ranges.size() ||
        cursor.peer_offset >= action.ranges[cursor.peer_range].bytes)
        throw std::out_of_range("Context peer cursor is outside its ranges");
    const auto end =
        cursor.peer_range + std::min(maximum_ranges, action.ranges.size() - cursor.peer_range);
    std::size_t bytes = 0;
    for (auto i = cursor.peer_range; i < end; ++i) {
        const auto count =
            action.ranges[i].bytes - (i == cursor.peer_range ? cursor.peer_offset : 0);
        if (!count || count > SIZE_MAX - bytes)
            throw std::invalid_argument("Context peer ranges have an invalid byte count");
        bytes += count;
    }
    return bytes;
}

PeerCopy peer_piece(const PeerCopy& action, ContextActionCursor& cursor, std::size_t bytes) {
    PeerCopy result{action.peer, action.send, {}, action.resource};
    if (cursor.peer_range == 0 && cursor.peer_offset == 0) {
        result.image_count = action.image_count;
        result.page_groups = action.page_groups;
    }
    while (bytes) {
        auto range       = action.ranges[cursor.peer_range];
        const auto count = std::min(bytes, range.bytes - cursor.peer_offset);
        if (range.offset < 0 || cursor.peer_offset > INT64_MAX - range.offset)
            throw std::overflow_error("Context peer range offset overflow");
        range.offset += cursor.peer_offset;
        range.bytes = count;
        result.ranges.push_back(range);
        cursor.peer_offset += count;
        bytes -= count;
        if (cursor.peer_offset == action.ranges[cursor.peer_range].bytes) {
            ++cursor.peer_range;
            cursor.peer_offset = 0;
        }
    }
    if (cursor.peer_range == action.ranges.size()) {
        ++cursor.operation;
        cursor.peer_range = 0;
    }
    return result;
}

} // namespace

std::vector<ContextActions> take_action_batches(std::span<const ContextActionBatchSource> ranks) {
    std::vector<ContextActions> result(ranks.size());
    std::vector<ContextActionCursor> next;
    next.reserve(ranks.size());
    bool local = false;
    std::optional<std::size_t> first;
    std::pair<std::uint32_t, std::uint32_t> edge;
    for (std::size_t i = 0; i < ranks.size(); ++i) {
        const auto& rank = ranks[i];
        if (!rank.actions || !rank.cursor)
            throw std::invalid_argument("Context batch source is missing actions or cursor");
        if (std::any_of(ranks.begin(), ranks.begin() + i,
                        [&](const auto& item) { return item.rank == rank.rank; }))
            throw std::invalid_argument("Context batch contains duplicate ranks");
        next.push_back(*rank.cursor);
        const auto& cursor = next.back();
        if (cursor.operation > rank.actions->operations.size() ||
            (cursor.operation == rank.actions->operations.size() &&
             !cursor.complete(*rank.actions)))
            throw std::out_of_range("Context batch cursor is outside its plan");
        if (cursor.complete(*rank.actions)) continue;
        const auto* peer = std::get_if<PeerCopy>(&rank.actions->operations[cursor.operation]);
        if (!peer) {
            local = true;
            continue;
        }
        const auto current =
            peer->send ? std::pair{rank.rank, peer->peer} : std::pair{peer->peer, rank.rank};
        if (!first || current < edge) {
            first = i;
            edge  = current;
        }
    }
    if (local) {
        for (std::size_t i = 0; i < ranks.size(); ++i) {
            const auto& actions = *ranks[i].actions;
            if (!next[i].complete(actions) &&
                !std::holds_alternative<PeerCopy>(actions.operations[next[i].operation]))
                result[i] = take_action_batch(actions, next[i]);
        }
    } else if (first) {
        const auto i       = *first;
        const auto& action = std::get<PeerCopy>(ranks[i].actions->operations[next[i].operation]);
        const auto found   = std::find_if(ranks.begin(), ranks.end(),
                                          [&](const auto& rank) { return rank.rank == action.peer; });
        if (found == ranks.end() || found->rank == ranks[i].rank)
            throw std::invalid_argument("Context peer action has no remote participant");
        const auto j = static_cast<std::size_t>(found - ranks.begin());
        if (next[j].complete(*found->actions))
            throw std::invalid_argument("Context peer action has no matching endpoint");
        const auto* other = std::get_if<PeerCopy>(&found->actions->operations[next[j].operation]);
        if (!other || other->peer != ranks[i].rank || other->send == action.send ||
            other->resource != action.resource)
            throw std::invalid_argument("Context peer action order differs between endpoints");
        if (peer_bytes(action, next[i], SIZE_MAX) != peer_bytes(*other, next[j], SIZE_MAX))
            throw std::invalid_argument("Context peer action byte counts differ between endpoints");
        const auto count = std::min(peer_bytes(action, next[i], kContextActionBatchOffsets),
                                    peer_bytes(*other, next[j], kContextActionBatchOffsets));
        result[i].operations.emplace_back(peer_piece(action, next[i], count));
        result[j].operations.emplace_back(peer_piece(*other, next[j], count));
    }
    for (std::size_t i = 0; i < ranks.size(); ++i) *ranks[i].cursor = next[i];
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
