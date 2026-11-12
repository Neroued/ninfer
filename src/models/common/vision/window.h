#pragma once

#include "models/common/frontend/media.h"
#include "runtime/executor/contract.h"
#include "runtime/contract/resources.h"

#include <memory>
#include <algorithm>
#include <stdexcept>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::vision {

// Quote the serial Native window lifecycle without payloads or executor reservations.
// A window produced for one chunk remains fixed until its last unit is consumed.
template <class Items, class ItemView>
std::uint64_t prefill_chunks(const Items& items, ItemView item_view, std::uint32_t begin,
                             std::uint32_t end, std::uint32_t chunk,
                             std::uint32_t feature_capacity) {
    if (begin >= end || !chunk) return 0;
    std::size_t first_item = 0, first_span = 0;
    const auto frame = [&](std::size_t& item, std::size_t& span) -> const frontend::TokenSpan* {
        while (item < items.size()) {
            const auto& spans = item_view(items[item]).token_spans;
            if (span < spans.size()) return &spans[span];
            ++item;
            span = 0;
        }
        return nullptr;
    };
    std::uint64_t count      = 0;
    std::uint32_t window_end = 0;
    while (begin < end) {
        while (const auto* span = frame(first_item, first_span)) {
            if (span->begin + span->count > begin) break;
            ++first_span;
        }
        auto next =
            static_cast<std::uint32_t>(std::min<std::uint64_t>(end, std::uint64_t(begin) + chunk));
        auto item = first_item, span_index = first_span;
        std::uint64_t used = 0;
        auto produced_end  = window_end;
        while (const auto* span = frame(item, span_index)) {
            if (span->begin >= next) break;
            if ((window_end && span->begin >= window_end) ||
                (!window_end && span->count > feature_capacity - used)) {
                next = static_cast<std::uint32_t>(std::max<std::size_t>(begin, span->begin));
                break;
            }
            used += span->count;
            if (!window_end) produced_end = static_cast<std::uint32_t>(span->begin + span->count);
            ++span_index;
        }
        if (next == begin) throw std::logic_error("recovery Vision unit exceeds its window");
        window_end = next >= produced_end ? 0 : produced_end;
        begin      = next;
        ++count;
    }
    return count;
}

// Native encoder units are images/video frames, not whole video payloads. A partially
// intersected unit must still be encoded in full; text-only gaps encode no media.
template <class Items, class ItemView>
runtime::PrefillWork recovery_work(const Items& items, ItemView item_view, std::uint32_t begin,
                                   std::uint32_t end, std::uint32_t chunk,
                                   std::uint32_t feature_capacity) {
    if (begin > end) throw std::invalid_argument("recovery begins after its target");
    auto work = runtime::make_prefill_work(begin, end - begin, 0, 0, chunk);
    if (begin == end || items.empty()) return work;
    for (const auto& entry : items) {
        const auto& item = item_view(entry);
        for (const auto& span : item.token_spans)
            if (span.begin < end && span.begin + span.count > begin) {
                ++work.vision_items;
                work.vision_patches += std::uint64_t(item.grid.height) * item.grid.width;
            }
    }
    work.chunks = prefill_chunks(items, item_view, begin, end, chunk, feature_capacity);
    return work;
}

// One independently encodable unit. Patch offsets are relative to its prepared payload.
struct Unit {
    std::uint32_t item = 0, index = 0;
    std::uint64_t patch_begin = 0;
    std::uint32_t patches     = 0;
    std::uint32_t token_begin = 0, features = 0;
    std::uint64_t cost = 0;

    [[nodiscard]] std::uint32_t end() const { return token_begin + features; }
};

struct Placement {
    std::uint32_t unit, rank, feature_begin;
};

struct OutputBlock {
    std::uint32_t rank, feature_begin;
    std::vector<std::int32_t> token_positions;
};

struct WindowPlan {
    std::vector<Unit> units;
    std::vector<Placement> placements;
    std::vector<OutputBlock> outputs;
    std::uint32_t consume_end = 0;
    std::uint32_t features    = 0;
};

// Select complete units intersecting [begin,end), stopping before a unit that would exceed
// capacity. Output is rank-major, with monotonically ordered token positions within each rank.
[[nodiscard]] std::shared_ptr<const WindowPlan> plan_window(std::span<const Unit>,
                                                            std::uint32_t begin, std::uint32_t end,
                                                            std::uint32_t feature_capacity,
                                                            std::uint32_t ranks);
void validate_units(std::span<const Unit>, std::uint32_t feature_capacity,
                    std::uint32_t prompt_tokens);

struct EmbeddingSlice {
    std::uint32_t source_begin = 0;
    std::vector<std::int32_t> destination_columns;
};

[[nodiscard]] std::vector<EmbeddingSlice> embedding_slices(const WindowPlan&, std::uint32_t begin,
                                                           std::uint32_t tokens);

struct WindowUse {
    std::uint64_t id = 0;
    runtime::ExecutionMember owner;
    std::shared_ptr<const WindowPlan> plan;
    bool produce = false;
};

// Coordinator-owned single output lease. Logical completion never releases an in-flight
// reader; the native Session reports consumption and the Executor reports retirement.
class Window {
public:
    [[nodiscard]] bool available(runtime::ExecutionMember) const noexcept;
    [[nodiscard]] std::optional<runtime::ExecutionMember> owner() const noexcept;
    [[nodiscard]] const WindowUse* current() const noexcept;
    [[nodiscard]] std::uint32_t limit(runtime::ExecutionMember, std::span<const Unit>,
                                      std::uint32_t begin, std::uint32_t end,
                                      std::uint32_t capacity) const;
    [[nodiscard]] WindowUse reserve(runtime::ExecutionMember, std::shared_ptr<const WindowPlan>);
    void submitted(std::uint64_t id, runtime::ExecutionTicket);
    void consumed(std::uint64_t id, bool complete);
    void retired(runtime::ExecutionTicket);
    void revoke(runtime::ExecutionMember);

private:
    std::optional<WindowUse> active_;
    std::optional<runtime::ExecutionTicket> reader_;
    std::uint64_t generation_ = 0;
    bool produced_            = false;
    bool release_             = false;
};

} // namespace ninfer::models::vision
