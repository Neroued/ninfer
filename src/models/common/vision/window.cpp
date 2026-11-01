#include "models/common/vision/window.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace ninfer::models::vision {
namespace {
bool same(runtime::ExecutionMember a, runtime::ExecutionMember b) {
    return a.sequence == b.sequence && a.residency == b.residency;
}
} // namespace

void validate_units(std::span<const Unit> units, std::uint32_t capacity,
                    std::uint32_t prompt_tokens) {
    std::uint32_t end = 0;
    for (const auto& unit : units) {
        if (!unit.patches || !unit.features || unit.token_begin < end ||
            unit.token_begin > prompt_tokens || unit.features > prompt_tokens - unit.token_begin)
            throw std::invalid_argument("Vision units do not describe ordered prompt features");
        if (unit.features > capacity)
            throw RequestError(RequestErrorKind::MediaBudgetExceeded,
                               "Vision segment needs " + std::to_string(unit.features) +
                                   " features; the configured Vision memory budget allows " +
                                   std::to_string(capacity));
        end = unit.end();
    }
}

std::shared_ptr<const WindowPlan> plan_window(std::span<const Unit> units, std::uint32_t begin,
                                              std::uint32_t end, std::uint32_t capacity,
                                              std::uint32_t ranks) {
    if (!ranks || !capacity || begin >= end)
        throw std::invalid_argument("Vision window requires a positive range and capacity");
    auto out         = std::make_shared<WindowPlan>();
    const auto first = std::lower_bound(units.begin(), units.end(), begin,
                                        [](const Unit& unit, auto p) { return unit.end() <= p; });
    for (auto it = first; it != units.end() && it->token_begin < end; ++it) {
        if (it->features > capacity)
            throw RequestError(RequestErrorKind::MediaBudgetExceeded,
                               "Vision segment exceeds the configured execution window");
        if (it->features > capacity - out->features) break;
        out->units.push_back(*it);
        out->features += it->features;
        out->consume_end = it->end();
    }
    if (out->units.empty()) return {};
    std::vector<std::uint32_t> order(out->units.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](auto a, auto b) { return out->units[a].cost > out->units[b].cost; });
    std::vector<std::uint64_t> loads(ranks);
    out->placements.resize(order.size());
    for (auto index : order) {
        const auto rank = static_cast<std::uint32_t>(std::min_element(loads.begin(), loads.end()) -
                                                     loads.begin());
        out->placements[index] = {index, rank, 0};
        const auto cost        = out->units[index].cost;
        loads[rank]            = cost > UINT64_MAX - loads[rank] ? UINT64_MAX : loads[rank] + cost;
    }
    std::uint32_t cursor = 0;
    for (std::uint32_t rank = 0; rank < ranks; ++rank) {
        OutputBlock block{rank, cursor, {}};
        for (auto& placement : out->placements) {
            if (placement.rank != rank) continue;
            const auto& unit        = out->units[placement.unit];
            placement.feature_begin = cursor;
            for (std::uint32_t i = 0; i < unit.features; ++i)
                block.token_positions.push_back(static_cast<std::int32_t>(unit.token_begin + i));
            cursor += unit.features;
        }
        if (!block.token_positions.empty()) out->outputs.push_back(std::move(block));
    }
    return out;
}

std::vector<EmbeddingSlice> embedding_slices(const WindowPlan& plan, std::uint32_t begin,
                                             std::uint32_t tokens) {
    std::vector<EmbeddingSlice> result;
    for (const auto& block : plan.outputs) {
        const auto first =
            std::lower_bound(block.token_positions.begin(), block.token_positions.end(), begin);
        const auto last =
            std::lower_bound(first, block.token_positions.end(), std::uint64_t(begin) + tokens);
        if (first == last) continue;
        EmbeddingSlice slice;
        slice.source_begin = block.feature_begin + (first - block.token_positions.begin());
        slice.destination_columns.reserve(last - first);
        for (auto it = first; it != last; ++it) slice.destination_columns.push_back(*it - begin);
        result.push_back(std::move(slice));
    }
    return result;
}

bool Window::available(runtime::ExecutionMember owner) const noexcept {
    return !active_ || (!release_ && same(active_->owner, owner));
}

std::optional<runtime::ExecutionMember> Window::owner() const noexcept {
    return active_ && !release_ ? std::optional(active_->owner) : std::nullopt;
}

const WindowUse* Window::current() const noexcept { return active_ ? &*active_ : nullptr; }

std::uint32_t Window::limit(runtime::ExecutionMember owner, std::span<const Unit> units,
                            std::uint32_t begin, std::uint32_t end, std::uint32_t capacity) const {
    std::uint32_t used = 0;
    const auto first   = std::lower_bound(units.begin(), units.end(), begin,
                                          [](const Unit& unit, auto p) { return unit.end() <= p; });
    for (auto it = first; it != units.end() && it->token_begin < end; ++it) {
        const bool blocked =
            active_ && (!available(owner) || it->token_begin >= active_->plan->consume_end);
        if (blocked || (!active_ && it->features > capacity - used))
            return std::max(begin, it->token_begin);
        used += it->features;
    }
    return end;
}

WindowUse Window::reserve(runtime::ExecutionMember owner, std::shared_ptr<const WindowPlan> plan) {
    if (!available(owner) || reader_)
        throw std::logic_error("Vision output window is held by an unfinished execution");
    if (active_) return {active_->id, owner, active_->plan, !produced_};
    if (!plan || plan->features == 0 || generation_ == UINT64_MAX)
        throw std::invalid_argument("Vision window has no executable plan or identity");
    active_   = WindowUse{++generation_, owner, std::move(plan), true};
    produced_ = release_ = false;
    return *active_;
}

void Window::submitted(std::uint64_t id, runtime::ExecutionTicket ticket) {
    if (!active_ || active_->id != id || reader_ || release_)
        throw std::logic_error("Vision submission differs from its reserved window");
    reader_          = ticket;
    produced_        = true;
    active_->produce = false;
}

void Window::consumed(std::uint64_t id, bool complete) {
    if (!active_ || active_->id != id || !reader_)
        throw std::logic_error("Vision consumption has no matching submitted reader");
    release_ |= complete;
}

void Window::retired(runtime::ExecutionTicket ticket) {
    if (!reader_ || *reader_ != ticket) return;
    reader_.reset();
    if (release_) active_.reset();
}

void Window::revoke(runtime::ExecutionMember owner) {
    if (!active_ || !same(active_->owner, owner)) return;
    release_ = true;
    if (!reader_) active_.reset();
}

} // namespace ninfer::models::vision
