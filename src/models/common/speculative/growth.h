#pragma once

#include "models/common/context/directory.h"
#include "models/common/speculative/round_plan.h"

#include <array>

namespace ninfer::models::speculative {
struct HistoryGrowth {
    context::HistoryDirectory& store;
    context::HistoryId id;
    std::uint32_t begin, anchor, held;
    context::ContextDemandId item;
    std::optional<context::RecordReservation> grant;

    HistoryGrowth(context::HistoryDirectory& directory, context::HistoryId history,
                  std::uint32_t first, std::uint32_t permitted)
        : store(directory), id(history), begin(first), anchor(permitted),
          held(store.reserved_growth_pages(id)) {}

    std::uint32_t extra(std::uint32_t depth) const {
        const auto required = store.write_pages(id, begin, anchor + depth);
        return required > held ? required - held : 0;
    }

    void request(context::ContextDemand& demand, std::uint32_t pages) {
        item = pages ? demand.add_growth(store, id, pages) : context::ContextDemandId{};
    }

    void receive(context::ContextReservation& reserved) {
        if (item.valid()) grant.emplace(reserved.take(item));
    }

    void adopt() {
        if (grant) store.adopt_growth(id, std::move(*grant));
    }

    void prepare(std::uint32_t depth) {
        if (!grant) return;
        const auto count = extra(depth);
        if (!count) {
            grant.reset();
            return;
        }
        grant->shrink(count);
        store.prepare_growth(id, *grant);
    }
};

struct RowGrowth {
    std::optional<HistoryGrowth> text, backend;
    using Pages = std::array<std::uint32_t, 2>;

    Pages pages(std::uint32_t depth) const {
        return {text ? text->extra(depth) : 0, backend ? backend->extra(depth) : 0};
    }

    void request(context::ContextDemand& demand, Pages pages) {
        if (text) text->request(demand, pages[0]);
        if (backend) backend->request(demand, pages[1]);
    }

    void receive(context::ContextReservation& reserved) {
        if (text) text->receive(reserved);
        if (backend) backend->receive(reserved);
    }

    void prepare(std::uint32_t depth) {
        if (text) text->prepare(depth);
        if (backend) backend->prepare(depth);
    }

    void adopt() {
        if (text) text->adopt();
        if (backend) backend->adopt();
    }
};

void grant_windows(context::ContextDirectory&, std::span<RowGrowth>, std::span<RoundMember>);

} // namespace ninfer::models::speculative
