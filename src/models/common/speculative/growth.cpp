#include "models/common/speculative/growth.h"

#include <stdexcept>

namespace ninfer::models::speculative {
void grant_windows(context::ContextDirectory& directory, std::span<RowGrowth> growth,
                   std::span<RoundMember> rows) {
    if (growth.size() != rows.size())
        throw std::invalid_argument("optional growth differs from round membership");
    context::ContextDemand all;
    std::size_t requesting_rows = 0;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        rows[i].granted_limit = rows[i].hard_limit;
        const auto pages      = growth[i].pages(rows[i].hard_limit);
        if (pages == RowGrowth::Pages{}) continue;
        growth[i].request(all, pages);
        ++requesting_rows;
    }
    if (!requesting_rows) return;
    if (auto reserved = directory.try_reserve_optional(all)) {
        for (auto& row : growth) row.receive(*reserved);
        return;
    }
    for (std::size_t i = 0; i < rows.size(); ++i) {
        auto& depth         = rows[i].granted_limit;
        auto& row           = growth[i];
        bool already_denied = requesting_rows == 1;
        while (depth) {
            const auto pages = row.pages(depth);
            if (pages == RowGrowth::Pages{}) break;
            if (!already_denied) {
                context::ContextDemand demand;
                row.request(demand, pages);
                if (auto reserved = directory.try_reserve_optional(demand)) {
                    row.receive(*reserved);
                    break;
                }
            }
            already_denied = false;
            do { --depth; } while (depth && row.pages(depth) == pages);
        }
    }
}
} // namespace ninfer::models::speculative
